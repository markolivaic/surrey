"""How large a model this card will hold, and how fast it runs, by a rule written before the probe.

The published model is 726,225 parameters because the laptop has 4.29 GB. This walks a ladder of
widths and batch sizes on the rented card, measures the peak memory and the step rate of each, and
reports the largest configuration that fits under a budget with room to spare.

The rule, fixed before anything ran:

  a configuration fits when a forward and backward pass peaks below 75 percent of the budget, leaving
  a quarter for fragmentation and for the evaluation pass, which holds a second batch of activations
  the ladder walks width over batch: width doubles the model, batch only repeats it, and the question
  being asked is about capacity
  the budget is 24 GB, the card this project planned for; the rented card's own memory is reported
  beside it, so a larger choice can be made deliberately rather than by accident
  every figure here is from this card and says so

It trains on real crops from the prepared tiles, because a synthetic block would not carry the same
neighbour structure and the step rate would flatter the card.

Run:
    python scripts/pipeline/rented_probe.py
    python scripts/pipeline/rented_probe.py --steps 8 --budget-gb 24
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, CLASS_NAMES, N_CLASSES, POINTS_PER_CROP  # noqa: E402
from surrey.model import SurreyNet, parameter_count  # noqa: E402

PREP = ROOT / "data" / "prep" / "dales"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
OUT = ROOT / "data" / "artifacts" / "rented_probe.json"
WIDTHS = (128, 192, 256, 384, 512)
BATCHES = (4, 8, 16, 32)
HEADROOM = 0.75
RULE = (
    "written before the probe ran: a configuration fits when the peak of a forward and backward "
    "pass stays under 75 percent of the budget, leaving a quarter for fragmentation and for the "
    "evaluation pass; the ladder walks width before batch because width is the capacity question; "
    "the budget is the 24 GB this project planned for, with the rented card's own memory reported "
    "beside it"
)


def crops(tiles: list[CachedTile], batch: int, rng: np.random.Generator) -> np.ndarray:
    feats = []
    for _ in range(batch):
        tile = tiles[rng.integers(len(tiles))]
        feat, _label, _rep = tile.random_crop(rng, BLOCK_M, POINTS_PER_CROP, 3)
        feats.append(feat)
    return np.stack(feats)


def probe(width: int, batch: int, tiles: list[CachedTile], steps: int, seed: int) -> dict:
    """Peak memory and step rate for one configuration, on real crops."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = SurreyNet(N_CLASSES, width=width, extra_features=0).to("cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    weights = torch.ones(N_CLASSES, device="cuda")
    t0 = None
    for step in range(steps):
        x = torch.from_numpy(crops(tiles, batch, rng)).to("cuda", non_blocking=True)
        y = torch.randint(0, len(CLASS_NAMES), (batch, POINTS_PER_CROP), device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            logits = net(x)
            loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), y.reshape(-1), weight=weights)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step == 1:  # the first step allocates; timing starts after it
            torch.cuda.synchronize()
            t0 = time.time()
    torch.cuda.synchronize()
    seconds = time.time() - t0 if t0 is not None else float("nan")
    timed = max(steps - 2, 1)
    peak = torch.cuda.max_memory_reserved() / 1e9
    out = {
        "width": width,
        "batch": batch,
        "parameters": parameter_count(net),
        "peak_reserved_gb": round(peak, 2),
        "steps_per_second": round(timed / seconds, 3) if seconds == seconds else None,
        "points_per_second": round(timed * batch * POINTS_PER_CROP / seconds)
        if seconds == seconds
        else None,
    }
    del net, opt, scaler
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--steps", type=int, default=8, help="steps per configuration, two of them untimed"
    )
    ap.add_argument("--budget-gb", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this probe is for a card")
    names = [
        t["name"]
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "train"
    ][:8]
    tiles = [CachedTile(PREP / n, "train") for n in names]
    card = torch.cuda.get_device_properties(0)
    result = {
        "rule_written_before_the_probe": RULE,
        "budget_gb": args.budget_gb,
        "card": {
            "name": card.name,
            "memory_gb": round(card.total_memory / 1e9, 2),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "where": "a rented card, not the laptop",
        "steps_per_configuration": args.steps,
        "tiles_read": names,
        "ladder": [],
    }
    for width in WIDTHS:
        for batch in BATCHES:
            try:
                row = probe(width, batch, tiles, args.steps, args.seed)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {"width": width, "batch": batch, "out_of_memory": True}
            row["fits_budget"] = bool(row.get("peak_reserved_gb", 1e9) <= HEADROOM * args.budget_gb)
            row["fits_card"] = bool(
                row.get("peak_reserved_gb", 1e9) <= HEADROOM * result["card"]["memory_gb"]
            )
            result["ladder"].append(row)
            print(row, flush=True)
    fitting = [r for r in result["ladder"] if r.get("fits_budget")]
    on_card = [r for r in result["ladder"] if r.get("fits_card")]

    def largest(rows: list[dict]) -> dict | None:
        return max(rows, key=lambda r: (r["width"], r["batch"])) if rows else None

    result["chosen_under_the_budget"] = largest(fitting)
    result["largest_on_this_card"] = largest(on_card)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "chosen_under_the_budget": result["chosen_under_the_budget"],
                "largest_on_this_card": result["largest_on_this_card"],
            },
            indent=1,
        )
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
