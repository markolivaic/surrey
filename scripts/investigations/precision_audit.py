"""Where half precision loses the geometry: every layer of the model, under fp16 autocast, against an exact pass.

Training and prediction run under fp16 autocast. The neighbour search expands |q|^2 + |r|^2 - 2 q.r, and
under autocast the q.r product ran in fp16; with z up to 200 m above a block's floor, fp16 cannot tell
neighbours a few centimetres apart. This measures it on the tallest Financial District blocks, with the
tallest Sunset blocks for contrast, on the published checkpoint:

    for each of the six neighbour searches (three set abstraction levels, k = 32, and three feature
    propagation levels, k = 3), the share of the exact k nearest neighbours a version keeps, over
    all queries and over queries more than 100 m up;
    for each layer's output, its relative error against the exact pass;
    the share of points whose predicted class matches the exact pass, over all points and over points
    more than 100 m up.

The exact pass runs with autocast off and neighbours found in float64. The versions:

    before          fp16 autocast, the neighbour search as it was before the fix
    after           fp16 autocast, backend/surrey/model.knn_idx as it is now (fp32, centred)
    fp32_uncentred  neighbours only: the old expansion in fp32 without centring, the choice not taken

The centroid draws are seeded identically for every version, so every version asks the same queries.

Writes data/artifacts/precision_audit.json.

Run:
    python scripts/investigations/precision_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import surrey.model as model_module  # noqa: E402
from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, POINTS_PER_CROP  # noqa: E402
from surrey.inference import load_checkpoint  # noqa: E402

CKPT = ROOT / "data" / "checkpoints" / "m0_seedA.pt"
PREP = ROOT / "data" / "prep" / "target"
OUT = ROOT / "data" / "artifacts" / "precision_audit.json"
AREAS = {"financial": 4, "sunset": 2}  # the tallest blocks of each
TALL_M = 100.0
LAYERS = ["sa1", "sa2", "sa3", "fp3", "fp2", "fp1"]  # the order knn_idx is called in a forward pass
FIXED_KNN = model_module.knn_idx


def legacy_knn(query, ref, k, chunk=1024):
    """The neighbour search before the fix, verbatim: the expansion under whatever autocast is active."""
    b, s, _ = query.shape
    out = torch.empty(b, s, k, dtype=torch.long, device=query.device)
    ref_sq = (ref**2).sum(-1).unsqueeze(1)
    for i in range(0, s, chunk):
        j = min(i + chunk, s)
        q = query[:, i:j]
        d = (q**2).sum(-1).unsqueeze(-1) + ref_sq - 2.0 * torch.bmm(q, ref.transpose(1, 2))
        out[:, i:j] = d.topk(k, dim=-1, largest=False).indices
    return out


def uncentred_fp32_knn(query, ref, k, chunk=1024):
    with torch.autocast(device_type=query.device.type, enabled=False):
        return legacy_knn(query.float(), ref.float(), k, chunk)


def exact_knn(query, ref, k, chunk=1024):
    b, s, _ = query.shape
    out = torch.empty(b, s, k, dtype=torch.long, device=query.device)
    with torch.autocast(device_type=query.device.type, enabled=False):
        q64, r64 = query.double(), ref.double()
        for i in range(0, s, chunk):
            j = min(i + chunk, s)
            d = torch.cdist(q64[:, i:j], r64, compute_mode="donot_use_mm_for_euclid_dist")
            out[:, i:j] = d.topk(k, dim=-1, largest=False).indices
    return out


def run(net, x, knn, amp: bool, seed: int = 0):
    """One forward pass; returns the knn calls (queries, indices) and each layer's output."""
    calls, outputs = [], {}

    def recorder(query, ref, k, chunk=1024):
        idx = knn(query, ref, k, chunk)
        calls.append((query.detach(), idx.detach()))
        return idx

    hooks = []
    for name in ("sa1", "sa2", "sa3", "fp3", "fp2", "fp1", "head"):
        layer = getattr(net, name)

        def hook(_m, _inp, out, name=name):
            val = out[1] if isinstance(out, tuple) else out
            outputs[name] = val.detach().float()

        hooks.append(layer.register_forward_hook(hook))
    model_module.knn_idx = recorder
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            net(x)
    finally:
        model_module.knn_idx = FIXED_KNN
        for h in hooks:
            h.remove()
    return calls, outputs


def kept(idx: torch.Tensor, ref_idx: torch.Tensor) -> torch.Tensor:
    """Per query, the share of the reference's neighbours this version also found."""
    return (idx.unsqueeze(-1) == ref_idx.unsqueeze(-2)).any(-1).float().mean(-1)


def tallest_crops(area: str, n: int, channels: int) -> list[np.ndarray]:
    tile = CachedTile(PREP / area, "sf")
    crops = [
        np.asarray(feats, dtype=np.float32)
        for feats, _l, _c, _v in tile.iter_full(BLOCK_M, POINTS_PER_CROP, channels=channels, seed=0)
    ]
    crops.sort(key=lambda c: -float(c[:, 2].max()))
    return crops[:n]


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit(
            "this measures the CUDA autocast path the models ran on; it needs a CUDA card"
        )
    net, _blob, channels = load_checkpoint(CKPT, "cuda", allow_pickle=True)
    net.eval()
    result = {"checkpoint": CKPT.relative_to(ROOT).as_posix(), "tall_m": TALL_M, "areas": {}}
    for area, n in AREAS.items():
        acc: dict = {}
        for crop in tallest_crops(area, n, channels):
            x = torch.from_numpy(crop).cuda().unsqueeze(0)
            ref_calls, ref_out = run(net, x, exact_knn, amp=False)
            versions = {
                "before": run(net, x, legacy_knn, amp=True),
                "after": run(net, x, FIXED_KNN, amp=True),
                "fp32_uncentred": run(net, x, uncentred_fp32_knn, amp=True),
            }
            for vname, (calls, out) in versions.items():
                for layer, (q, idx), (_q0, idx0) in zip(LAYERS, calls, ref_calls, strict=True):
                    share = kept(idx, idx0)[0]
                    tall = q[0, :, 2] > TALL_M
                    cell = acc.setdefault(vname, {}).setdefault(
                        layer, {"n": 0, "s": 0.0, "nt": 0, "st": 0.0}
                    )
                    cell["n"] += share.numel()
                    cell["s"] += float(share.sum())
                    cell["nt"] += int(tall.sum())
                    cell["st"] += float(share[tall].sum())
                if vname == "fp32_uncentred":
                    continue
                for layer in ("sa1", "sa2", "sa3", "fp3", "fp2", "fp1", "head"):
                    a, r = out[layer], ref_out[layer]
                    err = float(torch.linalg.norm(a - r) / torch.linalg.norm(r))
                    acc[vname].setdefault(f"out_{layer}", []).append(err)
                # The head's own output is (batch, classes, points), before the model transposes it.
                pred, pred0 = out["head"][0].argmax(0), ref_out["head"][0].argmax(0)
                tall = x[0, :, 2] > TALL_M
                cls = acc[vname].setdefault("class", {"n": 0, "s": 0, "nt": 0, "st": 0})
                cls["n"] += pred.numel()
                cls["s"] += int((pred == pred0).sum())
                cls["nt"] += int(tall.sum())
                cls["st"] += int((pred == pred0)[tall].sum())
            print(f"{area}: block up to {crop[:, 2].max():.0f} m done", flush=True)
        table = {}
        for vname, layers in acc.items():
            row = {}
            for layer in LAYERS:
                c = layers[layer]
                row[f"{layer}_neighbours_kept"] = round(c["s"] / c["n"], 4)
                row[f"{layer}_neighbours_kept_above_{TALL_M:g}m"] = (
                    round(c["st"] / c["nt"], 4) if c["nt"] else None
                )
            if vname != "fp32_uncentred":
                for layer in ("sa1", "sa2", "sa3", "fp3", "fp2", "fp1", "head"):
                    row[f"{layer}_output_relative_error"] = round(
                        float(np.mean(layers[f"out_{layer}"])), 4
                    )
                c = layers["class"]
                row["class_agreement"] = round(c["s"] / c["n"], 4)
                row[f"class_agreement_above_{TALL_M:g}m"] = (
                    round(c["st"] / c["nt"], 4) if c["nt"] else None
                )
            table[vname] = row
        result["areas"][area] = {"blocks": n, "versions": table}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
