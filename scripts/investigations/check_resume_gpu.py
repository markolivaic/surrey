"""Resume on the card: does a run stopped at step k and resumed draw the same batches as one that never stopped?

tests/unit/test_resume_exact.py holds the CPU to bit for bit weights. On the GPU some kernels are not
deterministic, so the weights are only reported here, and what must match is the batch order. Two
unbroken runs are compared too, so a weight difference after the resume can be read against the
difference the card makes on its own.

It trains on the test's synthetic tiles for a few steps, so it takes a minute or two, and it must run
with the card free: in a pause, by the script that holds the pause (train.py --ignore-hold).
Writes data/artifacts/resume_check_gpu.json.

Run:
    python scripts/investigations/check_resume_gpu.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "artifacts" / "resume_check_gpu.json"
STEPS, STOP = 6, 3


def load_test_module():
    spec = importlib.util.spec_from_file_location(
        "test_resume_exact", ROOT / "tests" / "unit" / "test_resume_exact.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device", default="cuda", help="cpu only to try the script itself while the card is busy"
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device; this check is for the card")
        return 1
    t = load_test_module()
    common = [a if a != "cpu" else args.device for a in t.COMMON]
    common[common.index("--steps") + 1] = str(STEPS)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        art = t.synthetic_tiles(tmp)

        def train(name: str, *extra: str) -> None:
            cmd = [
                sys.executable,
                str(t.TRAIN),
                *common,
                "--ignore-hold",
                "--prep",
                str(tmp),
                "--split-artifact",
                str(art),
                "--ckpt",
                str(tmp / f"{name}.pt"),
                "--out",
                str(tmp / f"{name}.json"),
                "--batch-log",
                str(tmp / f"{name}.batches"),
                *extra,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise SystemExit(proc.stdout[-2000:] + proc.stderr[-2000:])

        train("unbroken")
        train("unbroken_again")
        train("broken", "--stop-after-step", str(STOP))
        train("broken", "--resume")

        def batches(name: str) -> list[str]:
            return (tmp / f"{name}.batches").read_text(encoding="utf-8").split()

        def weights(name: str) -> dict:
            return torch.load(tmp / f"{name}.pt", map_location="cpu", weights_only=False)["model"]

        def max_abs(a: str, b: str) -> float:
            wa, wb = weights(a), weights(b)
            return max(float((wa[k].double() - wb[k].double()).abs().max()) for k in wa)

        unbroken = json.loads((tmp / "unbroken.json").read_text(encoding="utf-8"))
        result = {
            "device": unbroken["device"],
            "device_name": unbroken["device_name"],
            "torch": unbroken["torch"],
            "steps": STEPS,
            "stopped_after_step": STOP,
            "batch_order_matches_after_resume": batches("broken") == batches("unbroken"),
            "batch_order_matches_between_unbroken_runs": batches("unbroken_again")
            == batches("unbroken"),
            "weights_bit_for_bit_after_resume": max_abs("broken", "unbroken") == 0.0,
            "weights_max_abs_difference_after_resume": max_abs("broken", "unbroken"),
            "weights_max_abs_difference_between_unbroken_runs": max_abs(
                "unbroken_again", "unbroken"
            ),
            "interruptions_recorded": json.loads((tmp / "broken.json").read_text(encoding="utf-8"))[
                "interruptions"
            ],
            "note": "batch order is the requirement on the card; the weights are reported beside the difference "
            "between two unbroken runs",
        }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "interruptions_recorded"}, indent=2))
    return 0 if result["batch_order_matches_after_resume"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
