"""How much faster the rented card is than the laptop, on the same configuration, from the artefacts.

Both sides are read from what they wrote: the laptop's published M0 run and the rented card's probe at
the same width and batch. Steps a second is taken as steps over wall seconds where a whole run is the
source, and as the probe's own timed figure where the probe is, and each is labelled, because the two
are not the same measurement: a whole run carries its model build, its data loading and its final
evaluation, and the probe times eight steps in the middle of a run.

Run:
    python scripts/investigations/rented_speedup.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "data" / "artifacts"
OUT = ARTIFACTS / "rented_speedup.json"


def main() -> int:
    laptop = json.loads((ARTIFACTS / "train_m0_seedA.json").read_text(encoding="utf-8"))
    probe = json.loads((ARTIFACTS / "rented_probe.json").read_text(encoding="utf-8"))
    bench = json.loads((ARTIFACTS / "bench_rented_200.json").read_text(encoding="utf-8"))
    small = next(r for r in probe["ladder"] if r["width"] == 128 and r["batch"] == 4)
    laptop_rate = laptop["steps_trained"] / laptop["wall_seconds"]
    bench_rate = bench["steps_trained"] / bench["wall_seconds"]
    result = {
        "same_configuration": {
            "width": 128,
            "batch": 4,
            "points_per_crop": laptop["config"]["points"],
        },
        "laptop": {
            "card": laptop["device_name"],
            "steps": laptop["steps_trained"],
            "wall_seconds": laptop["wall_seconds"],
            "steps_per_second_whole_run": round(laptop_rate, 3),
            "note": "a whole 12,000 step run, model build and final evaluation included",
        },
        "rented": {
            "card": probe["card"]["name"],
            "memory_gb": probe["card"]["memory_gb"],
            "steps_per_second_probe": small["steps_per_second"],
            "steps_per_second_whole_run": round(bench_rate, 3),
            "bench_steps": bench["steps_trained"],
            "bench_wall_seconds": bench["wall_seconds"],
            "note": "the probe times eight steps in the middle; the bench is a whole 200 step run",
        },
        "speedup_like_for_like": round(small["steps_per_second"] / laptop_rate, 1),
        "speedup_whole_runs": round(bench_rate / laptop_rate, 1),
        "a_full_run_here_minutes": round(
            laptop["steps_trained"] / small["steps_per_second"] / 60, 1
        ),
        "a_full_run_on_the_laptop_minutes": round(laptop["wall_seconds"] / 60, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
