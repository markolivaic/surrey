"""Resolve every registered quantity from the artefacts into one committed file.

data/artifacts/quantities.json is the single place a reader, the README renderer
and the integrity tests all read a quantity from. It records each value with the
population it was measured over, so the same word never again stands for two
numbers without saying which is which.

Run:
    python scripts/pipeline/build_quantities.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.quantities import ARTIFACTS, REGISTRY, load_artifacts, resolve  # noqa: E402

OUT = ARTIFACTS / "quantities.json"


def main() -> int:
    artifacts = load_artifacts()
    artifacts.pop(OUT.name, None)
    values = resolve(REGISTRY, artifacts)
    broken = {k: v["error"] for k, v in values.items() if v["error"]}
    OUT.write_text(
        json.dumps(
            {
                "note": (
                    "one value per quantity, each with the population it was measured "
                    "over. Built from the other artefacts by scripts/pipeline/build_quantities.py"
                ),
                "quantities": values,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"{len(values)} quantities, {len(broken)} unresolved")
    for k, e in broken.items():
        print(f"  unresolved {k}: {e}")
    print(f"wrote {OUT}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
