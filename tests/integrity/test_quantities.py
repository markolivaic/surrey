"""The committed registry is what the committed artefacts say today.

The README and the page read every figure from data/artifacts/quantities.json, so every registered
quantity has to resolve and the file has to match a fresh resolution of the artefacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.quantities import ARTIFACTS, REGISTRY, load_artifacts, resolve  # noqa: E402

QUANTITIES_FILE = ARTIFACTS / "quantities.json"


@pytest.fixture(scope="module")
def values() -> dict:
    return resolve(REGISTRY, load_artifacts())


def test_every_registered_quantity_resolves(values):
    broken = {k: v["error"] for k, v in values.items() if v["error"] or v["value"] is None}
    assert not broken, f"quantities that do not resolve: {broken}"


def test_the_committed_quantities_file_is_current(values):
    assert QUANTITIES_FILE.exists(), "run scripts/pipeline/build_quantities.py"
    committed = json.loads(QUANTITIES_FILE.read_text(encoding="utf-8"))["quantities"]
    now = {k: v["value"] for k, v in values.items()}
    then = {k: v["value"] for k, v in committed.items()}
    drift = {k: (then.get(k), now[k]) for k in now if then.get(k) != now[k]}
    assert not drift, f"quantities.json is stale, rebuild it: {drift}"
