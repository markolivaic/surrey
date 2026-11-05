"""No DALES-2 label reaches the octrees hosted for the page.

The upstream annotations carry a non-commercial restriction the hosted predictions must not take on,
and a published label cannot be taken back. The octrees are built locally and not committed, so the
second half runs where they are.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_hosted_surrey_octrees_carry_predictions_not_dales_labels():
    source = (ROOT / "scripts" / "pipeline" / "build_octree.py").read_text(encoding="utf-8")
    assert not re.search(r"\.label\b|label\.npy", source), "build_octree.py reads DALES labels"
    viewer = ROOT / "data" / "viewer"
    if not viewer.exists():
        pytest.skip("the octrees are built locally and not committed")
    check_file = ROOT / "data" / "artifacts" / "octree_decode_check.json"
    check = json.loads(check_file.read_text(encoding="utf-8"))
    for manifest in viewer.glob("*/manifest.json"):
        m = json.loads(manifest.read_text(encoding="utf-8"))
        assert m.get("dales_labels_included") is False, manifest
        if m.get("city") == "surrey":
            assert check["areas"][f"surrey_{m['area']}"]["decoder_round_trip_exact"], manifest
