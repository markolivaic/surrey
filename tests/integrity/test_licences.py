"""The terms the data licences put on anything published from them.

Each dataset's licence sets terms the repository has to carry, and a breach cannot be taken back once
it is public: the City of Surrey attribution verbatim, the non-commercial restriction travelling with
the weights and predictions, and no DALES-2 label in the hosted octrees.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "data" / "artifacts"

# The Open Government Licence attribution, reproduced exactly as the licence gives it.
SURREY_ATTRIBUTION = (
    "Contains information licensed under the Open Government License \u2013 City of Surrey."
)


def collapsed(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def licence_problems(code: str, model: str, readme: str, card: str, notices: str) -> list[str]:
    problems = []
    if "code in this repository only" not in code:
        problems.append("LICENSE does not limit itself to the code")
    if "CC BY-NC 4.0" not in model or "weights" not in model or "predictions" not in model:
        problems.append(
            "LICENSE_MODEL.md does not license the weights and predictions CC BY-NC 4.0"
        )
    for name, text in (("README.md", readme), ("the dataset card", card)):
        if (
            "LICENSE_MODEL.md" not in text
            or "CC BY-NC 4.0" not in text
            or "non-commercial" not in text.lower()
        ):
            problems.append(
                f"{name} does not point to the non-commercial licence of the predictions"
            )
    if SURREY_ATTRIBUTION not in notices:
        problems.append(
            "THIRD_PARTY_NOTICES.md does not carry the City of Surrey attribution verbatim"
        )
    if (
        not ("Open Government Licence" in notices or "Open Government License" in notices)
        or "Creative Commons 3.0" not in notices
        or "MIT" not in notices
    ):
        problems.append("THIRD_PARTY_NOTICES.md does not name all three licence layers")
    return problems


def test_the_published_texts_carry_the_licence_terms():
    texts = [
        collapsed(ROOT / n)
        for n in (
            "LICENSE",
            "LICENSE_MODEL.md",
            "README.md",
            "docs/hosting/dataset_card.md",
            "THIRD_PARTY_NOTICES.md",
        )
    ]
    problems = licence_problems(*texts)
    assert not problems, "\n".join(problems)


def test_self_test_a_card_that_forgets_the_restriction_fails():
    code = "This licence covers the code in this repository only."
    model = "weights and predictions: CC BY-NC 4.0"
    readme = "weights and predictions: CC BY-NC 4.0, LICENSE_MODEL.md, non-commercial"
    notices = f"Open Government Licence, Creative Commons 3.0, MIT. {SURREY_ATTRIBUTION}"
    assert not licence_problems(code, model, readme, readme, notices)
    assert licence_problems(code, model, readme, "predictions released with this project", notices)
    assert licence_problems(code, model, readme, readme, notices.replace("\u2013", "-"))


def test_hosted_surrey_octrees_carry_predictions_not_dales_labels():
    source = (ROOT / "scripts" / "pipeline" / "build_octree.py").read_text(encoding="utf-8")
    assert not re.search(r"\.label\b|label\.npy", source), "build_octree.py reads DALES labels"
    viewer = ROOT / "data" / "viewer"
    if not viewer.exists():
        pytest.skip("the octrees are built locally and not committed")
    check = json.loads((ARTIFACTS / "octree_decode_check.json").read_text(encoding="utf-8"))
    for manifest in viewer.glob("*/manifest.json"):
        m = json.loads(manifest.read_text(encoding="utf-8"))
        assert m.get("dales_labels_included") is False, manifest
        if m.get("city") == "surrey":
            assert check["areas"][f"surrey_{m['area']}"]["decoder_round_trip_exact"], manifest
