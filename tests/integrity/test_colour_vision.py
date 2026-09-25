"""The palette against its rules, on tokens and on the committed bug frames, in both themes.

Rules and reasons in backend/surrey/palette.py. The token checks are quick; the frame checks
load four committed PNGs and simulate every changed pixel. Both carry a self-test: the first,
red-green palette must fail the token rules, and the frames drawn with it must fail the bug
frame rule, or the guard proves nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from surrey.palette import (  # noqa: E402
    MIN_BUG_RETENTION,
    PALETTE_FILE,
    check_theme,
    choose_floor,
    load_palette,
    old_palette,
)

FRAMES = ROOT / "docs" / "design" / "frames"
BUG_PAIRS = {
    "light": (
        "ggpark_ret0_oblique_e6e9e6_edl00_p2.webp",
        "ggpark_returns_ood_oblique_e6e9e6_edl00_p2.webp",
    ),
    "dark": (
        "ggpark_ret0_oblique_131815_edl00_p2.webp",
        "ggpark_returns_ood_oblique_131815_edl00_p2.webp",
    ),
}
OLD_BUG_PAIR = ("ggpark_ret0_oblique_e6e9e6.webp", "ggpark_returns_ood_oblique_e6e9e6.webp")


def retention(a_name: str, b_name: str) -> dict:
    from cvd_bug_frames import load, measure

    return measure(load(FRAMES / a_name), load(FRAMES / b_name))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_palette_tokens_pass_every_rule(theme):
    pal = load_palette()
    d = pal["themes"][theme]
    floor = pal["floor"] if pal["floor"] is not None else 0.2
    failures = check_theme(d["tokens"], d["ground"], floor)["failures"]
    assert not failures, failures


def test_the_floor_is_the_one_the_rule_chooses():
    pal = load_palette()
    assert pal["floor"] == choose_floor(pal)
    if pal["floor"] is None:
        assert "hides the class" in pal["floor_note"]


def test_self_test_the_first_palette_fails_the_token_rules():
    old = old_palette()
    failures = [
        f
        for th, d in old["themes"].items()
        for f in check_theme(d["tokens"], d["ground"], old["floor"])["failures"]
    ]
    assert failures, "the red-green palette passed; the rules are not checking anything"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_bug_frames_keep_the_difference_for_red_green_dichromacy(theme):
    r = retention(*BUG_PAIRS[theme])
    for kind in ("deuteranopia", "protanopia"):
        assert r[kind]["share_of_normal"] >= MIN_BUG_RETENTION, (theme, kind, r[kind])


def test_self_test_the_first_palette_frames_fail_the_bug_rule():
    r = retention(*OLD_BUG_PAIR)
    assert r["deuteranopia"]["share_of_normal"] < MIN_BUG_RETENTION, r


def test_isolated_points_reach_three_to_one_on_real_frames():
    doc = json.loads(
        (ROOT / "docs" / "design" / "sparse_contrast_check.json").read_text(encoding="utf-8")
    )
    for theme, groups in doc["themes"].items():
        for name, stats in groups.items():
            assert stats["pixels"] > 0, (theme, name)
            assert stats["passes"], (theme, name, stats)


def test_the_palette_file_names_its_rules_and_map():
    pal = json.loads(PALETTE_FILE.read_text(encoding="utf-8"))
    assert pal["rules"] == "backend/surrey/palette.py"
    assert pal["confidence_map"] == "data/artifacts/confidence_mapping.json"
