"""No channel a model reads may be dead in training or out of range on the target.

This is the guard that would have caught, on the first day, the bug the whole transfer
study ended up measuring: the return ratio was exactly 0 on every DALES-2 training
point and between 1/n and 1 on every San Francisco point. Every number was sourced
and consistent, and the input was meaningless.

Reads data/artifacts/input_channels.json, written by scripts/pipeline/input_channel_check.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "data" / "artifacts" / "input_channels.json"
MAJORITY = 0.5
# A channel broken on one flight line or one sensor would put a minority of an area's
# points out of range, and the majority rule alone would pass it. So every area and
# channel above this share must be listed below with the reason it is accepted, the
# same pattern as the UNSOURCED list for numbers. An entry that no longer applies fails
# too, so the list cannot rot into a blanket pass.
LISTED = 0.05
ACCEPTED_EXTRAPOLATIONS = {
    ("financial", "z_above_block_floor"): (
        "vertical extrapolation named before training in scripts/pipeline/input_range_check.py: 26 of "
        "64 Financial blocks rise above the tallest training block, so their upper points "
        "exceed the training z range; measured as a confound, not hidden"
    ),
}


@pytest.fixture(scope="module")
def channels() -> dict:
    assert ARTIFACT.exists(), "run scripts/pipeline/input_channel_check.py"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def dead(variant: dict) -> list[str]:
    return [c for c, s in variant["training_std"].items() if s <= 0.0]


def out_of_range(variant: dict) -> list[str]:
    return [
        f"{area}:{c}={share}"
        for area, per in variant["target_outside_share"].items()
        for c, share in per.items()
        if share > MAJORITY
    ]


def test_the_published_model_reads_no_dead_channel(channels):
    v = channels["variants"][channels["published_variant"]]
    assert not dead(v), f"channels with zero variance in training: {dead(v)}"


def test_the_published_model_stays_in_range_on_every_area(channels):
    v = channels["variants"][channels["published_variant"]]
    assert not out_of_range(v), f"out of the training range on most points: {out_of_range(v)}"


def test_the_guard_would_have_caught_the_return_ratio(channels):
    # The self-test. If this passes for returns_ood, the two tests above are too weak
    # to have caught the bug they exist for.
    v = channels["variants"]["returns_ood"]
    assert "return_ratio" in dead(v)
    assert any(":return_ratio=" in item for item in out_of_range(v))


def test_forcing_the_ratio_to_zero_is_still_a_dead_channel(channels):
    # ret0 is a diagnostic, not a model to publish: in range, but still reading an
    # input that never varied.
    v = channels["variants"]["ret0"]
    assert "return_ratio" in dead(v)
    assert not any(":return_ratio=" in item for item in out_of_range(v))


def test_every_area_is_reported(channels):
    areas = set(channels["san_francisco"])
    for name, v in channels["variants"].items():
        assert set(v["target_outside_share"]) == areas, name


def test_every_area_above_the_listing_share_is_accepted_with_a_reason(channels):
    v = channels["variants"][channels["published_variant"]]
    above = {
        (area, c)
        for area, per in v["target_outside_share"].items()
        for c, share in per.items()
        if share > LISTED
    }
    unlisted = sorted(above - set(ACCEPTED_EXTRAPOLATIONS))
    stale = sorted(set(ACCEPTED_EXTRAPOLATIONS) - above)
    assert not unlisted, (
        f"out of range on more than {LISTED:.0%} of points and not accepted: {unlisted}"
    )
    assert not stale, f"accepted extrapolations that no longer apply, remove them: {stale}"


def test_the_listing_rule_would_catch_a_minority_break(channels):
    # A channel out of range on 30 percent of one area: under the majority bar, over the
    # listing bar, and not accepted, so it must be caught.
    v = channels["variants"][channels["published_variant"]]
    broken = {a: dict(per) for a, per in v["target_outside_share"].items()}
    broken["mission"]["x"] = 0.30
    above = {
        (area, c) for area, per in broken.items() for c, share in per.items() if share > LISTED
    }
    assert ("mission", "x") in above - set(ACCEPTED_EXTRAPOLATIONS)
    assert not any(share > MAJORITY for per in broken.values() for share in per.values())
