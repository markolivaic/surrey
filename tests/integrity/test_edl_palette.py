"""Each theme's eye-dome strength is the one the palette's rules chose, and it keeps them.

Eye-dome lighting multiplies every drawn colour by a shade, and the palette's contrast and colour
vision rules were written for undarkened colours. scripts/investigations/edl_sweep.py measured the median shade on
the live page for each strength on a fine grid, per theme, on the first screen and on card 1, and
chose per theme by a rule written before it ran: the largest strength at which every rule passes on
both scenes, with every smaller strength passing too. These tests hold the page to that choice, and
read every rule again at the measured shades with the palette as it now is: a palette change the sweep
was not re-read for shows here.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.palette import check_theme, load_palette  # noqa: E402

SWEEP = ROOT / "data" / "artifacts" / "edl_sweep.json"


def sweep() -> dict:
    if not SWEEP.exists():
        pytest.skip("edl_sweep.json has not been measured")
    return json.loads(SWEEP.read_text(encoding="utf-8"))


def violations(doc: dict, theme: str, strength: float) -> list[str]:
    """Every rule the measurement shows broken at this strength in this theme, on any scene."""
    cells = [
        c
        for c in doc["cells"].values()
        if c["theme"] == theme and abs(c["strength"] - strength) < 1e-9
    ]
    if not cells:
        return [f"{theme} at {strength} was never measured"]
    pal = load_palette()
    d = pal["themes"][theme]
    floor = pal["floor"] if pal["floor"] is not None else 0.2
    return [
        f"{c['scene']}: {f}"
        for c in cells
        for f in check_theme(d["tokens"], d["ground"], floor, c["shade_median"])["failures"]
    ]


def by_rule(doc: dict, theme: str) -> float:
    """The rule itself: the largest grid strength that passes with every smaller one passing."""
    chosen = 0.0
    for s in sorted(doc["grid"]):
        if violations(doc, theme, s):
            break
        chosen = s
    return chosen


def test_the_page_takes_each_themes_strength_from_the_sweep():
    main = (ROOT / "web" / "src" / "main.js").read_text(encoding="utf-8")
    assert 'import edlSweep from "../../data/artifacts/edl_sweep.json"' in main
    assert re.search(r"edlSweep\.chosen\[theme\]\.strength", main), (
        "main.js no longer reads the sweep's choice"
    )


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_chosen_strength_is_the_rules_and_keeps_every_rule(theme):
    doc = sweep()
    chosen = doc["chosen"][theme]["strength"]
    assert chosen == by_rule(doc, theme), (
        f"{theme}: chosen {chosen}, the rule gives {by_rule(doc, theme)}"
    )
    broken = violations(doc, theme, chosen)
    assert not broken, f"{theme} at {chosen} breaks the palette rules:\n" + "\n".join(broken)


def test_self_test_the_strongest_strength_fails_in_the_dark_theme():
    doc = sweep()
    assert violations(doc, "dark", 0.6), "the dark theme at 0.6 passed, so the check proves nothing"
