"""The README is generated from docs/readme.template.md, and this fails if the committed one is not.

Matching a figure to a quantity by value guesses; naming the key does not. The template holds the key,
scripts/pipeline/render_readme.py fills it in, and this check keeps the committed README equal to what that
produces, so an edited figure or a moved quantity shows up as a failing test rather than as a sentence
nobody re-derived. The older value matching guard stays on beside it for anything typed by hand.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def renderer():
    spec = importlib.util.spec_from_file_location(
        "render_readme", ROOT / "scripts" / "pipeline" / "render_readme.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_committed_readme_is_what_the_template_and_the_registry_produce():
    mod = renderer()
    if not mod.TEMPLATE.exists():
        pytest.skip("no template yet")
    rendered = mod.render(mod.TEMPLATE.read_text(encoding="utf-8"), mod.values())
    current = mod.README.read_text(encoding="utf-8")
    if current != rendered:
        first = next(
            (
                i
                for i, (a, b) in enumerate(
                    zip(current.splitlines(), rendered.splitlines(), strict=False)
                )
                if a != b
            ),
            0,
        )
        pytest.fail(
            "README.md is not what the template produces; run scripts/pipeline/render_readme.py. "
            f"First difference at line {first + 1}:\n"
            f"  committed: {current.splitlines()[first][:160]}\n"
            f"  rendered : {rendered.splitlines()[first][:160]}"
        )


def test_self_test_a_changed_figure_is_caught():
    mod = renderer()
    vals = {"a.b": 0.5}
    assert mod.render("x {{a.b}} y", vals) == "x 0.5 y"
    assert mod.render("x {{a.b}} y", vals) != "x 0.6 y"


def test_a_key_the_registry_does_not_have_stops_the_render():
    mod = renderer()
    with pytest.raises(SystemExit):
        mod.render("{{not.a.real.key}}", {"a.b": 1.0})


def test_every_figure_in_the_template_is_a_key_rather_than_a_number():
    """A figure typed into the template would pass the check above and still be unsourced."""
    import re

    mod = renderer()
    if not mod.TEMPLATE.exists():
        pytest.skip("no template yet")
    text = mod.TEMPLATE.read_text(encoding="utf-8")
    # Not measurements: the licence version, the keep fractions named in words, the thresholds the page
    # was held to, the ordinary counts a sentence needs, 4.29, the laptop card's memory, and versions:
    # 1.4 of the LAS specification, 3.12 of Python and 2.1 of PotreeConverter, in the quick start, and
    # 2.9 and 2.11 of torch, the CI build and the one every model was trained with.
    allowed = {
        "4.0",
        "0.15",
        "0.25",
        "0.6",
        "1.0",
        "2.0",
        "3.0",
        "5.0",
        "20.0",
        "4.29",
        "1.4",
        "3.12",
        "2.1",
        "2.9",
        "2.11",
    }
    bare = [n for n in re.findall(r"(?<![\w.{])(-?\d+\.\d+)(?![\w}])", text) if n not in allowed]
    assert not bare, f"figures typed into the template instead of named: {sorted(set(bare))}"
