"""README.md and docs/method.md against the registry of quantities, and the registry against the artefacts.

The templates hold keys, scripts/pipeline/render_readme.py fills them in from data/artifacts/quantities.json,
so an edited figure or a moved quantity shows up here. The per-class IoU table and the mean IoU printed
beside it are also checked to come from one evaluation: a table taken from another pass, such as the single
pass logged during training, fails the per-seed check.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from surrey.quantities import ARTIFACTS, REGISTRY, load_artifacts, resolve

ROOT = Path(__file__).resolve().parents[2]
CLASSES = 15


def renderer():
    spec = importlib.util.spec_from_file_location(
        "render_readme", ROOT / "scripts" / "pipeline" / "render_readme.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def values() -> dict:
    return resolve(REGISTRY, load_artifacts())


def committed_quantities() -> dict:
    return json.loads((ARTIFACTS / "quantities.json").read_text(encoding="utf-8"))["quantities"]


def test_every_registered_quantity_resolves(values):
    broken = {k: v["error"] for k, v in values.items() if v["error"] or v["value"] is None}
    assert not broken, f"quantities that do not resolve: {broken}"


def test_the_committed_quantities_file_is_current(values):
    now = {k: v["value"] for k, v in values.items()}
    then = {k: v["value"] for k, v in committed_quantities().items()}
    drift = {k: (then.get(k), now[k]) for k in now if then.get(k) != now[k]}
    assert not drift, f"quantities.json is stale, run scripts/pipeline/build_quantities.py: {drift}"
    # A deleted artefact drops its quantities from a fresh build while the committed file, and every
    # figure read from it, still carries them.
    gone = sorted(set(then) - set(now))
    assert not gone, f"quantities no artefact produces any more: {gone[:20]}"


@pytest.mark.parametrize("which", [0, 1], ids=["README.md", "docs/method.md"])
def test_the_committed_file_is_what_its_template_and_the_registry_produce(which):
    mod = renderer()
    template, target, short = mod.DOCUMENTS[which]
    rendered = mod.render(template.read_text(encoding="utf-8"), mod.values(), short=short)
    current = target.read_text(encoding="utf-8")
    assert current == rendered, f"{target.name} is stale, run scripts/pipeline/render_readme.py"


def test_a_short_figure_is_fixed_point_with_two_significant_digits():
    mod = renderer()
    for value, text in ((0.000065, "0.000065"), (0.000995, "0.0010"), (0.05, "0.050"), (0.0, "0")):
        assert mod.render("{{a}}", {"a": value}, short=True) == text


def test_a_key_the_registry_does_not_have_stops_the_render():
    with pytest.raises(SystemExit):
        renderer().render("{{not.a.real.key}}", {"a.b": 1.0})


@pytest.mark.parametrize("recipe", ["m0", "m0fp16"])
def test_the_table_mean_is_the_stated_mean_iou(recipe):
    q = {k: v["value"] for k, v in committed_quantities().items()}
    prefix = f"seeds.{recipe}.surrey.test.iou."
    classes = sorted({k[len(prefix) :].rsplit(".", 1)[0] for k in q if k.startswith(prefix)})
    medians = [q[f"{prefix}{c}.median"] for c in classes]
    assert len(medians) == CLASSES
    stated = q[f"seeds.{recipe}.surrey.test.miou_15.median"]
    assert abs(float(np.mean(medians)) - stated) <= 0.005, (np.mean(medians), stated)


@pytest.mark.parametrize("recipe", ["m0", "m0fp16"])
def test_each_seed_s_class_ious_average_to_its_own_mean_iou(recipe):
    doc = json.loads((ARTIFACTS / f"seeds_{recipe}.json").read_text(encoding="utf-8"))
    held = doc["surrey_held_out"]
    for seed, miou in held["miou_15"]["per_seed"].items():
        per_class = [row["per_seed"][seed] for row in held["iou_per_class"].values()]
        assert abs(float(np.mean(per_class)) - miou) <= 1e-3, (seed, np.mean(per_class), miou)
