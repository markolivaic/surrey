"""The vegetation reference is built without access to the model's output.

Its NDVI threshold decides what counts as a tree in San Francisco, and the model is
then scored against that. A threshold chosen after seeing the predictions could be
nudged, knowingly or not, toward a flattering or a dramatic result. So the script
that fixes it must not be able to load a prediction or run the model at all, and
this reads its source to hold it to that rather than trusting a docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pipeline" / "naip_reference.py"

FORBIDDEN_IMPORTS = {"torch", "surrey.model", "surrey.inference"}
FORBIDDEN_STRINGS = {
    "predictions",
    "predict_cache",
    "confidence.json",
    "auroc.json",
    "transfer_sf.json",
}


def tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def docstrings(mod: ast.Module) -> set[int]:
    ids = set()
    for node in ast.walk(mod):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def offending(mod: ast.Module) -> list[str]:
    bad = []
    skip = docstrings(mod)
    for node in ast.walk(mod):
        if isinstance(node, ast.Import):
            bad += [
                a.name
                for a in node.names
                if a.name.split(".")[0] in {"torch"} or a.name in FORBIDDEN_IMPORTS
            ]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in FORBIDDEN_IMPORTS or node.module.split(".")[0] == "torch":
                bad.append(node.module)
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
        ):
            parts = {node.value} | set(node.value.replace("\\", "/").split("/"))
            bad += sorted(parts & FORBIDDEN_STRINGS)
    return bad


def test_the_reference_script_cannot_reach_the_model_or_its_output():
    assert SCRIPT.exists()
    assert not offending(tree()), f"naip_reference.py reaches model output: {offending(tree())}"


def test_the_blindness_check_itself_would_notice_a_leak():
    leaky = ast.parse('import torch\nP = ROOT / "data" / "predictions"\n')
    found = offending(leaky)
    assert "torch" in found and "predictions" in found
