"""A figure's words agree with what the figure counts.

The registry guards check that a number exists and carries a variant. They did not check the verb in
front of it. The first screen said "trained on 40 mbendjilali/DALES-2 tiles, 395,996,290 points", both
numbers true and both counting every tile prepared, the eleven held out for testing among them. The
model was trained on 29 tiles and 288,048,861 points. A true number under the wrong word.

So the claim words are read where the README template, the page and the provenance strip set a
figure, and each has to agree with the population of the registry key behind it:

- "trained on", "in training": the key's split is the training split.
- "held out", "held-out", "never trained on", "not trained on": the key's split is held out.
- "scored on": the key's split is not "all", since a score over everything includes training.
- "as delivered": the key comes from the one variant that saw the input as delivered, returns_ood.

A figure is governed by the nearest claim word before it in its sentence, line or strip cell, and
if none comes before, by the nearest after. A self-test feeds the rule the strip's old sentence.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUANTITIES = ROOT / "data" / "artifacts" / "quantities.json"
KEY = re.compile(r"\{\{([a-z0-9_.:]+)\}\}", re.I)

HELD_TOKENS = ("held_out", "holdout", "never_trained")
CLAIMS = [  # longest first, so "never trained on" is not read as "trained on"
    ("never trained on", "held_out"),
    ("not trained on", "held_out"),
    ("held-out", "held_out"),
    ("held out", "held_out"),
    ("trained on", "trained"),
    ("in training", "trained"),
    ("scored on", "scored"),
    ("as delivered", "delivered"),
]


def agrees(kind: str, pop: dict) -> bool:
    split = pop["split"]
    if kind == "trained":
        return split == "train" or "in_training" in split
    if kind == "held_out":
        return split == "test" or any(t in split for t in HELD_TOKENS)
    if kind == "scored":
        return split != "all"
    if kind == "delivered":
        return pop["variant"] == "returns_ood"
    raise ValueError(kind)


def claims_in(text: str) -> list[tuple[int, int, str, str]]:
    """Claim words with their spans, longest first and never overlapping."""
    found, taken = [], []
    low = text.lower()
    for phrase, kind in CLAIMS:
        for m in re.finditer(re.escape(phrase), low):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.end(), phrase, kind))
    return sorted(found)


def check(unit: str, text: str, reg: dict) -> list[str]:
    out = []
    claims = claims_in(text)
    keys = list(KEY.finditer(text))
    for i, m in enumerate(keys):
        key = m.group(1)
        if key.startswith("table:") or key not in reg or not claims:
            continue
        before = [c for c in claims if c[1] <= m.start()]
        # A claim word after a figure reaches back only to the figure right before it: in "the grouping
        # reaches 1.2 m against 0.9 m on the tiles the model trained on", it speaks for the 0.9 alone.
        next_start = keys[i + 1].start() if i + 1 < len(keys) else len(text)
        after = next((c for c in claims if m.end() <= c[0] < next_start), None)
        governing = before[-1] if before else after
        if after and before and after[0] - m.end() < m.start() - before[-1][1]:
            governing = after
        if governing and not agrees(governing[3], reg[key]["population"]):
            pop = reg[key]["population"]
            out.append(
                f'{unit}: "{governing[2]}" governs {key}, whose split is {pop["split"]}'
                f" and variant {pop['variant']}"
            )
    return out


def pieces(text: str) -> list[str]:
    """Sentences and lines: a claim word does not reach across either."""
    return [
        p for line in text.splitlines() for p in re.split(r"(?<=[.;:!?])\s+", line) if KEY.search(p)
    ]


def readme_units() -> dict[str, str]:
    """The README's template and the method notes' template, which the README links to."""
    units = {}
    for label, name in (("README", "readme.template.md"), ("method", "method.template.md")):
        path = ROOT / "docs" / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            units |= {f"{label}: {p[:50]}": p for p in pieces(text)}
    return units


def page_units() -> dict[str, str]:
    src = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    src = re.sub(r'<span data-q="([^"]+)"[^>]*></span>', r"{{\1}}", src)
    src = re.sub(r"<br\s*/?>|</(p|div|h2|section|li)>", "\n", src)
    text = html.unescape(re.sub(r"<[^>]+>", " ", src))
    text = "\n".join(" ".join(line.split()) for line in text.splitlines())
    return {f"page: {p[:50]}": p for p in pieces(text)}


def strip_units() -> dict[str, str]:
    """Each cell of the provenance strip, label and value, with its keys as {{key}}."""
    src = (ROOT / "web" / "src" / "hero.js").read_text(encoding="utf-8")
    body = src[src.index("export function drawStrip") :]
    body = body[body.index("const cells = [") : body.index("];")]
    body = body.replace("${HERO_AREA}", "sunset")
    body = re.sub(r'(?:fmt\()?get\((?:"|`)([a-z0-9_.]+)(?:"|`)\)\)?', r"{{\1}}", body)
    body = re.sub(r"\$\{(\{\{[a-z0-9_.]+\}\})\}", r"\1", body)
    units = {}
    for label, value in re.findall(
        r'\[\s*"([^"]+)",(.*?)\],?[ \t]*(?=\n\s*(?:\[|//|$))', body, re.S
    ):
        value = " ".join(re.sub(r"[`+]", " ", value).split())
        units[f"strip: {label}"] = f"{label} {value}"
    return units


def m1card_units() -> dict[str, str]:
    """The M1 card's text, built in JavaScript, with each figure it prints as {{key}}."""
    src = (ROOT / "web" / "src" / "m1card.js").read_text(encoding="utf-8")
    src = re.sub(
        r'\$\{spread\("([A-Za-z0-9_.]+)"\)\}', r"{{\1.median}} ({{\1.min}} to {{\1.max}})", src
    )
    src = re.sub(r'\$\{v\("([A-Za-z0-9_.]+)"\)\}', r"{{\1}}", src)
    # A paragraph is one line, whatever its source wrapping, so a claim word reaches every figure in its
    # sentence; paragraphs and captions stay apart.
    src = re.sub(r"</(p|figcaption|div|figure)>", "\x00", src)
    text = html.unescape(re.sub(r"<[^>]+>", " ", src))
    text = "\n".join(" ".join(part.split()) for part in text.split("\x00"))
    return {f"m1card: {p[:50]}": p for p in pieces(text)}


def registry() -> dict:
    return json.loads(QUANTITIES.read_text(encoding="utf-8"))["quantities"]


def test_claim_words_agree_with_their_populations():
    reg = registry()
    units = readme_units() | page_units() | strip_units() | m1card_units()
    found = [p for unit, text in units.items() for p in check(unit, text, reg)]
    assert not found, "\n".join(found)


def test_the_m1_card_is_read_at_all():
    units = m1card_units()
    assert any("swiss_holdout.m1_seeds.water.iou.median" in t for t in units.values()), sorted(
        units
    )[:5]


def test_the_strip_is_read_at_all():
    units = strip_units()
    assert "strip: trained on" in units and "strip: held out" in units, sorted(units)
    assert KEY.search(units["strip: trained on"]), units["strip: trained on"]


def test_self_test_the_old_strip_sentence_fails():
    reg = registry()
    old = "trained on {{provenance.dales.tiles}} mbendjilali/DALES-2 tiles, {{provenance.dales.points}} points"
    found = check("self test", old, reg)
    assert len(found) == 2, found
    new = (
        "trained on {{provenance.dales.train.tiles}} mbendjilali/DALES-2 tiles,"
        " {{surrey.train.voxel.training_points}} points"
    )
    assert not check("self test", new, reg)
    assert check("self test", "held out {{surrey.train.voxel.training_points}} points", reg)
