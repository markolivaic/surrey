"""No em dash, en dash or middle dot in any published text; a plain hyphen is fine.

Two licence attributions carry an en dash and have to be reproduced exactly, so those two strings,
and nothing else, are exempt. Third-party code under web/vendor is not ours to reword.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from published import published, rel  # noqa: E402

BANNED = {"\u2014": "em dash", "\u2013": "en dash", "\u00b7": "middle dot"}
ATTRIBUTIONS = (
    "Contains information licensed under the Open Government License \u2013 City of Surrey.",
    "Contains information licensed under the Open Government Licence \u2013 British Columbia.",
)
TEXT = {
    ".md",
    ".py",
    ".js",
    ".mjs",
    ".css",
    ".html",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".sh",
    "",
}


def banned_marks(text: str) -> list[str]:
    for exempt in ATTRIBUTIONS:
        text = text.replace(exempt, "")
    return [name for char, name in BANNED.items() if char in text]


def test_no_banned_punctuation_in_any_published_text():
    hits = [
        f"{rel(p)}: {', '.join(banned_marks(p.read_text(encoding='utf-8')))}"
        for p in published()
        if p.suffix in TEXT
        and not rel(p).startswith("web/vendor/")
        and banned_marks(p.read_text(encoding="utf-8"))
    ]
    assert not hits, "\n".join(hits)


def test_self_test_the_check_stops_the_dashes_and_lets_a_hyphen_and_the_attributions_through():
    assert banned_marks("Wire \u2013 Conductor") == ["en dash"]
    assert banned_marks("a pause \u2014 then") == ["em dash"]
    assert banned_marks("pole \u00b7 fence") == ["middle dot"]
    assert banned_marks("Wire-Conductor (Phase)") == []
    assert banned_marks(ATTRIBUTIONS[0] + " " + ATTRIBUTIONS[1]) == []
    assert banned_marks(ATTRIBUTIONS[0] + " and one more \u2013 here") == ["en dash"]
