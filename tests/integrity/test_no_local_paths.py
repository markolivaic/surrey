"""No published file carries an absolute local path (PROJECT_STANDARD 10.4).

Every published text is read raw: in a JSON file a Windows path sits with its backslashes doubled, and
that is the form that once leaked into five training artefacts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from published import published, rel  # noqa: E402

BACKSLASH = chr(92)
SEP = f"(?:{re.escape(BACKSLASH)}+|/)"
LOCAL = re.compile(
    rf"[A-Za-z]:{SEP}Users{SEP}|/home/[a-z][a-z0-9_-]*/|/Users/[A-Za-z][A-Za-z0-9_-]*/|/c/U(?:sers)/"
)
BINARY = {".png", ".webp", ".jpg", ".webm", ".bin", ".laz", ".ico", ".woff2", ".mp4", ".gif"}


def test_no_local_absolute_path():
    hits = []
    for path in published():
        # This file holds the forms it looks for, as examples for its own self-test.
        if path.suffix in BINARY or path.resolve() == Path(__file__).resolve():
            continue
        m = LOCAL.search(path.read_text(encoding="utf-8", errors="ignore"))
        if m:
            hits.append(f"{rel(path)}: {m.group(0)!r}")
    assert not hits, "\n".join(hits)


def test_self_test_the_pattern_catches_the_forms_that_leaked():
    b = BACKSLASH
    leaked = [
        f"C:{b}Users{b}someone",
        f"C:{b}{b}Users{b}{b}someone",
        "C:/" + "Users/someone/",
        "/home/someone/x",
        "/c/" + "Users/someone",
    ]
    for text in leaked:
        assert LOCAL.search(text), text
    assert not LOCAL.search("data/checkpoints/m0_seedA.pt")
