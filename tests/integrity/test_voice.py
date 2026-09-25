"""No published file names the author in the third person, names the tool that helped write it, or says I.

The author's name belongs in LICENSE and nowhere else. Third-party code under web/vendor is not ours
to reword.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from published import published, rel  # noqa: E402

BANNED = [
    re.compile(r"\bMarko\b"),
    re.compile(r"\bClaude\b"),
    re.compile(r"\bthe agent\b", re.I),
    re.compile(r"\bmy behalf\b", re.I),
    # First person: a lone capital i before a lowercase word. Code such as a struct format or
    # the abbreviation for input and output does not match.
    re.compile(r"(?<![\w/.\-'`\"<])I (?=[a-z])"),
]
EXEMPT = {"LICENSE"}
EXEMPT_DIRS = ("web/vendor/",)
BINARY = {".png", ".webp", ".jpg", ".webm", ".bin", ".laz", ".ico", ".woff2"}


def voice_hits(text: str) -> list[str]:
    return [m.group(0) for pat in BANNED for m in pat.finditer(text)]


def test_no_published_file_carries_the_agent_voice():
    hits = []
    for path in published():
        name = rel(path)
        if name in EXEMPT or name.startswith(EXEMPT_DIRS) or path.suffix in BINARY:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            hits += [f"{name}:{i}: {w!r}" for w in voice_hits(line)]
    assert not hits, "\n".join(hits)


def test_the_check_catches_each_banned_form():
    # Split so that this file does not trip its own check.
    for line in (
        "Mar" + "ko chose B.",
        "Nothing is run by Cla" + "ude.",
        "not the " + "agent's to close",
        "no account on my " + "behalf",
        "I print the command when it is done.",
    ):
        assert voice_hits(line), line
    for line in (
        'select = ["E", "I", "UP"]',
        'struct.unpack_from("<I", head)',
        "the I/O thread",
        "github.com/markolivaic",
        ".claude/",
    ):
        assert not voice_hits(line), line
