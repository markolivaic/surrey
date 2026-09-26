"""No published file names the author in the third person, names the tool that helped write it, or says I.

The author's name belongs in LICENSE and nowhere else. Third-party code under web/vendor is not ours
to reword. One section may say I: "Why this exists" in the README and its template, which the author
speaks in his own voice. Only the first person is allowed there; every other banned form still fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from published import published, rel  # noqa: E402

# First person: a lone capital i before a lowercase word. Code such as a struct format or the
# abbreviation for input and output does not match.
FIRST_PERSON = re.compile(r"(?<![\w/.\-'`\"<])I (?=[a-z])")
BANNED = [
    re.compile(r"\bMarko\b"),
    re.compile(r"\bClaude\b"),
    re.compile(r"\bthe agent\b", re.I),
    re.compile(r"\bmy behalf\b", re.I),
    FIRST_PERSON,
]
EXEMPT = {"LICENSE"}
EXEMPT_DIRS = ("web/vendor/",)
BINARY = {".png", ".webp", ".jpg", ".webm", ".bin", ".laz", ".ico", ".woff2", ".mp4", ".gif"}
OWN_VOICE_FILES = {"README.md", "docs/readme.template.md"}
OWN_VOICE_SECTION = "## Why this exists"


def voice_hits(text: str, own_voice: bool = False) -> list[str]:
    patterns = [p for p in BANNED if not (own_voice and p is FIRST_PERSON)]
    return [m.group(0) for pat in patterns for m in pat.finditer(text)]


def own_voice_lines(lines: list[str]) -> set[int]:
    """The line numbers of the author's own section, from its heading to the next heading."""
    inside, found = False, set()
    for i, line in enumerate(lines, 1):
        if line.startswith("## "):
            inside = line.strip() == OWN_VOICE_SECTION
        if inside:
            found.add(i)
    return found


def test_no_published_file_carries_the_agent_voice():
    hits = []
    for path in published():
        name = rel(path)
        if name in EXEMPT or name.startswith(EXEMPT_DIRS) or path.suffix in BINARY:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        own = own_voice_lines(lines) if name in OWN_VOICE_FILES else set()
        for i, line in enumerate(lines, 1):
            hits += [f"{name}:{i}: {w!r}" for w in voice_hits(line, own_voice=i in own)]
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


def test_only_the_authors_section_may_say_i():
    lines = [
        "## Why this exists",
        "I wanted to see that gap.",
        "## San Francisco",
        "I ran the model.",
    ]
    own = own_voice_lines(lines)
    assert own == {1, 2}
    assert not voice_hits(lines[1], own_voice=True)
    assert voice_hits(lines[3], own_voice=4 in own)
    # Inside the section the other forms still fail.
    assert voice_hits("Written by Cla" + "ude.", own_voice=True)
