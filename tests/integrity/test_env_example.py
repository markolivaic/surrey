""".env.example lists exactly the variables the code reads.

A setup file that documents a variable the code no longer reads, or misses one it does, sends a
stranger down a broken path. The variables are found where the code reads them, by the calls that read
the environment, in the backend, the scripts and the page's sources. Variables the operating system
sets for every program, and which no reader configures, are named here and left out.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Set by the system, not by a reader: LOCALAPPDATA by Windows, a place to look for installed tools, and
# MODE by Vite, the build mode the page was built in.
SYSTEM = {"LOCALAPPDATA", "MODE"}
READ = re.compile(
    r"""(?:os\.environ\.get|os\.getenv|os\.environ\[)\(?\s*["']?([A-Z][A-Z0-9_]*)["']?"""
)
CONSTANT = re.compile(r"""^([A-Z][A-Z0-9_]*_ENV)\s*=\s*["']([A-Z][A-Z0-9_]*)["']""", re.M)


def used() -> set[str]:
    names = set()
    for path in [*(ROOT / "backend").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        constants = dict(CONSTANT.findall(text))
        for name in READ.findall(text):
            names.add(constants.get(name, name))
    for path in (ROOT / "web" / "src").rglob("*.js"):
        names |= set(
            re.findall(
                r"(?:process\.env|import\.meta\.env)\.([A-Z][A-Z0-9_]*)",
                path.read_text(encoding="utf-8"),
            )
        )
    return names - SYSTEM


def declared() -> set[str]:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return {
        line.split("=", 1)[0].strip()
        for line in text.splitlines()
        if re.match(r"^[A-Z][A-Z0-9_]*=", line)
    }


def test_env_example_matches_what_the_code_reads():
    assert used() - declared() == set(), "read by the code but not in .env.example"
    assert declared() - used() == set(), "in .env.example but read nowhere"


def test_the_reader_finds_the_tool_variables_at_all():
    assert {"SURREY_FFMPEG", "SURREY_POTREECONVERTER"} <= used()
