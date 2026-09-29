"""The README keeps its disclosures on its first screen.

Every prose match runs on text with whitespace collapsed, so re-wrapping a sentence cannot disarm it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def disclosure_problems(readme: str) -> list[str]:
    head = re.sub(r"\s+", " ", " ".join(readme.split("\n")[:45])).lower()
    whole = re.sub(r"\s+", " ", readme).lower()
    problems = []
    if "surrey" not in head or not ("british columbia" in head or "canada" in head):
        problems.append(
            "the first screen does not say the lidar was flown over Surrey, British Columbia"
        )
    if "san francisco" not in head:
        problems.append("the first screen does not name the city in the pictures")
    if not re.search(r"not\s+(flown\s+)?(in|over)\s+dayton|no\s+dayton|not\s+dayton", head):
        problems.append("the first screen does not say the data is not from Dayton")
    if "prediction" not in whole or not re.search(
        r"not\s+(a\s+)?measurement|not\s+ground\s+truth", whole
    ):
        problems.append("the README does not say the colours are predictions and not measurements")
    if not re.search(r"^#+\s*what this is not", readme, re.I | re.M):
        problems.append("no 'What this is NOT' section")
    if not re.search(r"^#+\s*limitations", readme, re.I | re.M):
        problems.append("no 'Limitations' section")
    return problems


def test_the_readme_keeps_its_disclosures():
    problems = disclosure_problems((ROOT / "README.md").read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems)


def test_self_test_a_readme_without_them_fails():
    assert len(disclosure_problems("# surrey\n\nA point cloud model.\n")) == 6
