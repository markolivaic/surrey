"""The walkthrough is either real media, embedded, or plainly not there yet. Never anything between.

One rule for both states, so the repository is publishable before the recording and after it (kessler
once had two tests that contradicted each other, and could never pass once it succeeded):

- if docs/walkthrough/app-walkthrough.mp4 exists, the MP4, the GIF and the poster are what their first
  bytes say, the MP4 is under 5 MB and the GIF under 8 MB, and the README embeds the GIF linked to the
  MP4 with the poster beside it;
- if it does not, the README's walkthrough section says it is still to come and embeds nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "docs" / "walkthrough"
MP4, GIF, POSTER = (
    MEDIA / "app-walkthrough.mp4",
    MEDIA / "app-walkthrough.gif",
    MEDIA / "app-walkthrough-poster.jpg",
)


def walkthrough_section() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## Walkthrough\n(.*?)(?=^## )", text, re.M | re.S)
    assert m, "the README has no Walkthrough section"
    return m.group(1)


def test_the_walkthrough_is_real_media_or_plainly_absent():
    section = walkthrough_section()
    if MP4.exists():
        assert MP4.read_bytes()[4:8] == b"ftyp", "the MP4 is not an MP4"
        assert GIF.read_bytes()[:6] == b"GIF89a", "the GIF is not a GIF89a"
        assert POSTER.read_bytes()[:3] == b"\xff\xd8\xff", "the poster is not a JPEG"
        assert MP4.stat().st_size < 5_000_000 and GIF.stat().st_size < 8_000_000
        assert (
            "docs/walkthrough/app-walkthrough.gif" in section
            and "docs/walkthrough/app-walkthrough.mp4" in section
        )
        assert "No screen is mocked" in section
    else:
        assert not re.search(r"\.(gif|mp4|jpg)\b", section), (
            "the README embeds media that does not exist"
        )
        assert re.search(r"after|to come|not yet", section), (
            "with no media, the section must say it is to come"
        )
