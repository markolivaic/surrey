"""Every file under docs/design is read by the page, the README, the code or a test.

docs/design once held 101 MB of sheets and shots from the design rounds. Those now stay outside the
repository, and a file may sit here only if something that ships reads it by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from published import published, rel  # noqa: E402

DESIGN = "docs/design/"
READERS = ("README.md", "web/index.html", "web/src/", "backend/", "scripts/", "tests/")


def unread(files: list[str], texts: str) -> list[str]:
    return [f for f in files if f.rsplit("/", 1)[-1] not in texts and f not in texts]


def test_every_design_file_is_read_by_something_that_ships():
    files = [
        rel(p) for p in published() if rel(p).startswith(DESIGN) and rel(p) != DESIGN + "README.md"
    ]
    texts = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in published()
        if rel(p).startswith(READERS) and p.suffix in {".md", ".html", ".js", ".css", ".py"}
    )
    strays = unread(files, texts)
    assert not strays, "files under docs/design that nothing reads:\n" + "\n".join(strays)


def test_self_test_an_unread_file_is_caught():
    assert unread(
        ["docs/design/stray.png", "docs/design/palette.json"], 'import "docs/design/palette.json"'
    ) == ["docs/design/stray.png"]
