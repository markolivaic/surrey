"""No carriage returns in any published text file.

A shell script whose lines end in CRLF dies on its first line with "set: pipefail: invalid option
name", which says nothing about line endings. `.gitattributes` normalises what git commits; this
fails on anything that slipped past it, in the files a clone actually holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from published import published, rel  # noqa: E402

SUFFIXES = {
    ".sh",
    ".py",
    ".yml",
    ".yaml",
    ".js",
    ".mjs",
    ".json",
    ".md",
    ".css",
    ".html",
    ".toml",
    ".txt",
    ".cfg",
    ".ini",
    ".log",
}
NAMES = {".gitignore", ".gitattributes", ".env.example", "LICENSE"}


def with_carriage_returns(paths) -> list[str]:
    return [
        rel(p)
        for p in paths
        if (p.suffix in SUFFIXES or p.name in NAMES) and b"\r" in p.read_bytes()
    ]


def test_no_published_text_file_has_carriage_returns():
    files = [p for p in published() if p.suffix in SUFFIXES or p.name in NAMES]
    assert len(files) > 50, "the sweep found almost nothing, so it is not checking what it claims"
    bad = with_carriage_returns(files)
    assert not bad, "carriage returns in:\n" + "\n".join(bad)


def test_gitattributes_pins_shell_scripts_to_lf():
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in text
    assert "* text=auto eol=lf" in text


def test_self_test_the_check_catches_a_crlf_script(tmp_path, monkeypatch):
    import published as pub

    monkeypatch.setattr(pub, "ROOT", tmp_path)
    (tmp_path / "bad.sh").write_bytes(b"#!/usr/bin/env bash\r\nset -euo pipefail\r\n")
    (tmp_path / "good.sh").write_bytes(b"#!/usr/bin/env bash\nset -euo pipefail\n")
    found = with_carriage_returns([tmp_path / "bad.sh", tmp_path / "good.sh"])
    assert [Path(f).name for f in found] == ["bad.sh"]
