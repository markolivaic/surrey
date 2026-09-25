"""The files a clone of this repository holds, as git itself reads .gitignore.

In a git checkout that is the tracked files plus any untracked file git would pick up. Before the
repository has a history, git is pointed at a scratch directory for its own bookkeeping and asked the
same question about this working tree, which writes nothing here.
"""

from __future__ import annotations

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


@lru_cache(maxsize=1)
def published() -> tuple[Path, ...]:
    if (ROOT / ".git").exists():
        out = _git("-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard")
    else:
        with tempfile.TemporaryDirectory() as scratch:
            _git("init", "-q", scratch)
            out = _git(
                f"--git-dir={Path(scratch) / '.git'}",
                f"--work-tree={ROOT}",
                "ls-files",
                "--others",
                "--exclude-standard",
            )
    return tuple(
        sorted(ROOT / line for line in out.splitlines() if line and (ROOT / line).is_file())
    )


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()
