"""Self-test for tests/smoke/conftest.py: with CI=true and Playwright hidden, the smoke job fails.

Playwright is hidden by putting a package named playwright first on the path whose import raises,
which is what a runner without it looks like to pytest.importorskip.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_smoke(tmp_path: Path, ci: str) -> subprocess.CompletedProcess:
    fake = tmp_path / "no_playwright" / "playwright"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("raise ImportError('Playwright hidden for the self-test')\n")
    env = dict(
        os.environ,
        CI=ci,
        PYTHONPATH=str(fake.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(ROOT / "tests" / "smoke"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


def test_in_ci_a_skipped_smoke_test_fails_the_job(tmp_path):
    proc = run_smoke(tmp_path, "true")
    assert proc.returncode != 0, proc.stdout[-1500:]
    assert "must not happen" in proc.stdout


def test_locally_the_same_test_skips_with_its_reason(tmp_path):
    proc = run_smoke(tmp_path, "")
    assert proc.returncode == 0, proc.stdout[-1500:]
    assert "skipped" in proc.stdout
