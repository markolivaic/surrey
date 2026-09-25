"""Run every check the README's verification table lists, and write what each one said.

The table in the README is filled from this file's output, so a count or a result there is what a run
printed on a named machine on a named date, never a number typed in. Behavioural and integrity tests
are counted apart, because a guard that reads a file is not a test of what the code does.

Checks, in the order a reader would run them:

  behavioural tests     python -m pytest tests/unit -q
  integrity tests       python -m pytest tests/integrity -q
  README is generated   python scripts/pipeline/render_readme.py --check
  page figures          python scripts/pipeline/page_figures.py
  viewer tests          npm test          (in web/)
  viewer build          npm run build     (in web/)
  page smoke test       python -m pytest tests/smoke -q   (needs Playwright's Chromium)

Run:
    python scripts/pipeline/verify.py
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "artifacts" / "verification.json"
PY = sys.executable
NPM = shutil.which("npm") or "npm"

CHECKS = [
    (
        "behavioural tests",
        [PY, "-m", "pytest", "tests/unit", "-q", "-p", "no:cacheprovider"],
        ROOT,
        "python -m pytest tests/unit -q",
    ),
    (
        "integrity tests",
        [PY, "-m", "pytest", "tests/integrity", "-q", "-p", "no:cacheprovider"],
        ROOT,
        "python -m pytest tests/integrity -q",
    ),
    (
        "README is generated",
        [PY, "scripts/pipeline/render_readme.py", "--check"],
        ROOT,
        "python scripts/pipeline/render_readme.py --check",
    ),
    (
        "page figures",
        [PY, "scripts/pipeline/page_figures.py"],
        ROOT,
        "python scripts/pipeline/page_figures.py",
    ),
    ("viewer tests", [NPM, "test"], ROOT / "web", "npm test (in web/)"),
    ("viewer build", [NPM, "run", "build"], ROOT / "web", "npm run build (in web/)"),
    (
        "page smoke test",
        [PY, "-m", "pytest", "tests/smoke", "-q", "-p", "no:cacheprovider"],
        ROOT,
        "python -m pytest tests/smoke -q",
    ),
]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_tests import functions  # noqa: E402

PYTEST_SUMMARY = re.compile(r"(\d+) (passed|failed|skipped|error|errors|deselected)")


def machine() -> str:
    gpu = ""
    try:
        gpu = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except (OSError, IndexError, subprocess.SubprocessError):
        gpu = "no NVIDIA card"
    return f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}, {gpu}, Python {platform.python_version()}"


def result_of(name: str, code: int, text: str, suite: str = "") -> dict:
    counts = {}
    if name == "viewer tests":
        # node --test reports "# pass N" and "# fail N", or with an information sign in newer versions.
        m = re.search(r"(?:#|\u2139) pass (\d+)", text)
        f = re.search(r"(?:#|\u2139) fail (\d+)", text)
        counts = {"passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0}
        summary = f"{counts['passed']} passed, {counts['failed']} failed"
    elif "pytest" in name or name.endswith("tests") or name.endswith("smoke test"):
        for n, word in PYTEST_SUMMARY.findall(text):
            counts["errors" if word.startswith("error") else word] = int(n)
        cases = ", ".join(f"{v} {k}" for k, v in counts.items()) or "no tests ran"
        n = functions(suite)
        counts["functions"] = n
        summary = f"{n} test{'' if n == 1 else 's'}; cases {cases}"
    else:
        summary = "passes" if code == 0 else "fails"
    return {"exit_code": code, "passes": code == 0, "counts": counts, "summary": summary}


def main() -> int:
    started = dt.datetime.now(dt.UTC)
    rows = []
    for name, cmd, cwd, shown in CHECKS:
        run = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        text = run.stdout + run.stderr
        suite = next((a for a in cmd if a.startswith("tests/")), "")
        row = {"check": name, "command": shown} | result_of(name, run.returncode, text, suite)
        rows.append(row)
        print(f"{name}: {row['summary']} (exit {run.returncode})", flush=True)
        if run.returncode != 0:
            print(text[-3000:], flush=True)
    doc = {
        "what": "every check in the README's verification table, as one run printed it",
        "date": started.date().isoformat(),
        "machine": machine(),
        "checks": rows,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0 if all(r["passes"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
