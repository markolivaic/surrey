"""Count and run both test suites, and write the result as an artefact.

The README reports behavioural and integrity tests as two numbers. Those numbers
were being typed into prose from a console, which is exactly what the number guard
forbids for every other figure. A guard that demands a source for each decimal
should not exempt the figures about itself.

Every skipped test is counted by its reason, because a bare number of skips reads, from the
outside, like failures kept out of sight.

Collected counts say what exists. The run records what passed, failed and skipped,
on which machine and when, because a count of tests is only a claim once they ran.

Run:
    python scripts/pipeline/count_tests.py
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "artifacts" / "tests.json"
SUITES = {"unit": "tests/unit", "integrity": "tests/integrity", "smoke": "tests/smoke"}
SKIP_LINE = re.compile(r"^SKIPPED \[(\d+)\] [^:]+(?::\d+)?: (.+)$")
SUMMARY = re.compile(r"(\d+) (passed|failed|skipped|error|errors|deselected|xfailed|xpassed)")


def pytest(*args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args], cwd=ROOT, capture_output=True, text=True
    )
    return proc.stdout + proc.stderr


def collected(path: str) -> int:
    out = pytest(path, "--collect-only", "-q")
    m = re.search(r"(\d+) tests? collected", out) or re.search(r"collected (\d+) items?", out)
    if m:
        return int(m.group(1))
    return sum(1 for line in out.splitlines() if "::" in line)


def functions(path: str) -> int:
    """Test functions, each counted once however many cases a parametrisation makes of it. This is the
    number the README quotes; the cases are recorded beside it."""
    out = pytest(path, "--collect-only", "-q", "-p", "no:cacheprovider")
    ids = {line.split("[", 1)[0] for line in out.splitlines() if "::" in line}
    return len(ids)


def run(path: str) -> dict:
    out = pytest(path, "-q", "-rs", "-p", "no:cacheprovider")
    last = [line for line in out.splitlines() if SUMMARY.search(line)]
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    if last:
        for n, word in SUMMARY.findall(last[-1]):
            key = "errors" if word.startswith("error") else word
            if key in counts:
                counts[key] = int(n)
    reasons: dict[str, int] = {}
    for line in out.splitlines():
        m = SKIP_LINE.match(line.strip())
        if m:
            reasons[m.group(2)] = reasons.get(m.group(2), 0) + int(m.group(1))
    counts["skipped_by_reason"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    return counts


def main() -> int:
    t0 = time.time()
    suites = {}
    for name, path in SUITES.items():
        suites[name] = {
            "path": path,
            "functions": functions(path),
            "collected": collected(path),
            **run(path),
        }
        s = suites[name]
        print(
            f"{name:10s} functions {s['functions']:>4}  cases {s['collected']:>4}  passed {s['passed']:>4}  "
            f"failed {s['failed']:>3}  skipped {s['skipped']:>3}",
            flush=True,
        )
    result = {
        "suites": suites,
        "machine": platform.platform(),
        "python": platform.python_version(),
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(time.time() - t0, 1),
        "note": (
            "counted separately on purpose; a single total would let greps "
            "inflate a count of behavioural tests"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
