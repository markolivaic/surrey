"""In CI a skipped page test is a failed page test.

Locally the smoke test skips when Playwright or the built page is missing, and says why. In CI
(CI=true, which GitHub Actions sets) a skip would mean the workflow claims to test the page without
doing it, which the publish standard does not allow, so every skip here becomes a failure carrying
the skip's reason.
"""

from __future__ import annotations

import os

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.environ.get("CI", "").lower() == "true" and report.skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = f"a page test was skipped in CI, which must not happen: {reason}"
