"""Before an upload: every file in each area's octree folder matches its manifest's SHA-256, and
the folder holds nothing the manifest does not name.

Run:
    python scripts/pipeline/check_viewer_manifests.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "data" / "viewer"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    problems = []
    for manifest in sorted(VIEWER.glob("*/manifest.json")):
        m = json.loads(manifest.read_text(encoding="utf-8"))
        folder = manifest.parent
        files = {p.name for p in folder.iterdir() if p.is_file()} - {"manifest.json"}
        named = set(m["files"])
        if files != named:
            problems.append(
                f"{folder.name}: files {sorted(files ^ named)} not matched by the manifest"
            )
        for name, rec in m["files"].items():
            path = folder / name
            if not path.exists() or sha256(path) != rec["sha256"]:
                problems.append(f"{folder.name}/{name}: hash does not match")
        if m.get("dales_labels_included") is not False:
            problems.append(
                f"{folder.name}: manifest does not state that no DALES labels are included"
            )
        print(f"{folder.name}: {len(named)} files checked")
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
