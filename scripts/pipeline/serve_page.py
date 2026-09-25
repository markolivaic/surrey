"""Serve the built page locally, the way the host serves it.

The page is static. It needs a server that answers byte ranges, because the octree loader asks for each
node by range, and it needs its data under /data (the real octrees, built by scripts/pipeline/build_octree.py)
or /fixtures (the synthetic FIXTURE octree committed with the repository). This serves web/dist with
both mounted and prints the two addresses: the real page, if data/viewer exists, and the fixture page,
which works from a fresh clone.

Run, after `npm ci` and `npm run build` in web/:
    python scripts/pipeline/serve_page.py
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.static_server import serve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    dist = ROOT / "web" / "dist"
    if not (dist / "index.html").exists():
        raise SystemExit("web/dist is not built: run npm ci and npm run build in web/ first")
    mounts = {"/fixtures": str(ROOT / "web" / "fixtures")}
    real = ROOT / "data" / "viewer"
    if real.exists():
        mounts["/data"] = str(real)
    srv = serve(str(dist), args.port, mounts=mounts)
    base = f"http://127.0.0.1:{args.port}/index.html"
    print(f"fixture page, works from a fresh clone: {base}?fixture=1")
    print(
        f"real page: {base}"
        if real.exists()
        else "real page: needs the octrees in data/viewer, built by the pipeline in the README"
    )
    print("Ctrl+C stops the server")
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
