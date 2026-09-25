"""Download the DALES-2 tiles from HuggingFace and record what arrived.

Resumable: re-running skips files already present with a matching size, and
huggingface_hub resumes partial transfers on its own.

Licence chain for what this downloads, see THIRD_PARTY_NOTICES.md:
  raw points      Open Government Licence, City of Surrey (commercial use allowed)
  DALES labels    "Creative Commons 3.0", non-commercial, per the DALES project page
  DALES-2 labels  MIT per the HuggingFace card, which does not mention the two above

Nothing downloaded here is committed to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ID = "mbendjilali/DALES-2"
REPO_TYPE = "dataset"
DEFAULT_DEST = Path(__file__).resolve().parents[2] / "data" / "dales2"
MANIFEST = Path(__file__).resolve().parents[2] / "data" / "dales2_manifest.json"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="which subdirectories of the dataset repo to pull",
    )
    ap.add_argument(
        "--hash",
        action="store_true",
        help="compute sha256 for every tile (slow, ~2.3 GB of reads)",
    )
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    args.dest.mkdir(parents=True, exist_ok=True)
    patterns = [f"{s}/*.laz" for s in args.splits] + ["README.md"]

    t0 = time.time()
    print(f"repo   {REPO_ID}")
    print(f"dest   {args.dest}")
    print(f"take   {patterns}")
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(args.dest),
        allow_patterns=patterns,
        max_workers=4,
    )
    elapsed = time.time() - t0

    tiles = sorted(
        (p for s in args.splits for p in (Path(local) / s).glob("*.laz")),
        key=lambda p: (p.parent.name, p.name),
    )
    if not tiles:
        print("no tiles landed, refusing to write a manifest", file=sys.stderr)
        return 1

    records = []
    total = 0
    for p in tiles:
        size = p.stat().st_size
        total += size
        rec = {
            "split": p.parent.name,
            "name": p.name,
            "bytes": size,
        }
        if args.hash:
            rec["sha256"] = sha256_of(p)
        records.append(rec)

    counts: dict[str, int] = {}
    for r in records:
        counts[r["split"]] = counts.get(r["split"], 0) + 1

    manifest = {
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "downloaded_seconds": round(elapsed, 1),
        "tile_count": len(records),
        "tiles_per_split": counts,
        "total_bytes": total,
        "hashed": bool(args.hash),
        "tiles": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\ntiles  {len(records)}  {counts}")
    print(f"bytes  {total:,} ({total / 1e9:.2f} GB)")
    print(f"time   {elapsed:.0f} s")
    print(f"wrote  {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
