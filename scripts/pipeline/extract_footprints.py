"""Cut the San Francisco building footprints out of Microsoft's California file.

Microsoft Building Footprints, ODbL-1.0. The state file is 466 MB zipped, so it
is cached and never committed; the small city subset this writes is the artifact.

The state file is streamed rather than parsed whole. Microsoft writes these as a
FeatureCollection with one feature per line, so a line pass works, and a whole
document parse is kept as a fallback for the case where that stops being true.

Run:
    python scripts/pipeline/extract_footprints.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_URL = (
    "https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/California.geojson.zip"
)
DEFAULT_ZIP = ROOT / "data" / "cache" / "footprints" / "California.geojson.zip"
DEFAULT_OUT = ROOT / "data" / "target" / "sf_footprints.geojson"
DEFAULT_ARTIFACT = ROOT / "data" / "artifacts" / "sf_footprints.json"

# The whole of San Francisco, comfortably around every area of interest.
SF_BBOX = (-122.53, 37.70, -122.35, 37.84)


def ring_bbox(geom: dict) -> tuple[float, float, float, float] | None:
    """Bounding box of a Polygon or MultiPolygon, without building geometry."""
    coords = geom.get("coordinates")
    if not coords:
        return None
    xs: list[float] = []
    ys: list[float] = []

    def walk(node) -> None:
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], (int, float)) and len(node) >= 2:
                xs.append(float(node[0]))
                ys.append(float(node[1]))
            else:
                for child in node:
                    walk(child)

    walk(coords)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def iter_features(zf: zipfile.ZipFile, member: str):
    """Yield GeoJSON features, one line at a time where possible."""
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        line_mode = False
        buffered: list[str] = []
        for line in text:
            s = line.strip().rstrip(",")
            if not s or s in ("{", "}", "[", "]"):
                buffered.append(line)
                continue
            if s.startswith('{"type":"Feature"') or s.startswith('{ "type": "Feature"'):
                line_mode = True
                try:
                    yield json.loads(s)
                    continue
                except json.JSONDecodeError:
                    line_mode = False
            buffered.append(line)
        if not line_mode:
            # Fall back to a whole document parse.
            with zf.open(member) as raw2:
                doc = json.load(io.TextIOWrapper(raw2, encoding="utf-8"))
            yield from doc.get("features", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=list(SF_BBOX),
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
    )
    args = ap.parse_args()

    if not args.zip_path.exists():
        raise SystemExit(f"{args.zip_path} is missing. Download it from:\n  {SRC_URL}")

    bbox = tuple(args.bbox)
    t0 = time.time()
    with zipfile.ZipFile(args.zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".geojson")]
        if len(members) != 1:
            raise SystemExit(f"expected one .geojson in the zip, found {members}")
        member = members[0]
        member_size = zf.getinfo(member).file_size

        kept: list[dict] = []
        seen = 0
        for feat in iter_features(zf, member):
            seen += 1
            geom = feat.get("geometry") or {}
            fb = ring_bbox(geom)
            if fb is not None and intersects(fb, bbox):
                kept.append({"type": "Feature", "properties": {}, "geometry": geom})
            if seen % 1_000_000 == 0:
                print(f"  scanned {seen:,} features, kept {len(kept):,}", flush=True)

    out = {
        "type": "FeatureCollection",
        "name": "sf_building_footprints",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "source": SRC_URL,
        "license": "ODbL-1.0",
        "bbox_filter": list(bbox),
        "features": kept,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out), encoding="utf-8", newline="\n")

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    rec = {
        "source": SRC_URL,
        "license": "ODbL-1.0",
        "state_member": member,
        "state_member_uncompressed_bytes": member_size,
        "features_scanned": seen,
        "features_kept": len(kept),
        "bbox_filter": list(bbox),
        "output": str(args.out.relative_to(ROOT)).replace("\\", "/"),
        "output_bytes": args.out.stat().st_size,
        "output_sha256": digest,
        "seconds": round(time.time() - t0, 1),
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"scanned {seen:,} features")
    print(f"kept    {len(kept):,} inside {bbox}")
    print(f"wrote   {args.out}  ({rec['output_bytes'] / 1e6:.1f} MB)")
    print(f"sha256  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
