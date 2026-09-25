"""Build the FIXTURE octree the CI smoke test loads. Synthetic points, no project data, no model.

A 60 m square of made-up ground with a few boxes and cones on it, classes assigned from the
geometry (ground, building, vegetation), confidences drawn at random, colour a flat grey. It
exists so a Linux runner, which cannot run the Windows PotreeConverter binary, can build the page
and check that it loads and draws. Everything about it says FIXTURE: the folder name, the
manifest, and the banner the page shows in fixture mode.

Committed under web/fixtures/FIXTURE with a manifest of SHA-256 hashes.

Run:
    python scripts/pipeline/build_fixture_octree.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from build_octree import CONVERTER, SLOT_FIELDS, VIEWER_VARIANTS  # noqa: E402

OUT = ROOT / "web" / "fixtures" / "FIXTURE"
SEED = 7
VARIANTS = VIEWER_VARIANTS[:3]
GROUND, VEGETATION, BUILDING = 0, 1, 12


def points(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_ground = 40_000
    xy = rng.uniform(0, 60, (n_ground, 2))
    z = 0.02 * xy[:, 0] + rng.normal(0, 0.03, n_ground)
    pts, cls = [np.column_stack([xy, z])], [np.full(n_ground, GROUND)]
    for cx, cy, w, h in ((15, 15, 10, 8), (40, 20, 12, 12)):
        m = 6000
        top = np.column_stack(
            [rng.uniform(cx, cx + w, m), rng.uniform(cy, cy + w, m), np.full(m, h)]
        )
        pts.append(top)
        cls.append(np.full(m, BUILDING))
    for cx, cy in ((10, 45), (30, 45), (50, 45), (48, 5)):
        m = 3000
        r = rng.uniform(0, 3, m)
        a = rng.uniform(0, 2 * np.pi, m)
        hgt = rng.uniform(2, 9, m)
        pts.append(
            np.column_stack(
                [cx + r * np.cos(a) * (1 - hgt / 10), cy + r * np.sin(a) * (1 - hgt / 10), hgt]
            )
        )
        cls.append(np.full(m, VEGETATION))
    return np.concatenate(pts), np.concatenate(cls).astype(np.uint8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    import laspy

    rng = np.random.default_rng(SEED)
    xyz, cls = points(rng)
    n = len(xyz)
    work = Path(tempfile.mkdtemp(prefix="fixture_octree_"))
    try:
        hdr = laspy.LasHeader(version="1.4", point_format=7)
        hdr.offsets = np.array([500000.0, 4000000.0, 0.0])
        hdr.scales = np.full(3, 0.01)
        las = laspy.LasData(hdr)
        las.x, las.y, las.z = 500000 + xyz[:, 0], 4000000 + xyz[:, 1], xyz[:, 2]
        las.red = las.green = las.blue = np.full(n, 140 * 257, np.uint16)
        conf = [rng.integers(90, 256, n).astype(np.uint8) for _ in VARIANTS]
        las.classification, las.user_data = cls, conf[0]
        las.intensity = cls.astype(np.uint16) | (conf[1].astype(np.uint16) << 8)
        las.point_source_id = cls.astype(np.uint16) | (conf[2].astype(np.uint16) << 8)
        src = work / "fixture.las"
        las.write(src)
        attrs = ["rgb"] + [f for fields in SLOT_FIELDS[: len(VARIANTS)] for f in fields]
        if OUT.exists():
            shutil.rmtree(OUT)
        proc = subprocess.run(
            [
                str(CONVERTER),
                str(src),
                "-o",
                str(OUT),
                "--encoding",
                "BROTLI",
                "--attributes",
                *attrs,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"converter failed: {proc.returncode}\n{proc.stdout[-1500:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    (OUT / "log.txt").unlink(missing_ok=True)
    ids, counts = np.unique(cls, return_counts=True)
    manifest = {
        "area": "FIXTURE",
        "fixture": True,
        "note": "FIXTURE: synthetic points and random confidences, not data",
        "points": n,
        "variants": list(VARIANTS),
        "encoding": "BROTLI",
        "city": "fixture",
        "dales_labels_included": False,
        "slots": [{"variant": v, "fields": list(SLOT_FIELDS[i])} for i, v in enumerate(VARIANTS)],
        "classes_present": {
            v: {int(i): int(c) for i, c in zip(ids, counts, strict=True)} for v in VARIANTS
        },
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(OUT.iterdir())
            if p.is_file()
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"FIXTURE octree: {n} points, {total} bytes in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
