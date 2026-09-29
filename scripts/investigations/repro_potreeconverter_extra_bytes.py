"""Minimal reproduction: PotreeConverter 2.1.3 reads LAS extra bytes from the wrong offset.

A LAS 1.4 point format 7 file of 200,000 synthetic points, no project data involved, with N
uint8 extra byte attributes whose values are known: attribute k holds 10 * (k + 1) on every
point. The file is converted with --encoding UNCOMPRESSED, whose node layout is one record per
point in metadata order, and the extra attributes are read back from the first node. A correct
converter returns 10, 20, 30, ... in order. Run for N = 2 and N = 4. A second probe converts
the same kind of file at 20,000 points with two extra attributes and records the exit code: on
this machine that crashed while the same file without extra attributes converted.

What 2.1.3 did on this machine is recorded in docs/potreeconverter_extra_bytes.md. It has not
been reported upstream; this script publishes nothing.

Run:
    python scripts/investigations/repro_potreeconverter_extra_bytes.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = (
    ROOT
    / "data"
    / "tools"
    / "PotreeConverter"
    / "PotreeConverter_2.1.3_x64_windows"
    / "PotreeConverter.exe"
)
OUT = ROOT / "docs" / "potreeconverter_extra_bytes.json"
POINTS = 200_000
SMALL_POINTS = 20_000


def run(n_extra: int, work: Path, points: int = POINTS) -> dict:
    import laspy

    rng = np.random.default_rng(0)
    hdr = laspy.LasHeader(version="1.4", point_format=7)
    hdr.offsets = np.zeros(3)
    hdr.scales = np.full(3, 0.01)
    names = [f"x{k}" for k in range(n_extra)]
    for name in names:
        hdr.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint8))
    hdr.offsets = np.array([500000.0, 4000000.0, 10.0])
    las = laspy.LasData(hdr)
    las.x = 500000 + rng.uniform(0, 300, points)
    las.y = 4000000 + rng.uniform(0, 300, points)
    las.z = 10 + rng.uniform(0, 30, points)
    las.red = las.green = las.blue = np.full(points, 100 * 257, np.uint16)
    for k, name in enumerate(names):
        las[name] = np.full(points, 10 * (k + 1), np.uint8)
    src = work / f"extra_{n_extra}_{points}.las"
    las.write(src)
    back = laspy.read(src)
    written = [int(back[name][0]) for name in names]

    out_dir = work / f"out_{n_extra}_{points}"
    proc = subprocess.run(
        [
            str(CONVERTER),
            str(src),
            "-o",
            str(out_dir),
            "--encoding",
            "UNCOMPRESSED",
            "--attributes",
            "rgb",
            *names,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {"extra_attributes": n_extra, "points": points, "converter_exit": proc.returncode}
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    sizes = {a["name"]: a["size"] for a in meta["attributes"]}
    record = sum(sizes.values())
    hierarchy = (out_dir / "hierarchy.bin").read_bytes()
    offset = int.from_bytes(hierarchy[6:14], "little")
    octree = (out_dir / "octree.bin").read_bytes()
    first = octree[offset : offset + record]
    at, read = 0, {}
    for a in meta["attributes"]:
        if a["name"] in names:
            read[a["name"]] = first[at]
        at += a["size"]
    return {
        "extra_attributes": n_extra,
        "points": points,
        "converter_exit": 0,
        "written_by_laspy": written,
        "metadata_min_max": {
            a["name"]: [a["min"][0], a["max"][0]] for a in meta["attributes"] if a["name"] in names
        },
        "read_back_from_octree": [read[n] for n in names],
        "correct": [read[n] for n in names] == written,
    }


def main() -> int:
    if not CONVERTER.exists():
        raise SystemExit(f"PotreeConverter 2.1.3 not found at {CONVERTER}")
    work = Path(tempfile.mkdtemp(prefix="potree_repro_"))
    try:
        rows = [run(n, work) for n in (2, 4)]
        small = run(2, work, SMALL_POINTS)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    OUT.write_text(
        json.dumps(
            {
                "converter": "PotreeConverter 2.1.3 Windows x64",
                "runs": rows,
                "small_file_probe": small,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for r in [*rows, small]:
        print(r)
    return 0 if all(r.get("correct") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
