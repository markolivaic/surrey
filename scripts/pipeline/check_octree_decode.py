"""Does the viewer's decoder give back the points and predictions that went in?

Runs web/tools/decode_check.mjs, which decodes real nodes of a built octree with the
browser worker's own code, and matches every decoded point to the cached point at the
same position. A match must agree on NAIP colour (rebuilt the same way), and on class and
confidence for every variant, against the prediction caches. No WebGL.

For a Surrey tile (--surrey) it also answers the licence question: the class field must equal
the model's prediction on every decoded point, and is compared with the DALES label only to
show it is not the label (agreement near the model's accuracy, not 1). NAIP does not cover
Surrey, so colour must be zero there.

Run:
    python scripts/pipeline/check_octree_decode.py --area embarcadero
    python scripts/pipeline/check_octree_decode.py --surrey --area 5135_54430
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from build_octree import naip_at, variant_dir  # noqa: E402
from surrey.cache import CachedTile  # noqa: E402
from surrey.registration import naip_offsets  # noqa: E402

ARTIFACT = ROOT / "data" / "artifacts" / "octree_decode_check.json"


def main() -> int:
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--area", required=True)
    ap.add_argument("--nodes", type=int, default=6)
    ap.add_argument("--surrey", action="store_true")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "decoded.bin"
        subprocess.run(
            [
                "node",
                str(ROOT / "web" / "tools" / "decode_check.mjs"),
                str(ROOT / "data" / "viewer" / args.area),
                str(out),
                str(args.nodes),
            ],
            check=True,
        )
        info = json.loads(Path(f"{out}.json").read_text())
        slots = info["slots"]
        dt = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)] + [(e, "u1") for e in slots])
        dec = np.frombuffer(out.read_bytes(), dtype=dt)

    city = "surrey" if args.surrey else "sf"
    tile = CachedTile(
        ROOT / "data" / "prep" / ("dales" if args.surrey else "target") / args.area, city
    )
    idx = np.arange(tile.n)
    ux, uy = tile.utm_xy(idx)
    z = tile.z_abs(idx)
    off = np.asarray(info["offset"])
    tree = cKDTree(np.column_stack([ux - off[0], uy - off[1], z - off[2]]))
    # Four neighbours: the voxel pick can leave two cached points within 1.8 cm of each
    # other, and then the nearest is not necessarily the one that was encoded. A decoded
    # point passes when one of its neighbours within 1.8 cm agrees on everything.
    dists, nns = tree.query(dec["xyz"].astype(np.float64), k=4)
    dist = dists[:, 0]
    near = dists <= 0.018
    crs = json.loads((ROOT / "data" / "artifacts" / "crs_check.json").read_text(encoding="utf-8"))
    rgb = (
        np.zeros((tile.n, 3), np.uint8)
        if args.surrey
        else naip_at(tile, naip_offsets(crs).get(args.area))
    )

    # The converter requantises to its own centimetre grid from the bounding box, so a
    # decoded point may sit one centimetre off on each axis: up to 1.8 cm in 3D.
    row = {
        "area": args.area,
        "nodes": info["nodes"],
        "points_decoded": int(dec.size),
        "position_match_within_2cm": round(float((dist <= 0.018).mean()), 6),
        "position_distance_max_m": round(float(dist.max()), 4),
        "points_with_a_second_cached_point_within_2cm": int(near[:, 1].sum()),
        "rgb_equal": round(
            float(
                np.any(
                    [near[:, k] & (dec["rgb"] == rgb[nns[:, k]]).all(axis=1) for k in range(4)],
                    axis=0,
                ).mean()
            ),
            6,
        ),
        "variants": {},
    }
    manifest = json.loads(
        (ROOT / "data" / "viewer" / args.area / "manifest.json").read_text(encoding="utf-8")
    )
    for i, v in enumerate(manifest["variants"]):
        pz = np.load(variant_dir(v, city) / f"{city}_{args.area}.npz")
        conf = np.clip(np.round(pz["conf"].astype(np.float32) * 255), 0, 255).astype(np.uint8)
        agree = [
            near[:, k]
            & (dec[f"c_{i}"] == pz["pred"][nns[:, k]])
            & (dec[f"p_{i}"] == conf[nns[:, k]])
            for k in range(4)
        ]
        both = np.any(agree, axis=0)
        row["variants"][v] = {
            "slot": i,
            "class_equal": round(float(both.mean()), 6),
            "confidence_equal": round(float(both.mean()), 6),
        }
    if args.surrey:
        # Only to show the class field is not the DALES label; the label is never written.
        label = np.asarray(tile.label[:], np.int16)
        first = np.any([near[:, k] & (dec["c_0"] == label[nns[:, k]]) for k in range(4)], axis=0)
        row["class_field_agrees_with_dales_label"] = round(float(first.mean()), 4)
        row["manifest_dales_labels_included"] = manifest.get("dales_labels_included")
    ok = (
        row["position_match_within_2cm"] == 1.0
        and row["rgb_equal"] == 1.0
        and all(
            r["class_equal"] == 1.0 and r["confidence_equal"] == 1.0
            for r in row["variants"].values()
        )
    )
    row["decoder_round_trip_exact"] = ok
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8")) if ARTIFACT.exists() else {"areas": {}}
    doc["areas"][f"{city}_{args.area}" if args.surrey else args.area] = row
    ARTIFACT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(row, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
