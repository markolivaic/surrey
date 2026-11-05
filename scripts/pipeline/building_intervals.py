"""How uncertain the building figures are, and whether lift falls with block height everywhere.

Two measurements on the published recipe's three seeds, scored exactly as score_variant.py scores
buildings (global footprint placement, points more than 2 m above the label-free floor, water and
bridge deck left out):

    intervals   a bootstrap over footprints. Every eligible point goes with its nearest footprint, so
                each footprint carries its own share of hits, misses and null points, and 1,000
                resamples of the footprints give a 95% interval for precision and lift per area. With
                6 footprints at Embarcadero the interval is wide, and that is the point of showing it.
    by height   precision, null precision and lift over 40 m blocks pooled across the built areas,
                binned by how much height the block spans. If lift falls with height span in every
                area, that is consistent with the network meeting heights it was not trained on; if
                it falls only in the Financial District, something about that district is the more
                likely cause. Neither reading proves a cause.

Writes data/artifacts/building_intervals.json.

Run:
    python scripts/pipeline/building_intervals.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

import score_variant as sv  # noqa: E402
from surrey.areas import WITH_FOOTPRINTS  # noqa: E402
from surrey.cache import CachedTile  # noqa: E402
from surrey.footprints import clip_to_box, inside_mask, load_footprints  # noqa: E402
from surrey.metrics import coarse_lookup  # noqa: E402
from surrey.registration import footprint_offsets  # noqa: E402

ARTIFACTS = ROOT / "data" / "artifacts"
OUT = ARTIFACTS / "building_intervals.json"
SEEDS = ("A", "B", "C")
BUILT = WITH_FOOTPRINTS
RESAMPLES = 1000
RNG_SEED = 0


def lift(hits: float, pred: float, inside: float, eligible: float) -> tuple[float, float, float]:
    p = hits / pred if pred else float("nan")
    p0 = inside / eligible if eligible else float("nan")
    return p, p0, (p - p0) / (1 - p0) if p0 == p0 and 0 < p0 < 1 and p == p else float("nan")


def rounded(v: float) -> float | None:
    return None if v != v else round(float(v), 4)


def main() -> int:
    import shapely.affinity
    from scipy.spatial import cKDTree

    t0 = time.time()
    crs = json.loads((ARTIFACTS / "crs_check.json").read_text(encoding="utf-8"))
    off = footprint_offsets(crs)["global"]
    rng_doc = json.loads((ARTIFACTS / "input_range_check.json").read_text(encoding="utf-8"))
    block_m = float(rng_doc["block_m"])
    p99 = float(rng_doc["confounds_at_the_99th_percentile"]["training_p99"]["block_z_range_m"])
    top = float(rng_doc["vertical_extrapolation"]["surrey_training_block_z_max_m"])
    edges = [0.0, 15.0, 30.0, p99, top, float("inf")]
    table = coarse_lookup()
    building_group = sv.GROUPS.index("building")
    rng = np.random.default_rng(RNG_SEED)

    polys = None
    intervals: dict = {}
    bins = {s: np.zeros((len(edges) - 1, 4)) for s in SEEDS}  # hits, predicted, inside, eligible
    per_area_bins: dict = {}
    for name in BUILT:
        tile = CachedTile(sv.TARGET_PREP / name, "sf")
        if polys is None:
            polys = load_footprints(sv.FOOTPRINTS, tile.epsg)
        asprs = np.asarray(tile.asprs[:], np.int16)
        clean = ~np.isin(asprs, sv.ASPRS_NOISE)
        hag = np.asarray(tile.hag_cm[:], np.float32) / 100.0
        z = np.asarray(tile.z_cm[:], np.float64) / 100.0
        ux, uy = tile.utm_xy(np.arange(tile.n))
        clipped, _ = clip_to_box(polys, ux, uy)
        moved = [
            shapely.affinity.translate(q, xoff=off["east_m"], yoff=off["north_m"]) for q in clipped
        ]
        inside = inside_mask(moved, ux, uy)
        eligible = (
            clean & (hag > sv.HAG_GATE_M) & ~np.isin(asprs, (sv.ASPRS_WATER, sv.ASPRS_BRIDGE))
        )
        idx = np.flatnonzero(eligible)
        centroids = np.array([[q.centroid.x, q.centroid.y] for q in moved])
        owner = cKDTree(centroids).query(np.stack([ux[idx], uy[idx]], axis=1))[1]
        n_fp = len(moved)

        block_of = np.full(tile.n, -1, np.int64)
        spans = []
        for b, (bidx, zrange, _rel) in enumerate(sv.blocks(tile, block_m, asprs, clean, z)):
            block_of[bidx] = b
            spans.append(zrange)
        spans = np.asarray(spans, np.float64)
        bin_of_block = np.digitize(spans, edges[1:-1])

        intervals[name] = {"footprints": n_fp}
        per_area_bins[name] = {}
        for s in SEEDS:
            pred, _, _ = sv.load(sv.PRED / f"m0_seed{s}", f"sf_{name}")
            called = table[pred] == building_group
            c, i = called[idx], inside[idx]
            counts = np.stack(
                [
                    np.bincount(owner, weights=c & i, minlength=n_fp),
                    np.bincount(owner, weights=c, minlength=n_fp),
                    np.bincount(owner, weights=i, minlength=n_fp),
                    np.bincount(owner, minlength=n_fp).astype(float),
                ],
                axis=1,
            )
            p, p0, lf = lift(*counts.sum(axis=0))
            draws = rng.integers(0, n_fp, size=(RESAMPLES, n_fp))
            sums = counts[draws].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                bp = sums[:, 0] / sums[:, 1]
                bp0 = sums[:, 2] / sums[:, 3]
                bl = (bp - bp0) / (1 - bp0)
            intervals[name][s] = {
                "precision": rounded(p),
                "null_precision": rounded(p0),
                "lift": rounded(lf),
                "precision_95": [
                    rounded(np.nanpercentile(bp, 2.5)),
                    rounded(np.nanpercentile(bp, 97.5)),
                ],
                "lift_95": [
                    rounded(np.nanpercentile(bl, 2.5)),
                    rounded(np.nanpercentile(bl, 97.5)),
                ],
            }

            blk = block_of[idx]
            ok = blk >= 0
            bin_ids = bin_of_block[blk[ok]]
            area_bins = np.zeros((len(edges) - 1, 4))
            for col, w in enumerate((c & i, c, i, np.ones_like(c))):
                area_bins[:, col] = np.bincount(
                    bin_ids, weights=w[ok].astype(float), minlength=len(edges) - 1
                )
            bins[s] += area_bins
            per_area_bins[name][s] = area_bins.tolist()
        a = intervals[name]["A"]
        print(
            f"  {name:12s} {n_fp:3d} footprints  seed A lift {a['lift']} {a['lift_95']}  "
            f"precision {a['precision']} {a['precision_95']}",
            flush=True,
        )

    def summarise(counts: np.ndarray) -> list[dict]:
        out = []
        for k in range(len(edges) - 1):
            p, p0, lf = lift(*counts[k])
            out.append(
                {
                    "span_m": [edges[k], None if edges[k + 1] == float("inf") else edges[k + 1]],
                    "eligible_points": int(counts[k][3]),
                    "precision": rounded(p),
                    "null_precision": rounded(p0),
                    "lift": rounded(lf),
                }
            )
        return out

    by_height = {s: summarise(bins[s]) for s in SEEDS}
    without_fd = {
        s: summarise(sum(np.asarray(per_area_bins[a][s]) for a in BUILT if a != "financial"))
        for s in SEEDS
    }
    doc = {
        "what": "a bootstrap over footprints for building precision and lift, and lift by block "
        "height span pooled over the built areas, for the published recipe's three seeds",
        "placement": "global footprint offset, as score_variant.py",
        "eligible": "clean points more than 2 m above the label-free floor, not water or bridge deck",
        "bootstrap": {
            "unit": "footprint; every eligible point goes with its nearest footprint centroid",
            "resamples": RESAMPLES,
            "seed": RNG_SEED,
            "interval": "2.5th to 97.5th percentile",
        },
        "height_bins_m": [e if e != float("inf") else None for e in edges],
        "height_bin_edges_from": "15 and 30 m, then the 99th percentile and the maximum of the "
        "training blocks' height span (input_range_check.json)",
        "areas": intervals,
        "by_height_span": by_height,
        "by_height_span_without_financial": without_fd,
        "by_height_span_per_area": {
            a: {s: summarise(np.asarray(per_area_bins[a][s])) for s in SEEDS} for a in BUILT
        },
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    for s in SEEDS[:1]:
        for row in by_height[s]:
            print(
                f"  seed {s} span {row['span_m']}: lift {row['lift']} over {row['eligible_points']:,}"
            )
        for row in without_fd[s]:
            print(
                f"  seed {s} without FD {row['span_m']}: lift {row['lift']} over {row['eligible_points']:,}"
            )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
