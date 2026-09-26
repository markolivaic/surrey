"""Why the building lift fell: precision and coverage against footprints, old M0 against new M0.

The lift over the footprint null is a precision, so on its own it cannot tell a model that finds
fewer buildings from one that finds more of them less precisely. For every area with footprints this
sets, beside each other, the old M0 (its scores as the first commit, d62078f, holds them) and the new
M0 (Phase 4), median and range over three seeds: precision against footprints and its null, footprint
coverage and its null, and the footprints scored. Diagnostic runs of new M0 seed A, one vote and the
old iterator at one vote, are added when their scores exist, to separate what inference alone changed.

The reading was written before this ran, in docs/phase4_criteria.md: a figure rose or fell only when
every new seed lies beyond every old one (backend/surrey/seeds.py); coverage up with precision down is
a trade, both down is worse, anything else is written as measured.

Writes data/artifacts/building_breakdown.json.

Run, the two diagnoses first (both on the laptop's card, about a quarter of an hour each):
    python scripts/pipeline/predict_cache.py --ckpt data/checkpoints/m0_seedA.pt --out-dir data/predictions/m0_seedA_1vote --votes 1
    python scripts/investigations/predict_legacy_iterator.py
    python scripts/pipeline/score_variant.py --variant m0_seedA_1vote
    python scripts/pipeline/score_variant.py --variant m0_seedA_legacy_iter
    python scripts/investigations/building_breakdown.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "data" / "artifacts"
OUT = ARTIFACTS / "building_breakdown.json"
OLD_COMMIT = "d62078f"
SEEDS = "ABC"
DIAGNOSTICS = {
    "m0_seedA_1vote": "new M0 seed A, one vote, new iterator",
    "m0_seedA_legacy_iter": "new M0 seed A, one vote, the old iterator",
}
FIGURES = {
    "precision": ("building_vs_footprints_over_2m", "global_shift", "precision"),
    "null_precision": ("building_vs_footprints_over_2m", "global_shift", "null_precision"),
    "lift": ("building_vs_footprints_over_2m", "global_shift", "precision_lift_over_null"),
    "coverage_median": ("building_area_coverage_global_shift", "coverage_median"),
    "null_coverage_median": ("building_area_coverage_global_shift", "null_coverage_median"),
    "footprints_scored": ("building_area_coverage_global_shift", "footprints_scored"),
}


def from_commit(name: str) -> dict:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{OLD_COMMIT}:data/artifacts/{name}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if out.returncode != 0:
        raise SystemExit(f"{name} is not in commit {OLD_COMMIT}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def by_area(doc: dict) -> dict:
    return {r["aoi"]: r for r in doc["san_francisco_per_area"]}


def pick(row: dict, path: tuple) -> float | None:
    for k in path:
        row = row.get(k) if isinstance(row, dict) else None
    return row


def direction(new: list, old: list) -> str:
    if None in new or None in old:
        return "not measured"
    if min(new) > max(old):
        return "rose"
    if max(new) < min(old):
        return "fell"
    return "did not move"


def summary(vals: list) -> dict | None:
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"median": round(float(np.median(v)), 4), "min": min(v), "max": max(v), "per_seed": vals}


def main() -> int:
    old = {s: by_area(from_commit(f"sf_scores_m0_seed{s}.json")) for s in SEEDS}
    new = {
        s: by_area(
            json.loads((ARTIFACTS / f"sf_scores_m0_seed{s}.json").read_text(encoding="utf-8"))
        )
        for s in SEEDS
    }
    diag = {}
    for name in DIAGNOSTICS:
        p = ARTIFACTS / f"sf_scores_{name}.json"
        if p.exists():
            diag[name] = by_area(json.loads(p.read_text(encoding="utf-8")))
    areas = sorted(a for a in new["A"] if pick(new["A"][a], FIGURES["footprints_scored"]))
    rows = {}
    for a in areas:
        row = {}
        for fig, path in FIGURES.items():
            o = [pick(old[s][a], path) for s in SEEDS]
            n = [pick(new[s][a], path) for s in SEEDS]
            row[fig] = {"old": summary(o), "new": summary(n), "direction": direction(n, o)}
            row[fig]["seed_a"] = {
                "four_votes": n[0],
                **{k: pick(d[a], path) for k, d in diag.items()},
            }
        prec, cov = row["precision"]["direction"], row["coverage_median"]["direction"]
        if cov == "rose" and prec == "fell":
            reading = "a trade: more of each footprint covered, less precisely"
        elif cov == "fell" and prec == "fell":
            reading = "worse at buildings here"
        else:
            reading = f"precision {prec}, coverage {cov}"
        row["reading"] = reading
        rows[a] = row
        print(f"{a:12s} precision {prec:12s} coverage {cov:12s} -> {reading}", flush=True)
    doc = {
        "what": "building precision and coverage against footprints, old M0 against new M0",
        "old": f"sf_scores_m0_seed[ABC].json as commit {OLD_COMMIT} holds them",
        "new": "sf_scores_m0_seed[ABC].json, Phase 4",
        "rule": "rose or fell only when every new seed lies beyond every old one (backend/surrey/seeds.py); "
        "reading written in docs/phase4_criteria.md before this ran",
        "diagnostics": {k: v for k, v in DIAGNOSTICS.items() if k in diag},
        "not_separated": "class weights from 29 tiles against 40, and the dropped withheld points: "
        "separating them needs training",
        "areas": rows,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
