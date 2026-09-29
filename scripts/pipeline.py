"""Every step from the public sources to the README and the page, in order, with what each reads and writes.

The steps are the scripts in scripts/pipeline/. Each writes what a later one reads; nothing is read from
a place no earlier step writes, except the public sources named in a step's inputs. Scripts in
scripts/investigations/ answered one question each and are not part of this chain.

    python scripts/pipeline.py                 # the steps, their inputs and outputs
    python scripts/pipeline.py --check         # which outputs are on disk
    python scripts/pipeline.py --run prep_sf predict
    python scripts/pipeline.py --run all       # everything, in order

Downloading, training and prediction take hours and want an NVIDIA card; the last steps do not. On a
rented card the training and prediction run through scripts/pipeline/phase6_chain.sh
(docs/phase6_criteria.md).

Only M0, small and large, is trained and predicted here. Every other model's predictions were made
before the neighbour search ran in fp32 and are kept as they were (the step kept_predictions): Phase 4's
M0 as m0phase4, the first round's M0prime and M1 pairs, and the four channel checkpoint. This code would
predict them differently, and the first round's models were also trained with class weights from all 40
tiles and on Swiss water that the flag filter now removes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
P = "scripts/pipeline"

RECIPES = ["m0", "m0phase4", "m0prime", "m1"]
AREAS = ["financial", "soma", "mission", "sunset", "ggpark", "embarcadero", "baybridge", "steep"]
SEED = {"A": 0, "B": 1, "C": 2}
PREDICTED = ["m0_seedA", "m0_seedB", "m0_seedC", "m0_big_seedA"]
# Predicted in Phase 4 with the fp16 neighbour search (commit 2072357 holds their scores and
# data/artifacts/logs/phase4_outputs.sha256 their hashes) and kept: returns_ood is the four channel
# checkpoint on its input as delivered, ret0 the same with that channel set to 0.
KEPT = [f"{r}_seed{s}" for r in ("m0phase4", "m0prime", "m1") for s in "ABC"] + [
    "returns_ood",
    "ret0",
]
# Filled in when a step runs, from the number data/artifacts/votes.json holds.
VOTES = "{votes}"
BIG = "--recipe m0 --seed 0 --width 512 --batch 8 --lr 0.004 --steps 6000"


@dataclass
class Step:
    name: str
    commands: list[str]
    inputs: list[str]
    outputs: list[str]
    note: str = ""


def py(script: str, args: str = "") -> str:
    return f'"{PY}" {P}/{script} {args}'.strip()


STEPS = [
    Step(
        "fetch_dales2",
        [py("fetch_dales2.py")],
        ["huggingface.co/datasets/mbendjilali/DALES-2"],
        ["data/dales2/train/", "data/dales2/test/", "data/dales2_manifest.json"],
        "network",
    ),
    Step(
        "dales_stats",
        [py("dales_stats.py")],
        ["data/dales2/"],
        ["data/artifacts/dales_stats.json"],
        "class fractions per split; training weights come from the 29 training tiles",
    ),
    Step(
        "prep_dales",
        [py("prep_tiles.py", "--source dales --force")],
        ["data/dales2/"],
        ["data/prep/dales/", "data/artifacts/prep_dales.json"],
        "drops points with a dropped LAS flag",
    ),
    Step(
        "dales_water",
        [py("dales_water_check.py")],
        ["data/prep/dales/", "openmaps.gov.bc.ca Freshwater Atlas"],
        ["data/prep/dales_water_ignore/", "data/artifacts/dales_water_check.json"],
        "network",
    ),
    Step(
        "swiss_selection",
        [py("swiss_tile_selection.py"), py("swiss_tile_selection.py", "--holdout")],
        ["data.geo.admin.ch STAC"],
        ["data/artifacts/swiss_tile_selection.json", "data/artifacts/swiss_holdout_selection.json"],
        "network",
    ),
    Step(
        "prep_swiss",
        [
            py("prep_swiss.py"),
            py(
                "prep_swiss.py",
                "--selection data/artifacts/swiss_holdout_selection.json "
                "--out-dir data/prep/swiss_holdout "
                "--artifact data/artifacts/prep_swiss_holdout.json",
            ),
        ],
        ["data/artifacts/swiss_tile_selection.json", "data/artifacts/swiss_holdout_selection.json"],
        [
            "data/prep/swiss/",
            "data/prep/swiss_holdout/",
            "data/artifacts/prep_swiss.json",
            "data/artifacts/prep_swiss_holdout.json",
        ],
        "network; drops synthetic points",
    ),
    Step(
        "fetch_sf",
        [py("fetch_sf_aoi.py", "--aoi all")],
        ["s3 usgs-lidar-public CA_SanFrancisco_1_B23 EPT"],
        ["data/target/sf/", "data/artifacts/sf_aois.json"],
        "network",
    ),
    Step(
        "prep_sf",
        [py("prep_tiles.py", "--source target --force")],
        ["data/target/sf/"],
        ["data/prep/target/", "data/artifacts/prep_target.json"],
        "drops withheld points",
    ),
    Step(
        "datum",
        [py("datum_check.py")],
        ["s3 usgs-lidar-public source files"],
        ["data/artifacts/datum_check.json"],
        "network",
    ),
    Step(
        "footprints",
        [py("extract_footprints.py")],
        ["Microsoft US Building Footprints, California"],
        ["data/target/sf_footprints.geojson"],
        "network",
    ),
    Step(
        "naip_reference",
        [py("naip_reference.py")],
        ["data/prep/target/", "NAIP 2022 on the Planetary Computer"],
        ["data/reference/naip_<area>.npz", "data/artifacts/naip_reference.json"],
        "network; blind to the model",
    ),
    Step(
        "crs",
        [py("crs_check.py")],
        ["data/prep/target/", "data/target/sf_footprints.geojson", "data/cache/naip/"],
        ["data/artifacts/crs_check.json"],
        "lidar against NAIP and footprints, per area",
    ),
    Step(
        "naip_registered",
        [py("naip_reference.py", "--registered")],
        ["data/artifacts/crs_check.json"],
        [
            "data/reference/registered/naip_<area>.npz",
            "data/artifacts/naip_reference_registered.json",
        ],
    ),
    Step(
        "naip_mask_choice",
        [py("naip_mask_choice.py")],
        ["data/reference/", "data/prep/target/"],
        ["data/artifacts/naip_mask_choice.json"],
        "decided on multiple returns, never on predictions",
    ),
    Step(
        "water_polygon",
        [py("water_polygon.py")],
        ["data/prep/target/", "USGS NHD water polygons"],
        ["data/reference/water_polygon/", "data/artifacts/water_polygon.json"],
        "network",
    ),
    Step(
        "water_null",
        [py("water_null.py")],
        ["data/prep/target/", "NOAA tide gauge 9414290"],
        ["data/reference/water_null/", "data/artifacts/water_null.json"],
        "network",
    ),
    Step(
        "footprint_null",
        [py("footprint_null.py")],
        ["data/prep/target/", "data/target/sf_footprints.geojson"],
        ["data/artifacts/footprint_null.json"],
    ),
    Step(
        "input_range",
        [py("input_range_check.py")],
        ["data/prep/dales/", "data/prep/target/"],
        ["data/artifacts/input_range_check.json"],
    ),
    Step(
        "train_m0",
        [py("train.py", f"--recipe m0 --seed {SEED[x]}") for x in "ABC"]
        + [
            py(
                "train.py",
                f"{BIG} --ckpt data/checkpoints/m0_big_seedA.pt --out data/artifacts/train_m0_big_seedA.json",
            )
        ],
        ["data/prep/dales/", "data/artifacts/dales_stats.json", "data/artifacts/prep_dales.json"],
        ["data/checkpoints/m0_seed<A, B, C>.pt", "data/checkpoints/m0_big_seedA.pt"]
        + ["data/artifacts/train_m0_seed<A, B, C>.json", "data/artifacts/train_m0_big_seedA.json"],
        "GPU; class weights from the 29 training tiles",
    ),
    Step(
        "votes",
        [py("choose_votes.py", "--ckpt data/checkpoints/m0_seedA.pt")],
        ["data/checkpoints/m0_seedA.pt", "data/prep/dales/"],
        ["data/artifacts/votes.json"],
        "GPU; the number of voting passes, by its rule, on held-out Surrey only",
    ),
    Step(
        "input_channels",
        [py("input_channel_check.py")],
        ["data/prep/dales/", "data/prep/target/"],
        ["data/artifacts/input_channels.json"],
        "every channel a model reads is alive and in range",
    ),
    Step(
        "predict",
        [
            py(
                "predict_cache.py",
                f"--ckpt data/checkpoints/{v}.pt --out-dir data/predictions/{v} --votes {VOTES}",
            )
            for v in PREDICTED
        ],
        ["data/checkpoints/", "data/artifacts/votes.json", "data/prep/target/", "data/prep/dales/"],
        ["data/predictions/<variant>/"],
        "GPU",
    ),
    Step(
        "kept_predictions",
        [],
        ["the Phase 4 run on the rented card, with the fp16 neighbour search"],
        [
            "data/predictions/phase4/m0_seed<A, B, C>/",
            "data/predictions/m0prime_seed<A, B, C>/",
            "data/predictions/m1_seed<A, B, C>/",
            "data/predictions/manifest.json",
            "data/predictions/ret0/",
        ],
        "not run here; kept as made before the fix, each read only against models of its own search",
    ),
    Step(
        "score_sf",
        [py("score_variant.py", f"--variant {v}") for v in [*PREDICTED, *KEPT]],
        [
            "data/predictions/",
            "data/reference/",
            "data/target/sf_footprints.geojson",
            "data/artifacts/crs_check.json",
            "data/artifacts/naip_mask_choice.json",
        ],
        ["data/artifacts/sf_scores_<variant>.json"],
        "stops if any reference is missing",
    ),
    Step(
        "seeds",
        [py("aggregate_seeds.py", f"--recipe {r}") for r in RECIPES],
        ["data/artifacts/sf_scores_*.json", "data/artifacts/train_*.json"],
        ["data/artifacts/seeds_<recipe>.json"],
    ),
    Step(
        "compare",
        [
            py("compare_recipes.py", "--new m0 --old m0phase4"),
            py("compare_recipes.py", "--new m1 --old m0prime"),
        ],
        ["data/artifacts/seeds_*.json"],
        ["data/artifacts/compare_m0_vs_m0phase4.json", "data/artifacts/compare_m1_vs_m0prime.json"],
        "M0 against Phase 4's M0, and the old pair against itself only",
    ),
    Step(
        "quantities",
        [py("build_quantities.py")],
        ["data/artifacts/*.json"],
        ["data/artifacts/quantities.json"],
    ),
    Step(
        "frames",
        [
            py("render_frames.py", f"--area {a} --variant {v} --theme {t} --bg {g} --edl 0{x}")
            for a, v, x in (
                ("ggpark", "m0_seedA", ""),
                ("embarcadero", "m0_seedA", ""),
                ("sunset", "m0_seedA", ""),
                ("mission", "m0_seedA", ""),
                ("financial", "m0_seedA", ""),
                ("ggpark", "returns_ood", ""),
                ("5135_54430", "m0_seedA", " --surrey"),
            )
            for t, g in (("light", "e6e9e6"), ("dark", "131815"))
        ],
        ["data/predictions/m0_seedA/", "data/prep/target/", "docs/design/palette.json"],
        ["data/cache/frames/"],
    ),
    Step(
        "water_inset",
        [py("water_inset.py")],
        ["data/prep/swiss_holdout/", "data/prep/target/", "data/checkpoints/"],
        ["web/public/stills/water_inset_*.webp", "data/artifacts/water_inset.json"],
    ),
    Step(
        "page_figures",
        [py("page_figures.py")],
        ["data/artifacts/quantities.json", "data/cache/frames/"],
        ["web/public/figures.json", "web/public/asprs.json", "web/public/stills/"],
    ),
    Step(
        "octrees",
        [
            py(
                "build_octree.py",
                f"--areas {' '.join(AREAS)} --variants returns_ood ret0 m0_seedA m1_seedA",
            ),
            py(
                "build_octree.py",
                "--surrey --areas 5135_54430 --variants returns_ood ret0 m0_seedA m1_seedA",
            ),
        ]
        + [py("check_octree_decode.py", f"--area {a}") for a in AREAS]
        + [py("check_octree_decode.py", "--area 5135_54430 --surrey")],
        ["data/prep/target/", "data/predictions/", "PotreeConverter 2.1.3"],
        [
            "data/viewer/<area>/",
            "data/artifacts/octree_budget.json",
            "data/artifacts/octree_decode_check.json",
        ],
    ),
    Step(
        "readme",
        [py("render_readme.py")],
        ["docs/readme.template.md", "data/artifacts/quantities.json"],
        ["README.md"],
    ),
    Step("page", ["npm --prefix web ci", "npm --prefix web run build"], ["web/"], ["web/dist/"]),
    Step(
        "readme_images",
        [py("readme_images.py")],
        ["web/dist/", "data/viewer/"],
        ["docs/design/readme/first_screen.webp", "docs/design/readme/cards_1_and_3.webp"],
        "a browser on a hardware renderer",
    ),
    Step(
        "walkthrough",
        [py("record_walkthrough.py")],
        ["web/dist/", "data/viewer/", "ffmpeg"],
        ["docs/walkthrough/", "data/artifacts/walkthrough.json"],
        "one uninterrupted take; refuses a software renderer and a busy GPU",
    ),
    Step(
        "verify",
        [py("count_tests.py"), py("verify.py")],
        ["everything above"],
        ["data/artifacts/tests.json", "data/artifacts/verification.json"],
    ),
]


def run(step: Step) -> None:
    for cmd in step.commands:
        if VOTES in cmd:
            votes = json.loads(
                (ROOT / "data" / "artifacts" / "votes.json").read_text(encoding="utf-8")
            )
            cmd = cmd.replace(VOTES, str(votes["votes"]))
        print(f"$ {cmd}", flush=True)
        code = subprocess.call(cmd, shell=True, cwd=ROOT)
        if code:
            raise SystemExit(f"{step.name}: exit {code} from {cmd}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="say which outputs are on disk")
    ap.add_argument("--run", nargs="+", metavar="STEP", help="step names, or 'all'")
    args = ap.parse_args()
    names = [s.name for s in STEPS]
    if args.run:
        wanted = names if args.run == ["all"] else args.run
        unknown = [n for n in wanted if n not in names]
        if unknown:
            raise SystemExit(f"no such step: {unknown}; the steps are {names}")
        for step in STEPS:
            if step.name in wanted:
                run(step)
        return 0
    for i, s in enumerate(STEPS, 1):
        print(f"{i:>2}. {s.name}" + (f"  ({s.note})" if s.note else ""))
        print(f"      reads   {', '.join(s.inputs)}")
        for out in s.outputs:
            mark = ""
            if args.check:
                pattern = re.sub(r"<[^>]+>", "*", out).rstrip("/")
                mark = "  [on disk]" if any(ROOT.glob(pattern)) else "  [missing]"
            print(f"      writes  {out}{mark}")
        for cmd in s.commands[:2]:
            print(f"      $ {cmd.replace(PY, 'python')}")
        if len(s.commands) > 2:
            print(f"      ... and {len(s.commands) - 2} more of the same")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
