"""Write README.md and docs/method.md from their templates, each figure fetched from the registry.

Matching a README figure to a quantity by its value is a guess. With fourteen variants over hundreds of
metrics two unrelated quantities land on the same four decimals often enough that the guard which did
that was reporting the wrong model as the source. So the README is generated: each figure in the
template is a named registry key, `{{scores.m0_seedA.sf.soma.ground_recall}}`, and this fills it in from
data/artifacts/quantities.json. A number that has no key cannot be written into the README at all, and a
number whose key moves changes the README the moment the registry is rebuilt.

Tables whose cells are one quantity each are directives instead of cell by cell keys, because a table is
data and writing thirty keys by hand invites exactly the error this removes: `{{table:sf_buildings}}`.

    python scripts/pipeline/render_readme.py            write README.md and docs/method.md
    python scripts/pipeline/render_readme.py --check    fail if either is not what its template produces

tests/integrity/test_readme_numbers.py runs the check.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from surrey.areas import AREAS  # noqa: E402
from surrey.data import CLASS_NAMES, COARSE  # noqa: E402
from surrey.reference import REFERENCE_NOT_VALID  # noqa: E402

TEMPLATE = ROOT / "docs" / "readme.template.md"
README = ROOT / "README.md"
# The method notes the README links to for its detail, generated the same way from their own template.
METHOD_TEMPLATE = ROOT / "docs" / "method.template.md"
METHOD = ROOT / "docs" / "method.md"
# The README shows figures short; the method notes keep the four decimals the artefacts hold.
DOCUMENTS = ((TEMPLATE, README, True), (METHOD_TEMPLATE, METHOD, False))
SHORT = False
QUANTITIES = ROOT / "data" / "artifacts" / "quantities.json"
# {{key}} prints in the document's format; {{key@1}} with a fixed number of decimals; {{key%}} a share
# as a percentage, the way the page prints one.
KEY = re.compile(r"\{\{([a-z0-9_.:]+)(?:@(\d)|(%))?\}\}", re.I)

# Tables the template asks for by name. Each row names an area and each cell one registry key, so a
# table is data and no figure in it is typed.
SF_AREAS = [(label, key) for key, label in AREAS.items()]
# Each class with the figure that would catch the opposite mistake beside it: vegetation found beside
# vegetation called where the photograph sees none, ground found beside the band its precision lies in.
BOTH_SIDES = [
    ("NAIP vegetation cores called vegetation", "vegetation_headline"),
    ("NAIP not-vegetation cores called vegetation", "not_vegetation_core_called_vegetation"),
    ("Delivery ground called ground", "ground_recall"),
    ("Ground precision, floor", "ground_precision_floor"),
    ("Ground precision, ceiling", "ground_precision_ceiling"),
]
# Buildings in the areas where the footprint null calls anything a building, with the null beside.
BUILDING_COLUMNS = [
    ("Building lift", "building_lift_global_shift"),
    ("Precision against footprints", "building_precision"),
    ("The null's precision", "building_null_precision"),
    ("Footprint covered", "building_coverage"),
    ("The null's coverage", "building_null_coverage"),
]
# Ground and vegetation beside one-line rules scored the same way (scripts/pipeline/rule_baselines.py).
RULE_COLUMNS = [
    ("Delivery ground called ground", "ground_recall", "ground.recall"),
    ("Ground precision, floor", "ground_precision_floor", "ground.precision_floor"),
    (
        "NAIP vegetation cores called vegetation",
        "vegetation_headline",
        "vegetation.core_called_vegetation",
    ),
    (
        "NAIP not-vegetation cores called vegetation",
        "not_vegetation_core_called_vegetation",
        "vegetation.not_core_called_vegetation",
    ),
]
WATER_AREAS = [("Embarcadero", "embarcadero"), ("Bay Bridge", "baybridge")]


def values() -> dict:
    return {
        k: v["value"]
        for k, v in json.loads(QUANTITIES.read_text(encoding="utf-8"))["quantities"].items()
    }


def fmt(value) -> str:
    if isinstance(value, float):
        if SHORT:
            # Two decimals between 0.1 and 10, one above, two significant digits below, so a small
            # share never prints as 0.00.
            if value == 0:
                return "0"
            # A count taken as a median over seeds arrives as a float; 23.0 footprints are 23.
            if abs(value) >= 2 and value.is_integer():
                return f"{int(value):,}"
            if abs(value) >= 10:
                return f"{value:.1f}"
            if abs(value) >= 0.1:
                return f"{value:.2f}"
            # Fixed point, never 6.5e-05: as many decimals as two significant digits need, counted
            # after rounding, so 0.000995 prints as 0.0010 and not 0.00100.
            rounded = float(f"{value:.2g}")
            return f"{rounded:.{1 - math.floor(math.log10(abs(rounded)))}f}"
        text = f"{value:.4f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    return f"{value:,}" if isinstance(value, int) and abs(value) >= 10000 else str(value)


def pct(value: float) -> str:
    """A share as a percentage: whole at 10% and above, one decimal below, and never a real share
    printed as 0.0%. The same rule as web/src/format.js."""
    p = value * 100
    if p == 0:
        return "0%"
    if abs(p) < 0.1:
        return "under 0.1%"
    return f"{p:.0f}%" if abs(p) >= 10 else f"{p:.1f}%"


def spread(vals: dict, key: str) -> str:
    """A median with the seed range behind it, the way every per area figure is reported; where every
    seed gives the same figure there is no range to show."""
    median, low, high = (fmt(vals[f"{key}.{part}"]) for part in ("median", "min", "max"))
    return median if low == high else f"{median} ({low} to {high})"


def has_buildings(vals: dict, area: str) -> bool:
    return bool(vals.get(f"seeds.m0.sf.{area}.building_null_precision.median"))


def table_sf_both_sides(vals: dict) -> str:
    """A figure whose reference backend/surrey/reference.py marks as not valid in that area carries
    a dagger, and the template says why under the table."""
    head = "| Area | " + " | ".join(name for name, _ in BOTH_SIDES) + " |"
    rule = "|" + "---|" * (len(BOTH_SIDES) + 1)
    rows = []
    for label, area in SF_AREAS:
        cells = []
        for _, metric in BOTH_SIDES:
            key = f"seeds.m0.sf.{area}.{metric}"
            if vals.get(f"{key}.median") is None:
                cells.append("no core")
                continue
            mark = " \u2020" if (area, metric) in REFERENCE_NOT_VALID else ""
            cells.append(spread(vals, key) + mark)
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def table_sf_buildings(vals: dict) -> str:
    """Whether the footprints could be registered to the area's own lidar stands beside the count:
    where they could not, the area's figures rest on the offset measured elsewhere."""
    head = (
        "| Area | Footprints scored | Offset measured here | "
        + " | ".join(n for n, _ in BUILDING_COLUMNS)
        + " | Lift, seed A, 95% interval | The rule's lift |"
    )
    rule = "|" + "---|" * (len(BUILDING_COLUMNS) + 5)
    rows = []
    for label, area in SF_AREAS:
        if not has_buildings(vals, area):
            continue
        measured = vals[f"crs.sf.{area}.footprints_registered_by_correlation"]
        cells = [
            fmt(int(vals[f"seeds.m0.sf.{area}.footprints_scored.median"])),
            "yes" if measured else "no",
        ]
        cells += [spread(vals, f"seeds.m0.sf.{area}.{metric}") for _, metric in BUILDING_COLUMNS]
        ci = f"buildings_ci.sf.{area}.m0_seedA"
        cells.append(f"{fmt(vals[f'{ci}.lift_lo'])} to {fmt(vals[f'{ci}.lift_hi'])}")
        cells.append(fmt(vals[f"rules.sf.{area}.building.lift"]))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


HEIGHT_BINS = [
    ("under 15 m", "0_15"),
    ("15 to 30 m", "15_30"),
    ("30 m to the training blocks' 99th percentile", "30_p99"),
    ("from there to the tallest training block", "p99_max"),
    ("taller than any training block", "above_max"),
]


def table_by_height(vals: dict) -> str:
    """Building lift of seed A over 40 m blocks pooled by how much height they span, in the Financial
    District and in the other built areas (scripts/pipeline/building_intervals.py)."""
    rows = [
        "| Block spans | Financial District: lift (points) | Other built areas: lift (points) |",
        "|---|---|---|",
    ]
    for label, key in HEIGHT_BINS:
        cells = []
        for scope in ("financial", "without_financial"):
            base = f"byheight.{scope}.m0_seedA.{key}"
            if f"{base}.eligible_points" not in vals:
                cells.append("no blocks")
                continue
            lift = vals.get(f"{base}.lift")
            cells.append(
                f"{fmt(lift) if lift is not None else 'no footprint'} "
                f"({fmt(int(vals[f'{base}.eligible_points']))})"
            )
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


NDVI_SLUGS = ("t0_050", "otsu", "t0_100", "t0_150", "t0_200")


def table_ndvi(vals: dict) -> str:
    """The vegetation pair, median over seeds, against references rebuilt at other NDVI thresholds
    (scripts/pipeline/ndvi_sensitivity.py): vegetation found / false vegetation."""
    head = "| Area | " + " | ".join(
        f"NDVI {fmt(vals[f'ndvi.threshold.{s}'])}{' (Otsu)' if s == 'otsu' else ''}"
        for s in NDVI_SLUGS
    )
    rows = [head + " |", "|" + "---|" * (len(NDVI_SLUGS) + 1)]
    for label, area in SF_AREAS:
        if f"ndvi.sf.{area}.otsu.found" not in vals:
            continue
        cells = [
            f"{fmt(vals[f'ndvi.sf.{area}.{s}.found'])} / {fmt(vals[f'ndvi.sf.{area}.{s}.false'])}"
            for s in NDVI_SLUGS
        ]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def table_rules(vals: dict) -> str:
    """The model's ground and vegetation figures, each beside the same figure for a one-line rule:
    below 0.5 m is ground, 2-25 m up is vegetation. The vegetation rule scores 1.0 on both sides
    of the cores by construction."""
    head = "| Area | " + " | ".join(f"{n}, model / rule" for n, _, _ in RULE_COLUMNS) + " |"
    rows = [head, "|" + "---|" * (len(RULE_COLUMNS) + 1)]
    for label, area in SF_AREAS:
        cells = []
        for _, metric, rule_key in RULE_COLUMNS:
            model = vals.get(f"seeds.m0.sf.{area}.{metric}.median")
            rule = vals.get(f"rules.sf.{area}.{rule_key}")
            cells.append(
                "no core" if model is None or rule is None else f"{fmt(model)} / {fmt(rule)}"
            )
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def drop_rule(new: tuple, old: tuple) -> str:
    """backend/surrey/seeds.py's rule on two (min, max) seed ranges: moved only when every seed of one
    lies beyond every seed of the other."""
    if new[0] > old[1]:
        return "rose"
    if new[1] < old[0]:
        return "fell"
    return "no detectable move"


# Every claim written before the models it judges were trained, with what it said and any change
# made to it afterwards. Easy errors are read per area: a rate of raised points called ground above
# the home range is a transfer failure, inside it a weakness the model already had at home.
CLAIMS = [
    (
        "extrapolation",
        "Buildings score lower in the Financial District, the one area taller than training, than the "
        "median of Embarcadero, Mission, SoMa and Sunset: lift lower by at least 0.05",
        "",
    ),
    (
        "relief",
        "On Russian Hill, delivery ground in steep 40 m blocks is called building more often than in "
        "the rest, by at least 0.05",
        "the primary metric was changed after the first run, to the one the earliest written "
        "prediction named",
    ),
    (
        "hard_errors_home_vs_water",
        "On water at Embarcadero, confidence separates wrong ground calls from right ones less well "
        "than near-ground mistakes at home, by at least 0.05 AUROC",
        "the two references differ: delivery water against human labels at home",
    ),
    (
        "extrapolation_within_financial",
        "Inside the Financial District, blocks taller than any training block score lower than the "
        "district's other blocks, lift lower by at least 0.05",
        "an addition, written after the claim above and not before training",
    ),
]


def table_claims(vals: dict) -> str:
    """Each pre-registered claim with the number of seeds it holds on; a claim stands only on all."""
    rows = [
        "| Claim | Holds on | Stands | Changed afterwards |",
        "|---|---|---|---|",
    ]
    for key, said, change in CLAIMS:
        k, n = int(vals[f"seeds.m0.claims.{key}.k"]), int(vals[f"seeds.m0.claims.{key}.n"])
        rows.append(
            f"| {said} | {k} of {n} seeds | {'yes' if k == n else 'no'} | {change or 'no'} |"
        )
    return "\n".join(rows)


def table_easy_errors(vals: dict) -> str:
    """The reading fixed for easy errors, per area: delivery unassigned points more than 3 m up called
    ground, at a rate above every held-out Surrey tile (a transfer failure) or inside that range (a
    weakness the model has at home too). A reading, so it has no side that is expected."""
    rows = [
        "| Area | Rate | Above the home range | Inside it |",
        "|---|---|---|---|",
    ]
    for label, area in SF_AREAS:
        base = f"seeds.m0.claims.easy.{area}"
        if f"{base}.transfer.k" not in vals:
            continue
        n = int(vals[f"{base}.transfer.n"])
        rows.append(
            f"| {label} | {spread(vals, f'seeds.m0.sf.{area}.raised_unassigned_called_ground_rate')} "
            f"| {int(vals[f'{base}.transfer.k'])} of {n} seeds "
            f"| {int(vals[f'{base}.home_weakness.k'])} of {n} seeds |"
        )
    return "\n".join(rows)


def table_surrey_classes(vals: dict) -> str:
    """Held-out Surrey IoU per class, the published recipe beside the same recipe trained with the fp16 search, which differs from it only
    in the neighbour search."""
    head = (
        "| Class | Drawn as | Share of held-out points | IoU, published model | IoU, fp16-search model "
        "| Moved |"
    )
    rule = "|---|---|---|---|---|---|"
    group_of = {name: group for group, names in COARSE.items() for name in names}
    rows = []
    for name in CLASS_NAMES:
        new, old = (f"seeds.{r}.surrey.test.iou.{name}" for r in ("m0", "m0fp16"))
        share = fmt(vals[f"surrey.test.voxel.class_fraction.{name}"])
        move = drop_rule(
            (vals[f"{new}.min"], vals[f"{new}.max"]), (vals[f"{old}.min"], vals[f"{old}.max"])
        )
        rows.append(
            f"| {name} | {group_of[name]} | {share} | {spread(vals, new)} | {spread(vals, old)} "
            f"| {move} |"
        )
    return "\n".join([head, rule, *rows])


MOVE_LABELS = {
    "ground_precision_floor": "ground precision, floor",
    "ground_recall": "delivery ground called ground",
    "not_vegetation_core_called_vegetation": "not-vegetation cores called vegetation",
    "vegetation_headline": "vegetation cores called vegetation",
    "raised_unassigned_called_ground_rate": "raised unassigned points called ground",
    "bridge_deck_called_building": "bridge deck called building",
    "building_lift_global_shift": "building lift",
    "water_called_ground": "water called ground",
}


def moves_table(pair: str, old_name: str, new_name: str):
    """Every figure that moved between two recipes by the drop rule, with both recipes' median and
    seed range from the registry's compare.<pair>.move keys, so a move is read with its size."""

    def build(vals: dict) -> str:
        prefix = f"compare.{pair}.move."
        area_label = {a: label for label, a in SF_AREAS} | {"surrey": "held-out Surrey"}
        rows = []
        for key in sorted(k for k in vals if k.startswith(prefix)):
            if key.endswith((".median", ".min", ".max")):
                continue
            area, metric = key[len(prefix) :].split(".", 1)
            what = f"IoU, {metric}" if area == "surrey" else MOVE_LABELS.get(metric, metric)
            rows.append(
                f"| {area_label[area]} | {what} | {spread(vals, f'{key}.old')} "
                f"| {spread(vals, f'{key}.new')} | {'better' if vals[key] > 0 else 'worse'} |"
            )
        head = f"| Where | Figure | {old_name} | {new_name} | Moved |"
        return "\n".join([head, "|---|---|---|---|---|", *rows])

    return build


def table_bug(vals: dict) -> str:
    """One checkpoint with one input changed, so single figures and no seed range; the false
    vegetation beside each, so a model that calls everything vegetation cannot look fixed."""
    head = (
        "| Area | Vegetation cores called vegetation, as delivered | the same, ratio set to 0 "
        "| Not-vegetation cores called vegetation, as delivered | the same, ratio set to 0 |"
    )
    rule = "|---|---|---|---|---|"
    rows = []
    for label, area in SF_AREAS:
        cells = []
        for metric in ("vegetation_headline", "not_vegetation_core_called_vegetation"):
            for variant in ("returns_ood", "ret0"):
                v = vals.get(f"scores.{variant}.sf.{area}.{metric}")
                mark = " \u2020" if (area, metric) in REFERENCE_NOT_VALID else ""
                cells.append("no core" if v is None else fmt(v) + mark)
        if cells[0] == "no core":
            continue
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def table_m1_water(vals: dict) -> str:
    head = "| area | seed A | seed B | seed C | result |"
    rule = "|---|---|---|---|---|"
    rows = []
    for label, area in WATER_AREAS:
        seeds = [
            fmt(vals[f"compare.m1_vs_m0prime.water.{area}.lift.seed_{s}"]) for s in ("a", "b", "c")
        ]
        k = fmt(vals[f"compare.m1_vs_m0prime.water.{area}.positive_on"])
        n = fmt(vals[f"compare.m1_vs_m0prime.water.{area}.seeds"])
        rows.append(f"| {label} | " + " | ".join(seeds) + f" | positive on {k} of {n} seeds |")
    return "\n".join([head, rule, *rows])


VERIFY_CHECKS = [
    "behavioural tests",
    "integrity tests",
    "README is generated",
    "page figures",
    "viewer tests",
    "viewer build",
    "page smoke test",
]


def table_verification(vals: dict) -> str:
    """Each check with the command a reader runs and what the last run of scripts/pipeline/verify.py printed."""
    head = "| Check | Command | Result |"
    rule = "|---|---|---|"
    rows = []
    for check in VERIFY_CHECKS:
        slug = check.replace(" ", "_")
        rows.append(
            f"| {check} | `{vals[f'verification.{slug}.command']}` | {vals[f'verification.{slug}.summary']} |"
        )
    return "\n".join([head, rule, *rows])


TABLES = {
    "sf_both_sides": table_sf_both_sides,
    "sf_buildings": table_sf_buildings,
    "surrey_classes": table_surrey_classes,
    "claims": table_claims,
    "rules": table_rules,
    "by_height": table_by_height,
    "ndvi": table_ndvi,
    "easy_errors": table_easy_errors,
    "moves_m0_vs_m0fp16": moves_table("m0_vs_m0fp16", "fp16 search", "fp32 search"),
    "moves_m1_vs_m0prime": moves_table("m1_vs_m0prime", "M0prime", "M1"),
    "bug": table_bug,
    "m1_water": table_m1_water,
    "verification": table_verification,
}


def render(template: str, vals: dict, short: bool = False) -> str:
    global SHORT
    SHORT = short
    try:
        return _render(template, vals)
    finally:
        SHORT = False


def _render(template: str, vals: dict) -> str:
    missing = []

    def one(m: re.Match) -> str:
        name = m.group(1)
        if name.startswith("table:"):
            builder = TABLES.get(name.split(":", 1)[1])
            if builder is None:
                missing.append(name)
                return m.group(0)
            return builder(vals)
        if name not in vals:
            missing.append(name)
            return m.group(0)
        if m.group(2) is not None:
            return f"{vals[name]:.{int(m.group(2))}f}"
        if m.group(3):
            return pct(vals[name])
        return fmt(vals[name])

    out = KEY.sub(one, template)
    # A range typed into the prose, "0.54 (0.54 to 0.54)", says nothing when every seed agrees.
    out = re.sub(r"(?<![\w.])(-?[\d.,]+%?) \(\1 to \1\)", r"\1", out)
    if missing:
        raise SystemExit(
            "the template names things the registry does not have: "
            + ", ".join(sorted(set(missing)))
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="compare instead of writing, for CI and the guard"
    )
    args = ap.parse_args()
    vals = values()
    failed = 0
    for template, target, short in DOCUMENTS:
        rendered = render(template.read_text(encoding="utf-8"), vals, short=short)
        name = target.relative_to(ROOT).as_posix()
        if not args.check:
            # The template is LF and the check below compares the text as it is, so writing Windows
            # line endings here would make every line differ and fail a guard on an untouched file.
            target.write_text(rendered, encoding="utf-8", newline=chr(10))
            print(f"wrote {name}")
            continue
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current == rendered:
            print(f"{name} is what its template and the registry produce")
            continue
        failed += 1
        diff = difflib.unified_diff(
            current.splitlines(),
            rendered.splitlines(),
            name,
            "rendered from the template",
            lineterm="",
            n=1,
        )
        print("\n".join(list(diff)[:60]))
    if failed:
        print("\nrun: python scripts/pipeline/render_readme.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
