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

tests/integrity/test_readme_generated.py runs the check, and fails any decimal typed into a template
instead of named.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "docs" / "readme.template.md"
README = ROOT / "README.md"
# The method notes the README links to for its detail, generated the same way from their own template.
METHOD_TEMPLATE = ROOT / "docs" / "method.template.md"
METHOD = ROOT / "docs" / "method.md"
# The README shows figures short; the method notes keep the four decimals the artefacts hold.
DOCUMENTS = ((TEMPLATE, README, True), (METHOD_TEMPLATE, METHOD, False))
SHORT = False
QUANTITIES = ROOT / "data" / "artifacts" / "quantities.json"
KEY = re.compile(r"\{\{([a-z0-9_.:]+)\}\}", re.I)

# Tables the template asks for by name. Each row names an area and each cell one registry key, so a
# table is data and no figure in it is typed.
SF_AREAS = [
    ("Golden Gate Park", "ggpark"),
    ("Sunset", "sunset"),
    ("Mission", "mission"),
    ("Russian Hill", "steep"),
    ("SoMa", "soma"),
    ("Financial District", "financial"),
    ("Embarcadero", "embarcadero"),
    ("Bay Bridge", "baybridge"),
]
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
# The first round's M0 against the retrained M0, as scripts/investigations/building_breakdown.py set
# them side by side.
CHANGE_COLUMNS = [
    ("Precision, first round", "precision", "old"),
    ("Precision, retrained", "precision", "new"),
    ("Coverage, first round", "coverage_median", "old"),
    ("Coverage, retrained", "coverage_median", "new"),
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
            if abs(value) >= 10:
                return f"{value:.1f}"
            if abs(value) >= 0.1:
                return f"{value:.2f}"
            return f"{value:.2g}"
        text = f"{value:.4f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    return f"{value:,}" if isinstance(value, int) and abs(value) >= 10000 else str(value)


def spread(vals: dict, key: str) -> str:
    """A median with the seed range behind it, the way every per area figure is reported; where every
    seed gives the same figure there is no range to show."""
    median, low, high = (fmt(vals[f"{key}.{part}"]) for part in ("median", "min", "max"))
    return median if low == high else f"{median} ({low} to {high})"


def has_buildings(vals: dict, area: str) -> bool:
    return bool(vals.get(f"seeds.m0.sf.{area}.building_null_precision.median"))


def table_sf_both_sides(vals: dict) -> str:
    head = "| Area | " + " | ".join(name for name, _ in BOTH_SIDES) + " |"
    rule = "|" + "---|" * (len(BOTH_SIDES) + 1)
    rows = []
    for label, area in SF_AREAS:
        cells = []
        for _, metric in BOTH_SIDES:
            key = f"seeds.m0.sf.{area}.{metric}"
            cells.append(spread(vals, key) if vals.get(f"{key}.median") is not None else "no core")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def table_sf_buildings(vals: dict) -> str:
    head = "| Area | Footprints scored | " + " | ".join(n for n, _ in BUILDING_COLUMNS) + " |"
    rule = "|" + "---|" * (len(BUILDING_COLUMNS) + 2)
    rows = []
    for label, area in SF_AREAS:
        if not has_buildings(vals, area):
            continue
        cells = [fmt(int(vals[f"seeds.m0.sf.{area}.footprints_scored.median"]))]
        cells += [spread(vals, f"seeds.m0.sf.{area}.{metric}") for _, metric in BUILDING_COLUMNS]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def moved(vals: dict, area: str, fig: str) -> str:
    """The drop rule of backend/surrey/seeds.py: a figure moved only when every retrained seed lies
    beyond every first round seed."""
    old, new = (
        (vals[f"buildings.{area}.{fig}.{w}.min"], vals[f"buildings.{area}.{fig}.{w}.max"])
        for w in ("old", "new")
    )
    if new[0] > old[1]:
        return "rose"
    if new[1] < old[0]:
        return "fell"
    return "did not move"


def building_reading(vals: dict, area: str) -> str:
    """The reading written in docs/phase4_criteria.md before the breakdown ran."""
    precision, coverage = moved(vals, area, "precision"), moved(vals, area, "coverage_median")
    if coverage == "rose" and precision == "fell":
        return "a trade: more of each footprint covered, less precisely"
    if coverage == "fell" and precision == "fell":
        return "worse at buildings here"
    return f"precision {precision}, coverage {coverage}"


def table_building_change(vals: dict) -> str:
    head = (
        "| Area | Lift, first round to retrained | "
        + " | ".join(n for n, _, _ in CHANGE_COLUMNS)
        + " | Reading |"
    )
    rule = "|" + "---|" * (len(CHANGE_COLUMNS) + 3)
    rows = []
    for label, area in SF_AREAS:
        if not has_buildings(vals, area):
            continue
        lift = (
            f"{fmt(vals[f'buildings.{area}.lift.old.median'])} to "
            f"{fmt(vals[f'buildings.{area}.lift.new.median'])}, {moved(vals, area, 'lift')}"
        )
        cells = [lift]
        cells += [spread(vals, f"buildings.{area}.{fig}.{when}") for _, fig, when in CHANGE_COLUMNS]
        cells.append(building_reading(vals, area))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def table_bug(vals: dict) -> str:
    head = "| Area | NAIP vegetation cores called vegetation, input as delivered | the same, return ratio set to 0 |"
    rule = "|---|---|---|"
    rows = []
    for label, area in SF_AREAS:
        a = vals.get(f"scores.returns_ood.sf.{area}.vegetation_headline")
        b = vals.get(f"scores.ret0.sf.{area}.vegetation_headline")
        if a is None or b is None:
            continue
        rows.append(f"| {label} | {fmt(a)} | {fmt(b)} |")
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
    "building_change": table_building_change,
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
        return fmt(vals[name])

    out = KEY.sub(one, template)
    # A range typed into the prose, "0.54 (0.54 to 0.54)", says nothing when every seed agrees.
    out = re.sub(r"(?<![\w.])(-?[\d.,]+) \(\1 to \1\)", r"\1", out)
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
