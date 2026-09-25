"""Write README.md from docs/readme.template.md, with every figure fetched from the registry by name.

Matching a README figure to a quantity by its value is a guess. With fourteen variants over hundreds of
metrics two unrelated quantities land on the same four decimals often enough that the guard which did
that was reporting the wrong model as the source. So the README is generated: each figure in the
template is a named registry key, `{{scores.m0_seedA.sf.soma.ground_recall}}`, and this fills it in from
data/artifacts/quantities.json. A number that has no key cannot be written into the README at all, and a
number whose key moves changes the README the moment the registry is rebuilt.

Tables whose cells are one quantity each are directives instead of cell by cell keys, because a table is
data and writing thirty keys by hand invites exactly the error this removes: `{{table:sf_areas}}`.

    python scripts/pipeline/render_readme.py            write README.md
    python scripts/pipeline/render_readme.py --check    fail if README.md is not what the template produces

tests/integrity/test_readme_generated.py runs the check, and the older guard that matches figures by
value stays on as the last net for anything typed in by hand.
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
QUANTITIES = ROOT / "data" / "artifacts" / "quantities.json"
KEY = re.compile(r"\{\{([a-z0-9_.:]+)\}\}", re.I)

# Tables the template asks for by name. Each is (heading row, alignment row, rows), where a row names the
# label and the keys of its cells; a key of None prints the literal text given instead.
SF_AREAS = [
    ("Golden Gate Park", "ggpark"),
    ("Sunset", "sunset"),
    ("Mission", "mission"),
    ("Russian Hill", "steep"),
    ("SoMa", "soma"),
    ("Financial District", "financial"),
    ("Embarcadero", "embarcadero"),
]
SF_COLUMNS = [
    ("Building lift over the footprint null", "building_lift_global_shift"),
    ("NAIP vegetation cores called vegetation", "vegetation_headline"),
    ("Delivery ground called ground", "ground_recall"),
]
# The bug table is in the order the README told it, park first and Embarcadero last, and reads the
# pair of keys that measurement wrote: the input as delivered, and the same with the ratio forced to 0.
BUG_AREAS = [
    ("Golden Gate Park", "ggpark"),
    ("Sunset", "sunset"),
    ("Mission", "mission"),
    ("SoMa", "soma"),
    ("Russian Hill", "steep"),
    ("Financial District", "financial"),
    ("Embarcadero", "embarcadero"),
]
WATER_AREAS = [("Embarcadero", "embarcadero"), ("Bay Bridge", "baybridge")]


def values() -> dict:
    return {
        k: v["value"]
        for k, v in json.loads(QUANTITIES.read_text(encoding="utf-8"))["quantities"].items()
    }


def fmt(value) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    return f"{value:,}" if isinstance(value, int) and abs(value) >= 10000 else str(value)


def spread(vals: dict, key: str) -> str:
    """A median with the seed range behind it, the way every per area figure is reported."""
    median, low, high = (vals[f"{key}.{part}"] for part in ("median", "min", "max"))
    return f"{fmt(median)} ({fmt(low)} to {fmt(high)})"


def table_sf_areas(vals: dict) -> str:
    head = "| Area | " + " | ".join(name for name, _ in SF_COLUMNS) + " |"
    rule = "|" + "---|" * (len(SF_COLUMNS) + 1)
    rows = []
    for label, area in SF_AREAS:
        cells = []
        for _, metric in SF_COLUMNS:
            key = f"seeds.m0.sf.{area}.{metric}"
            no_footprints = (
                metric == "building_lift_global_shift"
                and vals.get(f"footprints.{area}.share_of_tall_points_inside") == 0.0
            )
            if no_footprints or f"{key}.median" not in vals:
                cells.append("no buildings")
            else:
                cells.append(spread(vals, key))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def table_bug(vals: dict) -> str:
    head = "| Area | NAIP vegetation cores called vegetation, input as delivered | the same, return ratio set to 0 |"
    rule = "|---|---|---|"
    rows = []
    for label, area in BUG_AREAS:
        a = vals[f"ret0.sf.{area}.before.naip_core_called_vegetation"]
        b = vals[f"ret0.sf.{area}.after.naip_core_called_vegetation"]
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
    "sf_areas": table_sf_areas,
    "bug": table_bug,
    "m1_water": table_m1_water,
    "verification": table_verification,
}


def render(template: str, vals: dict) -> str:
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
    rendered = render(TEMPLATE.read_text(encoding="utf-8"), values())
    if not args.check:
        # The template is LF and the check below compares the text as it is, so writing Windows
        # line endings here would make every line differ and fail a guard on an untouched file.
        README.write_text(rendered, encoding="utf-8", newline=chr(10))
        print(f"wrote {README}")
        return 0
    current = README.read_text(encoding="utf-8")
    if current == rendered:
        print("README.md is what the template and the registry produce")
        return 0
    diff = difflib.unified_diff(
        current.splitlines(),
        rendered.splitlines(),
        "README.md",
        "rendered from the template",
        lineterm="",
        n=1,
    )
    print("\n".join(list(diff)[:60]))
    print("\nrun: python scripts/pipeline/render_readme.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
