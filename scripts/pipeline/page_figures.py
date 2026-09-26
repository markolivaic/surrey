"""Resolve every figure on the viewer page from the registry, and copy the phone stills.

Each number on web/index.html is a placeholder naming a registry key and where it stands
(data-q and data-where). This writes web/public/figures.json and fails the build when:

    a key is not in data/artifacts/quantities.json or has no value
    a key's variant may not stand where it is placed: the old four channel model (returns_ood)
    and its forced-zero input (ret0) only on card 3, the bug card; the M1 card only the old M1 and
    figures with no model; everywhere else the published model, the spread over its seeds, or no model

The M1 card's figures are optional until M1 is trained; m1card.js shows "not trained yet" while the
comparison artefact does not exist. The phone stills come from data/cache/frames and are drawn
with the page palette.

Run:
    python scripts/pipeline/page_figures.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
PAGE = ROOT / "web" / "index.html"
M1CARD = ROOT / "web" / "src" / "m1card.js"
PUBLIC = ROOT / "web" / "public"
FRAMES = ROOT / "data" / "cache" / "frames"
# The first screen draws the published model over one area, and names the data itself. Its own
# figures may therefore be the published model's or nobody's, which is the default rule.
# The M1 card quotes every seed where it shows a change per seed, and the one density run by name.
ALLOWED = {
    "card3": {"returns_ood", "ret0", "none"},
    "m1card": {"m1_seeds", "m1_seedA", "m1_seedB", "m1_seedC", "m1_densityaug_seedA", "none"},
}
# The first screen can be pointed at any of these; the page takes one and the figures for all of
# them travel with it, so switching area never leaves a number unresolved.
HERO_AREAS = ("sunset", "mission", "financial", "ggpark")
# The model's coarse groups that have no code in the standard, drawn below its table.
NOCODE_GROUPS = ("pole", "vehicle", "fence")
DEFAULT_ALLOWED = {"m0_seedA", "m0_seeds", "none"}
PLACEHOLDER = re.compile(r'data-q="([^"]+)"[^>]*data-where="([^"]+)"')
# The first screen can draw either candidate area, so both stills travel with the page.
STILLS = [
    "ggpark_m0_seedA",
    "5135_54430_home_m0_seedA",
    "ggpark_returns_ood",
    "embarcadero_m0_seedA",
    "sunset_m0_seedA",
    "mission_m0_seedA",
    "financial_m0_seedA",
]


def check(
    registry: dict, placements: list[tuple[str, str]], optional: set[str] = frozenset()
) -> tuple[dict, list[str]]:
    figures, problems = {}, []
    for key, where in placements:
        q = registry.get(key)
        if q is None or q.get("value") is None:
            if key not in optional:
                problems.append(f"{key}: not in the registry")
            continue
        variant = q["population"]["variant"]
        if variant not in ALLOWED.get(where, DEFAULT_ALLOWED):
            problems.append(f"{key}: variant {variant} may not stand on {where}")
            continue
        figures[key] = {"value": q["value"], "variant": variant, "where": where}
    return figures, problems


def hero_keys() -> list[str]:
    """What the first screen needs: the standard's codes counted over the area it draws, and the
    provenance strip. Listed here rather than scraped from the page, because the legend's rows are
    built from the standard's table at run time and a scrape would find no placeholder to read."""
    from surrey.asprs import ASPRS_CLASSES

    keys = [
        "provenance.sf.flight_date",
        "provenance.sf.naip_year",
        "provenance.sf.days_from_imagery_to_lidar",
        "provenance.dales.train.tiles",
        "provenance.dales.test.tiles",
        "surrey.train.voxel.training_points",
        "provenance.sf.embarcadero.flight_first_utc",
        "waternull.sf.embarcadero.gauge_level_navd88_m.median",
    ]
    for area in HERO_AREAS:
        keys.append(f"provenance.sf.{area}.points_per_m2")
        keys.append(f"provenance.sf.{area}.drawn_points_per_m2")
        for group in NOCODE_GROUPS:
            keys += [
                f"asprs.{area}.m0_seedA.nocode.{group}.points",
                f"asprs.{area}.m0_seedA.nocode.{group}.share",
            ]
        for code in ASPRS_CLASSES:
            for part in ("points", "share"):
                keys.append(f"asprs.{area}.m0_seedA.code{code}.{part}")
    return keys


def asprs_table() -> dict:
    """The standard's table as the page draws it: the code, the standard's wording, whether the
    published model has a word for it, and whether the delivery itself labels it."""
    from surrey.asprs import ASPRS_CLASSES, DELIVERY_LABELS, SOURCE, codes_for
    from surrey.data import CLASS_NAMES
    from surrey.sources import M1_CLASS_NAMES

    m0, m1 = codes_for(CLASS_NAMES), codes_for(M1_CLASS_NAMES)
    return {
        "source": SOURCE,
        "classes": [
            {
                "code": code,
                "name": name,
                "m0": sorted(m0.get(code, [])),
                "m1": sorted(m1.get(code, [])),
                "delivery_labels": code in DELIVERY_LABELS,
            }
            for code, name in ASPRS_CLASSES.items()
        ],
    }


def main() -> int:
    registry = json.loads(
        (ROOT / "data" / "artifacts" / "quantities.json").read_text(encoding="utf-8")
    )
    registry = registry.get("quantities", registry)
    placements = PLACEHOLDER.findall(PAGE.read_text(encoding="utf-8"))
    m1card_text = M1CARD.read_text(encoding="utf-8")
    m1card_keys = sorted(set(re.findall(r"compare\.m1_vs_m0prime\.water\.[a-z_.]+", m1card_text)))
    # The inset's figures, named in full in the card's source; a spread names its stem and gets all three.
    # Keys carry capitals where they name a seed, m1_seedA, so the pattern has to take them.
    inset_keys = re.findall(
        r'v\("((?:swiss_holdout|water|crop_density|density|lattice_test|lattice|rented|realwater|flags)\.[A-Za-z0-9_.]+)"\)',
        m1card_text,
    )
    inset_keys += [
        f"{stem}.{part}"
        for stem in re.findall(r'spread\("([a-z0-9_.]+)"\)', m1card_text)
        for part in ("median", "min", "max")
    ]
    expanded = []
    for key in m1card_keys:
        if "${key}" in key or key.endswith("."):
            continue
        expanded.append(key)
    for area in ("embarcadero", "baybridge"):
        for part in ("positive_on", "seeds", "lift.median", "lift.min", "lift.max"):
            expanded.append(f"compare.m1_vs_m0prime.water.{area}.{part}")
    expanded.append("compare.m1_vs_m0prime.water.success")
    expanded += inset_keys
    placements += [(k, "m1card") for k in sorted(set(expanded))]
    placements += [(k, "hero") for k in hero_keys()]
    # A standard code the published model has no word for has no figure, and that absence is what the
    # legend is for: those keys are allowed to be missing, and only those.
    from surrey.asprs import ASPRS_CLASSES, codes_for
    from surrey.data import CLASS_NAMES

    can_name = codes_for(CLASS_NAMES)
    optional = {k for k, w in placements if w == "m1card"}
    optional |= {
        f"asprs.{area}.m0_seedA.code{code}.{part}"
        for area in HERO_AREAS
        for code in ASPRS_CLASSES
        if code not in can_name
        for part in ("points", "share")
    }
    figures, problems = check(registry, placements, optional=optional)
    if problems:
        print("\n".join(problems))
        return 1
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "figures.json").write_text(
        json.dumps(figures, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (PUBLIC / "asprs.json").write_text(
        json.dumps(asprs_table(), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # The stills are committed with the page. Their frames are working material in data/cache, which a
    # clone does not have, so a frame that is there refreshes its still and a still with no frame is
    # kept as committed; only a still that is in neither place is a fault.
    stills = PUBLIC / "stills"
    stills.mkdir(exist_ok=True)
    copied, kept, lost = 0, 0, []
    for name in STILLS:
        for ground in ("e6e9e6", "131815"):
            frame, still = (
                FRAMES / f"{name}_oblique_{ground}_edl00_p2.png",
                stills / f"{name}_{ground}.webp",
            )
            if frame.exists():
                from PIL import Image

                Image.open(frame).convert("RGB").save(still, "WEBP", quality=80, method=6)
                copied += 1
            elif still.exists():
                kept += 1
            else:
                lost.append(still.name)
    if lost:
        print("stills with no frame and no committed copy: " + ", ".join(lost))
        return 1
    print(
        f"{len(figures)} figures resolved, {copied} stills copied and {kept} kept as committed, "
        f"{len(asprs_table()['classes'])} standard classes written"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
