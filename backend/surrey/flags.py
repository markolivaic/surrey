"""The four LAS point flags, and which of them keep a point out of the prepared data.

LAS 1.4 carries four flags beside a point's class: synthetic (the point was made, not measured), key
point (kept by a thinning for a surface model), withheld (not to be used), and overlap (from the
overlapping edge of a flight line). Formats 0 to 5 have the first three and no overlap bit.

What each source actually sets, class by class, is measured by scripts/investigations/las_flags_audit.py
into data/artifacts/las_flags_audit.json. The decision for each flag in each source is here, with its
reason, and the preparation scripts apply it: a point with a dropped flag never reaches a prepared tile,
so it is neither trained on nor predicted.
"""

from __future__ import annotations

import numpy as np

SYNTHETIC, KEY_POINT, WITHHELD, OVERLAP = 1, 2, 4, 8
BITS = {"synthetic": SYNTHETIC, "key_point": KEY_POINT, "withheld": WITHHELD, "overlap": OVERLAP}
DROP = SYNTHETIC | WITHHELD

SOURCES = ("mbendjilali/DALES-2", "swissSURFACE3D", "USGS 3DEP CA_SanFrancisco_1_B23")

# flag -> source -> (decision, reason). A decision is "drop" or "keep"; no flag is set on a labelled
# point in a way that calls for keeping the point and ignoring it in the loss.
DECISIONS = {
    "synthetic": {
        "mbendjilali/DALES-2": (
            "drop",
            "never set; the rule every source follows, and it removes nothing here",
        ),
        "swissSURFACE3D": (
            "drop",
            "set on almost all class 9 water and on a small share of ground: points "
            "swisstopo placed to fill the surface, not returns, and the regular "
            "lattice M1 learned water from",
        ),
        "USGS 3DEP CA_SanFrancisco_1_B23": (
            "drop",
            "never set in the sampled nodes; the same rule",
        ),
    },
    "key_point": {
        "mbendjilali/DALES-2": (
            "keep",
            "never set; it marks a measured point a thinning kept, nothing against the point",
        ),
        "swissSURFACE3D": ("keep", "never set; the same reason"),
        "USGS 3DEP CA_SanFrancisco_1_B23": (
            "keep",
            "never set in the sampled nodes; the same reason",
        ),
    },
    "withheld": {
        "mbendjilali/DALES-2": (
            "drop",
            "never set; the specification says a withheld point is not to be used",
        ),
        "swissSURFACE3D": ("drop", "never set; the same rule"),
        "USGS 3DEP CA_SanFrancisco_1_B23": (
            "drop",
            "set on every point of the two noise classes, 7 and 18, "
            "and on nothing else in the sampled nodes",
        ),
    },
    "overlap": {
        "mbendjilali/DALES-2": (
            "keep",
            "never set; an overlap point is a measured return from the edge of a flight line",
        ),
        "swissSURFACE3D": ("keep", "never set; the same reason"),
        "USGS 3DEP CA_SanFrancisco_1_B23": (
            "keep",
            "the Entwine (EPT) copy read here carries no overlap bit, so overlap points "
            "cannot be told apart in it",
        ),
    },
}


def flag_bits(las) -> np.ndarray:
    """One byte per point, with a bit for each flag the file's point format carries."""
    names = set(las.point_format.dimension_names)
    out = np.zeros(len(las.points), dtype=np.uint8)
    for name, bit in BITS.items():
        if name in names:
            out |= np.asarray(getattr(las, name), dtype=bool).astype(np.uint8) * np.uint8(bit)
    return out


def kept(bits: np.ndarray) -> np.ndarray:
    """True for every point no dropped flag is set on."""
    return (np.asarray(bits, dtype=np.uint8) & DROP) == 0


def dropped_counts(bits: np.ndarray, classes: np.ndarray) -> dict:
    """Points each dropped flag removes, per class, for the preparation's own record."""
    out = {}
    for name, bit in BITS.items():
        if not DROP & bit:
            continue
        hit = (np.asarray(bits) & bit) != 0
        codes, counts = np.unique(np.asarray(classes)[hit], return_counts=True)
        out[name] = {str(int(c)): int(n) for c, n in zip(codes, counts, strict=True)}
    return out
