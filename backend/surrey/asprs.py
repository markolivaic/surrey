"""The standard's own class table, and which of its codes this project's models can name.

Quoted from Table 17, "ASPRS Standard Point Classes (Point Data Record Formats 6-10)", LAS
Specification v.1.4 - R15, American Society for Photogrammetry and Remote Sensing. The point of
carrying it here rather than inventing a legend is that the table is not ours: a reader can check it
against the standard, and the gap between what the standard defines, what the delivery labelled, and
what the model has a word for is the whole project in one object.

Only the codes this project touches are listed. The ones left out are in the standard and not in this
data: rail, road surface, the guard wire, transmission tower, the wire to structure connector, the
noise classes, overhead structure, snow and the rest.

Read from the publisher's own source, ASPRS's repository github.com/ASPRSorg/LAS at the tag for
1.4 R15 (09 July 2019, commit 72a54a4), and from a PDF of R15 whose title page names the same commit
(page 30). R16, published by ASPRS on 25 August 2025, leaves every row of this table unchanged.

The source writes 13 and 14 with a double hyphen, which typesets as an en dash. No dash of that kind
appears on the page, so they are written with a plain hyphen, as the standard itself writes 16,
Wire-Structure Connector: Wire-Guard (Shield) and Wire-Conductor (Phase).
"""

from __future__ import annotations

from .data import CLASS_NAMES

# code: (the standard's own wording, what this project can say about it)
ASPRS_CLASSES = {
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    9: "Water",
    14: "Wire-Conductor (Phase)",
    17: "Bridge Deck",
}
SOURCE = (
    "Table 17, ASPRS Standard Point Classes (Point Data Record Formats 6-10), "
    "LAS Specification 1.4 R15, ASPRS, 09 July 2019"
)

# Which of this project's fifteen DALES classes, and M1's two extra, answer to a standard code. DALES
# calls every wire a powerline and never separates the guard wire (13) from the conductor (14); it is
# counted under 14, the wire a power line is for. A class with no code here is one the standard has no
# place for: vehicles, fences, and poles. The standard's nearest codes for a pole are for other things,
# 15 a transmission tower and 19 an overhead structure such as a traffic light, so a pole is related to
# them and not covered by them.
FINE_TO_ASPRS = {
    "ground": 2,
    "vegetation": 4,
    "tree": 5,
    "habitat": 6,
    "complex": 6,
    "annex": 6,
    "water": 9,
    "powerline": 14,
    "bridge_deck": 17,
}
# The model's groups that have no code, and the standard's codes that are near them without covering them.
RELATED_CODES = {"pole": (15, 19)}
# What the San Francisco delivery itself labels, which is most of why the project exists.
DELIVERY_LABELS = (1, 2, 9, 17)


def codes_for(taxonomy: list[str]) -> dict[int, list[str]]:
    """Standard code to the classes of this taxonomy that fall under it, in taxonomy order."""
    out: dict[int, list[str]] = {}
    for name in taxonomy:
        code = FINE_TO_ASPRS.get(name)
        if code is not None:
            out.setdefault(code, []).append(name)
    return out


def model_can_name(code: int, taxonomy: list[str] | None = None) -> bool:
    """Whether a model with this taxonomy has any word that answers to this standard code."""
    return code in codes_for(taxonomy or CLASS_NAMES)
