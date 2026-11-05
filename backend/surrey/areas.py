"""The eight San Francisco areas, each named once, with the label a reader sees.

Ordered from parkland through the built areas to open water, which is the order the README's tables
use. The areas with building footprints are the six where buildings can be scored.
"""

from __future__ import annotations

AREAS = {
    "ggpark": "Golden Gate Park",
    "sunset": "Sunset",
    "mission": "Mission",
    "russianhill": "Russian Hill",
    "soma": "SoMa",
    "financial": "Financial District",
    "embarcadero": "Embarcadero",
    "baybridge": "Bay Bridge",
}
SF_AREAS = tuple(AREAS)
WITH_FOOTPRINTS = ("sunset", "mission", "russianhill", "soma", "financial", "embarcadero")
