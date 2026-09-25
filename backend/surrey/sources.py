"""The M1 taxonomy, and how each swissSURFACE3D code enters it.

M1 is M0 with two classes DALES-2 does not have, water and bridge deck, learned from
swissSURFACE3D tiles. Everything else about the model is M0's: three channels, the same
voxel, the same steps. The fifteen DALES-2 classes keep their ids; water and bridge
deck follow them.

swissSURFACE3D labels a building as one class, and DALES-2 splits buildings three ways.
A Swiss building point is not forced into one of the three: it carries the building
superclass label, and the loss rewards probability on any of habitat, complex and
annex (surrey.sources.superclass_nll). Codes with no counterpart are ignored, never
guessed.

Every code the 2024 swisstopo table lists has a row here with its decision and reason.
A code the table does not list is ignored and counted by the preparation script.
"""

from __future__ import annotations

import numpy as np

from .data import CLASS_NAMES

M1_CLASS_NAMES = CLASS_NAMES + ["water", "bridge_deck"]
WATER = M1_CLASS_NAMES.index("water")
BRIDGE_DECK = M1_CLASS_NAMES.index("bridge_deck")
IGNORE = -1
BUILDING_SUPER = 100
SUPERCLASSES = {BUILDING_SUPER: [CLASS_NAMES.index(c) for c in ("habitat", "complex", "annex")]}

SWISS_2024 = {
    1: (
        "unclassified and temporary objects",
        IGNORE,
        "mixes vehicles, cranes and anything unresolved; DALES splits vehicles four ways and "
        "has no class for the rest",
    ),
    2: ("ground", CLASS_NAMES.index("ground"), "the same class"),
    3: (
        "vegetation (low - medium - high)",
        IGNORE,
        "flown in February and March with no leaves; bare branches could undo the vegetation "
        "San Francisco just got back, so M1 learns vegetation from DALES alone",
    ),
    6: (
        "building roofs",
        BUILDING_SUPER,
        "one Swiss building class against three DALES ones; scored as the building superclass",
    ),
    9: ("water", WATER, "the class M1 exists to add"),
    14: (
        "high-voltage / power line",
        CLASS_NAMES.index("powerline"),
        "a conductor span, the object DALES labels powerline",
    ),
    15: (
        "masts with a load-bearing function",
        IGNORE,
        "lattice pylons and masts; DALES poles are utility, light and traffic poles, a "
        "different object, and no class fits",
    ),
    17: (
        "bridge-, overpass-, viaduct- and footbridge deck",
        BRIDGE_DECK,
        "the class M1 exists to add, and the only bridge part the San Francisco delivery labels",
    ),
    26: ("building facades", BUILDING_SUPER, "part of the building, as DALES labels walls"),
    27: (
        "bridge piers and viaduct stay cables",
        IGNORE,
        "the San Francisco reference labels decks only, piers are not deck, and stay cables "
        "look like powerline; teaching either as deck or wire would be wrong",
    ),
}


# Three recipes on the same data, so that each comparison moves one thing:
#
#   m0       DALES-2 alone, fifteen classes
#   m0prime  DALES-2 and the Swiss sub-tiles, fifteen classes; Swiss water and bridge
#            deck are ignored, so the model sees the same points M1 sees and has no word
#            for either
#   m1       the same data, with water and bridge deck as classes
#
# M1 against M0prime isolates the word; M0prime against M0 isolates the Swiss data and
# its density. DALES water is ignored in both recipes that add Swiss data.
RECIPES = ("m0", "m0prime", "m1")
M0PRIME_IGNORED = (WATER, BRIDGE_DECK)


def class_names_for(recipe: str) -> list[str]:
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}")
    return M1_CLASS_NAMES if recipe == "m1" else CLASS_NAMES


def recipe_ignore_mask(recipe: str, labels: np.ndarray) -> np.ndarray | None:
    """Points of a Swiss sub-tile that a recipe must not supervise, beyond the codes the
    table already ignores. None when nothing extra is ignored."""
    if recipe == "m0prime":
        return np.isin(labels, M0PRIME_IGNORED)
    if recipe in ("m0", "m1"):
        return None
    raise ValueError(f"unknown recipe {recipe!r}")


def map_swiss(codes: np.ndarray) -> tuple[np.ndarray, dict]:
    """int8 M1 labels for swissSURFACE3D codes, and counts of codes not in the table."""
    out = np.full(codes.shape, IGNORE, dtype=np.int8)
    for code, (_, label, _) in SWISS_2024.items():
        out[codes == code] = label
    unknown = {
        int(c): int(n)
        for c, n in zip(
            *np.unique(codes[~np.isin(codes, list(SWISS_2024))], return_counts=True), strict=True
        )
    }
    return out, unknown


def superclass_nll(log_probs, labels, weight=None):
    """Mean negative log likelihood over labelled points, where a superclass label counts
    the summed probability of its members.

    log_probs (N, C) log-softmax, labels (N,) long with IGNORE, fine ids, or a superclass
    id. With `weight` (C,), a fine label weighs its class and a superclass the mean of its
    members, the same units as torch's weighted cross entropy.
    """
    import torch

    keep = labels != IGNORE
    lp, lab = log_probs[keep], labels[keep]
    if lab.numel() == 0:
        return log_probs.sum() * 0.0
    # A label that is neither a class this head has nor a known superclass would add
    # weight and no loss. For M0prime that is Swiss water left unmasked.
    known = (lab < log_probs.shape[1]) | torch.isin(
        lab, torch.as_tensor(list(SUPERCLASSES), device=lab.device)
    )
    if not bool(known.all()):
        raise ValueError(
            f"labels outside the head and its superclasses: {sorted(set(lab[~known].tolist()))}"
        )
    loss = torch.zeros(lab.shape, dtype=lp.dtype, device=lp.device)
    w = torch.ones(lab.shape, dtype=lp.dtype, device=lp.device)
    fine = lab < log_probs.shape[1]
    loss[fine] = -lp[fine].gather(1, lab[fine].unsqueeze(1)).squeeze(1)
    if weight is not None:
        w[fine] = weight[lab[fine]]
    for sid, members in SUPERCLASSES.items():
        m = lab == sid
        if m.any():
            idx = torch.as_tensor(members, device=lp.device)
            loss[m] = -torch.logsumexp(lp[m][:, idx], dim=1)
            if weight is not None:
                w[m] = weight[idx].mean()
    return (loss * w).sum() / w.sum()
