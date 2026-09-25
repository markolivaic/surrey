"""The M1 taxonomy and its loss."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.data import CLASS_NAMES  # noqa: E402
from surrey.sources import (  # noqa: E402
    BRIDGE_DECK,
    BUILDING_SUPER,
    IGNORE,
    M1_CLASS_NAMES,
    SWISS_2024,
    WATER,
    map_swiss,
    superclass_nll,
)


def test_every_code_in_the_2024_table_has_a_decision_and_a_reason():
    assert set(SWISS_2024) == {1, 2, 3, 6, 9, 14, 15, 17, 26, 27}
    for code, (name, label, reason) in SWISS_2024.items():
        assert name and reason, code
        assert label in (IGNORE, BUILDING_SUPER) or 0 <= label < len(M1_CLASS_NAMES), code


def test_the_dales_classes_keep_their_ids_and_the_new_ones_follow():
    assert M1_CLASS_NAMES[:15] == CLASS_NAMES
    assert (WATER, BRIDGE_DECK) == (15, 16)


def test_vegetation_is_ignored_and_unknown_codes_are_counted_not_guessed():
    labels, unknown = map_swiss(np.array([3, 9, 17, 26, 6, 42, 42, 7]))
    assert labels.tolist() == [
        IGNORE,
        WATER,
        BRIDGE_DECK,
        BUILDING_SUPER,
        BUILDING_SUPER,
        IGNORE,
        IGNORE,
        IGNORE,
    ]
    assert unknown == {7: 1, 42: 2}


def test_the_superclass_loss_rewards_any_building_member_and_equals_ce_on_fine_labels():
    torch.manual_seed(0)
    logits = torch.randn(6, len(M1_CLASS_NAMES))
    lp = F.log_softmax(logits, dim=1)
    fine = torch.tensor([0, 5, WATER, BRIDGE_DECK, 1, IGNORE])
    assert torch.allclose(
        superclass_nll(lp, fine), F.cross_entropy(logits, fine, ignore_index=IGNORE)
    )
    members = [CLASS_NAMES.index(c) for c in ("habitat", "complex", "annex")]
    sup = torch.tensor([BUILDING_SUPER])
    expected = -torch.logsumexp(lp[0, members], dim=0)
    assert torch.allclose(superclass_nll(lp[:1], sup), expected)
    # Probability moved between building members does not change the superclass loss.
    shifted = logits[:1].clone()
    shifted[0, members[0]], shifted[0, members[1]] = logits[0, members[1]], logits[0, members[0]]
    assert torch.allclose(superclass_nll(F.log_softmax(shifted, 1), sup), expected)


def test_weights_match_torch_weighted_cross_entropy_on_fine_labels():
    torch.manual_seed(1)
    logits = torch.randn(8, len(M1_CLASS_NAMES))
    w = torch.rand(len(M1_CLASS_NAMES)) + 0.5
    lab = torch.tensor([0, 2, 3, WATER, BRIDGE_DECK, 0, 12, IGNORE])
    got = superclass_nll(F.log_softmax(logits, 1), lab, w)
    want = F.cross_entropy(logits, lab, weight=w, ignore_index=IGNORE)
    assert torch.allclose(got, want, atol=1e-6)
