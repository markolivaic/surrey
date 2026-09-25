"""AUROC from binned scores, checked against cases with known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.metrics import auroc_from_histograms, score_histogram  # noqa: E402


def exact_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank based AUROC with ties counted half, the definition the binned one approximates."""
    scores = np.concatenate([pos, neg])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks over ties
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def test_identical_distributions_score_one_half():
    rng = np.random.default_rng(0)
    s = rng.random(50000)
    a = auroc_from_histograms(score_histogram(s[:25000]), score_histogram(s[25000:]))
    assert a == pytest.approx(0.5, abs=0.01)


def test_perfect_separation_scores_one_and_reversed_scores_zero():
    pos = np.full(1000, 0.95)
    neg = np.full(1000, 0.10)
    assert auroc_from_histograms(score_histogram(pos), score_histogram(neg)) == pytest.approx(1.0)
    assert auroc_from_histograms(score_histogram(neg), score_histogram(pos)) == pytest.approx(0.0)


def test_binned_matches_the_exact_rank_definition():
    rng = np.random.default_rng(1)
    pos = np.clip(rng.normal(0.8, 0.1, 20000), 0, 1)
    neg = np.clip(rng.normal(0.6, 0.15, 8000), 0, 1)
    binned = auroc_from_histograms(score_histogram(pos), score_histogram(neg))
    assert binned == pytest.approx(exact_auroc(pos, neg), abs=2e-4)


def test_pooling_histograms_equals_scoring_the_concatenation():
    # The full pass pools tile by tile. Adding histograms has to give exactly
    # what one histogram over all the points would.
    rng = np.random.default_rng(2)
    p1, p2 = rng.random(3000), rng.random(5000) * 0.5 + 0.5
    n1, n2 = rng.random(4000) * 0.7, rng.random(2000)
    pooled = auroc_from_histograms(
        score_histogram(p1) + score_histogram(p2), score_histogram(n1) + score_histogram(n2)
    )
    whole = auroc_from_histograms(score_histogram(np.r_[p1, p2]), score_histogram(np.r_[n1, n2]))
    assert pooled == pytest.approx(whole, abs=1e-12)


def test_an_empty_side_is_none_not_a_number():
    h = score_histogram(np.array([0.5, 0.6]))
    assert auroc_from_histograms(h, np.zeros_like(h)) is None
    assert auroc_from_histograms(np.zeros_like(h), h) is None


def test_scores_at_the_edges_land_in_range():
    h = score_histogram(np.array([0.0, 1.0, 1.0000001, -1e-9]))
    assert h.sum() == 4
