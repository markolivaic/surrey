"""Per-class scoring.

Per class first. Four of the fifteen DALES-2 classes hold under half a percent of
the points, several are absent from whole tiles, and a fifteen class mean moves when
those four move, so a mean is reported beside the per-class figures, never instead of
them. The question is which classes survive a change of city, and that has per-class
answers.

A class with no support returns None rather than zero. Zero is a measurement and
None is an absence, and reporting an absence as a zero is how a model that never
saw a traffic pole ends up looking like a model that got every traffic pole wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import CLASS_NAMES, COARSE, N_CLASSES


class Confusion:
    """Truth by prediction counts, accumulated over any number of batches."""

    def __init__(self, names: list[str] | None = None):
        self.names = list(names or CLASS_NAMES)
        self.m = np.zeros((len(self.names), len(self.names)), dtype=np.int64)

    @property
    def n(self) -> int:
        return len(self.names)

    def add(self, truth: np.ndarray, pred: np.ndarray) -> None:
        truth = np.asarray(truth)
        pred = np.asarray(pred)
        if truth.shape != pred.shape:
            raise ValueError(f"shape mismatch {truth.shape} vs {pred.shape}")
        ok = (truth >= 0) & (truth < self.n) & (pred >= 0) & (pred < self.n)
        if ok.any():
            np.add.at(self.m, (truth[ok], pred[ok]), 1)

    def support(self) -> dict[str, int]:
        return {n: int(self.m[i, :].sum()) for i, n in enumerate(self.names)}

    def iou(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for i, name in enumerate(self.names):
            tp = int(self.m[i, i])
            union = int(self.m[i, :].sum() + self.m[:, i].sum() - tp)
            if int(self.m[i, :].sum()) == 0:
                out[name] = None  # never present, so nothing was measured
            elif union == 0:
                out[name] = 0.0
            else:
                out[name] = round(tp / union, 4)
        return out

    def precision(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for i, name in enumerate(self.names):
            predicted = int(self.m[:, i].sum())
            out[name] = None if predicted == 0 else round(int(self.m[i, i]) / predicted, 4)
        return out

    def recall(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for i, name in enumerate(self.names):
            actual = int(self.m[i, :].sum())
            out[name] = None if actual == 0 else round(int(self.m[i, i]) / actual, 4)
        return out

    def accuracy(self) -> float:
        total = int(self.m.sum())
        return float(np.trace(self.m) / total) if total else float("nan")

    def coarse(self) -> Confusion:
        """Regrouped into the classes a target delivery could plausibly tell apart."""
        groups = list(COARSE)
        index = {name: g for g, members in COARSE.items() for name in members}
        out = Confusion(groups)
        for i, ni in enumerate(self.names):
            for j, nj in enumerate(self.names):
                if not self.m[i, j]:
                    continue
                out.m[groups.index(index[ni]), groups.index(index[nj])] += self.m[i, j]
        return out


def coarse_lookup() -> np.ndarray:
    """Map a fine class id to its coarse group id, for use on whole arrays."""
    groups = list(COARSE)
    table = np.full(N_CLASSES, -1, dtype=np.int8)
    for g, members in COARSE.items():
        for name in members:
            table[CLASS_NAMES.index(name)] = groups.index(g)
    if (table < 0).any():
        missing = [CLASS_NAMES[i] for i in np.flatnonzero(table < 0)]
        raise ValueError(f"classes missing from the coarse grouping: {missing}")
    return table


# --------------------------------------------------------------------------
# Confidence as a detector of the model's own error
# --------------------------------------------------------------------------

SCORE_BINS = 20000


def score_histogram(scores: np.ndarray, bins: int = SCORE_BINS) -> np.ndarray:
    """Counts of scores in [0, 1] on a fine fixed grid.

    Histograms rather than raw arrays so a full held-out pass of about 108 million
    points can be pooled tile by tile without holding every probability at once.
    At twenty thousand bins the AUROC error from binning is far below anything
    this project reports.
    """
    idx = np.clip((np.asarray(scores, dtype=np.float64) * bins).astype(np.int64), 0, bins - 1)
    return np.bincount(idx, minlength=bins).astype(np.int64)


def auroc_from_histograms(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Probability that a random positive scores above a random negative.

    Ties within a bin count half, which is the standard convention. Returns None
    when either side is empty, because an AUROC over one class is not a number.
    """
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    n_pos, n_neg = positive.sum(), negative.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    below = np.concatenate([[0.0], np.cumsum(negative)[:-1]])
    return float((positive * (below + 0.5 * negative)).sum() / (n_pos * n_neg))


def auroc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """The same AUROC for raw values: each distinct value is its own bin, so it is exact."""
    positive, negative = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    if positive.size == 0 or negative.size == 0:
        return None
    values, inv = np.unique(np.concatenate([positive, negative]), return_inverse=True)
    return auroc_from_histograms(
        np.bincount(inv[: positive.size], minlength=values.size),
        np.bincount(inv[positive.size :], minlength=values.size),
    )


def auroc_row(conf, right, wrong, h=None) -> dict:
    """Confidence as a detector of the model's own error, with what the errors physically are."""
    r, w = conf[right], conf[wrong]
    a = auroc_from_histograms(score_histogram(r), score_histogram(w)) if r.size and w.size else None
    row: dict[str, Any] = {
        "right": int(r.size),
        "wrong": int(w.size),
        "auroc": None if a is None else round(a, 4),
        "median_conf_right": round(float(np.median(r)), 4) if r.size else None,
        "median_conf_wrong": round(float(np.median(w)), 4) if w.size else None,
    }
    if h is not None and w.size:
        # What the wrong calls physically are: a roof sits metres up, a kerb does not.
        row["wrong_height_m_p10_p50_p90"] = [
            round(float(v), 1) for v in np.nanpercentile(h[wrong], [10, 50, 90])
        ]
    if h is not None and r.size:
        # The surface check: right calls must sit on it.
        row["right_height_m_p50_p99"] = [
            round(float(v), 2) for v in np.nanpercentile(h[right], [50, 99])
        ]
    return row


def coarse_shares(
    coarse: np.ndarray, groups: list[str] | None = None, min_points: int = 1000
) -> dict | None:
    """Share of points in each coarse group, or None below `min_points`, where a share says little."""
    n = coarse.size
    if n < max(min_points, 1):
        return None
    return {
        g: round(float((coarse == i).sum() / n), 4) for i, g in enumerate(groups or list(COARSE))
    }
