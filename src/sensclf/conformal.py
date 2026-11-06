"""Split conformal prediction for each binary category, and a joint 6-label guarantee.

For one category, a calibrated probability p gives p(1) = p and p(0) = 1 - p. The
nonconformity of label y for a document is a(y) = 1 - p(y): small when the model finds y
plausible. On n held-out documents (the cal_test half, one document per group, see below) with
true labels y_i, let q be the ceil((n + 1)(1 - alpha))-th smallest a_i(y_i) (q = +inf if that
rank exceeds n). The prediction set of a new document is

    C(x) = { y in {0, 1} : a(y) <= q }.

If the new document and the held-out ones are exchangeable, P(y in C(x)) >= 1 - alpha (Vovk et
al., 2005; Angelopoulos & Bates, 2021). The set is {0}, {1} (a decision), {0, 1} (the model
cannot exclude either label: abstain) or empty (neither label is plausible enough).

* ``mondrian``: the same with one quantile per TRUE class, which gives coverage >= 1 - alpha
  separately for positive and for negative documents (class-conditional validity). It matters
  here because a missed sensitive document (y = 1) is the costly error.
* Joint guarantee for the 6-label vector: using alpha / 6 for each category, the union bound
  gives P(every true label is in its set) >= 1 - alpha.

Exchangeability. The two Nemotron renderings of a persona (us / intl) are not independent. The
quantile is therefore computed on ONE document per group of the held-out half, picked with a
fixed seed (``one_per_group``); the guarantee then holds over groups, under the assumption that
groups are exchangeable with new documents' groups. Under domain shift (test_shift, test_ood)
nothing is guaranteed; Milestone 5 measures the coverage that is actually reached.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

ALPHAS = (0.01, 0.05, 0.10)
SEED = 0


def nonconformity(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """a(y) = 1 - p(y) with p(1) = p, p(0) = 1 - p."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
    return np.where(y == 1, 1 - p, p)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score; +inf when that rank exceeds n."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        return math.inf
    rank = math.ceil((n + 1) * (1 - alpha))
    return math.inf if rank > n else float(scores[rank - 1])


def fit_quantiles(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float,
                  mondrian: bool = False) -> dict[int, float]:
    """Quantile per label value (the same value for both labels unless ``mondrian``)."""
    a = nonconformity(p_cal, y_cal)
    if not mondrian:
        q = conformal_quantile(a, alpha)
        return {0: q, 1: q}
    y_cal = np.asarray(y_cal, dtype=int)
    return {c: conformal_quantile(a[y_cal == c], alpha) for c in (0, 1)}


def prediction_sets(p: np.ndarray, q: dict[int, float]) -> np.ndarray:
    """(n, 2) boolean: column c says whether label c is in the set."""
    p = np.asarray(p, dtype=float)
    return np.column_stack([nonconformity(p, np.full(len(p), c)) <= q[c] for c in (0, 1)])


def set_statistics(sets: np.ndarray, y: np.ndarray) -> dict:
    """Coverage (overall and per true class) and the distribution of set sizes."""
    y = np.asarray(y, dtype=int)
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(axis=1)
    out = {"n": len(y), "coverage": float(covered.mean()),
           "singleton": float((size == 1).mean()), "both": float((size == 2).mean()),
           "empty": float((size == 0).mean())}
    for c in (0, 1):
        m = y == c
        out[f"coverage_y{c}"] = float(covered[m].mean()) if m.any() else None
        out[f"n_y{c}"] = int(m.sum())
    singles = size == 1
    out["singleton_error"] = (float((~covered[singles]).mean()) if singles.any() else None)
    return out


def one_per_group(groups: Sequence[str], seed: int = SEED) -> np.ndarray:
    """Indices of one document per group, chosen with a fixed seed (sorted)."""
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    picks = [members[int(rng.integers(len(members)))]
             for _, members in sorted(by_group.items())]
    return np.array(sorted(picks), dtype=int)


def coverage_bounds(n: int, alpha: float) -> tuple[float, float]:
    """Theoretical marginal coverage range for split conformal with n calibration points.

    Lower bound 1 - alpha; upper bound 1 - alpha + 1 / (n + 1) when scores have no ties.
    """
    return 1 - alpha, min(1.0, 1 - alpha + 1 / (n + 1))
