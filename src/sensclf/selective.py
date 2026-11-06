"""Selective prediction: decide automatically when confident, abstain (send to a human) otherwise.

For one category, a calibrated probability p gives the decision yhat = 1[p >= 0.5] and the
confidence c = max(p, 1 - p). Accepting only documents with c >= lambda trades COVERAGE (the
share of documents decided automatically) for SELECTIVE RISK (the error rate among them).

* ``risk_coverage``  the whole trade-off curve, sorted by confidence (ties kept together);
* ``aurc``           area under the risk-coverage curve (lower is better; Geifman &
                     El-Yaniv, 2017), with the oracle value of a perfect ranking for reference;
* ``choose_threshold`` a threshold whose selective risk is below a target r* with probability
                     at least 1 - delta, chosen on held-out data (the cal_test half).

Threshold choice. The candidates are the confidence levels reached at a FIXED grid of coverages
(5%, 10%, ..., 100%; K = 20 candidates), fixed before seeing any data. For each candidate the
accepted documents are n, of which k are errors; H0 "selective risk > r*" is rejected when the
exact binomial tail P(K <= k | n, r*) <= delta / K (Bonferroni). Among the rejected candidates
the loosest (largest coverage) is returned; None if none is rejected. By the union bound, the
probability of returning a threshold whose true selective risk exceeds r* is at most delta --
the multiple-testing view of risk control (Learn-then-Test, Angelopoulos et al., 2021) applied
to selective risk as in Geifman & El-Yaniv (2017). It assumes the held-out documents and the
target documents are exchangeable. Under domain shift that assumption fails, and Milestone 4
measures what happens then. (A first version scanned candidates from the strictest one with
fixed-sequence testing; it always stopped at once, because a handful of accepted documents can
never certify a 1% risk. The unit test caught it.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

DELTA = 0.05
RISK_TARGETS = (0.01, 0.02, 0.05)
COVERAGE_POINTS = (0.8, 0.9, 0.95)
COVERAGE_GRID = tuple(round(0.05 * i, 2) for i in range(1, 21))


def decisions(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(yhat, confidence) from calibrated probabilities."""
    p = np.asarray(p, dtype=float)
    return (p >= 0.5).astype(int), np.maximum(p, 1 - p)


def errors(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    yhat, _ = decisions(p)
    return (yhat != np.asarray(y, dtype=int)).astype(int)


@dataclass(frozen=True)
class Curve:
    coverage: np.ndarray       # accepted share at each distinct confidence level
    risk: np.ndarray           # selective risk at that level
    threshold: np.ndarray      # the confidence level (accept c >= threshold)


def risk_coverage(conf: np.ndarray, err: np.ndarray) -> Curve:
    """Risk-coverage curve; documents with equal confidence are accepted together."""
    conf, err = np.asarray(conf, dtype=float), np.asarray(err, dtype=float)
    order = np.argsort(-conf, kind="stable")
    c, e = conf[order], err[order]
    cum_e = np.cumsum(e)
    last = np.r_[np.nonzero(np.diff(c))[0], len(c) - 1]     # end index of each tie block
    n_acc = last + 1
    return Curve(coverage=n_acc / len(c), risk=cum_e[last] / n_acc, threshold=c[last])


def aurc(conf: np.ndarray, err: np.ndarray) -> float:
    """Area under the risk-coverage curve, as the mean selective risk over all prefixes.

    Ties are resolved by the tie-aware curve: every document of a tie block gets the risk of
    the whole block, so a constant confidence gives AURC = overall error rate.
    """
    curve = risk_coverage(conf, err)
    counts = np.diff(np.r_[0, np.round(curve.coverage * len(conf)).astype(int)])
    return float(np.sum(curve.risk * counts) / len(conf))


def oracle_aurc(err: np.ndarray) -> float:
    """AURC of a perfect confidence ranking (all errors last): the best achievable value."""
    err = np.sort(np.asarray(err, dtype=float))
    n = len(err)
    i = np.arange(1, n + 1)
    return float(np.mean(np.cumsum(err) / i))


def risk_at_coverage(conf: np.ndarray, err: np.ndarray, target: float) -> dict:
    """Selective risk when accepting (at least) the ``target`` share of documents."""
    curve = risk_coverage(conf, err)
    j = int(np.searchsorted(curve.coverage, target - 1e-12))
    j = min(j, len(curve.coverage) - 1)
    return {"coverage": float(curve.coverage[j]), "risk": float(curve.risk[j]),
            "threshold": float(curve.threshold[j])}


def binomial_pvalue(k: int, n: int, r: float) -> float:
    """P(K <= k) for K ~ Binomial(n, r): small when k errors are unlikely if risk were r."""
    return float(binom.cdf(k, n, r)) if n > 0 else 1.0


def choose_threshold(conf: np.ndarray, err: np.ndarray, target: float,
                     delta: float = DELTA, grid: tuple[float, ...] = COVERAGE_GRID) -> dict:
    """Loosest confidence threshold certified to have selective risk <= target (see module)."""
    conf, err = np.asarray(conf, dtype=float), np.asarray(err, dtype=int)
    level = delta / len(grid)
    chosen = None
    for cov_target in grid:
        point = risk_at_coverage(conf, err, cov_target)
        acc = conf >= point["threshold"]
        n, k = int(acc.sum()), int(err[acc].sum())
        certified = binomial_pvalue(k, n, target) <= level
        if certified and (chosen is None or n > chosen["cal_accepted"]):
            chosen = {"threshold": point["threshold"], "cal_coverage": n / len(conf),
                          "cal_risk": k / n, "cal_accepted": n, "cal_errors": k}
    return {"target": target, "delta": delta, "candidates": len(grid), "test_level": level,
            "certified": chosen is not None, **(chosen or {"threshold": None})}


def apply_threshold(conf: np.ndarray, err: np.ndarray, threshold: float | None) -> dict:
    """Coverage and selective risk on new documents; nothing accepted when threshold is None."""
    conf, err = np.asarray(conf, dtype=float), np.asarray(err, dtype=float)
    if threshold is None:
        return {"coverage": 0.0, "risk": None, "accepted": 0}
    acc = conf >= threshold
    n = int(acc.sum())
    return {"coverage": float(acc.mean()), "risk": float(err[acc].mean()) if n else None,
            "accepted": n}
