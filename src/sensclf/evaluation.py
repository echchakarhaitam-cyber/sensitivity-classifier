"""Evaluate score matrices on one set: point metrics, cluster-bootstrap CIs, paired differences.

Inputs are (n, 6) score matrices (NaN column = category not covered), the gold (n, 6) label
matrix and one group id per document. Operating points are per category thresholds chosen
beforehand on dev (:mod:`sensclf.selection`). Macro averages run over the categories that are
covered by the model AND defined in the set (at least one positive and one negative); the
excluded categories are listed next to every macro value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

from sensclf.bootstrap import cluster_bootstrap, interval
from sensclf.labels import CATEGORIES
from sensclf.metrics import (
    APScorer,
    category_metrics,
    confusion,
    f1_from_counts,
    is_defined,
    vector_metrics,
)

OperatingPoints = Mapping[str, Mapping[str, float | None]]   # op name -> category -> threshold
CI_OP = "dev_f1"   # F1 intervals are computed at the dev-F1 operating point


def evaluable(S: np.ndarray, Y: np.ndarray) -> list[int]:
    """Column indices that the model covers and that are defined in this set."""
    return [k for k in range(len(CATEGORIES))
            if not np.isnan(S[:, k]).any() and is_defined(Y[:, k])]


def point_metrics(S: np.ndarray, Y: np.ndarray, ops: OperatingPoints,
                  probabilistic: bool) -> dict:
    cols = evaluable(S, Y)
    per_category = {}
    for k, c in enumerate(CATEGORIES):
        if np.isnan(S[:, k]).any():
            per_category[c] = {"covered": False}
            continue
        thresholds = {op: t[c] for op, t in ops.items() if t[c] is not None}
        per_category[c] = {"covered": True,
                           **category_metrics(Y[:, k], S[:, k], thresholds, probabilistic)}
    macro: dict = {"categories": [CATEGORIES[k] for k in cols],
                   "excluded": [c for k, c in enumerate(CATEGORIES) if k not in cols]}
    for key in ("average_precision", "roc_auc", "brier", "log_loss"):
        vals = [per_category[CATEGORIES[k]][key] for k in cols]
        macro[key] = float(np.mean(vals)) if vals and None not in vals else None
    vector = {}
    for op, t in ops.items():
        usable = [k for k in cols if t[CATEGORIES[k]] is not None]
        P = np.zeros_like(Y)
        for k in usable:
            P[:, k] = S[:, k] >= t[CATEGORIES[k]]
        vector[op] = vector_metrics(Y, P, usable)
        vector[op]["categories"] = [CATEGORIES[k] for k in usable]
    return {"per_category": per_category, "macro": macro, "vector": vector}


def _statistics(name: str, S: np.ndarray, Y: np.ndarray, t: Mapping[str, float | None]) -> dict:
    """Bootstrap statistics of one model: per-category AP and F1, macro AP/F1, micro F1."""
    cols = evaluable(S, Y)
    stats = {}
    ap = {k: APScorer(Y[:, k], S[:, k]) for k in cols}
    preds = {k: (S[:, k] >= t[CATEGORIES[k]]).astype(float) for k in cols
             if t[CATEGORIES[k]] is not None}
    for k in cols:
        stats[f"{name}|ap|{CATEGORIES[k]}"] = ap[k]
    for k, p in preds.items():
        stats[f"{name}|f1|{CATEGORIES[k]}"] = (
            lambda W, y=Y[:, k], p=p: f1_from_counts(*confusion(y, p, W)))
    if cols:
        stats[f"{name}|macro_ap"] = lambda W: np.mean([ap[k](W) for k in cols], axis=0)
    if preds:
        def counts(W):
            return [confusion(Y[:, k], p, W) for k, p in preds.items()]

        stats[f"{name}|macro_f1"] = lambda W: np.mean([f1_from_counts(*c) for c in counts(W)],
                                                      axis=0)
        stats[f"{name}|micro_f1"] = lambda W: f1_from_counts(
            *(np.sum([c[i] for c in counts(W)], axis=0) for i in range(3)))
    return stats


def bootstrap_metrics(models: Mapping[str, tuple[np.ndarray, Mapping[str, float | None]]],
                      Y: np.ndarray, groups: Sequence[str],
                      paired: Sequence[str] = (), n_resamples: int = 1000,
                      seed: int = 0) -> dict:
    """95% cluster-bootstrap CIs per model and paired macro-AP differences.

    ``models`` maps a name to (scores, dev-F1 thresholds). ``paired`` names the models whose
    macro AP is compared pairwise; a pair whose evaluable categories differ is skipped and
    reported as such. Differences are A - B on identical resamples.
    """
    stats: dict = {}
    for name, (S, t) in models.items():
        stats.update(_statistics(name, S, Y, t))
    draws = cluster_bootstrap(groups, stats, n_resamples=n_resamples, seed=seed)
    ones = np.ones((1, len(Y)))
    cis: dict = {name: {} for name in models}
    for key, values in draws.items():
        name, rest = key.split("|", 1)
        cis[name][rest] = interval(values)
    pairs = {}
    for a, b in combinations(paired, 2):
        ka, kb = f"{a}|macro_ap", f"{b}|macro_ap"
        same = evaluable(models[a][0], Y) == evaluable(models[b][0], Y)
        if not same or ka not in stats or kb not in stats:
            pairs[f"{a} - {b}"] = {"skipped": "different evaluable categories"}
            continue
        point = float(stats[ka](ones)[0] - stats[kb](ones)[0])
        diff = draws[ka] - draws[kb]
        pairs[f"{a} - {b}"] = {"model_a": a, "model_b": b, "point": point,
                               "ci": interval(diff), "share_a_better": float(np.mean(diff > 0))}
    return {"ci": cis, "paired_macro_ap": pairs, "n_resamples": n_resamples, "seed": seed,
            "level": 0.95, "groups": len(set(groups)), "ci_operating_point": CI_OP}
