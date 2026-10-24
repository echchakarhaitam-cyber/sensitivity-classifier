"""Document-level metrics for the 6 binary categories and the 6-label vector.

Ranking metrics (average precision, ROC-AUC) use scores; F1 / precision / recall use a
threshold (``score >= threshold`` is positive); Brier score and log loss only make sense for
probabilities. A category with no positive or no negative document in a set is *undefined*
there: all its metrics are ``None`` and it is excluded from macro averages.

Average precision and F1 are implemented for **weighted** documents and for many weight
vectors at once (rows of ``W``), because a bootstrap resample is exactly a vector of
multiplicities. With unit weights they equal scikit-learn's ``average_precision_score`` and
``f1_score`` (tests check this, ties included).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-15  # log-loss clipping, as in scikit-learn


def is_defined(y: np.ndarray) -> bool:
    """A category is evaluable only with at least one positive and one negative document."""
    y = np.asarray(y)
    return bool(y.size) and 0 < int(y.sum()) < y.size


class APScorer:
    """Average precision of fixed scores under many weightings (sort once, reuse).

    AP = sum over distinct thresholds k (descending) of (TP_k - TP_{k-1}) / TP_total * P_k,
    with P_k = TP_k / (TP_k + FP_k): scikit-learn's step-wise definition, where tied scores
    form one threshold.
    """

    def __init__(self, y: np.ndarray, scores: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=float)
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        self.order = order
        self.y_sorted = np.asarray(y, dtype=float)[order]
        # last index of each block of tied scores
        self.block_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])

    def __call__(self, W: np.ndarray) -> np.ndarray:
        """AP for each row of ``W`` (shape (R, n) or (n,)); NaN where no positive weight."""
        W = np.atleast_2d(np.asarray(W, dtype=float))[:, self.order]
        tp = np.cumsum(W * self.y_sorted, axis=1)[:, self.block_ends]
        seen = np.cumsum(W, axis=1)[:, self.block_ends]
        precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        delta_tp = np.diff(tp, axis=1, prepend=0.0)
        total = tp[:, -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, (delta_tp * precision).sum(axis=1) / total, np.nan)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    return float(APScorer(y, scores)(np.ones(len(y)))[0])


def confusion(y: np.ndarray, pred: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, ...]:
    """Weighted (TP, FP, FN) per row of ``W``."""
    W = np.atleast_2d(np.asarray(W, dtype=float))
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return W @ (y * pred), W @ ((1 - y) * pred), W @ (y * (1 - pred))


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """F1 = 2TP / (2TP + FP + FN); 0 when TP = 0 (zero_division=0)."""
    tp, fp, fn = (np.asarray(v, dtype=float) for v in (tp, fp, fn))
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(tp), where=(tp > 0) & (den > 0))


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def category_metrics(y: np.ndarray, scores: np.ndarray, thresholds: dict[str, float],
                     probabilistic: bool) -> dict:
    """All per-category metrics of one model on one set; ``None`` everywhere if undefined."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    out: dict = {"positives": int(y.sum()), "documents": int(y.size), "defined": is_defined(y)}
    if not out["defined"]:
        out.update(roc_auc=None, average_precision=None, brier=None, log_loss=None,
                   at={name: None for name in thresholds})
        return out
    out["roc_auc"] = float(roc_auc_score(y, scores))
    out["average_precision"] = average_precision(y, scores)
    if probabilistic:
        p = np.clip(scores, EPS, 1 - EPS)
        out["brier"] = float(np.mean((scores - y) ** 2))
        out["log_loss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    else:
        out["brier"] = out["log_loss"] = None
    out["at"] = {}
    for name, t in thresholds.items():
        pred = (scores >= t).astype(int)
        tp, fp, fn = (float(v[0]) for v in confusion(y, pred, np.ones(y.size)))
        out["at"][name] = {"threshold": float(t), "precision": _ratio(tp, tp + fp),
                           "recall": _ratio(tp, tp + fn),
                           "f1": float(f1_from_counts(tp, fp, fn))}
    return out


def vector_metrics(Y: np.ndarray, P: np.ndarray, columns: Sequence[int]) -> dict:
    """Micro/macro F1, subset accuracy and Hamming loss over the label columns ``columns``.

    ``Y`` and ``P`` are (n, 6) 0/1 arrays; only the given columns (the categories that are
    defined in this set and covered by the model) enter, so all four numbers describe the
    same label vector.
    """
    if not columns:
        return {"micro_f1": None, "macro_f1": None, "subset_accuracy": None,
                "hamming_loss": None, "categories": []}
    Y = np.asarray(Y, dtype=int)[:, list(columns)]
    P = np.asarray(P, dtype=int)[:, list(columns)]
    tp = (Y & P).sum(axis=0)
    fp = ((1 - Y) & P).sum(axis=0)
    fn = (Y & (1 - P)).sum(axis=0)
    return {
        "micro_f1": float(f1_from_counts(tp.sum(), fp.sum(), fn.sum())),
        "macro_f1": float(np.mean(f1_from_counts(tp, fp, fn))),
        "subset_accuracy": float(np.mean((Y == P).all(axis=1))),
        "hamming_loss": float(np.mean(Y != P)),
        "categories": list(columns),
    }


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """The score threshold (``score >= t`` positive) with the highest F1 on these documents.

    Candidates are the distinct scores. Ties in F1 go to the highest threshold (the most
    conservative one). Callers must only pass DEV data (see sensclf.selection).
    """
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if not is_defined(y):
        raise ValueError("cannot choose a threshold for a category with one class only")
    order = np.argsort(-scores, kind="mergesort")
    s, ys = scores[order], y[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    tp = np.cumsum(ys)[ends]
    f1 = 2 * tp / (ends + 1.0 + ys.sum())
    return float(s[ends[int(np.argmax(f1))]])
