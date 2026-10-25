"""Model selection and decision thresholds -- on the DEV split only.

Two operating points per category (docs/results.md):

* ``default``: 0.5 on probabilities, 0 on LinearSVC margins (0.5 on the detector's 0/1 scores);
* ``dev_f1``: the threshold maximising F1 on dev (:func:`sensclf.metrics.best_f1_threshold`).

Thresholds and hyper-parameters are never chosen on a test set or on the calibration split:
:func:`dev_thresholds` and :func:`select_by_dev` refuse any other set name and any
calibration document.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sensclf.guard import ensure_no_calibration
from sensclf.labels import CATEGORIES
from sensclf.metrics import average_precision, best_f1_threshold, is_defined

SELECTION_SET = "dev"


class SelectionSetError(ValueError):
    """Selection was attempted on a set other than dev."""


def _require_dev(set_name: str, doc_ids: Sequence[str], forbidden: frozenset[str]) -> None:
    if set_name != SELECTION_SET:
        raise SelectionSetError(f"selection must use {SELECTION_SET!r}, got {set_name!r}")
    ensure_no_calibration(doc_ids, forbidden)


def default_threshold(model: object) -> float:
    explicit = getattr(model, "default_threshold", None)
    if explicit is not None:
        return float(explicit)
    return 0.5 if model.probabilistic else 0.0


def per_category_ap(Y: np.ndarray, S: np.ndarray) -> dict[str, float | None]:
    """AP per category; None if the category is undefined or the model does not cover it."""
    out: dict[str, float | None] = {}
    for k, c in enumerate(CATEGORIES):
        covered = not np.isnan(S[:, k]).any()
        out[c] = average_precision(Y[:, k], S[:, k]) if covered and is_defined(Y[:, k]) else None
    return out


def macro_ap(Y: np.ndarray, S: np.ndarray) -> float | None:
    values = [v for v in per_category_ap(Y, S).values() if v is not None]
    return float(np.mean(values)) if values else None


def select_by_dev(candidates: Sequence[tuple[float, np.ndarray]], Y_dev: np.ndarray,
                  *, set_name: str, doc_ids: Sequence[str],
                  forbidden: frozenset[str]) -> tuple[float, list[dict]]:
    """Pick the hyper-parameter with the highest dev macro AP; ties go to the smaller value.

    ``candidates`` are (hyper-parameter, dev score matrix) pairs.
    """
    _require_dev(set_name, doc_ids, forbidden)
    rows = [{"C": c, "dev_macro_ap": macro_ap(Y_dev, S), "dev_ap": per_category_ap(Y_dev, S)}
            for c, S in candidates]
    best = max(rows, key=lambda r: (r["dev_macro_ap"], -r["C"]))
    return best["C"], rows


def operating_points(model: object, S_dev: np.ndarray, Y_dev: np.ndarray, *, set_name: str,
                     doc_ids: Sequence[str],
                     forbidden: frozenset[str]) -> dict[str, dict[str, float | None]]:
    """Both operating points of a model; None for categories it does not cover.

    A model with ``tunable_threshold = False`` (the 0/1 detector) keeps its own decision as
    its dev-F1 operating point.
    """
    _require_dev(set_name, doc_ids, forbidden)
    t0 = default_threshold(model)
    covered = {c: not np.isnan(S_dev[:, k]).any() for k, c in enumerate(CATEGORIES)}
    default = {c: (t0 if covered[c] else None) for c in CATEGORIES}
    if not getattr(model, "tunable_threshold", True):
        return {"default": default, "dev_f1": dict(default)}
    return {"default": default,
            "dev_f1": dev_thresholds(S_dev, Y_dev, set_name=set_name, doc_ids=doc_ids,
                                     forbidden=forbidden)}


def dev_thresholds(S_dev: np.ndarray, Y_dev: np.ndarray, *, set_name: str,
                   doc_ids: Sequence[str], forbidden: frozenset[str]) -> dict[str, float | None]:
    """Per category the F1-maximising threshold on dev (None if undefined or not covered)."""
    _require_dev(set_name, doc_ids, forbidden)
    out: dict[str, float | None] = {}
    for k, c in enumerate(CATEGORIES):
        covered = not np.isnan(S_dev[:, k]).any()
        out[c] = (best_f1_threshold(Y_dev[:, k], S_dev[:, k])
                  if covered and is_defined(Y_dev[:, k]) else None)
    return out
