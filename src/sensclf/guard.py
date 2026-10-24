"""Guard: the calibration split must not touch training or model selection before Milestone 3.

The calibration split is reserved for calibration and conformal prediction. A single
calibration document used to fit a model or to choose a hyper-parameter or threshold would
invalidate those later guarantees, so every fit/selection entry point calls
:func:`ensure_no_calibration` and fails loudly instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache


class CalibrationLeakError(RuntimeError):
    """A calibration-split document reached training or selection."""


@lru_cache(maxsize=1)
def calibration_ids() -> frozenset[str]:
    """doc_ids of the verified calibration split (hash-checked against the manifest)."""
    from sensclf.processed import load_set

    return frozenset(d.doc_id for d in load_set("calibration"))


def ensure_no_calibration(doc_ids: Iterable[str], forbidden: frozenset[str]) -> None:
    """Raise :class:`CalibrationLeakError` if any of ``doc_ids`` is in ``forbidden``."""
    leaked = [i for i in doc_ids if i in forbidden]
    if leaked:
        raise CalibrationLeakError(
            f"{len(leaked)} calibration documents passed to training/selection, e.g. {leaked[0]}"
        )
