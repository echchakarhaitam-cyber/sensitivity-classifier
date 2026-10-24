"""Shared pieces of every model: label matrices, the score interface, save/load.

A model maps documents to an (n, 6) score matrix in :data:`~sensclf.labels.CATEGORIES`
order. Probability models return P(category | document); margin models (LinearSVC) return
decision values; a column of NaN means the model does not cover that category.

Models are stored with joblib (``models/``, gitignored) next to a JSON sidecar describing
how they were made, so a score can always be traced to a training set and to versions.
joblib files are pickles: only load files this project wrote.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from sensclf.labels import CATEGORIES
from sensclf.runinfo import versions
from sensclf.splits import LabelledDocument


class ScoreModel(Protocol):
    name: str
    probabilistic: bool

    def predict_scores(self, docs: Sequence[LabelledDocument]) -> np.ndarray: ...


def label_matrix(docs: Sequence[LabelledDocument]) -> np.ndarray:
    """(n, 6) int array of gold labels, columns in CATEGORIES order."""
    return np.array([[d.labels[c] for c in CATEGORIES] for d in docs], dtype=int).reshape(
        len(docs), len(CATEGORIES))


def doc_ids_sha256(docs: Sequence[LabelledDocument]) -> str:
    """Fingerprint of a training set: SHA-256 of its sorted doc_ids, one per line."""
    return hashlib.sha256("\n".join(sorted(d.doc_id for d in docs)).encode("utf-8")).hexdigest()


def save_model(model: object, path: Path, info: dict) -> Path:
    """``path`` (.joblib) plus ``path.with_suffix('.json')`` holding ``info`` and versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    sidecar = {**info, "versions": versions(), "file": path.name}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8", newline="\n")
    return path


def load_model(path: Path) -> object:
    return joblib.load(Path(path))


def load_sidecar(path: Path) -> dict:
    return json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
