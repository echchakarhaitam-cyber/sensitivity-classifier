"""Prediction files: one JSON line per document, ``{"doc_id", "scores"}`` (6 floats or null).

Written by scripts/m2_evaluate.py (gitignored: they are large and derived), read back by the
shortcut analysis so that every analysis uses the exact scores that were evaluated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def write_predictions(path: Path, doc_ids: Sequence[str], S: np.ndarray) -> None:
    if len(doc_ids) != len(S):
        raise ValueError("one score row per document is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for doc_id, row in zip(doc_ids, S):
            scores = [None if math.isnan(v) else float(v) for v in row]
            out.write(json.dumps({"doc_id": doc_id, "scores": scores}) + "\n")


def read_predictions(path: Path, doc_ids: Sequence[str]) -> np.ndarray:
    """Scores in the order of ``doc_ids``; raises if the file covers other documents."""
    rows = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            rows[rec["doc_id"]] = [np.nan if v is None else v for v in rec["scores"]]
    if set(rows) != set(doc_ids):
        raise ValueError(f"{path}: predictions do not match the documents of the set")
    return np.array([rows[i] for i in doc_ids], dtype=float)
