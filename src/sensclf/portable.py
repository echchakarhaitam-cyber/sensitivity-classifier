"""A NumPy-only copy of the selected model, for the in-browser demo.

The selected model (Milestone 3: ``word_svc`` + isotonic calibration) is a scikit-learn TF-IDF
vectorizer, six LinearSVC heads and six isotonic maps. scikit-learn pickles cannot be loaded
safely across versions, so ``scripts/export_demo_model.py`` writes the learned numbers
(vocabulary, idf, weights, isotonic breakpoints) to ``demo/model.npz`` and this module recomputes
the same scores with NumPy only. Tests check the result against scikit-learn.

On top of the probabilities the demo shows the two uncertainty tools of the study, with the
thresholds and quantiles taken unchanged from the committed results:

* selective prediction (Milestone 4): for a target risk r*, a category's decision is given only
  if its confidence max(p, 1 - p) reaches the threshold certified on the calibration data;
  otherwise the model abstains ("not sure"). Not certified -> always abstain.
* conformal prediction (Milestone 5): the set of labels {no, yes} whose nonconformity is within
  the calibrated quantile; a two-label set means "could be either".
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

CATEGORIES = ("IDENTITY", "CONTACT", "FINANCIAL", "HEALTH", "CREDENTIALS", "SPECIAL_CATEGORY")
TOKEN = re.compile(r"(?u)\b\w\w+\b")   # scikit-learn's default token_pattern
MAX_CHARS = 20_000


def analyze(text: str) -> list[str]:
    """Lowercase, tokenize and add word bigrams, as TfidfVectorizer(ngram_range=(1, 2))."""
    tokens = TOKEN.findall(text.lower())
    return tokens + [f"{a} {b}" for a, b in pairwise(tokens)]


@dataclass
class PortableModel:
    vocabulary: dict[str, int]
    idf: np.ndarray            # (V,)
    coef: np.ndarray           # (6, V)
    intercept: np.ndarray      # (6,)
    iso_x: list[np.ndarray]    # isotonic breakpoints per category
    iso_y: list[np.ndarray]

    @classmethod
    def load(cls, path: Path) -> PortableModel:
        with np.load(Path(path)) as z:
            terms = bytes(z["terms"]).decode("utf-8").split("\n")
            return cls(vocabulary={t: i for i, t in enumerate(terms)}, idf=z["idf"],
                       coef=z["coef"], intercept=z["intercept"],
                       iso_x=[z[f"iso_x_{k}"] for k in range(len(CATEGORIES))],
                       iso_y=[z[f"iso_y_{k}"] for k in range(len(CATEGORIES))])

    def features(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Sparse TF-IDF row: (column indices, values), sublinear tf and l2 norm."""
        counts = Counter(i for i in (self.vocabulary.get(t) for t in analyze(text))
                         if i is not None)
        if not counts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        tf = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
        val = (np.log(tf) + np.float32(1.0)) * self.idf[idx]
        norm = np.sqrt(np.sum(val * val))
        return idx, (val / norm if norm > 0 else val)

    def margins(self, text: str) -> np.ndarray:
        idx, val = self.features(text)
        return self.coef[:, idx].astype(np.float64) @ val.astype(np.float64) + self.intercept

    def probabilities(self, text: str) -> np.ndarray:
        m = self.margins(text)
        return np.array([float(np.interp(m[k], self.iso_x[k], self.iso_y[k]))
                         for k in range(len(CATEGORIES))])


def load_rules(selective_json: dict, conformal_json: dict) -> dict:
    """Thresholds and quantiles from the committed M4/M5 results, keyed for the demo."""
    thresholds = {}
    for cat in CATEGORIES:
        per = selective_json["categories"][cat]["thresholds"]
        thresholds[cat] = {r: (c["choice"]["threshold"] if c["choice"]["certified"] else None)
                           for r, c in per.items()}
    quantiles = {}
    for variant, by_alpha in conformal_json["variants"].items():
        quantiles[variant] = {a: {cat: {str(int(y)): v for y, v in
                                        block["per_category"][cat]["q"].items()}
                                  for cat in CATEGORIES}
                              for a, block in by_alpha.items()}
    return {"thresholds": thresholds, "quantiles": quantiles}


def explain(model: PortableModel, rules: dict, text: str, risk: str = "0.01",
            alpha: str = "0.1", variant: str = "mondrian") -> dict:
    """Everything the demo shows for one text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty text")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text longer than {MAX_CHARS} characters")
    p = model.probabilities(text)
    rows = []
    for k, cat in enumerate(CATEGORIES):
        conf = max(p[k], 1 - p[k])
        t = rules["thresholds"][cat].get(risk)
        if t is None:
            decision = "abstain (no certified threshold)"
        elif conf >= t:
            decision = "yes" if p[k] >= 0.5 else "no"
        else:
            decision = "not sure"
        q = rules["quantiles"][variant][alpha][cat]
        labels = [y for y in (0, 1) if (1 - p[k] if y == 1 else p[k]) <= q[str(y)] + 1e-12]
        rows.append({"category": cat, "p": float(p[k]), "confidence": float(conf),
                     "threshold": t, "decision": decision,
                     "set": ["yes" if y else "no" for y in labels]})
    return {"rows": rows}


def rules_from_files(selective_path: Path, conformal_path: Path) -> dict:
    return load_rules(json.loads(Path(selective_path).read_text(encoding="utf-8")),
                      json.loads(Path(conformal_path).read_text(encoding="utf-8")))
