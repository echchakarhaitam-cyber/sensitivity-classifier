"""Two reference models: a span detector without training, and a domain-only classifier.

``regex_detector`` -- privgate's RegexDetector, unchanged. A document is CONTACT if it finds an
EMAIL or PHONE, FINANCIAL if it finds a CREDIT_CARD or IBAN; scores are 0/1. The other four
categories are not covered (NaN columns, reported as "n/a"). It shows how far "a detector
fires" is from the document labels, which are defined by every annotated label of a category.

``domain_lr`` -- one LogisticRegression per category on the ONE-HOT DOMAIN ALONE (no text).
Whatever it achieves is reachable without reading the document; if it comes close to the text
models in-domain, the task contains a domain shortcut. Domain strings are case-folded (Gretel's
``finance`` then meets Nemotron's ``Finance``); a domain unseen in training maps to the
all-zero vector, so the prediction falls back to the intercept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from privgate.detectors.regex import RegexDetector
from privgate.spans import EntityType
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from sensclf.guard import ensure_no_calibration
from sensclf.labels import CATEGORIES, CONTACT, FINANCIAL
from sensclf.models.base import label_matrix
from sensclf.splits import LabelledDocument

DETECTOR_RULES: dict[str, frozenset[EntityType]] = {
    CONTACT: frozenset({EntityType.EMAIL, EntityType.PHONE}),
    FINANCIAL: frozenset({EntityType.CREDIT_CARD, EntityType.IBAN}),
}


class DetectorBaseline:
    """No training; NaN for categories the detector does not cover."""

    name = "regex_detector"
    probabilistic = False
    default_threshold = 0.5   # scores are 0/1
    covered = tuple(c for c in CATEGORIES if c in DETECTOR_RULES)

    def __init__(self) -> None:
        self.detector = RegexDetector()

    def predict_scores(self, docs: Sequence[LabelledDocument]) -> np.ndarray:
        out = np.full((len(docs), len(CATEGORIES)), np.nan)
        for i, doc in enumerate(docs):
            found = {s.label for s in self.detector.detect(doc.text)}
            for k, category in enumerate(CATEGORIES):
                if category in DETECTOR_RULES:
                    out[i, k] = float(bool(found & DETECTOR_RULES[category]))
        return out


def domain_of(doc: LabelledDocument) -> str:
    return str(doc.meta.get("domain", "")).casefold()


@dataclass
class DomainModel:
    """LogisticRegression per category on the one-hot (case-folded) domain."""

    C: float = 1.0
    seed: int = 0
    encoder: OneHotEncoder | None = field(default=None, repr=False)
    heads: list = field(default_factory=list, repr=False)

    name = "domain_lr"
    probabilistic = True

    def fit(self, docs: Sequence[LabelledDocument], *,
            forbidden_ids: frozenset[str]) -> DomainModel:
        ensure_no_calibration((d.doc_id for d in docs), forbidden_ids)
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        X = self.encoder.fit_transform([[domain_of(d)] for d in docs])
        Y = label_matrix(docs)
        self.heads = [LogisticRegression(C=self.C, solver="liblinear", max_iter=1000,
                                         random_state=self.seed).fit(X, Y[:, k])
                      for k in range(len(CATEGORIES))]
        return self

    @property
    def domains(self) -> list[str]:
        if self.encoder is None:
            raise RuntimeError("model is not fitted")
        return [str(v) for v in self.encoder.categories_[0]]

    def predict_scores(self, docs: Sequence[LabelledDocument]) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("model is not fitted")
        X = self.encoder.transform([[domain_of(d)] for d in docs])
        return np.column_stack([h.predict_proba(X)[:, 1] for h in self.heads])
