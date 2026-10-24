"""Linear text classifiers: TF-IDF features + one binary linear model per category.

Families (docs/results.md, Milestone 2):

* ``word_lr``  word 1-2 gram TF-IDF (sublinear tf, min_df 3, 200k features) + LogisticRegression
* ``char_lr``  char_wb 3-5 gram TF-IDF (sublinear tf, 300k features) + LogisticRegression
* ``word_svc`` the word features + LinearSVC; scores are margins, NOT probabilities (kept as
  the uncalibrated-score model for Milestone 3)

One-vs-rest: the six heads share one vectorizer and are fitted independently. The vectorizer
is fitted on the training documents only. All solvers are liblinear with a fixed seed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sensclf.guard import ensure_no_calibration
from sensclf.labels import CATEGORIES
from sensclf.models.base import label_matrix
from sensclf.splits import LabelledDocument

FAMILIES = ("word_lr", "char_lr", "word_svc")
C_GRIDS = {"word_lr": (0.1, 1.0, 10.0), "char_lr": (0.1, 1.0, 10.0), "word_svc": (0.01, 0.1, 1.0)}
SEED = 0


def make_vectorizer(family: str) -> TfidfVectorizer:
    if family in ("word_lr", "word_svc"):
        return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=3,
                               max_features=200_000, dtype=np.float32)
    if family == "char_lr":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                               max_features=300_000, dtype=np.float32)
    raise ValueError(f"unknown family {family!r}")


def make_head(family: str, C: float, seed: int):
    if family == "word_svc":
        return LinearSVC(C=C, dual=True, max_iter=10_000, random_state=seed)
    return LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=seed)


@dataclass
class TextModel:
    """One family at one C: a fitted vectorizer and six binary heads."""

    family: str
    C: float
    seed: int = SEED
    vectorizer: TfidfVectorizer | None = field(default=None, repr=False)
    heads: list = field(default_factory=list, repr=False)

    @property
    def name(self) -> str:
        return self.family

    @property
    def probabilistic(self) -> bool:
        return self.family != "word_svc"

    def fit(self, docs: Sequence[LabelledDocument], *, forbidden_ids: frozenset[str]) -> TextModel:
        ensure_no_calibration((d.doc_id for d in docs), forbidden_ids)
        self.vectorizer = make_vectorizer(self.family)
        X = self.vectorizer.fit_transform([d.text for d in docs])
        self.fit_heads(X, label_matrix(docs))
        return self

    def fit_heads(self, X, Y: np.ndarray) -> None:
        self.heads = []
        for k, category in enumerate(CATEGORIES):
            if len(set(Y[:, k])) < 2:
                raise ValueError(f"{category}: training labels contain one class only")
            self.heads.append(make_head(self.family, self.C, self.seed).fit(X, Y[:, k]))

    def transform(self, docs: Sequence[LabelledDocument]):
        if self.vectorizer is None:
            raise RuntimeError("model is not fitted")
        return self.vectorizer.transform([d.text for d in docs])

    def scores_from_matrix(self, X) -> np.ndarray:
        if self.probabilistic:
            cols = [h.predict_proba(X)[:, 1] for h in self.heads]
        else:
            cols = [h.decision_function(X) for h in self.heads]
        return np.column_stack(cols).astype(np.float64)

    def predict_scores(self, docs: Sequence[LabelledDocument]) -> np.ndarray:
        return self.scores_from_matrix(self.transform(docs))


@dataclass(frozen=True)
class GridFit:
    model: TextModel
    head_seconds: float


def fit_grid(family: str, docs: Sequence[LabelledDocument], Cs: Sequence[float], *,
             forbidden_ids: frozenset[str], seed: int = SEED) -> tuple[list[GridFit], float]:
    """Fit the vectorizer once, then the heads for every C; return fits and vectorizer time.

    Every model of the grid shares the same fitted vectorizer object (identical features), so
    the grid differs only in C.
    """
    ensure_no_calibration((d.doc_id for d in docs), forbidden_ids)
    start = time.perf_counter()
    vectorizer = make_vectorizer(family)
    X = vectorizer.fit_transform([d.text for d in docs])
    vec_seconds = time.perf_counter() - start
    Y = label_matrix(docs)
    fits = []
    for C in Cs:
        start = time.perf_counter()
        model = TextModel(family, C, seed, vectorizer)
        model.fit_heads(X, Y)
        fits.append(GridFit(model, time.perf_counter() - start))
    return fits, vec_seconds
