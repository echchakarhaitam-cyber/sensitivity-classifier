"""Shortcut analysis (RQ2): per-domain performance and what the top features are.

Feature annotation rule (fixed before looking at any result):

1. **field label** -- the feature contains a word (unigram or either bigram token) from
   :data:`FIELD_WORDS`: the words of every raw dataset label name (``blood_type`` -> blood),
   minus generic tokens (:data:`GENERIC_LABEL_TOKENS`), plus a short list of standard
   abbreviations (:data:`ABBREVIATIONS`). Such a feature names the sensitive field itself.
2. **domain/topic word** -- otherwise, if the domain explains a large share of whether the
   feature occurs: ``U = MI(F; D) / H(F) >= 0.25``, where F is "feature occurs in the
   document" over the TRAIN documents and D their domain. U is the uncertainty coefficient of
   F given D: 0 if the feature occurs at the same rate in every domain, 1 if the domain
   determines it. Field labels are checked first because a field name that is frequent in
   one domain still names the sensitive content.
3. **other** -- everything else.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sensclf.labels import CATEGORIES, TABLES
from sensclf.metrics import average_precision, is_defined

GENERIC_LABEL_TOKENS = frozenset({
    "number", "code", "type", "id", "of", "date", "time", "level", "status", "unique",
    "local", "identifier", "view", "first", "last", "full", "plan",
})
ABBREVIATIONS = frozenset({"dob", "mrn", "zip", "tel", "username", "login", "e-mail", "passcode",
                           "ssn", "cvv", "iban", "pin"})
FIELD_WORDS = frozenset(
    {tok for table in TABLES.values() for raw in table for tok in raw.split("_")
     if len(tok) >= 3 and tok not in GENERIC_LABEL_TOKENS} | ABBREVIATIONS)
DOMAIN_U_THRESHOLD = 0.25
FIELD, DOMAIN, OTHER = "field label", "domain/topic word", "other"


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def uncertainty_coefficient(present: np.ndarray, domains: np.ndarray) -> float:
    """U(F|D) = MI(F; D) / H(F) for a binary feature F; 0 when F is constant."""
    present = np.asarray(present, dtype=bool)
    domains = np.asarray(domains)
    if present.size == 0:
        return 0.0
    h_f = _entropy(np.array([present.mean(), 1 - present.mean()]))
    if h_f == 0:
        return 0.0
    h_f_given_d = 0.0
    for d in np.unique(domains):
        mask = domains == d
        rate = present[mask].mean()
        h_f_given_d += mask.mean() * _entropy(np.array([rate, 1 - rate]))
    return max(0.0, (h_f - h_f_given_d) / h_f)


def annotate(feature: str, u: float) -> str:
    if any(tok in FIELD_WORDS for tok in feature.split()):
        return FIELD
    return DOMAIN if u >= DOMAIN_U_THRESHOLD else OTHER


def top_features(model, X_train, domains: Sequence[str], k: int = 25) -> dict[str, list[dict]]:
    """Top-``k`` positive coefficients per category of a TF-IDF + linear model, annotated.

    ``X_train`` is the model's own TF-IDF matrix of the training documents (presence = > 0).
    """
    names = model.vectorizer.get_feature_names_out()
    domains = np.asarray([str(d) for d in domains])
    X_csc = X_train.tocsc()
    out = {}
    for c, head in zip(CATEGORIES, model.heads):
        coef = head.coef_.ravel()
        top = np.argsort(-coef, kind="mergesort")[:k]
        rows = []
        for j in top:
            present = np.zeros(X_csc.shape[0], dtype=bool)
            present[X_csc.indices[X_csc.indptr[j]:X_csc.indptr[j + 1]]] = True
            u = uncertainty_coefficient(present, domains)
            rows.append({"feature": str(names[j]), "coef": float(coef[j]), "u_domain": u,
                         "doc_freq": int(present.sum()), "annotation": annotate(names[j], u)})
        out[c] = rows
    return out


def per_domain_macro_ap(S: np.ndarray, Y: np.ndarray, domains: Sequence[str],
                        seen: frozenset[str]) -> list[dict]:
    """Macro AP per domain over the categories defined (and covered) inside that domain.

    ``seen`` holds the case-folded training domains.
    """
    domains = np.asarray([str(d) for d in domains])
    rows = []
    for d in sorted(set(domains)):
        mask = domains == d
        aps = [average_precision(Y[mask, k], S[mask, k]) for k in range(len(CATEGORIES))
               if is_defined(Y[mask, k]) and not np.isnan(S[mask, k]).any()]
        rows.append({"domain": d, "documents": int(mask.sum()),
                     "macro_ap": float(np.mean(aps)) if aps else None,
                     "categories_defined": len(aps), "seen_in_train": d.casefold() in seen})
    return rows
