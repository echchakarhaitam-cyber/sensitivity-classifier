import numpy as np
import pytest
from toy import toy_docs

from sensclf.models import TextModel
from sensclf.shortcuts import (
    DOMAIN,
    FIELD,
    FIELD_WORDS,
    OTHER,
    annotate,
    per_domain_macro_ap,
    top_features,
    uncertainty_coefficient,
)


def test_field_words_come_from_raw_labels_without_generic_tokens():
    assert {"blood", "password", "ssn", "email", "religious", "passport"} <= FIELD_WORDS
    assert not {"number", "type", "code", "id"} & FIELD_WORDS


@pytest.mark.parametrize("feature, u, expected", [
    ("blood", 0.9, FIELD),           # field label wins even when domain-specific
    ("blood type", 0.0, FIELD),
    ("my password", 0.1, FIELD),
    ("patient", 0.6, DOMAIN),
    ("patient", 0.25, DOMAIN),       # threshold is inclusive
    ("patient", 0.24, OTHER),
    ("the", 0.0, OTHER),
    ("number", 0.0, OTHER),          # generic label token
])
def test_annotation_rule(feature, u, expected):
    assert annotate(feature, u) == expected


def test_uncertainty_coefficient_extremes():
    domains = np.array(["a"] * 50 + ["b"] * 50)
    only_a = np.r_[np.ones(50), np.zeros(50)].astype(bool)
    assert uncertainty_coefficient(only_a, domains) == pytest.approx(1.0)
    even = np.tile([True, False], 50)
    assert uncertainty_coefficient(even, domains) == pytest.approx(0.0)
    assert uncertainty_coefficient(np.ones(100, dtype=bool), domains) == 0.0


def test_uncertainty_coefficient_partial_dependence_is_between_0_and_1():
    domains = np.array(["a"] * 50 + ["b"] * 50)
    present = np.r_[np.ones(40), np.zeros(10), np.ones(10), np.zeros(40)].astype(bool)
    assert 0 < uncertainty_coefficient(present, domains) < 1


def test_top_features_on_toy_model_finds_the_cues():
    docs = toy_docs(60)
    model = TextModel("word_lr", 10.0).fit(docs, forbidden_ids=frozenset())
    X = model.transform(docs)
    top = top_features(model, X, [d.meta["domain"] for d in docs], k=5)
    words = {r["feature"] for r in top["HEALTH"]}
    assert words & {"medical", "blood", "record", "blood type", "medical record"}
    assert all(r["annotation"] in (FIELD, DOMAIN, OTHER) for r in top["HEALTH"])
    assert all(0 <= r["u_domain"] <= 1 for rows in top.values() for r in rows)


def test_per_domain_macro_ap_and_seen_flag():
    Y = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    S = Y * 0.9 + 0.05
    rows = per_domain_macro_ap(S, Y, ["A", "A", "B", "B"], frozenset({"a"}))
    a, b = rows
    assert a["domain"] == "A" and a["seen_in_train"] and a["categories_defined"] == 2
    assert a["macro_ap"] == pytest.approx(1.0)
    assert not b["seen_in_train"] and b["documents"] == 2
