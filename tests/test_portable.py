"""The NumPy-only demo model must reproduce the scikit-learn model and the committed rules."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from sensclf.portable import CATEGORIES, MAX_CHARS, PortableModel, analyze, explain, load_rules

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
TEXTS = ["Patient John Smith, born 1980, was treated for asthma at St Mary's.",
         "Please wire 2,500 EUR to IBAN DE89 3704 0044 0532 0130 00 today.",
         "My password is hunter2 and my login is jsmith.",
         "Call me on +44 20 7946 0958 or write to ana@example.org.",
         "The weather is nice. ÉTÉ à Paris, naïve café!", "a b c"]


@pytest.fixture(scope="module")
def model():
    return PortableModel.load(DEMO / "model.npz")


@pytest.fixture(scope="module")
def rules():
    return json.loads((DEMO / "rules.json").read_text(encoding="utf-8"))


def test_analyzer_matches_scikit_learn():
    an = TfidfVectorizer(ngram_range=(1, 2)).build_analyzer()
    for text in TEXTS:
        assert analyze(text) == an(text)


def test_shapes(model):
    v = len(model.vocabulary)
    assert model.idf.shape == (v,) and model.coef.shape == (len(CATEGORIES), v)
    assert len(model.iso_x) == len(model.iso_y) == len(CATEGORIES)


def test_probabilities_in_range_and_deterministic(model):
    for text in TEXTS:
        p = model.probabilities(text)
        assert p.shape == (len(CATEGORIES),) and ((p >= 0) & (p <= 1)).all()
        assert np.array_equal(p, model.probabilities(text))


def test_unknown_words_give_intercept(model):
    assert np.allclose(model.margins("zzqxv qqqzzx"), model.intercept)


def test_rules_file_matches_committed_results(rules):
    rebuilt = load_rules(json.loads((ROOT / "results/m4/selective.json").read_text("utf-8")),
                         json.loads((ROOT / "results/m5/conformal.json").read_text("utf-8")))
    assert json.loads(json.dumps(rebuilt)) == rules


def test_explain(model, rules):
    out = explain(model, rules, TEXTS[0], risk="0.01", alpha="0.1", variant="mondrian")
    assert [r["category"] for r in out["rows"]] == list(CATEGORIES)
    for r in out["rows"]:
        assert r["decision"] in ("yes", "no", "not sure", "abstain (no certified threshold)")
        assert set(r["set"]) <= {"yes", "no"}
        if r["threshold"] is not None and r["decision"] in ("yes", "no"):
            assert r["confidence"] >= r["threshold"]


@pytest.mark.parametrize("text", ["", "   ", "x" * (MAX_CHARS + 1)])
def test_explain_rejects_bad_text(model, rules, text):
    with pytest.raises(ValueError):
        explain(model, rules, text)


@pytest.mark.data
def test_margins_match_the_trained_model(model):
    path = ROOT / "models" / "m2" / "word_svc.joblib"
    if not path.exists():
        pytest.skip("trained model not present")
    from sensclf.models.base import load_model

    m = load_model(path)
    X = m.vectorizer.transform(TEXTS)
    S = np.column_stack([h.decision_function(X) for h in m.heads])
    P = np.array([model.margins(t) for t in TEXTS])
    assert np.abs(S - P).max() < 1e-5
