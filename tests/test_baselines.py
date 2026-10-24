import numpy as np
import pytest

from sensclf.guard import CalibrationLeakError
from sensclf.labels import CATEGORIES
from sensclf.models import load_model, save_model
from sensclf.models.baselines import DetectorBaseline, DomainModel
from sensclf.splits import LabelledDocument

ZERO = dict.fromkeys(CATEGORIES, 0)
K = {c: i for i, c in enumerate(CATEGORIES)}


def _doc(i, text="", domain="x", **labels):
    return LabelledDocument(f"d{i}", text, {**ZERO, **labels}, {"domain": domain})


def test_detector_baseline_on_fixed_texts():
    docs = [
        _doc(0, "Write to ann@example.org today."),
        _doc(1, "Card 4111 1111 1111 1111 was charged."),          # Luhn-valid test number
        _doc(2, "IBAN DE89 3704 0044 0532 0130 00 for the refund."),  # checksum-valid IBAN
        _doc(3, "The meeting moved to Tuesday."),
    ]
    s = DetectorBaseline().predict_scores(docs)
    assert s[:, K["CONTACT"]].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert s[:, K["FINANCIAL"]].tolist() == [0.0, 1.0, 1.0, 0.0]
    for c in ("IDENTITY", "HEALTH", "CREDENTIALS", "SPECIAL_CATEGORY"):
        assert np.isnan(s[:, K[c]]).all()
    assert DetectorBaseline.covered == ("CONTACT", "FINANCIAL")


def _toy_domain_docs():
    # Health docs are HEALTH-positive 9 times out of 10, finance docs FINANCIAL-positive.
    docs = []
    for i in range(40):
        domain = "Health" if i % 2 else "Finance"
        hit = i % 20 not in (0, 1)
        labels = {"HEALTH": int(domain == "Health" and hit),
                  "FINANCIAL": int(domain == "Finance" and hit),
                  "IDENTITY": i % 4 == 0, "CONTACT": i % 3 == 0, "CREDENTIALS": i % 5 == 0,
                  "SPECIAL_CATEGORY": i % 7 == 0}
        docs.append(_doc(i, "text is ignored", domain, **{k: int(v) for k, v in labels.items()}))
    return docs


def test_domain_only_baseline_learns_the_domain_and_ignores_text():
    model = DomainModel(C=10.0).fit(_toy_domain_docs(), forbidden_ids=frozenset())
    health, finance = _doc(100, "card number 4111", "HEALTH"), _doc(101, "blood type", "finance")
    s = model.predict_scores([health, finance])
    assert s[0, K["HEALTH"]] > 0.7 > s[1, K["HEALTH"]]          # case-folded domain
    assert s[1, K["FINANCIAL"]] > 0.7 > s[0, K["FINANCIAL"]]
    assert model.domains == ["finance", "health"]


def test_domain_only_baseline_falls_back_to_intercept_for_unseen_domains():
    model = DomainModel().fit(_toy_domain_docs(), forbidden_ids=frozenset())
    s = model.predict_scores([_doc(1, "a", "Space"), _doc(2, "b", "Ocean"), _doc(3, "c", "")])
    assert np.allclose(s[0], s[1]) and np.allclose(s[0], s[2])


def test_domain_only_baseline_guard_and_roundtrip(tmp_path):
    docs = _toy_domain_docs()
    with pytest.raises(CalibrationLeakError):
        DomainModel().fit(docs, forbidden_ids=frozenset({"d5"}))
    model = DomainModel().fit(docs, forbidden_ids=frozenset())
    loaded = load_model(save_model(model, tmp_path / "d.joblib", {"family": "domain_lr"}))
    assert np.array_equal(loaded.predict_scores(docs), model.predict_scores(docs))


def test_unfitted_domain_model_raises():
    with pytest.raises(RuntimeError):
        DomainModel().predict_scores([_doc(0)])
