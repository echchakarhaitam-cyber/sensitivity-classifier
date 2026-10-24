import numpy as np
import pytest
from toy import toy_docs

from sensclf.guard import CalibrationLeakError, ensure_no_calibration
from sensclf.labels import CATEGORIES
from sensclf.models import TextModel, fit_grid, label_matrix, load_model, load_sidecar, save_model
from sensclf.models.base import doc_ids_sha256

NONE: frozenset[str] = frozenset()
TRAIN = toy_docs(60, seed=0)
TEST = toy_docs(20, seed=1, prefix="test")


@pytest.mark.parametrize("family", ["word_lr", "char_lr"])
def test_probability_models_learn_the_cues(family):
    model = TextModel(family, C=10.0).fit(TRAIN, forbidden_ids=NONE)
    scores = model.predict_scores(TEST)
    assert scores.shape == (20, 6) and ((scores >= 0) & (scores <= 1)).all()
    Y = label_matrix(TEST)
    assert ((scores >= 0.5).astype(int) == Y).mean() > 0.9


def test_svc_returns_margins_not_probabilities():
    model = TextModel("word_svc", C=1.0).fit(TRAIN, forbidden_ids=NONE)
    scores = model.predict_scores(TEST)
    assert not model.probabilistic
    assert (scores < 0).any() and (scores > 1).any()
    assert ((scores >= 0).astype(int) == label_matrix(TEST)).mean() > 0.9


def test_fit_rejects_calibration_documents():
    forbidden = frozenset({TRAIN[3].doc_id})
    with pytest.raises(CalibrationLeakError):
        TextModel("word_lr", 1.0).fit(TRAIN, forbidden_ids=forbidden)
    with pytest.raises(CalibrationLeakError):
        fit_grid("word_lr", TRAIN, [1.0], forbidden_ids=forbidden)


def test_ensure_no_calibration_accepts_clean_ids():
    ensure_no_calibration(["a", "b"], frozenset({"c"}))


def test_fit_rejects_a_category_with_one_class():
    docs = [d for d in TRAIN if d.labels["HEALTH"] == 1]
    with pytest.raises(ValueError, match="one class"):
        TextModel("word_lr", 1.0).fit(docs, forbidden_ids=NONE)


def test_grid_shares_the_vectorizer_and_matches_a_direct_fit():
    fits, vec_seconds = fit_grid("word_lr", TRAIN, [0.1, 10.0], forbidden_ids=NONE)
    assert vec_seconds >= 0 and [f.model.C for f in fits] == [0.1, 10.0]
    assert fits[0].model.vectorizer is fits[1].model.vectorizer
    direct = TextModel("word_lr", 10.0).fit(TRAIN, forbidden_ids=NONE)
    np.testing.assert_allclose(fits[1].model.predict_scores(TEST), direct.predict_scores(TEST))


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        TextModel("word_lr", 1.0).predict_scores(TEST)


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        TextModel("bert", 1.0).fit(TRAIN, forbidden_ids=NONE)


def test_save_load_gives_identical_scores_and_a_sidecar(tmp_path):
    model = TextModel("word_svc", 0.1).fit(TRAIN, forbidden_ids=NONE)
    info = {"family": "word_svc", "training_set": {"doc_ids_sha256": doc_ids_sha256(TRAIN)}}
    path = save_model(model, tmp_path / "m.joblib", info)
    loaded = load_model(path)
    assert np.array_equal(loaded.predict_scores(TEST), model.predict_scores(TEST))
    sidecar = load_sidecar(path)
    assert sidecar["family"] == "word_svc" and "scikit-learn" in sidecar["versions"]


def test_doc_ids_sha256_ignores_order():
    assert doc_ids_sha256(TRAIN) == doc_ids_sha256(list(reversed(TRAIN)))
    assert doc_ids_sha256(TRAIN) != doc_ids_sha256(TRAIN[1:])


def test_label_matrix_order_and_empty():
    assert label_matrix(TRAIN)[0].tolist() == [TRAIN[0].labels[c] for c in CATEGORIES]
    assert label_matrix([]).shape == (0, 6)


@pytest.mark.slow
@pytest.mark.data
def test_full_training_word_lr_save_load_on_real_data(tmp_path):
    from sensclf.guard import calibration_ids
    from sensclf.metrics import average_precision
    from sensclf.processed import load_set

    train, dev = load_set("train"), load_set("dev")[:2000]
    model = TextModel("word_lr", 1.0).fit(train, forbidden_ids=calibration_ids())
    loaded = load_model(save_model(model, tmp_path / "w.joblib", {"family": "word_lr"}))
    scores = loaded.predict_scores(dev)
    assert np.array_equal(scores, model.predict_scores(dev))
    Y = label_matrix(dev)
    assert np.mean([average_precision(Y[:, k], scores[:, k]) for k in range(6)]) > 0.5
