import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from sensclf import metrics as M


def test_average_precision_hand_computed():
    # ranking: + - + -   -> precision at the two positives 1/1 and 2/3 -> AP = (1 + 2/3) / 2
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert M.average_precision(y, s) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_with_ties_matches_sklearn():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.1])
    assert M.average_precision(y, s) == pytest.approx(average_precision_score(y, s))


def test_weighted_average_precision_matches_sklearn_sample_weight():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 2)          # rounding creates ties
    W = rng.integers(0, 3, (5, 200)).astype(float)
    got = M.APScorer(y, s)(W)
    for r in range(5):
        assert got[r] == pytest.approx(average_precision_score(y, s, sample_weight=W[r]))


def test_ap_is_nan_when_a_resample_has_no_positive_weight():
    y, s = np.array([1, 0]), np.array([0.9, 0.1])
    assert np.isnan(M.APScorer(y, s)(np.array([[0.0, 2.0]]))[0])


def test_f1_from_counts_and_zero_division():
    assert M.f1_from_counts(2, 1, 1) == pytest.approx(4 / 6)
    assert M.f1_from_counts(0, 3, 2) == 0.0
    assert M.f1_from_counts(0, 0, 0) == 0.0


def test_category_metrics_hand_computed():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.1])
    m = M.category_metrics(y, s, {"default": 0.5}, probabilistic=True)
    at = m["at"]["default"]                    # predicted positive: docs 0 and 2
    assert (at["precision"], at["recall"], at["f1"]) == (0.5, 0.5, 0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)
    expected_ll = -np.mean(np.log([0.9, 0.4, 0.4, 0.9]))
    assert m["log_loss"] == pytest.approx(expected_ll)


def test_category_metrics_svm_scores_have_no_probability_metrics():
    m = M.category_metrics(np.array([1, 0]), np.array([1.3, -2.0]), {"default": 0.0}, False)
    assert m["brier"] is None and m["log_loss"] is None and m["at"]["default"]["f1"] == 1.0


@pytest.mark.parametrize("y", [np.array([0, 0, 0]), np.array([1, 1]), np.array([], dtype=int)])
def test_undefined_category_gives_none(y):
    m = M.category_metrics(y, np.zeros(len(y)), {"default": 0.5}, True)
    assert not m["defined"]
    assert m["average_precision"] is None and m["at"]["default"] is None


def test_vector_metrics_hand_computed():
    Y = np.array([[1, 0, 1], [0, 1, 0]])
    P = np.array([[1, 0, 0], [0, 1, 1]])
    v = M.vector_metrics(Y, P, [0, 1, 2])
    assert v["subset_accuracy"] == 0.0
    assert v["hamming_loss"] == pytest.approx(2 / 6)
    assert v["micro_f1"] == pytest.approx(f1_score(Y, P, average="micro"))
    assert v["macro_f1"] == pytest.approx(f1_score(Y, P, average="macro", zero_division=0))


def test_vector_metrics_use_only_given_columns():
    Y = np.array([[1, 0], [1, 0]])
    P = np.array([[1, 1], [1, 1]])
    assert M.vector_metrics(Y, P, [0])["subset_accuracy"] == 1.0
    assert M.vector_metrics(Y, P, [])["micro_f1"] is None


def test_best_f1_threshold_hand_computed():
    y = np.array([1, 1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    # cut at 0.6 -> TP 3, FP 1 -> F1 6/7, better than cut at 0.8 (TP 2 -> F1 4/5)
    assert M.best_f1_threshold(y, s) == 0.6


def test_best_f1_threshold_tie_takes_highest_threshold():
    y = np.array([1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6])
    # cut at 0.9: TP 1, FN 1 -> 2/3; cut at 0.6: TP 2, FP 2 -> 2/3: tie -> 0.9
    assert M.best_f1_threshold(y, s) == 0.9


def test_best_f1_threshold_rejects_single_class():
    with pytest.raises(ValueError):
        M.best_f1_threshold(np.array([0, 0]), np.array([0.1, 0.2]))
