import numpy as np
import pytest

from sensclf.selective import (
    apply_threshold,
    aurc,
    binomial_pvalue,
    choose_threshold,
    decisions,
    errors,
    oracle_aurc,
    risk_at_coverage,
    risk_coverage,
)


def test_decisions_and_errors():
    p = np.array([0.9, 0.2, 0.5, 0.6])
    yhat, conf = decisions(p)
    assert yhat.tolist() == [1, 0, 1, 1]
    assert np.allclose(conf, [0.9, 0.8, 0.5, 0.6])
    assert errors(p, np.array([1, 1, 1, 0])).tolist() == [0, 1, 0, 1]


def test_risk_coverage_hand_example():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    err = np.array([0, 0, 1, 1])
    c = risk_coverage(conf, err)
    assert np.allclose(c.coverage, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(c.risk, [0, 0, 1 / 3, 0.5])
    # AURC = mean of (0, 0, 1/3, 1/2)
    assert aurc(conf, err) == pytest.approx((1 / 3 + 0.5) / 4)
    assert oracle_aurc(err) == pytest.approx((1 / 3 + 0.5) / 4)   # this ranking is perfect


def test_ties_are_accepted_together():
    conf = np.array([0.9, 0.9, 0.5])
    err = np.array([1, 0, 0])
    c = risk_coverage(conf, err)
    assert np.allclose(c.coverage, [2 / 3, 1.0])
    assert np.allclose(c.risk, [0.5, 1 / 3])
    assert aurc(np.full(4, 0.7), np.array([1, 0, 0, 0])) == pytest.approx(0.25)


def test_bad_ranking_has_higher_aurc_than_oracle():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    err = np.array([1, 1, 0, 0])
    assert aurc(conf, err) > oracle_aurc(err)


def test_risk_at_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    err = np.array([0, 0, 1, 1])
    r = risk_at_coverage(conf, err, 0.5)
    assert r["coverage"] == 0.5 and r["risk"] == 0.0


def test_binomial_pvalue_values():
    # 0 errors in 300 at r = 0.01: P(K=0) = 0.99^300
    assert binomial_pvalue(0, 300, 0.01) == pytest.approx(0.99 ** 300)
    assert binomial_pvalue(0, 0, 0.01) == 1.0


def test_choose_threshold_certifies_and_controls_risk():
    rng = np.random.default_rng(0)

    def draw(n):
        # confidences concentrated near 1, as for a good model; perfectly calibrated errors
        conf = 1 - 0.5 * rng.beta(1, 8, n)
        return conf, (rng.random(n) < (1 - conf)).astype(int)

    conf, err = draw(20_000)
    out = choose_threshold(conf, err, target=0.02, delta=0.05)
    assert out["certified"]
    assert out["cal_risk"] <= 0.02 and 0 < out["cal_coverage"] < 1
    # with the chosen threshold, the true risk on fresh data from the same law stays below r*
    conf2, err2 = draw(400_000)
    assert apply_threshold(conf2, err2, out["threshold"])["risk"] <= 0.02


def test_choose_threshold_not_certified_when_too_few_docs():
    out = choose_threshold(np.array([0.99, 0.98]), np.array([0, 0]), target=0.01)
    assert not out["certified"] and out["threshold"] is None
    assert apply_threshold(np.array([0.9]), np.array([0]), None)["coverage"] == 0.0
