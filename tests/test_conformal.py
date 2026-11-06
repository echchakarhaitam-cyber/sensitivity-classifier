import math

import numpy as np
import pytest

from sensclf.conformal import (
    conformal_quantile,
    coverage_bounds,
    fit_quantiles,
    nonconformity,
    one_per_group,
    prediction_sets,
    set_statistics,
)


def test_nonconformity():
    p = np.array([0.9, 0.9, 0.2])
    assert np.allclose(nonconformity(p, np.array([1, 0, 0])), [0.1, 0.9, 0.2])


def test_quantile_rank_and_infinity():
    s = np.arange(1, 10) / 10.0          # n = 9
    # alpha 0.2: rank ceil(10 * 0.8) = 8 -> 0.8
    assert conformal_quantile(s, 0.2) == pytest.approx(0.8)
    # alpha 0.05: rank ceil(10 * 0.95) = 10 > 9 -> inf
    assert math.isinf(conformal_quantile(s, 0.05))
    assert math.isinf(conformal_quantile(np.array([]), 0.1))


def test_sets_and_statistics_hand_example():
    q = {0: 0.3, 1: 0.3}
    p = np.array([0.9, 0.5, 0.1, 0.75])
    sets = prediction_sets(p, q)
    # p=0.9 -> a(1)=0.1 in, a(0)=0.9 out -> {1}; p=0.5 -> both 0.5 > 0.3 -> empty
    # p=0.1 -> {0}; p=0.75 -> a(1)=0.25 in, a(0)=0.75 out -> {1}
    assert sets.tolist() == [[False, True], [False, False], [True, False], [False, True]]
    stats = set_statistics(sets, np.array([1, 1, 0, 0]))
    assert stats["coverage"] == 0.5
    assert stats["singleton"] == 0.75 and stats["empty"] == 0.25 and stats["both"] == 0.0
    assert stats["coverage_y1"] == 0.5 and stats["coverage_y0"] == 0.5
    assert stats["singleton_error"] == pytest.approx(1 / 3)


def test_marginal_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(0)

    def draw(n):
        p = rng.random(n)
        return p, (rng.random(n) < p).astype(int)

    covs = []
    for alpha in (0.05, 0.1):
        p_cal, y_cal = draw(2000)
        q = fit_quantiles(p_cal, y_cal, alpha)
        p_new, y_new = draw(100_000)
        covs.append(set_statistics(prediction_sets(p_new, q), y_new)["coverage"])
        lo, hi = coverage_bounds(2000, alpha)
        assert covs[-1] >= lo - 0.01
        assert covs[-1] <= hi + 0.01


def test_mondrian_gives_class_conditional_coverage():
    rng = np.random.default_rng(1)
    # a model that is bad on positives: their p is low
    y = (rng.random(20000) < 0.2).astype(int)
    p = np.clip(np.where(y == 1, rng.beta(2, 3, len(y)), rng.beta(1, 6, len(y))), 0, 1)
    cal, new = slice(0, 10000), slice(10000, None)
    q_m = fit_quantiles(p[cal], y[cal], 0.1, mondrian=True)
    q_s = fit_quantiles(p[cal], y[cal], 0.1)
    st_m = set_statistics(prediction_sets(p[new], q_m), y[new])
    st_s = set_statistics(prediction_sets(p[new], q_s), y[new])
    assert st_m["coverage_y1"] >= 0.88
    assert st_m["coverage_y0"] >= 0.88
    assert st_s["coverage_y1"] < st_m["coverage_y1"]


def test_one_per_group_deterministic():
    groups = ["a", "a", "b", "c", "c", "c"]
    idx = one_per_group(groups, seed=0)
    assert len(idx) == 3 and len({groups[i] for i in idx}) == 3
    assert np.array_equal(idx, one_per_group(groups, seed=0))
