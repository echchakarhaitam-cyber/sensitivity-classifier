import numpy as np
import pytest

from sensclf import bootstrap as B

GROUPS = ["g1", "g1", "g2", "g2", "g3", "solo"]


def test_documents_of_one_group_always_get_the_same_weight():
    index, n_groups = B.group_index(GROUPS)
    first_doc = [list(index).index(g) for g in range(n_groups)]
    for W in B.iter_weights(GROUPS, n_resamples=300, seed=0, chunk=64):
        assert np.array_equal(W[:, 0], W[:, 1])
        assert np.array_equal(W[:, 2], W[:, 3])
        # each resample draws exactly G groups
        assert (W[:, first_doc].sum(axis=1) == n_groups).all()


def test_bootstrap_is_deterministic_and_chunking_does_not_matter():
    stats = {"total": lambda W: W.sum(axis=1), "first": lambda W: W[:, 0]}
    a = B.cluster_bootstrap(GROUPS, stats, n_resamples=250, seed=0, chunk=100)
    b = B.cluster_bootstrap(GROUPS, stats, n_resamples=250, seed=0, chunk=7)
    assert np.array_equal(a["first"], b["first"]) and len(a["first"]) == 250
    c = B.cluster_bootstrap(GROUPS, stats, n_resamples=250, seed=1)
    assert not np.array_equal(a["first"], c["first"])


def test_statistics_are_paired_on_identical_resamples():
    stats = {"a": lambda W: W[:, 4], "b": lambda W: W[:, 4] * 2}
    out = B.cluster_bootstrap(GROUPS, stats, n_resamples=50)
    assert np.array_equal(out["b"], 2 * out["a"])


def test_group_index_in_order_of_first_appearance():
    index, n = B.group_index(["b", "a", "b"])
    assert list(index) == [0, 1, 0] and n == 2


def test_interval_ignores_nan_and_handles_all_nan():
    values = np.r_[np.arange(101.0), np.nan]
    assert B.interval(values) == pytest.approx((2.5, 97.5))
    assert B.interval(np.array([np.nan])) is None


def test_mean_interval_covers_the_truth_for_independent_groups():
    rng = np.random.default_rng(0)
    groups = [f"g{i // 2}" for i in range(400)]
    x = np.repeat(rng.normal(size=200), 2)
    out = B.cluster_bootstrap(groups, {"mean": lambda W: (W @ x) / W.sum(axis=1)})
    lo, hi = B.interval(out["mean"])
    assert lo < x.mean() < hi
