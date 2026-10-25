import numpy as np
import pytest

from sensclf.evaluation import bootstrap_metrics, evaluable, point_metrics
from sensclf.labels import CATEGORIES

rng = np.random.default_rng(0)
N = 120
Y = rng.integers(0, 2, (N, 6))
Y[:, 5] = 0                                    # SPECIAL_CATEGORY undefined in this set
S_GOOD = np.clip(Y * 0.6 + rng.random((N, 6)) * 0.4, 0, 1)
S_NOISE = rng.random((N, 6))
S_DET = np.full((N, 6), np.nan)
S_DET[:, 1] = Y[:, 1]                          # covers CONTACT only
GROUPS = [f"g{i // 2}" for i in range(N)]
T = dict.fromkeys(CATEGORIES, 0.5)
OPS = {"default": T, "dev_f1": T}


def test_evaluable_excludes_undefined_and_uncovered():
    assert evaluable(S_GOOD, Y) == [0, 1, 2, 3, 4]
    assert evaluable(S_DET, Y) == [1]


def test_point_metrics_macro_excludes_undefined_category():
    m = point_metrics(S_GOOD, Y, OPS, probabilistic=True)
    assert m["macro"]["excluded"] == ["SPECIAL_CATEGORY"]
    assert m["per_category"]["SPECIAL_CATEGORY"]["defined"] is False
    aps = [m["per_category"][c]["average_precision"] for c in CATEGORIES[:5]]
    assert m["macro"]["average_precision"] == pytest.approx(np.mean(aps))
    assert m["vector"]["dev_f1"]["categories"] == list(CATEGORIES[:5])


def test_point_metrics_marks_uncovered_categories():
    m = point_metrics(S_DET, Y, OPS, probabilistic=False)
    assert m["per_category"]["IDENTITY"] == {"covered": False}
    assert m["macro"]["categories"] == ["CONTACT"]
    assert m["per_category"]["CONTACT"]["at"]["default"]["f1"] == 1.0


def test_bootstrap_intervals_contain_point_and_pairs_are_paired():
    out = bootstrap_metrics({"good": (S_GOOD, T), "noise": (S_NOISE, T), "same": (S_GOOD, T),
                             "det": (S_DET, T)}, Y, GROUPS,
                            paired=["good", "noise", "same", "det"], n_resamples=200)
    point = point_metrics(S_GOOD, Y, OPS, True)["macro"]["average_precision"]
    lo, hi = out["ci"]["good"]["macro_ap"]
    assert lo <= point <= hi
    assert out["paired_macro_ap"]["good - same"]["ci"] == (0.0, 0.0)
    better = out["paired_macro_ap"]["good - noise"]
    assert better["point"] > 0 and better["ci"][0] > 0 and better["share_a_better"] == 1.0
    assert "skipped" in out["paired_macro_ap"]["good - det"]
    assert set(out["ci"]["det"]) == {"ap|CONTACT", "f1|CONTACT", "macro_ap", "macro_f1",
                                     "micro_f1"}
    assert out["groups"] == N // 2


def test_bootstrap_is_deterministic():
    a = bootstrap_metrics({"m": (S_GOOD, T)}, Y, GROUPS, n_resamples=100)
    b = bootstrap_metrics({"m": (S_GOOD, T)}, Y, GROUPS, n_resamples=100)
    assert a == b
