import numpy as np
import pytest

from sensclf.calibration import (
    BinaryCalibrator,
    Calibrator,
    bin_index,
    brier,
    ece,
    log_loss,
    reliability_curve,
    score_to_z,
    sigmoid,
    to_logit,
)


def test_logit_sigmoid_roundtrip():
    p = np.array([0.01, 0.2, 0.5, 0.9, 0.999])
    assert np.allclose(sigmoid(to_logit(p)), p)
    assert np.isfinite(to_logit(np.array([0.0, 1.0]))).all()   # clipped, no infinities


def test_ece_hand_example():
    # 4 docs: two in bin [0.0,1/15) with p=0 (one positive), two in last bin with p=1 (one pos)
    y = np.array([0, 1, 1, 0])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    # each bin: |0.5 - 0| = 0.5 and |0.5 - 1| = 0.5, weights 1/2 each -> ECE 0.5
    assert ece(y, p)[0] == pytest.approx(0.5)
    assert brier(y, p)[0] == pytest.approx(0.5)


def test_perfectly_calibrated_bins_have_zero_ece():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert ece(y, p)[0] == pytest.approx(0.0)
    assert log_loss(y, p)[0] == pytest.approx(np.log(2))


def test_weighted_metrics_match_duplication():
    y = np.array([1, 0, 1])
    p = np.array([0.9, 0.3, 0.4])
    W = np.array([[2.0, 1.0, 0.0]])
    dup_y, dup_p = np.array([1, 1, 0]), np.array([0.9, 0.9, 0.3])
    assert ece(y, p, W)[0] == pytest.approx(ece(dup_y, dup_p)[0])
    assert brier(y, p, W)[0] == pytest.approx(brier(dup_y, dup_p)[0])
    assert log_loss(y, p, W)[0] == pytest.approx(log_loss(dup_y, dup_p)[0])


def test_bin_index_edges():
    assert bin_index(np.array([0.0, 1.0, 0.999999]), 15).tolist() == [0, 14, 14]


def _synthetic(T_true: float, n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    z_true = rng.normal(0, 2, n)
    y = (rng.random(n) < sigmoid(z_true)).astype(int)
    return z_true * T_true, y      # overconfident scores when T_true > 1


def test_temperature_recovers_known_temperature():
    z, y = _synthetic(3.0)
    cal = BinaryCalibrator("temperature").fit(z, y)
    assert cal.params["T"] == pytest.approx(3.0, rel=0.08)


def test_platt_recovers_slope_and_reduces_ece():
    z, y = _synthetic(2.0)
    cal = BinaryCalibrator("platt").fit(z, y)
    assert cal.params["a"] == pytest.approx(0.5, rel=0.08)
    assert ece(y, cal.predict(z))[0] < ece(y, sigmoid(z))[0]


def test_isotonic_is_monotone_and_bounded():
    z, y = _synthetic(2.0, n=3000)
    cal = BinaryCalibrator("isotonic").fit(z, y)
    grid = np.linspace(-20, 20, 200)
    out = cal.predict(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0 and out.max() <= 1


def test_one_class_and_unknown_method_raise():
    with pytest.raises(ValueError):
        BinaryCalibrator("platt").fit(np.array([0.1, 0.2]), np.array([1, 1]))
    with pytest.raises(ValueError):
        BinaryCalibrator("magic").fit(np.array([0.1, 0.2]), np.array([0, 1]))


def test_calibrator_matrix_and_raw_rules():
    rng = np.random.default_rng(1)
    S = rng.random((500, 6))
    Y = (rng.random((500, 6)) < S).astype(int)
    cal = Calibrator("platt", probabilistic=True).fit(S, Y)
    assert cal.transform(S).shape == (500, 6)
    raw = Calibrator("raw", probabilistic=True).fit(S, Y)
    assert np.allclose(raw.transform(S), np.clip(S, 1e-7, 1 - 1e-7))
    with pytest.raises(ValueError):
        Calibrator("raw", probabilistic=False).fit(S, Y)
    with pytest.raises(RuntimeError):
        Calibrator("platt", probabilistic=True).transform(S)


def test_margin_input_is_unchanged():
    m = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(score_to_z(m, probabilistic=False), m)


def test_reliability_curve_counts():
    y = np.array([0, 1, 1, 1])
    p = np.array([0.05, 0.05, 0.95, 0.95])
    rows = reliability_curve(y, p)
    assert [r["n"] for r in rows] == [2, 2]
    assert rows[0]["frac_pos"] == 0.5 and rows[1]["frac_pos"] == 1.0
