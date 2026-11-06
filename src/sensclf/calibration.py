"""Post-hoc calibration: turn model scores into probabilities that mean what they say.

A score s is *calibrated* when, among all documents that receive s = 0.8, about 80% are
positive. Milestone 2 showed good ranking (high average precision), but ranking says nothing
about calibration, and LinearSVC margins are not probabilities at all.

Every calibrator is fitted per category on the reserved CALIBRATION split only (never on
train, dev or a test set), on a one-dimensional input z:

* probability models (``word_lr``, ``char_lr``): z = logit(p), clipped to avoid infinities;
* margin models (``word_svc``): z = the decision value.

Methods:

* ``raw``          no mapping (probability models only): the baseline;
* ``temperature``  p = sigmoid(z / T) with one parameter T > 0 (Guo et al., 2017). It can only
                   soften or sharpen; it never moves the decision boundary z = 0;
* ``platt``        p = sigmoid(a·z + b), two parameters (Platt, 1999), fitted by maximum
                   likelihood. It can also shift the boundary;
* ``isotonic``     the best non-decreasing step function of z in squared error
                   (Zadrozny & Elkan, 2002). Most flexible; it creates ties, so it can change
                   average precision slightly, and it needs more data than the parametric maps.

Metrics (weighted, so the cluster bootstrap of Milestone 2 can resample them):

* expected calibration error (ECE) with 15 equal-width bins on [0, 1]:
  ECE = sum_b (n_b / n) · |mean(y in b) - mean(p in b)|;
* Brier score mean((p - y)^2) and log loss -mean(y log p + (1-y) log(1-p)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODS = ("raw", "temperature", "platt", "isotonic")
N_BINS = 15
EPS = 1e-12
LOGIT_CLIP = 1e-7


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return expit(np.asarray(z, dtype=float))


def score_to_z(scores: np.ndarray, probabilistic: bool) -> np.ndarray:
    """The 1-D calibration input: logit of a probability, or the margin itself."""
    return to_logit(scores) if probabilistic else np.asarray(scores, dtype=float)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


@dataclass
class BinaryCalibrator:
    """One fitted map z -> probability for one category."""

    method: str
    params: dict = field(default_factory=dict)
    _iso: IsotonicRegression | None = field(default=None, repr=False)

    def fit(self, z: np.ndarray, y: np.ndarray) -> BinaryCalibrator:
        z, y = np.asarray(z, dtype=float), np.asarray(y, dtype=int)
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}")
        if len(set(y.tolist())) < 2:
            raise ValueError("calibration labels contain one class only")
        if self.method == "temperature":
            res = minimize_scalar(lambda logt: _log_loss(y, sigmoid(z / np.exp(logt))),
                                  bounds=(np.log(0.05), np.log(20.0)), method="bounded",
                                  options={"xatol": 1e-6})
            self.params = {"T": float(np.exp(res.x))}
        elif self.method == "platt":
            # Essentially unpenalised maximum likelihood (C very large); one feature.
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z[:, None], y)
            self.params = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        elif self.method == "isotonic":
            self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(z, y)
            self.params = {"steps": len(self._iso.X_thresholds_)}
        return self

    def predict(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.method == "raw":
            return sigmoid(z)
        if self.method == "temperature":
            return sigmoid(z / self.params["T"])
        if self.method == "platt":
            return sigmoid(self.params["a"] * z + self.params["b"])
        if self._iso is None:
            raise RuntimeError("isotonic calibrator is not fitted")
        return self._iso.predict(z)


@dataclass
class Calibrator:
    """Six binary calibrators (one per category column) of one method for one model."""

    method: str
    probabilistic: bool
    heads: list[BinaryCalibrator] = field(default_factory=list)

    def fit(self, S_cal: np.ndarray, Y_cal: np.ndarray) -> Calibrator:
        if self.method == "raw" and not self.probabilistic:
            raise ValueError("'raw' is only defined for probability models")
        Z = score_to_z(S_cal, self.probabilistic)
        self.heads = [BinaryCalibrator(self.method).fit(Z[:, k], Y_cal[:, k])
                      for k in range(Z.shape[1])]
        return self

    def transform(self, S: np.ndarray) -> np.ndarray:
        if not self.heads:
            raise RuntimeError("calibrator is not fitted")
        Z = score_to_z(S, self.probabilistic)
        return np.column_stack([h.predict(Z[:, k]) for k, h in enumerate(self.heads)])


# ---------------------------------------------------------------- weighted metrics
def _w(W: np.ndarray | None, n: int) -> np.ndarray:
    """(k, n) weights; a single row of ones when W is None."""
    return np.ones((1, n)) if W is None else np.atleast_2d(np.asarray(W, dtype=float))


def bin_index(p: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Equal-width bins on [0, 1]; p = 1 falls in the last bin."""
    return np.minimum((np.asarray(p, dtype=float) * n_bins).astype(int), n_bins - 1)


def ece(y: np.ndarray, p: np.ndarray, W: np.ndarray | None = None,
        n_bins: int = N_BINS) -> np.ndarray:
    """Expected calibration error for each weight row of W (shape (k,))."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    Wm = _w(W, len(y))
    b = bin_index(p, n_bins)
    onehot = np.zeros((len(y), n_bins))
    onehot[np.arange(len(y)), b] = 1.0
    nb = Wm @ onehot                       # (k, bins) weighted counts
    sy = Wm @ (onehot * y[:, None])
    sp = Wm @ (onehot * p[:, None])
    total = nb.sum(axis=1)
    gap = np.abs(sy - sp)                  # = n_b · |mean y - mean p|
    return gap.sum(axis=1) / total


def brier(y: np.ndarray, p: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    Wm = _w(W, len(y))
    return (Wm @ (p - y) ** 2) / Wm.sum(axis=1)


def log_loss(y: np.ndarray, p: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    Wm = _w(W, len(y))
    losses = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return (Wm @ losses) / Wm.sum(axis=1)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Per non-empty bin: count, mean predicted probability, observed positive rate."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = bin_index(p, n_bins)
    rows = []
    for i in range(n_bins):
        m = b == i
        if m.any():
            rows.append({"bin": i, "lo": i / n_bins, "hi": (i + 1) / n_bins, "n": int(m.sum()),
                         "mean_p": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return rows
