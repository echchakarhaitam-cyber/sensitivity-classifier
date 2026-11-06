"""Milestone 3: calibrate the Milestone 2 models on the calibration split and evaluate.

Usage:  python scripts/m3_calibrate.py

1. Score the CALIBRATION split with the frozen Milestone 2 models (the first time this split is
   touched by a model; it was refused to every training and selection step before).
2. Fit, per model and per category, the calibrators of sensclf.calibration on the cal_fit half
   of the split only (sensclf.calsplit); the cal_test half is kept for Milestones 4-5.
3. Apply them to the stored dev / test_shift / test_ood scores of Milestone 2 and measure ECE,
   Brier score and log loss with 95% cluster-bootstrap CIs (macro over evaluable categories).
4. Choose ONE calibrator per model for Milestones 4-5 by a rule fixed here, before any test-set
   number is looked at: the method with the lowest macro Brier score on DEV.

Writes results/m3/calibration.json (committed), results/m3/calibrators.json (fitted
parameters), results/m3/predictions_<model>_calibration.jsonl (gitignored raw scores) and
results/m3/predictions_<model>_<method>_<set>.jsonl (gitignored calibrated probabilities).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from sensclf.bootstrap import cluster_bootstrap, interval
from sensclf.calibration import (
    METHODS,
    Calibrator,
    brier,
    ece,
    log_loss,
    reliability_curve,
)
from sensclf.calsplit import split_calibration
from sensclf.labels import CATEGORIES
from sensclf.metrics import is_defined
from sensclf.models import label_matrix, load_model
from sensclf.predictions import read_predictions, write_predictions
from sensclf.processed import load_set
from sensclf.rawdata import REPO_ROOT, load_manifest

MODELS = ("word_lr", "char_lr", "word_svc")
EVAL_SETS = ("dev", "test_shift", "test_ood")
SELECTION_SET = "dev"
SELECTION_METRIC = "brier"
M2 = REPO_ROOT / "results" / "m2"
OUT = REPO_ROOT / "results" / "m3"
MODEL_DIR = REPO_ROOT / "models" / "m2"
N_RESAMPLES = 1000


def groups_of(docs) -> list[str]:
    return [str(d.meta.get("group_id") or d.doc_id) for d in docs]


def methods_for(model) -> tuple[str, ...]:
    return METHODS if model.probabilistic else tuple(m for m in METHODS if m != "raw")


def evaluate(P: np.ndarray, Y: np.ndarray, groups: list[str]) -> dict:
    cols = [k for k in range(len(CATEGORIES)) if is_defined(Y[:, k])]
    per_cat = {CATEGORIES[k]: {"ece": float(ece(Y[:, k], P[:, k])[0]),
                               "brier": float(brier(Y[:, k], P[:, k])[0]),
                               "log_loss": float(log_loss(Y[:, k], P[:, k])[0]),
                               "mean_p": float(P[:, k].mean()),
                               "positive_rate": float(Y[:, k].mean())}
               for k in range(len(CATEGORIES))}

    def macro(fn):
        return lambda W: np.mean([fn(Y[:, k], P[:, k], W) for k in cols], axis=0)

    stats = {"ece": macro(ece), "brier": macro(brier), "log_loss": macro(log_loss)}
    draws = cluster_bootstrap(groups, stats, n_resamples=N_RESAMPLES, seed=0)
    ones = np.ones((1, len(Y)))
    out = {"categories": [CATEGORIES[k] for k in cols],
           "excluded": [c for k, c in enumerate(CATEGORIES) if k not in cols],
           "per_category": per_cat}
    for name, fn in stats.items():
        out[name] = float(fn(ones)[0])
        out[f"{name}_ci"] = interval(draws[name])
    pooled_y = np.concatenate([Y[:, k] for k in cols])
    pooled_p = np.concatenate([P[:, k] for k in cols])
    out["reliability_pooled"] = reliability_curve(pooled_y, pooled_p)
    return out


def main() -> int:
    manifest = load_manifest()
    cal_docs = load_set("calibration", manifest=manifest)
    cal_fit, cal_test = split_calibration(cal_docs)
    fit_ids = {d.doc_id for d in cal_fit}
    sets = {name: load_set(name, manifest=manifest) for name in EVAL_SETS}
    sets["cal_test"] = cal_test
    results = {"set_sha256": {s: manifest["processed_sets"][s]["sha256"]
                              for s in ("calibration", *EVAL_SETS)},
               "selection_rule": f"lowest macro {SELECTION_METRIC} on {SELECTION_SET}",
               "n_resamples": N_RESAMPLES, "models": {},
               "calibration_halves": {"cal_fit": len(cal_fit), "cal_test": len(cal_test),
                                      "rule": "group hash bucket 10-14 -> cal_fit, 15-19 -> "
                                              "cal_test"}}
    params: dict = {}
    for name in MODELS:
        model = load_model(MODEL_DIR / f"{name}.joblib")
        start = time.perf_counter()
        S_cal = model.predict_scores(cal_docs)
        write_predictions(OUT / f"predictions_{name}_calibration.jsonl",
                          [d.doc_id for d in cal_docs], S_cal)
        print(f"scored {name} on calibration in {time.perf_counter() - start:.0f} s",
              flush=True)
        cal_ids = [d.doc_id for d in cal_docs]
        mask = np.array([i in fit_ids for i in cal_ids])
        S_fit, Y_fit = S_cal[mask], label_matrix(cal_docs)[mask]
        S_by_set = {"cal_test": S_cal[~mask]}
        if [d.doc_id for d in cal_test] != [i for i in cal_ids if i not in fit_ids]:
            raise RuntimeError("cal_test order differs from the calibration split order")
        entry: dict = {"probabilistic": model.probabilistic, "methods": {}}
        params[name] = {}
        for method in methods_for(model):
            cal = Calibrator(method, model.probabilistic).fit(S_fit, Y_fit)
            params[name][method] = {c: h.params for c, h in zip(CATEGORIES, cal.heads)}
            entry["methods"][method] = {}
            for set_name, docs in sets.items():
                ids = [d.doc_id for d in docs]
                S = S_by_set.get(set_name)
                if S is None:
                    S = read_predictions(M2 / f"predictions_{name}_{set_name}.jsonl", ids)
                P = cal.transform(S)
                write_predictions(OUT / f"predictions_{name}_{method}_{set_name}.jsonl", ids, P)
                entry["methods"][method][set_name] = evaluate(P, label_matrix(docs),
                                                              groups_of(docs))
            print(f"  {name}/{method} done", flush=True)
        dev_scores = {m: v[SELECTION_SET][SELECTION_METRIC] for m, v in entry["methods"].items()}
        entry["selected"] = min(dev_scores, key=lambda m: (dev_scores[m], m))
        entry["selection_scores"] = dev_scores
        results["models"][name] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8", newline="\n")
    (OUT / "calibrators.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8", newline="\n")
    print("selected:", {m: results["models"][m]["selected"] for m in MODELS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
