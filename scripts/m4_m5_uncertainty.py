"""Milestones 4-5: abstention (selective prediction) and conformal prediction sets.

Usage:  python scripts/m4_m5_uncertainty.py      (after scripts/m3_calibrate.py)

Protocol fixed before any result of this script was seen:

* Model: the Milestone 3 model whose selected calibrator has the lowest macro Brier score on
  DEV; its calibrated probabilities are used everywhere below.
* Held-out data for thresholds and quantiles: the cal_test half of the calibration split
  (never used to fit anything before). Conformal quantiles use one document per group
  (seed 0).
* Milestone 4: per category, risk-coverage curves and AURC on every set; thresholds certified
  on cal_test for target selective risks r* in {1%, 2%, 5%} with delta = 5%; then coverage and
  achieved risk on dev (in-domain), test_shift and test_ood. A document is decided
  automatically when every category is accepted.
* Milestone 5: split conformal per category, marginal and Mondrian (per true class), for
  alpha in {1%, 5%, 10%}, plus the joint 6-label version (alpha / 6 per category); coverage,
  class-conditional coverage and set sizes on every set; the spread of in-domain coverage over
  100 different one-per-group draws of the calibration documents.

Writes results/m4/selective.json and results/m5/conformal.json (committed).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from sensclf.bootstrap import cluster_bootstrap, interval
from sensclf.calsplit import split_calibration
from sensclf.conformal import (
    ALPHAS,
    coverage_bounds,
    fit_quantiles,
    one_per_group,
    prediction_sets,
    set_statistics,
)
from sensclf.labels import CATEGORIES
from sensclf.metrics import is_defined
from sensclf.models import label_matrix
from sensclf.predictions import read_predictions
from sensclf.processed import load_set
from sensclf.rawdata import REPO_ROOT, load_manifest
from sensclf.selective import (
    COVERAGE_POINTS,
    DELTA,
    RISK_TARGETS,
    apply_threshold,
    aurc,
    choose_threshold,
    decisions,
    errors,
    oracle_aurc,
    risk_at_coverage,
)

M3 = REPO_ROOT / "results" / "m3"
OUT4 = REPO_ROOT / "results" / "m4"
OUT5 = REPO_ROOT / "results" / "m5"
EVAL_SETS = ("dev", "test_shift", "test_ood")
N_RESAMPLES = 1000
N_DRAWS = 100


def groups_of(docs) -> list[str]:
    return [str(d.meta.get("group_id") or d.doc_id) for d in docs]


def pick_model(calibration: dict) -> tuple[str, str]:
    best = min(calibration["models"].items(),
               key=lambda kv: (kv[1]["selection_scores"][kv[1]["selected"]], kv[0]))
    return best[0], best[1]["selected"]


def risk_ci(err: np.ndarray, acc: np.ndarray, groups: list[str]) -> list[float] | None:
    if not acc.any():
        return None

    def stat(W):
        a = W @ acc.astype(float)
        return np.where(a > 0, (W @ (err * acc)) / np.maximum(a, 1e-12), np.nan)

    return interval(cluster_bootstrap(groups, {"r": stat}, N_RESAMPLES, seed=0)["r"])


def selective(P: dict, Y: dict, groups: dict) -> dict:
    out: dict = {"targets": list(RISK_TARGETS), "delta": DELTA, "categories": {}}
    for k, c in enumerate(CATEGORIES):
        _, conf_cal = decisions(P["cal_test"][:, k])
        err_cal = errors(P["cal_test"][:, k], Y["cal_test"][:, k])
        entry: dict = {"sets": {}, "thresholds": {}}
        for s in EVAL_SETS:
            _, conf = decisions(P[s][:, k])
            err = errors(P[s][:, k], Y[s][:, k])
            entry["sets"][s] = {
                "defined": bool(is_defined(Y[s][:, k])),
                "error_rate_full_coverage": float(err.mean()),
                "aurc": aurc(conf, err), "oracle_aurc": oracle_aurc(err),
                "at_coverage": {str(t): risk_at_coverage(conf, err, t) for t in COVERAGE_POINTS},
            }
        for r in RISK_TARGETS:
            choice = choose_threshold(conf_cal, err_cal, r, DELTA)
            applied = {}
            for s in EVAL_SETS:
                _, conf = decisions(P[s][:, k])
                err = errors(P[s][:, k], Y[s][:, k])
                res = apply_threshold(conf, err, choice["threshold"])
                acc = (conf >= choice["threshold"]) if choice["threshold"] is not None \
                    else np.zeros(len(conf), bool)
                res["risk_ci"] = risk_ci(err, acc, groups[s])
                res["violation"] = res["risk"] is not None and res["risk"] > r
                applied[s] = res
            entry["thresholds"][str(r)] = {"choice": choice, "applied": applied}
        out["categories"][c] = entry
    # document level: decided automatically iff every category is accepted
    out["document_level"] = {}
    for r in RISK_TARGETS:
        thr = [out["categories"][c]["thresholds"][str(r)]["choice"]["threshold"]
               for c in CATEGORIES]
        rows = {}
        for s in EVAL_SETS:
            acc_all = np.ones(len(Y[s]), bool)
            wrong_any = np.zeros(len(Y[s]), bool)
            for k, t in enumerate(thr):
                _, conf = decisions(P[s][:, k])
                acc_all &= (conf >= t) if t is not None else False
                wrong_any |= errors(P[s][:, k], Y[s][:, k]).astype(bool)
            n = int(acc_all.sum())
            rows[s] = {"coverage": float(acc_all.mean()), "accepted": n,
                       "doc_error_rate": float(wrong_any[acc_all].mean()) if n else None}
        out["document_level"][str(r)] = rows
    return out


def conformal(P: dict, Y: dict, cal_groups: list[str]) -> dict:
    idx = one_per_group(cal_groups, seed=0)
    out: dict = {"alphas": list(ALPHAS), "n_cal": len(idx), "variants": {}}
    for mondrian in (False, True):
        name = "mondrian" if mondrian else "marginal"
        var: dict = {}
        for alpha in ALPHAS:
            per_cat, joint_sets = {}, {s: [] for s in EVAL_SETS}
            for k, c in enumerate(CATEGORIES):
                q = fit_quantiles(P["cal_test"][idx, k], Y["cal_test"][idx, k], alpha, mondrian)
                qj = fit_quantiles(P["cal_test"][idx, k], Y["cal_test"][idx, k], alpha / 6,
                                   mondrian)
                per_cat[c] = {"q": {str(y): (None if not np.isfinite(v) else v)
                                    for y, v in q.items()},
                              "sets": {s: set_statistics(prediction_sets(P[s][:, k], q),
                                                         Y[s][:, k]) for s in EVAL_SETS}}
                for s in EVAL_SETS:
                    joint_sets[s].append(prediction_sets(P[s][:, k], qj))
            joint = {}
            for s in EVAL_SETS:
                sets = np.stack(joint_sets[s], axis=1)              # (n, 6, 2)
                y = Y[s]
                covered = np.take_along_axis(sets, y[:, :, None], axis=2)[:, :, 0]
                singletons = sets.sum(axis=2) == 1
                joint[s] = {"joint_coverage": float(covered.all(axis=1).mean()),
                            "all_singleton": float(singletons.all(axis=1).mean())}
            lo, hi = coverage_bounds(len(idx), alpha)
            var[str(alpha)] = {"per_category": per_cat, "joint": joint,
                               "theory": {"lower": lo, "upper": hi}}
        out["variants"][name] = var
    # spread of in-domain coverage over different calibration draws (marginal, alpha 0.1)
    spread = {}
    for k, c in enumerate(CATEGORIES):
        covs = []
        for seed in range(N_DRAWS):
            ix = one_per_group(cal_groups, seed=seed)
            q = fit_quantiles(P["cal_test"][ix, k], Y["cal_test"][ix, k], 0.1)
            covs.append(set_statistics(prediction_sets(P["dev"][:, k], q), Y["dev"][:, k])
                        ["coverage"])
        spread[c] = {"min": float(np.min(covs)), "mean": float(np.mean(covs)),
                     "max": float(np.max(covs))}
    out["dev_coverage_spread_alpha_0.1"] = {"draws": N_DRAWS, "per_category": spread}
    return out


def main() -> int:
    manifest = load_manifest()
    calibration = json.loads((M3 / "calibration.json").read_text(encoding="utf-8"))
    model, method = pick_model(calibration)
    print(f"model {model} with {method} calibration", flush=True)
    _, cal_test = split_calibration(load_set("calibration", manifest=manifest))
    docs = {s: load_set(s, manifest=manifest) for s in EVAL_SETS}
    docs["cal_test"] = cal_test
    P, Y, G = {}, {}, {}
    for s, ds in docs.items():
        ids = [d.doc_id for d in ds]
        P[s] = read_predictions(M3 / f"predictions_{model}_{method}_{s}.jsonl", ids)
        Y[s] = label_matrix(ds)
        G[s] = groups_of(ds)
    header = {"model": model, "calibration": method,
              "set_sha256": {s: manifest["processed_sets"][s]["sha256"]
                             for s in ("calibration", *EVAL_SETS)},
              "documents": {s: len(ds) for s, ds in docs.items()}}
    sel = {**header, **selective(P, Y, G)}
    OUT4.mkdir(parents=True, exist_ok=True)
    (OUT4 / "selective.json").write_text(json.dumps(sel, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8", newline="\n")
    print("wrote results/m4/selective.json", flush=True)
    conf = {**header, **conformal(P, Y, G["cal_test"])}
    OUT5.mkdir(parents=True, exist_ok=True)
    (OUT5 / "conformal.json").write_text(json.dumps(conf, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8", newline="\n")
    print("wrote results/m5/conformal.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
