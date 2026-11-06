"""Figures for docs/results-uncertainty.md.

Usage:  python scripts/m3_m4_figures.py      (after m3_calibrate and m4_m5_uncertainty)

* docs/figures/m3_reliability.png -- pooled reliability diagrams (all six categories) of the
  carried-forward model, raw vs selected calibrator, on dev / test_shift / test_ood; read from
  results/m3/calibration.json.
* docs/figures/m4_risk_coverage.png -- per-category risk-coverage curves of the carried-forward
  model on dev and test_ood; read from the (gitignored) calibrated predictions in results/m3.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sensclf.labels import CATEGORIES
from sensclf.models import label_matrix
from sensclf.predictions import read_predictions
from sensclf.processed import load_set
from sensclf.rawdata import REPO_ROOT, load_manifest
from sensclf.selective import decisions, errors, risk_coverage

FIG = REPO_ROOT / "docs" / "figures"
SETS = ("dev", "test_shift", "test_ood")


def reliability(cal: dict, model: str, method: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, s in zip(axes, SETS, strict=True):
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect")
        methods = cal["models"][model]["methods"]
        base = "raw" if "raw" in methods else "temperature"   # SVC scores are not probabilities
        for meth, color in ((base, "#c0504d"), (method, "#1f77b4")):
            r = cal["models"][model]["methods"][meth][s]
            bins = [b for b in r["reliability_pooled"] if b["n"] > 0]
            ax.plot([b["mean_p"] for b in bins], [b["frac_pos"] for b in bins], "o-",
                    color=color, ms=4, label=f"{meth} (ECE {r['ece']:.3f})")
        ax.set_title(s)
        ax.set_xlabel("mean predicted probability")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_ylabel("observed positive rate")
    fig.suptitle(f"Reliability, {model}, six categories pooled (15 bins)")
    fig.tight_layout()
    fig.savefig(FIG / "m3_reliability.png", dpi=130)
    plt.close(fig)


def risk_coverage_fig(model: str, method: str) -> None:
    manifest = load_manifest()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, s in zip(axes, ("dev", "test_ood"), strict=True):
        docs = load_set(s, manifest=manifest)
        P = read_predictions(REPO_ROOT / "results" / "m3" / f"predictions_{model}_{method}_{s}"
                             ".jsonl", [d.doc_id for d in docs])
        Y = label_matrix(docs)
        for k, c in enumerate(CATEGORIES):
            _, conf = decisions(P[:, k])
            curve = risk_coverage(conf, errors(P[:, k], Y[:, k]))
            ax.plot(curve.coverage, curve.risk, lw=1.4, label=c)
        ax.set_title(f"{s} ({method})")
        ax.set_xlabel("coverage (share decided automatically)")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("selective risk (error rate among accepted)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Risk-coverage curves, {model}")
    fig.tight_layout()
    fig.savefig(FIG / "m4_risk_coverage.png", dpi=130)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    cal = json.loads((REPO_ROOT / "results" / "m3" / "calibration.json").read_text("utf-8"))
    sel = json.loads((REPO_ROOT / "results" / "m4" / "selective.json").read_text("utf-8"))
    model, method = sel["model"], sel["calibration"]
    reliability(cal, model, method)
    risk_coverage_fig(model, method)
    print("wrote docs/figures/m3_reliability.png and m4_risk_coverage.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
