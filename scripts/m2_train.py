"""Milestone 2: train every model family on train, select C on dev, save the selected model.

Usage:
  python scripts/m2_train.py                 # all families, each in its own process
  python scripts/m2_train.py --family word_lr

Families: word_lr, char_lr, word_svc (C grids in sensclf.models.text) and domain_lr
(C in {0.1, 1, 10}). Selection criterion: macro average precision over the six categories on
DEV only; the calibration split is refused by the guard. Training set: the full train split
(decided before any evaluation, see docs/results.md).

Outputs: models/m2/<family>.joblib + .json (gitignored) and results/m2/selection/<family>.json
(the dev grid, train times and the peak memory of the family's own process).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings

from sklearn.exceptions import ConvergenceWarning

from sensclf.guard import calibration_ids
from sensclf.models import C_GRIDS, FAMILIES, fit_grid, label_matrix, save_model
from sensclf.models.base import doc_ids_sha256
from sensclf.models.baselines import DomainModel
from sensclf.processed import load_set
from sensclf.rawdata import REPO_ROOT, load_manifest
from sensclf.runinfo import peak_memory_mb
from sensclf.selection import select_by_dev

ALL_FAMILIES = (*FAMILIES, "domain_lr")
DOMAIN_GRID = (0.1, 1.0, 10.0)
MODEL_DIR = REPO_ROOT / "models" / "m2"
SELECTION_DIR = REPO_ROOT / "results" / "m2" / "selection"
SEED = 0


def _grid(family: str, train, forbidden) -> tuple[list[tuple], float]:
    """[(C, fitted model, head seconds)] and the vectorizer seconds (0 for domain_lr)."""
    if family == "domain_lr":
        out = []
        for C in DOMAIN_GRID:
            start = time.perf_counter()
            model = DomainModel(C, SEED).fit(train, forbidden_ids=forbidden)
            out.append((C, model, time.perf_counter() - start))
        return out, 0.0
    fits, vec_seconds = fit_grid(family, train, C_GRIDS[family], forbidden_ids=forbidden,
                                 seed=SEED)
    return [(f.model.C, f.model, f.head_seconds) for f in fits], vec_seconds


def train_family(family: str) -> dict:
    forbidden = calibration_ids()
    train, dev = load_set("train"), load_set("dev")
    Y_dev = label_matrix(dev)
    dev_ids = [d.doc_id for d in dev]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        grid, vec_seconds = _grid(family, train, forbidden)
    scores = [(C, model.predict_scores(dev)) for C, model, _ in grid]
    best_C, rows = select_by_dev(scores, Y_dev, set_name="dev", doc_ids=dev_ids,
                                 forbidden=forbidden)
    for row, (_, _, seconds) in zip(rows, grid):
        row["head_seconds"] = seconds
    chosen, chosen_head_seconds = next((m, s) for C, m, s in grid if C == best_C)
    report = {
        "family": family,
        "grid": rows,
        "selected_C": best_C,
        "criterion": "macro average precision over the six categories on dev",
        "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        "vectorizer_seconds": vec_seconds,
        "grid_train_seconds": vec_seconds + sum(s for _, _, s in grid),
        "selected_model_train_seconds": vec_seconds + chosen_head_seconds,
        "training_set": {
            "name": "train", "documents": len(train), "doc_ids_sha256": doc_ids_sha256(train),
            "processed_sha256": load_manifest()["processed_sets"]["train"]["sha256"],
        },
        "seed": SEED,
        "peak_memory_mb": peak_memory_mb(),
    }
    info = {k: report[k] for k in ("family", "training_set", "seed", "peak_memory_mb")}
    info.update(params={"C": best_C}, train_time_s=report["selected_model_train_seconds"])
    save_model(chosen, MODEL_DIR / f"{family}.joblib", info)
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    (SELECTION_DIR / f"{family}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=ALL_FAMILIES)
    args = parser.parse_args(argv)
    if args.family:
        r = train_family(args.family)
        print(f"{r['family']}: C={r['selected_C']}  grid "
              + ", ".join(f"C={row['C']}: {row['dev_macro_ap']:.4f}" for row in r["grid"])
              + f"  ({r['grid_train_seconds']:.0f} s, peak {r['peak_memory_mb']:.0f} MB, "
              f"{r['convergence_warnings']} convergence warnings)", flush=True)
        return 0
    for family in ALL_FAMILIES:   # one process per family: peak memory is per family
        subprocess.run([sys.executable, __file__, "--family", family], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
