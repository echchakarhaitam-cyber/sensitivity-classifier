"""Export the selected model (word_svc + isotonic) for the NumPy-only demo (sensclf.portable).

Needs models/m2/word_svc.joblib, data/processed/calibration.jsonl and
results/m3/predictions_word_svc_calibration.jsonl (all produced by the milestone scripts and
gitignored). The isotonic maps are refitted exactly as scripts/m3_calibrate.py fits them (same
cal_fit half, same scores), then checked against the committed-run predictions on cal_test.

Writes demo/model.npz and demo/rules.json.
Usage:  python scripts/export_demo_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sensclf.calibration import Calibrator
from sensclf.calsplit import split_calibration
from sensclf.labels import CATEGORIES
from sensclf.models.base import label_matrix, load_model
from sensclf.portable import load_rules
from sensclf.processed import load_set
from sensclf.rawdata import load_manifest

ROOT = Path(__file__).resolve().parents[1]
M3 = ROOT / "results" / "m3"
OUT = ROOT / "demo"


def read_scores(path: Path) -> dict[str, list[float]]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {r["doc_id"]: r["scores"] for r in rows}


def main() -> int:
    model = load_model(ROOT / "models" / "m2" / "word_svc.joblib")
    cal_docs = load_set("calibration", manifest=load_manifest())
    cal_fit, cal_test = split_calibration(cal_docs)
    scores = read_scores(M3 / "predictions_word_svc_calibration.jsonl")
    S_fit = np.array([scores[d.doc_id] for d in cal_fit])
    iso = Calibrator("isotonic", probabilistic=False).fit(S_fit, label_matrix(cal_fit))

    # check: the refitted maps reproduce the committed run's cal_test probabilities
    expected = read_scores(M3 / "predictions_word_svc_isotonic_cal_test.jsonl")
    S_test = np.array([scores[d.doc_id] for d in cal_test])
    P = iso.transform(S_test)
    E = np.array([expected[d.doc_id] for d in cal_test])
    if not np.allclose(P, E, atol=1e-12):
        print("refitted isotonic maps differ from the committed run", file=sys.stderr)
        return 1

    vec = model.vectorizer
    terms = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    arrays = {"terms": np.frombuffer("\n".join(terms).encode("utf-8"), dtype=np.uint8),
              "idf": vec.idf_.astype(np.float32),
              "coef": np.vstack([h.coef_[0] for h in model.heads]).astype(np.float32),
              "intercept": np.array([h.intercept_[0] for h in model.heads], dtype=np.float64)}
    for k, head in enumerate(iso.heads):
        arrays[f"iso_x_{k}"] = head._iso.X_thresholds_.astype(np.float64)
        arrays[f"iso_y_{k}"] = head._iso.y_thresholds_.astype(np.float64)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "model.npz", **arrays)
    rules = load_rules(json.loads((ROOT / "results" / "m4" / "selective.json").read_text("utf-8")),
                       json.loads((ROOT / "results" / "m5" / "conformal.json").read_text("utf-8")))
    (OUT / "rules.json").write_text(json.dumps(rules, indent=1, sort_keys=True) + "\n",
                                    encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'model.npz'} ({(OUT / 'model.npz').stat().st_size} bytes), "
          f"{len(terms)} terms, categories {list(CATEGORIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
