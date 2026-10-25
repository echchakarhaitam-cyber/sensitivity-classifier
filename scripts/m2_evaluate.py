"""Milestone 2: score dev, test_shift and test_ood with every model; metrics with CIs.

Usage:  python scripts/m2_evaluate.py      (after scripts/m2_train.py)

Every set is loaded through its manifest hash. Operating points come from DEV scores only
(sensclf.selection). Writes results/m2/predictions_<model>_<set>.jsonl (gitignored) and
results/m2/metrics.json (committed).
"""

from __future__ import annotations

import json
import sys
import time

from sensclf.evaluation import bootstrap_metrics, point_metrics
from sensclf.guard import calibration_ids, ensure_no_calibration
from sensclf.models import label_matrix, load_model, load_sidecar
from sensclf.models.baselines import DetectorBaseline
from sensclf.predictions import write_predictions
from sensclf.processed import load_set
from sensclf.rawdata import REPO_ROOT, load_manifest
from sensclf.selection import operating_points

SETS = ("dev", "test_shift", "test_ood")
TRAINED = ("word_lr", "char_lr", "word_svc", "domain_lr")
MODELS = (*TRAINED, "regex_detector")
PAIRED = TRAINED            # same evaluable categories; the detector covers only two
MODEL_DIR = REPO_ROOT / "models" / "m2"
OUT = REPO_ROOT / "results" / "m2"


def groups_of(docs) -> list[str]:
    return [str(d.meta.get("group_id") or d.doc_id) for d in docs]


def load_models() -> tuple[dict, dict]:
    models, info = {}, {}
    for name in TRAINED:
        models[name] = load_model(MODEL_DIR / f"{name}.joblib")
        sidecar = load_sidecar(MODEL_DIR / f"{name}.joblib")
        info[name] = {"probabilistic": models[name].probabilistic, "params": sidecar["params"],
                      "training_set": sidecar["training_set"]}
    models["regex_detector"] = DetectorBaseline()
    info["regex_detector"] = {"probabilistic": False, "params": {},
                              "covered": list(DetectorBaseline.covered)}
    return models, info


def main() -> int:
    manifest = load_manifest()
    forbidden = calibration_ids()
    sets = {name: load_set(name, manifest=manifest) for name in SETS}
    for docs in sets.values():
        ensure_no_calibration((d.doc_id for d in docs), forbidden)
    models, info = load_models()
    scores: dict = {name: {} for name in MODELS}
    for name, model in models.items():
        for set_name, docs in sets.items():
            start = time.perf_counter()
            S = model.predict_scores(docs)
            scores[name][set_name] = S
            write_predictions(OUT / f"predictions_{name}_{set_name}.jsonl",
                              [d.doc_id for d in docs], S)
            print(f"scored {name} on {set_name} in {time.perf_counter() - start:.0f} s",
                  flush=True)
    dev = sets["dev"]
    Y_dev, dev_ids = label_matrix(dev), [d.doc_id for d in dev]
    ops = {name: operating_points(model, scores[name]["dev"], Y_dev, set_name="dev",
                                  doc_ids=dev_ids, forbidden=forbidden)
           for name, model in models.items()}
    results = {"set_sha256": {s: manifest["processed_sets"][s]["sha256"] for s in SETS},
               "models": info, "operating_points": ops, "sets": {}}
    for set_name, docs in sets.items():
        Y = label_matrix(docs)
        point = {name: point_metrics(scores[name][set_name], Y, ops[name],
                                     info[name]["probabilistic"]) for name in MODELS}
        start = time.perf_counter()
        boot = bootstrap_metrics({name: (scores[name][set_name], ops[name]["dev_f1"])
                                  for name in MODELS}, Y, groups_of(docs), paired=PAIRED)
        print(f"bootstrap {set_name} in {time.perf_counter() - start:.0f} s", flush=True)
        results["sets"][set_name] = {"documents": len(docs), "models": point, "bootstrap": boot}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
