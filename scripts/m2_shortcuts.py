"""Milestone 2, RQ2: per-domain performance and annotated top features of word_lr.

Usage:  python scripts/m2_shortcuts.py     (after scripts/m2_evaluate.py)

Reads the evaluated predictions (results/m2/predictions_*.jsonl), the verified sets and the
saved word_lr model; writes results/m2/shortcuts.json. The domain-only comparison is built
from results/m2/metrics.json by scripts/make_results.py.
"""

from __future__ import annotations

import json
import sys

from sensclf.models import label_matrix, load_model
from sensclf.models.baselines import domain_of
from sensclf.predictions import read_predictions
from sensclf.processed import load_set
from sensclf.rawdata import REPO_ROOT
from sensclf.shortcuts import (
    DOMAIN_U_THRESHOLD,
    FIELD_WORDS,
    per_domain_macro_ap,
    top_features,
)

OUT = REPO_ROOT / "results" / "m2"
MODELS = ("word_lr", "char_lr", "word_svc", "domain_lr")
SETS = ("dev", "test_shift")
TOP_K = 25


def main() -> int:
    train = load_set("train")
    seen = frozenset(domain_of(d) for d in train)
    per_domain = {}
    for set_name in SETS:
        docs = load_set(set_name)
        ids, Y = [d.doc_id for d in docs], label_matrix(docs)
        domains = [str(d.meta.get("domain", "")) for d in docs]
        per_domain[set_name] = {}
        for name in MODELS:
            S = read_predictions(OUT / f"predictions_{name}_{set_name}.jsonl", ids)
            per_domain[set_name][name] = per_domain_macro_ap(S, Y, domains, seen)
    model = load_model(REPO_ROOT / "models" / "m2" / "word_lr.joblib")
    features = top_features(model, model.transform(train),
                            [str(d.meta.get("domain", "")) for d in train], k=TOP_K)
    result = {
        "per_domain": per_domain,
        "train_domains": sorted(seen),
        "top_features": features,
        "rule": {"domain_u_threshold": DOMAIN_U_THRESHOLD, "field_words": sorted(FIELD_WORDS),
                 "statistic": "U = MI(F; D) / H(F) on train documents (F = feature occurs)"},
    }
    (OUT / "shortcuts.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'shortcuts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
