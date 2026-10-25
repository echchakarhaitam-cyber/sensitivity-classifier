import numpy as np
import pytest
from toy import toy_docs

from sensclf import report as R
from sensclf.evaluation import bootstrap_metrics, point_metrics
from sensclf.labels import CATEGORIES
from sensclf.models import label_matrix


def test_compare_sentence_direction_and_equality():
    assert R.compare_sentence("a", 0.9, "b", 0.8, "macro PR-AUC") == \
        "`a` has a higher macro PR-AUC than `b` (0.900 vs 0.800)"
    assert "lower" in R.compare_sentence("a", 0.1, "b", 0.8, "x")
    assert "the same" in R.compare_sentence("a", 0.8001, "b", 0.8004, "x")   # equal displayed


def test_ci_sentence():
    assert R.ci_sentence((0.01, 0.02)).endswith("excludes 0")
    assert R.ci_sentence((-0.01, 0.02)).endswith("includes 0")
    assert R.ci_sentence((-0.03, -0.02)).endswith("excludes 0")
    assert R.ci_sentence(None) == "no interval could be computed"


def test_wrong_comparisons_detects_contradictions():
    good = R.compare_sentence("a", 0.9, "b", 0.8, "m") + "; " + R.ci_sentence((0.1, 0.2))
    assert R.wrong_comparisons(good) == []
    bad = "`a` has a lower m than `b` (0.900 vs 0.800); [0.100, 0.200] includes 0"
    assert len(R.wrong_comparisons(bad)) == 2


def _synthetic_inputs():
    """Result dicts with the real structure, built with the real metric functions."""
    rng = np.random.default_rng(0)
    metrics_sets = {}
    for name, seed in (("dev", 1), ("test_shift", 2), ("test_ood", 3)):
        docs = toy_docs(40, seed=seed, prefix=name)
        Y = label_matrix(docs)
        if name == "test_ood":
            Y[:, 3] = 0                                  # HEALTH undefined, as in Gretel
        scores = {m: np.clip(Y * 0.5 + rng.random(Y.shape) * 0.5, 0, 1) for m in R.TRAINED}
        det = np.full(Y.shape, np.nan)
        det[:, 1:3] = Y[:, 1:3]
        scores["regex_detector"] = det
        t = dict.fromkeys(CATEGORIES, 0.5)
        ops = {"default": t, "dev_f1": t}
        metrics_sets[name] = {
            "documents": len(docs),
            "models": {m: point_metrics(S, Y, ops, m != "regex_detector")
                       for m, S in scores.items()},
            "bootstrap": bootstrap_metrics({m: (S, t) for m, S in scores.items()}, Y,
                                           [d.meta["group_id"] for d in docs],
                                           paired=R.TRAINED, n_resamples=50),
        }
    t = dict.fromkeys(CATEGORIES, 0.5)
    metrics = {"sets": metrics_sets,
               "operating_points": {m: {"default": t, "dev_f1": t} for m in R.ALL_MODELS}}
    selection = {m: {"grid": [{"C": 0.1, "dev_macro_ap": 0.8}, {"C": 1.0, "dev_macro_ap": 0.9}],
                     "selected_C": 1.0, "grid_train_seconds": 60.0,
                     "selected_model_train_seconds": 30.0, "peak_memory_mb": 100.0,
                     "convergence_warnings": 0, "training_set": {"documents": 40}}
                 for m in R.TRAINED}
    rows = [{"domain": "Finance", "documents": 10, "macro_ap": 0.9, "categories_defined": 6,
             "seen_in_train": True},
            {"domain": "Space", "documents": 5, "macro_ap": 0.7, "categories_defined": 4,
             "seen_in_train": False}]
    feats = [{"feature": "blood type", "coef": 3.0, "u_domain": 0.1, "doc_freq": 9,
              "annotation": "field label"}]
    shortcuts = {"per_domain": {s: dict.fromkeys(R.TRAINED, rows) for s in ("dev", "test_shift")},
                 "top_features": dict.fromkeys(CATEGORIES, feats),
                 "rule": {"domain_u_threshold": 0.25}}
    return metrics, selection, shortcuts


def test_render_results_end_to_end_on_synthetic_inputs():
    text = R.render_results(*_synthetic_inputs())
    assert R.wrong_comparisons(text) == []
    for heading in ("## Setup", "## RQ1", "### test_ood", "## Paired model comparison",
                    "### Domain-only baseline vs text models", "### Per-domain macro PR-AUC",
                    "#### SPECIAL_CATEGORY", "## Observations", "## Threats to validity"):
        assert heading in text
    assert "excluded from every macro average: HEALTH" in text
    assert "(grid edge)" in text and "Space (*unseen*)" in text
    assert "test_ood: 5" in text                     # generated count of defined categories


@pytest.mark.data
def test_committed_results_doc_is_current_and_consistent():
    import sys

    from sensclf.rawdata import REPO_ROOT

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import make_results

    text = R.render_results(*make_results.load_inputs())
    assert R.wrong_comparisons(text) == []
    assert make_results.DOC.read_text(encoding="utf-8") == text
