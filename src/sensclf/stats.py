"""Descriptive statistics of the labelled sets and the trivial majority baseline.

Pure functions over :class:`~sensclf.splits.LabelledDocument` lists; rendering to markdown
is kept separate so the numbers in docs/DATA.md and results/data_stats.json are the same
objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sensclf.labels import CATEGORIES
from sensclf.splits import LabelledDocument

# Below/above these prevalences some later metrics (per-category F1, calibration error,
# per-class conformal coverage) rest on very few positives or negatives.
LOW_PREVALENCE = 0.02
HIGH_PREVALENCE = 0.98


def set_stats(docs: Sequence[LabelledDocument]) -> dict:
    """Documents, mean length, per-category prevalence, label cardinality, share unlabelled."""
    n = len(docs)
    counts = {c: sum(d.labels[c] for d in docs) for c in CATEGORIES}
    positives_per_doc = [sum(d.labels.values()) for d in docs]
    return {
        "documents": n,
        "mean_length_chars": _ratio(sum(len(d.text) for d in docs), n),
        "prevalence": {c: {"count": counts[c], "rate": _ratio(counts[c], n)} for c in CATEGORIES},
        "label_cardinality": _ratio(sum(positives_per_doc), n),
        "share_no_category": _ratio(sum(k == 0 for k in positives_per_doc), n),
    }


def cooccurrence(docs: Sequence[LabelledDocument]) -> dict[str, dict[str, int]]:
    """``m[a][b]`` = documents positive for both a and b (diagonal: positives of a)."""
    return {a: {b: sum(d.labels[a] and d.labels[b] for d in docs) for b in CATEGORIES}
            for a in CATEGORIES}


def majority_baseline(fit: Sequence[LabelledDocument],
                      evaluate: Sequence[LabelledDocument]) -> dict[str, dict]:
    """Per category: predict the majority class of ``fit`` for every document of ``evaluate``.

    F1 is for the positive class; when the baseline predicts 0 everywhere there are no true
    positives and F1 is 0 (the ``zero_division=0`` convention), flagged ``f1_defined=False``
    because precision is 0/0. Ties (exactly half positive) predict 1.
    """
    out = {}
    for c in CATEGORIES:
        positives = sum(d.labels[c] for d in fit)
        majority = int(2 * positives >= len(fit)) if fit else 0
        truth = [d.labels[c] for d in evaluate]
        tp = sum(t == 1 and majority == 1 for t in truth)
        fp = sum(t == 0 and majority == 1 for t in truth)
        fn = sum(t == 1 and majority == 0 for t in truth)
        correct = sum(t == majority for t in truth)
        out[c] = {
            "majority": majority,
            "accuracy": _ratio(correct, len(truth)),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn) if tp else 0.0,
            "f1_defined": majority == 1 or tp > 0,
        }
    return out


def flag_prevalence(per_set: Mapping[str, dict]) -> list[dict]:
    """(set, category, rate) for every prevalence outside [2%, 98%]."""
    flags = []
    for name, stats in per_set.items():
        for c in CATEGORIES:
            rate = stats["prevalence"][c]["rate"]
            if rate < LOW_PREVALENCE or rate > HIGH_PREVALENCE:
                flags.append({"set": name, "category": c, "rate": rate,
                              "side": "low" if rate < LOW_PREVALENCE else "high"})
    return flags


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


# --- markdown ------------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def sets_table(per_set: Mapping[str, dict]) -> str:
    head = "| set | documents | mean length (chars) | label cardinality | no category |"
    rows = [head, "|---|---:|---:|---:|---:|"]
    for name, s in per_set.items():
        rows.append(f"| {name} | {s['documents']:,} | {s['mean_length_chars']:,.1f} | "
                    f"{s['label_cardinality']:.3f} | {_pct(s['share_no_category'])} |")
    return "\n".join(rows)


def prevalence_table(per_set: Mapping[str, dict]) -> str:
    rows = ["| category | " + " | ".join(per_set) + " |", "|---" + "|---:" * len(per_set) + "|"]
    for c in CATEGORIES:
        cells = [f"{_pct(s['prevalence'][c]['rate'])} ({s['prevalence'][c]['count']:,})"
                 for s in per_set.values()]
        rows.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def cooccurrence_table(matrix: Mapping[str, Mapping[str, int]]) -> str:
    rows = ["| | " + " | ".join(CATEGORIES) + " |", "|---" + "|---:" * len(CATEGORIES) + "|"]
    for a in CATEGORIES:
        rows.append(f"| {a} | " + " | ".join(f"{matrix[a][b]:,}" for b in CATEGORIES) + " |")
    return "\n".join(rows)


def baseline_table(baseline: Mapping[str, dict]) -> str:
    rows = ["| category | majority class (train) | accuracy (dev) | F1 (dev, positive class) |",
            "|---|---:|---:|---:|"]
    for c, b in baseline.items():
        f1 = f"{b['f1']:.4f}" + ("" if b["f1_defined"] else " (undefined, set to 0)")
        rows.append(f"| {c} | {b['majority']} | {b['accuracy']:.4f} | {f1} |")
    return "\n".join(rows)


def flags_list(flags: Sequence[dict]) -> str:
    if not flags:
        return "No category has prevalence below 2% or above 98% in any set."
    return "\n".join(f"- **{f['category']}** in `{f['set']}`: {_pct(f['rate'])} ({f['side']})"
                     for f in flags)
