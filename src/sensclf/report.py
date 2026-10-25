"""Render docs/results.md from the committed result JSON files. No number is typed by hand.

Every sentence that compares two numbers is produced by :func:`compare_sentence` or
:func:`ci_sentence` from the numbers themselves, and :func:`wrong_comparisons` re-checks every
such sentence in the final text (tests run it), so the prose cannot contradict the tables.
Comparisons are made on the values as displayed (three decimals).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence

from sensclf.labels import CATEGORIES

TEXT_MODELS = ("word_lr", "char_lr", "word_svc")
TRAINED = (*TEXT_MODELS, "domain_lr")
ALL_MODELS = (*TRAINED, "regex_detector")
SETS = ("dev", "test_shift", "test_ood")
NA = "n/a"


def fmt(x: float | None, digits: int = 3) -> str:
    return NA if x is None else f"{x:.{digits}f}"


def fmt_ci(ci: Sequence[float] | None) -> str:
    return "" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"


def compare_sentence(name_a: str, a: float, name_b: str, b: float, metric: str) -> str:
    """'`A` has a higher <metric> than `B` (0.979 vs 0.975)': direction from the numbers."""
    ra, rb = round(a, 3), round(b, 3)
    if ra == rb:
        return f"`{name_a}` has the same {metric} as `{name_b}` ({ra:.3f} vs {rb:.3f})"
    word = "higher" if ra > rb else "lower"
    return f"`{name_a}` has a {word} {metric} than `{name_b}` ({ra:.3f} vs {rb:.3f})"


def ci_sentence(ci: Sequence[float] | None) -> str:
    """'the 95% CI of the difference [x, y] excludes 0' (or includes), from the bounds."""
    if ci is None:
        return "no interval could be computed"
    lo, hi = round(ci[0], 3), round(ci[1], 3)
    word = "excludes" if lo > 0 or hi < 0 else "includes"
    return f"the 95% CI of the difference [{lo:.3f}, {hi:.3f}] {word} 0"


_COMPARE = re.compile(r"`[^`]+` has (?:a (higher|lower)|(the same)) [^`(]+? (?:than|as) `[^`]+` "
                      r"\((-?\d+\.\d{3}) vs (-?\d+\.\d{3})\)")
_CI = re.compile(r"\[(-?\d+\.\d{3}), (-?\d+\.\d{3})\] (excludes|includes) 0")


def wrong_comparisons(text: str) -> list[str]:
    """Generated comparison sentences whose wording contradicts their own numbers."""
    wrong = []
    for m in _COMPARE.finditer(text):
        word = m.group(1) or m.group(2)
        a, b = float(m.group(3)), float(m.group(4))
        truth = "higher" if a > b else "lower" if a < b else "the same"
        if word != truth:
            wrong.append(m.group(0))
    for m in _CI.finditer(text):
        lo, hi = float(m.group(1)), float(m.group(2))
        truth = "excludes" if lo > 0 or hi < 0 else "includes"
        if m.group(3) != truth:
            wrong.append(m.group(0))
    return wrong


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# --- setup -----------------------------------------------------------------------------------

def setup_section(selection: Mapping[str, dict], budget_minutes: float = 45.0) -> str:
    rows = []
    for name in TRAINED:
        s = selection[name]
        grid = ", ".join(f"C={g['C']:g}: {fmt(g['dev_macro_ap'], 4)}" for g in s["grid"])
        cs = [g["C"] for g in s["grid"]]
        edge = " (grid edge)" if s["selected_C"] in (min(cs), max(cs)) else ""
        rows.append([f"`{name}`", grid, f"{s['selected_C']:g}{edge}",
                     f"{s['grid_train_seconds']:.0f}", f"{s['selected_model_train_seconds']:.0f}",
                     fmt(s["peak_memory_mb"], 0), str(s["convergence_warnings"])])
    total_min = sum(selection[n]["grid_train_seconds"] for n in TRAINED) / 60
    docs = selection["word_lr"]["training_set"]["documents"]
    relation = "within" if total_min <= budget_minutes else "over"
    decision = (f"All models were trained on the full train split ({docs:,} documents). "
                f"Training every grid took {total_min:.1f} minutes in total, {relation} the "
                f"{budget_minutes:.0f}-minute budget that would have triggered a 40,000-document "
                "group-aware sample (the decision was taken from a timing probe before any "
                "evaluation).")
    return "\n\n".join([
        "## Setup",
        decision,
        ("Hyper-parameters were selected on dev only (macro average precision over the six "
        "categories). The dev grid, train times (seconds) and the peak memory of each "
        "family's own process:"),
        _table(["model", "dev macro PR-AUC per C", "selected C", "grid train s",
                "selected model train s", "peak MB", "convergence warnings"], rows),
        ("`regex_detector` needs no training. `word_svc` scores are margins (default threshold "
        "0); the other trained models output probabilities (default threshold 0.5)."),
    ])


# --- RQ1 -------------------------------------------------------------------------------------

def _cell(point: float | None, ci: Sequence[float] | None) -> str:
    return NA if point is None else fmt(point) + fmt_ci(ci)


def _category_cells(models: Mapping, boot: Mapping, c: str, m: str) -> tuple[str, str]:
    pc = models[m]["per_category"][c]
    if not pc.get("covered", True):
        return NA, NA
    if not pc["defined"]:
        return "undefined", "undefined"
    at = pc["at"].get("dev_f1")
    return (_cell(pc["average_precision"], boot[m].get(f"ap|{c}")),
            _cell(at["f1"] if at else None, boot[m].get(f"f1|{c}")))


def rq1_set_section(set_name: str, data: Mapping) -> str:
    models, boot = data["models"], data["bootstrap"]["ci"]
    ap_rows, f1_rows = [], []
    for c in CATEGORIES:
        first = models["word_lr"]["per_category"][c]
        prev = f"{first['positives']:,} / {first['documents']:,}"
        cells = [_category_cells(models, boot, c, m) for m in ALL_MODELS]
        ap_rows.append([c, prev, *[a for a, _ in cells]])
        f1_rows.append([c, prev, *[f for _, f in cells]])
    agg_rows = []
    for m in ALL_MODELS:
        mm, vec = models[m]["macro"], models[m]["vector"]
        scope = (", ".join(mm["categories"]) if m == "regex_detector"
                 else ", ".join(mm["excluded"]) or "none")
        agg_rows.append([
            f"`{m}`", _cell(mm["average_precision"], boot[m].get("macro_ap")),
            fmt(mm["roc_auc"]), _cell(vec["dev_f1"]["macro_f1"], boot[m].get("macro_f1")),
            _cell(vec["dev_f1"]["micro_f1"], boot[m].get("micro_f1")),
            fmt(vec["default"]["macro_f1"]), fmt(vec["dev_f1"]["subset_accuracy"]),
            fmt(vec["dev_f1"]["hamming_loss"]), fmt(mm["brier"]), fmt(mm["log_loss"]), scope])
    excluded = models["word_lr"]["macro"]["excluded"]
    note = (f"Undefined in this set (no positive or no negative document), excluded from every "
            f"macro average: {', '.join(excluded)}." if excluded else
            "Every category is defined in this set.")
    head = ["category", "positives / documents", *[f"`{m}`" for m in ALL_MODELS]]
    return "\n\n".join([
        (f"### {set_name} ({data['documents']:,} documents, "
        f"{data['bootstrap']['groups']:,} groups)"),
        note,
        "Average precision (PR-AUC) with 95% cluster-bootstrap CI:",
        _table(head, ap_rows),
        "F1 at the dev-F1 threshold with 95% cluster-bootstrap CI:",
        _table(head, f1_rows),
        ("Aggregates (vector metrics over the evaluable categories; F1, subset accuracy and "
        "Hamming loss at the dev-F1 thresholds unless marked default; the detector's macro "
        "values cover only its two categories):"),
        _table(["model", "macro PR-AUC", "macro ROC-AUC", "macro F1", "micro F1",
                "macro F1 (default thr.)", "subset accuracy", "Hamming loss", "macro Brier",
                "macro log loss", "excluded categories (detector: covered)"], agg_rows),
    ])


def thresholds_section(metrics: Mapping) -> str:
    ops = metrics["operating_points"]
    rows = [[f"`{m}`", fmt(next(t for t in ops[m]["default"].values() if t is not None)),
             *[fmt(ops[m]["dev_f1"][c]) for c in CATEGORIES]] for m in ALL_MODELS]
    return "\n\n".join([
        "### Operating points (chosen on dev)",
        ("`regex_detector` has no score to tune (0/1 output), so both of its operating points "
         "are its own decision; on 0/1 scores the F1-maximising threshold could only be that "
         "decision or 'everything positive'."),
        _table(["model", "default", *[f"dev-F1 {c}" for c in CATEGORIES]], rows),
    ])


def paired_section(metrics: Mapping) -> str:
    parts = ["## Paired model comparison (macro PR-AUC, same group resamples)",
             "`regex_detector` is not paired: it covers two categories only."]
    for set_name in SETS:
        pairs = metrics["sets"][set_name]["bootstrap"]["paired_macro_ap"]
        models = metrics["sets"][set_name]["models"]
        rows, sentences = [], []
        for key, p in pairs.items():
            if "skipped" in p:
                rows.append([key, NA, NA, NA])
                continue
            a, b = p["model_a"], p["model_b"]
            rows.append([f"`{a}` − `{b}`", f"{p['point']:+.3f}", fmt_ci(p["ci"]).strip(),
                         f"{p['share_a_better']:.3f}"])
            va = models[a]["macro"]["average_precision"]
            vb = models[b]["macro"]["average_precision"]
            sentences.append(f"- {compare_sentence(a, va, b, vb, 'macro PR-AUC')}; "
                             f"{ci_sentence(p['ci'])}.")
        parts += [f"### {set_name}", _table(["difference", "point", "95% CI",
                                             "share of resamples with A > B"], rows),
                  "\n".join(sentences)]
    return "\n\n".join(parts)


# --- RQ2 -------------------------------------------------------------------------------------

def _macro_ap(metrics: Mapping, set_name: str, model: str) -> float | None:
    return metrics["sets"][set_name]["models"][model]["macro"]["average_precision"]


def no_skill(metrics: Mapping, set_name: str) -> float | None:
    """Macro AP of a random ranking = mean prevalence over the evaluable categories."""
    pc = metrics["sets"][set_name]["models"]["word_lr"]["per_category"]
    vals = [pc[c]["positives"] / pc[c]["documents"] for c in CATEGORIES if pc[c]["defined"]]
    return sum(vals) / len(vals) if vals else None


def domain_only_section(metrics: Mapping) -> str:
    rows = []
    for m in TRAINED:
        vals = [_macro_ap(metrics, s, m) for s in SETS]
        drops = [None if vals[0] is None or v is None else vals[0] - v for v in vals[1:]]
        rows.append([f"`{m}`", *[fmt(v) for v in vals], *[fmt(d) for d in drops]])
    rows.append(["no-skill (mean prevalence)", *[fmt(no_skill(metrics, s)) for s in SETS],
                 NA, NA])
    best = max(TEXT_MODELS, key=lambda m: _macro_ap(metrics, "dev", m))
    dev_best = _macro_ap(metrics, "dev", best)
    dev_dom = _macro_ap(metrics, "dev", "domain_lr")
    drop = {m: _macro_ap(metrics, "dev", m) - _macro_ap(metrics, "test_shift", m)
            for m in (best, "domain_lr")}
    n_ood = len(metrics["sets"]["test_ood"]["models"]["word_lr"]["macro"]["categories"])
    in_domain = compare_sentence("domain_lr", dev_dom, best, dev_best, "macro PR-AUC")
    shift = compare_sentence("domain_lr", drop["domain_lr"], best, drop[best], "drop")
    lines = [
        (f"- In-domain (dev): {in_domain}; the no-skill level is "
        f"{fmt(no_skill(metrics, 'dev'))}."),
        f"- Drop from dev to test_shift: {shift}.",
        ("- test_ood has one domain value (`finance`, equal to the training domain `Finance` "
        "after case-folding), so the domain-only model gives every test_ood document the same score "
        "and its PR-AUC there equals each category's prevalence."),
    ]
    return "\n\n".join([
        "### Domain-only baseline vs text models",
        ("Macro PR-AUC; the drop is dev minus the shifted set (positive = worse under shift). "
        f"Macro averages run over the categories defined in each set (test_ood: {n_ood})."),
        _table(["model", *SETS, "drop to test_shift", "drop to test_ood"], rows),
        "\n".join(lines),
    ])


def per_domain_section(shortcuts: Mapping) -> str:
    parts = ["### Per-domain macro PR-AUC (dev and test_shift)",
             ("Macro over the categories defined inside each domain (count in the table). "
             "Domains marked *unseen* do not occur in the training split (case-folded match).")]
    for set_name in ("dev", "test_shift"):
        by_model = shortcuts["per_domain"][set_name]
        rows = []
        for i, row in enumerate(by_model["word_lr"]):
            cells = [fmt(by_model[m][i]["macro_ap"]) for m in TRAINED]
            rows.append([row["domain"] + ("" if row["seen_in_train"] else " (*unseen*)"),
                         f"{row['documents']:,}", str(row["categories_defined"]), *cells])
        word = [r for r in by_model["word_lr"] if r["macro_ap"] is not None]
        lo = min(word, key=lambda r: r["macro_ap"])
        hi = max(word, key=lambda r: r["macro_ap"])
        unseen = sum(not r["seen_in_train"] for r in by_model["word_lr"])
        summary = (f"{set_name}: {len(by_model['word_lr'])} domains, {unseen} unseen in train. "
                   f"`word_lr` per-domain macro PR-AUC ranges from {fmt(lo['macro_ap'])} "
                   f"({lo['domain']}) to {fmt(hi['macro_ap'])} ({hi['domain']}).")
        parts += [f"#### {set_name}", summary,
                  _table(["domain", "documents", "categories", *[f"`{m}`" for m in TRAINED]],
                         rows)]
    return "\n\n".join(parts)


def annotation_counts(shortcuts: Mapping) -> dict[str, Counter]:
    return {c: Counter(r["annotation"] for r in rows)
            for c, rows in shortcuts["top_features"].items()}


def features_section(shortcuts: Mapping) -> str:
    rule = shortcuts["rule"]
    counts = annotation_counts(shortcuts)
    kinds = ("field label", "domain/topic word", "other")
    summary = _table(["category", *kinds],
                     [[c, *[str(counts[c][k]) for k in kinds]] for c in CATEGORIES])
    parts = [
        "### Top-25 positive features of `word_lr` per category",
        ("Annotation rule (fixed before the results; `sensclf.shortcuts`): **field label** if "
        "the feature contains a word of a raw dataset label name (generic tokens removed) or a "
        "standard abbreviation; otherwise **domain/topic word** if U = MI(F; D) / H(F) ≥ "
        f"{rule['domain_u_threshold']} on the training documents (F: the feature occurs; D: "
        "the domain; U is the share of the feature's occurrence entropy explained by the "
        "domain); otherwise **other**."),
        "Counts per category:", summary,
    ]
    for c in CATEGORIES:
        rows = [[f"`{r['feature']}`", f"{r['coef']:.2f}", f"{r['u_domain']:.3f}",
                 f"{r['doc_freq']:,}", r["annotation"]] for r in shortcuts["top_features"][c]]
        parts += [f"#### {c}", _table(["feature", "coefficient", "U (domain)",
                                       "train documents", "annotation"], rows)]
    return "\n\n".join(parts)


# --- observations and the whole document ------------------------------------------------------

def _best(metrics: Mapping, set_name: str, models: Sequence[str]) -> str:
    return max(models, key=lambda m: _macro_ap(metrics, set_name, m) or -1.0)


def _f1(metrics: Mapping, set_name: str, model: str, category: str) -> float | None:
    pc = metrics["sets"][set_name]["models"][model]["per_category"][category]
    if not pc.get("covered", True) or not pc["defined"]:
        return None
    return pc["at"]["dev_f1"]["f1"]


def observations(metrics: Mapping, shortcuts: Mapping) -> str:
    lines = []
    for s in SETS:
        best = _best(metrics, s, TRAINED)
        lines.append(f"- {s}: the highest macro PR-AUC among the trained models is "
                     f"`{best}` ({fmt(_macro_ap(metrics, s, best))}).")
    for s in ("test_shift", "test_ood"):
        lines.append("- " + compare_sentence(
            f"word_lr on {s}", _macro_ap(metrics, s, "word_lr"), "word_lr on dev",
            _macro_ap(metrics, "dev", "word_lr"), "macro PR-AUC") + ".")
    for s in SETS:
        pc = metrics["sets"][s]["models"]["word_lr"]["per_category"]
        defined = [c for c in CATEGORIES if pc[c]["defined"]]
        worst = min(defined, key=lambda c: pc[c]["average_precision"])
        lines.append(f"- {s}: `word_lr`'s lowest per-category PR-AUC is {worst} "
                     f"({fmt(pc[worst]['average_precision'])}).")
    for c in ("CONTACT", "FINANCIAL"):
        det, word = _f1(metrics, "dev", "regex_detector", c), _f1(metrics, "dev", "word_lr", c)
        if det is not None and word is not None:
            lines.append(f"- {c} on dev: " + compare_sentence(
                "regex_detector", det, "word_lr", word, "F1 (dev-F1 threshold)") + ".")
    counts = annotation_counts(shortcuts)
    total = Counter()
    for c in counts.values():
        total.update(c)
    lines.append("- Top-25 features of `word_lr` over the six categories: "
                 + ", ".join(f"{total[k]} {k}" for k in ("field label", "domain/topic word",
                                                         "other")) + ".")
    return "\n".join(["## Observations", "", "Generated from the tables above; descriptive only.",
                      "", *lines])


THREATS = """## Threats to validity

- **Synthetic data.** Both datasets are generated by language-model pipelines. Their templates,
  vocabularies and formatting regularities are easier for n-gram models than real documents,
  so absolute scores say little about real-world performance.
- **Labels come from annotated spans.** A document is positive when the generator annotated a
  span of a mapped raw label. Unannotated mentions make false negatives, and the category
  mapping (docs/TASK.md, Decisions) decides what each category means. A model can be "wrong"
  where the annotation is incomplete.
- **Missing categories in test_ood.** Gretel has no health or GDPR Art. 9 labels, so HEALTH and
  SPECIAL_CATEGORY are undefined there and test_ood macro averages cover fewer categories than
  dev or test_shift; macro values are not comparable across these sets.
- **Prevalence differs between sets.** PR-AUC depends on prevalence (the no-skill level is
  listed next to the domain-only comparison), so part of a PR-AUC change between sets is not
  a change in ranking quality.
- **Selection on dev.** C and the dev-F1 thresholds are chosen on dev, so dev numbers are
  optimistic; test_shift and test_ood are untouched by any choice. The grid is small, and a
  selected C at the edge of its grid (marked in the setup table) may not be the optimum.
- **Uncertainty.** Intervals come from a cluster bootstrap over groups of the evaluation set
  (1000 resamples, seed 0); they do not cover variability from training data or seeds.
- **Domain field.** The shortcut analysis trusts the dataset's `domain` field; the per-domain
  sets are small for some domains, and test_shift domains are mostly unseen by design.
"""


def render_results(metrics: Mapping, selection: Mapping, shortcuts: Mapping) -> str:
    parts = [
        "# Results",
        ("<!-- Generated by scripts/make_results.py from results/m2/*.json. Do not edit by hand. "
        "-->"),
        ("Research questions:\n\n"
        "- **RQ1** How well do linear text classifiers detect each sensitive category at "
        "document level, in-domain (dev), under domain shift (test_shift) and out of "
        "distribution (test_ood)?\n"
        "- **RQ2** Do they learn the sensitive content or the document's domain (shortcut "
        "learning)?"),
        ("Task and sets: [TASK.md](TASK.md); data: [DATA.md](DATA.md); background: "
        "[theory/01-linear-text-classification.md](theory/01-linear-text-classification.md). "
        "The calibration split is not used in this milestone (a guard refuses it)."),
        setup_section(selection),
        "## RQ1: detection per category",
        *[rq1_set_section(s, metrics["sets"][s]) for s in SETS],
        thresholds_section(metrics),
        paired_section(metrics),
        "## RQ2: content or domain?",
        domain_only_section(metrics),
        per_domain_section(shortcuts),
        features_section(shortcuts),
        observations(metrics, shortcuts),
        THREATS,
    ]
    return "\n\n".join(p.strip() for p in parts) + "\n"
