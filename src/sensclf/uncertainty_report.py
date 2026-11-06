"""Render docs/results-uncertainty.md (RQ3-RQ5) from the committed result JSON files.

No number is typed by hand: every value and every counted statement ("k of 18 thresholds were
violated") is computed here from results/m3/calibration.json, results/m4/selective.json and
results/m5/conformal.json. Comparative sentences reuse :func:`sensclf.report.compare_sentence`,
so :func:`sensclf.report.wrong_comparisons` can re-check them.
"""

from __future__ import annotations

from collections.abc import Mapping

from sensclf.labels import CATEGORIES
from sensclf.report import SETS, compare_sentence, fmt, fmt_ci

METHODS = ("raw", "temperature", "platt", "isotonic")
SET_NAMES = {"dev": "dev (in-domain)", "test_shift": "test_shift (us -> intl)",
             "test_ood": "test_ood (Gretel)", "cal_test": "cal_test"}


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _calibration(cal: Mapping) -> list[str]:
    out = ["## RQ3 -- Are the probabilities calibrated, and does calibration survive shift?", "",
           (f"Calibrators are fitted on cal_fit ({cal['calibration_halves']['cal_fit']} documents) "
           "only; the calibrator of each model is selected by "
           f"**{cal['selection_rule']}**. Macro values over the six categories; 95% cluster-"
           f"bootstrap intervals by group ({cal['n_resamples']} resamples). ECE uses 15 "
           "equal-width bins."), ""]
    for model, m in sorted(cal["models"].items()):
        avail = [x for x in METHODS if x in m["methods"]]
        out += [f"### `{model}` (selected: **{m['selected']}**)", "",
                "| method | " + " | ".join(f"ECE {s}" for s in SETS) + " | "
                + " | ".join(f"Brier {s}" for s in SETS) + " |",
                "|---|" + "---:|" * (2 * len(SETS))]
        for meth in avail:
            r = m["methods"][meth]
            out.append(f"| {meth} | " + " | ".join(fmt(r[s]["ece"], 4) for s in SETS) + " | "
                       + " | ".join(fmt(r[s]["brier"], 4) for s in SETS) + " |")
        out.append("")
        base = "raw" if "raw" in m["methods"] else avail[0]
        sel = m["selected"]
        for s in SETS:
            a, b = m["methods"][sel][s]["ece"], m["methods"][base][s]["ece"]
            out.append(f"- {SET_NAMES[s]}: "
                       + compare_sentence(f"{model}+{sel}", a, f"{model}+{base}", b, "ECE")
                       + ".")
        out.append("")
    out += [("Note on intervals: ECE is biased upwards in small resamples (empty or sparse bins "
            "add noise that never cancels), so the bootstrap distribution of ECE can sit "
            "entirely above the point value; the intervals of the Brier score and log loss do "
            "not have this problem and are the ones to read for uncertainty. Full intervals are "
            "in `results/m3/calibration.json`."), ""]
    return out


def _selective(sel: Mapping) -> list[str]:
    targets = [str(t) for t in sel["targets"]]
    out = ["## RQ4 -- Abstention: how much can be decided automatically at a target risk?", "",
           (f"Model: `{sel['model']}` + {sel['calibration']} calibration. Per category, the "
           "decision is p >= 0.5 and the confidence max(p, 1 - p). Thresholds are certified on "
           f"cal_test ({sel['documents']['cal_test']} documents) so that the selective risk is "
           f"<= r* with probability >= {1 - sel['delta']:.2f} under exchangeability "
           "(20 candidate coverages, Bonferroni, exact binomial test)."), "",
           "### Risk-coverage summary (AURC; lower is better)", "",
           "| category | " + " | ".join(f"AURC {s}" for s in SETS) + " | "
           + " | ".join(f"oracle {s}" for s in SETS) + " |",
           "|---|" + "---:|" * (2 * len(SETS))]
    for c in CATEGORIES:
        e = sel["categories"][c]["sets"]
        out.append(f"| {c} | " + " | ".join(fmt(e[s]["aurc"], 4) for s in SETS) + " | "
                   + " | ".join(fmt(e[s]["oracle_aurc"], 4) for s in SETS) + " |")
    out += ["", "### Certified thresholds: coverage / achieved risk", "",
            ("A cell shows coverage, achieved selective risk and, in brackets, the 95% bootstrap "
            "interval of the risk; **bold** marks a risk above the target. *(no positives)*: the "
            "category has no positive document in that set (HEALTH and SPECIAL_CATEGORY in "
            "test_ood), so the risk only counts false alarms; such cells are left out of the "
            "counts."), ""]
    violations: dict[str, int] = {s: 0 for s in SETS}
    total: dict[str, int] = {s: 0 for s in SETS}
    uncertified = []
    for r in targets:
        out += [f"#### r* = {pct(float(r))}", "",
                "| category | certified on cal_test | " + " | ".join(SETS) + " |",
                "|---|---|" + "---|" * len(SETS)]
        for c in CATEGORIES:
            t = sel["categories"][c]["thresholds"][r]
            ch = t["choice"]
            if not ch["certified"]:
                uncertified.append(f"{c} at r* = {pct(float(r))}")
                out.append(f"| {c} | no (abstain on everything) | "
                           + " | ".join("coverage 0" for _ in SETS) + " |")
                continue
            cells = []
            for s in SETS:
                a = t["applied"][s]
                risk = pct(a["risk"])
                if not sel["categories"][c]["sets"][s]["defined"]:
                    cells.append(f"{pct(a['coverage'])} / {risk} (no positives)")
                    continue
                total[s] += 1
                if a["violation"]:
                    violations[s] += 1
                    risk = f"**{risk}**"
                cells.append(f"{pct(a['coverage'])} / {risk}{fmt_ci(a['risk_ci'])}")
            out.append(f"| {c} | yes, coverage {pct(ch['cal_coverage'])} | "
                       + " | ".join(cells) + " |")
        out.append("")
    out += ["Counted from the tables above:", ""]
    for s in SETS:
        out.append(f"- {SET_NAMES[s]}: the achieved risk exceeds the target for {violations[s]} "
                   f"of {total[s]} certified thresholds"
                   + (" on categories with positives." if s == "test_ood" else "."))
    if uncertified:
        out.append("- Not certifiable on cal_test (too few accepted documents with few enough "
                   "errors): " + ", ".join(uncertified) + ".")
    out += ["", "### Whole documents", "",
            ("A document is decided automatically only if all six categories are accepted; its "
            "error rate counts a document as wrong if any accepted category is wrong. The "
            "per-category guarantees do not add up to a document-level one (no union bound is "
            "applied here), which the error rates below make visible."), "",
            "| r* | " + " | ".join(f"coverage {s}" for s in SETS) + " | "
            + " | ".join(f"doc error {s}" for s in SETS) + " |",
            "|---|" + "---:|" * (2 * len(SETS))]
    for r in targets:
        d = sel["document_level"][r]
        out.append(f"| {pct(float(r))} | " + " | ".join(pct(d[s]["coverage"]) for s in SETS)
                   + " | " + " | ".join(pct(d[s]["doc_error_rate"]) for s in SETS) + " |")
    out.append("")
    return out


def _conformal(conf: Mapping) -> list[str]:
    out = ["## RQ5 -- Conformal prediction sets: is the coverage guarantee met?", "",
           (f"Model: `{conf['model']}` + {conf['calibration']}. Quantiles computed on "
           f"{conf['n_cal']} cal_test documents (one per group). Coverage is the share of "
           "documents whose true label is in the set; *both* is the share of sets {0, 1} "
           "(the model abstains), *empty* the share of empty sets. Under exchangeability the "
           "coverage is at least 1 - alpha; the upper bound 1 - alpha + 1/(n + 1) holds only "
           "without ties in the scores, and isotonic calibration produces many ties, so "
           "coverage above it is expected and not an error."), ""]
    shortfalls: dict[str, list[str]] = {}
    n_cells: dict[str, int] = {}
    for variant in ("marginal", "mondrian"):
        out += [f"### {variant.capitalize()}", ""]
        for a, v in sorted(conf["variants"][variant].items(), key=lambda kv: float(kv[0])):
            alpha = float(a)
            out += [f"#### alpha = {pct(alpha)} (target coverage {pct(1 - alpha)})", "",
                    "| category | " + " | ".join(f"coverage {s}" for s in SETS)
                    + " | coverage y=1 dev | both dev | empty dev |",
                    "|---|" + "---:|" * (len(SETS) + 3)]
            for c in CATEGORIES:
                st = v["per_category"][c]["sets"]
                cells = []
                for s in SETS:
                    cov = st[s]["coverage"]
                    txt = pct(cov)
                    if st[s]["n_y1"] == 0:
                        cells.append(f"{txt} (no positives)")
                        continue
                    n_cells[f"{variant}/{s}"] = n_cells.get(f"{variant}/{s}", 0) + 1
                    if round(cov, 3) < round(1 - alpha, 3):
                        txt = f"**{txt}**"
                        shortfalls.setdefault(f"{variant}/{s}", []).append(
                            f"{c} at alpha {pct(alpha)}")
                    cells.append(txt)
                d = st["dev"]
                out.append(f"| {c} | " + " | ".join(cells) + f" | {pct(d['coverage_y1'])} | "
                           f"{pct(d['both'])} | {pct(d['empty'])} |")
            j = v["joint"]
            out += ["", "Joint 6-label sets (alpha / 6 per category): "
                    + "; ".join(f"{s} joint coverage {pct(j[s]['joint_coverage'])}, all six "
                                f"decided {pct(j[s]['all_singleton'])}" for s in SETS) + ".",
                    ""]
    out += ["**Bold** marks a coverage below the target. Counted:", ""]
    for variant in ("marginal", "mondrian"):
        for s in SETS:
            items = shortfalls.get(f"{variant}/{s}", [])
            out.append(f"- {variant}, {SET_NAMES[s]}: {len(items)} of "
                       f"{n_cells.get(f'{variant}/{s}', 0)} cells (categories with positives) below "
                       "target" + (f" ({', '.join(items)})" if items else "") + ".")
    sp = conf["dev_coverage_spread_alpha_0.1"]
    out += ["", (f"Variability of the dev coverage over {sp['draws']} different one-per-group "
            "draws of the calibration documents (marginal, alpha = 10%):"), "",
            "| category | min | mean | max |", "|---|---:|---:|---:|"]
    for c in CATEGORIES:
        x = sp["per_category"][c]
        out.append(f"| {c} | {pct(x['min'])} | {pct(x['mean'])} | {pct(x['max'])} |")
    out += ["", ("The guarantee is on average over calibration draws, so a single draw can fall "
            "slightly below 1 - alpha; the mean is the quantity the theorem bounds."), ""]
    return out


def render_uncertainty(cal: Mapping, sel: Mapping, conf: Mapping) -> str:
    lines = ["# Results -- calibration, abstention and conformal sets (RQ3-RQ5)", "",
             ("Generated by `scripts/make_uncertainty_results.py` from `results/m3`, `results/m4` "
             "and `results/m5`. Do not edit by hand. Methods: "
             "[theory note](theory/02-calibration-abstention-conformal.md)."), "",
             "![Reliability diagrams](figures/m3_reliability.png)", "",
             "![Risk-coverage curves](figures/m4_risk_coverage.png)", ""]
    lines += _calibration(cal) + _selective(sel) + _conformal(conf)
    return "\n".join(lines).rstrip() + "\n"
