import pytest

from sensclf import stats
from sensclf.labels import CATEGORIES
from sensclf.splits import LabelledDocument

ZERO = dict.fromkeys(CATEGORIES, 0)


def _doc(i, text="abcd", **labels):
    return LabelledDocument(f"d{i}", text, {**ZERO, **labels}, {})


# 4 documents: two IDENTITY+CONTACT, one IDENTITY, one with nothing.
DOCS = [_doc(0, "ab", IDENTITY=1, CONTACT=1), _doc(1, "abcd", IDENTITY=1, CONTACT=1),
        _doc(2, "abcdef", IDENTITY=1), _doc(3, "abcdefgh")]


def test_set_stats_on_tiny_fixture():
    s = stats.set_stats(DOCS)
    assert s["documents"] == 4
    assert s["mean_length_chars"] == 5.0
    assert s["prevalence"]["IDENTITY"] == {"count": 3, "rate": 0.75}
    assert s["prevalence"]["CONTACT"] == {"count": 2, "rate": 0.5}
    assert s["prevalence"]["HEALTH"] == {"count": 0, "rate": 0.0}
    assert s["label_cardinality"] == 5 / 4
    assert s["share_no_category"] == 0.25


def test_set_stats_on_empty_set_is_zero_not_an_error():
    s = stats.set_stats([])
    assert s["documents"] == 0 and s["mean_length_chars"] == 0.0


def test_cooccurrence_is_symmetric_with_counts_on_the_diagonal():
    m = stats.cooccurrence(DOCS)
    assert m["IDENTITY"]["IDENTITY"] == 3
    assert m["IDENTITY"]["CONTACT"] == m["CONTACT"]["IDENTITY"] == 2
    assert m["HEALTH"]["IDENTITY"] == 0
    assert all(m[a][b] == m[b][a] for a in CATEGORIES for b in CATEGORIES)


def test_majority_baseline_predicting_positive():
    b = stats.majority_baseline(fit=DOCS, evaluate=DOCS)["IDENTITY"]
    # predicts 1 everywhere: 3 TP, 1 FP -> accuracy 0.75, F1 = 6/7
    assert b["majority"] == 1 and b["accuracy"] == 0.75
    assert b["f1"] == pytest.approx(6 / 7) and b["f1_defined"]


def test_majority_baseline_predicting_negative_has_zero_undefined_f1():
    b = stats.majority_baseline(fit=DOCS, evaluate=DOCS)["HEALTH"]
    assert b["majority"] == 0 and b["accuracy"] == 1.0
    assert b["f1"] == 0.0 and not b["f1_defined"]


def test_majority_baseline_tie_predicts_positive_and_uses_fit_set_only():
    b = stats.majority_baseline(fit=DOCS, evaluate=[_doc(9)])["CONTACT"]  # 2 of 4 positive
    assert b["majority"] == 1 and b["accuracy"] == 0.0 and b["f1"] == 0.0


def test_flags_mark_low_and_high_prevalence():
    per_set = {"s": stats.set_stats(DOCS),
               "all": stats.set_stats([_doc(i, IDENTITY=1) for i in range(50)])}
    flags = stats.flag_prevalence(per_set)
    assert {"set": "s", "category": "HEALTH", "rate": 0.0, "side": "low"} in flags
    assert {"set": "all", "category": "IDENTITY", "rate": 1.0, "side": "high"} in flags
    assert not any(f["set"] == "s" and f["category"] == "IDENTITY" for f in flags)


def test_markdown_tables_have_one_row_per_category():
    per_set = {"s": stats.set_stats(DOCS)}
    assert len(stats.prevalence_table(per_set).splitlines()) == 2 + 6
    assert "75.00% (3)" in stats.prevalence_table(per_set)
    assert len(stats.cooccurrence_table(stats.cooccurrence(DOCS)).splitlines()) == 2 + 6
    baseline = stats.baseline_table(stats.majority_baseline(DOCS, DOCS))
    assert "undefined, set to 0" in baseline
    assert stats.flags_list([]).startswith("No category")
