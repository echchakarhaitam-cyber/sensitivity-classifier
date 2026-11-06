import json

import pytest

from sensclf.labels import CATEGORIES
from sensclf.rawdata import REPO_ROOT
from sensclf.report import wrong_comparisons
from sensclf.uncertainty_report import pct, render_uncertainty

RES = REPO_ROOT / "results"
FILES = ("m3/calibration.json", "m4/selective.json", "m5/conformal.json")


def test_pct():
    assert pct(0.1234) == "12.3%" and pct(None) == "n/a"


@pytest.mark.skipif(not all((RES / f).exists() for f in FILES), reason="results not generated")
def test_committed_doc_matches_results_and_is_consistent():
    inputs = [json.loads((RES / f).read_text(encoding="utf-8")) for f in FILES]
    text = render_uncertainty(*inputs)
    assert wrong_comparisons(text) == []
    for c in CATEGORIES:
        assert c in text
    doc = REPO_ROOT / "docs" / "results-uncertainty.md"
    assert doc.read_text(encoding="utf-8") == text, "regenerate with make_uncertainty_results"


def test_violation_is_counted_and_bolded():
    sets = {s: {"aurc": 0.1, "oracle_aurc": 0.01, "error_rate_full_coverage": 0.1,
                "defined": True, "at_coverage": {}} for s in ("dev", "test_shift", "test_ood")}
    applied = {s: {"coverage": 0.5, "risk": 0.02 if s == "test_ood" else 0.005, "accepted": 5,
                   "risk_ci": None, "violation": s == "test_ood"}
               for s in ("dev", "test_shift", "test_ood")}
    thr = {"0.01": {"choice": {"certified": True, "cal_coverage": 0.5, "threshold": 0.9},
                    "applied": applied}}
    sel = {"model": "m", "calibration": "isotonic", "delta": 0.05, "targets": [0.01],
           "documents": {"cal_test": 10},
           "categories": {c: {"sets": sets, "thresholds": thr} for c in CATEGORIES}}
    sel["document_level"] = {"0.01": {s: {"coverage": 0.1, "doc_error_rate": 0.0}
                                      for s in ("dev", "test_shift", "test_ood")}}
    from sensclf.uncertainty_report import _selective
    text = "\n".join(_selective(sel))
    assert "**2.0%**" in text
    assert "exceeds the target for 6 of 6 certified thresholds on categories" in text
    assert "exceeds the target for 0 of 6" in text
