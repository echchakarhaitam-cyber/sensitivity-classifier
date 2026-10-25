import numpy as np
import pytest

from sensclf.guard import CalibrationLeakError
from sensclf.models.baselines import DetectorBaseline, DomainModel
from sensclf.models.text import TextModel
from sensclf.selection import (
    SelectionSetError,
    default_threshold,
    dev_thresholds,
    macro_ap,
    operating_points,
    per_category_ap,
    select_by_dev,
)

Y = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0]])
GOOD = Y * 0.8 + 0.1                       # perfect ranking
BAD = 1 - GOOD                             # reversed ranking
IDS = ["a", "b", "c", "d"]
NONE: frozenset[str] = frozenset()


def test_macro_ap_excludes_undefined_and_uncovered_columns():
    assert per_category_ap(Y, GOOD)["SPECIAL_CATEGORY"] is None   # no positive in column 5
    assert macro_ap(Y, GOOD) == pytest.approx(1.0)
    partial = GOOD.copy()
    partial[:, 0] = np.nan
    assert per_category_ap(Y, partial)["IDENTITY"] is None


def test_select_by_dev_picks_best_and_breaks_ties_to_smaller_c():
    best, rows = select_by_dev([(1.0, BAD), (10.0, GOOD), (0.1, GOOD)], Y, set_name="dev",
                               doc_ids=IDS, forbidden=NONE)
    assert best == 0.1 and len(rows) == 3


@pytest.mark.parametrize("set_name", ["test_shift", "test_ood", "calibration", "train"])
def test_selection_refuses_any_set_but_dev(set_name):
    with pytest.raises(SelectionSetError):
        select_by_dev([(1.0, GOOD)], Y, set_name=set_name, doc_ids=IDS, forbidden=NONE)
    with pytest.raises(SelectionSetError):
        dev_thresholds(GOOD, Y, set_name=set_name, doc_ids=IDS, forbidden=NONE)


def test_selection_refuses_calibration_documents():
    with pytest.raises(CalibrationLeakError):
        dev_thresholds(GOOD, Y, set_name="dev", doc_ids=IDS, forbidden=frozenset({"c"}))
    with pytest.raises(CalibrationLeakError):
        select_by_dev([(1.0, GOOD)], Y, set_name="dev", doc_ids=IDS, forbidden=frozenset({"a"}))


def test_dev_thresholds_values():
    t = dev_thresholds(GOOD, Y, set_name="dev", doc_ids=IDS, forbidden=NONE)
    assert t["IDENTITY"] == pytest.approx(0.9)       # positives scored 0.9, negatives 0.1
    assert t["SPECIAL_CATEGORY"] is None


def test_default_thresholds_per_model_kind():
    assert default_threshold(TextModel("word_lr", 1.0)) == 0.5
    assert default_threshold(TextModel("word_svc", 1.0)) == 0.0
    assert default_threshold(DomainModel()) == 0.5
    assert default_threshold(DetectorBaseline()) == 0.5


def test_operating_points_for_tunable_models_use_dev_f1():
    ops = operating_points(TextModel("word_svc", 1.0), GOOD, Y, set_name="dev", doc_ids=IDS,
                           forbidden=NONE)
    assert ops["default"]["IDENTITY"] == 0.0
    assert ops["dev_f1"]["IDENTITY"] == pytest.approx(0.9)


def test_operating_points_keep_the_detectors_own_decision():
    S = np.full(Y.shape, np.nan)
    S[:, 1] = [0.0, 1.0, 1.0, 0.0]                  # CONTACT only
    ops = operating_points(DetectorBaseline(), S, Y, set_name="dev", doc_ids=IDS,
                           forbidden=NONE)
    assert ops["dev_f1"] == ops["default"]
    assert ops["default"]["CONTACT"] == 0.5 and ops["default"]["IDENTITY"] is None


def test_operating_points_refuse_non_dev_sets():
    with pytest.raises(SelectionSetError):
        operating_points(DomainModel(), GOOD, Y, set_name="test_ood", doc_ids=IDS,
                         forbidden=NONE)
