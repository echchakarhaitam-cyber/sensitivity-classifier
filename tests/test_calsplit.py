import pytest

from sensclf.calsplit import FIT_BUCKETS, TEST_BUCKETS, split_calibration
from sensclf.labels import CATEGORIES
from sensclf.splits import LabelledDocument, bucket


def _doc(i: int, group: str) -> LabelledDocument:
    return LabelledDocument(doc_id=f"d{i}", text="t", labels={c: 0 for c in CATEGORIES},
                            meta={"group_id": group})


def _groups_in(buckets, n):
    out, i = [], 0
    while len(out) < n:
        g = f"g{i}"
        if bucket(g) in buckets:
            out.append(g)
        i += 1
    return out


def test_split_by_group_and_disjoint():
    fit_g = _groups_in(FIT_BUCKETS, 5)
    test_g = _groups_in(TEST_BUCKETS, 5)
    docs = [_doc(i, g) for i, g in enumerate(fit_g + test_g + fit_g)]
    fit, test = split_calibration(docs)
    assert {d.meta["group_id"] for d in fit} == set(fit_g)
    assert {d.meta["group_id"] for d in test} == set(test_g)
    assert len(fit) == 10 and len(test) == 5      # both copies of a group stay together


def test_non_calibration_bucket_raises():
    g = _groups_in(range(10), 1)[0]
    with pytest.raises(ValueError):
        split_calibration([_doc(0, g)])
