"""The calibration split, cut in two halves by group, decided before any calibration result.

Milestone 3 fits calibrators; Milestones 4-5 choose abstention thresholds and conformal
quantiles. If both used the same documents, the calibrated probabilities of those documents
would depend on their own labels, and the exchangeability that conformal prediction needs
would break. So the calibration split (hash buckets 10-19 of privgate's split hash) is cut by
the same hash of the GROUP id:

* ``cal_fit``   buckets 10-14: fitting the calibrators (Milestone 3);
* ``cal_test``  buckets 15-19: thresholds and conformal quantiles (Milestones 4-5).

The two copies of a persona (us / intl) share a group and therefore always land in the same
half.
"""

from __future__ import annotations

from collections.abc import Sequence

from sensclf.splits import LabelledDocument, bucket

FIT_BUCKETS = range(10, 15)
TEST_BUCKETS = range(15, 20)


def group_key(doc: LabelledDocument) -> str:
    return str(doc.meta.get("group_id") or doc.doc_id)


def split_calibration(docs: Sequence[LabelledDocument]
                      ) -> tuple[list[LabelledDocument], list[LabelledDocument]]:
    """(cal_fit, cal_test); raises if a document is not in a calibration bucket."""
    fit, test = [], []
    for d in docs:
        h = bucket(group_key(d))
        if h in FIT_BUCKETS:
            fit.append(d)
        elif h in TEST_BUCKETS:
            test.append(d)
        else:
            raise ValueError(f"{d.doc_id}: bucket {h} is not a calibration bucket")
    return fit, test
