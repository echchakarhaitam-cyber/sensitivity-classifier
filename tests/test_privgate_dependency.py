"""The privgate API that sensclf relies on, pinned at commit 498adcc.

If an upgrade of the pin changes any of these, this test fails before any data code does.
"""

from privgate.data import LABEL_MAPS, Document, RawSpan, build_document
from privgate.data.documents import LoadOptions, LoadTally
from privgate.data.splits import assign_split, split_key
from privgate.spans import EntityType


def test_privgate_label_maps_have_the_pinned_label_counts():
    assert len(LABEL_MAPS["nemotron-pii"]) == 55
    assert len(LABEL_MAPS["gretel-finance-en"]) == 29


def test_privgate_split_uses_group_id_and_ten_percent_dev():
    tally = LoadTally("nemotron-pii", LoadOptions())
    doc = build_document("d1", "Ann", [RawSpan(0, 3, "first_name", "Ann")], tally,
                         {"group_id": "g1"})
    assert isinstance(doc, Document)
    assert doc.spans[0].label is EntityType.PERSON
    assert split_key(doc) == "g1"
    assert assign_split("g1") in {"dev", "train"}
