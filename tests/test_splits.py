import pytest
from fixtures import raw_fixture
from privgate.data.splits import assign_split

from sensclf import processed
from sensclf.labels import CATEGORIES
from sensclf.rawdata import DataIntegrityError
from sensclf.splits import (
    SET_NAMES,
    LabelledDocument,
    bucket,
    build_sets,
    check_disjoint,
    nemotron_split,
    read_jsonl,
    write_jsonl,
)

ZERO = dict.fromkeys(CATEGORIES, 0)


def _doc(doc_id, group="g", **labels):
    return LabelledDocument(doc_id, "text", {**ZERO, **labels}, {"group_id": group})


def test_bucket_agrees_with_privgate_dev_split():
    for i in range(3000):
        key = f"key-{i}"
        assert (bucket(key) < 10) == (assign_split(key) == "dev")


def test_split_ranges_and_rough_proportions():
    splits = [nemotron_split(f"k{i}") for i in range(20000)]
    for i in range(200):
        h = bucket(f"k{i}")
        expected = "dev" if h < 10 else "calibration" if h < 20 else "train"
        assert splits[i] == expected
    assert 0.08 < splits.count("dev") / len(splits) < 0.12
    assert 0.08 < splits.count("calibration") / len(splits) < 0.12


def test_split_is_a_pure_function_of_the_key():
    keys = [f"k{i}" for i in range(500)]
    assert [nemotron_split(k) for k in keys] == [nemotron_split(k) for k in reversed(keys)][::-1]


def test_build_sets_on_fixture(tmp_path):
    sets = build_sets(raw_fixture(tmp_path)).sets
    assert set(sets) == set(SET_NAMES)
    assert [len(sets[n]) for n in ("train", "calibration", "dev")] == [6, 4, 4]
    assert len(sets["test_shift"]) == 4 and len(sets["test_ood"]) == 2
    # both renderings of each persona stay together
    for name in ("train", "calibration", "dev"):
        for doc in sets[name]:
            assert nemotron_split(doc.meta["group_id"]) == name
            assert doc.meta["split_key"] == doc.meta["group_id"]


def test_build_sets_labels_documents_from_raw_spans(tmp_path):
    sets = build_sets(raw_fixture(tmp_path)).sets
    us = next(d for d in sets["test_shift"] if d.doc_id.endswith(":us"))
    intl = next(d for d in sets["test_shift"] if d.doc_id.endswith(":intl"))
    assert dict(us.labels) == {**ZERO, "IDENTITY": 1, "CONTACT": 1}
    assert us.meta["raw_labels"] == ["email", "first_name"]
    assert dict(intl.labels) == ZERO
    iban = next(d for d in sets["test_ood"] if d.doc_id == "gretel:0")
    assert dict(iban.labels) == {**ZERO, "FINANCIAL": 1}


def test_gretel_pool_drops_texts_of_the_test_file(tmp_path):
    result = build_sets(raw_fixture(tmp_path))
    assert result.dropped_duplicates == ("gretel-train:0",)
    assert [d.doc_id for d in result.sets["gretel_pool"]] == ["gretel-train:1"]
    assert result.counts["gretel_en_train"].documents == 2


def test_build_is_deterministic(tmp_path):
    paths = raw_fixture(tmp_path)
    a, b = build_sets(paths), build_sets(paths)
    for name in SET_NAMES:
        assert write_jsonl(a.sets[name], tmp_path / "a.jsonl") == \
               write_jsonl(b.sets[name], tmp_path / "b.jsonl")


def test_check_disjoint_rejects_a_group_in_two_splits():
    with pytest.raises(DataIntegrityError, match="group"):
        check_disjoint({"train": [_doc("a", "g1")], "dev": [_doc("b", "g1")]})


def test_check_disjoint_rejects_a_repeated_doc_id():
    with pytest.raises(DataIntegrityError, match="doc_id"):
        check_disjoint({"test_shift": [_doc("a")], "test_ood": [_doc("a")]})
    with pytest.raises(DataIntegrityError):
        check_disjoint({"test_ood": [_doc("a"), _doc("a")]})


def test_labelled_document_validates_labels():
    with pytest.raises(ValueError):
        LabelledDocument("d", "t", {"IDENTITY": 1}, {})
    with pytest.raises(ValueError):
        LabelledDocument("d", "t", {**ZERO, "HEALTH": 2}, {})


def test_jsonl_roundtrip(tmp_path):
    docs = [_doc("a", IDENTITY=1), _doc("b")]
    write_jsonl(docs, tmp_path / "s.jsonl")
    back = read_jsonl(tmp_path / "s.jsonl")
    assert [d.to_record() for d in back] == [d.to_record() for d in docs]


def test_saved_sets_load_verified_and_tampering_aborts(tmp_path):
    out = tmp_path / "processed"
    manifest = processed.save_sets(build_sets(raw_fixture(tmp_path)), {"files": []}, out)
    assert manifest["gretel_pool_dropped_duplicates"]["count"] == 1
    assert len(processed.load_set("dev", out, manifest)) == 4
    path = out / "dev.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("Ann", "Bob"), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        processed.load_set("dev", out, manifest)
    with pytest.raises(DataIntegrityError):
        processed.load_set("unknown", out, manifest)


# --- integration tests on the real processed sets -------------------------------------------

@pytest.fixture(scope="module")
def real_sets():
    return {name: processed.load_set(name) for name in SET_NAMES}


@pytest.mark.data
def test_real_nemotron_splits_are_group_disjoint_and_complete(real_sets):
    check_disjoint(real_sets)
    total = sum(len(real_sets[n]) for n in ("train", "calibration", "dev"))
    assert total == 100_000
    assert len(real_sets["test_shift"]) == 100_000


@pytest.mark.data
def test_real_dev_equals_privgate_dev(real_sets):
    from privgate.data.splits import iter_file

    from sensclf.rawdata import raw_dir_from_env, verified_raw_paths

    verified_raw_paths(allow_download=False)
    # privgate expects its data root (the parent of a "raw" directory).
    pairs, _ = iter_file(raw_dir_from_env().parent, "nemotron_train_file")
    privgate_dev = {doc.doc_id for split, doc in pairs if split == "dev"}
    assert {d.doc_id for d in real_sets["dev"]} == privgate_dev


@pytest.mark.data
def test_real_gretel_pool_drops_the_known_duplicates(real_sets):
    from sensclf.rawdata import load_manifest

    assert load_manifest()["gretel_pool_dropped_duplicates"]["count"] == 11
    test_texts = {d.text for d in real_sets["test_ood"]}
    assert not any(d.text in test_texts for d in real_sets["gretel_pool"])
