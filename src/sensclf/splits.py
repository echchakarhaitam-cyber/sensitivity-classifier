"""Labelled documents, the group-aware splits and the processed JSONL sets.

Split rule (docs/TASK.md), decided before any model exists. From the Nemotron TRAIN file,
with ``h = int(sha256(group_id), 16) % 100``:

* ``dev``          h in [0, 10)   -- exactly privgate's dev split (same key, same hash);
* ``calibration``  h in [10, 20)  -- reserved for calibration and conformal prediction;
* ``train``        h in [20, 100).

The Nemotron TEST file is ``test_shift``, the Gretel English test file ``test_ood``, and the
Gretel English train file, minus documents whose text occurs in the Gretel test file, is
``gretel_pool``. A hash rather than a random draw: a document's split never depends on file
order, on other documents or on a seed, and both renderings of a Nemotron persona (shared
``group_id``) always land together.

Rows are read with privgate's parquet reader, span parsers and document ids, and every
document passes through privgate's :class:`~privgate.data.Document` (id, text and meta
validation; the split key comes from privgate's ``split_key``). sensclf keeps the raw
labels, which privgate's documents do not store (see :mod:`sensclf.labels`).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from privgate.data import Document, gretel, nemotron
from privgate.data._parquet import iter_rows
from privgate.data.splits import split_key

from sensclf.labels import CATEGORIES, GRETEL, NEMOTRON, document_labels, is_valid_gold_span
from sensclf.rawdata import DataIntegrityError

DEV_RANGE = range(10)
CALIBRATION_RANGE = range(10, 20)
NEMOTRON_SPLITS = ("train", "calibration", "dev")
SET_NAMES = ("train", "calibration", "dev", "test_shift", "test_ood", "gretel_pool")
SET_DEFINITIONS: Mapping[str, str] = MappingProxyType({
    "train": "Nemotron train file, h in [20, 100)",
    "calibration": "Nemotron train file, h in [10, 20); calibration/conformal only",
    "dev": "Nemotron train file, h in [0, 10); equal to privgate's dev split",
    "test_shift": "Nemotron test file, all rows (domain-shifted)",
    "test_ood": "Gretel English test file, all rows (out-of-distribution)",
    "gretel_pool": ("Gretel English train file minus texts found in the Gretel test file; "
                    "reserved, not used in Milestones 2-5"),
})
SPLIT_RULE = "h = int(sha256(group_id), 16) % 100; dev h<10, calibration 10<=h<20, train h>=20"


def bucket(key: str) -> int:
    """``int(sha256(key), 16) % 100``: privgate's split hash, kept as the full bucket.

    privgate exposes only the dev/train decision (``assign_split``: dev iff bucket < 10); the
    calibration split needs the bucket itself. tests/test_splits.py checks the agreement.
    """
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100


def nemotron_split(key: str) -> str:
    h = bucket(key)
    if h in DEV_RANGE:
        return "dev"
    if h in CALIBRATION_RANGE:
        return "calibration"
    return "train"


@dataclass(frozen=True)
class LabelledDocument:
    """One document with its 6-bit label vector. ``meta`` is JSON-serialisable."""

    doc_id: str
    text: str
    labels: Mapping[str, int]
    meta: Mapping[str, object]

    def __post_init__(self) -> None:
        if tuple(self.labels) != CATEGORIES or set(self.labels.values()) - {0, 1}:
            raise ValueError(f"{self.doc_id}: labels must map {CATEGORIES} to 0/1")
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    __hash__ = None  # type: ignore[assignment]

    def to_record(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text, "labels": dict(self.labels),
                "meta": dict(self.meta)}

    @classmethod
    def from_record(cls, record: Mapping) -> LabelledDocument:
        labels = {c: record["labels"][c] for c in CATEGORIES}
        return cls(record["doc_id"], record["text"], labels, record["meta"])


@dataclass
class LoadCounts:
    """What was read from one raw file: documents, spans, and spans failing the checks."""

    documents: int = 0
    spans: int = 0
    invalid_spans: int = 0
    raw_labels: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {"documents": self.documents, "spans": self.spans,
                "invalid_spans": self.invalid_spans,
                "raw_label_counts": dict(sorted(self.raw_labels.items()))}


@dataclass(frozen=True)
class _Source:
    dataset: str
    columns: tuple[str, ...]
    text_column: str
    span_column: str
    meta_columns: tuple[str, ...]
    parse: Callable
    doc_id: Callable[[Mapping], str]
    group_column: str | None


def _source(kind: str, gretel_prefix: str) -> _Source:
    if kind == "nemotron":
        return _Source(NEMOTRON, nemotron.COLUMNS, "text", "spans", nemotron.META_COLUMNS,
                       nemotron.parse_spans, nemotron.doc_id, "uid")
    if kind == "gretel":
        return _Source(GRETEL, gretel.COLUMNS, "generated_text", "pii_spans",
                       gretel.META_COLUMNS, gretel.parse_spans,
                       lambda row: gretel.doc_id(row, gretel_prefix), None)
    raise ValueError(f"unknown source kind {kind!r}")


def iter_labelled(path: Path, kind: str, counts: LoadCounts,
                  gretel_prefix: str = gretel.ID_PREFIX) -> Iterator[LabelledDocument]:
    """Stream labelled documents from a raw parquet file (``kind``: nemotron or gretel)."""
    src = _source(kind, gretel_prefix)
    for row in iter_rows(path, src.columns):
        text = row[src.text_column]
        raw = src.parse(row[src.span_column])
        valid = [s for s in raw if is_valid_gold_span(s, text)]
        meta = {c: row[c] for c in src.meta_columns if row[c] is not None}
        if src.group_column:
            meta["group_id"] = row[src.group_column]
        doc = Document(src.doc_id(row), text, (), src.dataset, meta)  # privgate's validation
        counts.documents += 1
        counts.spans += len(raw)
        counts.invalid_spans += len(raw) - len(valid)
        counts.raw_labels.update(s.label for s in valid)
        yield LabelledDocument(
            doc.doc_id, text, document_labels(src.dataset, text, valid),
            {**doc.meta, "source": src.dataset, "split_key": split_key(doc),
             "raw_labels": sorted({s.label for s in valid})},
        )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BuildResult:
    sets: Mapping[str, list[LabelledDocument]]
    counts: Mapping[str, LoadCounts]
    dropped_duplicates: tuple[str, ...]   # gretel_pool doc_ids whose text is in the test file


def build_sets(paths: Mapping[str, Path]) -> BuildResult:
    """All six sets from the four raw files (short names as in :mod:`sensclf.rawdata`)."""
    counts = {name: LoadCounts() for name in paths}
    sets: dict[str, list[LabelledDocument]] = {name: [] for name in SET_NAMES}
    for doc in iter_labelled(paths["nemotron_train"], "nemotron", counts["nemotron_train"]):
        sets[nemotron_split(doc.meta["split_key"])].append(doc)
    sets["test_shift"] = list(
        iter_labelled(paths["nemotron_test"], "nemotron", counts["nemotron_test"]))
    sets["test_ood"] = list(
        iter_labelled(paths["gretel_en_test"], "gretel", counts["gretel_en_test"]))
    test_texts = {text_sha256(d.text) for d in sets["test_ood"]}
    dropped = []
    for doc in iter_labelled(paths["gretel_en_train"], "gretel", counts["gretel_en_train"],
                             gretel.TRAIN_ID_PREFIX):
        if text_sha256(doc.text) in test_texts:
            dropped.append(doc.doc_id)
        else:
            sets["gretel_pool"].append(doc)
    check_disjoint(sets)
    return BuildResult(sets, counts, tuple(dropped))


def check_disjoint(sets: Mapping[str, Sequence[LabelledDocument]]) -> None:
    """Raise if a group_id occurs in two Nemotron splits or a doc_id occurs twice anywhere."""
    owner: dict[str, str] = {}
    for name in NEMOTRON_SPLITS:
        for doc in sets.get(name, ()):
            group = str(doc.meta["group_id"])
            if owner.setdefault(group, name) != name:
                raise DataIntegrityError(f"group {group} is in {owner[group]} and {name}")
    ids = Counter(doc.doc_id for docs in sets.values() for doc in docs)
    repeated = [i for i, n in ids.items() if n > 1]
    if repeated:
        raise DataIntegrityError(f"{len(repeated)} doc_ids occur twice, e.g. {repeated[0]}")


def write_jsonl(docs: Iterable[LabelledDocument], path: Path) -> str:
    """One JSON object per line (UTF-8, LF, sorted keys); return the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for doc in docs:
            line = json.dumps(doc.to_record(), ensure_ascii=False, sort_keys=True) + "\n"
            out.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[LabelledDocument]:
    with Path(path).open(encoding="utf-8") as fh:
        return [LabelledDocument.from_record(json.loads(line)) for line in fh if line.strip()]
