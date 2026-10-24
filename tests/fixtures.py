"""Tiny synthetic parquet files with the columns privgate's loaders read. All text is invented."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sensclf.splits import nemotron_split

NEMOTRON_COLUMNS = ("uid", "domain", "document_type", "document_format", "locale", "text", "spans")


def span(text: str, sub: str, label: str) -> dict:
    start = text.index(sub)
    return {"start": start, "end": start + len(sub), "text": sub, "label": label}


def uid_in(split: str, n: int = 1, prefix: str = "u") -> list[str]:
    """The first ``n`` uids ``<prefix><i>`` whose hash lands in ``split`` (deterministic)."""
    found, i = [], 0
    while len(found) < n:
        if nemotron_split(f"{prefix}{i}") == split:
            found.append(f"{prefix}{i}")
        i += 1
    return found


def nemotron_rows(uids: list[str]) -> list[dict]:
    """Two renderings (us, intl) per uid; us names a person and an email, intl has no span."""
    rows = []
    for uid in uids:
        for locale in ("us", "intl"):
            if locale == "us":
                text = f"Patient Ann Lee ({uid}) wrote to ann@example.org."
                spans = [span(text, "Ann", "first_name"), span(text, "ann@example.org", "email")]
            else:
                text = f"Customer {uid} logged in."
                spans = []
            rows.append({"uid": uid, "domain": "Healthcare", "document_type": "note",
                         "document_format": "unstructured", "locale": locale, "text": text,
                         "spans": repr(spans)})
    return rows


def write_nemotron(path: Path, rows: list[dict]) -> Path:
    schema = pa.schema([(c, pa.string()) for c in NEMOTRON_COLUMNS])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def gretel_row(index: int, text: str, spans: list[dict]) -> dict:
    return {"__index_level_0__": index, "generated_text": text,
            "pii_spans": json.dumps([{k: s[k] for k in ("start", "end", "label")} for s in spans]),
            "domain": "finance", "document_type": "Email", "expanded_type": "test",
            "language": "English"}


def write_gretel(path: Path, rows: list[dict]) -> Path:
    schema = pa.schema([("__index_level_0__", pa.int64())] + [
        (c, pa.string()) for c in ("generated_text", "pii_spans", "domain", "document_type",
                                   "expanded_type", "language")])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def raw_fixture(tmp_path: Path) -> dict[str, Path]:
    """Four raw files; the Gretel train file repeats one Gretel test text."""
    train_uids = uid_in("train", 3) + uid_in("calibration", 2) + uid_in("dev", 2)
    iban_text = "Transfer to IBAN DE00 1234 now."
    test_rows = [gretel_row(0, iban_text, [span(iban_text, "DE00 1234", "iban")]),
                 gretel_row(1, "Nothing here.", [])]
    pin_text = "Your PIN is 9999."
    train_rows = [gretel_row(0, "Nothing here.", []),          # duplicate of test row 1
                  gretel_row(1, pin_text, [span(pin_text, "9999", "account_pin")])]
    return {
        "nemotron_train": write_nemotron(tmp_path / "ntrain.parquet", nemotron_rows(train_uids)),
        "nemotron_test": write_nemotron(tmp_path / "ntest.parquet", nemotron_rows(["t1", "t2"])),
        "gretel_en_test": write_gretel(tmp_path / "gtest.parquet", test_rows),
        "gretel_en_train": write_gretel(tmp_path / "gtrain.parquet", train_rows),
    }
