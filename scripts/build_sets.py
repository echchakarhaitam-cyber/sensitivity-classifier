"""Build the six processed sets from the verified raw files.

Usage:  python scripts/build_sets.py     (raw files from SENSCLF_RAW_DIR, see docs/DATA.md)

Writes data/processed/<set>.jsonl (gitignored) and records their SHA-256, sizes, the split
rule, the Gretel train/test duplicates and per-file load counts in data/manifest.json.
"""

from __future__ import annotations

import sys

from sensclf.processed import build_and_save


def main() -> int:
    manifest = build_and_save()
    for name, entry in manifest["processed_sets"].items():
        print(f"{name:12s} {entry['documents']:>7,d} docs  sha256={entry['sha256']}")
    dup = manifest["gretel_pool_dropped_duplicates"]["count"]
    print(f"gretel_pool: {dup} documents dropped (text also in the Gretel test file)")
    for name, counts in manifest["load_counts"].items():
        print(f"{name}: {counts['spans']} spans, {counts['invalid_spans']} failed the checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
