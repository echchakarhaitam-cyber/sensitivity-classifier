"""Statistics of the six processed sets -> results/data_stats.json and docs/DATA.md tables.

Usage:  python scripts/data_stats.py     (after scripts/build_sets.py)

Every set is loaded through sensclf.processed.load_set, i.e. only after its SHA-256 matches
data/manifest.json; the hashes are copied into the output for provenance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sensclf import stats
from sensclf.docblocks import update_file
from sensclf.processed import load_set
from sensclf.rawdata import REPO_ROOT, load_manifest
from sensclf.splits import SET_NAMES

OUT = REPO_ROOT / "results" / "data_stats.json"
DATA_DOC = REPO_ROOT / "docs" / "DATA.md"


def compute() -> dict:
    manifest = load_manifest()
    sets = {name: load_set(name, manifest=manifest) for name in SET_NAMES}
    per_set = {name: stats.set_stats(docs) for name, docs in sets.items()}
    return {
        "set_sha256": {n: manifest["processed_sets"][n]["sha256"] for n in SET_NAMES},
        "sets": per_set,
        "cooccurrence_train": stats.cooccurrence(sets["train"]),
        "majority_baseline_dev": stats.majority_baseline(sets["train"], sets["dev"]),
        "prevalence_flags": stats.flag_prevalence(per_set),
        "gretel_pool_dropped_duplicates": manifest["gretel_pool_dropped_duplicates"]["count"],
        "load_counts": {k: {f: v[f] for f in ("documents", "spans", "invalid_spans")}
                        for k, v in manifest["load_counts"].items()},
    }


def render(result: dict) -> dict[str, str]:
    loads = "\n".join(f"| {k} | {v['documents']:,} | {v['spans']:,} | {v['invalid_spans']:,} |"
                      for k, v in result["load_counts"].items())
    stats_md = "\n\n".join([
        "### Raw files read\n\n| raw file | documents | gold spans | spans failing the checks |"
        "\n|---|---:|---:|---:|\n" + loads,
        "### Sets\n\n" + stats.sets_table(result["sets"]),
        "### Category prevalence (rate and count)\n\n" + stats.prevalence_table(result["sets"]),
        "### Co-occurrence matrix (train; documents positive for both)\n\n"
        + stats.cooccurrence_table(result["cooccurrence_train"]),
        "### Trivial baseline: always predict the train majority class, evaluated on dev\n\n"
        + stats.baseline_table(result["majority_baseline_dev"]),
        "### Prevalence flags (< 2% or > 98%; reported, definition unchanged)\n\n"
        + stats.flags_list(result["prevalence_flags"]),
    ])
    dup = result["gretel_pool_dropped_duplicates"]
    return {"stats": stats_md,
            "duplicates": f"Documents dropped from `gretel_pool`: **{dup}**."}


def main() -> int:
    result = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                   newline="\n")
    blocks = render(result)
    update_file(DATA_DOC, blocks)
    print(blocks["duplicates"])
    print(blocks["stats"])
    print(f"wrote {Path(OUT).relative_to(REPO_ROOT)} and updated docs/DATA.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
