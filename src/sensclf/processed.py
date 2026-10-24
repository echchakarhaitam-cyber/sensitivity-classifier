"""Writing the processed sets with their hashes, and reading them back verified.

Every later milestone loads a set through :func:`load_set`, which refuses a file whose
SHA-256 differs from ``data/manifest.json``: results can then only ever come from the exact
sets defined in Milestone 1.
"""

from __future__ import annotations

from pathlib import Path

from sensclf.rawdata import (
    MANIFEST_PATH,
    REPO_ROOT,
    DataIntegrityError,
    load_manifest,
    sha256_of,
    verified_raw_paths,
    write_manifest,
)
from sensclf.splits import (
    SET_DEFINITIONS,
    SET_NAMES,
    SPLIT_RULE,
    BuildResult,
    LabelledDocument,
    build_sets,
    read_jsonl,
    write_jsonl,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def save_sets(result: BuildResult, manifest: dict, out_dir: Path = PROCESSED_DIR) -> dict:
    """Write every set to ``out_dir/<name>.jsonl``; return the manifest with their entries."""
    entries = {}
    for name in SET_NAMES:
        docs = result.sets[name]
        path = Path(out_dir) / f"{name}.jsonl"
        entries[name] = {
            "path": f"data/processed/{name}.jsonl",
            "sha256": write_jsonl(docs, path),
            "documents": len(docs),
            "definition": SET_DEFINITIONS[name],
        }
    return {
        **manifest,
        "processed_sets": entries,
        "split_rule": SPLIT_RULE,
        "gretel_pool_dropped_duplicates": {
            "count": len(result.dropped_duplicates),
            "doc_ids": sorted(result.dropped_duplicates),
        },
        "load_counts": {name: c.to_dict() for name, c in sorted(result.counts.items())},
    }


def load_set(name: str, processed_dir: Path = PROCESSED_DIR,
             manifest: dict | None = None) -> list[LabelledDocument]:
    """The processed set ``name`` after checking its SHA-256 against the manifest."""
    manifest = load_manifest(MANIFEST_PATH) if manifest is None else manifest
    try:
        expected = manifest["processed_sets"][name]["sha256"]
    except KeyError:
        raise DataIntegrityError(f"manifest has no processed set {name!r}") from None
    path = Path(processed_dir) / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run scripts/build_sets.py")
    digest = sha256_of(path)
    if digest != expected:
        raise DataIntegrityError(f"{path}: sha256 {digest} differs from the manifest")
    return read_jsonl(path)


def build_and_save(manifest_path: Path = MANIFEST_PATH, out_dir: Path = PROCESSED_DIR,
                   raw_dir: Path | None = None) -> dict:
    """Verify the raw files, build all sets, write them and record them in the manifest."""
    manifest = load_manifest(manifest_path)
    paths = verified_raw_paths(raw_dir, manifest)
    updated = save_sets(build_sets(paths), manifest, out_dir)
    write_manifest(updated, manifest_path)
    return updated
