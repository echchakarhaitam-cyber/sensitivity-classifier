"""Verified access to the pinned raw dataset files.

The raw parquet files are not copied into this repository: they are read from a configurable
directory (``SENSCLF_RAW_DIR``, default ``../llm-privacy-gateway/data/raw`` next to this repo),
which uses privgate's layout ``<raw dir>/<repo with "/" -> "__">/<file name>``.

Why verify every time: every number in this project is only meaningful for the exact bytes
pinned in ``data/manifest.json`` (entries copied unchanged from privgate's manifest). A file
whose SHA-256 differs aborts with :class:`DataIntegrityError`; nothing is ever re-hashed into
the manifest. A missing file is downloaded from its pinned URL (revision in the URL) and must
then match the recorded size and hash too.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_RAW_DIR = "SENSCLF_RAW_DIR"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT.parent / "llm-privacy-gateway" / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"
CHUNK = 1 << 20

# Short names used across sensclf -> (repo, path inside the repo) of the manifest entry.
RAW_FILES: Mapping[str, tuple[str, str]] = {
    "nemotron_train": ("nvidia/Nemotron-PII", "data/train-00000-of-00001.parquet"),
    "nemotron_test": ("nvidia/Nemotron-PII", "data/test-00000-of-00001.parquet"),
    "gretel_en_test": (
        "gretelai/synthetic_pii_finance_multilingual", "data/English_test-00000-of-00001.parquet",
    ),
    "gretel_en_train": (
        "gretelai/synthetic_pii_finance_multilingual", "data/English_train-00000-of-00001.parquet",
    ),
}


class DataIntegrityError(RuntimeError):
    """A raw or processed file does not match data/manifest.json."""


@dataclass(frozen=True)
class RawFile:
    """One pinned upstream file as recorded in the manifest."""

    name: str
    repo: str
    path: str
    size: int
    sha256: str
    url: str
    license: str
    revision: str

    def local_path(self, raw_dir: Path) -> Path:
        return Path(raw_dir) / self.repo.replace("/", "__") / Path(self.path).name


def raw_dir_from_env(env: Mapping[str, str] | None = None) -> Path:
    """``SENSCLF_RAW_DIR`` if set (relative paths resolve against the cwd), else the default."""
    value = (os.environ if env is None else env).get(ENV_RAW_DIR)
    return Path(value).resolve() if value else DEFAULT_RAW_DIR.resolve()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    """Sorted keys, two-space indent, LF: the file diffs cleanly in git."""
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def raw_files(manifest: dict) -> dict[str, RawFile]:
    """The four pinned files by short name; raises if the manifest lacks one."""
    entries = {(e["repo"], e["path"]): e for e in manifest.get("files", [])}
    out = {}
    for name, key in RAW_FILES.items():
        if key not in entries:
            raise DataIntegrityError(f"manifest has no entry for {name} {key}")
        e = entries[key]
        out[name] = RawFile(name, e["repo"], e["path"], e["size"], e["sha256"], e["url"],
                            e["license"], e["revision"])
    return out


def download(url: str, dest: Path, expected_size: int) -> None:
    """Stream ``url`` to ``dest`` through ``dest.part``; rename only after the size check.

    An interrupted or truncated download therefore never leaves a file that looks complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    written = 0
    with urllib.request.urlopen(url, timeout=60) as resp, part.open("wb") as out:
        for block in iter(lambda: resp.read(CHUNK), b""):
            out.write(block)
            written += len(block)
    if written != expected_size:
        part.unlink()
        raise DataIntegrityError(f"{url}: expected {expected_size} bytes, got {written}")
    os.replace(part, dest)


def verify_file(spec: RawFile, raw_dir: Path, *, allow_download: bool = True,
                url: str | None = None) -> Path:
    """Path of the verified local copy of ``spec``; download it first if it is missing.

    ``url`` overrides the download URL (tests use a file:// URL instead of the network).
    """
    path = spec.local_path(raw_dir)
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"{path} is missing and downloading is disabled")
        download(url or spec.url, path, spec.size)
    digest = sha256_of(path)
    if digest != spec.sha256:
        raise DataIntegrityError(
            f"{path}: sha256 {digest} differs from the manifest ({spec.sha256}); refusing to use it"
        )
    return path


def verified_raw_paths(raw_dir: Path | None = None, manifest: dict | None = None,
                       *, allow_download: bool = True) -> dict[str, Path]:
    """Verify (and fetch if missing) every pinned raw file; short name -> local path."""
    raw_dir = raw_dir_from_env() if raw_dir is None else Path(raw_dir)
    manifest = load_manifest() if manifest is None else manifest
    return {name: verify_file(spec, raw_dir, allow_download=allow_download)
            for name, spec in raw_files(manifest).items()}
