import hashlib
from pathlib import Path

import pytest

from sensclf import rawdata
from sensclf.rawdata import DataIntegrityError, RawFile

CONTENT = b"synthetic parquet stand-in"


def _spec(content: bytes = CONTENT, name: str = "nemotron_train") -> RawFile:
    return RawFile(name, "org/data-set", "data/file.parquet", len(content),
                   hashlib.sha256(content).hexdigest(), "https://example.invalid/never-used",
                   "CC-BY-4.0", "rev")


def _manifest(tmp_path: Path) -> dict:
    """A manifest naming all four files, each with its own synthetic content on disk."""
    files = []
    for name, (repo, path) in rawdata.RAW_FILES.items():
        content = f"bytes of {name}".encode()
        dest = tmp_path / repo.replace("/", "__") / Path(path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        files.append({"repo": repo, "path": path, "size": len(content),
                      "sha256": hashlib.sha256(content).hexdigest(), "url": "x",
                      "license": "test", "revision": "r"})
    return {"files": files}


def test_local_path_uses_privgate_layout(tmp_path):
    assert _spec().local_path(tmp_path) == tmp_path / "org__data-set" / "file.parquet"


def test_verify_file_accepts_matching_hash(tmp_path):
    spec = _spec()
    path = spec.local_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(CONTENT)
    assert rawdata.verify_file(spec, tmp_path, allow_download=False) == path


def test_verify_file_aborts_on_tampered_file(tmp_path):
    spec = _spec()
    path = spec.local_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(CONTENT.replace(b"synthetic", b"tampered!"))  # same size, other bytes
    with pytest.raises(DataIntegrityError, match="differs from the manifest"):
        rawdata.verify_file(spec, tmp_path, allow_download=False)


def test_missing_file_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rawdata.verify_file(_spec(), tmp_path, allow_download=False)


def test_missing_file_is_downloaded_then_verified(tmp_path):
    source = tmp_path / "upstream.bin"
    source.write_bytes(CONTENT)
    raw_dir = tmp_path / "raw"
    path = rawdata.verify_file(_spec(), raw_dir, url=source.as_uri())
    assert path.read_bytes() == CONTENT
    assert not path.with_name(path.name + ".part").exists()


def test_download_with_wrong_size_leaves_no_file(tmp_path):
    source = tmp_path / "upstream.bin"
    source.write_bytes(CONTENT + b"extra")
    raw_dir = tmp_path / "raw"
    with pytest.raises(DataIntegrityError, match="expected"):
        rawdata.verify_file(_spec(), raw_dir, url=source.as_uri())
    assert list(raw_dir.rglob("*.parquet*")) == []


def test_downloaded_file_with_wrong_hash_aborts(tmp_path):
    source = tmp_path / "upstream.bin"
    source.write_bytes(CONTENT.upper())  # right size, wrong bytes
    with pytest.raises(DataIntegrityError):
        rawdata.verify_file(_spec(), tmp_path / "raw", url=source.as_uri())


def test_verified_raw_paths_covers_all_four_files(tmp_path):
    paths = rawdata.verified_raw_paths(tmp_path, _manifest(tmp_path), allow_download=False)
    assert set(paths) == set(rawdata.RAW_FILES)


def test_verified_raw_paths_aborts_when_any_file_is_tampered(tmp_path):
    manifest = _manifest(tmp_path)
    repo, path = rawdata.RAW_FILES["gretel_en_test"]
    (tmp_path / repo.replace("/", "__") / Path(path).name).write_bytes(b"bytes of gretel_xx_test")
    with pytest.raises(DataIntegrityError):
        rawdata.verified_raw_paths(tmp_path, manifest, allow_download=False)


def test_manifest_without_an_entry_is_rejected():
    with pytest.raises(DataIntegrityError, match="no entry"):
        rawdata.raw_files({"files": []})


def test_raw_dir_from_env_and_default(tmp_path):
    assert rawdata.raw_dir_from_env({rawdata.ENV_RAW_DIR: str(tmp_path)}) == tmp_path.resolve()
    assert rawdata.raw_dir_from_env({}) == rawdata.DEFAULT_RAW_DIR.resolve()


def test_write_manifest_roundtrip(tmp_path):
    path = tmp_path / "m.json"
    rawdata.write_manifest({"b": 1, "a": [1]}, path)
    assert rawdata.load_manifest(path) == {"a": [1], "b": 1}
    assert b"\r\n" not in path.read_bytes()


def test_committed_manifest_pins_four_files_with_licences():
    specs = rawdata.raw_files(rawdata.load_manifest())
    assert {s.license for s in specs.values()} == {"CC-BY-4.0", "Apache-2.0"}
    for s in specs.values():
        assert s.revision in s.url and len(s.sha256) == 64


@pytest.mark.data
def test_real_raw_files_match_the_manifest():
    paths = rawdata.verified_raw_paths(allow_download=False)
    assert all(p.exists() for p in paths.values())
