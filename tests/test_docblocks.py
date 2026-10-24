import pytest

from sensclf.docblocks import replace_block, update_file

DOC = "intro\n<!-- t:begin -->\nold\n<!-- t:end -->\noutro\n"


def test_replace_block_swaps_only_the_body():
    assert replace_block(DOC, "t", "new\n") == "intro\n<!-- t:begin -->\nnew\n<!-- t:end -->\noutro\n"


def test_replace_block_is_idempotent():
    once = replace_block(DOC, "t", "x")
    assert replace_block(once, "t", "x") == once


@pytest.mark.parametrize("bad", [
    "no markers",
    "<!-- t:begin -->a<!-- t:end --><!-- t:begin -->b<!-- t:end -->",
    "<!-- t:end -->a<!-- t:begin -->",
])
def test_replace_block_rejects_missing_duplicate_or_reversed_markers(bad):
    with pytest.raises(ValueError):
        replace_block(bad, "t", "x")


def test_update_file_writes_lf(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(DOC, encoding="utf-8")
    update_file(path, {"t": "row"})
    assert b"row" in path.read_bytes() and b"\r\n" not in path.read_bytes()
