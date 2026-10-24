"""Replace generated blocks in markdown docs.

A block is the text between ``<!-- name:begin -->`` and ``<!-- name:end -->``. Docs hold no
hand-typed numbers or tables: scripts regenerate the blocks, so docs cannot drift from code.
"""

from __future__ import annotations

from pathlib import Path


def replace_block(text: str, name: str, content: str) -> str:
    """``text`` with the body of block ``name`` replaced; the markers must occur exactly once."""
    begin, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"block {name!r} must have exactly one begin and one end marker")
    head, rest = text.split(begin)
    if end not in rest:
        raise ValueError(f"block {name!r}: end marker before begin marker")
    _, tail = rest.split(end)
    return f"{head}{begin}\n{content.strip()}\n{end}{tail}"


def update_file(path: Path, blocks: dict[str, str]) -> None:
    text = Path(path).read_text(encoding="utf-8")
    for name, content in blocks.items():
        text = replace_block(text, name, content)
    Path(path).write_text(text, encoding="utf-8", newline="\n")
