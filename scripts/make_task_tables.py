"""Regenerate the raw-label -> category table in docs/TASK.md from sensclf.labels.

Usage:  python scripts/make_task_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sensclf.docblocks import update_file
from sensclf.labels import table_markdown

TASK_DOC = Path(__file__).resolve().parents[1] / "docs" / "TASK.md"


def main() -> int:
    update_file(TASK_DOC, {"labels": table_markdown()})
    print(f"updated {TASK_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
