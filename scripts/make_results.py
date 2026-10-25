"""Generate docs/results.md from results/m2/*.json (never edit the output by hand).

Usage:  python scripts/make_results.py     (after m2_train, m2_evaluate, m2_shortcuts)

Refuses to write a document in which a generated comparison contradicts its own numbers.
"""

from __future__ import annotations

import json
import sys

from sensclf.rawdata import REPO_ROOT
from sensclf.report import TRAINED, render_results, wrong_comparisons

RESULTS = REPO_ROOT / "results" / "m2"
DOC = REPO_ROOT / "docs" / "results.md"


def load_inputs() -> tuple[dict, dict, dict]:
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    selection = {name: read(RESULTS / "selection" / f"{name}.json") for name in TRAINED}
    return read(RESULTS / "metrics.json"), selection, read(RESULTS / "shortcuts.json")


def main() -> int:
    text = render_results(*load_inputs())
    wrong = wrong_comparisons(text)
    if wrong:
        print("contradictory generated sentences:", *wrong, sep="\n  ", file=sys.stderr)
        return 1
    DOC.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
