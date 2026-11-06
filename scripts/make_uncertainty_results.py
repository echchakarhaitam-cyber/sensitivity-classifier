"""Generate docs/results-uncertainty.md from results/m3, m4, m5 (never edit the output by hand).

Usage:  python scripts/make_uncertainty_results.py     (after m3_calibrate, m4_m5_uncertainty)
"""

from __future__ import annotations

import json
import sys

from sensclf.rawdata import REPO_ROOT
from sensclf.report import wrong_comparisons
from sensclf.uncertainty_report import render_uncertainty

DOC = REPO_ROOT / "docs" / "results-uncertainty.md"


def main() -> int:
    def read(rel):
        return json.loads((REPO_ROOT / "results" / rel).read_text(encoding="utf-8"))

    text = render_uncertainty(read("m3/calibration.json"), read("m4/selective.json"),
                              read("m5/conformal.json"))
    wrong = wrong_comparisons(text)
    if wrong:
        print("contradictory generated sentences:", *wrong, sep="\n  ", file=sys.stderr)
        return 1
    DOC.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
