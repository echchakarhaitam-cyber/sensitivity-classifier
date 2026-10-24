"""Facts about a run that go into model sidecars: package versions and peak memory."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ("sensclf", "privgate", "scikit-learn", "numpy", "scipy", "joblib")


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def peak_memory_mb() -> float | None:
    """Peak resident memory of THIS process so far, in MB (None if it cannot be read).

    Windows: peak working set (psutil). Elsewhere: ``ru_maxrss`` (KB on Linux, bytes on
    macOS). It is a process-lifetime peak, so scripts train each model family in its own
    process to attribute the peak to one family.
    """
    try:
        if sys.platform == "win32":
            import psutil

            return psutil.Process().memory_info().peak_wset / 2**20
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    except (ImportError, AttributeError, OSError):
        return None
