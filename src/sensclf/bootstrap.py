"""Cluster (group) bootstrap: resample GROUPS of documents, not documents.

The two Nemotron renderings of a persona (``us`` / ``intl``, one ``group_id``) share the
person, the document type and often the very values that are annotated; treating them as
independent draws would make intervals too narrow. Each resample draws G groups with
replacement out of the G groups in the set; a document's weight is the number of times its
group was drawn, so every document of a group always has the same weight. Documents without
a group form a group of their own (Gretel).

Resamples are generated in a fixed order from ``numpy.random.default_rng(seed)``, so the same
seed and the same document order give identical resamples for every model: statistics of
different models are PAIRED, and differences between them get their own interval.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np

N_RESAMPLES = 1000
SEED = 0
LEVEL = 0.95
CHUNK = 100   # resamples held in memory at once: (CHUNK, n) weights


def group_index(groups: Sequence[str]) -> tuple[np.ndarray, int]:
    """Integer group id per document (in order of first appearance) and the group count."""
    ids: dict[str, int] = {}
    index = np.fromiter((ids.setdefault(g, len(ids)) for g in groups), dtype=np.int64,
                        count=len(groups))
    return index, len(ids)


def iter_weights(groups: Sequence[str], n_resamples: int = N_RESAMPLES, seed: int = SEED,
                 chunk: int = CHUNK) -> Iterator[np.ndarray]:
    """Yield (k, n) document-weight matrices, ``n_resamples`` rows in total."""
    index, n_groups = group_index(groups)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_resamples:
        k = min(chunk, n_resamples - done)
        counts = np.stack([np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
                           for _ in range(k)])
        yield counts[:, index].astype(np.float64)
        done += k


def cluster_bootstrap(groups: Sequence[str],
                      statistics: Mapping[str, Callable[[np.ndarray], np.ndarray]],
                      n_resamples: int = N_RESAMPLES, seed: int = SEED,
                      chunk: int = CHUNK) -> dict[str, np.ndarray]:
    """Value of every statistic on every resample (each maps (k, n) weights to (k,) values)."""
    parts: dict[str, list[np.ndarray]] = {name: [] for name in statistics}
    for W in iter_weights(groups, n_resamples, seed, chunk):
        for name, fn in statistics.items():
            parts[name].append(np.asarray(fn(W), dtype=float))
    return {name: np.concatenate(vals) for name, vals in parts.items()}


def interval(values: np.ndarray, level: float = LEVEL) -> tuple[float, float] | None:
    """Percentile interval, ignoring resamples where the statistic is undefined (NaN)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    alpha = (1 - level) / 2
    lo, hi = np.percentile(finite, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)
