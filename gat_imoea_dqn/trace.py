"""E5: optional convergence trace. Off unless `start()` is called.

Every algorithm loop calls `record(gen, F_lo, F_hi)` after each generation (gen = generations
completed, 0 = initial population) and `final(F_lo, F_hi)` at the end. While a trace is active,
the population's first interval front is stored at the checkpoint generations. Nothing here
touches a random stream, so traced runs give the same final front as untraced ones.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_ACTIVE: Optional[Dict] = None


def checkpoints(G: int):
    """0, 10, 25, 50, then every G/20; always G."""
    pts = {0, 10, 25, 50} | set(range(0, G + 1, max(G // 20, 1))) | {G}
    return sorted(p for p in pts if p <= G)


def start(G: int):
    global _ACTIVE
    _ACTIVE = {"G": G, "at": set(checkpoints(G)), "fronts": {}}


def stop() -> Dict:
    global _ACTIVE
    out, _ACTIVE = _ACTIVE, None
    return out["fronts"] if out else {}


def _store(gen, F_lo, F_hi):
    from .intervals import interval_dominance_matrix, nondominated_fronts
    f = nondominated_fronts(interval_dominance_matrix(F_lo, F_hi))[0]
    _ACTIVE["fronts"][int(gen)] = (np.asarray(F_lo)[f].tolist(), np.asarray(F_hi)[f].tolist())


def record(gen, F_lo, F_hi):
    if _ACTIVE is not None and gen in _ACTIVE["at"] and gen < _ACTIVE["G"]:
        _store(gen, F_lo, F_hi)


def final(F_lo, F_hi):
    if _ACTIVE is not None:
        _store(_ACTIVE["G"], F_lo, F_hi)
