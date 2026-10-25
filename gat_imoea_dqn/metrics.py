"""Quality indicators for comparing final fronts (minimisation, normalized objectives)."""
from __future__ import annotations

import numpy as np


def hypervolume_3d(F: np.ndarray, ref) -> float:
    """Exact 3-D hypervolume by sweeping the third objective.

    Points that do not strictly dominate `ref` contribute nothing.
    """
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 2])]
    hv = 0.0
    for i in range(len(F)):
        z_next = F[i + 1, 2] if i + 1 < len(F) else ref[2]
        if z_next > F[i, 2]:
            hv += _hv2d_union(F[:i + 1, :2], ref[:2]) * (z_next - F[i, 2])
    return float(hv)


def _hv2d_union(P: np.ndarray, ref: np.ndarray) -> float:
    """Area dominated by the union of boxes [p, ref] (minimisation)."""
    P = P[np.argsort(P[:, 0], kind="stable")]
    area, y_cap = 0.0, ref[1]
    for i in range(len(P)):
        if P[i, 1] < y_cap:
            area += (ref[0] - P[i, 0]) * (y_cap - P[i, 1])
            y_cap = P[i, 1]
    return area
