"""Quality indicators for comparing final fronts (minimisation, normalized objectives).

Jin et al. (2026), Section 5.2, used for comparability:
  IGD          mean distance from each reference point to the nearest front
               midpoint; the reference = non-dominated midpoints of all runs pooled
  HV (Jin)     (HV of lower bounds + HV of upper bounds) / 2
  Imprecision  sum over the front's solutions of the summed interval lengths
"""
from __future__ import annotations

import numpy as np


def nondominated(F: np.ndarray) -> np.ndarray:
    """Pareto non-dominated rows of F (minimisation), duplicates removed."""
    F = np.unique(np.asarray(F, float), axis=0)
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dominated = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
        keep &= ~dominated
    return F[keep]


def pooled_reference(fronts_mid) -> np.ndarray:
    """Non-dominated union of many fronts, merged one front at a time (memory stays small)."""
    ref = np.empty((0, 3))
    for F in fronts_mid:
        ref = nondominated(np.vstack([ref, np.asarray(F, float)]))
    return ref


def igd(front_mid: np.ndarray, reference: np.ndarray) -> float:
    d = np.linalg.norm(reference[:, None, :] - np.asarray(front_mid)[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def hv_jin(front_lo: np.ndarray, front_hi: np.ndarray, ref) -> float:
    return 0.5 * (hypervolume_3d(front_lo, ref) + hypervolume_3d(front_hi, ref))


def imprecision(front_lo: np.ndarray, front_hi: np.ndarray) -> float:
    return float((np.asarray(front_hi) - np.asarray(front_lo)).sum())


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
