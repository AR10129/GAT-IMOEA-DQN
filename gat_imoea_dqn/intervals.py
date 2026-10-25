"""Interval helpers (PDF sections 2.2, 2.4 and steps 4/6).

An interval vector is a pair of arrays (lo, hi) with lo <= hi elementwise.
Every operation the timing calculus needs (+, max, division of a positive
quantity by a positive interval) is monotone in each bound, so an interval
result is always [f(lower inputs), f(upper inputs)].
"""
from __future__ import annotations

from typing import List

import numpy as np


def normalize_intervals(F_lo: np.ndarray, F_hi: np.ndarray):
    """Scale each objective to [0, 1] over the population: (F - min lo) / (max hi - min lo).

    Objectives are in s, J and $ with very different magnitudes; widths and
    distances only become comparable across objectives after this scaling.
    """
    F_lo, F_hi = np.asarray(F_lo, float), np.asarray(F_hi, float)
    lo = F_lo.min(axis=0)
    span = F_hi.max(axis=0) - lo
    span = np.where(span > 0, span, 1.0)
    return (F_lo - lo) / span, (F_hi - lo) / span


def uncertainty_metrics(F_lo: np.ndarray, F_hi: np.ndarray):
    """U1 = sum of objective interval lengths, U2 = product of them.

    Works for one solution (shape (M,)) or a population (shape (P, M)).
    """
    width = np.asarray(F_hi, dtype=float) - np.asarray(F_lo, dtype=float)
    return width.sum(axis=-1), width.prod(axis=-1)


def uncertainty_dominates(Ua, Ub) -> bool:
    """X_a <_UN X_b: no worse on both U1 and U2 and strictly better on one."""
    Ua, Ub = np.asarray(Ua), np.asarray(Ub)
    return bool(np.all(Ua <= Ub) and np.any(Ua < Ub))


# ------------------------------------------------------- dominance --------
def possibility_leq(a_lo, a_hi, b_lo, b_hi) -> np.ndarray:
    """Possibility degree P(A <= B) for intervals A = [a_lo, a_hi], B = [b_lo, b_hi].

    P(A <= B) = min(1, max(0, (b_hi - a_lo) / (w_A + w_B))), the standard
    possibility degree for interval numbers. It satisfies P(A<=B) + P(B<=A) = 1.
    For two points (w_A + w_B = 0) it is 1, 0.5 or 0 for a < b, a = b, a > b.
    Inputs broadcast.
    """
    a_lo, a_hi, b_lo, b_hi = (np.asarray(x, dtype=float) for x in (a_lo, a_hi, b_lo, b_hi))
    w = (a_hi - a_lo) + (b_hi - b_lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.clip((b_hi - a_lo) / w, 0.0, 1.0)
    crisp = np.where(a_lo < b_lo, 1.0, np.where(a_lo > b_lo, 0.0, 0.5))
    return np.where(w > 0, p, crisp)


def interval_dominance_matrix(F_lo: np.ndarray, F_hi: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """D[i, j] = True when solution i interval-dominates solution j (minimisation).

    i <_IN j  iff  P(F_m(i) <= F_m(j)) >= threshold for every objective m and
    > threshold for at least one.
    """
    F_lo, F_hi = np.asarray(F_lo, float), np.asarray(F_hi, float)
    P = possibility_leq(F_lo[:, None, :], F_hi[:, None, :], F_lo[None, :, :], F_hi[None, :, :])
    D = np.all(P >= threshold, axis=2) & np.any(P > threshold, axis=2)
    np.fill_diagonal(D, False)
    return D


def pareto_dominance_matrix(F: np.ndarray) -> np.ndarray:
    """D[i, j] = True when crisp vector F[i] Pareto-dominates F[j] (minimisation)."""
    F = np.asarray(F, float)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    return le & lt


def nondominated_fronts(D: np.ndarray) -> List[np.ndarray]:
    """Peel non-dominated fronts from a dominance matrix D[i, j] = i dominates j.

    Possibility-degree dominance need not be transitive, so a cycle can leave
    every remaining solution dominated; those with the fewest dominators then
    form the next front, which guarantees termination.
    """
    n = D.shape[0]
    remaining = np.ones(n, dtype=bool)
    fronts = []
    while remaining.any():
        counts = D[remaining][:, remaining].sum(axis=0)
        idx = np.flatnonzero(remaining)
        front = idx[counts == 0]
        if front.size == 0:
            front = idx[counts == counts.min()]
        fronts.append(front)
        remaining[front] = False
    return fronts


def front_ranks(fronts: List[np.ndarray], n: int) -> np.ndarray:
    rank = np.empty(n, dtype=np.int64)
    for r, f in enumerate(fronts):
        rank[f] = r
    return rank
