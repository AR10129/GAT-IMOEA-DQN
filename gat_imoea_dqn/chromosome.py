"""Two-string chromosome (PDF step 1).

An individual is a pair:
  X  allocation string, X[k] in {0..n+1}: 0 device, 1..n edge, n+1 cloud
  O  priority string, a permutation of the K tasks in topological order

A population stores them as two int64 arrays, X (N, K) and O (N, K).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dataset_tools.loader import Scenario, WorkflowDAG

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                     # pragma: no cover
    HAVE_NUMBA = False


@dataclass
class Population:
    X: np.ndarray                       # (N, K) allocation strings
    O: np.ndarray                       # (N, K) priority strings

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx) -> "Population":
        """Always returns a Population (an int index gives a population of one)."""
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        return Population(self.X[idx].copy(), self.O[idx].copy())

    def copy(self) -> "Population":
        return Population(self.X.copy(), self.O.copy())

    @staticmethod
    def concat(a: "Population", b: "Population") -> "Population":
        return Population(np.concatenate([a.X, b.X]), np.concatenate([a.O, b.O]))


class DagIndex:
    """Adjacency lists in CSR form, built once per workflow."""

    def __init__(self, wf: WorkflowDAG):
        K = wf.num_tasks
        src, dst = wf.edge_index
        self.K = K
        self.src, self.dst = src.astype(np.int64), dst.astype(np.int64)
        by_dst = np.argsort(dst, kind="stable")
        by_src = np.argsort(src, kind="stable")
        self.pred = src[by_dst].astype(np.int64)
        self.pred_ptr = np.concatenate([[0], np.cumsum(np.bincount(dst, minlength=K))]).astype(np.int64)
        self.succ = dst[by_src].astype(np.int64)
        self.succ_ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=K))]).astype(np.int64)
        self.indeg = np.diff(self.pred_ptr)

    def preds(self, v: int) -> np.ndarray:
        return self.pred[self.pred_ptr[v]:self.pred_ptr[v + 1]]

    def succs(self, v: int) -> np.ndarray:
        return self.succ[self.succ_ptr[v]:self.succ_ptr[v + 1]]


# --------------------------------------------------------------- checks ---
def is_topological(O: np.ndarray, dag: DagIndex) -> bool:
    O = np.asarray(O)
    if O.shape != (dag.K,):
        return False
    pos = np.full(dag.K, -1, dtype=np.int64)
    pos[O] = np.arange(dag.K)
    return bool((pos >= 0).all() and np.all(pos[dag.src] < pos[dag.dst]))


# ----------------------------------------------------------- generation ---
def _random_kahn_py(indeg, succ_ptr, succ, u):
    K = indeg.shape[0]
    indeg = indeg.copy()
    ready = np.empty(K, dtype=np.int64)
    n_ready = 0
    for v in range(K):
        if indeg[v] == 0:
            ready[n_ready] = v
            n_ready += 1
    order = np.empty(K, dtype=np.int64)
    for i in range(K):
        j = int(u[i] * n_ready)
        t = ready[j]
        ready[j] = ready[n_ready - 1]          # swap-remove
        n_ready -= 1
        order[i] = t
        for k in range(succ_ptr[t], succ_ptr[t + 1]):
            v = succ[k]
            indeg[v] -= 1
            if indeg[v] == 0:
                ready[n_ready] = v
                n_ready += 1
    return order


_random_kahn = njit(cache=True)(_random_kahn_py) if HAVE_NUMBA else _random_kahn_py


def random_topological_order(dag: DagIndex, rng: np.random.Generator) -> np.ndarray:
    """Kahn's algorithm, picking uniformly among ready tasks (randomized tie-breaking)."""
    return _random_kahn(dag.indeg, dag.succ_ptr, dag.succ, rng.random(dag.K))


def init_population(wf: WorkflowDAG, sc: Scenario, size: int, rng: np.random.Generator,
                    dag: Optional[DagIndex] = None) -> Population:
    """P0: allocation genes drawn uniformly from the nodes the origin device can
    reach (communication range), priority strings from randomized Kahn."""
    dag = dag or DagIndex(wf)
    targets = sc.allowed_targets()
    X = rng.choice(targets, size=(size, wf.num_tasks)).astype(np.int64)
    O = np.stack([random_topological_order(dag, rng) for _ in range(size)])
    return Population(X, O)


# --------------------------------------------------------------- repair ---
def repair_order(O: np.ndarray, dag: DagIndex) -> np.ndarray:
    """Closest topological order to O: Kahn's algorithm that always takes the
    ready task appearing earliest in O. Valid orders come back unchanged.
    Missing or duplicated tasks are fixed too (missing ones are ranked last)."""
    K = dag.K
    key = np.full(K, K, dtype=np.int64)
    O = np.asarray(O, dtype=np.int64)
    for p in range(len(O) - 1, -1, -1):           # first occurrence wins
        if 0 <= O[p] < K:
            key[O[p]] = p
    indeg = dag.indeg.copy()
    heap = [(int(key[v]), int(v)) for v in np.flatnonzero(indeg == 0)]
    heapq.heapify(heap)
    out = np.empty(K, dtype=np.int64)
    for i in range(K):
        _, u = heapq.heappop(heap)
        out[i] = u
        for v in dag.succs(u):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (int(key[v]), int(v)))
    return out


def repair_allocation(X: np.ndarray, allowed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace genes pointing at unreachable nodes with random reachable ones."""
    X = np.asarray(X, dtype=np.int64).copy()
    ok = np.isin(X, allowed)
    if not ok.all():
        X[~ok] = rng.choice(allowed, size=int((~ok).sum()))
    return X
