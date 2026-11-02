"""Reference heuristics for DAG offloading (task B0): no search, a handful of plans.

    res = run_heuristics(wf, sc, EvaluatorConfig(...))    # RunResult; res.history["names"]

Plans, all scored by the same IntervalEvaluator as the evolutionary methods:
  all_local     every task on its own device
  all_edge      every task on its device's nearest in-range edge server (local if none)
  all_cloud     every task on the cloud (local if the device cannot reach it)
  heft_<obj>    HEFT-style list scheduling, one plan per objective (makespan, energy, cost):
                tasks in decreasing upward rank (HEFT's priority: mean compute time plus the
                longest mean compute + transfer path to an exit), each placed in that order.
                heft_makespan is classic HEFT: the allowed node with the earliest finish time of
                the task. heft_energy / heft_cost: the node that minimises the plan's midpoint
                energy / cost, with the tasks not yet placed kept on their device.
The priority string of every plan is the upward-rank order (topological by construction).
The result's front is the first interval front of the distinct plans, as for the EAs.
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from .algorithm import RunResult
from .chromosome import DagIndex, Population, allowed_for
from .evaluator import EvaluatorConfig, IntervalEvaluator
from .intervals import interval_dominance_matrix, nondominated_fronts

OBJECTIVES = ("makespan", "energy", "cost")


def _allowed_mask(wf, sc, K: int, N: int) -> np.ndarray:
    allowed = allowed_for(wf, sc)
    if allowed.ndim == 2:
        return allowed.astype(bool)
    mask = np.zeros((K, N), dtype=bool)
    mask[:, allowed] = True
    return mask


def upward_rank(ev: IntervalEvaluator, mask: np.ndarray) -> np.ndarray:
    """HEFT upward rank on interval midpoints: mean compute time over each task's allowed
    nodes (full capacity) plus the max over children of mean transfer + the child's rank."""
    K = ev.K
    C = (ev.C_lo + ev.C_hi) / 2
    f = (ev.f_lo + ev.f_hi) / 2
    spk = (ev.spk_lo + ev.spk_hi) / 2
    nodes = [np.where(np.flatnonzero(mask[v]) == 0, ev.dev_node[v], np.flatnonzero(mask[v])) for v in range(K)]
    w = np.array([np.mean(C[v] / f[nd]) for v, nd in enumerate(nodes)])
    D = (ev.D_lo + ev.D_hi) / 2
    cbar = np.empty(len(D))
    for e in range(len(D)):
        a, b = nodes[ev.src[e]], nodes[ev.dst[e]]
        S = spk[np.ix_(a, b)]
        S = np.where(np.isfinite(S), S, np.nan)
        cbar[e] = D[e] * np.nanmean(S) if D[e] > 0 and np.isfinite(np.nanmean(S)) else 0.0
    rank = np.zeros(K)
    by_src = np.argsort(ev.src, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(np.bincount(ev.src, minlength=K))])
    for v in np.argsort(-_depth_order(ev), kind="stable"):    # children before parents
        es = by_src[ptr[v]:ptr[v + 1]]
        rank[v] = w[v] + (max(cbar[e] + rank[ev.dst[e]] for e in es) if len(es) else 0.0)
    return rank


def _depth_order(ev: IntervalEvaluator) -> np.ndarray:
    """Longest-path depth from the entry tasks (parents have smaller depth than children)."""
    depth = np.zeros(ev.K, dtype=np.int64)
    indeg = np.bincount(ev.dst, minlength=ev.K)
    succ_by_src = np.argsort(ev.src, kind="stable")
    sptr = np.concatenate([[0], np.cumsum(np.bincount(ev.src, minlength=ev.K))])
    ready = [v for v in range(ev.K) if indeg[v] == 0]
    while ready:
        v = ready.pop()
        for e in succ_by_src[sptr[v]:sptr[v + 1]]:
            c = ev.dst[e]
            depth[c] = max(depth[c], depth[v] + 1)
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    return depth


def rank_order(dag: DagIndex, rank: np.ndarray) -> np.ndarray:
    """Topological order taking, among ready tasks, the highest upward rank (ties: lower id)."""
    indeg = dag.indeg.copy()
    heap = [(-rank[v], v) for v in range(dag.K) if indeg[v] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, v = heapq.heappop(heap)
        order.append(v)
        for c in dag.succs(v):
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(heap, (-rank[c], int(c)))
    return np.array(order, dtype=np.int64)


def run_heuristics(wf, sc, evaluator: EvaluatorConfig = EvaluatorConfig()) -> RunResult:
    t0 = time.perf_counter()
    ev = IntervalEvaluator(wf, sc, evaluator)
    K, N = ev.K, ev.N
    mask = _allowed_mask(wf, sc, K, N)
    dag = DagIndex(wf)
    O = rank_order(dag, upward_rank(ev, mask))

    def pick(node_of_task):
        return np.where(mask[np.arange(K), node_of_task], node_of_task, 0).astype(np.int64)

    near = ev.near_node                       # internal node index = allocation gene for edges
    plans = {"all_local": np.zeros(K, dtype=np.int64),
             "all_edge": pick(np.where(near > 0, near, 0)),
             "all_cloud": pick(np.full(K, ev.cloud))}
    for j, obj in enumerate(OBJECTIVES):
        X = np.zeros(K, dtype=np.int64)
        for v in O:
            cand = np.flatnonzero(mask[v])
            if obj == "makespan":
                # classic HEFT: the node with the earliest finish time of v (midpoint), given the
                # tasks placed so far; tasks later in O cannot delay v in list scheduling
                score = []
                for x in cand:
                    X[v] = x
                    FT_lo, FT_hi = ev.evaluate(X, O, detail=True).FT
                    score.append((FT_lo[v] + FT_hi[v]) / 2)
            else:
                Xs = np.repeat(X[None], len(cand), axis=0)
                Xs[:, v] = cand
                lo, hi = ev.evaluate_population(Xs, np.repeat(O[None], len(cand), axis=0))
                score = (lo[:, j] + hi[:, j]) / 2
            X[v] = cand[int(np.argmin(score))]                # ties: the first (local) node
        plans[f"heft_{obj}"] = X
    names = list(plans)
    Xs = np.stack([plans[n] for n in names])
    Os = np.repeat(O[None], len(names), axis=0)
    F_lo, F_hi = ev.evaluate_population(Xs, Os)
    # distinct plans only, so duplicates (e.g. heft_cost = all_local) do not count twice
    _, first = np.unique(np.round(np.hstack([F_lo, F_hi]), 12), axis=0, return_index=True)
    keep = np.sort(first)
    front = keep[nondominated_fronts(interval_dominance_matrix(F_lo[keep], F_hi[keep]))[0]]
    hist = {"names": names, "action": [], "reward": [],
            "objectives": {n: (F_lo[i].tolist(), F_hi[i].tolist()) for i, n in enumerate(names)}}
    return RunResult(population=Population(Xs, Os), F_lo=F_lo, F_hi=F_hi, front=np.sort(front),
                     history=hist, seconds=time.perf_counter() - t0)
