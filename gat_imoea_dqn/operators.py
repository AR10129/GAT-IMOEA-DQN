"""Variation for the two-string chromosome (PDF step 6).

    off, info = make_offspring(pop, F_lo, F_hi, dag, sc.allowed_targets(), rng)

Mating procedure (as in the PDF):
  1. Sort P into interval non-dominated fronts (possibility-degree dominance).
     The top half is the elite pool TopPI and mates only inside TopPI; the
     bottom half BottomPI picks partners from the whole population.
  2. Sort P into uncertainty fronts (Pareto on U1, U2, computed on objectives
     scaled to [0, 1] over P unless normalize=False). The top half gets
     crossover probability p_c = 0.7, the bottom half p_c = 0.9.
  3. Allocation string: DAEC. Priority string: POX.
Every individual of P is the first parent of exactly one child, so N parents
give N offspring (merged pool 2N for environmental selection).

Operator details the PDF leaves open (choices, all in VariationConfig):
  * DAEC gene operator: uniform crossover where each gene comes from the better
    parent (lower interval front, then lower uncertainty front) with
    probability `elite_bias`. The default 0.5 is plain uniform crossover: the
    "elite" and "dual-adaptive" parts are the mating-pool restriction (1) and
    the front-based p_c (2).
  * POX: precedence-preserving order-based crossover (Bierwirth et al. 1996).
    A random binary mask picks, position by position, which parent supplies
    its leftmost not-yet-used task. Any mask yields a valid topological order.
    mode="one_point" uses a mask 1..1 0..0 (prefix of parent 1, rest in
    parent 2's order).
  * Mutation: each allocation gene is resampled with probability p_m_alloc
    (default 1/K) among the reachable nodes; each task is moved with
    probability p_m_order (default 1/K) to a random position between its
    last parent and first child, which keeps the order valid.
  * Repair: never needed by the operators above, but every child is checked
    and repaired (Kahn ordered by current position) as a safety net.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .chromosome import DagIndex, Population, is_allowed, is_topological, repair_allocation, repair_order
from .intervals import (front_ranks, interval_dominance_matrix, nondominated_fronts, normalize_intervals,
                        pareto_dominance_matrix, uncertainty_metrics)

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                     # pragma: no cover
    HAVE_NUMBA = False


def _pox_py(p1, p2, mask):
    K = p1.shape[0]
    used = np.zeros(K, dtype=np.bool_)
    child = np.empty(K, dtype=np.int64)
    i1 = 0
    i2 = 0
    for k in range(K):
        if mask[k]:
            while used[p1[i1]]:
                i1 += 1
            t = p1[i1]
        else:
            while used[p2[i2]]:
                i2 += 1
            t = p2[i2]
        used[t] = True
        child[k] = t
    return child


_pox = njit(cache=True)(_pox_py) if HAVE_NUMBA else _pox_py


@dataclass(frozen=True)
class VariationConfig:
    p_c_low: float = 0.7                # top half of uncertainty fronts
    p_c_high: float = 0.9               # bottom half
    elite_bias: float = 0.5             # DAEC: chance a gene comes from the better parent
    p_m_alloc: Optional[float] = None   # per gene; None -> 1/K
    p_m_order: Optional[float] = None   # per task; None -> 1/K
    pox_mode: str = "mask"              # "mask" or "one_point"
    dominance_threshold: float = 0.5    # possibility-degree threshold for interval fronts
    normalize: bool = True              # scale objectives to [0, 1] before U1/U2 (as in selection)


@dataclass
class MatingInfo:
    interval_rank: np.ndarray           # front index under interval dominance
    uncertainty_rank: np.ndarray        # front index under UDR (U1, U2)
    elite: np.ndarray                   # bool, member of TopPI
    p_c: np.ndarray                     # crossover probability per first parent
    partner: np.ndarray                 # partner index per first parent
    crossed: np.ndarray                 # bool, crossover actually applied


# ------------------------------------------------------------ crossover ---
def pox(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, mode: str = "mask") -> np.ndarray:
    """Precedence-preserving order-based crossover of two topological orders."""
    K = len(p1)
    if mode == "mask":
        mask = rng.random(K) < 0.5
    elif mode == "one_point":
        mask = np.arange(K) < int(rng.integers(1, K)) if K > 1 else np.ones(K, bool)
    else:
        raise ValueError(f"unknown POX mode {mode!r}")
    return _pox(np.asarray(p1, np.int64), np.asarray(p2, np.int64), mask)


def daec(x_better: np.ndarray, x_other: np.ndarray, rng: np.random.Generator,
         elite_bias: float = 0.5) -> np.ndarray:
    """Allocation-string crossover: gene k from the better parent with prob elite_bias."""
    take = rng.random(len(x_better)) < elite_bias
    return np.where(take, x_better, x_other)


# ------------------------------------------------------------- mutation ---
def mutate_allocation(x: np.ndarray, allowed: np.ndarray, rng: np.random.Generator,
                      rate: float) -> np.ndarray:
    """Resample each gene with probability `rate` to a different reachable node.
    `allowed`: 1-D targets shared by all tasks, or a (K, n+2) bool mask per task."""
    x = x.copy()
    hit = np.flatnonzero(rng.random(len(x)) < rate)
    if not hit.size:
        return x
    if allowed.ndim == 1:
        if len(allowed) > 1:
            for k in hit:
                choices = allowed[allowed != x[k]]
                x[k] = choices[rng.integers(len(choices))]
        return x
    for k in hit:
        choices = np.flatnonzero(allowed[k])
        choices = choices[choices != x[k]]
        if len(choices):
            x[k] = choices[rng.integers(len(choices))]
    return x


def shift_task(O: np.ndarray, v: int, dag: DagIndex, rng: np.random.Generator) -> np.ndarray:
    """Move task v to a random position that keeps O topological."""
    K = len(O)
    pos = np.empty(K, dtype=np.int64)
    pos[O] = np.arange(K)
    p, s = pos[dag.preds(v)], pos[dag.succs(v)]
    lo = int(p.max()) + 1 if p.size else 0               # index in O without v
    hi = int(s.min()) - 1 if s.size else K - 1
    if hi <= lo:
        return O
    rest = np.delete(O, pos[v])
    return np.insert(rest, int(rng.integers(lo, hi + 1)), v)


def mutate_order(O: np.ndarray, dag: DagIndex, rng: np.random.Generator, rate: float) -> np.ndarray:
    for v in np.flatnonzero(rng.random(len(O)) < rate):
        O = shift_task(O, int(v), dag, rng)
    return O


# --------------------------------------------------------------- mating ---
def _top_half(rank: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bool mask of the best ceil(N/2) by rank; ties at the cut broken at random."""
    n = len(rank)
    order = np.lexsort((rng.random(n), rank))
    top = np.zeros(n, dtype=bool)
    top[order[:(n + 1) // 2]] = True
    return top


def make_offspring(pop: Population, F_lo: np.ndarray, F_hi: np.ndarray, dag: DagIndex,
                   allowed: np.ndarray, rng: np.random.Generator,
                   cfg: VariationConfig = VariationConfig()):
    N, K = pop.X.shape
    if N < 2:
        raise ValueError("need at least 2 individuals to mate")
    allowed = np.asarray(allowed)
    allowed = allowed.astype(bool) if allowed.ndim == 2 else allowed.astype(np.int64)
    rate_x = cfg.p_m_alloc if cfg.p_m_alloc is not None else 1.0 / K
    rate_o = cfg.p_m_order if cfg.p_m_order is not None else 1.0 / K

    iv_rank = front_ranks(nondominated_fronts(
        interval_dominance_matrix(F_lo, F_hi, cfg.dominance_threshold)), N)
    U1, U2 = uncertainty_metrics(*(normalize_intervals(F_lo, F_hi) if cfg.normalize else (F_lo, F_hi)))
    un_rank = front_ranks(nondominated_fronts(pareto_dominance_matrix(np.stack([U1, U2], 1))), N)

    elite = _top_half(iv_rank, rng)
    p_c = np.where(_top_half(un_rank, rng), cfg.p_c_low, cfg.p_c_high)
    elite_idx = np.flatnonzero(elite)

    X_off = np.empty_like(pop.X)
    O_off = np.empty_like(pop.O)
    partner = np.empty(N, dtype=np.int64)
    crossed = np.zeros(N, dtype=bool)
    for i in range(N):
        pool = elite_idx if (elite[i] and len(elite_idx) > 1) else np.arange(N)
        j = i
        while j == i:
            j = int(pool[rng.integers(len(pool))])
        partner[i] = j
        x, o = pop.X[i], pop.O[i]
        if rng.random() < p_c[i]:
            crossed[i] = True
            i_better = (iv_rank[i], un_rank[i]) <= (iv_rank[j], un_rank[j])
            xb, xo = (pop.X[i], pop.X[j]) if i_better else (pop.X[j], pop.X[i])
            x = daec(xb, xo, rng, cfg.elite_bias)
            o = pox(pop.O[i], pop.O[j], rng, cfg.pox_mode)
        x = mutate_allocation(x, allowed, rng, rate_x)
        o = mutate_order(o.copy(), dag, rng, rate_o)
        if not is_allowed(x, allowed).all():
            x = repair_allocation(x, allowed, rng)
        if not is_topological(o, dag):
            o = repair_order(o, dag)
        X_off[i], O_off[i] = x, o

    info = MatingInfo(interval_rank=iv_rank, uncertainty_rank=un_rank, elite=elite, p_c=p_c,
                      partner=partner, crossed=crossed)
    return Population(X_off, O_off), info
