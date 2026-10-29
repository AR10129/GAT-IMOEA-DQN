"""Three-tier environmental selection (PDF step 4) and the DQN action set (step 3).

    actions = selection_actions()                     # (28, 3) rows (alpha, beta, gamma)
    keep, info = environmental_selection(F_lo, F_hi, N, actions[a], rng, genotypes=(X, O))

The merged pool P_M = P_t U O_t (size 2N) is ranked three ways:
  Rank C  ascending interval dominance degree  R_IN(x) = sum_{y <_IN x} S_IN(y),
          S_IN(y) = number of pool members y interval-dominates.
  Rank D  descending interval crowding distance, summed over the k = sqrt(|P_M|)
          nearest neighbours, with
          ICD(a, b) = sum_j |m(F_j(a)) - m(F_j(b))| / (mu(a, b) + V(a) + V(b) + 1)
          mu = overlap volume of the two objective hyper-boxes, V = box volume.
  Rank U  ascending uncertainty dominance degree R_UN, built like R_IN from the
          UDR relation on (U1, U2).
P_{t+1} = top alpha*N of Rank C, then top beta*N of Rank D not yet chosen,
then top gamma*N of Rank U not yet chosen.

Choices beyond the PDF (SelectionConfig):
  * normalize=True: objectives are scaled to [0, 1] over P_M before ICD and
    U1/U2 are computed, so s, J and $ contribute comparably. Interval dominance
    (Rank C) is scale-free and uses raw values.
  * "Nearest neighbours" of a are the k members with the smallest ICD(a, .),
    i.e. the most similar ones, and ICD(a) is the sum over them.
  * Counts: n_C = round(alpha N), n_D = round(beta N), n_U = N - n_C - n_D.
  * dedupe_genotypes=True: an exact copy (same X and O) of an individual already
    chosen is skipped while other candidates remain, so clones do not crowd the
    population. Set False for plain index-based deduplication.
  * convergence="tchebycheff" replaces R_IN in Rank C with a decomposition rank:
    n_weights Das-Dennis vectors w (default |P'| = N), objectives normalised over P_M
    (ideal -> 0), interval Tchebycheff g~(x | w) = [max_j w_j lo_j, max_j w_j hi_j], and
    R_TCH(x) = min over w of x's position when P_M is sorted by g~(. | w). Sorting
    intervals by possibility degree at 0.5 equals sorting by midpoint, so each list is
    the order IMOEA/D's replacement rule uses; every vector's best solution has R = 0.
  * Ties inside a ranking are broken at random (tie_break="random", the default).
    tie_break="icd" breaks ties in Rank C by descending ICD instead (as NSGA-II
    truncates its last front by crowding). Late in a run more than alpha*N pool
    members share R_IN = 0, so a random cut lets the population drift along the
    front and lose its extremes.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .intervals import interval_dominance_matrix, normalize_intervals, pareto_dominance_matrix, uncertainty_metrics


# -------------------------------------------------------------- actions ---
def das_dennis(divisions: int = 6, n_obj: int = 3) -> np.ndarray:
    """Das-Dennis simplex lattice: all weight vectors with entries in {0, 1/q, ..., 1}
    summing to 1. q = 6, m = 3 gives C(8, 2) = 28 vectors."""
    rows = [c for c in itertools.product(range(divisions + 1), repeat=n_obj) if sum(c) == divisions]
    return np.array(sorted(rows, reverse=True), dtype=float) / divisions


def selection_actions(divisions: int = 6) -> np.ndarray:
    """The H = 28 discrete DQN actions a = (alpha, beta, gamma)."""
    return das_dennis(divisions, 3)


ACTION_SETS = ("all", "convergent")


def action_subset(name: str = "all") -> np.ndarray:
    """Indices into selection_actions() an agent may choose from: "all" 28, or
    "convergent" the 10 actions with alpha >= 1/2 (convergence tier at least half of N)."""
    A = selection_actions()
    if name == "all":
        return np.arange(len(A))
    if name == "convergent":
        return np.flatnonzero(A[:, 0] >= 0.5 - 1e-9)
    raise ValueError(f"action set must be one of {ACTION_SETS}, got {name!r}")


def simplex_weights(n: int, n_obj: int = 3) -> np.ndarray:
    """n weight vectors: the largest Das-Dennis lattice with at most n points, topped up
    with fixed-seed random simplex points (q = 13 gives exactly 105 for 3 objectives)."""
    q = 1
    while len(das_dennis(q + 1, n_obj)) <= n:
        q += 1
    W = das_dennis(q, n_obj)
    if len(W) < n:
        W = np.vstack([W, np.random.default_rng(0).dirichlet(np.ones(n_obj), size=n - len(W))])
    return W[:n]


def tchebycheff_rank(G_lo: np.ndarray, G_hi: np.ndarray, W: np.ndarray) -> np.ndarray:
    """R_TCH: best position of each solution over the weight vectors' midpoint-Tchebycheff orders.
    G_lo, G_hi: objectives already normalised so the ideal point is 0."""
    W = np.maximum(W, 1e-6)
    g = 0.5 * (np.max(W[:, None, :] * G_lo[None], axis=2) + np.max(W[:, None, :] * G_hi[None], axis=2))
    pos = np.empty_like(g, dtype=np.int64)
    order = np.argsort(g, axis=1, kind="stable")
    np.put_along_axis(pos, order, np.arange(g.shape[1])[None, :].repeat(len(W), 0), axis=1)
    return pos.min(axis=0).astype(float)


# ------------------------------------------------------------- rankings ---
def dominance_degree(D: np.ndarray) -> np.ndarray:
    """R(x) = sum of strengths S(y) of every y that dominates x; S(y) = #dominated by y."""
    D = np.asarray(D, dtype=bool)
    S = D.sum(axis=1).astype(float)
    return D.T.astype(float) @ S


def interval_crowding(F_lo: np.ndarray, F_hi: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """ICD(a) = sum of ICD(a, b) over the k most similar other solutions b."""
    F_lo, F_hi = np.asarray(F_lo, float), np.asarray(F_hi, float)
    n = len(F_lo)
    if n < 2:
        return np.full(n, np.inf)
    k = int(round(np.sqrt(n))) if k is None else int(k)
    k = max(1, min(k, n - 1))
    mid = (F_lo + F_hi) / 2.0
    V = np.prod(F_hi - F_lo, axis=1)
    num = np.abs(mid[:, None, :] - mid[None, :, :]).sum(axis=2)
    overlap = np.clip(np.minimum(F_hi[:, None, :], F_hi[None, :, :])
                      - np.maximum(F_lo[:, None, :], F_lo[None, :, :]), 0.0, None)
    mu = np.prod(overlap, axis=2)
    pair = num / (mu + V[:, None] + V[None, :] + 1.0)
    np.fill_diagonal(pair, np.inf)
    nearest = np.partition(pair, k - 1, axis=1)[:, :k]
    return nearest.sum(axis=1)


def _order(key: np.ndarray, rng: np.random.Generator, descending: bool = False) -> np.ndarray:
    key = -key if descending else key
    return np.lexsort((rng.random(len(key)), key))


# ------------------------------------------------------------- assembly ---
@dataclass(frozen=True)
class SelectionConfig:
    normalize: bool = True
    k_neighbors: Optional[int] = None      # None -> round(sqrt(|P_M|))
    dedupe_genotypes: bool = True
    dominance_threshold: float = 0.5
    tie_break: str = "random"              # Rank C ties: "random" or "icd" (descending crowding)
    convergence: str = "dominance"         # Rank C: "dominance" (R_IN, Jin et al.) or "tchebycheff"
    n_weights: Optional[int] = None        # tchebycheff: weight vectors; None -> the selection size N


@dataclass
class SelectionInfo:
    R_C: np.ndarray                        # interval dominance degree (lower = better)
    ICD: np.ndarray                        # interval crowding distance (higher = better)
    R_U: np.ndarray                        # uncertainty dominance degree (lower = better)
    counts: Tuple[int, int, int]           # n_C, n_D, n_U requested
    tiers: dict = field(default_factory=dict)   # "C" / "D" / "U" -> chosen pool indices


def rankings(F_lo, F_hi, cfg: SelectionConfig = SelectionConfig(), n_select: Optional[int] = None):
    G_lo, G_hi = normalize_intervals(F_lo, F_hi) if cfg.normalize else (np.asarray(F_lo), np.asarray(F_hi))
    if cfg.convergence == "dominance":
        R_C = dominance_degree(interval_dominance_matrix(F_lo, F_hi, cfg.dominance_threshold))
    elif cfg.convergence == "tchebycheff":
        n_w = cfg.n_weights or n_select or len(F_lo) // 2
        R_C = tchebycheff_rank(*normalize_intervals(F_lo, F_hi), simplex_weights(n_w, np.shape(F_lo)[1]))
    else:
        raise ValueError(f"unknown convergence ranking {cfg.convergence!r}")
    ICD = interval_crowding(G_lo, G_hi, cfg.k_neighbors)
    U1, U2 = uncertainty_metrics(G_lo, G_hi)
    R_U = dominance_degree(pareto_dominance_matrix(np.stack([U1, U2], axis=1)))
    return R_C, ICD, R_U


def environmental_selection(F_lo: np.ndarray, F_hi: np.ndarray, n_select: int, action,
                            rng: np.random.Generator, genotypes=None,
                            cfg: SelectionConfig = SelectionConfig()):
    """Pick n_select pool indices. `action` = (alpha, beta, gamma), summing to 1.
    `genotypes` = (X, O) of the pool, used only for clone removal."""
    F_lo, F_hi = np.asarray(F_lo, float), np.asarray(F_hi, float)
    n_pool = len(F_lo)
    if not 0 < n_select <= n_pool:
        raise ValueError(f"cannot select {n_select} from a pool of {n_pool}")
    a = np.asarray(action, float)
    if a.shape != (3,) or np.any(a < 0) or not np.isclose(a.sum(), 1.0):
        raise ValueError(f"action must be (alpha, beta, gamma) >= 0 summing to 1, got {action}")
    n_C = int(round(a[0] * n_select))
    n_D = min(int(round(a[1] * n_select)), n_select - n_C)
    n_U = n_select - n_C - n_D

    R_C, ICD, R_U = rankings(F_lo, F_hi, cfg, n_select)
    if cfg.tie_break == "icd":
        order_C = np.lexsort((rng.random(n_pool), -ICD, R_C))
    elif cfg.tie_break == "random":
        order_C = _order(R_C, rng)
    else:
        raise ValueError(f"unknown tie_break {cfg.tie_break!r}")
    orders = {"C": order_C, "D": _order(ICD, rng, descending=True), "U": _order(R_U, rng)}

    keys = None
    if cfg.dedupe_genotypes and genotypes is not None:
        X, O = genotypes
        keys = [X[i].tobytes() + O[i].tobytes() for i in range(n_pool)]

    chosen = np.zeros(n_pool, dtype=bool)
    seen = set()
    tiers = {}
    for tier, want in (("C", n_C), ("D", n_D), ("U", n_U)):
        picked = []
        for i in orders[tier]:
            if len(picked) == want:
                break
            if chosen[i] or (keys is not None and keys[i] in seen):
                continue
            picked.append(i)
            chosen[i] = True
            if keys is not None:
                seen.add(keys[i])
        tiers[tier] = picked
    # only clones were left: fill the remaining slots from Rank C regardless
    short = n_select - int(chosen.sum())
    if short:
        extra = [i for i in orders["C"] if not chosen[i]][:short]
        chosen[extra] = True
        tiers["fill"] = extra
    keep = np.concatenate([np.asarray(tiers[t], dtype=np.int64) for t in tiers])
    return keep, SelectionInfo(R_C=R_C, ICD=ICD, R_U=R_U, counts=(n_C, n_D, n_U),
                               tiers={t: np.asarray(v, dtype=np.int64) for t, v in tiers.items()})
