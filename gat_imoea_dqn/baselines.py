"""Re-implemented interval MOEA baselines (Jin et al. 2026 compare against these).

    res = run_baseline("ip_moea", wf, sc, BaselineConfig(generations=1000, seed=0))
    res = run_baseline("imoead",  wf, sc, BaselineConfig(generations=1000, seed=0))

Fairness: every baseline uses OUR two-string chromosome, the same initial
population for a given seed (`init_population`), the same variation operators
(uniform allocation crossover + POX + mutation at rate 1/K, with repair), the
same evaluator, N = 105 and N new evaluations per generation. Only the
selection mechanisms differ. (Jin et al.'s baselines used SBX on an integer
string, which cannot keep a DAG order valid.)

IP-MOEA (Limbourg & Aponte, 2005) - NSGA-II with the imprecise Pareto relation:
  a dominates b iff the upper bound of a is <= the lower bound of b in every
  objective and < in one (every point of box a dominates every point of box b).
  Fronts by non-dominated sorting under that relation; crowding distance on
  interval midpoints (normalised); binary tournament on (front, crowding);
  crossover probability 0.9; (mu + lambda) survival with the last front cut by crowding.

IMOEA/D (Gan et al., 2023) - decomposition with interval scalar functions:
  N Das-Dennis weight vectors (q = 13 gives 105 for 3 objectives), neighbourhood T = 20;
  objectives normalised by the running ideal / nadir; interval reference point
  z~ = [min lower bound, min upper bound]; interval Tchebycheff
  g~(x | w) = [max_j w_j (lo_j - z_lo_j), max_j w_j (hi_j - z_lo_j)];
  a child replaces up to n_r = 2 neighbours whose g~ it beats with possibility
  degree P(g~_child <= g~_old) > 0.5; mating within the neighbourhood with
  probability delta = 0.9. Reference-vector adaptation (simplified from the paper,
  which adjusts vectors by interval crowding distance): every 10% of the run, the
  vector of the most crowded solution (smallest ICD) is moved to the normalised
  direction of the most isolated solution (largest ICD), and neighbourhoods are rebuilt.

NSGA-II (Deb et al., 2002), two interval variants (task B3); the loop is IP-MOEA's, only the
dominance relation used for non-dominated sorting changes:
  nsga2_mid  crisp Pareto dominance on interval midpoints (the "ignore the uncertainty" baseline);
  nsga2_int  possibility-degree interval dominance (P >= 0.5 in every objective, > 0.5 in one),
             the relation of Jin et al.'s beta tier.
  Crowding distance on midpoints, binary tournament on (front, crowding), p_c = 0.9.

NSGA-III (Deb & Jain, 2014) on interval midpoints (nsga3_mid): random mating, crossover
  always applied; (mu + lambda) survival by Pareto fronts on midpoints, the last front
  filled by reference-point niching: N Das-Dennis reference points (105 for N = 105),
  normalisation by the ideal point and the hyperplane through the extreme points (ASF with
  weights e_j, 1e-6 elsewhere; falls back to the worst value per objective when the plane is
  degenerate), association by perpendicular distance, and niche-count selection (an empty
  niche takes its closest member, otherwise a random one).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .algorithm import RunResult
from .chromosome import (DagIndex, Population, allowed_for, init_population, is_allowed, is_topological,
                         repair_allocation, repair_order)
from .evaluator import EvaluatorConfig, IntervalEvaluator
from .intervals import (interval_dominance_matrix, nondominated_fronts, normalize_intervals, pareto_dominance_matrix,
                        possibility_leq)
from .operators import daec, mutate_allocation, mutate_order, pox
from .selection import interval_crowding, simplex_weights

BASELINES = ("ip_moea", "imoead", "nsga2_mid", "nsga2_int", "nsga3_mid")


@dataclass(frozen=True)
class BaselineConfig:
    pop_size: int = 105
    generations: int = 1000
    seed: int = 0
    evaluator: EvaluatorConfig = EvaluatorConfig()
    p_crossover: float = 0.9          # IP-MOEA
    neighbours: int = 20              # IMOEA/D T
    delta: float = 0.9                # IMOEA/D mating-restriction probability
    max_replace: int = 2              # IMOEA/D n_r
    adapt_every: float = 0.1          # IMOEA/D: fraction of the run between vector adaptations


# --------------------------------------------------------------- shared ---
def _child(x1, o1, x2, o2, allowed, dag, rng, crossover: bool):
    K = len(x1)
    if crossover:
        x = daec(x1, x2, rng, 0.5)                   # uniform crossover (no elite bias)
        o = pox(o1, o2, rng)
    else:
        x, o = x1.copy(), o1.copy()
    x = mutate_allocation(x, allowed, rng, 1.0 / K)
    o = mutate_order(o.copy(), dag, rng, 1.0 / K)
    if not is_allowed(x, allowed).all():
        x = repair_allocation(x, allowed, rng)
    if not is_topological(o, dag):
        o = repair_order(o, dag)
    return x, o


def _final(pop: Population, F_lo, F_hi, t0: float) -> RunResult:
    front = nondominated_fronts(interval_dominance_matrix(F_lo, F_hi))[0]
    return RunResult(population=pop, F_lo=F_lo, F_hi=F_hi, front=np.sort(front),
                     history={"action": [], "reward": []}, seconds=time.perf_counter() - t0)


# -------------------------------------------------------------- IP-MOEA ---
def imprecise_dominance_matrix(F_lo: np.ndarray, F_hi: np.ndarray) -> np.ndarray:
    """D[i, j]: every point of box i dominates every point of box j (Limbourg & Aponte)."""
    le = np.all(F_hi[:, None, :] <= F_lo[None, :, :], axis=2)
    lt = np.any(F_hi[:, None, :] < F_lo[None, :, :], axis=2)
    D = le & lt
    np.fill_diagonal(D, False)
    return D


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance (objectives normalised by the set's range)."""
    n, M = F.shape
    cd = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(M):
        order = np.argsort(F[:, j], kind="stable")
        span = F[order[-1], j] - F[order[0], j]
        cd[order[0]] = cd[order[-1]] = np.inf
        if span > 0:
            cd[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return cd


def midpoint_dominance_matrix(F_lo: np.ndarray, F_hi: np.ndarray) -> np.ndarray:
    """D[i, j]: the midpoint of i Pareto-dominates the midpoint of j."""
    D = pareto_dominance_matrix((F_lo + F_hi) / 2.0)
    np.fill_diagonal(D, False)
    return D


DOMINANCE = {"ip_moea": imprecise_dominance_matrix, "nsga2_mid": midpoint_dominance_matrix,
             "nsga2_int": interval_dominance_matrix}


def _rank_and_crowd(F_lo, F_hi, dominance=imprecise_dominance_matrix):
    fronts = nondominated_fronts(dominance(F_lo, F_hi))
    rank = np.empty(len(F_lo), dtype=np.int64)
    crowd = np.empty(len(F_lo))
    mid = (F_lo + F_hi) / 2.0
    for r, f in enumerate(fronts):
        rank[f] = r
        crowd[f] = crowding_distance(mid[f])
    return fronts, rank, crowd


def run_ip_moea(wf, sc, cfg: BaselineConfig, dominance=imprecise_dominance_matrix) -> RunResult:
    """IP-MOEA; with another `dominance` relation, the matching NSGA-II variant."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    dag = DagIndex(wf)
    ev = IntervalEvaluator(wf, sc, cfg.evaluator)
    allowed = allowed_for(wf, sc)
    N = cfg.pop_size
    pop = init_population(wf, sc, N, rng, dag)
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    for _ in range(cfg.generations):
        _, rank, crowd = _rank_and_crowd(F_lo, F_hi, dominance)

        def tournament() -> int:
            a, b = rng.integers(N, size=2)
            if rank[a] != rank[b]:
                return int(a if rank[a] < rank[b] else b)
            return int(a if crowd[a] >= crowd[b] else b)

        Xc = np.empty_like(pop.X)
        Oc = np.empty_like(pop.O)
        for i in range(N):
            p, q = tournament(), tournament()
            Xc[i], Oc[i] = _child(pop.X[p], pop.O[p], pop.X[q], pop.O[q], allowed, dag, rng,
                                  rng.random() < cfg.p_crossover)
        G_lo, G_hi = ev.evaluate_population(Xc, Oc)
        merged = Population.concat(pop, Population(Xc, Oc))
        M_lo, M_hi = np.concatenate([F_lo, G_lo]), np.concatenate([F_hi, G_hi])
        fronts, _, crowd_m = _rank_and_crowd(M_lo, M_hi, dominance)
        keep = []
        for f in fronts:
            if len(keep) + len(f) <= N:
                keep.extend(f.tolist())
            else:
                rest = f[np.argsort(-crowd_m[f], kind="stable")][:N - len(keep)]
                keep.extend(rest.tolist())
                break
        keep = np.asarray(keep, dtype=np.int64)
        pop, F_lo, F_hi = merged[keep], M_lo[keep], M_hi[keep]
    return _final(pop, F_lo, F_hi, t0)


# -------------------------------------------------------------- IMOEA/D ---
def _weights(n: int) -> np.ndarray:
    return simplex_weights(n, 3)


def _neighbourhoods(W: np.ndarray, T: int) -> np.ndarray:
    d = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
    return np.argsort(d, axis=1, kind="stable")[:, :T]


def run_imoead(wf, sc, cfg: BaselineConfig) -> RunResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    dag = DagIndex(wf)
    ev = IntervalEvaluator(wf, sc, cfg.evaluator)
    allowed = allowed_for(wf, sc)
    N = cfg.pop_size
    W = np.maximum(_weights(N), 1e-6)
    B = _neighbourhoods(W, min(cfg.neighbours, N))
    pop = init_population(wf, sc, N, rng, dag)
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    ideal = F_lo.min(axis=0)
    nadir = F_hi.max(axis=0)
    adapt = max(1, int(round(cfg.adapt_every * cfg.generations)))

    def tcheby(w, lo, hi):
        """Interval Tchebycheff on normalised objectives; the reference (ideal) maps to 0."""
        span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
        nlo, nhi = (lo - ideal) / span, (hi - ideal) / span
        return np.max(w * nlo, axis=-1), np.max(w * nhi, axis=-1)

    for gen in range(cfg.generations):
        for i in rng.permutation(N):
            pool = B[i] if rng.random() < cfg.delta else np.arange(N)
            p, q = rng.choice(pool, size=2, replace=False)
            x, o = _child(pop.X[p], pop.O[p], pop.X[q], pop.O[q], allowed, dag, rng, True)
            c_lo, c_hi = (f[0] for f in ev.evaluate_population(x[None], o[None]))   # one per subproblem per generation
            ideal = np.minimum(ideal, c_lo)
            nadir = np.maximum(nadir, c_hi)
            cand = pool
            w = W[cand]
            g_old_lo, g_old_hi = tcheby(w, F_lo[cand], F_hi[cand])
            g_new_lo, g_new_hi = tcheby(w, np.broadcast_to(c_lo, (len(cand), 3)), np.broadcast_to(c_hi, (len(cand), 3)))
            better = possibility_leq(g_new_lo, g_new_hi, g_old_lo, g_old_hi) > 0.5
            replaced = 0
            for j_idx in rng.permutation(np.flatnonzero(better)):
                j = cand[j_idx]
                pop.X[j], pop.O[j], F_lo[j], F_hi[j] = x, o, c_lo, c_hi
                replaced += 1
                if replaced >= cfg.max_replace:
                    break
        if (gen + 1) % adapt == 0 and gen + 1 < cfg.generations:   # reference-vector adaptation
            nlo, nhi = normalize_intervals(F_lo, F_hi)
            icd = interval_crowding(nlo, nhi)
            worst, best = int(np.argmin(icd)), int(np.argmax(icd))
            direction = np.maximum((nlo[best] + nhi[best]) / 2.0, 1e-6)
            W[worst] = direction / direction.sum()
            B = _neighbourhoods(W, min(cfg.neighbours, N))
    return _final(pop, F_lo, F_hi, t0)


# ------------------------------------------------------------- NSGA-III ---
def _normalise_nsga3(F: np.ndarray) -> np.ndarray:
    """Deb & Jain's adaptive normalisation of the candidate set F (rows = solutions)."""
    M = F.shape[1]
    Ft = F - F.min(axis=0)
    W = np.full((M, M), 1e-6) + np.eye(M) * (1 - 1e-6)
    extreme = np.array([np.argmin(np.max(Ft / W[j], axis=1)) for j in range(M)])
    a = None
    try:
        b = np.linalg.solve(Ft[extreme], np.ones(M))
        if np.all(b > 1e-12):
            a = 1.0 / b
            if not np.all(np.isfinite(a)) or np.any(a <= 1e-10):
                a = None
    except np.linalg.LinAlgError:
        a = None
    if a is None:
        a = Ft.max(axis=0)
    a = np.where(a > 1e-10, a, 1.0)
    return Ft / a


def niching_select(F: np.ndarray, fronts, n: int, Z: np.ndarray, rng) -> np.ndarray:
    """NSGA-III survival: indices of n members of F (midpoints), fronts in order."""
    chosen, last = [], None
    for f in fronts:
        if len(chosen) + len(f) <= n:
            chosen.extend(f.tolist())
            if len(chosen) == n:
                return np.asarray(chosen, dtype=np.int64)
        else:
            last = f
            break
    cand = np.asarray(chosen + last.tolist(), dtype=np.int64)
    Fn = _normalise_nsga3(F[cand])
    Zu = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    proj = Fn @ Zu.T
    dist = np.sqrt(np.maximum((Fn ** 2).sum(1)[:, None] - proj ** 2, 0.0))
    pi, d = dist.argmin(axis=1), dist.min(axis=1)
    k = len(chosen)
    rho = np.bincount(pi[:k], minlength=len(Z))
    pool = {}                                           # reference point -> last-front positions
    for pos in range(k, len(cand)):
        pool.setdefault(int(pi[pos]), []).append(pos)
    active = np.ones(len(Z), dtype=bool)
    picked = []
    while len(chosen) + len(picked) < n:
        r_min = rho[active].min()
        J = np.flatnonzero(active & (rho == r_min))
        j = int(rng.choice(J))
        members = pool.get(j, [])
        if not members:
            active[j] = False
            continue
        if rho[j] == 0:
            pos = min(members, key=lambda q: d[q])
        else:
            pos = members[int(rng.integers(len(members)))]
        members.remove(pos)
        picked.append(int(cand[pos]))
        rho[j] += 1
    return np.asarray(chosen + picked, dtype=np.int64)


def run_nsga3(wf, sc, cfg: BaselineConfig) -> RunResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    dag = DagIndex(wf)
    ev = IntervalEvaluator(wf, sc, cfg.evaluator)
    allowed = allowed_for(wf, sc)
    N = cfg.pop_size
    Z = _weights(N)
    pop = init_population(wf, sc, N, rng, dag)
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    for _ in range(cfg.generations):
        Xc = np.empty_like(pop.X)
        Oc = np.empty_like(pop.O)
        for i in range(N):
            p, q = rng.choice(N, size=2, replace=False)
            Xc[i], Oc[i] = _child(pop.X[p], pop.O[p], pop.X[q], pop.O[q], allowed, dag, rng, True)
        G_lo, G_hi = ev.evaluate_population(Xc, Oc)
        merged = Population.concat(pop, Population(Xc, Oc))
        M_lo, M_hi = np.concatenate([F_lo, G_lo]), np.concatenate([F_hi, G_hi])
        fronts = nondominated_fronts(midpoint_dominance_matrix(M_lo, M_hi))
        keep = niching_select((M_lo + M_hi) / 2.0, fronts, N, Z, rng)
        pop, F_lo, F_hi = merged[keep], M_lo[keep], M_hi[keep]
    return _final(pop, F_lo, F_hi, t0)


def run_baseline(name: str, wf, sc, cfg: Optional[BaselineConfig] = None) -> RunResult:
    cfg = cfg or BaselineConfig()
    if name in DOMINANCE:
        return run_ip_moea(wf, sc, cfg, DOMINANCE[name])
    if name == "nsga3_mid":
        return run_nsga3(wf, sc, cfg)
    if name == "imoead":
        return run_imoead(wf, sc, cfg)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
