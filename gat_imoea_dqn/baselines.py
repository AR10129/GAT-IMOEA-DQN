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
from .intervals import interval_dominance_matrix, nondominated_fronts, normalize_intervals, possibility_leq
from .operators import daec, mutate_allocation, mutate_order, pox
from .selection import interval_crowding, simplex_weights

BASELINES = ("ip_moea", "imoead")


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


def _rank_and_crowd(F_lo, F_hi):
    fronts = nondominated_fronts(imprecise_dominance_matrix(F_lo, F_hi))
    rank = np.empty(len(F_lo), dtype=np.int64)
    crowd = np.empty(len(F_lo))
    mid = (F_lo + F_hi) / 2.0
    for r, f in enumerate(fronts):
        rank[f] = r
        crowd[f] = crowding_distance(mid[f])
    return fronts, rank, crowd


def run_ip_moea(wf, sc, cfg: BaselineConfig) -> RunResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    dag = DagIndex(wf)
    ev = IntervalEvaluator(wf, sc, cfg.evaluator)
    allowed = allowed_for(wf, sc)
    N = cfg.pop_size
    pop = init_population(wf, sc, N, rng, dag)
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    for _ in range(cfg.generations):
        _, rank, crowd = _rank_and_crowd(F_lo, F_hi)

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
        fronts, _, crowd_m = _rank_and_crowd(M_lo, M_hi)
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
            res = ev.evaluate(x, o)                  # one evaluation per subproblem per generation
            c_lo, c_hi = res.F_lo, res.F_hi
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


def run_baseline(name: str, wf, sc, cfg: Optional[BaselineConfig] = None) -> RunResult:
    cfg = cfg or BaselineConfig()
    if name == "ip_moea":
        return run_ip_moea(wf, sc, cfg)
    if name == "imoead":
        return run_imoead(wf, sc, cfg)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
