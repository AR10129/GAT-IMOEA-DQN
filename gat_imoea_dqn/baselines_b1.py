"""B1 interval baselines from the Jin / Cui group, re-implemented from the papers.

    res = run_baseline("imoead_awn", wf, sc, BaselineConfig(generations=1000, seed=0))
    res = run_baseline("imea_ied",   wf, sc, BaselineConfig(generations=1000, seed=0))

Same fairness rules as baselines.py: our chromosome, initial population, variation
operators, evaluator, N = 105 and N new evaluations per generation; only selection differs.

IMOEA/D-AWN (Jin, Zhang, Xie, Cui; Egyptian Informatics Journal 24 (2023) 100405)
  MOEA/D with the interval Tchebycheff function g = max_k w_k (f_k - Z_k) in interval
  arithmetic (Eq. 7), Z = [min lower bound, min upper bound] per objective (the reference
  hypercube). A child made from two parents in B(i) replaces EVERY x_j in B(i) with
  P(g(child | w_j) <= g(x_j | w_j)) > 0.5, P = Sun's interval confidence level (Eqs. 3-4).
  Neighbourhood size T' = T_max exp(-G / G_max), T_max = 1.5 T, T = 10 (Eq. 12, Alg. 2),
  recomputed every generation. From G >= 0.8 G_max (rate_evol): the interval elite population
  IEP = interval non-dominated solutions of (P or IEP) and the offspring, truncated to n_IEP by
  the comprehensive ranking R; each weight vector takes its best IEP member when that member's
  g is better (Alg. 1, lines 12-16). Every wag = 100 generations (from 0.8 G_max): delete the
  n_us = 0.05 N subproblems with the worst R = alpha R_ISL + (1 - alpha) R_I, alpha = 0.8
  (Eqs. 8-10: interval sparsity level from the interval distance of Eq. 6 to the two nearest
  individuals, ranked descending; uncertainty = product of interval widths, ranked ascending)
  and insert w' = w_r (1 - c) + w_{r+1} c, c = 0.5 (|Remain| - 1) / |Remain|, between
  consecutively ranked best vectors (Eq. 11), each new vector taking its best solution in
  P and IEP. Output: the population.
  Choices the paper leaves open (ours): objectives normalised by the running ideal / nadir
  before scalarising (as in our IMOEA/D; the paper's benchmarks have comparable scales);
  n_IEP = N; "nearest" individuals by the summed interval distance over objectives; no weight
  adjustment at the very last generation (it could not affect the output).

IMEA-IED (Cui, Qu, Zhang, Jin, Cai, Zhang, Chen; Swarm Evol. Comput. 91 (2024) 101749)
  NSGA-III framework (Alg. 1). Non-dominated sorting of P u Q under the interval dominance
  based on information entropy (Alg. 2): per objective k, the convergence entropy H_c (of the
  midpoints, Eqs. 15-16) and the uncertainty entropy H_w (of the widths, Eqs. 17-18) of the
  sorted population pick the comparison: H_c < H_w -> P_c (Eq. 19, midpoint difference),
  H_c > H_w -> P_w (Eq. 20, width difference), equal -> IDM3 (Eq. 8). x dominates y when
  P_k(x <= y) >= 1 - gamma in every objective and > gamma in one, gamma = 0.5 (Eq. 10).
  Last front: the adaptive niche selection (Alg. 3) - association by perpendicular distance to
  N reference vectors, the least-crowded niche first, ties broken by the largest increase of
  the niche's summed interval crowding distance (Eqs. 13-14, 23-24), the closest member for an
  empty niche, otherwise a random one; once per generation an empty niche borrows a temporary
  vector r_new = (r_k + r_l) / 2 from the most crowding-isolated last-front solution (Eq. 25).
  Mating: random pairs, p_c = 0.9 (Section 4.3). Output: the population.
  Choices the paper leaves open (ours): association on normalised interval midpoints (min-max
  over P u Q); the "two closest solutions" of Eq. 13 by the Eq. 14 distance itself; ties in
  the crowding increment broken at random.
"""
from __future__ import annotations

import time

import numpy as np

from . import trace
from .algorithm import RunResult
from .chromosome import DagIndex, Population, allowed_for, init_population
from .evaluator import IntervalEvaluator
from .intervals import nondominated_fronts

# ===================================================== interval helpers ===


def confidence_level(a_lo, a_hi, b_lo, b_hi):
    """Sun's interval confidence level P(A <= B) (AWN Eqs. 3-4), elementwise.
    K = [smallest of the four bounds, second smallest]; d(B, K) = sqrt(((B_lo - K_lo)^2 +
    (B_hi - K_hi)^2) / 2); P = d(B, K) / (d(A, K) + d(B, K)), 0.5 when both are 0."""
    a_lo, a_hi, b_lo, b_hi = np.broadcast_arrays(*(np.asarray(v, float) for v in (a_lo, a_hi, b_lo, b_hi)))
    s = np.sort(np.stack([a_lo, a_hi, b_lo, b_hi]), axis=0)
    k_lo, k_hi = s[0], s[1]
    dA = np.sqrt(((a_lo - k_lo) ** 2 + (a_hi - k_hi) ** 2) / 2.0)
    dB = np.sqrt(((b_lo - k_lo) ** 2 + (b_hi - k_hi) ** 2) / 2.0)
    tot = dA + dB
    return np.where(tot > 0, dB / np.where(tot > 0, tot, 1.0), 0.5)


def confidence_dominance_matrix(F_lo, F_hi, gamma: float = 0.5) -> np.ndarray:
    """AWN Def. 4: i dominates j iff P(f_k(i) <= f_k(j)) >= gamma for all k, > gamma for one."""
    P = confidence_level(F_lo[:, None, :], F_hi[:, None, :], F_lo[None, :, :], F_hi[None, :, :])
    D = np.all(P >= gamma, axis=2) & np.any(P > gamma, axis=2)
    np.fill_diagonal(D, False)
    return D


def interval_distance(a_lo, a_hi, b_lo, b_hi):
    """AWN Eq. 6: sqrt((M(A) - M(B))^2 + (w(A)^2 + w(B)^2) / 3 - 2 w(A n B)^2 / 3), w = half-width."""
    mA, mB = (a_lo + a_hi) / 2, (b_lo + b_hi) / 2
    wA, wB = (a_hi - a_lo) / 2, (b_hi - b_lo) / 2
    wI = np.maximum(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0.0) / 2
    return np.sqrt(np.maximum((mA - mB) ** 2 + (wA ** 2 + wB ** 2) / 3 - 2 * wI ** 2 / 3, 0.0))


def _neighbourhoods(W: np.ndarray, T: int) -> np.ndarray:
    d = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
    return np.argsort(d, axis=1, kind="stable")[:, :T]


def _norm(lo, hi, ideal, nadir):
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    return (lo - ideal) / span, (hi - ideal) / span


# ========================================================== IMOEA/D-AWN ===


def awn_tcheby(W, nlo, nhi, z_lo, z_hi):
    """Interval Tchebycheff (AWN Eq. 7) on normalised objectives: w_k (f_k - Z_k) with interval
    subtraction [f_lo - Z_hi, f_hi - Z_lo], then the interval maximum over k."""
    return np.max(W * (nlo - z_hi), axis=-1), np.max(W * (nhi - z_lo), axis=-1)


def awn_ranking(nlo, nhi, alpha: float = 0.8) -> np.ndarray:
    """Comprehensive ranking R (AWN Eqs. 8-10); larger = denser and more uncertain = worse."""
    n = len(nlo)
    Dk = interval_distance(nlo[:, None, :], nhi[:, None, :], nlo[None, :, :], nhi[None, :, :])  # n x n x M
    dsum = Dk.sum(axis=2)
    np.fill_diagonal(dsum, np.inf)
    isl = np.zeros(n)
    if n > 2:
        nearest = np.argsort(dsum, axis=1, kind="stable")[:, :2]
        prod = np.prod(Dk[np.arange(n)[:, None], nearest], axis=2)                         # n x 2
        isl = prod.mean(axis=1)
    unc = np.prod(nhi - nlo, axis=1)
    r_isl = np.empty(n)
    r_isl[np.argsort(-isl, kind="stable")] = np.arange(1, n + 1)
    r_i = np.empty(n)
    r_i[np.argsort(unc, kind="stable")] = np.arange(1, n + 1)
    return alpha * r_isl + (1 - alpha) * r_i


def run_imoead_awn(wf, sc, cfg) -> RunResult:
    from .baselines import _child, _final, _weights
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    dag = DagIndex(wf)
    ev = IntervalEvaluator(wf, sc, cfg.evaluator)
    allowed = allowed_for(wf, sc)
    N, G_max = cfg.pop_size, cfg.generations
    T0, alpha, rate_evol, wag = 10, 0.8, 0.8, 100
    n_us = max(1, int(round(0.05 * N)))
    n_iep = N
    T_max = 1.5 * T0
    W = np.maximum(_weights(N), 1e-6)
    pop = init_population(wf, sc, N, rng, dag)
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    ideal, nadir = F_lo.min(axis=0), F_hi.max(axis=0)
    iep = None                                             # (Population, F_lo, F_hi)

    def T_at(G):
        return int(min(N, max(2, round(T_max * np.exp(-G / G_max)))))

    B = _neighbourhoods(W, T_at(1))
    trace.record(0, F_lo, F_hi)
    for G in range(1, G_max):                              # Alg. 3: G = 1 .. G_max - 1
        Xo, Oo, Lo, Ho = [], [], [], []
        for i in range(N):
            pool = B[i]
            p, q = rng.choice(pool, size=2, replace=False)
            x, o = _child(pop.X[p], pop.O[p], pop.X[q], pop.O[q], allowed, dag, rng, rng.random() < 0.9)
            c_lo, c_hi = (f[0] for f in ev.evaluate_population(x[None], o[None]))
            Xo.append(x), Oo.append(o), Lo.append(c_lo), Ho.append(c_hi)
            ideal, nadir = np.minimum(ideal, c_lo), np.maximum(nadir, c_hi)
            nlo, nhi = _norm(F_lo[pool], F_hi[pool], ideal, nadir)
            clo, chi = _norm(c_lo, c_hi, ideal, nadir)
            Pn_lo, Pn_hi = _norm(F_lo, F_hi, ideal, nadir)
            z_lo, z_hi = Pn_lo.min(axis=0), Pn_hi.min(axis=0)
            z_lo, z_hi = np.minimum(z_lo, clo), np.minimum(z_hi, chi)    # reference hypercube incl. the child
            w = W[pool]
            g_old = awn_tcheby(w, nlo, nhi, z_lo, z_hi)
            g_new = awn_tcheby(w, np.broadcast_to(clo, nlo.shape), np.broadcast_to(chi, nhi.shape), z_lo, z_hi)
            better = confidence_level(g_new[0], g_new[1], g_old[0], g_old[1]) > 0.5
            for j in pool[better]:
                pop.X[j], pop.O[j], F_lo[j], F_hi[j] = x, o, c_lo, c_hi
        G_next = G + 1
        B = _neighbourhoods(W, T_at(G_next))
        if G_next >= rate_evol * G_max:
            O_pop = Population(np.array(Xo), np.array(Oo))
            O_lo, O_hi = np.array(Lo), np.array(Ho)
            base = (pop, F_lo, F_hi) if iep is None else iep
            U = Population.concat(base[0], O_pop)
            U_lo, U_hi = np.concatenate([base[1], O_lo]), np.concatenate([base[2], O_hi])
            nd = nondominated_fronts(confidence_dominance_matrix(*_norm(U_lo, U_hi, ideal, nadir)))[0]
            if len(nd) > n_iep:
                R = awn_ranking(*_norm(U_lo[nd], U_hi[nd], ideal, nadir), alpha)
                nd = nd[np.argsort(R, kind="stable")[:n_iep]]
            iep = (U[nd], U_lo[nd].copy(), U_hi[nd].copy())
            # each weight vector takes its best IEP member if that one is better (Alg. 1, 12-16)
            Pn_lo, Pn_hi = _norm(F_lo, F_hi, ideal, nadir)
            En_lo, En_hi = _norm(iep[1], iep[2], ideal, nadir)
            z_lo = np.minimum(Pn_lo.min(0), En_lo.min(0))
            z_hi = np.minimum(Pn_hi.min(0), En_hi.min(0))
            ge_lo, ge_hi = awn_tcheby(W[:, None, :], En_lo[None], En_hi[None], z_lo, z_hi)     # N x |IEP|
            best = np.argmin((ge_lo + ge_hi) / 2, axis=1)
            gp_lo, gp_hi = awn_tcheby(W, Pn_lo, Pn_hi, z_lo, z_hi)
            gb_lo, gb_hi = ge_lo[np.arange(N), best], ge_hi[np.arange(N), best]
            for j in np.flatnonzero(confidence_level(gb_lo, gb_hi, gp_lo, gp_hi) > 0.5):
                b = best[j]
                pop.X[j], pop.O[j] = iep[0].X[b].copy(), iep[0].O[b].copy()
                F_lo[j], F_hi[j] = iep[1][b], iep[2][b]
            if G_next % wag == 0 and G_next < G_max:      # weight-vector adjustment (Alg. 1, 17-28)
                R = awn_ranking(*_norm(F_lo, F_hi, ideal, nadir), alpha)
                order = np.argsort(R, kind="stable")       # best first
                keep = order[:N - n_us]
                Wk = W[keep]
                rem = len(keep)
                c = 0.5 * (rem - 1) / rem
                W_new = np.array([Wk[r] * (1 - c) + Wk[r + 1] * c for r in range(n_us)])
                pop_k, Fk_lo, Fk_hi = pop[keep], F_lo[keep], F_hi[keep]
                # each new vector takes its best solution from P and IEP
                C_pop = Population.concat(pop_k, iep[0])
                C_lo, C_hi = np.concatenate([Fk_lo, iep[1]]), np.concatenate([Fk_hi, iep[2]])
                Cn_lo, Cn_hi = _norm(C_lo, C_hi, ideal, nadir)
                gz_lo, gz_hi = Cn_lo.min(0), Cn_hi.min(0)
                gn_lo, gn_hi = awn_tcheby(W_new[:, None, :], Cn_lo[None], Cn_hi[None], gz_lo, gz_hi)
                pick = np.argmin((gn_lo + gn_hi) / 2, axis=1)
                pop = Population.concat(pop_k, C_pop[pick])
                F_lo = np.concatenate([Fk_lo, C_lo[pick]])
                F_hi = np.concatenate([Fk_hi, C_hi[pick]])
                W = np.vstack([Wk, W_new])
                W = np.maximum(W / W.sum(axis=1, keepdims=True), 1e-6)
                B = _neighbourhoods(W, T_at(G_next))
        trace.record(G, F_lo, F_hi)
    return _final(pop, F_lo, F_hi, t0)


# ============================================================= IMEA-IED ===


def _entropy(v: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy per column of a non-negative matrix (IED Eqs. 15-18)."""
    n = len(v)
    s = v.sum(axis=0)
    p = np.where(s > 0, v / np.where(s > 0, s, 1.0), 1.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=0) / np.log(n)
    return h


def ied_probabilities(F_lo: np.ndarray, F_hi: np.ndarray) -> np.ndarray:
    """P[i, j, k] = P_k(f_k(i) <= f_k(j)) under the entropy-chosen relation (IED Alg. 2)."""
    m = (F_lo + F_hi) / 2
    w = F_hi - F_lo
    Hc, Hw = _entropy(m), _entropy(w)
    dm = m[:, None, :] - m[None, :, :]                     # m(a) - m(b)
    dw = w[:, None, :] - w[None, :, :]                     # w(a) - w(b)
    S = w[:, None, :] + w[None, :, :]
    x3 = 0.5 + dm / np.where(S > 0, S, 1.0)                # IDM3:  (a_hi - b_lo) / (w(a) + w(b))
    xc = 0.5 + 2 * dm / np.where(S > 0, S, 1.0)            # Eq. 19: minus delta_m = m(b) - m(a)
    xw = 0.5 + (dm + dw) / np.where(S > 0, S, 1.0)         # Eq. 20: minus delta_w = w(b) - w(a)
    x = np.where(Hc < Hw, xc, np.where(Hc > Hw, xw, x3))
    P = np.clip(1.0 - x, 0.0, 1.0)
    crisp = np.where(dm < 0, 1.0, np.where(dm > 0, 0.0, 0.5))   # both intervals are points
    return np.where(S > 0, P, crisp)


def ied_dominance_matrix(F_lo, F_hi, gamma: float = 0.5) -> np.ndarray:
    P = ied_probabilities(F_lo, F_hi)
    D = np.all(P >= 1 - gamma, axis=2) & np.any(P > gamma, axis=2)
    np.fill_diagonal(D, False)
    return D


def ied_crowding(nlo, nhi) -> np.ndarray:
    """IED Eqs. 13-14: mean of the distances to the two closest solutions, with
    D(x1, x2) = sum_g |m_g(x1) - m_g(x2)| / (overlap(x1, x2) + V(x1) + V(x2) + 1)."""
    n = len(nlo)
    if n < 2:
        return np.zeros(n)
    m = (nlo + nhi) / 2
    V = np.prod(nhi - nlo, axis=1)
    ov = np.prod(np.maximum(np.minimum(nhi[:, None, :], nhi[None, :, :]) -
                            np.maximum(nlo[:, None, :], nlo[None, :, :]), 0.0), axis=2)
    D = np.abs(m[:, None, :] - m[None, :, :]).sum(axis=2) / (ov + V[:, None] + V[None, :] + 1.0)
    np.fill_diagonal(D, np.inf)
    k = min(2, n - 1)
    return np.sort(D, axis=1)[:, :k].mean(axis=1)


def ied_niche_select(nlo, nhi, fronts, n: int, Z: np.ndarray, rng, C_prev: np.ndarray):
    """IED Alg. 3 on candidates P_{t+1} u F_l; returns (selected indices, C_t per vector)."""
    chosen, last = [], None
    for f in fronts:
        if len(chosen) + len(f) <= n:
            chosen.extend(f.tolist())
        else:
            last = f
            break
    cand = np.asarray(chosen + ([] if last is None else last.tolist()), dtype=np.int64)
    mid = (nlo[cand] + nhi[cand]) / 2
    Zu = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    proj = mid @ Zu.T
    d2 = np.sqrt(np.maximum((mid ** 2).sum(1)[:, None] - proj ** 2, 0.0))        # |cand| x H
    pi = d2.argmin(axis=1)
    cd = ied_crowding(nlo[cand], nhi[cand])
    C_t = np.bincount(pi, weights=cd, minlength=len(Z))
    if last is None or len(chosen) == n:
        return np.asarray(chosen[:n], dtype=np.int64), C_t
    CI = C_t - C_prev
    k = len(chosen)
    rho = np.bincount(pi[:k], minlength=len(Z))
    remaining = set(range(k, len(cand)))
    active = np.ones(len(Z), dtype=bool)
    picked, flag = [], False
    while k + len(picked) < n:
        r_min = rho[active].min()
        Jm = np.flatnonzero(active & (rho == r_min))
        if len(Jm) > 1:
            top = CI[Jm].max()
            Jm = Jm[CI[Jm] == top]
        J = int(rng.choice(Jm))
        IJ = [q for q in remaining if pi[q] == J]
        if IJ:
            q = min(IJ, key=lambda u: d2[u, J]) if rho[J] == 0 else IJ[int(rng.integers(len(IJ)))]
        else:
            q = None
            if not flag:                                   # temporary vector r_new (Eq. 25), once per generation
                rem = sorted(remaining)
                xk = rem[int(np.argmax(cd[rem]))]
                rk, rl = np.argsort(d2[xk], kind="stable")[:2]
                r_new = (Z[rk] + Z[rl]) / 2
                r_new = r_new / np.linalg.norm(r_new)
                pr = mid[rem] @ r_new
                dn = np.sqrt(np.maximum((mid[rem] ** 2).sum(1) - pr ** 2, 0.0))
                assoc = [u for u, dd in zip(rem, dn) if dd < d2[u].min()]
                if assoc:
                    q = min(assoc, key=lambda u: dn[rem.index(u)])
                    flag = True
            if q is None:
                active[J] = False
                continue
        remaining.discard(q)
        picked.append(q)
        rho[J] += 1
    return cand[np.asarray(list(range(k)) + picked, dtype=np.int64)], C_t


def run_imea_ied(wf, sc, cfg) -> RunResult:
    from .baselines import _child, _final, _weights
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    dag = DagIndex(wf)
    ev = IntervalEvaluator(wf, sc, cfg.evaluator)
    allowed = allowed_for(wf, sc)
    N = cfg.pop_size
    Z = _weights(N)
    C_prev = np.zeros(len(Z))
    pop = init_population(wf, sc, N, rng, dag)
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    trace.record(0, F_lo, F_hi)
    for gen in range(cfg.generations):
        Xc = np.empty_like(pop.X)
        Oc = np.empty_like(pop.O)
        for i in range(N):
            p, q = rng.choice(N, size=2, replace=False)
            Xc[i], Oc[i] = _child(pop.X[p], pop.O[p], pop.X[q], pop.O[q], allowed, dag, rng, rng.random() < 0.9)
        G_lo, G_hi = ev.evaluate_population(Xc, Oc)
        merged = Population.concat(pop, Population(Xc, Oc))
        M_lo, M_hi = np.concatenate([F_lo, G_lo]), np.concatenate([F_hi, G_hi])
        fronts = nondominated_fronts(ied_dominance_matrix(M_lo, M_hi))
        lo_, hi_ = M_lo.min(axis=0), M_hi.max(axis=0)
        nlo, nhi = _norm(M_lo, M_hi, lo_, hi_)
        keep, C_prev = ied_niche_select(nlo, nhi, fronts, N, Z, rng, C_prev)
        pop, F_lo, F_hi = merged[keep], M_lo[keep], M_hi[keep]
        trace.record(gen + 1, F_lo, F_hi)
    return _final(pop, F_lo, F_hi, t0)
