"""Interval critical-path evaluator: makespan F1, energy F2 and cost F3.

Implements sections 2.2-2.3 of GAT-IMOEA-DQN_Workflow.pdf for one individual
(allocation string X, priority string O) on a WorkflowDAG + Scenario.

    ev = IntervalEvaluator(load_workflow(...), load_scenario(...))
    res = ev.evaluate(X, O)            # res.F_lo, res.F_hi: [makespan s, energy J, cost $]
    F_lo, F_hi = ev.evaluate_population(Xs, Os)      # (P, 3) each

Node ids follow the PDF: 0 = origin device, 1..n = edge servers, n+1 = cloud.

Model, per the PDF unless marked (choice):
  * Effective capacity: device f_i; edge f_u / |{j : x_j = u}| (equal static
    share over every task placed on u); cloud f_C per task.
  * t_comp = C / f_eff.  Transfer on edge (p, k) = D_pk / R(x_p, x_k), zero
    when co-located.  Rates come from Scenario.rate_matrix_mbps().
  * AT(k) = max over parents (FT(p) + transfer); [0, 0] for entry tasks.
    ST(k) = max(AT(k), avail(x_k)),  FT(k) = ST(k) + t_comp(k).
    Tasks are list-scheduled in the order O.
  * Queues (choice): the device runs one task at a time; the cloud has no
    queue ("capacity per task"); edge servers share their CPU instead of
    queueing, because the f_u / n_u share already models contention.
    Set EvaluatorConfig(edge_queue=True) to also queue on edge servers.
  * Energy: terminal = sum_{local} C e_i + p_i * (time spent uploading data for
    offloaded tasks) + p_idle * (compute time of offloaded tasks);
    edge = sum_u kappa_E f_u^2 sum C + P_static_u * max FT on u;
    cloud (choice; not defined in the PDF) = kappa_C f_C^2 sum C + P_static_C * max FT.
  * Cost: eta_E * f_eff * t_comp on edges + eta_C * t_comp on the cloud,
    evaluated with plain interval arithmetic, as written in the PDF.
  * External I/O (choice): external_io="pdf" ignores files that no task
    produces/consumes (entry tasks start at 0). external_io="device" uploads
    each offloaded task's external inputs from the device before it may start,
    and returns external outputs of offloaded tasks to the device (counted in
    makespan and upload/idle energy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from dataset_tools.loader import KB_TO_MBIT, Scenario, WorkflowDAG

try:                                    # numba is optional; same results without it
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                     # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap if not (args and callable(args[0])) else args[0]

OBJECTIVES = ("makespan_s", "energy_j", "cost_usd")


def _xfer(kb: np.ndarray, sec_per_kb: np.ndarray) -> np.ndarray:
    """kb * sec_per_kb, with 0 KB taking 0 s even over an unreachable (inf) link."""
    with np.errstate(invalid="ignore"):
        return np.where(kb > 0, kb * sec_per_kb, 0.0)


def _schedule_py(order, X, pred_ptr, pred_edge, src, tt_lo, tt_hi, tc_lo, tc_hi,
                 ready_lo, ready_hi, queued, n_nodes):
    K = order.shape[0]
    ST_lo = np.zeros(K)
    ST_hi = np.zeros(K)
    FT_lo = np.zeros(K)
    FT_hi = np.zeros(K)
    av_lo = np.zeros(n_nodes)
    av_hi = np.zeros(n_nodes)
    for i in range(K):
        v = order[i]
        a_lo = ready_lo[v]
        a_hi = ready_hi[v]
        for j in range(pred_ptr[v], pred_ptr[v + 1]):
            e = pred_edge[j]
            p = src[e]
            t = FT_lo[p] + tt_lo[e]
            if t > a_lo:
                a_lo = t
            t = FT_hi[p] + tt_hi[e]
            if t > a_hi:
                a_hi = t
        node = X[v]
        if queued[node]:
            if av_lo[node] > a_lo:
                a_lo = av_lo[node]
            if av_hi[node] > a_hi:
                a_hi = av_hi[node]
        ST_lo[v] = a_lo
        ST_hi[v] = a_hi
        FT_lo[v] = a_lo + tc_lo[v]
        FT_hi[v] = a_hi + tc_hi[v]
        if queued[node]:
            av_lo[node] = FT_lo[v]
            av_hi[node] = FT_hi[v]
    return ST_lo, ST_hi, FT_lo, FT_hi


_schedule = njit(cache=True)(_schedule_py) if HAVE_NUMBA else _schedule_py


@njit(cache=True)
def _xf(kb, spk):
    return kb * spk if kb > 0 else 0.0


@njit(cache=True)
def _psum(a, lo, n):
    """Sum of a[lo:lo+n] in exactly NumPy's pairwise order, so the fast path is bit-identical.
    NumPy splits n > 128 in halves (rounded to multiples of 8) recursively; this walks the same
    tree with an explicit stack, because numba cannot cache recursive functions safely."""
    if n <= 128:
        return _pblock(a, lo, n)
    s_lo = np.empty(64, dtype=np.int64)
    s_n = np.empty(64, dtype=np.int64)
    s_st = np.empty(64, dtype=np.int64)
    vals = np.empty(64)
    sp = 0
    vp = 0
    s_lo[0], s_n[0], s_st[0] = lo, n, 0
    sp = 1
    while sp > 0:
        sp -= 1
        l, m, st = s_lo[sp], s_n[sp], s_st[sp]
        if m <= 128:
            vals[vp] = _pblock(a, l, m)
            vp += 1
        elif st == 0:
            m2 = m // 2
            m2 -= m2 % 8
            s_lo[sp], s_n[sp], s_st[sp] = l, m, 1              # combine after both halves
            s_lo[sp + 1], s_n[sp + 1], s_st[sp + 1] = l + m2, m - m2, 0
            s_lo[sp + 2], s_n[sp + 2], s_st[sp + 2] = l, m2, 0  # left half first
            sp += 3
        else:
            right = vals[vp - 1]
            left = vals[vp - 2]
            vp -= 1
            vals[vp - 1] = left + right
    return vals[0]


@njit(cache=True)
def _pblock(a, lo, n):
    if n < 8:
        r = 0.0
        for i in range(n):
            r += a[lo + i]
        return r
    if n <= 128:
        r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
        r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
        i = 8
        while i < n - (n % 8):
            r0 += a[lo + i]
            r1 += a[lo + i + 1]
            r2 += a[lo + i + 2]
            r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]
            r5 += a[lo + i + 5]
            r6 += a[lo + i + 6]
            r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    return 0.0                                  # not reached: callers pass n <= 128


@njit(cache=True)
def _evaluate_population_nb(Xs, Os, validate, allowed_mask, dev_node, near_node, cloud, NI, is_edge, queued,
                            f_lo, f_hi, C_lo, C_hi, src, dst, D_lo, D_hi, pred_ptr, pred_edge,
                            spk_lo, spk_hi, device_io, ext_in, ext_out, sum_apps, app, n_apps,
                            e_lo_t, e_hi_t, p_tx_t, p_idle_t, kappa_E, kappa_C, P_static, eta_E, eta_C,
                            device_energy):
    """Fast path of IntervalEvaluator.evaluate for a whole population (same model, one compiled
    loop). Returns F_lo, F_hi (P, 3), feasible (P,), error (P,): 0 ok, 1 invalid individual."""
    P, K = Xs.shape
    E = src.shape[0]
    F_lo = np.empty((P, 3))
    F_hi = np.empty((P, 3))
    feas = np.ones(P, dtype=np.bool_)
    err = np.zeros(P, dtype=np.int64)
    I = np.empty(K, dtype=np.int64)
    cnt = np.zeros(NI, dtype=np.int64)
    pos = np.empty(K, dtype=np.int64)
    feff_lo = np.empty(K)
    feff_hi = np.empty(K)
    tc_lo = np.empty(K)
    tc_hi = np.empty(K)
    tt_lo = np.empty(E)
    tt_hi = np.empty(E)
    ready_lo = np.zeros(K)
    ready_hi = np.zeros(K)
    app_lo = np.zeros(n_apps)
    app_hi = np.zeros(n_apps)
    work_lo = np.zeros(NI)
    work_hi = np.zeros(NI)
    busy_lo = np.zeros(NI)
    busy_hi = np.zeros(NI)
    b_lo = np.empty(max(K, E, NI, n_apps) + 1)        # scratch for NumPy-order sums
    b_hi = np.empty(max(K, E, NI, n_apps) + 1)
    N = allowed_mask.shape[1]
    for p in range(P):
        X = Xs[p]
        O = Os[p]
        if validate:
            bad = False
            for v in range(K):
                if X[v] < 0 or X[v] >= N or not allowed_mask[v, X[v]]:
                    bad = True
                pos[v] = -1
            for i in range(K):
                o = O[i]
                if o < 0 or o >= K or pos[o] >= 0:
                    bad = True
                else:
                    pos[o] = i
            if not bad:
                for e in range(E):
                    if pos[src[e]] >= pos[dst[e]]:
                        bad = True
                        break
            if bad:
                err[p] = 1
                continue
        cnt[:] = 0
        for v in range(K):
            I[v] = dev_node[v] if X[v] == 0 else X[v]
            cnt[I[v]] += 1
        for v in range(K):
            s = cnt[I[v]] if (is_edge[I[v]] and cnt[I[v]] > 1) else 1
            feff_lo[v] = f_lo[I[v]] / s
            feff_hi[v] = f_hi[I[v]] / s
            tc_lo[v] = C_lo[v] / feff_hi[v]
            tc_hi[v] = C_hi[v] / feff_lo[v]
        ok = True
        for e in range(E):
            a, b = I[src[e]], I[dst[e]]
            tt_lo[e] = _xf(D_lo[e], spk_lo[a, b])
            tt_hi[e] = _xf(D_hi[e], spk_hi[a, b])
            if not np.isfinite(tt_hi[e]):
                ok = False
        if device_io:
            for v in range(K):
                ready_lo[v] = _xf(ext_in[v], spk_lo[dev_node[v], I[v]])
                ready_hi[v] = _xf(ext_in[v], spk_hi[dev_node[v], I[v]])
                if not np.isfinite(ready_hi[v]) or not np.isfinite(_xf(ext_out[v], spk_hi[I[v], dev_node[v]])):
                    ok = False
        feas[p] = ok
        ST_lo, ST_hi, FT_lo, FT_hi = _schedule(O, I, pred_ptr, pred_edge, src, tt_lo, tt_hi, tc_lo, tc_hi,
                                               ready_lo, ready_hi, queued, NI)
        # F1
        mk_lo = 0.0
        mk_hi = 0.0
        if sum_apps:
            app_lo[:] = 0.0
            app_hi[:] = 0.0
        for v in range(K):
            el = FT_lo[v]
            eh = FT_hi[v]
            if device_io:
                el += _xf(ext_out[v], spk_lo[I[v], dev_node[v]])
                eh += _xf(ext_out[v], spk_hi[I[v], dev_node[v]])
            if sum_apps:
                if el > app_lo[app[v]]:
                    app_lo[app[v]] = el
                if eh > app_hi[app[v]]:
                    app_hi[app[v]] = eh
            else:
                if v == 0 or el > mk_lo:
                    mk_lo = el
                if v == 0 or eh > mk_hi:
                    mk_hi = eh
        if sum_apps:
            mk_lo = _psum(app_lo, 0, n_apps)
            mk_hi = _psum(app_hi, 0, n_apps)
        # F2: device side (each sum gathers its terms in task / edge order, then sums as NumPy does)
        m = 0
        for v in range(K):
            if X[v] == 0:
                b_lo[m] = C_lo[v] * e_lo_t[v]
                b_hi[m] = C_hi[v] * e_hi_t[v]
                m += 1
        ec_lo = _psum(b_lo, 0, m)
        ec_hi = _psum(b_hi, 0, m)
        m = 0
        for e in range(E):                       # parent on its device -> offloaded child
            s = src[e]
            if X[s] == 0 and X[dst[e]] != 0:
                d = I[dst[e]]
                nr = near_node[s]
                to_c = d == cloud and nr > 0
                wn = nr if to_c else d
                w_lo = _xf(D_lo[e], spk_lo[nr, cloud]) if to_c else 0.0
                w_hi = _xf(D_hi[e], spk_hi[nr, cloud]) if to_c else 0.0
                b_lo[m] = p_tx_t[s] * _xf(D_lo[e], spk_lo[dev_node[s], wn]) + p_idle_t[s] * w_lo
                b_hi[m] = p_tx_t[s] * _xf(D_hi[e], spk_hi[dev_node[s], wn]) + p_idle_t[s] * w_hi
                m += 1
        tx_lo = _psum(b_lo, 0, m)
        tx_hi = _psum(b_hi, 0, m)
        if device_io:
            m = 0
            for v in range(K):                   # external inputs of offloaded tasks
                if X[v] != 0:
                    d = I[v]
                    nr = near_node[v]
                    to_c = d == cloud and nr > 0
                    wn = nr if to_c else d
                    w_lo = _xf(ext_in[v], spk_lo[nr, cloud]) if to_c else 0.0
                    w_hi = _xf(ext_in[v], spk_hi[nr, cloud]) if to_c else 0.0
                    b_lo[m] = p_tx_t[v] * _xf(ext_in[v], spk_lo[dev_node[v], wn]) + p_idle_t[v] * w_lo
                    b_hi[m] = p_tx_t[v] * _xf(ext_in[v], spk_hi[dev_node[v], wn]) + p_idle_t[v] * w_hi
                    m += 1
            tx_lo += _psum(b_lo, 0, m)
            tx_hi += _psum(b_hi, 0, m)
        m = 0
        for v in range(K):
            if X[v] != 0:
                b_lo[m] = p_idle_t[v] * tc_lo[v]
                b_hi[m] = p_idle_t[v] * tc_hi[v]
                m += 1
        ei_lo = _psum(b_lo, 0, m)
        ei_hi = _psum(b_hi, 0, m)
        # F2: edge and cloud
        eE_lo = 0.0
        eE_hi = 0.0
        eC_lo = 0.0
        eC_hi = 0.0
        if not device_energy:
            work_lo[:] = 0.0
            work_hi[:] = 0.0
            busy_lo[:] = 0.0
            busy_hi[:] = 0.0
            for v in range(K):
                work_lo[I[v]] += C_lo[v]
                work_hi[I[v]] += C_hi[v]
                if FT_lo[v] > busy_lo[I[v]]:
                    busy_lo[I[v]] = FT_lo[v]
                if FT_hi[v] > busy_hi[I[v]]:
                    busy_hi[I[v]] = FT_hi[v]
            m = 0
            for u in range(NI):
                if is_edge[u]:
                    b_lo[m] = kappa_E * f_lo[u] ** 2 * work_lo[u] + P_static[u] * busy_lo[u]
                    b_hi[m] = kappa_E * f_hi[u] ** 2 * work_hi[u] + P_static[u] * busy_hi[u]
                    m += 1
            eE_lo = _psum(b_lo, 0, m)
            eE_hi = _psum(b_hi, 0, m)
            eC_lo = kappa_C * f_lo[cloud] ** 2 * work_lo[cloud] + P_static[cloud] * busy_lo[cloud]
            eC_hi = kappa_C * f_hi[cloud] ** 2 * work_hi[cloud] + P_static[cloud] * busy_hi[cloud]
        # F3
        m = 0
        for v in range(K):
            if is_edge[I[v]]:
                b_lo[m] = feff_lo[v] * tc_lo[v]
                b_hi[m] = feff_hi[v] * tc_hi[v]
                m += 1
        ce_lo = _psum(b_lo, 0, m)
        ce_hi = _psum(b_hi, 0, m)
        m = 0
        for v in range(K):
            if I[v] == cloud:
                b_lo[m] = tc_lo[v]
                b_hi[m] = tc_hi[v]
                m += 1
        cc_lo = _psum(b_lo, 0, m)
        cc_hi = _psum(b_hi, 0, m)
        F_lo[p, 0] = mk_lo
        F_hi[p, 0] = mk_hi
        F_lo[p, 1] = ec_lo + tx_lo + ei_lo + eE_lo + eC_lo
        F_hi[p, 1] = ec_hi + tx_hi + ei_hi + eE_hi + eC_hi
        F_lo[p, 2] = eta_E * ce_lo + eta_C * cc_lo
        F_hi[p, 2] = eta_E * ce_hi + eta_C * cc_hi
    return F_lo, F_hi, feas, err


@dataclass(frozen=True)
class EvaluatorConfig:
    external_io: str = "pdf"            # "pdf" or "device"
    edge_queue: bool = False
    validate: bool = True               # check X is reachable and O is topological
    # "all": device + edge + cloud energy (the workflow PDF); "device": device-side
    # energy only (local compute, uploads, idle), as in Jin et al. (2026)
    energy_scope: str = "all"
    # F1: "max" = latest finish over all exit tasks (the PDF); "sum_apps" = sum over
    # applications of each application's makespan (Suite A, many devices). With one
    # application they are the same; with single-task applications and no device queue,
    # "sum_apps" is exactly Jin et al.'s summed task latency.
    makespan: str = "max"
    device_queue: bool = True           # a device runs one task at a time (False: unlimited, as in Jin et al.)


@dataclass
class EvalResult:
    F_lo: np.ndarray                    # (3,) makespan [s], energy [J], cost [$]
    F_hi: np.ndarray
    feasible: bool = True
    breakdown: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    ST: Optional[Tuple[np.ndarray, np.ndarray]] = None   # per task, when detail=True
    FT: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def midpoint(self) -> np.ndarray:
        return (self.F_lo + self.F_hi) / 2.0

    @property
    def width(self) -> np.ndarray:
        return self.F_hi - self.F_lo


class InvalidIndividual(ValueError):
    pass


class IntervalEvaluator:
    def __init__(self, wf: WorkflowDAG, sc: Scenario, config: EvaluatorConfig = EvaluatorConfig()):
        if config.external_io not in ("pdf", "device"):
            raise ValueError("external_io must be 'pdf' or 'device'")
        if config.energy_scope not in ("all", "device"):
            raise ValueError("energy_scope must be 'all' or 'device'")
        if config.makespan not in ("max", "sum_apps"):
            raise ValueError("makespan must be 'max' or 'sum_apps'")
        self.wf, self.sc, self.cfg = wf, sc, config
        K, n, m = wf.num_tasks, sc.n_edge, sc.num_devices
        self.K, self.n, self.m = K, n, m
        self.N = n + 2                  # allocation genes 0..n+1 (0 = the task's own device)
        self.NI = n + 2 + m             # internal nodes: 1..n edges, n+1 cloud, n+2+d device d
        self.cloud = n + 1
        self.owner = (np.asarray(wf.owner, dtype=np.int64) if wf.owner is not None
                      else np.full(K, sc.origin, dtype=np.int64))
        if self.owner.min() < 0 or self.owner.max() >= m:
            raise ValueError(f"task owners must be devices 0..{m - 1}")
        self.dev_node = n + 2 + self.owner          # internal node of each task's own device
        app = np.asarray(wf.app, dtype=np.int64) if wf.app is not None else np.zeros(K, dtype=np.int64)
        _, self.app = np.unique(app, return_inverse=True)
        self.n_apps = int(self.app.max()) + 1 if K else 0

        self.C_lo, self.C_hi = wf.C_lo.astype(float), wf.C_hi.astype(float)
        self.src = wf.edge_index[0].astype(np.int64)
        self.dst = wf.edge_index[1].astype(np.int64)
        self.D_lo, self.D_hi = wf.D_lo.astype(float), wf.D_hi.astype(float)
        self.ext_in, self.ext_out = wf.ext_input_kb.astype(float), wf.ext_output_kb.astype(float)

        order = np.argsort(self.dst, kind="stable")
        self.pred_edge = order.astype(np.int64)
        self.pred_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.dst, minlength=K))]).astype(np.int64)
        self.is_exit = wf.out_degree == 0

        # per internal node capacity [GHz]
        NI = self.NI
        self.f_lo = np.ones(NI)
        self.f_hi = np.ones(NI)
        self.f_lo[1:n + 1], self.f_hi[1:n + 1] = sc.f_edge[:, 0], sc.f_edge[:, 1]
        self.f_lo[self.cloud], self.f_hi[self.cloud] = sc.f_cloud
        self.f_lo[n + 2:], self.f_hi[n + 2:] = sc.dev_f[:, 0], sc.dev_f[:, 1]
        self.is_edge = np.zeros(NI, dtype=bool)
        self.is_edge[1:n + 1] = True
        self.queued = np.zeros(NI, dtype=np.bool_)
        self.queued[n + 2:] = config.device_queue
        self.queued[1:n + 1] = config.edge_queue

        R_lo, R_hi = sc.internal_rate_matrix_mbps()
        # transfer seconds per KB, interval [1/R_hi, 1/R_lo]; 0 on the diagonal, inf if unreachable
        with np.errstate(divide="ignore"):
            self.spk_lo = np.where(np.isinf(R_hi), 0.0, KB_TO_MBIT / R_hi)
            self.spk_hi = np.where(np.isinf(R_lo), 0.0, KB_TO_MBIT / R_lo)
        self.allowed_mask = sc.allowed_mask(self.owner)            # (K, n+2)
        # device -> cloud traffic is relayed by the device's nearest in-range edge; for
        # energy the wireless leg costs transmit power and the wired leg idle power
        # (Jin et al. 2026, eq. 11). Internal node of that edge per task (0 = none).
        self.near_node = sc.dev_nearest[self.owner].astype(np.int64)

        raw = sc.raw
        # per-task device parameters (the task's own device)
        self.e_lo_t, self.e_hi_t = sc.dev_e[self.owner, 0], sc.dev_e[self.owner, 1]
        self.p_tx_t, self.p_idle_t = sc.dev_ptx[self.owner], sc.dev_pidle[self.owner]
        self.kappa_E = float(raw["edge_servers"]["kappa_j_per_gcycle_ghz2"])
        self.P_static = np.zeros(NI)
        self.P_static[1:n + 1] = np.asarray(raw["edge_servers"]["static_power_w"], dtype=float)
        self.kappa_C = float(raw["cloud"]["kappa_j_per_gcycle_ghz2"])
        self.P_static[self.cloud] = float(raw["cloud"].get("static_power_w", 0.0))
        self.eta_E = float(raw["pricing"]["edge_usd_per_ghz_s"])
        self.eta_C = float(raw["pricing"]["cloud_usd_per_s"])

    # ------------------------------------------------------------------
    def check(self, X: np.ndarray, O: np.ndarray) -> None:
        K = self.K
        if X.shape != (K,) or O.shape != (K,):
            raise InvalidIndividual(f"X and O must have shape ({K},)")
        if X.min() < 0 or X.max() >= self.N:
            raise InvalidIndividual(f"allocation genes must be in 0..{self.N - 1}")
        bad = np.flatnonzero(~self.allowed_mask[np.arange(K), X])
        if bad.size:
            raise InvalidIndividual(f"tasks {bad[:5].tolist()} use node(s) {np.unique(X[bad]).tolist()} "
                                    f"their device cannot reach")
        pos = np.full(K, -1, dtype=np.int64)
        pos[O] = np.arange(K)
        if (pos < 0).any():
            raise InvalidIndividual("O is not a permutation of the tasks")
        viol = np.flatnonzero(pos[self.src] >= pos[self.dst])
        if viol.size:
            e = viol[0]
            raise InvalidIndividual(f"O is not topological: task {self.dst[e]} is placed before its "
                                    f"parent {self.src[e]}")

    def _upload_energy(self, sender, dest_node, kb, spk, mask) -> float:
        """Device energy to send kb from each sender task's device to dest_node.
        To an edge: transmit power x wireless time. To the cloud: transmit power x the
        wireless leg to the nearest edge + idle power x the wired leg (Jin et al. eq. 11)."""
        if not np.any(mask):
            return 0.0
        s, dst, kb = sender[mask], dest_node[mask], kb[mask]
        dev = self.dev_node[s]
        near = self.near_node[s]
        to_cloud = (dst == self.cloud) & (near > 0)
        wireless_node = np.where(to_cloud, near, dst)
        t_wireless = _xfer(kb, spk[dev, wireless_node])
        t_wired = np.where(to_cloud, _xfer(kb, spk[near, self.cloud]), 0.0)
        return float((self.p_tx_t[s] * t_wireless + self.p_idle_t[s] * t_wired).sum())

    def internal(self, X) -> np.ndarray:
        """Internal node of each task: its own device for gene 0, else the edge / cloud node."""
        X = np.asarray(X, dtype=np.int64)
        return np.where(X == 0, self.dev_node, X)

    def evaluate(self, X, O, detail: bool = False) -> EvalResult:
        X = np.asarray(X, dtype=np.int64)
        O = np.asarray(O, dtype=np.int64)
        if self.cfg.validate:
            self.check(X, O)
        n, cloud = self.n, self.cloud
        I = self.internal(X)

        # effective capacity and compute time
        share = np.where(self.is_edge, np.maximum(np.bincount(I, minlength=self.NI), 1), 1)
        feff_lo = (self.f_lo / share)[I]
        feff_hi = (self.f_hi / share)[I]
        tc_lo = self.C_lo / feff_hi
        tc_hi = self.C_hi / feff_lo

        # inter-node transfers
        xs, xd = I[self.src], I[self.dst]
        tt_lo = _xfer(self.D_lo, self.spk_lo[xs, xd])
        tt_hi = _xfer(self.D_hi, self.spk_hi[xs, xd])
        feasible = bool(np.all(np.isfinite(tt_hi)))

        offloaded = X != 0
        ready_lo = np.zeros(self.K)
        ready_hi = np.zeros(self.K)
        ret_lo = ret_hi = None
        if self.cfg.external_io == "device":
            # the diagonal is 0 s/KB, so local tasks get 0 automatically
            ready_lo = _xfer(self.ext_in, self.spk_lo[self.dev_node, I])
            ready_hi = _xfer(self.ext_in, self.spk_hi[self.dev_node, I])
            ret_lo = _xfer(self.ext_out, self.spk_lo[I, self.dev_node])
            ret_hi = _xfer(self.ext_out, self.spk_hi[I, self.dev_node])
            feasible &= bool(np.all(np.isfinite(ready_hi)) and np.all(np.isfinite(ret_hi)))

        ST_lo, ST_hi, FT_lo, FT_hi = _schedule(
            O, I, self.pred_ptr, self.pred_edge, self.src, tt_lo, tt_hi, tc_lo, tc_hi,
            ready_lo, ready_hi, self.queued, self.NI)

        # F1: makespan (latest finish incl. returning results), overall or summed per application
        end_lo = FT_lo + ret_lo if ret_lo is not None else FT_lo
        end_hi = FT_hi + ret_hi if ret_hi is not None else FT_hi
        if self.cfg.makespan == "sum_apps":
            app_lo = np.zeros(self.n_apps)
            app_hi = np.zeros(self.n_apps)
            np.maximum.at(app_lo, self.app, end_lo)
            np.maximum.at(app_hi, self.app, end_hi)
            mk_lo, mk_hi = float(app_lo.sum()), float(app_hi.sum())
        else:
            mk_lo, mk_hi = float(end_lo.max()), float(end_hi.max())

        # F2: energy (each task's own device parameters)
        local = ~offloaded
        e_comp = ((self.C_lo * self.e_lo_t)[local].sum(), (self.C_hi * self.e_hi_t)[local].sum())
        from_dev = (X[self.src] == 0) & offloaded[self.dst]        # sent from the parent's device
        e_tx_lo = self._upload_energy(self.src, I[self.dst], self.D_lo, self.spk_lo, from_dev)
        e_tx_hi = self._upload_energy(self.src, I[self.dst], self.D_hi, self.spk_hi, from_dev)
        if self.cfg.external_io == "device":
            tasks = np.arange(self.K)
            e_tx_lo += self._upload_energy(tasks, I, self.ext_in, self.spk_lo, offloaded)
            e_tx_hi += self._upload_energy(tasks, I, self.ext_in, self.spk_hi, offloaded)
        e_tx = (float(e_tx_lo), float(e_tx_hi))
        e_idle = ((self.p_idle_t * tc_lo)[offloaded].sum(), (self.p_idle_t * tc_hi)[offloaded].sum())

        work_lo = np.bincount(I, weights=self.C_lo, minlength=self.NI)
        work_hi = np.bincount(I, weights=self.C_hi, minlength=self.NI)
        busy_lo = np.zeros(self.NI)
        busy_hi = np.zeros(self.NI)
        np.maximum.at(busy_lo, I, FT_lo)
        np.maximum.at(busy_hi, I, FT_hi)
        e_sl = slice(1, n + 1)
        e_edge = (float((self.kappa_E * self.f_lo[e_sl] ** 2 * work_lo[e_sl]
                         + self.P_static[e_sl] * busy_lo[e_sl]).sum()),
                  float((self.kappa_E * self.f_hi[e_sl] ** 2 * work_hi[e_sl]
                         + self.P_static[e_sl] * busy_hi[e_sl]).sum()))
        e_cloud = (self.kappa_C * self.f_lo[cloud] ** 2 * work_lo[cloud] + self.P_static[cloud] * busy_lo[cloud],
                   self.kappa_C * self.f_hi[cloud] ** 2 * work_hi[cloud] + self.P_static[cloud] * busy_hi[cloud])
        if self.cfg.energy_scope == "device":
            e_edge = e_cloud = (0.0, 0.0)
        energy = [e_comp[b] + e_tx[b] + e_idle[b] + e_edge[b] + e_cloud[b] for b in (0, 1)]

        # F3: monetary cost
        on_edge = self.is_edge[I]
        on_cloud = I == cloud
        c_edge = (self.eta_E * (feff_lo * tc_lo)[on_edge].sum(), self.eta_E * (feff_hi * tc_hi)[on_edge].sum())
        c_cloud = (self.eta_C * tc_lo[on_cloud].sum(), self.eta_C * tc_hi[on_cloud].sum())
        cost = [c_edge[b] + c_cloud[b] for b in (0, 1)]

        F_lo = np.array([mk_lo, energy[0], cost[0]], dtype=float)
        F_hi = np.array([mk_hi, energy[1], cost[1]], dtype=float)
        res = EvalResult(F_lo=F_lo, F_hi=F_hi, feasible=feasible)
        if detail:
            res.breakdown = {
                "energy_local_compute": e_comp, "energy_upload": e_tx, "energy_idle": e_idle,
                "energy_edge": e_edge, "energy_cloud": e_cloud,
                "cost_edge": c_edge, "cost_cloud": c_cloud,
            }
            res.ST, res.FT = (ST_lo, ST_hi), (FT_lo, FT_hi)
        return res

    def evaluate_population(self, Xs, Os, fast: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """(P, 3) lower / upper objectives. fast=True uses the compiled loop (needs numba);
        fast=False calls evaluate() per individual (the reference implementation)."""
        Xs = np.ascontiguousarray(Xs, dtype=np.int64)
        Os = np.ascontiguousarray(Os, dtype=np.int64)
        if fast and HAVE_NUMBA and Xs.ndim == 2 and Xs.shape[1] == self.K:
            c = self.cfg
            F_lo, F_hi, _, err = _evaluate_population_nb(
                Xs, Os, c.validate, self.allowed_mask, self.dev_node, self.near_node, self.cloud, self.NI,
                self.is_edge, self.queued, self.f_lo, self.f_hi, self.C_lo, self.C_hi, self.src, self.dst,
                self.D_lo, self.D_hi, self.pred_ptr, self.pred_edge, self.spk_lo, self.spk_hi,
                c.external_io == "device", self.ext_in, self.ext_out, c.makespan == "sum_apps",
                self.app.astype(np.int64), max(self.n_apps, 1), self.e_lo_t, self.e_hi_t,
                self.p_tx_t.astype(float), self.p_idle_t.astype(float), self.kappa_E, self.kappa_C,
                self.P_static, self.eta_E, self.eta_C, c.energy_scope == "device")
            if err.any():
                i = int(np.flatnonzero(err)[0])
                self.check(Xs[i], Os[i])                  # raises with the reason
            return F_lo, F_hi
        F_lo = np.empty((len(Xs), 3))
        F_hi = np.empty((len(Xs), 3))
        for i in range(len(Xs)):
            r = self.evaluate(Xs[i], Os[i])
            F_lo[i], F_hi[i] = r.F_lo, r.F_hi
        return F_lo, F_hi
