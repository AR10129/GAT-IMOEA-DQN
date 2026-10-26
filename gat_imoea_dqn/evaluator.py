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

    def evaluate_population(self, Xs, Os) -> Tuple[np.ndarray, np.ndarray]:
        Xs, Os = np.asarray(Xs), np.asarray(Os)
        F_lo = np.empty((len(Xs), 3))
        F_hi = np.empty((len(Xs), 3))
        for i in range(len(Xs)):
            r = self.evaluate(Xs[i], Os[i])
            F_lo[i], F_hi[i] = r.F_lo, r.F_hi
        return F_lo, F_hi
