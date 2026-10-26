"""Load workflow datasets and scenarios for GAT-IMOEA-DQN.

    from dataset_tools.loader import load_workflow, load_scenario
    wf = load_workflow("datasets/workflows/montage/montage_200.json")
    sc = load_scenario("datasets/scenarios/default.yaml")
    x, e = wf.gat_node_features(), wf.gat_edge_features()      # (K,5), (E,2)
    R_lo, R_hi = sc.rate_matrix_mbps()                         # (n+2, n+2)

Node indexing for allocation genes x_k (PDF section 2.1):
    0 = originating mobile device, 1..n = edge servers, n+1 = cloud.
Intervals are stored as two arrays (lower, upper).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

KB_TO_MBIT = 8.0 / 1000.0          # 1 KB = 1000 bytes = 0.008 Mbit


# ------------------------------------------------------------ workflow ---
@dataclass
class WorkflowDAG:
    name: str
    family: str
    C_lo: np.ndarray               # (K,) workload lower bound [Gcycles]
    C_hi: np.ndarray               # (K,) workload upper bound
    edge_index: np.ndarray         # (2, E) src, dst; src < dst always
    D_lo: np.ndarray               # (E,) data lower bound [KB]
    D_hi: np.ndarray               # (E,) data upper bound
    ext_input_kb: np.ndarray       # (K,) external input uploaded from the device
    ext_output_kb: np.ndarray      # (K,) final results returned to the device
    depth: np.ndarray              # (K,) topological depth, entry tasks = 0
    task_types: List[str]
    meta: Dict = field(default_factory=dict)
    # Multi-device instances (Suite A): the mobile device that submits each task and the
    # application (one DAG) it belongs to. None = every task belongs to the scenario's
    # origin device, as one application.
    owner: Optional[np.ndarray] = None
    app: Optional[np.ndarray] = None

    @property
    def num_tasks(self) -> int:
        return len(self.C_lo)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def in_degree(self) -> np.ndarray:
        return np.bincount(self.edge_index[1], minlength=self.num_tasks)

    @property
    def out_degree(self) -> np.ndarray:
        return np.bincount(self.edge_index[0], minlength=self.num_tasks)

    @property
    def entry_tasks(self) -> np.ndarray:
        return np.flatnonzero(self.in_degree == 0)

    @property
    def exit_tasks(self) -> np.ndarray:
        return np.flatnonzero(self.out_degree == 0)

    def predecessors(self) -> List[np.ndarray]:
        """preds[k] = parent ids of task k; parallel edge ids in pred_edges()."""
        order = np.argsort(self.edge_index[1], kind="stable")
        splits = np.cumsum(self.in_degree)[:-1]
        return np.split(self.edge_index[0][order], splits)

    def pred_edges(self) -> List[np.ndarray]:
        order = np.argsort(self.edge_index[1], kind="stable")
        return np.split(order, np.cumsum(self.in_degree)[:-1])

    def successors(self) -> List[np.ndarray]:
        order = np.argsort(self.edge_index[0], kind="stable")
        return np.split(self.edge_index[1][order], np.cumsum(self.out_degree)[:-1])

    # -- GAT inputs (PDF step 2) -------------------------------------------
    def gat_node_features(self, normalize: Optional[str] = "log") -> np.ndarray:
        """h_k^(0) = [m(C_k), w(C_k), in_deg, out_deg, depth]  -> (K, 5).

        normalize="log": log1p on the workload midpoint/width (they span several
        orders of magnitude); degrees and depth divided by their max.
        normalize=None: raw values exactly as in the PDF.
        """
        mid = (self.C_lo + self.C_hi) / 2.0
        wid = self.C_hi - self.C_lo
        feats = [mid, wid, self.in_degree.astype(float), self.out_degree.astype(float),
                 self.depth.astype(float)]
        if normalize == "log":
            feats[0], feats[1] = np.log1p(mid), np.log1p(wid)
            for i in (2, 3, 4):
                m = feats[i].max()
                feats[i] = feats[i] / m if m > 0 else feats[i]
        return np.stack(feats, axis=1).astype(np.float32)

    def gat_edge_features(self, normalize: Optional[str] = "log") -> np.ndarray:
        """e_ij = [m(D_ij), w(D_ij)]  -> (E, 2)."""
        mid = (self.D_lo + self.D_hi) / 2.0
        wid = self.D_hi - self.D_lo
        if normalize == "log":
            mid, wid = np.log1p(mid), np.log1p(wid)
        return np.stack([mid, wid], axis=1).astype(np.float32)

    def random_topological_order(self, rng: np.random.Generator) -> np.ndarray:
        """Kahn's algorithm with random tie-breaking (PDF step 1, priority string O)."""
        indeg = self.in_degree.copy()
        succ = self.successors()
        ready = list(np.flatnonzero(indeg == 0))
        order = []
        while ready:
            u = ready.pop(int(rng.integers(len(ready))))
            order.append(u)
            for v in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
        return np.asarray(order, dtype=np.int64)


def load_workflow(path) -> WorkflowDAG:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    t, e = rec["tasks"], rec["edges"]
    W = np.array([x["workload_gcycles"] for x in t], dtype=float).reshape(-1, 2)
    D = np.array([x["data_kb"] for x in e], dtype=float).reshape(-1, 2)
    ei = np.array([[x["src"] for x in e], [x["dst"] for x in e]], dtype=np.int64).reshape(2, -1)
    meta = {k: v for k, v in rec.items() if k not in ("tasks", "edges")}
    has_owner = bool(t) and "device" in t[0]
    return WorkflowDAG(
        name=rec["name"], family=rec["family"],
        C_lo=W[:, 0], C_hi=W[:, 1], edge_index=ei, D_lo=D[:, 0], D_hi=D[:, 1],
        ext_input_kb=np.array([x["ext_input_kb"] for x in t], dtype=float),
        ext_output_kb=np.array([x["ext_output_kb"] for x in t], dtype=float),
        depth=np.array([x["depth"] for x in t], dtype=np.int64),
        task_types=[x["type"] for x in t], meta=meta,
        owner=np.array([x["device"] for x in t], dtype=np.int64) if has_owner else None,
        app=np.array([x["app"] for x in t], dtype=np.int64) if has_owner else None)


def load_manifest(root="datasets") -> Dict[str, Dict]:
    return json.loads((Path(root) / "manifest.json").read_text())["workflows"]


# ------------------------------------------------------------ scenario ---
def _iv(x) -> Tuple[float, float]:
    lo, hi = (float(x[0]), float(x[1])) if isinstance(x, (list, tuple)) else (float(x), float(x))
    if lo > hi:
        raise ValueError(f"interval lower bound > upper bound: {x}")
    return lo, hi


@dataclass
class Scenario:
    raw: Dict
    n_edge: int
    origin: int
    f_dev: Tuple[float, float]            # origin device CPU [GHz]
    e_dev: Tuple[float, float]            # origin device energy [J / Gcycle]
    p_tx: float
    p_idle: float
    f_edge: np.ndarray                    # (n, 2)
    f_cloud: Tuple[float, float]
    edge_in_range: np.ndarray             # (n,) bool, for the origin device
    uplink_mbps: np.ndarray               # (n, 2) origin device -> edge u
    nearest_edge: int                     # 1-based edge id used to reach the cloud
    # every device (Suite A: many devices each submit tasks); index d = 0..m-1
    dev_f: np.ndarray = None              # (m, 2) CPU [GHz]
    dev_e: np.ndarray = None              # (m, 2) energy per Gcycle [J]
    dev_ptx: np.ndarray = None            # (m,) transmit power [W]
    dev_pidle: np.ndarray = None          # (m,) idle power [W]
    dev_in_range: np.ndarray = None       # (m, n) bool
    dev_uplink: np.ndarray = None         # (m, n, 2) device d -> edge u [Mbps]
    dev_nearest: np.ndarray = None        # (m,) 1-based nearest in-range edge, 0 = none

    @property
    def num_nodes(self) -> int:
        return self.n_edge + 2

    @property
    def num_devices(self) -> int:
        return len(self.dev_f)

    def allowed_targets(self, device: Optional[int] = None) -> np.ndarray:
        """Allocation values x_k a device may use (communication range); default the origin."""
        d = self.origin if device is None else device
        in_range, nearest = self.dev_in_range[d], self.dev_nearest[d]
        targets = [0] + [u + 1 for u in range(self.n_edge) if in_range[u]]
        if nearest > 0:
            targets.append(self.n_edge + 1)
        return np.asarray(targets, dtype=np.int64)

    def allowed_mask(self, owner: np.ndarray) -> np.ndarray:
        """(K, n+2) bool: allocation values each task may take, given its owner device."""
        per_dev = np.zeros((self.num_devices, self.n_edge + 2), dtype=bool)
        for d in range(self.num_devices):
            per_dev[d, self.allowed_targets(d)] = True
        return per_dev[np.asarray(owner, dtype=np.int64)]

    def internal_rate_matrix_mbps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Rates between internal nodes: 1..n edges, n+1 cloud, n+2+d device d (index 0 unused).

        Edge/cloud links as in rate_matrix_mbps(); each device row/column is that
        device's own wireless links. Device <-> another device is unreachable (0).
        """
        n, m = self.n_edge, self.num_devices
        N = n + 2 + m
        lo, hi = np.zeros((N, N)), np.zeros((N, N))
        base_lo, base_hi = self.rate_matrix_mbps(self.origin)
        lo[1:n + 2, 1:n + 2], hi[1:n + 2, 1:n + 2] = base_lo[1:, 1:], base_hi[1:, 1:]
        for d in range(m):
            r_lo, r_hi = self.rate_matrix_mbps(d)
            i = n + 2 + d
            lo[i, 1:n + 2], hi[i, 1:n + 2] = r_lo[0, 1:], r_hi[0, 1:]
            lo[1:n + 2, i], hi[1:n + 2, i] = r_lo[1:, 0], r_hi[1:, 0]
            lo[i, i] = hi[i, i] = np.inf
        return lo, hi

    def rate_matrix_mbps(self, device: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Interval rate R~[a, b] between nodes a, b (0=device, 1..n edge, n+1 cloud),
        for one device (default the origin).

        Diagonal is +inf (co-located, no transfer). Unreachable pairs are 0.
        Device<->cloud is relayed via the nearest in-range edge: its rate is the
        series combination 1 / (1/r_wireless + 1/W_C).
        """
        d = self.origin if device is None else device
        in_range, uplink, nearest_edge = self.dev_in_range[d], self.dev_uplink[d], int(self.dev_nearest[d])
        L = self.raw["links"]
        n, c = self.n_edge, self.n_edge + 1
        lo = np.zeros((n + 2, n + 2))
        hi = np.zeros((n + 2, n + 2))
        np.fill_diagonal(lo, np.inf)
        np.fill_diagonal(hi, np.inf)
        # "uplink" = the edge -> device downlink runs at that device-edge uplink rate
        down_spec = L["wireless_downlink_mbps"]
        same_as_up = isinstance(down_spec, str)
        if same_as_up and down_spec != "uplink":
            raise ValueError(f"wireless_downlink_mbps must be an interval or 'uplink', got {down_spec!r}")
        down_fixed = None if same_as_up else _iv(down_spec)
        bh = _iv(L["backhaul_mbps"])
        e2e = _iv(L["edge_to_edge_mbps"])

        def down_rate(u: int) -> Tuple[float, float]:
            return tuple(uplink[u - 1]) if same_as_up else down_fixed

        for u in range(1, n + 1):
            if in_range[u - 1]:
                lo[0, u], hi[0, u] = uplink[u - 1]
                lo[u, 0], hi[u, 0] = down_rate(u)
            lo[u, c], hi[u, c] = bh
            lo[c, u], hi[c, u] = bh
            for v in range(1, n + 1):
                if v != u:
                    lo[u, v], hi[u, v] = e2e
        if nearest_edge > 0:
            up = uplink[nearest_edge - 1]
            series = lambda a, b: 1.0 / (1.0 / a + 1.0 / b)  # noqa: E731
            lo[0, c], hi[0, c] = series(up[0], bh[0]), series(up[1], bh[1])
            down = down_rate(nearest_edge)
            lo[c, 0], hi[c, 0] = series(down[0], bh[0]), series(down[1], bh[1])
        return lo, hi

    @staticmethod
    def transfer_time_s(data_lo, data_hi, rate_lo, rate_hi):
        """Interval D~ / R~ in seconds: [D_lo / R_hi, D_hi / R_lo]."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = np.where(np.isinf(rate_hi), 0.0, data_lo * KB_TO_MBIT / rate_hi)
            t_hi = np.where(np.isinf(rate_lo), 0.0, data_hi * KB_TO_MBIT / rate_lo)
        return t_lo, t_hi


def load_scenario(path, origin_device: Optional[int] = None) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    md, es, cl, L = raw["mobile_devices"], raw["edge_servers"], raw["cloud"], raw["links"]
    i = md["origin_device"] if origin_device is None else origin_device
    m, n = md["count"], es["count"]
    for key in ("cpu_ghz", "energy_per_gcycle_j", "tx_power_w", "idle_power_w", "positions_m"):
        if len(md[key]) != m:
            raise ValueError(f"mobile_devices.{key} has {len(md[key])} entries, expected {m}")
    for key in ("cpu_ghz", "static_power_w", "positions_m"):
        if len(es[key]) != n:
            raise ValueError(f"edge_servers.{key} has {len(es[key])} entries, expected {n}")

    edge_pos = np.asarray(es["positions_m"], dtype=float)
    up = L["wireless_uplink"]
    B = _iv(up["bandwidth_mhz"])

    def links(d: int):
        dist = np.linalg.norm(edge_pos - np.asarray(md["positions_m"][d], dtype=float), axis=1)
        in_range = dist <= float(es["coverage_radius_m"])
        if up.get("channel_gain") is not None:        # fixed gain g_{i,u}, as in Jin et al.
            gain = np.full(n, float(up["channel_gain"]))
        else:                                         # distance path loss d^-alpha
            gain = np.maximum(dist, 1.0) ** (-float(up["path_loss_exponent"]))
        se = np.log2(1.0 + float(md["tx_power_w"][d]) * gain / float(up["noise_power_w"]))
        uplink = np.stack([B[0] * se, B[1] * se], axis=1)      # MHz * bit/s/Hz = Mbps
        nearest = int(np.argmin(np.where(in_range, dist, np.inf))) + 1 if in_range.any() else 0
        return in_range, uplink, nearest

    per = [links(d) for d in range(m)]
    in_range, uplink, nearest = per[i]
    return Scenario(
        raw=raw, n_edge=n, origin=i,
        f_dev=_iv(md["cpu_ghz"][i]), e_dev=_iv(md["energy_per_gcycle_j"][i]),
        p_tx=float(md["tx_power_w"][i]), p_idle=float(md["idle_power_w"][i]),
        f_edge=np.array([_iv(x) for x in es["cpu_ghz"]]), f_cloud=_iv(cl["cpu_ghz"]),
        edge_in_range=in_range, uplink_mbps=uplink, nearest_edge=nearest,
        dev_f=np.array([_iv(x) for x in md["cpu_ghz"]]),
        dev_e=np.array([_iv(x) for x in md["energy_per_gcycle_j"]]),
        dev_ptx=np.asarray(md["tx_power_w"], dtype=float), dev_pidle=np.asarray(md["idle_power_w"], dtype=float),
        dev_in_range=np.stack([p[0] for p in per]), dev_uplink=np.stack([p[1] for p in per]),
        dev_nearest=np.asarray([p[2] for p in per], dtype=np.int64))
