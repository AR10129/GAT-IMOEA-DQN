"""Suite A: Jin-scale multi-device instances with small application DAGs.

    python scripts/generate_suite_a.py          # writes datasets/suite_a/ and datasets/scenarios/suite_a/

Configurations follow Jin et al. (2026), Section 5.1:
    small   3 edge servers, 100 tasks, 20 / 50 mobile devices
    basic   6 edge servers, 300 tasks, 60 / 100 mobile devices
    medium 10 edge servers, 500 tasks, 100 / 250 mobile devices

Each device submits ONE application: a layered random DAG (daggen-style). The
total task count is split across devices at random (every device gets >= 1
task). Task sizes follow Jin et al. Table 1: workload C in {1, 2, 3, 4, 5}
Gcycles; entry tasks upload their input data S ~ U[500, 3000] KB from the
device; each DAG edge carries U[500, 3000] KB. Tasks are crisp (zero-width
intervals): as in Jin et al., uncertainty comes from the resources
(bandwidth, CPU), which the scenario gives as intervals.

DAG shape (per application with K tasks): layers of width 1..ceil(fat * sqrt(K));
every task outside the first layer gets one random parent in the previous
layer, plus each other task of the previous layer becomes a parent with
probability `density`. With K = 1 the application is a single task, as in
Jin et al.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

CONFIGS = {
    "small_20": (3, 100, 20), "small_50": (3, 100, 50),
    "basic_60": (6, 300, 60), "basic_100": (6, 300, 100),
    "medium_100": (10, 500, 100), "medium_250": (10, 500, 250),
}
JIN_DEVICE_CPU = [[0.45, 0.59], [0.54, 0.71], [0.63, 0.82], [0.72, 0.94], [0.81, 1.06], [0.9, 1.18]]
JIN_EDGE_CPU = [[78.5, 90.0], [102.0, 120.0], [127.5, 150.0]]
JIN_TX_POWER = [0.1, 0.2, 0.4, 0.5]
AREA_M = 1000.0


def layered_dag(K: int, rng: np.random.Generator, fat: float = 1.0, density: float = 0.3):
    """Edges (parent, child) of a layered random DAG on tasks 0..K-1, numbered layer by layer."""
    layers, start = [], 0
    max_w = max(1, math.ceil(fat * math.sqrt(K)))
    while start < K:
        w = int(min(rng.integers(1, max_w + 1), K - start))
        layers.append(list(range(start, start + w)))
        start += w
    edges = set()
    for prev, cur in zip(layers, layers[1:]):
        for c in cur:
            edges.add((int(rng.choice(prev)), c))
            for p in prev:
                if rng.random() < density:
                    edges.add((p, c))
    depth = np.zeros(K, dtype=np.int64)
    for li, layer in enumerate(layers):
        depth[layer] = li
    return sorted(edges), depth


def split_tasks(total: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Random positive integer sizes summing to total (one per device)."""
    if total < m:
        raise ValueError("need at least one task per device")
    return 1 + rng.multinomial(total - m, np.full(m, 1.0 / m))


def scenario(n_edge: int, m: int, rng: np.random.Generator, name: str) -> Dict:
    g = math.ceil(math.sqrt(n_edge))
    cells = [((i + 0.5) * AREA_M / g, (j + 0.5) * AREA_M / g) for i in range(g) for j in range(g)]
    edges = [list(map(float, c)) for c in cells[:n_edge]]
    radius = 0.75 * AREA_M / g
    E = np.array(edges)
    devices = []
    while len(devices) < m:                                     # every device reaches >= 1 edge
        p = rng.uniform(0, AREA_M, size=2)
        if np.min(np.linalg.norm(E - p, axis=1)) <= radius:
            devices.append([round(float(p[0]), 1), round(float(p[1]), 1)])
    return {
        "name": name,
        "description": f"Suite A, Jin et al. scale: {n_edge} edge servers, {m} mobile devices",
        "seed": 0,
        "mobile_devices": {
            "count": m, "origin_device": 0,
            "cpu_ghz": [JIN_DEVICE_CPU[d % 6] for d in range(m)],
            "energy_per_gcycle_j": [[0.9, 1.1]] * m,
            "tx_power_w": [JIN_TX_POWER[d % 4] for d in range(m)],
            "idle_power_w": [0.02] * m,
            "positions_m": devices,
        },
        "edge_servers": {
            "count": n_edge, "cpu_ghz": [JIN_EDGE_CPU[u % 3] for u in range(n_edge)],
            "kappa_j_per_gcycle_ghz2": 0.00005, "static_power_w": [15.0] * n_edge,
            "positions_m": edges, "coverage_radius_m": round(radius, 1),
        },
        "cloud": {"cpu_ghz": [18.0, 22.0], "kappa_j_per_gcycle_ghz2": 0.002, "static_power_w": 0.0},
        "links": {
            "wireless_uplink": {"bandwidth_mhz": [9.0, 12.0], "noise_power_w": 1.0e-13, "channel_gain": 0.001,
                                "path_loss_exponent": 4.0},
            "wireless_downlink_mbps": "uplink", "backhaul_mbps": [408.0, 624.0],
            "edge_to_edge_mbps": [408.0, 624.0], "device_cloud_route": "via_nearest_edge",
        },
        "pricing": {"edge_usd_per_ghz_s": 0.001, "cloud_usd_per_s": 0.01},
    }


def instance(config: str, seed: int, scenario_name: str) -> Dict:
    n_edge, total, m = CONFIGS[config]
    rng = np.random.default_rng(seed)
    sizes = split_tasks(total, m, rng)
    tasks, edges, offset = [], [], 0
    for d, K in enumerate(sizes):
        es, depth = layered_dag(int(K), rng)
        has_parent = np.zeros(K, dtype=bool)
        for p, c in es:
            has_parent[c] = True
            kb = float(rng.uniform(500, 3000))
            edges.append({"src": offset + p, "dst": offset + c, "data_kb": [round(kb, 3), round(kb, 3)]})
        for k in range(K):
            c = float(rng.integers(1, 6))
            inp = 0.0 if has_parent[k] else round(float(rng.uniform(500, 3000)), 3)
            tasks.append({"id": offset + k, "name": f"d{d}_t{k}", "type": "task", "workload_gcycles": [c, c],
                          "ext_input_kb": inp, "ext_output_kb": 0.0, "depth": int(depth[k]),
                          "device": d, "app": d})
        offset += int(K)
    edges.sort(key=lambda e: (e["src"], e["dst"]))
    indeg = np.zeros(offset, dtype=int)
    outdeg = np.zeros(offset, dtype=int)
    for e in edges:
        indeg[e["dst"]] += 1
        outdeg[e["src"]] += 1
    name = f"{config}_i{seed % 1000}"
    return {
        "schema_version": "1.0", "name": name, "family": "suite_a", "scenario": scenario_name,
        "source": {"generator": "layered-random-dag", "seed": seed, "config": config},
        "units": {"workload": "Gcycles", "data": "KB (1 KB = 1000 bytes)"},
        "num_tasks": offset, "num_edges": len(edges), "num_devices": m, "num_apps": m,
        "entry_tasks": [int(i) for i in np.flatnonzero(indeg == 0)],
        "exit_tasks": [int(i) for i in np.flatnonzero(outdeg == 0)],
        "stats": {"components": m, "tasks_per_device": [int(s) for s in sizes]},
        "tasks": tasks, "edges": edges,
    }


def write_suite(root: Path, instances_per_config: int = 4, base_seed: int = 2026) -> Dict[str, Dict]:
    """i0 = the test instance (as in Jin et al., one per configuration); i1.. = training/validation."""
    (root / "suite_a").mkdir(parents=True, exist_ok=True)
    (root / "scenarios" / "suite_a").mkdir(parents=True, exist_ok=True)
    manifest = {}
    for ci, (config, (n_edge, total, m)) in enumerate(CONFIGS.items()):
        sc_name = f"suite_a/{config}"
        sc = scenario(n_edge, m, np.random.default_rng(base_seed + ci), config)
        (root / "scenarios" / f"{sc_name}.yaml").write_text(
            "# generated by scripts/generate_suite_a.py; values from Jin et al. (2026) Table 1\n"
            + yaml.safe_dump(sc, sort_keys=False, default_flow_style=None), encoding="utf-8")
        for i in range(instances_per_config):
            rec = instance(config, base_seed * 1000 + ci * 100 + i, sc_name)
            rec["name"] = f"{config}_i{i}"
            path = root / "suite_a" / f"{rec['name']}.json"
            path.write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
            manifest[rec["name"]] = {"path": f"suite_a/{rec['name']}.json", "scenario": sc_name,
                                     "config": config, "role": "test" if i == 0 else "train",
                                     "num_tasks": rec["num_tasks"], "num_edges": rec["num_edges"],
                                     "num_devices": m, "edge_servers": n_edge}
    (root / "suite_a" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
