"""RawWorkflow -> interval workflow record (dataset schema v1).

Conversions (also recorded in each file under "conversion"):
  workload  C_k  [Gcycles] = runtime_ref_s * reference_ghz, widened to
            [C(1 - d), C(1 + d)] with d ~ U[width_min, width_max] per task.
  data      D_ij [KB]      = bytes carried on edge i->j / 1000, widened the same
            way with an independent d per edge.
  ext. I/O  [KB]           = files a task reads that no parent sends (inputs that
            must be uploaded from the originating device) and files it writes that
            no child reads (results returned to the device).
Tasks are renumbered in a topological order, so every edge has src < dst.
"""
from __future__ import annotations

import json
import re
from collections import deque
from typing import Any, Dict, List

import numpy as np

from .raw import RawWorkflow

SCHEMA_VERSION = "1.0"
MIN_RUNTIME_S = 1e-3


def _topo_order(n: int, edges) -> List[int]:
    indeg = [0] * n
    succ: List[List[int]] = [[] for _ in range(n)]
    for p, c in edges:
        succ[p].append(c)
        indeg[c] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(succ[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != n:
        raise ValueError(f"workflow has a cycle ({len(order)}/{n} tasks sortable)")
    return order


def _interval(mid: float, rng: np.random.Generator, wmin: float, wmax: float) -> List[float]:
    d = rng.uniform(wmin, wmax)
    return [round(mid * (1.0 - d), 6), round(mid * (1.0 + d), 6)]


def build_record(raw: RawWorkflow, name: str, seed: int, reference_ghz: float,
                 width_min: float, width_max: float) -> Dict[str, Any]:
    n = len(raw.tasks)
    order = _topo_order(n, raw.links.keys())
    new_id = {old: new for new, old in enumerate(order)}
    rng = np.random.default_rng(seed + 7919)

    edges = sorted((new_id[p], new_id[c], files) for (p, c), files in raw.links.items())
    in_files: Dict[int, set] = {i: set() for i in range(n)}
    out_files: Dict[int, set] = {i: set() for i in range(n)}
    for p, c, files in edges:
        out_files[p].update(files)
        in_files[c].update(files)

    indeg = [0] * n
    outdeg = [0] * n
    preds: List[List[int]] = [[] for _ in range(n)]
    for p, c, _ in edges:
        outdeg[p] += 1
        indeg[c] += 1
        preds[c].append(p)
    depth = [0] * n
    for v in range(n):                       # ids are topological
        if preds[v]:
            depth[v] = max(depth[p] for p in preds[v]) + 1

    tasks = []
    for new, old in enumerate(order):
        t = raw.tasks[old]
        rt = max(t.runtime_s, MIN_RUNTIME_S)
        ext_in = sum(s for f, s in t.inputs.items() if f not in in_files[new]) / 1000.0
        ext_out = sum(s for f, s in t.outputs.items() if f not in out_files[new]) / 1000.0
        tasks.append({
            "id": new,
            "name": t.name,
            "type": t.type,
            "runtime_ref_s": round(rt, 4),
            "workload_gcycles": _interval(rt * reference_ghz, rng, width_min, width_max),
            "ext_input_kb": round(ext_in, 3),
            "ext_output_kb": round(ext_out, 3),
            "depth": depth[new],
        })

    edge_rows = []
    for p, c, files in edges:
        kb = sum(files.values()) / 1000.0
        edge_rows.append({"src": p, "dst": c, "data_kb": _interval(kb, rng, width_min, width_max)})

    # weakly connected components
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p, c, _ in edges:
        parent[find(p)] = find(c)
    components = len({find(i) for i in range(n)})

    # critical path on reference runtimes, ignoring communication
    finish = [0.0] * n
    for v in range(n):
        start = max((finish[p] for p in preds[v]), default=0.0)
        finish[v] = start + tasks[v]["runtime_ref_s"]

    types: Dict[str, int] = {}
    for t in tasks:
        types[t["type"]] = types.get(t["type"], 0) + 1
    total_work = sum(sum(t["workload_gcycles"]) / 2 for t in tasks)
    total_data = sum(sum(e["data_kb"]) / 2 for e in edge_rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "family": raw.family,
        "source": {"generator": raw.generator, "seed": seed},
        "units": {"workload": "Gcycles", "data": "KB (1 KB = 1000 bytes)",
                  "runtime": "s on the generator's reference core"},
        "conversion": {"reference_ghz": reference_ghz,
                       "interval_half_width": [width_min, width_max]},
        "num_tasks": n,
        "num_edges": len(edge_rows),
        "entry_tasks": [i for i in range(n) if indeg[i] == 0],
        "exit_tasks": [i for i in range(n) if outdeg[i] == 0],
        "stats": {
            "depth": max(depth) + 1 if n else 0,
            "max_width": int(np.bincount(depth).max()) if n else 0,
            "components": components,
            "max_in_degree": max(indeg) if n else 0,
            "max_out_degree": max(outdeg) if n else 0,
            "total_workload_gcycles_mid": round(total_work, 3),
            "total_edge_data_kb_mid": round(total_data, 3),
            "data_to_compute_kb_per_gcycle": round(total_data / total_work, 4) if total_work else None,
            "critical_path_ref_s": round(max(finish), 3) if n else 0.0,
            "task_types": dict(sorted(types.items())),
        },
        "tasks": tasks,
        "edges": edge_rows,
    }


def dump_record(record: Dict[str, Any], path) -> None:
    """Readable JSON: one task / edge per line."""
    head = {k: v for k, v in record.items() if k not in ("tasks", "edges")}
    with open(path, "w", encoding="utf-8") as f:
        body = json.dumps(head, indent=2, default=int)[:-2]
        # put lists of plain numbers on one line
        body = re.sub(r"\[\s+([-\d.,\s]*?)\s+\]",
                      lambda mt: "[" + ", ".join(x.strip() for x in mt.group(1).split(",")) + "]", body)
        f.write(body + ",\n")
        for key in ("tasks", "edges"):
            rows = record[key]
            f.write(f'  "{key}": [\n')
            for i, r in enumerate(rows):
                f.write("    " + json.dumps(r, separators=(",", ":")) + ("," if i < len(rows) - 1 else "") + "\n")
            f.write("  ]" + ("," if key == "tasks" else "") + "\n")
        f.write("}\n")
