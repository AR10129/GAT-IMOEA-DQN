"""Checks that a workflow record satisfies what GAT-IMOEA-DQN assumes."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _is_interval(x) -> bool:
    return (isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x)
            and 0 <= x[0] <= x[1])


def validate_record(rec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Errors make the record unusable."""
    errors: List[str] = []
    warnings: List[str] = []
    n = rec.get("num_tasks")
    tasks, edges = rec.get("tasks", []), rec.get("edges", [])
    if n != len(tasks):
        errors.append(f"num_tasks={n} but {len(tasks)} tasks listed")
    if rec.get("num_edges") != len(edges):
        errors.append(f"num_edges={rec.get('num_edges')} but {len(edges)} edges listed")
    for i, t in enumerate(tasks):
        if t.get("id") != i:
            errors.append(f"task {i} has id {t.get('id')} (ids must be 0..K-1 in order)")
            break
    bad_w = [t["id"] for t in tasks if not _is_interval(t.get("workload_gcycles"))]
    if bad_w:
        errors.append(f"{len(bad_w)} tasks with an invalid workload interval, e.g. {bad_w[:5]}")
    zero_w = [t["id"] for t in tasks if _is_interval(t.get("workload_gcycles")) and t["workload_gcycles"][1] <= 0]
    if zero_w:
        warnings.append(f"{len(zero_w)} tasks with zero workload")

    seen = set()
    indeg = [0] * len(tasks)
    outdeg = [0] * len(tasks)
    for e in edges:
        s, d = e.get("src"), e.get("dst")
        if not (isinstance(s, int) and isinstance(d, int) and 0 <= s < len(tasks) and 0 <= d < len(tasks)):
            errors.append(f"edge with bad endpoints {s}->{d}")
            continue
        if s >= d:   # ids are topological, so src < dst also proves acyclicity
            errors.append(f"edge {s}->{d} breaks topological numbering (self-loop or cycle)")
        if (s, d) in seen:
            errors.append(f"duplicate edge {s}->{d}")
        seen.add((s, d))
        if not _is_interval(e.get("data_kb")):
            errors.append(f"edge {s}->{d} has an invalid data interval")
        indeg[d] += 1
        outdeg[s] += 1
    zero_d = sum(1 for e in edges if _is_interval(e.get("data_kb")) and e["data_kb"][1] == 0)
    if zero_d:
        warnings.append(f"{zero_d} edges carry no data (precedence only)")

    entries = [i for i in range(len(tasks)) if indeg[i] == 0]
    exits = [i for i in range(len(tasks)) if outdeg[i] == 0]
    if rec.get("entry_tasks") != entries:
        errors.append("entry_tasks does not match the edge list")
    if rec.get("exit_tasks") != exits:
        errors.append("exit_tasks does not match the edge list")
    isolated = [i for i in range(len(tasks)) if indeg[i] == 0 and outdeg[i] == 0]
    if isolated:
        warnings.append(f"{len(isolated)} isolated tasks")
    comps = rec.get("stats", {}).get("components", 1)
    if comps > 1:
        warnings.append(f"{comps} weakly connected components")
    return errors, warnings
