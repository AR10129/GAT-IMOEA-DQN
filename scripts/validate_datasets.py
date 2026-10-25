"""Validate every workflow in datasets/workflows and the scenario files.

    python scripts/validate_datasets.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_tools.loader import load_scenario, load_workflow  # noqa: E402
from dataset_tools.validate import validate_record  # noqa: E402


def main() -> int:
    bad = 0
    files = sorted((ROOT / "datasets" / "workflows").glob("*/*.json"))
    for path in files:
        rec = json.loads(path.read_text(encoding="utf-8"))
        errors, warnings = validate_record(rec)
        wf = load_workflow(path)
        x, e = wf.gat_node_features(), wf.gat_edge_features()
        order = wf.random_topological_order(np.random.default_rng(0))
        pos = np.empty_like(order)
        pos[order] = np.arange(len(order))
        if not np.all(pos[wf.edge_index[0]] < pos[wf.edge_index[1]]):
            errors.append("random topological order violates an edge")
        if x.shape != (wf.num_tasks, 5) or e.shape != (wf.num_edges, 2) or not np.isfinite(x).all():
            errors.append(f"bad GAT feature shapes {x.shape} {e.shape}")
        status = "OK  " if not errors else "FAIL"
        bad += bool(errors)
        print(f"{status} {path.relative_to(ROOT)}  {errors or ''}{'  note: ' + '; '.join(warnings) if warnings else ''}")

    for sc_path in sorted((ROOT / "datasets" / "scenarios").glob("*.yaml")):
        try:
            sc = load_scenario(sc_path)
            lo, hi = sc.rate_matrix_mbps()
            assert np.all(lo <= hi), "rate interval with lower > upper"
            print(f"OK   {sc_path.relative_to(ROOT)}  targets={sc.allowed_targets().tolist()} "
                  f"uplink_mbps={np.round(sc.uplink_mbps, 1).tolist()}")
        except Exception as exc:
            bad += 1
            print(f"FAIL {sc_path.relative_to(ROOT)}  {exc}")
    print(f"\n{len(files)} workflows checked, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
