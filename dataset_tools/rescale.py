"""Map a workflow's workloads and data sizes into a target range at load time.

    wf_j = rescale_workflow(load_workflow(path))          # Jin et al.: 1-5 Gcycles, 500-3000 KB
    wf_x = rescale_workflow(wf, data_multiplier=10)       # 10x the data, for CCR sweeps

Why: the generated workflows keep the generators' scientific-cluster sizes
(hundreds of Gcycles, MB-GB files). Jin et al. (2026) use 1-5 Gcycles per task
and 500-3000 KB per task; mapping into those ranges makes the two papers'
settings comparable.

How: midpoints are mapped by an order-preserving log-linear map,
    m' = lo + (ln m - ln m_min) / (ln m_max - ln m_min) * (hi - lo),
because raw sizes span several orders of magnitude (a linear map would squeeze
all but the largest tasks onto the lower bound). Each interval keeps its
relative half-width d = (hi - lo) / (hi + lo). Edges that carry no data stay
at zero (precedence only). Files no task produces/consumes (ext_input_kb,
ext_output_kb) are mapped into the same data range. `data_multiplier` then
scales every data size (edges and external files) to raise or lower the
communication-to-computation ratio.
"""
from __future__ import annotations

import copy
from typing import Tuple

import numpy as np

from .loader import WorkflowDAG

JIN_WORKLOAD_GCYCLES = (1.0, 5.0)
JIN_DATA_KB = (500.0, 3000.0)


def _log_map(mid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Order-preserving log-linear map of positive values into [lo, hi]; zeros stay zero."""
    out = np.zeros_like(mid, dtype=float)
    pos = mid > 0
    if not pos.any():
        return out
    lm = np.log(mid[pos])
    span = lm.max() - lm.min()
    out[pos] = (lo + hi) / 2 if span <= 0 else lo + (lm - lm.min()) / span * (hi - lo)
    return out


def _rescale_interval(i_lo: np.ndarray, i_hi: np.ndarray, rng: Tuple[float, float]):
    mid = (i_lo + i_hi) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(mid > 0, (i_hi - i_lo) / (i_hi + i_lo), 0.0)
    new_mid = _log_map(mid, *rng)
    return new_mid * (1.0 - rel), new_mid * (1.0 + rel)


def multiply_data(wf: WorkflowDAG, data_multiplier: float) -> WorkflowDAG:
    """Scale every data size (DAG edges and external files) without remapping; for Suite A,
    whose sizes are already in Jin et al.'s ranges."""
    if data_multiplier < 0:
        raise ValueError("data_multiplier must be >= 0")
    out = copy.deepcopy(wf)
    out.D_lo, out.D_hi = wf.D_lo * data_multiplier, wf.D_hi * data_multiplier
    out.ext_input_kb, out.ext_output_kb = wf.ext_input_kb * data_multiplier, wf.ext_output_kb * data_multiplier
    out.meta = {**wf.meta, "data_multiplier": data_multiplier}
    return out


def rescale_workflow(wf: WorkflowDAG,
                     workload_range: Tuple[float, float] = JIN_WORKLOAD_GCYCLES,
                     data_range: Tuple[float, float] = JIN_DATA_KB,
                     data_multiplier: float = 1.0) -> WorkflowDAG:
    if not (0 < workload_range[0] <= workload_range[1]) or not (0 < data_range[0] <= data_range[1]):
        raise ValueError("ranges must be positive (lo, hi) with lo <= hi")
    if data_multiplier < 0:
        raise ValueError("data_multiplier must be >= 0")
    out = copy.deepcopy(wf)
    out.C_lo, out.C_hi = _rescale_interval(wf.C_lo, wf.C_hi, workload_range)
    D_lo, D_hi = _rescale_interval(wf.D_lo, wf.D_hi, data_range)
    out.D_lo, out.D_hi = D_lo * data_multiplier, D_hi * data_multiplier
    out.ext_input_kb = _log_map(wf.ext_input_kb, *data_range) * data_multiplier
    out.ext_output_kb = _log_map(wf.ext_output_kb, *data_range) * data_multiplier
    out.meta = {**wf.meta, "rescaled": {"workload_gcycles": list(workload_range), "data_kb": list(data_range),
                                        "data_multiplier": data_multiplier, "mapping": "log-linear"}}
    return out
