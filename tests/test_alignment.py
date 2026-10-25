"""Jin et al. alignment: rescaling, fixed channel gain, downlink = uplink, energy scope."""
import math

import numpy as np
import pytest

from conftest import ROOT
from dataset_tools.loader import load_scenario, load_workflow
from dataset_tools.rescale import JIN_DATA_KB, JIN_WORKLOAD_GCYCLES, rescale_workflow
from gat_imoea_dqn import EvaluatorConfig, IntervalEvaluator, init_population, DagIndex

WF = load_workflow(ROOT / "datasets" / "workflows" / "cybershake" / "cybershake_200.json")
JIN = load_scenario(ROOT / "datasets" / "scenarios" / "jin.yaml")


def test_rescale_ranges_order_and_widths():
    r = rescale_workflow(WF)
    mid, mid0 = (r.C_lo + r.C_hi) / 2, (WF.C_lo + WF.C_hi) / 2
    assert mid.min() == pytest.approx(JIN_WORKLOAD_GCYCLES[0]) and mid.max() == pytest.approx(JIN_WORKLOAD_GCYCLES[1])
    assert np.array_equal(np.argsort(mid, kind="stable"), np.argsort(mid0, kind="stable"))   # order kept
    np.testing.assert_allclose((r.C_hi - r.C_lo) / (r.C_hi + r.C_lo), (WF.C_hi - WF.C_lo) / (WF.C_hi + WF.C_lo))
    d = (r.D_lo + r.D_hi) / 2
    pos = (WF.D_lo + WF.D_hi) > 0
    assert d[pos].min() == pytest.approx(JIN_DATA_KB[0]) and d[pos].max() == pytest.approx(JIN_DATA_KB[1])
    assert np.all(d[~pos] == 0)
    assert np.all(r.C_lo <= r.C_hi) and np.all(r.D_lo <= r.D_hi)
    np.testing.assert_array_equal(r.edge_index, WF.edge_index)          # graph untouched
    assert WF.C_lo.max() > 100                                          # original not modified


def test_data_multiplier():
    a, b = rescale_workflow(WF), rescale_workflow(WF, data_multiplier=10)
    np.testing.assert_allclose(b.D_hi, 10 * a.D_hi)
    np.testing.assert_allclose(b.C_hi, a.C_hi)


def test_jin_scenario_rates():
    # Jin et al.: r = B log2(1 + p g / v0) with g = 0.001, v0 = -100 dBm; device 2 transmits at 0.4 W
    se = math.log2(1 + 0.4 * 1e-3 / 1e-13)
    assert JIN.edge_in_range.all()
    np.testing.assert_allclose(JIN.uplink_mbps, [[9 * se, 12 * se]] * 3)
    lo, hi = JIN.rate_matrix_mbps()
    np.testing.assert_allclose([lo[1, 0], hi[1, 0]], [9 * se, 12 * se])  # downlink = uplink
    assert (lo[1, 4], hi[1, 4]) == (408.0, 624.0)                        # edge -> cloud = W_C
    assert JIN.f_edge.tolist() == [[78.5, 90.0], [102.0, 120.0], [127.5, 150.0]]


def test_energy_scope_device_only():
    wf = rescale_workflow(WF)
    pop = init_population(wf, JIN, 4, np.random.default_rng(0), DagIndex(wf))
    full = IntervalEvaluator(wf, JIN).evaluate(pop.X[0], pop.O[0], detail=True)
    dev = IntervalEvaluator(wf, JIN, EvaluatorConfig(energy_scope="device")).evaluate(pop.X[0], pop.O[0])
    server = sum(full.breakdown[k][0] for k in ("energy_edge", "energy_cloud"))
    assert server > 0
    assert dev.F_lo[1] == pytest.approx(full.F_lo[1] - server)
    np.testing.assert_allclose(dev.F_lo[[0, 2]], full.F_lo[[0, 2]])     # makespan and cost unchanged
    with pytest.raises(ValueError):
        IntervalEvaluator(wf, JIN, EvaluatorConfig(energy_scope="edge"))
