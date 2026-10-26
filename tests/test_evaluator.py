"""Hand-checked cases for the interval critical-path evaluator.

Diamond DAG 0 -> {1, 2} -> 3, crisp scenario (every interval has lo == hi),
so each expected number below can be worked out on paper.
"""
import math

import numpy as np
import pytest
import yaml

from conftest import ROOT
from dataset_tools.loader import WorkflowDAG, load_scenario, load_workflow
from gat_imoea_dqn import EvaluatorConfig, IntervalEvaluator, InvalidIndividual, uncertainty_metrics
from gat_imoea_dqn import evaluator as evmod

C = [2.0, 4.0, 6.0, 8.0]                       # Gcycles
EDGES = [(0, 1, 1000.0), (0, 2, 2000.0), (1, 3, 500.0), (2, 3, 1500.0)]   # KB


def diamond(C_lo=C, C_hi=C, D_scale=(1.0, 1.0), ext_in=(0, 0, 0, 0), ext_out=(0, 0, 0, 0)):
    src = [e[0] for e in EDGES]
    dst = [e[1] for e in EDGES]
    D = np.array([e[2] for e in EDGES])
    return WorkflowDAG(
        name="diamond", family="test", C_lo=np.array(C_lo, float), C_hi=np.array(C_hi, float),
        edge_index=np.array([src, dst]), D_lo=D * D_scale[0], D_hi=D * D_scale[1],
        ext_input_kb=np.array(ext_in, float), ext_output_kb=np.array(ext_out, float),
        depth=np.array([0, 1, 1, 2]), task_types=["t"] * 4)


def scenario_dict(f_dev=(1, 1), f_e1=(10, 10), f_e2=(5, 5), f_c=(20, 20), B=(10, 10), down=(50, 50),
                  bh=(100, 100), e2e=(80, 80)):
    return {
        "name": "tiny", "seed": 0,
        "mobile_devices": {"count": 1, "origin_device": 0, "cpu_ghz": [list(f_dev)],
                           "energy_per_gcycle_j": [[1.0, 1.0]], "tx_power_w": [0.5],
                           "idle_power_w": [0.1], "positions_m": [[0, 0]]},
        "edge_servers": {"count": 2, "cpu_ghz": [list(f_e1), list(f_e2)], "kappa_j_per_gcycle_ghz2": 0.01,
                         "static_power_w": [2.0, 3.0], "positions_m": [[100, 0], [0, 100]],
                         "coverage_radius_m": 150},
        "cloud": {"cpu_ghz": list(f_c), "kappa_j_per_gcycle_ghz2": 0.001, "static_power_w": 0.0},
        "links": {"wireless_uplink": {"bandwidth_mhz": list(B), "noise_power_w": 1e-13,
                                      "path_loss_exponent": 4.0},
                  "wireless_downlink_mbps": list(down), "backhaul_mbps": list(bh),
                  "edge_to_edge_mbps": list(e2e), "device_cloud_route": "via_nearest_edge"},
        "pricing": {"edge_usd_per_ghz_s": 0.001, "cloud_usd_per_s": 0.01},
    }


@pytest.fixture
def make_sc(tmp_path):
    def _make(**kw):
        p = tmp_path / f"sc_{len(list(tmp_path.iterdir()))}.yaml"
        p.write_text(yaml.safe_dump(scenario_dict(**kw)))
        return load_scenario(p)
    return _make


UPLINK = 10 * math.log2(1 + 0.5 * 100.0 ** -4 / 1e-13)    # Mbps, both edges are 100 m away
O = np.array([0, 1, 2, 3])


def test_uplink_rate_formula(make_sc):
    sc = make_sc()
    assert sc.allowed_targets().tolist() == [0, 1, 2, 3]
    assert sc.uplink_mbps[0, 0] == pytest.approx(UPLINK)
    lo, hi = sc.rate_matrix_mbps()
    assert lo[0, 3] == pytest.approx(1 / (1 / UPLINK + 1 / 100))     # device -> cloud via edge 1


def test_all_local_runs_sequentially(make_sc):
    r = IntervalEvaluator(diamond(), make_sc()).evaluate([0, 0, 0, 0], O)
    np.testing.assert_allclose(r.F_lo, [20.0, 20.0, 0.0])         # 2+4+6+8 s, 20 Gcycles * 1 J
    np.testing.assert_allclose(r.F_hi, r.F_lo)


def test_all_cloud_runs_in_parallel(make_sc):
    r = IntervalEvaluator(diamond(), make_sc()).evaluate([3, 3, 3, 3], O)
    # 0.1 + max(0.2, 0.3) + 0.4; idle 0.1 W * 1.0 s + 0.001*20^2*20 J; 0.01 $/s * 1.0 s
    np.testing.assert_allclose(r.F_lo, [0.8, 0.1 + 8.0, 0.01])


def test_mixed_allocation_by_hand(make_sc):
    r = IntervalEvaluator(diamond(), make_sc()).evaluate([0, 1, 2, 0], O, detail=True)
    t01 = 1000 * 0.008 / UPLINK
    t02 = 2000 * 0.008 / UPLINK
    FT1 = 2 + t01 + 4 / 10
    FT2 = 2 + t02 + 6 / 5
    AT3 = max(FT1 + 500 * 0.008 / 50, FT2 + 1500 * 0.008 / 50)
    FT3 = max(AT3, 2.0) + 8
    energy = (2 + 8) + 0.5 * (t01 + t02) + 0.1 * (0.4 + 1.2) \
        + (0.01 * 100 * 4 + 2 * FT1) + (0.01 * 25 * 6 + 3 * FT2)
    cost = 0.001 * (10 * 0.4 + 5 * 1.2)
    np.testing.assert_allclose(r.F_lo, [FT3, energy, cost])
    np.testing.assert_allclose(r.FT[0], [2, FT1, FT2, FT3])


def test_edge_processor_sharing_and_optional_queue(make_sc):
    sc = make_sc()
    shared = IntervalEvaluator(diamond(), sc).evaluate([1, 1, 1, 1], O)
    # 4 tasks share 10 GHz -> 2.5 GHz each, tasks 1 and 2 overlap
    assert shared.F_lo[0] == pytest.approx(0.8 + 2.4 + 3.2)
    assert shared.F_lo[2] == pytest.approx(0.001 * 20)            # eta_E * f_eff * t = eta_E * C
    queued = IntervalEvaluator(diamond(), sc, EvaluatorConfig(edge_queue=True)).evaluate([1, 1, 1, 1], O)
    assert queued.F_lo[0] == pytest.approx(0.8 + 1.6 + 2.4 + 3.2)


def test_priority_order_changes_device_queue(make_sc):
    ev = IntervalEvaluator(diamond(), make_sc())
    a = ev.evaluate([0, 0, 0, 0], [0, 1, 2, 3], detail=True)
    b = ev.evaluate([0, 0, 0, 0], [0, 2, 1, 3], detail=True)
    # tasks 1 and 2 compete for the single device CPU; O decides who goes first
    np.testing.assert_allclose(a.FT[0], [2, 6, 12, 20])
    np.testing.assert_allclose(b.FT[0], [2, 12, 8, 20])


def test_interval_bounds_match_extreme_crisp_worlds(make_sc):
    """Makespan lower bound = crisp run with small work and fast resources, and vice versa."""
    C_lo, C_hi = [1.5, 3.0, 5.0, 7.0], [2.5, 5.0, 7.0, 9.0]
    sc = make_sc(f_dev=(1, 2), f_e1=(8, 12), f_e2=(4, 6), f_c=(18, 22), B=(10, 20), down=(40, 60),
                 bh=(90, 110), e2e=(70, 90))
    wf = diamond(C_lo, C_hi, D_scale=(0.8, 1.2))
    X = [0, 1, 2, 3]
    r = IntervalEvaluator(wf, sc).evaluate(X, O)
    fast = make_sc(f_dev=(2, 2), f_e1=(12, 12), f_e2=(6, 6), f_c=(22, 22), B=(20, 20), down=(60, 60),
                   bh=(110, 110), e2e=(90, 90))
    slow = make_sc(f_dev=(1, 1), f_e1=(8, 8), f_e2=(4, 4), f_c=(18, 18), B=(10, 10), down=(40, 40),
                   bh=(90, 90), e2e=(70, 70))
    lo = IntervalEvaluator(diamond(C_lo, C_lo, D_scale=(0.8, 0.8)), fast).evaluate(X, O)
    hi = IntervalEvaluator(diamond(C_hi, C_hi, D_scale=(1.2, 1.2)), slow).evaluate(X, O)
    assert r.F_lo[0] == pytest.approx(lo.F_lo[0])
    assert r.F_hi[0] == pytest.approx(hi.F_hi[0])
    assert np.all(r.F_lo <= r.F_hi)


def test_external_io_mode(make_sc):
    wf = diamond(ext_in=(4000, 0, 0, 0), ext_out=(0, 0, 0, 1000))
    sc = make_sc()
    X = [1, 1, 1, 1]
    pdf = IntervalEvaluator(wf, sc).evaluate(X, O)
    dev = IntervalEvaluator(wf, sc, EvaluatorConfig(external_io="device")).evaluate(X, O)
    up = 4000 * 0.008 / UPLINK
    back = 1000 * 0.008 / 50
    assert dev.F_lo[0] == pytest.approx(pdf.F_lo[0] + up + back)
    assert dev.F_lo[1] > pdf.F_lo[1]                               # upload energy added


def test_rejects_invalid_individuals(make_sc):
    ev = IntervalEvaluator(diamond(), make_sc())
    with pytest.raises(InvalidIndividual, match="topological"):
        ev.evaluate([0, 0, 0, 0], [1, 0, 2, 3])
    with pytest.raises(InvalidIndividual, match="permutation"):
        ev.evaluate([0, 0, 0, 0], [0, 1, 1, 3])
    with pytest.raises(InvalidIndividual):
        ev.evaluate([0, 0, 0, 9], O)


def test_out_of_range_edge_is_rejected(tmp_path):
    d = scenario_dict()
    d["edge_servers"]["positions_m"][1] = [0, 900]                 # edge 2 out of range
    p = tmp_path / "far.yaml"
    p.write_text(yaml.safe_dump(d))
    sc = load_scenario(p)
    assert sc.allowed_targets().tolist() == [0, 1, 3]
    with pytest.raises(InvalidIndividual, match="cannot reach"):
        IntervalEvaluator(diamond(), sc).evaluate([0, 2, 0, 0], O)


def test_uncertainty_metrics():
    U1, U2 = uncertainty_metrics(np.array([[1, 2, 3], [1, 2, 0]]), np.array([[2, 4, 6], [1, 5, 0]]))
    np.testing.assert_allclose(U1, [6, 3])
    np.testing.assert_allclose(U2, [6, 0])


@pytest.mark.parametrize("name", ["montage/montage_200", "cybershake/cybershake_400", "sipht/sipht_200"])
def test_real_workflows_numba_matches_python(name):
    wf = load_workflow(ROOT / "datasets" / "workflows" / f"{name}.json")
    sc = load_scenario(ROOT / "datasets" / "scenarios" / "default.yaml")
    ev = IntervalEvaluator(wf, sc)
    rng = np.random.default_rng(1)
    targets = sc.allowed_targets()
    for _ in range(5):
        X = rng.choice(targets, size=wf.num_tasks)
        Ord = wf.random_topological_order(rng)
        r = ev.evaluate(X, Ord, detail=True)
        assert r.feasible and np.all(np.isfinite(r.F_hi)) and np.all(r.F_lo <= r.F_hi)
        ref = evmod._schedule_py(Ord, ev.internal(X), ev.pred_ptr, ev.pred_edge, ev.src,
                                 *_tt_tc(ev, X), np.zeros(wf.num_tasks), np.zeros(wf.num_tasks),
                                 ev.queued, ev.NI)
        np.testing.assert_allclose(r.FT[0], ref[2])
        np.testing.assert_allclose(r.FT[1], ref[3])
        # every task starts after all parents finish
        assert np.all(r.ST[0][ev.dst] >= r.FT[0][ev.src] - 1e-9)


def _tt_tc(ev, X):
    I = ev.internal(X)
    share = np.where(ev.is_edge, np.maximum(np.bincount(I, minlength=ev.NI), 1), 1)
    xs, xd = I[ev.src], I[ev.dst]
    return (ev.D_lo * ev.spk_lo[xs, xd], ev.D_hi * ev.spk_hi[xs, xd],
            ev.C_lo / (ev.f_hi / share)[I], ev.C_hi / (ev.f_lo / share)[I])
