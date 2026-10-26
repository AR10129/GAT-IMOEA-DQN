"""Multi-device evaluator, Suite A instances, and the exact reduction to Jin et al. (2026)."""
import numpy as np
import pytest

from conftest import ROOT
from dataset_tools.loader import WorkflowDAG, load_scenario, load_workflow
from dataset_tools.suite_a import layered_dag, split_tasks
from gat_imoea_dqn import (DagIndex, EvaluatorConfig, IntervalEvaluator, InvalidIndividual, allowed_for,
                           init_population, is_allowed, is_topological, make_offspring)
from gat_imoea_dqn.algorithm import RunConfig, run
from gat_imoea_dqn.evaluator import KB_TO_MBIT

SC = load_scenario(ROOT / "datasets" / "scenarios" / "suite_a" / "small_20.yaml")
WF = load_workflow(ROOT / "datasets" / "suite_a" / "small_20_i0.json")
SUITE_CFG = EvaluatorConfig(external_io="device", makespan="sum_apps", energy_scope="device")


def test_generator_shapes():
    rng = np.random.default_rng(0)
    for K in (1, 2, 5, 12):
        edges, depth = layered_dag(K, rng)
        assert all(p < c for p, c in edges)                       # numbered topologically
        has_parent = {c for _, c in edges}
        assert all(k in has_parent for k in range(K) if depth[k] > 0)
    s = split_tasks(100, 20, rng)
    assert s.sum() == 100 and s.min() >= 1


def test_suite_instance_and_ownership():
    assert WF.num_tasks == 100 and len(np.unique(WF.owner)) == 20
    assert np.array_equal(WF.owner, WF.app)
    mask = allowed_for(WF, SC)
    assert mask.shape == (100, SC.n_edge + 2) and mask[:, 0].all()
    for d in range(SC.num_devices):                               # a task may use exactly its device's range
        rows = mask[WF.owner == d]
        assert np.array_equal(rows[0], np.isin(np.arange(SC.n_edge + 2), SC.allowed_targets(d)))


def test_population_respects_each_device_range():
    rng = np.random.default_rng(1)
    dag = DagIndex(WF)
    pop = init_population(WF, SC, 30, rng, dag)
    mask = allowed_for(WF, SC)
    assert all(is_allowed(x, mask).all() for x in pop.X)
    ev = IntervalEvaluator(WF, SC, SUITE_CFG)
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    off, _ = make_offspring(pop, F_lo, F_hi, dag, mask, rng)
    assert all(is_allowed(x, mask).all() for x in off.X) and all(is_topological(o, dag) for o in off.O)
    bad = pop.X[0].copy()
    k = int(np.flatnonzero(~mask.all(axis=1))[0])                 # a task with an unreachable edge
    bad[k] = int(np.flatnonzero(~mask[k])[0])
    with pytest.raises(InvalidIndividual, match="their device cannot reach"):
        ev.evaluate(bad, pop.O[0])


def test_devices_queue_separately():
    # two independent single-task apps on different devices: both local, run in parallel
    wf = _single_task_apps(np.array([3.0, 3.0]), np.array([0.0, 0.0]), owners=[0, 1])
    ev = IntervalEvaluator(wf, SC, EvaluatorConfig(makespan="max"))
    r = ev.evaluate([0, 0], [0, 1], detail=True)
    np.testing.assert_allclose(r.FT[0], [3.0 / SC.dev_f[0, 1], 3.0 / SC.dev_f[1, 1]])
    # same device: the second waits for the first
    wf2 = _single_task_apps(np.array([3.0, 3.0]), np.array([0.0, 0.0]), owners=[0, 0])
    r2 = IntervalEvaluator(wf2, SC, EvaluatorConfig(makespan="max")).evaluate([0, 0], [0, 1], detail=True)
    np.testing.assert_allclose(r2.FT[0], [3.0 / SC.dev_f[0, 1], 6.0 / SC.dev_f[0, 1]])


def _single_task_apps(C, S, owners):
    K = len(C)
    return WorkflowDAG(name="jin", family="test", C_lo=C, C_hi=C, edge_index=np.zeros((2, 0), dtype=np.int64),
                       D_lo=np.zeros(0), D_hi=np.zeros(0), ext_input_kb=S, ext_output_kb=np.zeros(K),
                       depth=np.zeros(K, dtype=np.int64), task_types=["t"] * K,
                       owner=np.asarray(owners), app=np.arange(K))


def test_reduces_exactly_to_jin_et_al():
    """Single-task applications, no device queue: F1/F2/F3 equal Jin et al. eqs. (2)-(20)."""
    rng = np.random.default_rng(3)
    K = 40
    owners = rng.integers(0, SC.num_devices, size=K)
    C = rng.integers(1, 6, size=K).astype(float)
    S = rng.uniform(500, 3000, size=K)
    wf = _single_task_apps(C, S, owners)
    ev = IntervalEvaluator(wf, SC, EvaluatorConfig(external_io="device", makespan="sum_apps",
                                                   energy_scope="device", device_queue=False))
    mask = allowed_for(wf, SC)
    X = np.array([rng.choice(np.flatnonzero(mask[k])) for k in range(K)])
    r = ev.evaluate(X, np.arange(K))
    n, cloud = SC.n_edge, SC.n_edge + 1
    n_on = np.bincount(X, minlength=n + 2)
    W_lo, W_hi = SC.raw["links"]["backhaul_mbps"]
    for b, (fsel, rsel, wc) in enumerate(((1, 1, W_hi), (0, 0, W_lo))):   # lower bound: fast end
        lat = en = cost = 0.0
        for k in range(K):
            d, x = owners[k], X[k]
            if x == 0:                                                     # eqs. (2), (3)
                lat += C[k] / SC.dev_f[d, fsel]
                en += C[k] * SC.dev_e[d, 1 - fsel]
                continue
            u = x if x <= n else SC.dev_nearest[d]
            t_up = S[k] * KB_TO_MBIT / SC.dev_uplink[d, u - 1, rsel]       # eq. (4)
            if x <= n:                                                     # eqs. (5)-(8)
                f_share = SC.f_edge[x - 1, fsel] / n_on[x]
                t_exe = C[k] / f_share
                lat += t_up + t_exe
                en += SC.dev_ptx[d] * t_up + SC.dev_pidle[d] * t_exe
            else:                                                          # eqs. (9)-(13)
                t_wired = S[k] * KB_TO_MBIT / wc
                t_exe = C[k] / SC.f_cloud[fsel]
                lat += t_up + t_wired + t_exe
                en += SC.dev_ptx[d] * t_up + SC.dev_pidle[d] * t_wired + SC.dev_pidle[d] * t_exe
                cost += 0.01 * t_exe
        F = r.F_lo if b == 0 else r.F_hi
        assert F[0] == pytest.approx(lat)
        assert F[1] == pytest.approx(en)
    # cost, eqs. (8) and (13) in interval arithmetic: eta_E * f~_eff * t~_exe gives the lower
    # bound eta_E * f_lo * C / f_hi (the product does not cancel), and eta_C * t~_exe
    exp_cost_lo = sum(0.001 * (SC.f_edge[X[k] - 1, 0] / n_on[X[k]]) * C[k] / (SC.f_edge[X[k] - 1, 1] / n_on[X[k]])
                      for k in range(K) if 0 < X[k] <= n) \
        + sum(0.01 * C[k] / SC.f_cloud[1] for k in range(K) if X[k] == cloud)
    exp_cost_hi = sum(0.001 * (SC.f_edge[X[k] - 1, 1] / n_on[X[k]]) * C[k] / (SC.f_edge[X[k] - 1, 0] / n_on[X[k]])
                      for k in range(K) if 0 < X[k] <= n) \
        + sum(0.01 * C[k] / SC.f_cloud[0] for k in range(K) if X[k] == cloud)
    assert r.F_lo[2] == pytest.approx(exp_cost_lo)
    assert r.F_hi[2] == pytest.approx(exp_cost_hi)


def test_suite_a_run_end_to_end():
    res = run(WF, SC, RunConfig(pop_size=16, generations=8, policy="fixed", fixed_action=(5 / 6, 1 / 6, 0),
                                evaluator=SUITE_CFG))
    mask = allowed_for(WF, SC)
    assert all(is_allowed(x, mask).all() for x in res.population.X)
    assert np.all(res.F_lo <= res.F_hi) and len(res.front) >= 1
