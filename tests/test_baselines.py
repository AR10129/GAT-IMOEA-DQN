"""IP-MOEA and IMOEA/D re-implementations."""
import numpy as np
import pytest

from conftest import ROOT
from dataset_tools.loader import load_scenario, load_workflow
from gat_imoea_dqn import DagIndex, EvaluatorConfig, IntervalEvaluator, allowed_for, is_allowed, is_topological, \
    init_population, pareto_dominance_matrix
from gat_imoea_dqn.baselines import (BaselineConfig, _weights, crowding_distance, imprecise_dominance_matrix,
                                     run_baseline)

SC_A = load_scenario(ROOT / "datasets" / "scenarios" / "suite_a" / "small_20.yaml")
WF_A = load_workflow(ROOT / "datasets" / "suite_a" / "small_20_i0.json")
SUITE_CFG = EvaluatorConfig(external_io="device", makespan="sum_apps", energy_scope="device")


def test_imprecise_dominance():
    lo = np.array([[0, 0, 0], [2, 2, 2], [0.5, 0.5, 0.5]], float)
    hi = lo + 1.0
    D = imprecise_dominance_matrix(lo, hi)
    assert D[0, 1] and not D[0, 2] and not D[1, 0]        # boxes 0 and 2 overlap -> incomparable


def test_crowding_distance():
    F = np.array([[0, 3], [1, 2], [2, 1], [3, 0]], float)
    cd = crowding_distance(F)
    assert np.isinf(cd[[0, 3]]).all() and cd[1] == pytest.approx(cd[2]) == pytest.approx(4 / 3)


def test_weights_count():
    W = _weights(105)
    assert W.shape == (105, 3) and np.allclose(W.sum(1), 1)


@pytest.mark.parametrize("name", ["ip_moea", "imoead"])
def test_baselines_valid_and_improve(name):
    cfg = BaselineConfig(pop_size=24, generations=25, seed=3, evaluator=SUITE_CFG)
    res = run_baseline(name, WF_A, SC_A, cfg)
    dag = DagIndex(WF_A)
    mask = allowed_for(WF_A, SC_A)
    assert len(res.population) == 24 and len(res.front) >= 1
    assert all(is_allowed(x, mask).all() for x in res.population.X)
    assert all(is_topological(o, dag) for o in res.population.O)
    # final objectives are what the evaluator says
    ev = IntervalEvaluator(WF_A, SC_A, SUITE_CFG)
    F_lo, F_hi = ev.evaluate_population(res.population.X, res.population.O)
    np.testing.assert_allclose(F_lo, res.F_lo) and np.testing.assert_allclose(F_hi, res.F_hi)
    # progress: the final population dominates most of the (same-seed) initial one on midpoints
    pop0 = init_population(WF_A, SC_A, 24, np.random.default_rng(3), dag)
    G_lo, G_hi = ev.evaluate_population(pop0.X, pop0.O)
    D = pareto_dominance_matrix(np.concatenate([(F_lo + F_hi) / 2, (G_lo + G_hi) / 2]))
    # IP-MOEA's box relation leaves almost everything non-dominated, so selection runs on
    # crowding and converges slowly (the weakness Jin et al. 2026 report); IMOEA/D converges.
    # (In this setting the whole initial population is ONE front under the box relation.)
    need, allow = (0.3, 0.3) if name == "ip_moea" else (0.8, 0.1)
    assert D[:24, 24:].any(axis=0).mean() >= need
    assert D[24:, :24].any(axis=0).mean() <= allow          # initial solutions rarely beat survivors


def test_unknown_baseline():
    with pytest.raises(ValueError):
        run_baseline("nsga9", WF_A, SC_A)
