"""Three-tier environmental selection (step 4) and the action set (step 3)."""
import numpy as np
import pytest

from conftest import ROOT
from dataset_tools.loader import load_scenario, load_workflow
from gat_imoea_dqn import (DagIndex, IntervalEvaluator, Population, SelectionConfig, das_dennis,
                           dominance_degree, environmental_selection, init_population, interval_crowding,
                           is_topological, make_offspring, normalize_intervals, pareto_dominance_matrix,
                           rankings, selection_actions)

SC = load_scenario(ROOT / "datasets" / "scenarios" / "default.yaml")


def test_das_dennis_actions():
    A = selection_actions()
    assert A.shape == (28, 3)
    np.testing.assert_allclose(A.sum(1), 1.0)
    assert len({tuple(r) for r in A}) == 28
    np.testing.assert_allclose(A * 6, np.round(A * 6))
    assert das_dennis(2, 3).shape == (6, 3)


def test_dominance_degree_chain():
    # a dominates b and c, b dominates c
    D = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], bool)
    np.testing.assert_allclose(dominance_degree(D), [0, 2, 3])   # S = [2, 1, 0]


def test_interval_crowding_by_hand():
    lo = np.array([[0, 0, 0], [2, 2, 2]], float)
    assert interval_crowding(lo, lo + 1, k=1) == pytest.approx([2.0, 2.0])     # 6 / (0 + 1 + 1 + 1)
    lo2 = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], float)
    assert interval_crowding(lo2, lo2 + 1, k=1) == pytest.approx([1.5 / 3.125] * 2)   # overlap 0.125


def test_crowding_prefers_isolated_points():
    mid = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [1, 1, 1]], float)
    icd = interval_crowding(mid, mid + 0.01, k=2)
    assert icd.argmax() == 3


def test_normalize_intervals():
    lo, hi = normalize_intervals(np.array([[0, 10], [5, 30]], float), np.array([[1, 20], [10, 50]], float))
    np.testing.assert_allclose(lo, [[0, 0], [0.5, 0.5]])
    np.testing.assert_allclose(hi, [[0.1, 0.25], [1, 1]])


def _pool(n=40, seed=0):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(0, 10, size=(n, 3))
    return lo, lo + rng.uniform(0, 2, size=(n, 3)), rng


def test_pure_convergence_takes_best_dominance_degree():
    lo, hi, rng = _pool()
    keep, info = environmental_selection(lo, hi, 20, (1, 0, 0), rng)
    assert len(keep) == 20 and len(set(keep.tolist())) == 20
    assert info.R_C[keep].max() <= np.sort(info.R_C)[19]


def test_pure_uncertainty_takes_best_uncertainty_degree():
    lo, hi, rng = _pool()
    keep, info = environmental_selection(lo, hi, 20, (0, 0, 1), rng)
    assert info.R_U[keep].max() <= np.sort(info.R_U)[19]


def test_pure_diversity_takes_highest_icd():
    lo, hi, rng = _pool()
    keep, info = environmental_selection(lo, hi, 20, (0, 1, 0), rng)
    assert info.ICD[keep].min() >= np.sort(info.ICD)[::-1][19]


def test_mixed_action_tiers_are_sequential_and_disjoint():
    lo, hi, rng = _pool(210)
    keep, info = environmental_selection(lo, hi, 105, (1 / 3, 1 / 2, 1 / 6), rng)
    assert info.counts == (35, 52, 18) and len(keep) == 105 == len(set(keep.tolist()))
    C, D, U = info.tiers["C"], info.tiers["D"], info.tiers["U"]
    assert (len(C), len(D), len(U)) == info.counts
    # tier D = best ICD among those not taken by tier C
    rest = np.setdiff1d(np.arange(210), C)
    assert info.ICD[D].min() >= np.sort(info.ICD[rest])[::-1][len(D) - 1]
    rest = np.setdiff1d(rest, D)
    assert info.R_U[U].max() <= np.sort(info.R_U[rest])[len(U) - 1]


def test_clones_are_skipped():
    lo, hi, rng = _pool(10)
    lo[1], hi[1] = lo[0], hi[0]
    X = np.arange(10)[:, None].repeat(4, 1)
    O = np.tile(np.arange(4), (10, 1))
    X[1] = X[0]                                                  # individual 1 is a clone of 0
    R_C, _, _ = rankings(lo, hi)
    keep, _ = environmental_selection(lo, hi, 9, (1, 0, 0), rng, genotypes=(X, O))
    assert not ({0, 1} <= set(keep.tolist()))                    # never both copies while others remain
    keep_all, _ = environmental_selection(lo, hi, 10, (1, 0, 0), rng, genotypes=(X, O))
    assert sorted(keep_all.tolist()) == list(range(10))          # filled once only clones remain
    keep_raw, _ = environmental_selection(lo, hi, 9, (1, 0, 0), rng, genotypes=(X, O),
                                          cfg=SelectionConfig(dedupe_genotypes=False))
    assert len(keep_raw) == 9


def test_rejects_bad_action():
    lo, hi, rng = _pool(10)
    with pytest.raises(ValueError):
        environmental_selection(lo, hi, 5, (0.5, 0.6, 0.0), rng)


@pytest.mark.parametrize("action", [(1, 0, 0), (1 / 3, 1 / 3, 1 / 3)])
def test_evolution_loop_improves(action):
    """init -> (mate -> merge -> select) x G; objectives must improve on a real workflow."""
    wf = load_workflow(ROOT / "datasets" / "workflows" / "montage" / "montage_100.json")
    dag, ev, rng = DagIndex(wf), IntervalEvaluator(wf, SC), np.random.default_rng(7)
    allowed = SC.allowed_targets()
    N = 30
    pop = init_population(wf, SC, N, rng, dag)
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    start = (F_lo + F_hi) / 2
    for _ in range(25):
        off, _ = make_offspring(pop, F_lo, F_hi, dag, allowed, rng)
        G_lo, G_hi = ev.evaluate_population(off.X, off.O)
        merged = Population.concat(pop, off)
        M_lo, M_hi = np.concatenate([F_lo, G_lo]), np.concatenate([F_hi, G_hi])
        keep, _ = environmental_selection(M_lo, M_hi, N, action, rng, genotypes=(merged.X, merged.O))
        pop, F_lo, F_hi = merged[keep], M_lo[keep], M_hi[keep]
    end = (F_lo + F_hi) / 2
    assert len(pop) == N and all(is_topological(o, dag) for o in pop.O)
    # Medians can trade off (e.g. lower cost for longer makespan), so measure
    # progress by dominance on interval midpoints: the final population must
    # dominate (almost) every initial solution.
    D = pareto_dominance_matrix(np.concatenate([end, start]))
    assert D[:N, N:].any(axis=0).mean() >= 0.9
    if action == (1, 0, 0):
        assert not D[N:, :N].any()                               # nothing initial beats a survivor


def test_icd_tie_break_keeps_spread():
    # 6 mutually non-dominated points on a line (all R_C = 0) plus two near-copies of the middle;
    # 4 C-tier slots must go to the most isolated non-dominated points, not a random subset
    lo = np.array([[0, 5, 0], [1, 4, 0], [2, 3, 0], [2.01, 2.99, 0], [2.02, 2.98, 0], [3, 2, 0],
                   [4, 1, 0], [5, 0, 0]], float)
    hi = lo + 0.01
    R_C, ICD, _ = rankings(lo, hi)
    assert (R_C == 0).all()
    for seed in range(5):
        keep, _ = environmental_selection(lo, hi, 4, (1.0, 0.0, 0.0), np.random.default_rng(seed),
                                          cfg=SelectionConfig(tie_break="icd"))
        assert set(keep.tolist()) == set(np.argsort(-ICD)[:4].tolist())
    with pytest.raises(ValueError):
        environmental_selection(lo, hi, 4, (1.0, 0.0, 0.0), np.random.default_rng(0),
                                cfg=SelectionConfig(tie_break="nope"))
