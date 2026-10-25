"""Chromosome, operators and front sorting."""
import numpy as np
import pytest

from conftest import ROOT
from dataset_tools.loader import load_scenario, load_workflow
from gat_imoea_dqn import (DagIndex, IntervalEvaluator, VariationConfig, daec, init_population,
                           interval_dominance_matrix, is_topological, make_offspring, mutate_allocation,
                           mutate_order, nondominated_fronts, pareto_dominance_matrix, possibility_leq, pox,
                           repair_allocation, repair_order, shift_task)
from gat_imoea_dqn import operators as opmod

WF_DIR = ROOT / "datasets" / "workflows"
SC = load_scenario(ROOT / "datasets" / "scenarios" / "default.yaml")
NAMES = ["montage/montage_400", "epigenomics/epigenomics_200", "cybershake/cybershake_200",
         "inspiral/inspiral_400", "sipht/sipht_200"]


@pytest.fixture(scope="module", params=NAMES)
def problem(request):
    wf = load_workflow(WF_DIR / f"{request.param}.json")
    return wf, DagIndex(wf)


# ------------------------------------------------------------- intervals --
def test_possibility_degree():
    assert possibility_leq(1, 2, 3, 4) == 1.0                  # A entirely left of B
    assert possibility_leq(3, 4, 1, 2) == 0.0
    assert possibility_leq(1, 3, 1, 3) == pytest.approx(0.5)  # identical intervals
    assert possibility_leq(0, 2, 1, 3) == pytest.approx(0.75)  # (3 - 0) / (2 + 2)
    assert possibility_leq(2, 2, 2, 2) == 0.5                  # equal points
    a = possibility_leq(0, 4, 1, 2) + possibility_leq(1, 2, 0, 4)
    assert a == pytest.approx(1.0)


def test_interval_fronts_by_hand():
    F_lo = np.array([[1, 1], [2, 2], [1, 3], [5, 5]], float)
    F_hi = F_lo + 0.5
    D = interval_dominance_matrix(F_lo, F_hi)
    assert D[0, 1] and D[0, 3] and D[1, 3] and not D[1, 0]
    fronts = nondominated_fronts(D)
    assert [sorted(f.tolist()) for f in fronts] == [[0], [1, 2], [3]]


def test_fronts_terminate_on_cycles():
    D = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], bool)        # 0>1>2>0
    fronts = nondominated_fronts(D)
    assert sorted(np.concatenate(fronts).tolist()) == [0, 1, 2]


def test_pareto_fronts():
    F = np.array([[1, 5], [2, 2], [5, 1], [3, 3], [6, 6]], float)
    fronts = nondominated_fronts(pareto_dominance_matrix(F))
    assert [sorted(f.tolist()) for f in fronts] == [[0, 1, 2], [3], [4]]


# ------------------------------------------------------------ chromosome --
def test_init_population(problem):
    wf, dag = problem
    pop = init_population(wf, SC, 105, np.random.default_rng(0), dag)
    assert pop.X.shape == pop.O.shape == (105, wf.num_tasks)
    assert np.isin(pop.X, SC.allowed_targets()).all()
    assert all(is_topological(o, dag) for o in pop.O)
    assert len({o.tobytes() for o in pop.O}) > 1                 # orders are randomized
    again = init_population(wf, SC, 105, np.random.default_rng(0), dag)
    assert np.array_equal(pop.X, again.X) and np.array_equal(pop.O, again.O)


def test_repair_order(problem):
    wf, dag = problem
    rng = np.random.default_rng(1)
    valid = init_population(wf, SC, 1, rng, dag).O[0]
    assert np.array_equal(repair_order(valid, dag), valid)        # already valid: unchanged
    for _ in range(5):
        fixed = repair_order(rng.permutation(wf.num_tasks), dag)
        assert is_topological(fixed, dag)
    broken = valid.copy()
    broken[:3] = broken[0]                                        # duplicates / missing tasks
    assert is_topological(repair_order(broken, dag), dag)


def test_repair_allocation():
    allowed = np.array([0, 1, 5])
    x = repair_allocation(np.array([0, 2, 5, 9, 1]), allowed, np.random.default_rng(0))
    assert np.isin(x, allowed).all() and x[0] == 0 and x[2] == 5 and x[4] == 1


# ------------------------------------------------------------- operators --
@pytest.mark.parametrize("mode", ["mask", "one_point"])
def test_pox_children_are_always_valid(problem, mode):
    wf, dag = problem
    rng = np.random.default_rng(2)
    pop = init_population(wf, SC, 20, rng, dag)
    for _ in range(40):
        i, j = rng.choice(20, 2, replace=False)
        child = pox(pop.O[i], pop.O[j], rng, mode)
        assert is_topological(child, dag)                         # no repair needed


def test_pox_masks():
    p1, p2 = np.array([0, 1, 2, 3, 4]), np.array([4, 3, 2, 1, 0])
    assert opmod._pox_py(p1, p2, np.ones(5, bool)).tolist() == p1.tolist()
    assert opmod._pox_py(p1, p2, np.zeros(5, bool)).tolist() == p2.tolist()
    assert opmod._pox_py(p1, p2, np.array([1, 0, 1, 0, 1], bool)).tolist() == [0, 4, 1, 3, 2]
    m = np.random.default_rng(0).random(5) < .5
    assert np.array_equal(opmod._pox(p1, p2, m), opmod._pox_py(p1, p2, m))   # numba == python


def test_daec():
    rng = np.random.default_rng(0)
    a, b = np.zeros(1000, np.int64), np.ones(1000, np.int64)
    assert np.array_equal(daec(a, b, rng, 1.0), a)
    child = daec(a, b, rng, 0.5)
    assert set(child.tolist()) == {0, 1} and 400 < (child == 0).sum() < 600


def test_mutations_keep_validity(problem):
    wf, dag = problem
    rng = np.random.default_rng(3)
    allowed = SC.allowed_targets()
    pop = init_population(wf, SC, 1, rng, dag)
    x = mutate_allocation(pop.X[0], allowed, rng, 0.2)
    assert np.isin(x, allowed).all() and (x != pop.X[0]).mean() > 0.1
    o = pop.O[0]
    moved = False
    for _ in range(30):
        new = mutate_order(o.copy(), dag, rng, 0.05)
        assert is_topological(new, dag)
        moved |= not np.array_equal(new, o)
        o = new
    assert moved


def test_shift_task_window():
    from dataset_tools.loader import WorkflowDAG
    # chain 0 -> 1 -> 2 plus free task 3: task 1 can never move, task 3 can go anywhere
    wf = WorkflowDAG("t", "t", np.ones(4), np.ones(4), np.array([[0, 1], [1, 2]]), np.ones(2), np.ones(2),
                     np.zeros(4), np.zeros(4), np.array([0, 1, 2, 0]), ["t"] * 4)
    dag = DagIndex(wf)
    rng = np.random.default_rng(0)
    O = np.array([0, 1, 2, 3])
    assert all(np.array_equal(shift_task(O, 1, dag, rng), O) for _ in range(10))
    seen = {tuple(shift_task(O, 3, dag, rng)) for _ in range(200)}
    assert len(seen) == 4 and all(is_topological(np.array(s), dag) for s in seen)


# ---------------------------------------------------------------- mating --
def test_make_offspring_rules(problem):
    wf, dag = problem
    rng = np.random.default_rng(4)
    ev = IntervalEvaluator(wf, SC)
    pop = init_population(wf, SC, 105, rng, dag)
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    off, info = make_offspring(pop, F_lo, F_hi, dag, SC.allowed_targets(), rng)

    assert len(off) == 105
    assert np.isin(off.X, SC.allowed_targets()).all()
    assert all(is_topological(o, dag) for o in off.O)
    ev.evaluate_population(off.X, off.O)                           # evaluator accepts all children

    assert info.elite.sum() == 53                                  # ceil(105 / 2)
    assert np.all(info.elite[info.partner[info.elite]])            # TopPI mates only inside TopPI
    assert info.partner.tolist() != list(range(105)) and np.all(info.partner != np.arange(105))
    worst_elite = info.interval_rank[info.elite].max()
    assert np.all(info.interval_rank[~info.elite] >= worst_elite)  # elite = best fronts
    assert set(np.unique(info.p_c).tolist()) <= {0.7, 0.9}
    assert (info.p_c == 0.7).sum() == 53
    low = info.p_c == 0.7
    assert np.all(info.uncertainty_rank[~low] >= info.uncertainty_rank[low].max())
    assert 0.6 < info.crossed.mean() < 0.98


def test_no_crossover_means_mutated_copies(problem):
    wf, dag = problem
    rng = np.random.default_rng(5)
    pop = init_population(wf, SC, 10, rng, dag)
    F = np.abs(rng.normal(size=(10, 3)))
    cfg = VariationConfig(p_c_low=0.0, p_c_high=0.0, p_m_alloc=0.0, p_m_order=0.0)
    off, info = make_offspring(pop, F, F + 1, dag, SC.allowed_targets(), rng, cfg)
    assert not info.crossed.any()
    assert np.array_equal(off.X, pop.X) and np.array_equal(off.O, pop.O)
