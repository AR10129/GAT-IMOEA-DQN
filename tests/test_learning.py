"""GAT encoder, population features, PIBI reward, DQN agent and the full run loop."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import ROOT  # noqa: E402
from dataset_tools.loader import WorkflowDAG, load_scenario, load_workflow  # noqa: E402
from gat_imoea_dqn import is_topological, DagIndex  # noqa: E402
from gat_imoea_dqn.algorithm import RunConfig, run  # noqa: E402
from gat_imoea_dqn.dqn import DQNAgent, DQNConfig  # noqa: E402
from gat_imoea_dqn.gat import GATEncoder, GraphTensors, segment_softmax  # noqa: E402
from gat_imoea_dqn.state import PopulationTracker, decision_clusters, pibi  # noqa: E402

SC = load_scenario(ROOT / "datasets" / "scenarios" / "default.yaml")
WF = load_workflow(ROOT / "datasets" / "workflows" / "montage" / "montage_60.json")


# ------------------------------------------------------------------- GAT --
def test_segment_softmax_sums_to_one():
    scores = torch.randn(7, 3)
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    p = segment_softmax(scores, idx, 3)
    s = torch.zeros(3, 3).index_add_(0, idx, p)
    assert torch.allclose(s, torch.ones(3, 3))


def test_gat_output_and_gradients():
    torch.manual_seed(0)
    enc = GATEncoder()
    g = GraphTensors.from_workflow(WF)
    assert g.x.shape == (WF.num_tasks, 5) and g.edge_attr.shape[1] == 2
    assert g.dst.numel() == 2 * WF.num_edges + WF.num_tasks              # both directions + self-loops
    out = enc(g)
    assert out.shape == (16,) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def _permuted(wf, perm):
    """Same DAG with node i renamed to perm[i]."""
    inv = np.argsort(perm)
    return WorkflowDAG(wf.name, wf.family, wf.C_lo[inv], wf.C_hi[inv], perm[wf.edge_index], wf.D_lo, wf.D_hi,
                       wf.ext_input_kb[inv], wf.ext_output_kb[inv], wf.depth[inv],
                       [wf.task_types[i] for i in inv])


def test_graph_embedding_ignores_node_numbering():
    torch.manual_seed(0)
    enc = GATEncoder()
    perm = np.random.default_rng(0).permutation(WF.num_tasks)
    a = enc(GraphTensors.from_workflow(WF))
    b = enc(GraphTensors.from_workflow(_permuted(WF, perm)))
    assert torch.allclose(a, b, atol=1e-5)


def test_edge_features_change_the_embedding():
    torch.manual_seed(0)
    enc = GATEncoder()
    g = GraphTensors.from_workflow(WF)
    g2 = GraphTensors(g.x, g.edge_attr * 3.0, g.dst, g.src, g.num_nodes)
    assert not torch.allclose(enc(g), enc(g2))


# ------------------------------------------------------- state / reward --
def test_decision_clusters():
    X = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [5, 5, 5, 5], [0, 0, 0, 0]])
    assert decision_clusters(X, 0.25).tolist() == [0, 0, 1, 0]
    assert len(set(decision_clusters(X, 0.0).tolist())) == 3


def test_pibi_values():
    W = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    origin = np.zeros((1, 3))
    assert pibi(origin, origin, W) == 0.0
    pt = np.array([[1.0, 0, 0]])
    # w1: d1 = 1, d2 = 0 -> 1;  w2/w3: d1 = 0, d2 = 1 -> theta = 5
    assert pibi(pt, pt, W, theta=5.0) == pytest.approx((1 + 5 + 5) / 3)
    better = np.array([[0.5, 0.5, 0.5]])
    worse = better + 0.3
    assert pibi(better, better, W) < pibi(worse, worse, W)


def test_tracker_features_and_reward():
    rng = np.random.default_rng(0)
    tr = PopulationTracker(g_max=50)
    X = rng.integers(0, 3, size=(20, 30))
    lo = rng.uniform(1, 5, size=(20, 3))
    f0 = tr.features(X, lo, lo + 1, 0)
    assert f0.shape == (7,) and f0[0] == f0[1] == f0[3] == f0[4] == f0[5] == 0 and f0[6] == 0
    f1 = tr.features(X, lo * 0.5, lo * 0.5 + 0.5, 10)
    assert f1[0] > 0 and f1[4] < 0 and f1[6] == pytest.approx(0.2) and np.all(np.abs(f1[3:6]) <= 1)
    assert tr.reward((lo, lo + 1), (lo * 0.5, lo * 0.5 + 0.5)) > 0          # improvement -> positive
    assert tr.reward((lo * 0.5, lo * 0.5 + 0.5), (lo, lo + 1)) < 0


def test_hypervolume_reward():
    rng = np.random.default_rng(1)
    lo = rng.uniform(1, 5, size=(20, 3))
    tr = PopulationTracker(g_max=50, reward="hv")
    tr.observe(lo, lo + 1)
    better = (lo * 0.6, lo * 0.6 + 0.6)
    assert tr.reward((lo, lo + 1), better) > 0
    assert tr.reward(better, (lo, lo + 1)) < 0
    assert tr.reward((lo, lo + 1), (lo, lo + 1)) == 0.0
    with pytest.raises(ValueError):
        PopulationTracker(g_max=5, reward="nope")


def test_hv_jin_reward_tracks_the_metric():
    from gat_imoea_dqn.metrics import hv_jin
    rng = np.random.default_rng(2)
    lo = rng.uniform(1, 5, size=(20, 3))
    tr = PopulationTracker(g_max=50, reward="hvjin", normalization="fixed")
    tr.observe(lo, lo + 1)
    nl, nh = tr.normalize(lo, lo + 1)
    assert tr.hv_jin(lo, lo + 1) == pytest.approx(hv_jin(nl, nh, [1.1] * 3))
    narrower = (lo, lo + 0.5)                       # same lower bounds, tighter intervals: HV_jin rises
    assert tr.reward((lo, lo + 1), narrower) > 0


def test_log_hv_jin_reward_telescopes():
    rng = np.random.default_rng(3)
    lo = rng.uniform(1, 5, size=(20, 3))
    tr = PopulationTracker(g_max=50, reward="loghvjin", normalization="fixed")
    tr.observe(lo, lo + 1)
    steps = [(lo, lo + 1), (lo * 0.9, lo * 0.9 + 0.8), (lo * 0.7, lo * 0.7 + 0.5)]
    total = tr.reward(steps[0], steps[1]) + tr.reward(steps[1], steps[2])
    assert total == pytest.approx(np.log(tr.hv_jin(*steps[2]) / tr.hv_jin(*steps[0])))


def test_convergent_action_set():
    from gat_imoea_dqn.selection import action_subset, selection_actions
    idx = action_subset("convergent")
    A = selection_actions()
    assert len(idx) == 10 and np.all(A[idx, 0] >= 0.5 - 1e-9) and len(action_subset("all")) == 28
    agent = DQNAgent(DQNConfig(n_actions=10, erb_threshold=1000))
    res = run(WF, SC, RunConfig(pop_size=12, generations=6, decision_every=2, action_set="convergent"), agent=agent)
    assert set(res.history["action"]) <= set(idx.tolist())          # recorded as full-grid indices
    with pytest.raises(ValueError):
        run(WF, SC, RunConfig(pop_size=12, generations=2), agent=DQNAgent(DQNConfig(n_actions=10)))
    rnd = run(WF, SC, RunConfig(pop_size=12, generations=5, policy="random", action_set="convergent"))
    assert set(rnd.history["action"]) <= set(idx.tolist())


def test_run_with_hv_reward():
    agent = DQNAgent(DQNConfig(erb_threshold=10, update_period=5, grad_steps=2))
    res = run(WF, SC, RunConfig(pop_size=12, generations=15, reward="hv"), agent=agent)
    assert agent.train_rounds == 2 and np.all(np.abs(res.history["reward"]) <= 1)


# ------------------------------------------------------------------- DQN --
def test_epsilon_power_decay():
    ag = DQNAgent()
    assert ag.epsilon(0) == 1.0
    assert ag.epsilon(100) == pytest.approx(0.3)
    assert ag.epsilon(200) == pytest.approx(0.09)


def test_fixed_normalization_keeps_initial_bounds():
    tr = PopulationTracker(g_max=10, normalization="fixed")
    lo = np.array([[2.0, 4.0, 1.0], [3.0, 5.0, 2.0]])
    tr.observe(lo, lo + 1)
    np.testing.assert_allclose(tr.ideal, 0.0)
    np.testing.assert_allclose(tr.nadir, [4.0, 6.0, 3.0])
    tr.observe(lo * 10, lo * 10 + 1)                    # later, worse solutions do not move the bounds
    np.testing.assert_allclose(tr.nadir, [4.0, 6.0, 3.0])
    a, b = tr.reward((lo, lo + 1), (lo * 0.5, lo * 0.5 + 0.5)), tr.reward((lo, lo + 1), (lo * 0.5, lo * 0.5 + 0.5))
    assert a == b > 0                                    # same transition, same reward at any time
    with pytest.raises(ValueError):
        PopulationTracker(g_max=10, normalization="sometimes")


def test_n_step_transitions_by_hand():
    ag = DQNAgent(DQNConfig(n_step=3, gamma=0.5, erb_threshold=10 ** 6))
    gid = ag.register_graph("m", WF)
    S = [np.full(7, k, np.float32) for k in range(5)]
    for k, r in enumerate([1.0, 2.0, 3.0, 4.0]):
        ag.store(gid, S[k], k, r, S[k + 1], k == 3)
    b = ag.buffer
    assert len(b) == 4 and not ag._pending
    np.testing.assert_allclose(b.r[:4], [1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3 + 0.25 * 4, 3 + 0.5 * 4, 4])
    np.testing.assert_allclose(b.disc[:4], [0.125, 0.125, 0.25, 0.5])
    np.testing.assert_allclose(b.gsum[:4], [1.75, 1.75, 1.5, 1.0])
    np.testing.assert_allclose(b.done[:4], [0, 1, 1, 1])
    np.testing.assert_allclose(b.s2[0], S[3])            # bootstrap from s_{t+3}
    np.testing.assert_array_equal(b.a[:4], [0, 1, 2, 3])
    # standardisation of an n-step sum = sum of standardised per-step rewards
    mean, sd = ag.reward_stats()
    assert ag.cfg.reward_scaling == "none"
    ag2 = DQNAgent(DQNConfig(n_step=3, gamma=0.5, reward_scaling="running", erb_threshold=10 ** 6))
    ag2.r_count, ag2.r_mean, ag2.r_m2 = ag.r_count, ag.r_mean, ag.r_m2
    got = ag2._scale(torch.tensor([b.r[0]]), torch.tensor([b.gsum[0]]))
    want = sum(0.5 ** k * (r - mean) / sd for k, r in enumerate([1.0, 2.0, 3.0]))
    assert float(got) == pytest.approx(want, rel=1e-5)


def test_stable_agent_runs_and_trains():
    agent = DQNAgent(DQNConfig(n_step=5, loss="huber", reward_scaling="running", eps_floor=0.1,
                               erb_threshold=10, update_period=5, grad_steps=2))
    res = run(WF, SC, RunConfig(pop_size=12, generations=20, tracker_normalization="fixed"), agent=agent)
    # the 5-step buffer lags 4 steps: 6 transitions at generation 10 (< 10), so rounds at 15 and 20 only
    assert agent.train_rounds == 2 and len(agent.buffer) == 20 and np.isfinite(agent.last_loss)
    assert len(res.front) >= 1


def test_decision_blocks_hold_the_action():
    agent = DQNAgent(DQNConfig(erb_threshold=1000))
    res = run(WF, SC, RunConfig(pop_size=12, generations=23, decision_every=5, tracker_normalization="fixed"),
              agent=agent)
    acts, rew = res.history["action"], res.history["reward"]
    assert all(acts[t] == acts[t - t % 5] for t in range(23))          # held within each block
    assert [t for t, r in enumerate(rew) if r is not None] == [4, 9, 14, 19, 22]   # last block is short
    assert agent.lifetime_steps == 5 and len(agent.buffer) == 5 and agent.buffer.done[4] == 1.0
    fixed = run(WF, SC, RunConfig(pop_size=12, generations=6, policy="fixed", decision_every=3))
    assert [r is not None for r in fixed.history["reward"]] == [False, False, True, False, False, True]


def test_epsilon_floor():
    ag = DQNAgent(DQNConfig(eps_floor=0.1))
    assert ag.epsilon(0) == 1.0 and ag.epsilon(1000) == pytest.approx(0.1)


def test_running_reward_standardization(tmp_path):
    ag = DQNAgent(DQNConfig(reward_scaling="running", erb_threshold=1, update_period=1, grad_steps=1))
    gid = ag.register_graph("m", WF)
    s = np.zeros(7, np.float32)
    rewards = np.random.default_rng(0).normal(0.01, 0.002, size=200)
    for r in rewards:
        ag.store(gid, s, 0, float(r), s, True)
    mean, sd = ag.reward_stats()
    assert mean == pytest.approx(rewards.mean()) and sd == pytest.approx(rewards.std(ddof=1))
    scaled = ag._scale(torch.tensor(rewards, dtype=torch.float32))
    assert abs(float(scaled.mean())) < 1e-4 and float(scaled.std()) == pytest.approx(1.0, abs=1e-3)
    ag.save(tmp_path / "a.pt")
    assert DQNAgent.load(tmp_path / "a.pt").reward_stats() == pytest.approx((mean, sd))
    assert DQNAgent().r_count == 0 and torch.equal(DQNAgent()._scale(torch.ones(3)), torch.ones(3))


def test_schedule_policy():
    from gat_imoea_dqn.algorithm import schedule_action
    from gat_imoea_dqn.selection import selection_actions
    A = selection_actions()
    first = A[schedule_action(((1, 0, 0), (1 / 6, 5 / 6, 0)), 0, 100, A)]
    last = A[schedule_action(((1, 0, 0), (1 / 6, 5 / 6, 0)), 99, 100, A)]
    assert np.allclose(first, [1, 0, 0]) and np.allclose(last, [1 / 6, 5 / 6, 0])
    res = run(WF, SC, RunConfig(pop_size=12, generations=8, policy="schedule"))
    acts = [A[a] for a in res.history["action"]]
    assert np.allclose(acts[0], [1, 0, 0]) and all(acts[i][0] >= acts[i + 1][0] for i in range(7))


def test_training_waits_for_buffer_and_period():
    ag = DQNAgent(DQNConfig(erb_threshold=30, update_period=10, grad_steps=2))
    gid = ag.register_graph("m", WF)
    s = np.zeros(7, np.float32)
    trained_at = []
    for i in range(60):
        ag.store(gid, s, 0, 0.0, s, False)
        if ag.maybe_train() is not None:
            trained_at.append(i + 1)
    assert trained_at == [30, 40, 50, 60]


def test_double_dqn_learns_a_bandit():
    """Action 3 always pays 1, everything else 0 (terminal transitions)."""
    ag = DQNAgent(DQNConfig(erb_threshold=1, update_period=1, grad_steps=1, lr=5e-3, seed=1))
    gid = ag.register_graph("m", WF)
    rng = np.random.default_rng(0)
    for _ in range(400):
        a = int(rng.integers(28))
        s = rng.random(7).astype(np.float32)
        ag.store(gid, s, a, float(a == 3), s, True)
    for _ in range(30):
        ag.train_round()
    q = ag.q_values(gid, rng.random(7).astype(np.float32))
    assert q.argmax() == 3 and q[3] > 0.7
    assert ag.act(gid, np.zeros(7, np.float32), 10_000, rng)[0] == 3       # epsilon ~ 0


def test_agent_save_load(tmp_path):
    ag = DQNAgent(DQNConfig(seed=3))
    gid = ag.register_graph("m", WF)
    s = np.random.default_rng(0).random(7).astype(np.float32)
    ag.save(tmp_path / "a.pt")
    other = DQNAgent.load(tmp_path / "a.pt")
    assert np.allclose(ag.q_values(gid, s), other.q_values(other.register_graph("m", WF), s))
    assert ag.state(gid, s).shape == (23,)


def test_graph_ablations():
    s = np.random.default_rng(0).random(7).astype(np.float32)
    nog = DQNAgent(DQNConfig(use_graph=False, erb_threshold=4, update_period=4, grad_steps=2))
    gid = nog.register_graph("m", WF)
    assert torch.count_nonzero(nog.embedding(gid)) == 0
    for _ in range(4):
        nog.store(gid, s, 1, 0.5, s, True)
    assert nog.maybe_train() is not None                                 # trains without the GAT
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in nog.online.encoder.parameters())

    ag = DQNAgent(DQNConfig(seed=2))
    gid = ag.register_graph("m", WF)
    with_graph = ag.q_values(gid, s)
    ag.zero_graph_at_inference = True
    zeroed = ag.q_values(gid, s)
    expect = ag.online.q(torch.zeros(1, 16), torch.from_numpy(s)[None]).detach().numpy()[0]
    assert np.allclose(zeroed, expect) and not np.allclose(zeroed, with_graph)


# -------------------------------------------------------------- run loop --
def test_full_run_learns_online():
    agent = DQNAgent(DQNConfig(erb_threshold=20, update_period=10, grad_steps=5))
    cfg = RunConfig(pop_size=16, generations=40, seed=0)
    res = run(WF, SC, cfg, agent=agent)
    h = res.history
    assert all(len(v) == 40 for v in h.values())
    assert agent.train_rounds == 3 and agent.lifetime_steps == 40         # at 20, 30, 40
    assert len(res.population) == 16 and len(res.front) >= 1
    dag = DagIndex(WF)
    assert all(is_topological(o, dag) for o in res.population.O)
    assert np.isin(res.population.X, SC.allowed_targets()).all()
    assert h["pibi"][-1] <= h["pibi"][0] + 1e-9 or np.mean(h["reward"]) > 0
    assert h["explored"][0] is True                                       # epsilon(0) = 1
    # a second run keeps learning with the same agent (counter carries over)
    run(WF, SC, cfg, agent=agent)
    assert agent.lifetime_steps == 80 and agent.train_rounds == 7


@pytest.mark.parametrize("policy", ["random", "fixed"])
def test_baseline_policies(policy):
    res = run(WF, SC, RunConfig(pop_size=12, generations=10, policy=policy))
    assert len(res.history["action"]) == 10 and len(res.front) >= 1
