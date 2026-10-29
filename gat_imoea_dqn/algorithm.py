"""The GAT-IMOEA-DQN optimisation loop (PDF section 3, steps 1-6).

    result = run(wf, sc, RunConfig(generations=300), agent=DQNAgent())

Per generation t:
  6  mate P_t -> offspring O_t (interval / uncertainty fronts, DAEC + POX, mutation)
     evaluate O_t (interval critical-path calculus)
  2  state s_t = [GAT(G) || 7 population features of P_t]
  3  a_t = epsilon-greedy argmax Q(s_t, .) -> (alpha, beta, gamma)
  4  P_{t+1} = three-tier selection from P_t U O_t
  5  r_t = relative PIBI improvement, store (s_t, a_t, r_t, s_{t+1}), train when due

policy="dqn" (default) learns online as above. Baselines: "random" picks a
uniform action each generation, "fixed" always uses `fixed_action`, and
"schedule" moves linearly from `schedule[0]` to `schedule[1]` over the run
(default: pure convergence -> diversity-heavy), snapped to the action grid.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .chromosome import DagIndex, Population, allowed_for, init_population
from .dqn import DQNAgent
from .evaluator import EvaluatorConfig, IntervalEvaluator
from .intervals import interval_dominance_matrix, nondominated_fronts
from .operators import VariationConfig, make_offspring
from .selection import SelectionConfig, action_subset, environmental_selection, selection_actions
from .state import PopulationTracker


@dataclass(frozen=True)
class RunConfig:
    pop_size: int = 105
    generations: int = 300
    seed: int = 0
    policy: str = "dqn"                         # "dqn", "random", "fixed" or "schedule"
    fixed_action: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    # schedule: linear in t / G from the first action to the second, snapped to the 28-action grid
    schedule: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((1.0, 0.0, 0.0),
                                                                              (1 / 6, 5 / 6, 0.0))
    learn: bool = True                          # dqn: store transitions and train
    greedy: bool = False                        # dqn: no exploration (evaluation runs)
    reward: str = "pibi"                        # "pibi" (PDF) or "hv" (hypervolume improvement)
    tracker_normalization: str = "running"      # "running" or "fixed" (bounds from the initial population)
    # dqn: pick an action every `decision_every` generations and hold it in between; one
    # transition per decision, its reward the relative improvement over the whole block
    # and its state change features measured over the block (1 = the PDF's per-generation loop)
    decision_every: int = 1
    # dqn / random: choose among "all" 28 actions or the 10 "convergent" ones (alpha >= 1/2);
    # history["action"] always records the index into the full 28-action grid
    action_set: str = "all"
    evaluator: EvaluatorConfig = EvaluatorConfig()
    variation: VariationConfig = VariationConfig()
    selection: SelectionConfig = SelectionConfig()


@dataclass
class RunResult:
    population: Population
    F_lo: np.ndarray
    F_hi: np.ndarray
    front: np.ndarray                           # indices of the first interval front
    history: Dict[str, List] = field(default_factory=dict)
    seconds: float = 0.0

    def front_objectives(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.F_lo[self.front], self.F_hi[self.front]


def schedule_action(schedule, t: int, generations: int, actions: np.ndarray) -> int:
    """Index of the grid action nearest to start + (t / (G - 1)) * (end - start)."""
    start, end = np.asarray(schedule[0], float), np.asarray(schedule[1], float)
    p = t / max(generations - 1, 1)
    target = (1 - p) * start + p * end
    return int(np.argmin(np.abs(actions - target).sum(axis=1)))


def run(wf, sc, cfg: RunConfig = RunConfig(), agent: Optional[DQNAgent] = None,
        graph_key: Optional[str] = None,
        callback: Optional[Callable[[int, Dict], None]] = None) -> RunResult:
    if cfg.policy not in ("dqn", "random", "fixed", "schedule"):
        raise ValueError(f"unknown policy {cfg.policy!r}")
    t_start = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    dag = DagIndex(wf)
    ev = IntervalEvaluator(wf, sc, cfg.evaluator)
    allowed = allowed_for(wf, sc)
    actions = selection_actions()
    subset = action_subset(cfg.action_set)
    tracker = PopulationTracker(cfg.generations, reward=cfg.reward, normalization=cfg.tracker_normalization)
    N = cfg.pop_size

    gid = None
    if cfg.policy == "dqn":
        agent = agent if agent is not None else DQNAgent()
        agent.begin_episode()
        if agent.cfg.n_actions != len(subset):
            raise ValueError(f"agent has {agent.cfg.n_actions} actions, action set {cfg.action_set!r} has {len(subset)}")
        gid = agent.register_graph(graph_key or wf.name, wf)

    pop = init_population(wf, sc, N, rng, dag)                             # step 1
    F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
    tracker.observe(F_lo, F_hi)
    s = tracker.features(pop.X, F_lo, F_hi, 0)

    hist: Dict[str, List] = {k: [] for k in ("action", "explored", "reward", "pibi", "epsilon", "loss",
                                             "best_mid", "median_mid")}
    K = max(int(cfg.decision_every), 1)
    block_F = (F_lo, F_hi)                                                 # population at the block start
    a, explored = -1, False
    for t in range(cfg.generations):
        off, _ = make_offspring(pop, F_lo, F_hi, dag, allowed, rng, cfg.variation)   # step 6
        G_lo, G_hi = ev.evaluate_population(off.X, off.O)
        tracker.observe(G_lo, G_hi)

        if t % K == 0:
            block_F = (F_lo, F_hi)
        if cfg.policy == "dqn":                                            # steps 2-3
            if t % K == 0:
                a, explored = agent.act(gid, s, t, rng, greedy=cfg.greedy)
            else:
                explored = False
            action = actions[subset[a]]
        elif cfg.policy == "random":
            a = int(subset[rng.integers(len(subset))])
            action = actions[a]
        elif cfg.policy == "schedule":
            a = schedule_action(cfg.schedule, t, cfg.generations, actions)
            action = actions[a]
        else:
            a, action = -1, np.asarray(cfg.fixed_action, float)

        merged = Population.concat(pop, off)                               # step 4
        M_lo, M_hi = np.concatenate([F_lo, G_lo]), np.concatenate([F_hi, G_hi])
        keep, _ = environmental_selection(M_lo, M_hi, N, action, rng, genotypes=(merged.X, merged.O),
                                          cfg=cfg.selection)
        new_pop, nF_lo, nF_hi = merged[keep], M_lo[keep], M_hi[keep]

        loss = None
        last = t == cfg.generations - 1
        if K == 1:
            r = tracker.reward((F_lo, F_hi), (nF_lo, nF_hi))               # step 5
            s2 = tracker.features(new_pop.X, nF_lo, nF_hi, t + 1)
            if cfg.policy == "dqn" and cfg.learn:
                agent.store(gid, s, a, r, s2, last)
                loss = agent.maybe_train()
        elif (t + 1) % K == 0 or last:                                     # end of a decision block
            r = tracker.reward(block_F, (nF_lo, nF_hi))
            s2 = tracker.features(new_pop.X, nF_lo, nF_hi, t + 1)
            if cfg.policy == "dqn" and cfg.learn:
                agent.store(gid, s, a, r, s2, last)
                loss = agent.maybe_train()
        else:
            tracker.observe(nF_lo, nF_hi)
            r, s2 = None, s

        pop, F_lo, F_hi, s = new_pop, nF_lo, nF_hi, s2
        mid = (F_lo + F_hi) / 2
        rec = {"action": int(subset[a]) if cfg.policy == "dqn" else a, "explored": explored, "reward": r, "pibi": tracker.pibi(F_lo, F_hi),
               "epsilon": agent.epsilon(t) if cfg.policy == "dqn" else None, "loss": loss,
               "best_mid": mid.min(axis=0).tolist(), "median_mid": np.median(mid, axis=0).tolist()}
        for k, v in rec.items():
            hist[k].append(v)
        if callback:
            callback(t, rec)

    front = nondominated_fronts(interval_dominance_matrix(F_lo, F_hi))[0]
    return RunResult(population=pop, F_lo=F_lo, F_hi=F_hi, front=np.sort(front), history=hist,
                     seconds=time.perf_counter() - t_start)
