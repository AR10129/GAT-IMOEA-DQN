# GAT-IMOEA-DQN

Interval multi-objective DAG workflow offloading via graph-attentive
evolutionary optimization. The design is in `GAT-IMOEA-DQN_Workflow.pdf`.

| Part | Where | Status |
|---|---|---|
| Datasets: 40 interval workflows and the environment scenario | `datasets/`, `dataset_tools/` (see `datasets/README.md`) | done |
| Interval critical-path evaluator (F1 makespan, F2 energy, F3 cost, U1/U2) | `gat_imoea_dqn/evaluator.py`, `gat_imoea_dqn/intervals.py` | done |
| Two-string chromosome, init, POX / DAEC, mutation, repair, step-6 mating | `gat_imoea_dqn/chromosome.py`, `gat_imoea_dqn/operators.py` | done |
| Interval and uncertainty fronts (possibility-degree dominance, UDR) | `gat_imoea_dqn/intervals.py` | done |
| Three-tier environmental selection (Rank C / D / U, ICD) and the 28 Das-Dennis actions | `gat_imoea_dqn/selection.py` | done |
| GAT encoder, 7 population features, 23-dim state, PIBI reward | `gat_imoea_dqn/gat.py`, `gat_imoea_dqn/state.py` | done |
| DQN (ε-greedy, replay buffer, Double DQN) and the full loop | `gat_imoea_dqn/dqn.py`, `gat_imoea_dqn/algorithm.py` | done |
| Experiments: training across workflows, baselines, comparison | `scripts/train.py`, `scripts/run.py`, `scripts/compare.py` | tooling done |

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest tests -q
```

## Evaluator

```python
import numpy as np
from dataset_tools.loader import load_workflow, load_scenario
from gat_imoea_dqn import IntervalEvaluator, uncertainty_metrics

wf = load_workflow("datasets/workflows/montage/montage_200.json")
sc = load_scenario("datasets/scenarios/default.yaml")
ev = IntervalEvaluator(wf, sc)

rng = np.random.default_rng(0)
X = rng.choice(sc.allowed_targets(), size=wf.num_tasks)   # allocation string
O = wf.random_topological_order(rng)                       # priority string
res = ev.evaluate(X, O, detail=True)
res.F_lo, res.F_hi        # [makespan s, energy J, cost $] lower/upper bounds
res.breakdown, res.FT     # energy/cost parts, per-task finish-time intervals
F_lo, F_hi = ev.evaluate_population(Xs, Os)                # (P, 3) each
U1, U2 = uncertainty_metrics(F_lo, F_hi)
```

On a 2000-task workflow, a 210-individual population takes 0.15–0.27 s with
numba and about 1.5–2 s without it (`scripts/bench_evaluator.py`).

### Modelling choices beyond the PDF

The PDF leaves these points open. Each is either a config switch
(`EvaluatorConfig`) or a scenario value.

- **Queues.** The device runs one task at a time, in priority order O. The
  cloud has no queue ("capacity per task"). Edge servers share their CPU as
  f_u / n_u and run those tasks in parallel, without also queueing them, since
  the share already models contention. `edge_queue=True` makes edge servers
  queue as well (the literal reading of eq. 1).
- **External data.** `external_io="pdf"` (the default) follows the PDF: entry
  tasks are ready at time 0 and files no task produces are ignored.
  `external_io="device"` uploads an offloaded task's external inputs from the
  device before it starts, and returns its external outputs. This matters
  mostly for CyberShake, whose entry tasks read about 40 GB.
- **Upload energy.** The p_i·t_upload term counts the time the device spends
  sending data, either parent outputs or external inputs, to offloaded tasks.
- **Cloud energy.** κ_C·f_C²·ΣC + P_static_C·max FT, mirroring the edge term.
- **Cost.** η_E·f_eff·t_comp uses plain interval arithmetic, as written in the
  PDF. Because f_eff appears twice, the cost interval comes out wider than the
  workload interval alone would give.
- **Device ↔ cloud and edge → device links.** These are defined in
  `datasets/scenarios/default.yaml` (relay through the nearest edge server;
  downlink rate).

## Chromosome and variation

```python
from gat_imoea_dqn import DagIndex, init_population, make_offspring

dag = DagIndex(wf)
pop = init_population(wf, sc, 105, rng, dag)          # X: reachable nodes, O: random Kahn
F_lo, F_hi = ev.evaluate_population(pop.X, pop.O)
off, info = make_offspring(pop, F_lo, F_hi, dag, sc.allowed_targets(), rng)
# info: interval/uncertainty front ranks, elite mask, p_c, partners
```

`make_offspring` follows Step 6 of the PDF:

1. Interval fronts split the population into TopPI and BottomPI. TopPI mates
   only inside TopPI; BottomPI picks partners from the whole population.
2. Uncertainty fronts on (U1, U2) set the crossover probability: p_c = 0.7 for
   the top half and 0.9 for the bottom half.
3. Each parent produces one child: DAEC on the allocation string, POX on the
   priority string, then mutation.

On a 2000-task workflow, variation plus evaluation of the 105 children takes
about 0.15 s per generation.

### Operator choices beyond the PDF

All of these are set in `VariationConfig`.

- **Interval dominance.** i ≺_IN j when P(F_m(i) ≤ F_m(j)) ≥ 0.5 for every
  objective and > 0.5 for at least one, where P is the standard
  possibility degree of interval numbers. The threshold is set by
  `dominance_threshold`. The relation need not be transitive, so the front
  sorting has a guard against cycles.
- **DAEC gene operator.** This is uniform crossover: each gene comes from the
  better parent with probability `elite_bias`, and the default 0.5 means no
  bias. The "elite" and "dual-adaptive" parts are the TopPI-only mating and the
  front-based p_c. Raise `elite_bias` to bias genes toward the better parent.
- **POX.** This is the mask-based precedence-preserving crossover of Bierwirth
  et al. (1996), which always yields a valid topological order.
  `pox_mode="one_point"` takes a prefix from one parent instead.
- **Mutation.** Each allocation gene is resampled to another reachable node
  with rate 1/K. Each task moves with rate 1/K to a random position between its
  last parent and first child, which keeps the order valid.
- **Repair.** The operators never produce invalid children, but every child is
  checked. If needed it is repaired by Kahn's algorithm, ordered by each task's
  current position, plus re-drawing any unreachable allocation genes.
- **Ties at the half-way cut** are broken at random.

## Environmental selection

```python
from gat_imoea_dqn import Population, environmental_selection, selection_actions

A = selection_actions()                                   # (28, 3) rows (alpha, beta, gamma)
merged = Population.concat(pop, off)                      # P_M, size 2N
M_lo, M_hi = np.concatenate([F_lo, G_lo]), np.concatenate([F_hi, G_hi])
keep, info = environmental_selection(M_lo, M_hi, 105, A[a_t], rng, genotypes=(merged.X, merged.O))
pop, F_lo, F_hi = merged[keep], M_lo[keep], M_hi[keep]    # P_{t+1}
# info.R_C, info.ICD, info.R_U, info.tiers["C" / "D" / "U"]
```

A full generation on a 2000-task workflow takes about 0.2 s: 47 ms variation,
122 ms evaluation and 34 ms selection.

### Selection choices beyond the PDF

All of these are set in `SelectionConfig`.

- **Scaling.** Objectives are scaled to [0, 1] over the merged pool before the
  crowding distance and U1/U2 are computed. Raw values would let energy
  (about 10⁶ J) swamp makespan (s) and cost ($). The mating step uses the same
  scaling (`VariationConfig.normalize`). Interval dominance is unaffected by
  scale and uses raw values.
- **Nearest neighbours.** For the crowding distance these are the
  k = round(√|P_M|) other solutions with the smallest pairwise ICD, i.e. the
  most similar ones. ICD(a) is the sum over them.
- **Counts.** n_C = round(αN), n_D = round(βN) and n_U = N − n_C − n_D.
- **Clones.** Exact copies (the same X and O) of an individual already chosen
  are skipped while other candidates remain (`dedupe_genotypes`).
- **Ties** inside a ranking are broken at random.

## Full algorithm

```bash
# train one agent (GAT + DQN) across workflows, saved to checkpoints/agent.pt
.venv/Scripts/python scripts/train.py --workflows montage_100 sipht_100 --episodes 10 --generations 300
# run on a workflow: online learning from scratch, a trained agent, or a baseline
.venv/Scripts/python scripts/run.py --workflow montage_200
.venv/Scripts/python scripts/run.py --workflow montage_200 --agent checkpoints/agent.pt --greedy
.venv/Scripts/python scripts/run.py --workflow montage_200 --policy random
.venv/Scripts/python scripts/run.py --workflow montage_200 --policy fixed --action 0.5 0.5 0
# score several result files on one shared scale
.venv/Scripts/python scripts/compare.py results/montage_200_*.json
# full experiment: train agents, then compare policies on held-out workflows, in parallel
.venv/Scripts/python scripts/experiment.py --name main --policies trained fresh random fixed --train-seeds 0
# ablation: + fixed_fav, a no-GAT agent, graph zeroed at test time, 3 training seeds
.venv/Scripts/python scripts/experiment.py --name ablation --reuse-from main --workers 12
```

`experiment.py` writes `results/experiments/<name>/summary.md`. It reports, per
policy, workflow and training seed:

- PIBI
- the hypervolume of the front midpoints and of the upper bounds (worst case)
- U1
- paired per-seed wins and mean ranks

It resumes if interrupted, and `--reuse-from` copies finished runs from
another experiment.

```python
from gat_imoea_dqn.algorithm import RunConfig, run
from gat_imoea_dqn.dqn import DQNAgent
res = run(wf, sc, RunConfig(generations=300), agent=DQNAgent())
res.front_objectives(); res.population; res.history   # actions, rewards, PIBI, loss, ...
```

Each generation goes: mate (step 6) → evaluate the offspring → build the state
(step 2) → the ε-greedy DQN picks (α, β, γ) (step 3) → three-tier selection
(step 4) → PIBI reward, replay buffer and Double DQN training when due (step 5).

### Learning choices beyond the PDF

- **GAT.** Each node aggregates over its parents, its children and itself
  (self-loops get a zero edge feature). Hidden size 32, 4 heads, ELU
  activations, and gated attention pooling down to 16 dimensions. The GAT is
  trained end to end through the DQN's TD loss. The replay buffer stores a
  graph key, so one agent can learn across workflows.
- **The 7 population features.** The PDF names them but doesn't define them; the
  exact definitions are in `state.py`.
  - Objectives are normalised with running bounds (the ideal and nadir over the run).
  - c1 and c2 are distances between the population centroid and the cluster
    centroids, measured on interval midpoints.
  - d1 counts clusters of allocation strings: two strings are in the same
    cluster when they differ in at most 10% of genes.
  - d2, u1 and u2 are relative changes of the form (now − before) / (|now| + |before|),
    so they stay in [−1, 1].
- **PIBI.** This is MOEA/D's PBI (θ = 5) on the normalised lower and upper
  objective vectors, averaged over the two bounds, taking the best solution per
  Das-Dennis direction and then the mean over directions. Lower is better.
  - The reward is (PIBI_t − PIBI_{t+1}) / PIBI_t, clipped to [−1, 1]. The PDF writes
    (PIBI_{t+1} − PIBI_t) / PIBI_t; I flipped the sign so that an improvement
    (a smaller PIBI) earns a positive reward, matching "relative improvement".
- **HV reward (option, not in the PDF).** `RunConfig(reward="hv")` rewards the
  relative improvement in (HV(midpoints) + HV(upper bounds)) / 2 on normalised
  objectives (reference 1.1). In `results/experiments/hv` it did worse than the
  PIBI reward, even on hypervolume, and varied more across training seeds.
- **Learner fixes (options, off by default).**
  `DQNConfig(reward_scaling="running", eps_floor=0.1)` trains on rewards
  standardised by their running mean and standard deviation, and keeps ε ≥ 0.1.
  `results/diagnostics/main` shows why: the action strongly drives the
  reward, but the unscaled Q-values barely separated the actions.
- **Schedule baseline.** `RunConfig(policy="schedule")` moves linearly from
  (1, 0, 0) to (1/6, 5/6, 0) over the run, snapped to the 28-action grid.
- **ε schedule.** ε_t = 0.3^(t/100), so ε is 1 at t = 0, 0.3 at t = 100 and
  0.09 at t = 200. The PDF's "ϵt= 0.3t/100" is read as a power. ε restarts in
  each run.
- **DQN.** An MLP (23 → 64 → 64 → 28) trained with Adam (lr 1e-3), γ = 0.9,
  batch 32 and 50 gradient steps per training round. The target network is
  hard-synced after each round, gradients are clipped at norm 10, and the
  buffer holds 10 000 transitions. The PDF's buffer threshold (250) and update
  period (every 50 generations) count over the agent's lifetime, so they carry
  across runs.

## Layout

```
dataset_tools/     dataset generation (Pegasus port, WfCommons), validation, loader
gat_imoea_dqn/     the framework: evaluator, intervals, chromosome, operators, selection,
                   gat, state (features + PIBI), dqn, algorithm (the full loop)
datasets/          workflows, scenarios, manifest, legacy files
scripts/           generate/validate datasets, bench_evaluator, train, run, compare
tests/             pytest suite (hand-checked evaluator cases)
```
