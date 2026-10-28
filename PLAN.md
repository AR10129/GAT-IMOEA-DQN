# Paper plan

**Target:** Engineering Applications of Artificial Intelligence (EAAI), the venue of
Jin et al. (2026), "Reinforcement learning-driven interval multi-objective
evolutionary algorithm for task offloading in uncertain cloud–edge", EAAI 178,
115043. Backups: Swarm and Evolutionary Computation, Applied Soft Computing,
Expert Systems with Applications.

**Working title:** Interval multi-objective offloading of dependent-task workflows
in uncertain cloud–edge environments with learned, structure-aware environmental
selection.

**The gap:** Jin et al. treat every task as an independent offloading unit
(their §3.3) and sum per-task latency. We model applications as DAGs: precedence,
inter-task data transfer, and makespan along the interval critical path.

**Guiding rule:** keep everything except the DAG extension as close to Jin et al.
as possible (parameters, metrics, protocol, baselines), so reviewers can compare
the two papers directly. Anything beyond that is a sensitivity study or a stated
limitation.

## Contributions

| # | Contribution | Status |
|---|---|---|
| C1 | Interval critical-path model for DAG workflows in the cloud–edge–end architecture (transfer, makespan, energy, cost) | built; needs Jin-aligned parameters |
| C2 | DAG-valid two-string encoding: allocation + topological priority, POX, shift mutation, topological repair | built |
| C3 | RL-driven environmental selection trained across workflows and reused, not relearned per run | built; must be re-tested at 1000 generations |
| C4 | Structure-aware state: dynamic GAT, or critical-path descriptors if the GAT fails gate L6 | to do |
| C5 | Released benchmark: 40 interval WfCommons / Pegasus workflows + generator, and Suite A instances | Suite B built; Suite A to do |
| C6 | Evaluation stronger than the predecessor's: their 5 baselines + IMOEA-DQN + NSGA-II + tuned fixed action; 30 runs; Holm-corrected tests | to do |

The GAT is a headline claim only if it passes L6. Otherwise it is reported as an
ablation and the paper stands on C1–C3, C5 and C6.

## Benchmark suites

**Suite A — Jin-scale, many devices, small application DAGs** (directly comparable to Jin et al.)

| Scale | Edge servers | Tasks | Mobile devices |
|---|---|---|---|
| Small | 3 | 100 | 20 / 50 |
| Basic | 6 | 300 | 60 / 100 |
| Medium | 10 | 500 | 100 / 250 |

Each device submits small application DAGs from a standard layered random-DAG
generator (fan-out, depth and CCR as generator parameters). Task and data sizes
follow Jin et al.: 1–5 Gcycles, 500–3000 KB.

**Suite B — scientific workflows** (realism and scalability)

The 40 existing WfCommons / Pegasus workflows (5 families × 60–2000 tasks), one
originating device, with workloads and data mapped into Jin et al.'s ranges
(order-preserving log mapping). Test on 100 / 400 / 1000 tasks, plus 2000 for
scalability.

## Parameters and protocol (from Jin et al., Table 1 and §5.1)

| Item | Value |
|---|---|
| Device CPU | {[0.45,0.59], [0.54,0.71], [0.63,0.82], [0.72,0.94], [0.81,1.06], [0.9,1.18]} GHz, cycled over devices |
| Edge aggregate CPU | {[78.5,90], [102,120], [127.5,150]} GHz, cycled; shared equally by the tasks placed on it |
| Cloud CPU per task | [18, 22] GHz |
| Wireless bandwidth B | [9, 12] MHz; channel gain 0.001; noise −100 dBm; transmit power {0.1, 0.2, 0.4, 0.5} W; idle power 0.02 W |
| Wired edge–cloud W_C | [408, 624] Mbps |
| Prices | η_E 0.001 $/(GHz·s), η_C 0.01 $/s |
| Not in Jin et al. (our choice, documented) | edge→device downlink = the uplink rate; edge↔edge = W_C; per-cycle device energy as in our scenario |
| Algorithm | N = 105, G_max = 1000, buffer threshold 250, DQN updated every 50 generations, p_c 0.7 / 0.9 |
| Metrics | IGD (midpoints, pooled reference front), HV (average of lower- and upper-bound HV), Imprecision; extras: PIBI, worst-case HV, U1/U2 |
| Statistics | 30 runs; Wilcoxon rank-sum with Holm correction (+ / − / = tables as in Jin et al.); Friedman ranks; Vargha–Delaney A12 |

**Open decision: energy scope.** Jin et al. count device-side energy only. Our
PDF adds edge and cloud energy. Plan: a switch, with device-only as the main
setting for comparability and three-tier energy as a sensitivity study.

## Checklist (in order; each phase depends on the one before)

### P0 — Aligned model
- [x] R0. Project under git (tag `pre-paper`); still to do: every result folder records its commit
- [x] M1. Jin et al. scenario (`datasets/scenarios/jin.yaml`) + fixed channel gain + downlink = uplink option
- [x] M2. Load-time rescaling into Jin et al.'s ranges (`dataset_tools/rescale.py`) + data multiplier for CCR sweeps
- [x] M3. Energy-scope switch (`EvaluatorConfig(energy_scope="device" | "all")`)
- [x] M4. Re-measured with `scripts/comm_share.py` (five 200-task workflows, 100 generations):

  | Data multiplier | Fast-end makespan increase from communication | Median energy increase |
  |---|---|---|
  | ×1 (Jin et al.'s exact sizes) | 0.4–6.7% | 0.5–1.7% |
  | ×20 | 15–116% | 4.5–23% |
  | ×100 | 33–181% | 29–82% |

  Decision: CCR levels = data ×1 / ×20 / ×100 (low / medium / high). At ×1 the DAG model
  differs from Jin et al. mainly through precedence and makespan, not transfer time;
  say so in the paper. Magnitudes are now realistic: makespans of 3–180 s, energy of
  30–160 J, costs of a few cents.
- [x] M5. Multi-device, multi-DAG evaluator: one internal node and queue per device; gene 0 = the
  task's own device; per-device CPU / energy / power / range; `makespan="sum_apps"` (sum of
  application makespans) and `device_queue` switches. Tested to reduce EXACTLY to Jin et al.'s
  eqs. (2)-(20) for single-task applications without a device queue (`tests/test_suite_a.py`).
  Device->cloud uploads now cost transmit power on the wireless leg and idle power on the wired
  leg (Jin et al. eq. 11); L0 / L34 used transmit power on both legs (small energy difference).
- [x] M6. Suite A: `scripts/generate_suite_a.py` → `datasets/suite_a/` (24 instances: Jin et al.'s
  6 configurations × i0 test + i1–i3 train/val) and `datasets/scenarios/suite_a/`. One layered
  random DAG per device; C ∈ {1..5} Gcycles, entry inputs and edge data U[500, 3000] KB, crisp
  tasks (uncertainty from resources, as in Jin et al.). Run with
  `--external-io device --makespan sum_apps --energy-scope device`.
- [ ] B0. Reference heuristics: all-local, all-edge, all-cloud, HEFT-style list scheduling

### P1 — Core claims
- [x] L0. Re-run trained vs fresh vs fixed at G_max = 1000 under Jin et al.'s metrics (risk check for C3).
  `results/experiments/l0` (Jin scenario, data ×20, device energy, five 200-task workflows,
  10 paired seeds, agents averaged over 3 training seeds; Wilcoxon +/−/= over the 5 workflows):

  | Policy | Mean IGD rank | IGD | HV (Jin) | vs fresh: IGD · HV | vs fixed_fav: IGD · HV · Imprecision |
  |---|---|---|---|---|---|
  | trained (GAT + fixes) | 1.82 | 0.051 | 0.835 | 4/0/1 · 5/0/0 | 3/1/1 · 3/0/2 · 5/0/0 |
  | fixed_fav (5/6, 1/6, 0) | 2.40 | 0.053 | 0.803 | 5/0/0 · 3/0/2 | reference |
  | fixed (⅓, ⅓, ⅓) | 3.10 | 0.064 | 0.813 | 4/0/1 · 4/0/1 | 0/2/3 · 1/0/4 · 5/0/0 |
  | trained_nogat (+ fixes) | 3.36 | 0.065 | 0.822 | 4/0/1 · 4/0/1 | 0/3/2 · 3/0/2 · 5/0/0 |
  | fresh (Jin-style online) | 4.72 | 0.168 | 0.764 | reference | 0/5/0 · 0/3/2 · 5/0/0 |
  | random | 5.60 | 0.221 | 0.736 | 0/0/5 · 0/1/4 | 0/5/0 · 0/5/0 · 5/0/0 |

  Verdict: C3 risk retired. Even with 1000 generations, the fresh online agent is
  significantly worse than every policy except random on IGD. The trained GAT agent
  leads overall, and for the first time beats fixed_fav (IGD 3 of 5 workflows, HV 3 of 5,
  never worse on HV). Still open:
  - training-seed variance: no-GAT seed 1 collapsed (beats fixed_fav 0/50 runs), which drags the no-GAT average down; L3/L4 must fix this before the GAT-vs-no-GAT claim is safe;
  - Imprecision is a sum over the front, so it grows with front size (fixed_fav keeps 90 solutions); also report it per solution;
  - training reward still barely rises (0.0009 → 0.0010).
- [ ] L1. Dynamic structure-aware state (per-task placement shares, critical-path frequency, slack)
- [ ] L2. Training distribution: about 250 graphs; leave-one-family-out tests
- [x] L3. Stable reward: fixed per-run normalisation (ideal 0, nadir from the initial population), 5-step returns
- [x] L4. Learner tuning: Huber loss, lr 5e-4, 50k buffer, 40 episodes on 10 graphs (*_60 + *_100),
  validation on the five *_400 workflows every 10 episodes keeping the best checkpoint.
  `results/experiments/l34` (same setting as L0, 5 training seeds per agent, 10 paired seeds):

  | Policy | Mean IGD rank | IGD | HV (Jin) | vs fixed_fav: IGD · HV · Imprecision | Worst seed vs fixed_fav |
  |---|---|---|---|---|---|
  | no-GAT stable | 1.46 | 0.032 | 0.855 | 4/0/1 · 4/0/1 · 5/0/0 | 14/50 runs (seed 0) |
  | GAT stable | 1.84 | 0.035 | 0.847 | **5/0/0 · 5/0/0 · 5/0/0** | 32/50 runs (seed 3) |
  | fixed_fav | 3.04 | 0.048 | 0.802 | reference | — |
  | fixed (⅓, ⅓, ⅓) | 3.98 | 0.074 | 0.811 | 0/4/1 · 1/0/4 · 5/0/0 | — |
  | fresh (Jin-style online) | 5.08 | 0.199 | 0.762 | 0/5/0 · 0/3/2 · 5/0/0 | — |

  - Stabilisation works: GAT stable beats L0's GAT agent on IGD and HV in 5 of 5 workflows
    (and both L0 agents likewise).
  - No collapse: all 10 seeds beat the equal mix in ≥ 34 of 50 runs and the fresh agent in ≥ 42 of 50.
    The weakest (no-GAT seed 0) beats fixed_fav in only 14 of 50, against 0 of 50 for L0's collapsed seed.
  - GAT vs no-GAT (Wilcoxon, GAT as reference): no-GAT better on IGD in 2 workflows, worse in 1,
    tied in 2; HV 3/1/1. No consistent GAT advantage; the GAT is only more even across seeds.
  - Imprecision per solution: stable agents 0.67–0.69 vs fixed_fav 0.81 (fresh 0.53, but far worse IGD).
  - Validation score rises during training (≈ 0.65 → 0.70 for every seed): use it as the learning
    curve in the paper. The per-step training reward stays flat (≈ 0.0005).
  - Validation on *_400 at 300 generations is an imperfect proxy: no-GAT seed 0 had the best validation
    score but the weakest test result.
- [ ] L6. Gate: Q-ranking Spearman ≥ 0.6, greedy regret ≤ 30% of random's, GAT beats no-GAT
- [ ] B1. Re-implement IP-MOEA, IMOEA/D, IMOEA/D-AWN, IMEA-IED, IMaOEACI (same encoding and evaluator)
  - [x] IP-MOEA and IMOEA/D in `gat_imoea_dqn/baselines.py`; experiment policies `ip_moea`, `imoead`
    (same chromosome, initial population, operators, evaluator and evaluation budget as ours).
  - **Issue — IP-MOEA:** under its strict box relation the whole population is one front in our
    setting, so it has no convergence pressure (median makespan 159 → 155 s in 100 generations on
    small_20, vs 60 s for IMOEA/D). Faithful to the relation, but a reviewer may call it a
    strawman: check the original paper's tie-breaking / crowding before the final runs.
  - **Risk — IMOEA/D:** in a 40-generation smoke test on Suite A (2 seeds) it beat our selection with
    fixed_fav clearly (IGD 0.023 vs 0.080; HV 1.00 vs 0.86). Must be checked at G_max = 1000 against
    the trained agents before building more baselines.
  - **Result — Suite A at G_max = 1000** (`results/experiments/suite_a`, 2026-10-28; 6 i0 instances
    × {×1, ×20}; 10 eval seeds; 3 training seeds per agent, trained on i1/i2, validated on i3; summaries
    `summary_x1.md`, `summary_x20.md`). **The risk is confirmed: IMOEA/D wins IGD and HV on all 12
    cases (Wilcoxon −, 12/12 for both agents).**

    | ×1 / ×20 | IGD | HV_jin | Imprecision | mean rank |
    |---|---|---|---|---|
    | imoead | 0.017 / 0.019 | 0.99 / 0.98 | 20.9 / 26.2 | 1.0 / 1.0 |
    | trained_stable (GAT) | 0.058 / 0.067 | 0.92 / 0.90 | 16.5 / 20.1 | 2.9 / 2.8 |
    | trained_nogat_stable | 0.063 / 0.067 | 0.89 / 0.87 | 16.2 / 19.6 | 3.1 / 2.9 |
    | fixed_fav | 0.064 / 0.086 | 0.79 / 0.77 | 22.5 / 25.4 | 3.3 / 3.6 |
    | fresh | 0.150 / 0.154 | 0.86 / 0.83 | 10.0 / 14.4 | 4.8 / 4.7 |
    | ip_moea | 0.274 / 0.287 | 0.62 / 0.62 | 23.1 / 27.7 | 5.9 / 5.9 |

    - What holds: the GAT agent beats fixed_fav (HV on 12/12, IGD on 8/12; fixed_fav wins IGD on
      small_20/small_50 at both CCRs), fresh and IP-MOEA; no training seed collapsed; GAT > no-GAT on HV (12/12),
      tie on IGD.
    - Imprecision "+" vs IMOEA/D is weak evidence: it is a sum over the front, and our fronts are
      smaller (≈ 75–90 vs ≈ 100 solutions).
    - fixed_fav also loses by ≈ 4×, so the gap is in the IMOEA backbone (α/β/γ selection and
      generational mating), not in the learned controller. IMOEA/D's steady-state neighbourhood
      replacement converges much faster under the same operators and budget.
    - Jin et al. report IMOEA-DQN beating IMOEA/D; either our backbone is weaker than theirs or our
      IMOEA/D is stronger than theirs. This must be resolved before any more baselines or the 30-run
      protocol. Options: (a) audit our Step 4/6 against Jin et al. for missing convergence pressure;
      (b) hybridise — let the DQN control a decomposition-based backbone; (c) reframe the claim as
      adaptive control on a fixed backbone and drop IMOEA/D from the headline table (weak for EAAI).
  - [ ] IMOEA/D-AWN, IMEA-IED, IMaOEACI
- [ ] B2. Re-implement IMOEA-DQN (Jin et al.) on the DAG model
- [ ] B3. NSGA-II; tuned fixed-action baseline; ablations (UDR1, UDR2, DAEC, DQN-ES, GAT, POX, static vs dynamic state)
- [ ] B4. Equal budget for all methods; wall-clock times reported

### P2 — Evaluation
- [ ] E1. Suite A (3 scales × 2 device counts) and Suite B (5 families × 3 sizes × 3 CCR)
- [ ] E2. 30 runs per method and instance
- [ ] E3. Metrics as above, normalised once per instance
- [ ] E4. Statistics as above
- [ ] E5. IGD-vs-generation curves; (α, β, γ) traces per family
- [ ] E6. Sensitivity: CCR, uncertainty width, edge count, bandwidth, energy scope, multi-core queueing
- [ ] E7. Monte Carlo check of final fronts under sampled crisp values

### P3 — Paper
- [ ] R1. One script builds every table and figure (LaTeX)
- [ ] R2. System-model figure, pseudocode, complexity, parameter table
- [ ] R3. Threats to validity and limitations (no link contention, no receive energy, interval bounds assumed to hold)

## Risks

- At 1000 generations an online agent may catch up with the trained one (C3) → test first (L0).
- Under Jin et al.'s metrics the tuned fixed action may beat the DQN → report honestly; the claim then rests on adaptivity vs the predecessor's equal-mix ablation, plus C1, C2 and C5.
- Re-implementing six baselines without public code is the largest cost → start with IP-MOEA and IMOEA/D.
