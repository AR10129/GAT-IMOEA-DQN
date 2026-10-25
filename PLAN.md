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
- [ ] R0. Project under git; every result folder records its commit
- [ ] M1. Jin et al. scenario (`datasets/scenarios/jin.yaml`) + fixed channel gain option
- [ ] M2. Load-time rescaling of workloads and data into Jin et al.'s ranges; CCR multiplier for sweeps
- [ ] M3. Energy-scope switch (device-only / three-tier)
- [ ] M4. Re-measure how much communication moves the objectives (target at CCR = 1: ≥ 10%)
- [ ] M5. Multi-device, multi-DAG evaluator (one queue per device; x = 0 means the task's own device)
- [ ] M6. Suite A generator (layered random DAGs) + instances at the three scales
- [ ] B0. Reference heuristics: all-local, all-edge, all-cloud, HEFT-style list scheduling

### P1 — Core claims
- [ ] L0. Re-run trained vs fresh vs fixed at G_max = 1000 under Jin et al.'s metrics (risk check for C3)
- [ ] L1. Dynamic structure-aware state (per-task placement shares, critical-path frequency, slack)
- [ ] L2. Training distribution: about 250 graphs; leave-one-family-out tests
- [ ] L3. Stable reward: fixed per-run normalisation, 5-step returns
- [ ] L4. Learner tuning + training curves per seed
- [ ] L6. Gate: Q-ranking Spearman ≥ 0.6, greedy regret ≤ 30% of random's, GAT beats no-GAT
- [ ] B1. Re-implement IP-MOEA, IMOEA/D, IMOEA/D-AWN, IMEA-IED, IMaOEACI (same encoding and evaluator)
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
