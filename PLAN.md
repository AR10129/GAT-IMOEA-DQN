# Paper plan

**Target:** Engineering Applications of Artificial Intelligence (EAAI), the venue of
Jin et al. (2026), "Reinforcement learning-driven interval multi-objective
evolutionary algorithm for task offloading in uncertain cloud–edge", EAAI 178,
115043. Backups: Swarm and Evolutionary Computation, Applied Soft Computing,
Expert Systems with Applications.

**Reframed 2026-10-29** (after Suite A and three DQN learner designs; history in P1 below).
The paper now leads with the selection backbone, not the learned controller: on Suite A the
decomposition-ranked three-tier selection beats IMOEA/D on IGD in 12/12 cases, while no DQN
design beat the best fixed action. The DQN work becomes an analysis section.

**Working title:** Interval multi-objective offloading of DAG workflows in uncertain
cloud–edge–end environments via decomposition-ranked three-tier environmental selection.

**Working name of the method:** DR-IMOEA (decomposition-ranked IMOEA); rename before writing.

**The gap:** (1) Jin et al. treat every task as an independent offloading unit (their §3.3) and
sum per-task latency; we model applications as DAGs: precedence, inter-task data transfer, and
makespan along the interval critical path. (2) Their convergence ranking R_IN (dominance degree)
loses selection pressure once most of the merged population is non-dominated (measured: 105–123
of 210 pool members tie at R_IN = 0 for 88 slots), so the population drifts; a decomposition rank
restores the pressure while keeping their dominance / uncertainty tiers.

**Guiding rule:** keep everything except the DAG model and the selection backbone as close to
Jin et al. as possible (parameters, metrics, protocol, baselines), so reviewers can compare the
two papers directly. Anything beyond that is a sensitivity study or a stated limitation.

## Contributions

| # | Contribution | Status |
|---|---|---|
| C1 | Interval critical-path model for DAG workflows in the cloud–edge–end architecture (transfer, makespan, energy, cost); reduces exactly to Jin et al.'s model for single-task applications | built and tested |
| C2 | DAG-valid two-string encoding: allocation + topological priority, POX, shift mutation, topological repair | built |
| C3 | **Headline.** Decomposition-ranked three-tier selection: Rank C = interval Tchebycheff rank over N weight vectors (ICD tie-break) inside Jin et al.'s α/β/γ selection, plus neighbourhood mating; with a diagnosis of why R_IN stalls | built; Suite A: beats IMOEA/D IGD 12/12, HV 10/12 (10 seeds) |
| C4 | Analysis of learned control of (α, β, γ): reward audit (per-step SNR 0.007–0.06), three learner designs, GAT vs no-GAT; best learner reaches but does not beat the best fixed action, and GAT adds nothing | done (`reward_audit`, `suite_a_tch`, `dqn_fix_test`); write up as ablation / negative result |
| C5 | Released benchmark: Suite A instances + 40 interval WfCommons / Pegasus workflows + generators | Suite A and B built |
| C6 | Evaluation stronger than the predecessor's: their 5 baselines + IMOEA-DQN on the DAG model + NSGA-II + list-scheduling heuristics; 30 runs; Holm-corrected tests | partly (IMOEA/D, IP-MOEA; Jin-backbone DQN = IMOEA-DQN on our model) |

C4 is supporting material: it explains why per-generation DQN control fails here, which is
useful to the field, but it is not a performance claim. The GAT stays only in that section.

**What a reviewer will ask, and our answer (to prepare):**
- *Is C3 novel?* Decomposition + dominance hybrids exist (MOEA/DD, θ-DEA, NSGA-III, RVEA) but not
  for interval objectives inside a three-tier dominance / uncertainty selection. Literature check
  is task W1 below.
- *Which (α, β, γ) does the method use, and was it tuned on the test set?* Today's (5/6, 1/6, 0)
  came from a trained agent's favourite action on the Jin backbone. Re-select it on the training
  instances (i1–i3) only, and report the sensitivity over all 28 actions (task A2).
- *Why drop the RL?* We don't hide it: C4 reports it with the reasons. EAAI's scope covers
  evolutionary methods for engineering problems.

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
- [ ] B0. Reference heuristics → moved to P1b

### P1 — Core claims (history: the learned-controller line, closed 2026-10-29)
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
  - **Audit (a) done, 2026-10-28** (`scripts/backbone_audit.py`, `results/experiments/backbone_audit`;
    fixed actions, G_max = 1000, small_20_i0@x1 and basic_60_i0@x20, 4 seeds — means only, too few
    seeds for Wilcoxon).
    - Our code matches Jin et al.'s Algorithms 1 and 2. Gaps found where the paper is silent:
      (1) Rank C ties are cut at random, and after ≈ gen 100 more than αN pool members have
      R_IN = 0 (105–123 for 88 slots on basic_60), so the population drifts along the front;
      (2) the "mating selection" box of Jin et al.'s Fig. 2 is not implemented.
    - Front analysis of the Suite A runs: our fronts are not mostly dominated by IMOEA/D's; they
      cover only part of the front (trained agents miss the low-cost end: best normalised cost
      0.28–0.35 vs 0.03–0.09).
    - New options (defaults unchanged, so earlier results reproduce): `SelectionConfig.tie_break="icd"`,
      `VariationConfig.mating="tournament"`, `VariationConfig.partner="neighbours"`.

    | variant | small_20 ×1 IGD / HV | basic_60 ×20 IGD / HV |
    |---|---|---|
    | base (fixed_fav) | 0.045 / 0.82 | 0.096 / 0.58 |
    | + ICD tie-break | 0.032 / 0.80 | 0.097 / 0.56 |
    | + tournament mating | 0.136 / 0.74 | 0.158 / 0.53 |
    | + neighbourhood mating | 0.034 / 0.85 | 0.084 / 0.57 |
    | + neighbourhood + tie-break | **0.021** / 0.87 | 0.078 / 0.59 |
    | IMOEA/D without mating restriction | 0.059 / 0.88 | 0.045 / 0.73 |
    | IMOEA/D | 0.049 / **0.89** | **0.023 / 0.76** |

    - Conclusion: tie-break + neighbourhood mating closes the gap on the small instance (better IGD
      than IMOEA/D) but not on basic_60. IMOEA/D keeps most of its lead without mating restriction,
      so the remaining gap is its replacement rule: scalarised (Tchebycheff), steady-state updates
      keep convergence pressure when most of the pool is non-dominated, while a dominance-degree
      ranking loses it. Tournament mating hurts; drop it.
    - Next decision: option (b) — keep Jin et al.'s three rankings but give Rank C a scalarised
      (decomposition) convergence measure, or let the DQN steer a decomposition backbone — or
      accept (c).
  - **Rank C prototype done, 2026-10-28 — the gap is closed.** `SelectionConfig(convergence="tchebycheff")`:
    N Das-Dennis weights, objectives normalised over P_M, interval Tchebycheff per weight, and
    R_TCH(x) = best position of x over the weights' midpoint orders (sorting by possibility degree
    at 0.5 is sorting by midpoint, i.e. IMOEA/D's own comparison). D and U tiers unchanged; ICD tie-break.
    Same audit protocol (fixed actions, no learning, 4 seeds, means only):

    | variant | small_20 ×1 IGD / HV | basic_60 ×20 IGD / HV | medium_250 ×20 IGD / HV |
    |---|---|---|---|
    | base (fixed_fav, R_IN) | 0.047 / 0.82 | 0.102 / 0.60 | 0.155 / 0.55 |
    | tch (fixed_fav) | 0.016 / 0.89 | 0.031 / 0.77 | — |
    | **tch_near** (fixed_fav + neighbourhood mating) | **0.012 / 0.90** | **0.014 / 0.80** | 0.016 / 0.82 |
    | tch_conv_near (1, 0, 0) | 0.013 / 0.90 | 0.015 / 0.80 | **0.014 / 0.82** |
    | IMOEA/D | 0.043 / 0.87 | 0.026 / 0.78 | 0.035 / 0.79 |

    - tch_near beats IMOEA/D on IGD (≈ 2–4×) and HV on all three instances, with similar
      Imprecision (21.2 vs 18.9, 30.9 vs 31.2, 37.1 vs 40.7). The decomposition rank alone does most
      of it; neighbourhood mating adds the rest.
    - Paper framing: a decomposition-based convergence ranking inside Jin et al.'s three-tier
      DQN-ES is a new, defensible contribution (it fixes the loss of pressure of R_IN when most of
      P_M is non-dominated, measured above).
    - Open: (1) retrain the DQN on the new backbone — with tch Rank C the best fixed actions
      (5/6, 1/6, 0) and (1, 0, 0) are close, so the learned controller must show it still adds
      value; (2) confirm with 10+ seeds on all 12 Suite A cases; (3) ablate tie-break and mating.
  - **Suite A on the tch backbone done, 2026-10-29** (`results/experiments/suite_a_tch`, 10 seeds,
    Gmax 1000, 12 cases; agents retrained 48 episodes × 3 seeds; `summary_x1/x20.md`,
    `summary_vs_tch_fixed_fav.md`). Mean IGD / HV_jin over the 12 cases, Wilcoxon wins/losses/ties:

    | policy | IGD | HV_jin | IGD vs IMOEA/D | IGD vs tch_fixed_fav |
    |---|---|---|---|---|
    | **tch_fixed_fav** (5/6, 1/6, 0) | **0.0124** | **0.958** | 12/0/0 | — |
    | tch_conv (1, 0, 0) | 0.0134 | 0.955 | 12/0/0 | 1/6/5 |
    | tch_trained_nogat_stable | 0.0244 | 0.939 | 5/1/6 | 0/12/0 |
    | tch_trained_stable (GAT) | 0.0261 | 0.947 | 5/4/3 | 0/11/1 |
    | IMOEA/D | 0.0291 | 0.943 | — | 0/12/0 |
    | tch_fresh (online, untrained) | 0.0721 | 0.894 | 0/8/4 | 0/12/0 |
    | trained_stable (Jin backbone) | 0.107 | 0.872 | 0/12/0 | 0/12/0 |

    - The backbone fix holds at full scale: tch with a fixed action beats IMOEA/D on IGD in 12/12
      cases (×1 and ×20 alike) and on HV in 10/12 (2 ties).
    - **The DQN does not add value on this backbone.** Both trained agents lose to the best fixed
      action on IGD in 11–12/12 cases (about 2× the IGD); they only win on Imprecision (21.7 vs 24.5).
      Training reward stays flat (0.0004 → 0.0005) and validation scores hover at 0.47–0.52 for
      all six agents with no upward trend, so the agents have not learned a useful policy.
    - GAT vs no-GAT: no clear difference (IGD 0.026 vs 0.024, HV 0.947 vs 0.939).
    - This is the risk listed below ("the tuned fixed action may beat the DQN") coming true.
      Decision needed before more experiments: (a) fix the learning signal (the per-step reward is
      ~4e-4, likely too small and too noisy to learn from — e.g. reward scaling, IGD/HV-delta reward,
      fewer/coarser actions, longer training) and retrain; or (b) reframe the paper around the
      decomposition backbone with the DQN as an ablation/negative result.
  - **L7 reward audit, 2026-10-29** (`scripts/reward_audit.py`, `results/experiments/reward_audit`):
    each of the 28 actions fixed for 1000 generations, small_20 ×1 and basic_60 ×20, 2 seeds.
    - The summed PIBI reward ranks actions like IGD overall (Spearman 0.92 / 0.90) but not among
      the top ones: (0.67, 0, 0.33) earns near-top reward with ~2× the best IGD — the action the
      GAT agent picked most.
    - Per-step signal-to-noise after generation 100 is 0.007–0.06 (spread of the 28 action means /
      per-step sd). The per-generation reward is almost pure noise; this is why nothing was learned.
    - Static headroom is small: the best fixed action per instance (IGD 0.0087 / 0.0128) is within
      ~10% of tch_fixed_fav, so an adaptive controller can only win through the schedule over time.
  - **L7 fix, short test, 2026-10-29** (`results/experiments/dqn_fix_test`): one decision per 50
    generations (`RunConfig.decision_every`), reward = relative HV_jin improvement over the block,
    n-step 3, gamma 0.95; 2 training seeds × 24 episodes; 4 test cases × 4 seeds.
    - Training reward now rises (≈ 0.05 → 0.09 per block on revisited workflows; the old learner
      stayed at 0.0004), but test performance got worse: IGD 0.045 (no GAT) / 0.062 (GAT) vs 0.026 /
      0.030 for the old agents and 0.0136 for tch_fixed_fav, last on all 4 cases. The agents
      favour diversity-heavy actions ((0.5, 0.5, 0) 31–52%), which the audit shows are poor.
    - Likely cause: too little data for 28 actions — 24 episodes × 20 decisions = 480 transitions
      per agent — and early blocks (clipped large relative gains) dominate the return.
    - Status: the DQN still does not beat the best fixed action after two learner designs.
  - **L8 learner, short test, 2026-10-29** (same experiment dir and protocol; 48 training episodes):
    only the 10 actions with α ≥ 1/2 (`RunConfig.action_set="convergent"`), one decision per 25
    generations, reward = log HV_jin ratio (unclipped; sums to the final HV gain), n-step 5, γ 0.98.
    Mean over the 4 cases (one normalisation across all 10 policies):

    | policy | IGD | HV_jin | Imprecision |
    |---|---|---|---|
    | **tch_fixed_fav** | **0.0130** | **0.791** | 30.8 |
    | tch_conv (1, 0, 0) | 0.0145 | 0.787 | 33.6 |
    | tch_trained_nogat_l8 | 0.0149 | 0.789 | 29.2 |
    | tch_trained_l8 (GAT) | 0.0174 | 0.785 | 31.7 |
    | tch_random_conv (random among the 10) | 0.0223 | 0.757 | 29.4 |
    | tch_trained_stable (old) | 0.0311 | 0.772 | 28.3 |
    | IMOEA/D | 0.0379 | 0.766 | 30.8 |

    - Big step up: IGD halves vs the old agents, beats random choice among the same 10 actions on
      all 4 cases and IMOEA/D by ~2.5×; the no-GAT agent ties tch_conv and is within ~0.2% of
      tch_fixed_fav on HV_jin (higher on medium_100). It still trails tch_fixed_fav on IGD in all
      4 cases (by 10–25%). 4 seeds, so no test is significant; read the means.
    - Training reward stays flat (0.039 → 0.040), so most of the gain comes from the restricted
      action set, not from learned adaptation. GAT is again no better than no-GAT.
    - Conclusion after three learner designs: on the tch backbone the DQN reaches, but does not
      beat, the best tuned fixed action.
### P1b — Reframed core claims (from 2026-10-29; in this order)
- [ ] W1. Literature check for C3: decomposition / dominance hybrids (MOEA/DD, θ-DEA, NSGA-III,
  RVEA, SPEA/R, MOEA/D-DU) and interval MOEAs (IMOEA/D, IMOEA/D-AWN, IP-MOEA, interval NSGA-II
  variants). Write down exactly what is new: an interval Tchebycheff rank as the convergence tier
  of a dominance / uncertainty three-tier selection, and the R_IN-stall diagnosis.
- [ ] A2. Pick (α, β, γ) without the test set: the 28-action sweep (`scripts/reward_audit.py`) on
  the training instances i1–i3 of all 12 cases; choose the default by mean rank; report the
  sensitivity surface; keep (1, 0, 0) as the "no tuning" variant.
- [ ] A1. Backbone ablation on Suite A with the chosen action (10 seeds, 12 cases): Jin backbone →
  + Tchebycheff Rank C → + ICD tie-break → + neighbourhood mating; neighbourhood size {10, 20, 40},
  p_local {0.5, 0.9, 1.0}, number of weight vectors. The audit (4 seeds, 2–3 cases) is the pilot.
- [ ] A3. What the β (dominance) and γ (uncertainty) tiers buy under the new Rank C: IGD vs
  Imprecision trade-off from the A2 sweep, and the Rank C tie counts per generation (the stall
  figure: R_IN vs Tchebycheff).
- [ ] B0. Reference heuristics: all-local, all-edge, all-cloud, HEFT-style list scheduling
- [ ] B1. Remaining interval baselines: IMOEA/D-AWN, IMEA-IED, IMaOEACI (IMOEA/D, IP-MOEA done;
  check IP-MOEA's tie-breaking / crowding against the original before the final runs)
- [ ] B2. IMOEA-DQN (Jin et al.) on the DAG model = `trained_stable` on the Jin backbone (and
  `fresh` for the online version): check it against their Algorithm 1/2 and name it in tables
- [ ] B3. NSGA-II (midpoint and interval-dominance variants)
- [ ] B4. Equal budget for all methods; wall-clock times reported
- [ ] D1. Write-up material for C4: the reward-audit figures (per-step SNR by phase; reward
  ranking vs IGD ranking), the three-learner table (stable / blk / l8), GAT vs no-GAT. A full Suite A
  retrain of L8 only if a reviewer asks.

### P2 — Evaluation
- [ ] E1. Suite A (3 scales × 2 device counts × CCR ×1 / ×20) and Suite B (5 families × 3 sizes × 3 CCR)
- [ ] E2. 30 runs per method and instance
- [ ] E3. Metrics as above, normalised once per instance
- [ ] E4. Statistics as above
- [ ] E5. IGD / HV vs generation curves for DR-IMOEA and every baseline
- [ ] E6. Sensitivity: CCR, uncertainty width, edge count, bandwidth, energy scope, multi-core queueing
- [ ] E7. Monte Carlo check of final fronts under sampled crisp values

### P3 — Paper
- [ ] R1. One script builds every table and figure (LaTeX)
- [ ] R2. System-model figure, pseudocode, complexity, parameter table
- [ ] R3. Threats to validity and limitations (no link contention, no receive energy, interval
  bounds assumed to hold, fixed action chosen on training instances)

## Risks

- C3 judged incremental ("just add MOEA/D's Tchebycheff") → W1 first; make the diagnosis (tie
  counts), the ablation and the interval treatment explicit; the DAG model (C1) is the second pillar.
- Tuning on the test set → A2 picks the action on training instances only.
- Our IMOEA/D is weaker or stronger than the original → document every parameter from Gan et al.
  (2023); Jin et al. report IMOEA-DQN > IMOEA/D, we find the opposite on the DAG model: explain the
  differences (DAG model, budget, re-implementation) and report it plainly.
- Re-implementing the remaining baselines without public code is the largest cost → after A1/A2.
- Closed: the learned controller does not beat the best fixed action (three designs, 2026-10-29);
  now reported as C4 analysis.
