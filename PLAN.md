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
| C3 | **Headline.** Decomposition-ranked three-tier selection: Rank C = interval Tchebycheff rank over N weight vectors (ICD tie-break) inside Jin et al.'s α/β/γ selection, plus neighbourhood mating (T = 10, p_local = 1.0); with a diagnosis of why R_IN stalls | built; Suite A: beats IMOEA/D IGD 12/12, HV 10/12 (10 seeds) |
| C4 | Analysis of learned control of (α, β, γ): reward audit (per-step SNR 0.007–0.06), three learner designs, GAT vs no-GAT; best learner reaches but does not beat the best fixed action, and GAT adds nothing | done (`reward_audit`, `suite_a_tch`, `dqn_fix_test`); write up as ablation / negative result |
| C5 | Released benchmark: Suite A instances + 40 interval WfCommons / Pegasus workflows + generators | Suite A and B built |
| C6 | Evaluation stronger than the predecessor's: their 5 baselines + IMOEA-DQN on the DAG model + NSGA-II + list-scheduling heuristics; 30 runs; Holm-corrected tests | partly (IMOEA/D, IP-MOEA; Jin-backbone DQN = IMOEA-DQN on our model) |

C4 is supporting material: it explains why per-generation DQN control fails here, which is
useful to the field, but it is not a performance claim. The GAT stays only in that section.

**What a reviewer will ask, and our answer (to prepare):**
- *Is C3 novel?* The rank itself is not: it is MSOPS's best-position rank (Hughes 2003) on
  interval midpoint Tchebycheff values. New is its use as the convergence tier inside Jin et
  al.'s dominance / uncertainty selection, the diagnosis of why R_IN stalls, and the evidence.
  Say this plainly and cite MSOPS and Global WASF-GA (W1, `docs/novelty_check.md`).
- *Which (α, β, γ) does the method use, and was it tuned on the test set?* (5/6, 1/6, 0), chosen
  on the training instances (i1–i3) only: A2 (2026-10-31) ranks it first of 28 by mean IGD rank on
  the 36 training instances. The test set was never used. The top four actions (α ≥ 2/3, γ ≤ 1/6)
  are within 1.2× of the per-instance best IGD on average, so the method is not sensitive to the
  exact choice inside that region.
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
- [x] W1. Literature check (done 2026-11-02, `docs/novelty_check.md`). Findings:
  - **Our Rank C is MSOPS's ranking** (Hughes 2003: best position over per-weight Tchebycheff
    orderings); Global WASF-GA (2017) is close. Not new as a ranking: credit both.
  - Still new (no paper found): that rank, on interval objectives, as the convergence tier of
    Jin et al.'s dominance / uncertainty three-tier selection; the R_IN-stall diagnosis; the
    integrated method beating the interval baselines.
  - Closest C1 prior work: Shi et al., EIJ 2023 (interval, dependent workflows, devices + edge,
    migration). Get the full text and separate C1 from it.
  - New interval baseline candidate: IMOEA/D-DAW (Sci. Rep. 2026).
  - Consequences: add a midpoint NSGA-III or MOEA/DD baseline (B3), make A3's stall figure
    central, and run Suite B and the DAG heuristics before the hand-written baselines.
- [x] A2. Pick (α, β, γ) without the test set (done 2026-10-31, `scripts/action_sweep.py`,
  `results/experiments/action_sweep/summary.md`). 28 fixed actions × 36 training instances (i1–i3
  of the 6 configurations at CCR ×1 and ×20), 1 seed, G = 1000, tch backbone.
  - Chosen default: **(5/6, 1/6, 0)**, mean IGD rank 2.25 of 28, IGD ratio to the per-instance best
    1.13 on average and 1.71 at worst. It is the action already used, so the Suite A results stand.
  - Runners-up: (2/3, 1/6, 1/6) 2.67, (5/6, 0, 1/6) 2.69 (best mean HV rank, 1.75), (1, 0, 0) 2.97.
    Only these four ever win an instance (9 / 10 / 9 / 8 wins).
  - Sensitivity: IGD rank rises steadily as α falls. Below α = 1/2 the IGD ratio is above 4;
    pure γ (0, 0, 1) is 52× worse. Imprecision falls as γ rises (0.31 → 0.20 per solution): the
    A3 trade-off.
  - Robustness: the default is 1st on CCR ×1 and 3rd on CCR ×20, and in the top 3 on every
    configuration. Keep (1, 0, 0) as the "no tuning" variant.
- [x] A1. Backbone ablation (done 2026-11-01, `scripts/ablation.py`,
  `results/experiments/ablation/summary.md`). Action (5/6, 1/6, 0) fixed, 12 test cases × 10 seeds,
  Wilcoxon vs full per case. Mean IGD (W/L/T of the variant against full):
  - Build-up: Jin 0.116 → + Tchebycheff Rank C 0.053 → + ICD tie-break 0.044 → + neighbourhood
    mating (= full) **0.0125**. Every step is significant on 12/12 cases. IMOEA/D is 0.042.
  - Leave one out: without Tchebycheff 0.069 (0/12/0), without neighbourhood mating 0.044
    (0/12/0), without ICD 0.015 (0/7/5). The first two are the method, and they interact:
    neighbourhood mating alone (on R_IN) and Tchebycheff alone each recover only part of the
    gain, together they are 3.5× better than either. ICD is a small, consistent extra.
  - Sensitivity: smaller / more local mating is better. T = 10 beats T = 20 (8/0/4, IGD 0.0108),
    p_local = 1.0 beats 0.9 (7/0/5); T = 40 and p_local = 0.5 are worse on 11–12/12. The default
    number of weight vectors (N) beats N/2 and 2N.
  - **Resolved (A1b, 2026-11-02, `scripts/tune_mating.py`):** T ∈ {3, 5, 10, 20} × p_local ∈ {0.9, 1.0}
    re-tuned on the 36 training instances (2 seeds): **T = 10, p_local = 1.0** has the best mean
    IGD rank and beats the old default on 35/36 training instances; T = 3 is worse, so 10 is
    inside the range. On the test cases (`final` variant, 10 seeds): mean IGD 0.0103 vs 0.0135
    for the old default (9/0/3), HV 12/0/0; vs IMOEA/D 12/0/0 on IGD. **The method's settings
    are now frozen: (5/6, 1/6, 0), Tchebycheff Rank C with N weights, ICD tie-break,
    neighbourhood mating T = 10, p_local = 1.0.** Use these for all B and E runs.
  - The seconds column is not comparable: jin / full / conv / imoead were run in other batches
    under different machine load. B4 does timing properly.
- [x] A3. What the β (dominance) and γ (uncertainty) tiers buy under the new Rank C: IGD vs
  Imprecision trade-off from the A2 sweep, and the Rank C tie counts per generation (the stall
  figure: R_IN vs Tchebycheff).
  Done 2026-11-10, `scripts/a3_tiers.py` -> results/figures/a3_tradeoff.{png,pdf}, a3_stall.{png,pdf},
  results/stats/a3_stall.csv.
  - Trade-off (A2 sweep, 28 actions x 36 training instances): a small γ buys precision cheaply,
    (5/6, 0, 1/6) has Imprecision 0.276 vs 0.307 per solution (-10%) for IGD ratio 1.15 vs 1.13;
    β or γ above 1/6 costs IGD steeply (ratio > 2 below α = 1/2); pure γ reaches 0.20 at 52x IGD.
    So the tiers trade convergence for precision along one curve; final keeps γ = 0 (chosen on IGD).
  - Stall (medium_250_i0@x20, montage_400@x20, 3 seeds, G = 1000): share of the α N Rank C places
    decided by the tie-break. Jin's R_IN: 43% / 10% in G 1-100, 99.5% / 86% in G 101-500, 100% /
    99.9% after G 500; 117-121 pool members share R_IN = 0 at the end (> α N = 88), so selection is a
    random draw from the first front. Tchebycheff Rank C: 21% / 14%, 21% / 23%, 11% / 14%; about 80
    share the best rank. This is the mechanism behind the build-up in A1.
- [x] B0. Reference heuristics (2026-11-03, `gat_imoea_dqn/heuristics.py`, policy `heuristics`):
  all_local, all_edge (nearest edge), all_cloud, and HEFT-style list scheduling in upward-rank
  order: heft_makespan = classic earliest-finish-time placement, heft_energy / heft_cost = greedy
  on the plan's energy / cost. Note for the paper: HEFT loses to all_cloud on makespan because
  edge capacity is shared by every task placed on it, so early EFT choices are slowed by later
  ones. Pilot (4 Suite B cases, 2 seeds, G = 1000): the heuristics' IGD is 20–40× the method's.
- [x] Speed-up (2026-11-03): compiled population scorer, bit-identical to the reference
  (old runs reproduce exactly); runs are 5–10× faster.
- [x] B1. Remaining interval baselines (2026-11-05; PDFs in the repo root, user-supplied):
  - [x] IMOEA/D-AWN (Jin, Zhang, Xie, Cui, EIJ 24 (2023) 100405) and IMEA-IED (Cui et al., SWEVO 91
    (2024) 101749) re-implemented from the papers in `gat_imoea_dqn/baselines_b1.py` (policies
    `imoead_awn`, `imea_ied`); the docstring lists every equation used and every choice the paper
    leaves open. Tests: `tests/test_baselines_b1.py` (IED reproduces the paper's Fig. 3 example).
    Launch script `scripts/run_b1.cmd` (Suite A + B, 1710 runs with imoead_orig).
  - Dropped: IMaOEACI (Cai et al., Inf. Sci. 690 (2025) 121552). Its core quantity, the
    "individual comprehensive indicator", is defined only by citation to an unpublished PhD thesis
    (Zhang, Taiyuan Univ. Sci. Tech., 2023), and its environmental selection is one paragraph, so a
    faithful re-implementation is not possible. Say so in the paper.
  - Not a baseline: IMOEA/D-DAW (Sci. Rep. 2026). Its sparsity measure is built for a bi-objective
    problem with one crisp objective; adapting it to three interval objectives would be our own
    design. Related work only.
  - [x] Fidelity: IMEA-IED's Section 4.3 gives IMOEA/D's original settings as T = 10, reference
    vectors adapted from G_max / 2 every G_max / 6. Ours uses T = 20 and adapts every 10% from the
    start (baselines.py). Decided 2026-11-05: add policy `imoead_orig` (T = 10, adaptation from G_max / 2
    every floor(G_max / 6)), queued in run_b1.cmd (570 runs); `imoead` is unchanged and bit-identical.
    Report the original-settings version as IMOEA/D; the old one becomes a sensitivity note.
  - Results (done 2026-11-05, no errors; mean IGD, reference front now pooled over 13 policies):
    - Suite A: final 0.0123, imoead_awn 0.0206, imoead_orig 0.0224, imoead 0.0304, imea_ied 0.0824.
      final wins IGD and HV_jin 12/12 against all three.
    - Suite B: final 0.0158, imoead_orig 0.0183, imoead 0.0236, imoead_awn 0.0296, imea_ied 0.0916.
      vs final (IGD · HV_jin, better/worse/tie): imoead_orig 4/30/11 · 12/30/3;
      imoead_awn 3/30/12 · 2/41/2; imea_ied 0/45/0 · 0/45/0.
    - **Threat:** imoead_orig beats final on HV_jin in 12 of the 15 1000-task cases (on IGD in those 15 cases:
      imoead_orig better in 2, tie 3, final better in 10). final still wins every
      100- and 400-task case on HV. The original schedule (T = 10, late adaptation) is much
      stronger than our earlier setting at 1000 tasks. Decide before writing: investigate (e.g. G
      budget at 1000 tasks, late adaptation inside final), and state it in the paper either way.
    - Investigation (2026-11-05; `scripts/run_inv.cmd`, experiments `inv_act`, `inv_g2000`; 3 cases
      cybershake_1000@x1, epigenomics_1000@x20, sipht_1000@x20, 5 seeds, scored on the Suite B scale):
      - Where the fronts overlap, final is better: 61-90% of imoead_orig's front is dominated by
        final's (pooled over seeds), 0-5% the other way. imoead_orig wins HV by reaching cheaper
        solutions (normalised min cost about 0.24 vs 0.28-0.29). At 400 tasks it is the reverse
        (final 0.13-0.16, imoead_orig 0.14-0.19), which is why final wins HV there.
      - More diversity in selection hurts: action (2/3, 1/3, 0) IGD 0.0267, (1/2, 1/2, 0) 0.0378,
        vs final 0.0203. 10% global mating (`final_p09`): IGD 0.0175 (better), HV unchanged (0.920),
        min cost no lower. Neither closes the HV gap.
      - Budget: at G = 2000 both improve; final IGD 0.0061 vs imoead_orig 0.0094, HV 0.9535 vs
        0.9572 (final ahead on cybershake, behind on the other two). imoead_orig still reaches
        cheaper solutions (0.215-0.224 vs 0.231-0.245). So the gap is not only budget: IMOEA/D
        covers the low-cost extreme better at 1000 tasks.
      - Timing: the 898 s mean for final at 1000 tasks was 10 sleep-stretched runs (up to 10884 s);
        medians are final 176 s, imoead_orig 162 s, imoead_awn 226 s, nsga2_mid 70 s. Re-time in B4.
      - Options: report as is (IGD and dominance favour final, HV favours imoead_orig at 1000 tasks
        by reaching the cost extreme), or change the frozen method (e.g. keep each objective's
        extreme solution); a change means re-running every `final` run.
  - [x] Check IP-MOEA's tie-breaking / crowding against the original (2026-11-10). The original
    (Limbourg & Aponte, CEC 2005) was not obtainable (ResearchGate 403). IMOEA/D-AWN (Sec. 1) and
    IMaOEACI (Sec. 2) both describe its crowding as each individual's contribution to the
    hypervolume of the interval upper and lower limits, not midpoint crowding. Jin et al. 2026
    (Sec. 5.3.1) give the same reason for IP-MOEA's weakness that we see (the order-relation
    dominance leaves too many solutions non-dominated). New policy `ip_moea_hv`: exclusive HV
    contribution within each front, mean of lower and upper corners, normalised over the ranked
    set, ref 1.1 (fast exact form; ~5x slower per generation than ip_moea). Pilot (`scripts/
    run_checks.cmd`, checks_a / checks_b, the E5 instances, 10 seeds; E3 normalisation), IGD
    ip_moea -> ip_moea_hv: 0.309 -> 0.184, 0.290 -> 0.123, 0.321 -> 0.189, 0.331 -> 0.206 (p < 0.001
    on all 4, HV too); final 0.011 / 0.010 / 0.030 / 0.037, nsga2_mid 0.065 / 0.042 / 0.127 / 0.152.
    The HV-contribution version is both closer to the description and stronger, so it should
    replace ip_moea in the paper tables (needs Suite A + B reruns, 570 runs). Conclusions unchanged.
- [x] B2. IMOEA-DQN (Jin et al.) on the DAG model = `trained_stable` on the Jin backbone (and
  `fresh` for the online version): check it against their Algorithm 1/2 and name it in tables
  - [x] Suite B runs done 2026-11-04 (`scripts/run_b2.cmd`, 1800 runs: Suite A GAT agents ×3
    training seeds, greedy, plus online `fresh`; agents trained on the default scenario, so this
    is also a transfer test to Jin's scenario). Mean IGD: final 0.011, IMOEA/D 0.020, Jin backbone
    with fixed action 0.083, trained_stable 0.125, fresh 0.176. Both DQN versions lose to final and
    to the fixed-action Jin backbone on IGD and HV_jin in 45/45 cases; trained beats fresh
    (342–372 of 450 runs per training seed). By size (IGD trained / fixed_fav): 0.096 / 0.046,
    0.108 / 0.082, 0.171 / 0.120. Supports C4: the learned controller adds nothing over a fixed
    action. (Adding policies changes the pooled reference front, so final's IGD moved 0.0105 → 0.011.)
  - [x] Algorithm 1/2 fidelity check (2026-11-10, against the EAAI 178 (2026) 115043 text). Match:
    DAEC (interval ND sort, top half mates within the top half, uncertainty ND sort, p_c 0.7 / 0.9);
    DQN-ES with R_IN (dominance-strength sum), ICD diversity rank, R_UN uncertainty rank, sequential
    deduplicated α/β/γ selection; 28 Das-Dennis actions (H = 6); 7-feature state; PIBI-improvement
    reward (sign flipped so improvement is positive); update every 50 generations; lr 1e-3; N 105,
    G 1000. Differences: (1) the published learner is online with an MLP 32-64-32 on the 7 features;
    our `trained_stable` is pre-trained offline with a GAT graph embedding and (64, 64) head, and
    `fresh` is online but with the GAT; (2) ERB threshold and ε formula are not given in the paper
    (ours: 250, 0.3^(t/100)); (3) operators are ours (their SBX cannot keep a DAG order valid).
    New policy `imoea_dqn` = the published learner (online, no graph, MLP 32-64-32, agent seed = run
    seed). Pilot IGD (E5 instances): imoea_dqn 0.276 / 0.186 / 0.283 / 0.300, fresh 0.275 / 0.253 /
    0.309 / 0.320, trained_stable 0.162 / 0.123 / 0.256 / 0.260, fixed_fav 0.170 / 0.110 / 0.163 /
    0.207: the published learner is no better than `fresh` (HV slightly better on Suite B) and loses
    to the fixed-action backbone on IGD on 4/4 (p < 0.001).
    Table naming: "IMOEA-DQN" = `imoea_dqn` (as published, needs Suite A + B reruns, 570 runs);
    "IMOEA-DQN, offline GAT" = `trained_stable` (the C4 analysis); "Jin backbone, fixed action" =
    `fixed_fav`.
  - [x] Corrected-baseline reruns (2026-11-10, `scripts/run_rerun_imoeadqn.cmd`,
    `scripts/run_rerun_ipmoea.cmd`; experiments rerun_{imoeadqn,ipmoea}_{a,b}; 570 + 570 runs, 40 of
    each copied from the checks pilot (same settings and seeds); no errors). The E3 paper set now uses
    `ip_moea_hv` (label IP-MOEA) and `imoea_dqn` (label IMOEA-DQN) instead of `ip_moea` and
    `trained_stable`; still 10 policies. `scripts/stats.py` regenerated suite_{a,b}.md / *_runs.csv,
    pooling suite_a_main + suite_a_e2 + rerun_*_a and suite_b + suite_b_e2 + rerun_*_b.
    - Suite A: final better on IGD and HV against every policy on 12/12; Friedman IGD final 1.00,
      IMOEA/D-AWN 2.42, IMOEA/D 2.83, ..., IP-MOEA 8.50, IMOEA-DQN 9.92. Mean IGD final 0.012,
      IP-MOEA 0.147, IMOEA-DQN 0.254.
    - Suite B: IGD vs final - IMOEA/D 1/43/1 (wins only inspiral_1000@x100), NSGA-II mid 4/40/1,
      int 5/40/0 (the 100-task x1 cases), all others 0 wins. HV_jin - IMOEA/D 2/42/1
      (inspiral_1000@x100, sipht_1000@x20), all others 0. Friedman IGD 1.29 vs 2.69 (IMOEA/D),
      HV 1.09 vs 2.73, CD 2.02. Mean IGD final 0.019, IMOEA/D 0.026, IP-MOEA 0.122, IMOEA-DQN 0.229.
    - IMOEA/D's IGD wins fell from 4 to 1; cybershake_1000@x1, epigenomics_1000@x20 and
      inspiral_1000@x1 flipped to final (e.g. epigenomics_1000@x20 final / IMOEA/D 0.035 / 0.031 ->
      0.030 / 0.038). No run of either changed: the cause is the pool. Dropping ip_moea removes the
      cost nadir it set (2.7 -> 2.3, now IMOEA/D-AWN), and the reference front grows (595 -> 762
      points there), so IGD weights shift. The 1000-task comparison with IMOEA/D is within the
      sensitivity of the normalisation; do not claim it either way without a robustness check
      (e.g. normalise over the union of every policy, or per-objective ranks).
    - [x] Robustness check (2026-11-10, `scripts/robustness.py` -> results/stats/robustness.{md,csv}):
      final vs IMOEA/D (30 + 30 seeds, Wilcoxon, alpha 0.05) with the normalisation and reference
      front taken from four pools: paper10 (current tables), union12 (+ ip_moea, trained_stable),
      strong5 (final, IMOEA/D, IMOEA/D-AWN, NSGA-II mid / int), pair2 (final, IMOEA/D).
      - Suite A: final better on IGD and HV_jin on 12/12 in every pool.
      - Suite B, HV_jin: 42/2/1 in paper10, union12, strong5; 42/1/2 in pair2. Stable.
      - Suite B, IGD: 43/1/1, 42/1/2, 33/9/3, 34/6/5. Every change is at 1000 tasks (plus
        montage_100@x100, a tie in 3 of 4). Without the weak baselines in the pool, IMOEA/D has the
        lower IGD on 6-9 of the 15 1000-task instances (cybershake x1 / x20, epigenomics x1 / x20,
        inspiral x1 / x20, sipht x20; montage / sipht x1 become ties). inspiral_1000@x100 goes to
        IMOEA/D in every pool on both metrics.
      - Reading: up to 400 tasks every conclusion is pool-independent. At 1000 tasks final keeps
        the HV lead but the IGD verdict depends on what the reference front contains; the weak
        baselines add reference points (likely at the extremes) that final covers and IMOEA/D does
        not. Paper: use HV_jin as the headline metric at 1000 tasks, show the strong5 IGD as a
        sensitivity table, and state that at 1000 tasks IMOEA/D is level or ahead on IGD (consistent
        with E5: both still converging at G=1000).
    - Imprecision: IMOEA-DQN ranks 1.00 on both suites because its front collapses to a few
      solutions; report it next to front size.
    - Not updated: E5 curves and E7 Monte Carlo still use ip_moea / trained_stable (the reruns did
      not trace or save fronts). Redo with `--trace --save-front` on the 4 E5 instances if the
      figures must show the corrected baselines (80 runs).
- [x] B3. NSGA-II (midpoint and interval-dominance variants) and NSGA-III (midpoint), in
  `gat_imoea_dqn/baselines.py` (policies `nsga2_mid`, `nsga2_int`, `nsga3_mid`); IP-MOEA's loop is
  generalised over the dominance relation and stays bit-identical. Done 2026-11-05
  (`scripts/run_b3.cmd`): Suite A in the new experiment `suite_a_main` (every method in one table;
  the frozen method's 120 re-runs are identical to the ablation's `final` runs, checked file by
  file) and Suite B in `suite_b`.
  - Suite A (mean IGD): final 0.0123, nsga2_int 0.0325, nsga2_mid 0.0327, imoead 0.0352, nsga3_mid
    0.0912. final wins IGD and HV_jin in 12/12 against every baseline.
  - Suite B (mean IGD): final 0.0126, imoead 0.0217, nsga2_mid 0.0355, nsga2_int 0.0358, nsga3_mid
    0.0993. HV_jin: final wins 45/45 against every baseline (nsga2_int: 43 wins, 2 ties).
  - Caveat for the paper: on IGD both NSGA-II variants beat final in 9 cases, all of them 100-task
    workflows (cybershake/epigenomics/inspiral/montage/sipht_100 at x1, inspiral/montage/sipht_100
    at x20, sipht_100 at x100); final wins every 400- and 1000-task case. Report it. Untested
    guess: on 100 tasks the search space is small enough for plain NSGA-II to converge. Imprecision
    vs final: Suite A NSGA-II worse in 8, tie 4; Suite B nsga2_mid better in 24, worse in 16.
- [x] B4. Equal budget for all methods; wall-clock times reported
  Budget (2026-11-10, `scripts/budget_check.py`, evaluator calls counted, G=10, N=105): every
  method evaluates N(G+1) solutions except IMOEA/D-AWN, N*G (the original's loop runs generations
  1..G_max-1; 0.1% fewer at G=1000). Kept as published; say so in the setup section.
  Timing (`scripts/run_timing.cmd`, experiments timing_a / timing_b, one run at a time, G=1000,
  2 seeds, trained agent reused so no training time; median seconds per run):
  | Method | A small_20 | A medium_250 | cyber_100 | cyber_400 | cyber_1000 | cyber_2000 |
  | final | 29.2 | 33.2 | 34.6 | 41.0 | 53.1 | 61.9 |
  | imoead_orig | 27.0 | 33.7 | 37.5 | 45.2 | 59.2 | 59.0 |
  | imoead_awn | 36.9 | 45.8 | 52.6 | 60.0 | 73.4 | 74.0 |
  | imea_ied | 38.5 | 46.7 | 51.1 | 54.0 | 65.6 | 68.7 |
  | ip_moea | 15.2 | 20.9 | 21.4 | 26.7 | 38.0 | 49.4 |
  | nsga2_mid | 15.3 | 20.7 | 22.0 | 27.6 | 30.0 | 45.9 |
  | nsga2_int | 21.1 | 26.6 | 28.6 | 33.8 | 36.2 | 52.0 |
  | nsga3_mid | 18.1 | 23.0 | 24.4 | 29.3 | 32.8 | 47.7 |
  | fixed_fav | 30.5 | 37.0 | 38.5 | 44.8 | 47.1 | 64.7 |
  | trained_stable | 34.9 | 57.7 | 38.0 | 44.7 | 62.8 | 92.8 |
  - final costs about the same as IMOEA/D and the Jin backbone, less than IMOEA/D-AWN and
    IMEA-IED, and 1.3-1.7x the NSGA family (Tchebycheff ranking + neighbourhood mating).
  - IMOEA-DQN vs the fixed-action backbone: +0-56% (none at cybershake 100/400, +33% at 1000,
    +43% at 2000, +56% on Suite A medium_250): the per-generation GAT inference over the DAG.
    final drops it, which is part of its case.
- [ ] D1. Write-up material for C4: the reward-audit figures (per-step SNR by phase; reward
  ranking vs IGD ranking), the three-learner table (stable / blk / l8), GAT vs no-GAT. A full Suite A
  retrain of L8 only if a reviewer asks.

### P2 — Evaluation
- [ ] E1. Suite A (3 scales × 2 device counts × CCR ×1 / ×20) and Suite B (5 families × 3 sizes × 3 CCR)
  - [x] Suite B done 2026-11-04 (`scripts/run_suite_b.cmd`, `results/experiments/suite_b/summary.md`):
    45 cases (5 families × 100/400/1000 tasks × data ×1/×20/×100), Jin scenario, 10 seeds, G = 1000,
    2250 runs. Mean IGD: final 0.0105, IMOEA/D 0.0208, Jin backbone 0.084, IP-MOEA 0.225,
    heuristics 0.239. Final is best on IGD and HV_jin everywhere: vs IMOEA/D IGD 43 wins / 2 ties
    (inspiral_400@x1, sipht_100@x100) / 0 losses, HV_jin 45/0/0; vs every other method 45/0/0.
    IMOEA/D's IGD is ~2× final's at every size, data level and family (1.9–2.8×). The Jin backbone's
    gap grows with size (IGD 0.046 / 0.085 / 0.121 at 100 / 400 / 1000 tasks): the R_IN stall
    gets worse on larger DAGs. Imprecision: final's fronts are wider than IMOEA/D's (worse in 35/45),
    as in Suite A; the heuristics' 3-plan fronts have near-zero imprecision by construction.
- [x] E2. Seeds (trimmed 2026-11-04, user decision): 30 runs for the main comparison only
  (final, IMOEA/D, Jin backbone) on all 57 cases (Suite A 12 + Suite B 45) = 20 more seeds ×
  3 × 57 = 3420 runs. Every other method (B1, B2, B3, IP-MOEA) stays at 10 seeds; heuristics are
  deterministic. Sensitivity (E6) and scalability at 10 seeds. Remaining budget ≈ 10,000 runs
  (B1 1710, B2 1350, B3 1710, E2 3420, E6 ~1200, 2000-task scalability ~600), ≈ 22 h awake.
  Result (2026-11-10, 1080 + 4050 runs; Suite A first attempt hit a numba cache race at
  start-up, rerun with `scripts/run_e2a.cmd`). Means final / imoead_orig / fixed_fav:
  Suite A IGD 0.0111 / 0.0324 / 0.1162, HV_jin 0.814 / 0.792 / 0.631, Imprecision 29.5 / 26.5 / 25.6;
  imoead_orig vs final: IGD 0/12/0, HV 0/12/0, Imprecision 11/0/1 (imoead_orig narrower).
  Suite B IGD 0.0180 / 0.0230 / 0.1461, HV_jin 0.928 / 0.912 / 0.780, Imprecision 63.9 / 60.9 / 49.6;
  imoead_orig vs final: IGD 6/36/3, HV 1/42/2, Imprecision 26/8/11. fixed_fav loses every case on
  IGD and HV. imoead_orig's IGD wins are all 1000-task cases (cybershake x1/x20, epigenomics
  x1/x20, inspiral x20, sipht x20; it also wins HV on inspiral_1000@x100) - the cost-extreme gap.
  Note: HV vs imoead_orig at 1000 tasks is 1/42/2 here but 12/30/3 in B1, because the
  normalisation covers only these 3 policies; the paper tables need one fixed normalisation per
  instance (E3) and must say which.
- [x] E3. Metrics as above, normalised once per instance
- [x] E4. Statistics as above
  `scripts/stats.py` (2026-11-10) -> results/stats/suite_{a,b}.md and *_runs.csv. Per instance it
  pools every run of the paper set (final, imoead_orig, imoead_awn, imea_ied, ip_moea, nsga2_mid,
  nsga2_int, nsga3_mid, fixed_fav, trained_stable) from suite_a_main + suite_a_e2 / suite_b +
  suite_b_e2, and fixes one normalisation and one reference front over that pool. Heuristics, the
  old imoead schedule and fresh DQN are left out (heuristics would stretch the normalisation;
  report them separately). Seeds: final / imoead_orig / fixed_fav 30, the rest 10. Wilcoxon
  rank-sum vs final with Holm over the 9 policies per instance and metric; A12; Friedman + Nemenyi.
  - Suite A (12): final better on IGD and HV against every policy on 12/12, A12 large on 12/12;
    Friedman rank 1.00 on IGD and HV, but with 12 instances the Nemenyi CD is 3.91, so only
    IMEA-IED, IP-MOEA, NSGA-III, the Jin backbone and IMOEA-DQN are significantly behind on
    ranks; the per-instance tests carry the evidence. Imprecision: final ranks 7.75 of 10; IMOEA/D,
    IMOEA/D-AWN, NSGA-III, the Jin backbone and IMOEA-DQN give narrower intervals.
  - Suite B (45): IGD vs final - IMOEA/D 4/37/4 (wins cybershake_1000@x1, epigenomics_1000@x20,
    inspiral_1000@x1, inspiral_1000@x100), NSGA-II mid / int 5/40/0 (wins the five 100-task x1
    cases), every other policy 0 wins. HV_jin - IMOEA/D 2/42/1 (inspiral_1000@x100,
    sipht_1000@x20), all others 0 wins. Friedman ranks: IGD 1.42 vs next 2.64 (IMOEA/D), HV 1.09
    vs 2.73; CD 2.02, so on ranks final is significantly ahead of every policy except IMOEA/D and
    IMOEA/D-AWN (IGD gaps 1.22 / 1.58, HV 1.64 / 1.84).
    Imprecision: final ranks 7.64 of 10 (IMOEA/D 28/7/10 vs final).
  - Use these tables in the paper (one normalisation per instance); the per-experiment
    summaries differ because each normalises over its own policy set.
- [x] E5. IGD / HV vs generation curves for DR-IMOEA and every baseline
  Done 2026-11-10 (`scripts/run_e5.cmd`, experiments e5_a / e5_b, 400 runs, no errors). New
  `gat_imoea_dqn/trace.py` + `experiment.py --trace`: every loop stores the first interval front at
  G = 0, 10, 25, 50, 100, ..., 1000 (IMOEA/D-AWN's last loop step is 999; its end front is stored
  as 1000). Tracing touches no random stream: all 400 final fronts are identical to the same
  policy / seed in suite_a_main / suite_b (`scripts/convergence.py` checks this). Instances:
  medium_250_i0@x20 (A), montage_400@x20, cybershake_1000@x20, epigenomics_1000@x20 (B); the 10
  paper policies, 10 seeds; IMOEA-DQN with training seed 0 only. Normalisation and reference front
  as in E3, so G=1000 matches the stats tables. Output results/stats/e5_convergence.{md,csv},
  results/figures/e5_convergence.{png,pdf}.
  - medium_250: final leads from about G=150 and keeps falling (IGD 0.011 at G=1000); IMOEA/D
    0.033 at G=1000, never reaches final's G=500 value (0.026).
  - montage_400: IMOEA/D is ahead up to about G=350, final overtakes and flattens at 0.010 from
    G~700 (IMOEA/D 0.011).
  - cybershake_1000 / epigenomics_1000: IMOEA/D slightly ahead throughout (0.029 vs 0.030; 0.030
    vs 0.037), and both curves are still falling at G=1000: at 1000 tasks the budget is not
    saturated, so the gap there is convergence speed, not a plateau. HV: final catches up by
    G~600 and ends level or ahead.
  - Every other policy stalls far above (IGD 0.04-0.33); IMOEA-DQN starts worse than the initial
    population on IGD (its early actions push to one region) and only slowly recovers, the
    fixed-action backbone plateaus by G~300. Paper figure: the 2x4 panel as is.
  - Refreshed 2026-11-11 with the corrected baselines (`scripts/run_e57fix.cmd`, experiments
    e57fix_{a,b}: ip_moea_hv + imoea_dqn, 10 seeds, --trace --save-front, 80 runs, no errors; copied
    into e5_* and e7_*). Pools now include rerun_*; paper set as in stats.py. 400/400 final fronts
    identical to the table runs. G=1000 IGD final / IMOEA/D: 0.012 / 0.035, 0.011 / 0.012,
    0.028 / 0.033, 0.032 / 0.036; with the new pool final now ends ahead on both 1000-task cases
    (pool-dependent, see the robustness check under B2). IP-MOEA 0.18 / 0.13 / 0.17 / 0.16 and
    IMOEA-DQN 0.29 / 0.20 / 0.29 / 0.27 stay far above; IMOEA-DQN again starts worse than the
    initial population.
- [x] E6. Sensitivity: CCR, uncertainty width, edge count, bandwidth, energy scope, multi-core queueing
  Done 2026-11-07, no errors. Mean IGD (final / imoead_orig / nsga2_mid / fixed_fav) and final's
  record vs imoead_orig (IGD · HV_jin, better/worse/tie for imoead_orig, 5 cases):
  base 0.0082 / 0.0121 / 0.0496 / 0.122 (0/5/0 · 0/5/0); energy all-tier 0.0100 / 0.0108 / 0.113 /
  0.139 (**2/1/2 · 4/0/1**); width x0.5 0.0094 / 0.0178 (0/5/0 · 0/5/0); width x2 0.0073 / 0.0089
  (0/4/1 · **2/1/2**); bandwidth x0.5 0.0118 / 0.0160 and x2 0.0063 / 0.0100 (0/5/0 · 0/5/0);
  1 edge 0.0093 / 0.0112 (0/4/1 · 0/4/1); 6 edges 0.0108 / 0.0170 (0/4/1 · 0/5/0); edge queue
  0.0083 / 0.0106 (0/4/1 · 0/4/1). nsga2_mid and fixed_fav lose 5/5 on IGD and HV everywhere.
  Weak spots for final: all-tier energy (imoead_orig wins HV 4/5) and wide intervals (HV 2/5);
  check whether these are the same cost-extreme effect as at 1000 tasks.
  Running since 2026-11-05 (`scripts/run_e6.cmd, run_scale.cmd, run_e2.cmd (one at a time)`, logs results/experiment_{e6,scale,e2}.log; order E6 ->
  2000-task scalability -> E2). Cases: the five 400-task workflows at x20; policies final,
  imoead_orig (reported as IMOEA/D), fixed_fav (Jin backbone), nsga2_mid; 10 seeds. Levels, each its
  own experiment: e6_base (copied from suite_b), e6_energy_all, e6_width05 / e6_width2 (interval
  half-widths x0.5 / x2, `--width-scale`), e6_bw05 / e6_bw2 (wireless bandwidth, scenarios
  jin_bw05 / jin_bw2), e6_edges1 / e6_edges6 (scenarios jin_e1 / jin_e6), e6_edgequeue (edge servers
  queue, `--edge-queue`). CCR is already covered by Suite B's x1 / x20 / x100. 1600 runs.
  Scalability: `scale_2000`, 5 families x 2000 tasks x {x1, x20, x100}, same 4 policies, 600 runs.
  Scalability result (2026-11-07, 600/600 runs): mean IGD final 0.0699 / imoead_orig 0.0692,
  HV_jin 0.886 / 0.857, Imprecision 63.2 / 74.6. imoead_orig vs final: IGD 6/5/4, HV 0/13/2,
  Imprecision 1/13/1. nsga2_mid and fixed_fav lose 15/15 on IGD and HV. Front check
  (`front_cmp` on scale_2000): final dominates 64-79% of imoead_orig's front, about 0% the other
  way, and final wins HV 13/15. The cost-extreme gap is larger than at 1000 tasks: normalised min
  cost final 0.20-0.25 vs imoead_orig 0.12-0.15. imoead_orig's IGD wins come from that extreme
  (the pooled reference includes it), not from a better front. Same structural effect as B1 at
  1000 tasks, now visible on IGD instead of HV.
  Extreme-coverage pilot (2026-11-07, `scripts/run_pilot.cmd`, experiment `pilot_ext`; 8 cases:
  montage/sipht_400@x20, the three 1000-task investigation cases, cybershake_2000@x100,
  epigenomics/inspiral_2000@x20; 5 seeds; final / imoead_orig runs reused). Variants of final:
  `final_rb` (Rank C Tchebycheff normalised by the run's running ideal / nadir, as IMOEA/D does),
  `final_ax` (6 extra weight vectors near each axis), `final_rbax` (both); diagnostic
  `imoead_noadapt` (imoead_orig without vector adaptation; it moves only 3 of 105 vectors anyway).
  Mean over the 8 cases (IGD, HV_jin, normalised min cost): final 0.0439 / 0.8812 / 0.196,
  imoead_orig 0.0368 / 0.8650 / 0.145, final_rb 0.0464 / 0.8747 / 0.214, final_ax 0.0442 / 0.8796 /
  0.203, final_rbax 0.0463 / 0.8766 / 0.216, imoead_noadapt 0.0372 / 0.8650 / 0.143.
  - No variant reaches cheaper solutions; none improves IGD or HV. imoead_noadapt equals
    imoead_orig, so adaptation is not the cause; the normalisation is not either.
  - At 400 tasks final reaches the cheaper end (0.10 vs 0.16-0.18); at 1000-2000 it reaches the
    makespan and energy extremes better and IMOEA/D the cost extreme. The remaining differences are
    IMOEA/D's steady-state per-subproblem replacement and weight-space mating.
  - Decision: keep final frozen; report the cost-extreme gap as a limitation (R3) with the
    dominance analysis and this pilot. The switches (`tch_bounds`, `axis_weights`) stay, default off.
  E2: `suite_a_e2`, `suite_b_e2` (final, imoead_orig, fixed_fav; seeds 0-9 copied, 10-29 new;
  3420 runs). The E2 set uses imoead_orig as IMOEA/D (B1 decision).
- [x] E7. Monte Carlo check of final fronts under sampled crisp values
  Done 2026-11-10. Fronts: `scripts/run_e7.cmd` (experiments e7_a / e7_b, the E5 instances and
  policies, 10 seeds, `--save-front` stores each final front's X / O; 400 runs, no errors).
  `scripts/montecarlo.py`: 200 crisp scenarios per instance, every interval input (C, D, f, R,
  e) drawn uniformly and independently, common to all policies; ~9k solutions per instance.
  Output results/stats/e7_montecarlo.{md,csv}.
  - Containment 100.00% on all four instances: every crisp outcome lies inside its predicted
    interval in all three objectives (the interval evaluator is conservative, as designed).
  - Crisp values sit slightly below the interval midpoint (mean position 0.44-0.48). The 95%
    crisp spread covers 27-45% of the interval width on the Suite B instances, but only 9% for
    makespan / energy on Suite A medium_250 (many independent applications summed: independent
    draws average out, the interval is the joint worst case). Worth a sentence in R3: the
    intervals are worst-case bounds, not likely ranges.
  - Ranking holds: Kendall tau between the E3 interval IGD order and the Monte Carlo IGD order
    0.96 / 0.91 / 0.96 / 1.00.
  - final beats every policy in 100% of scenarios on HV and IGD on medium_250; on montage_400
    and cybershake_1000 on HV in 100% of scenarios against all, and on IGD against all except
    IMOEA/D (montage 0%: IMOEA/D 0.0103 vs 0.0115 although E3 interval IGD favours final;
    cybershake_1000 88% although E3 favours IMOEA/D). epigenomics_1000: IMOEA/D better (final
    wins 19% HV, 10% IGD), matching E3 / E5. Against all other policies: 100% everywhere.
  - Refreshed 2026-11-11 with the corrected baselines (same scenarios). Containment 100% on all 4;
    Kendall tau 0.91 / 0.96 / 1.00 / 0.91. final vs IMOEA/D, share of scenarios final better
    (HV / IGD): medium_250 100 / 100, montage_400 100 / 0, cybershake_1000 100 / 100,
    epigenomics_1000 10 / 100. The 1000-task IGD shares moved with the pool (it also defines the
    per-scenario crisp reference), so read them with the robustness caveat. Against IP-MOEA and
    IMOEA-DQN final is better in 100% of scenarios on both metrics everywhere.

### P3 — Paper
- [~] R1. One script builds every table and figure (LaTeX)
  Built 2026-11-11: `scripts/build_paper.py` -> paper/tables/*.tex (14 booktabs tables), paper/figures/
  (e5_convergence, a3_stall, a3_tradeoff PDFs, copied from results/figures) and paper/preview.tex
  (elsarticle, inputs everything; `\ours` is the method's name, one line to rename). Each scored table
  normalises per instance over its own policies, as E3; scores cached in results/stats/paper_*_runs.csv
  (`--rescore` to redo, ~30 min). Checked: Suite A / B counts and Friedman ranks identical to
  scripts/stats.py; column counts, braces and math balanced in every table. Not compiled (no LaTeX
  on this machine; use Overleaf).
  Tables: suite{a,b}_{igd,hv_jin,imprecision} (sideways, mean (sd) + Wilcoxon/Holm symbols), friedman,
  heuristics (scored on the main normalisation without joining the pool), robustness, ablation
  (build-up vs final, leave-one-out / sensitivity vs the T=20 default as in A1), sensitivity (E6),
  scalability (2000 tasks), timing (B4), montecarlo (E7).
  - [ ] Timing rows for ip_moea_hv and imoea_dqn are missing (the table prints a TODO): rerun
    `scripts/run_timing.cmd` with those two policies (6 instances x 2 seeds, one at a time).
  - [ ] Still to add: C4 figures (D1), front plots if wanted.
- [ ] R2. System-model figure, pseudocode, complexity, parameter table
- [ ] R3. Threats to validity and limitations (no link contention, no receive energy, interval
  bounds assumed to hold, fixed action chosen on training instances, at 1000-2000 tasks IMOEA/D
  reaches cheaper solutions though final dominates the overlap; see the E6 / pilot notes)

## Risks

- C3 judged incremental ("just MSOPS inside Jin et al.") → W1 confirmed the rank is MSOPS's; make the diagnosis (tie
  counts), the ablation and the interval treatment explicit; the DAG model (C1) is the second pillar.
- Tuning on the test set → A2 picks the action on training instances only.
- Our IMOEA/D is weaker or stronger than the original → document every parameter from Gan et al.
  (2023); Jin et al. report IMOEA-DQN > IMOEA/D, we find the opposite on the DAG model: explain the
  differences (DAG model, budget, re-implementation) and report it plainly.
- Re-implementing the remaining baselines without public code is the largest cost → after A1/A2.
- Closed: the learned controller does not beat the best fixed action (three designs, 2026-10-29);
  now reported as C4 analysis.
