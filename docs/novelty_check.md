# W1: novelty check (2026-11-02)

Web search plus full text where open access (PMC, MOEA Framework source). The ScienceDirect
and DOAJ pages (Egyptian Informatics Journal, Swarm and Evolutionary Computation) refused
automated access, so for those papers only abstracts were read. **Before writing, download
the papers marked "abstract only" through institutional access and confirm the differences
claimed below.**

## Verdict

| Contribution | Prior art found | Still new | Risk |
|---|---|---|---|
| C1 interval DAG model, cloud–edge–end | Shi et al. 2023 (interval + dependent workflow + many devices / edge servers, but migration); grey-number DAG scheduling 2026 (multi-cloud, PPO); crisp DAG offloading MOEAs (many) | Interval critical-path DAG offloading over device + edge + cloud, placement **and** order encoded, exact reduction to Jin et al. | Medium: must cite and separate from Shi et al. 2023 |
| C3 Rank C | **MSOPS (Hughes 2003)** ranks each solution by its best position over the per-weight min-max (Tchebycheff) orderings; that is our R_TCH. Global WASF-GA (2017) classifies fronts by ASF over weight vectors. Dominance + decomposition hybrids: MOEA/DD, θ-DEA, NSGA-III, secondary-ranking NSGA-II (Palakonda & Mallipeddi 2017) | Interval version (mean of the lower- and upper-bound Tchebycheff values) used as the convergence tier inside Jin et al.'s dominance / uncertainty three-tier selection; the R_IN-stall diagnosis; objective-space neighbourhood mating in that design. **No paper found that does this for interval objectives.** | **High if pitched as a new ranking.** Present it as an interval MSOPS rank, cite Hughes, and claim the integration, the diagnosis and the evidence, not the rank itself |
| C4 DQN analysis | Many RL-assisted MOEAs; negative results are rare | Reward audit (SNR), three learner designs, GAT adds nothing | Low (supporting material) |
| C5 benchmark | — | Released interval DAG instances | Low |

Bottom line: the paper is publishable as an application + method-integration paper (EAAI's
profile), not as a new selection operator. The novelty claim must be "first interval DAG
offloading model in cloud–edge–end, and a diagnosis-driven fix of the predecessor's selection
(interval MSOPS rank as the convergence tier) that beats the interval baselines", with MSOPS
credited.

## Evidence

### Ranking (C3)
- **MSOPS**, E. J. Hughes, "Multiple Single Objective Pareto Sampling", CEC 2003, 2678–2684;
  MSOPS-II, CEC 2007. MOEA Framework's `MSOPSRankedPopulation` (source read): per weight
  vector a weighted min-max score and a VADS score; each column sorted; a solution's rank vector
  sorted ascending and compared lexicographically, so the best (smallest) position over the
  weight vectors decides first; truncation keeps the best ranks. Ours: one score per weight
  vector (interval midpoint Tchebycheff), rank = minimum position, ties by ICD instead of the
  next-best position.
- **Global WASF-GA**, Saborido, Ruiz, Luque, Evolutionary Computation 25(2):309–349, 2017:
  fronts by achievement scalarizing function (Tchebycheff-type) over weight vectors — a close
  relative; cite it.
- Dominance / decomposition hybrids: MOEA/DD (Li et al. 2015), θ-DEA (Yuan et al. 2016),
  NSGA-III, RVEA, HS-MOEA (Sci. Rep. 2022, dominance + reference vectors + indicator),
  Palakonda & Mallipeddi, IEEE Access 2017 (average rank / weighted sum as a secondary
  criterion after Pareto dominance).
- Loss of dominance selection pressure when most of the population is non-dominated:
  Ishibuchi et al. (many-objective line) — cite for the R_IN-stall diagnosis; our new part
  is showing it already happens with three interval objectives under possibility-degree dominance.

### Interval MOEAs (decomposition)
- **IMOEA/D-DAW**, Zhang, Xu, Yi, Wang, Zong, Sci. Rep. 16:19141, 2026 (PMC13279808, read):
  interval Tchebycheff / PBI on interval objective vectors, interval ranking (Sengupta–Pal) to
  compare scalar intervals, MOEA/D per-subproblem replacement, dynamic neighbourhoods and
  sparsity-driven weight adaptation. No merged-population rank, no dominance / uncertainty
  tiers. Baselines: IP-MOEA, IMOEA/D-AWN, EG-IMOEA → **add it to related work; consider it
  as a baseline** (newest interval decomposition method).
- **IMOEA/D-AWN**, Egyptian Informatics Journal 2023 (abstract only): adaptive weights and
  neighbourhoods; interval sparsity + interval uncertainty ranking. Already in our B1 list.
- Interval many-objective with dynamic dual-sequence (SWEVO 2025), dynamic relation adjustment
  (SWEVO 2025), information-entropy dominance (SWEVO 2024), synthesized-ranking NSGA-II
  (interval) — all abstract only; none reported to use a weight-vector rank as a tier.

### Interval / uncertain DAG and workflow offloading (C1)
- **Shi, Zhao, Li, Zhang, Cui**, "Workflow migration in uncertain edge computing environments
  based on interval many-objective evolutionary algorithm", Egyptian Informatics Journal
  24(4):100418, 2023 (abstract only): workflows with inter-task dependencies, many devices and
  edge servers, bandwidth / computing power as intervals; objectives migration delay, maximum
  completion time, energy, load balance; MI-MaOEA. **Closest prior work for C1.** Differences to
  confirm from the full text: migration triggered by mobility (not offloading placement), no
  cloud tier (?), how completion time handles precedence (critical path?), no ordering string.
- Zhang, Cui et al., Information Sciences 2022: interval many-objective cloud task scheduling
  (independent tasks).
- Grey-number PPO for uncertainty-aware multi-objective DAG scheduling in multi-cloud, 2026
  (Springer): interval grey numbers for DAGs, but RL and multi-cloud only.
- Crisp dependent-task offloading MOEAs: FGCS 2023 (MOEA/D, cloud–edge–end, dependencies),
  FGCS 2025 (IIoT dependency-aware, reliability), multi-workflow offloading (2025).
- Jin et al., EAAI 178:115043, 2026 (IMOEA-DQN): independent tasks, interval latency / energy /
  cost — the predecessor.

## What this changes in the plan
1. Rename C3 in the paper: "interval MSOPS rank as the convergence tier"; cite MSOPS and
   Global WASF-GA in the method section, not only related work.
2. Strengthen the evidence instead of the claim: (a) A3 stall figure (tie counts R_IN vs
   R_TCH); (b) a baseline that uses a dominance / decomposition hybrid on midpoints
   (NSGA-III or MOEA/DD) so "why not an off-the-shelf hybrid?" is answered with numbers;
   (c) IMOEA/D-DAW as an extra interval baseline if feasible.
3. Related work must separate C1 from Shi et al. 2023 explicitly (get the full text).
4. Since C3's novelty is modest, C1 (the DAG model and benchmark) carries more weight: run
   Suite B (real workflows) and the DAG heuristics (HEFT-style) before the hand-written
   interval baselines.
