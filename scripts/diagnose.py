"""Why isn't the DQN learning? Measure the learning signal directly.

    python scripts/diagnose.py --agents-from hv              # full run, uses results/experiments/hv agents
    python scripts/diagnose.py --name smoke --workflows montage_60 --snap-gens 5 --horizon 3 --replicates 2

Three measurements, all on the held-out test workflows:

A. Q-value gaps. Each trained agent runs greedily once per workflow; at every
   generation the Q-values of all 28 actions are recorded. Reports the gap
   between the best and second-best action, the max-min spread, how often the
   greedy action changes between generations, and the share of the top action.

B. Immediate reward signal-to-noise. Population snapshots are taken from a
   random-action run at early / middle / late generations. From each snapshot,
   J offspring sets are drawn; for each offspring set all 28 actions are applied
   to the SAME merged pool and the reward is computed. A two-way decomposition
   splits reward variance into the part explained by the action (eta2_action),
   by the offspring draw (eta2_offspring, uncontrollable noise) and the rest.

C. Multi-step effect. From each snapshot, every action is followed for H
   generations (J' replicates) and the discounted return (gamma = 0.9) and the
   final front quality (PIBI, HV on one shared normalization per snapshot) are
   measured. This is the quantity Q(s, a) should estimate, so the agents'
   Q-values at the snapshot state are compared with it (Spearman rank
   correlation, and the regret of the greedy action).

Outputs: results/diagnostics/<name>/{snapshots/, reward_matrices/, qtraces/, summary.json, summary.md}
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import json
import multiprocessing as mp
import pathlib
import pickle
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TEST_SET = ["montage_200", "epigenomics_200", "cybershake_200", "inspiral_200", "sipht_200"]
GAMMA = 0.9


def _setup():
    import torch
    torch.set_num_threads(1)


def _load(name):
    from dataset_tools.loader import load_manifest, load_scenario, load_workflow
    manifest = load_manifest(ROOT / "datasets")
    return (load_workflow(ROOT / "datasets" / manifest[name]["path"]),
            load_scenario(ROOT / "datasets" / "scenarios" / "default.yaml"))


class Env:
    """One workflow's evaluator, operators and selection, stepped manually."""

    def __init__(self, wfname, generations):
        from gat_imoea_dqn import DagIndex, IntervalEvaluator
        self.wf, self.sc = _load(wfname)
        self.dag = DagIndex(self.wf)
        self.ev = IntervalEvaluator(self.wf, self.sc)
        from gat_imoea_dqn.chromosome import allowed_for
        self.allowed = allowed_for(self.wf, self.sc)
        self.generations = generations

    def offspring(self, pop, F_lo, F_hi, rng):
        from gat_imoea_dqn import make_offspring
        off, _ = make_offspring(pop, F_lo, F_hi, self.dag, self.allowed, rng)
        G_lo, G_hi = self.ev.evaluate_population(off.X, off.O)
        return off, G_lo, G_hi

    @staticmethod
    def select(pop, F_lo, F_hi, off, G_lo, G_hi, action, rng):
        import numpy as np
        from gat_imoea_dqn import Population, environmental_selection
        merged = Population.concat(pop, off)
        M_lo, M_hi = np.concatenate([F_lo, G_lo]), np.concatenate([F_hi, G_hi])
        keep, _ = environmental_selection(M_lo, M_hi, len(pop), action, rng, genotypes=(merged.X, merged.O))
        return merged[keep], M_lo[keep], M_hi[keep]


# ------------------------------------------------------------ phase 1 ---
def make_snapshots(spec) -> str:
    """Random-action run; save (population, objectives, tracker, state) at chosen generations."""
    _setup()
    import numpy as np
    from gat_imoea_dqn import init_population
    from gat_imoea_dqn.selection import selection_actions
    from gat_imoea_dqn.state import PopulationTracker

    out = pathlib.Path(spec["out"])
    wanted = sorted(spec["snap_gens"])
    if all((out / f"{spec['workflow']}_t{t}.pkl").exists() for t in wanted):
        return f"skip snapshots {spec['workflow']}"
    env = Env(spec["workflow"], spec["generations"])
    A = selection_actions()
    rng = np.random.default_rng(spec["seed"])
    tracker = PopulationTracker(spec["generations"])
    pop = init_population(env.wf, env.sc, spec["pop"], rng, env.dag)
    F_lo, F_hi = env.ev.evaluate_population(pop.X, pop.O)
    tracker.observe(F_lo, F_hi)
    s = tracker.features(pop.X, F_lo, F_hi, 0)
    for t in range(max(wanted) + 1):
        if t in wanted:
            with open(out / f"{spec['workflow']}_t{t}.pkl", "wb") as f:
                pickle.dump({"workflow": spec["workflow"], "t": t, "pop": pop, "F_lo": F_lo, "F_hi": F_hi,
                             "tracker": copy.deepcopy(tracker), "state": s}, f)
        off, G_lo, G_hi = env.offspring(pop, F_lo, F_hi, rng)
        tracker.observe(G_lo, G_hi)
        a = A[rng.integers(len(A))]
        pop, F_lo, F_hi = env.select(pop, F_lo, F_hi, off, G_lo, G_hi, a, rng)
        s = tracker.features(pop.X, F_lo, F_hi, t + 1)
    return f"snapshots {spec['workflow']} at {wanted}"


# ------------------------------------------------------------ phase 2 ---
def reward_matrices(spec) -> str:
    """Immediate reward R1[j, a] and H-step returns G[j, a] from one snapshot."""
    _setup()
    import numpy as np
    from gat_imoea_dqn.selection import selection_actions

    out = pathlib.Path(spec["out"])
    if out.exists():
        return f"skip {out.name}"
    snap = pickle.loads(pathlib.Path(spec["snapshot"]).read_bytes())
    env = Env(snap["workflow"], spec["generations"])
    A = selection_actions()
    pop0, F_lo0, F_hi0, tr0 = snap["pop"], snap["F_lo"], snap["F_hi"], snap["tracker"]

    J = spec["replicates_immediate"]
    R1 = np.zeros((J, len(A)))
    for j in range(J):
        off, G_lo, G_hi = env.offspring(pop0, F_lo0, F_hi0, np.random.default_rng(1000 + j))
        for a in range(len(A)):
            tr = copy.deepcopy(tr0)
            tr.observe(G_lo, G_hi)
            p1, l1, h1 = env.select(pop0, F_lo0, F_hi0, off, G_lo, G_hi, A[a], np.random.default_rng(2000 + j))
            R1[j, a] = tr.reward((F_lo0, F_hi0), (l1, h1))

    Jm, H = spec["replicates_return"], spec["horizon"]
    G = np.zeros((Jm, len(A)))
    finals = []                     # (j, a, F_lo, F_hi) of the population after H generations
    for j in range(Jm):
        for a in range(len(A)):
            rng = np.random.default_rng(5000 + j)
            tr = copy.deepcopy(tr0)
            pop, F_lo, F_hi = pop0, F_lo0, F_hi0
            ret = 0.0
            for k in range(H):
                off, G_lo, G_hi = env.offspring(pop, F_lo, F_hi, rng)
                tr.observe(G_lo, G_hi)
                npop, nlo, nhi = env.select(pop, F_lo, F_hi, off, G_lo, G_hi, A[a], rng)
                ret += GAMMA ** k * tr.reward((F_lo, F_hi), (nlo, nhi))
                pop, F_lo, F_hi = npop, nlo, nhi
            G[j, a] = ret
            finals.append((j, a, F_lo, F_hi))

    # final-population quality on one shared normalization for this snapshot
    from gat_imoea_dqn.metrics import hypervolume_3d
    from gat_imoea_dqn.selection import das_dennis
    from gat_imoea_dqn.state import pibi
    ideal = np.min([f[2].min(0) for f in finals], axis=0)
    nadir = np.max([f[3].max(0) for f in finals], axis=0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    W = das_dennis(6, 3)
    PIBI = np.zeros((Jm, len(A)))
    HV = np.zeros((Jm, len(A)))
    for j, a, lo, hi in finals:
        nlo, nhi = (lo - ideal) / span, (hi - ideal) / span
        PIBI[j, a] = pibi(nlo, nhi, W)
        HV[j, a] = hypervolume_3d((nlo + nhi) / 2, [1.1] * 3)
    np.savez(out, R1=R1, G=G, PIBI=PIBI, HV=HV, state=snap["state"])
    return f"done {out.name}"


# ------------------------------------------------------------ phase 3 ---
def q_trace(spec) -> str:
    """Greedy run of one agent on one workflow, recording Q(s_t, .) every generation."""
    _setup()
    import numpy as np
    from gat_imoea_dqn.algorithm import RunConfig, run
    from gat_imoea_dqn.dqn import DQNAgent

    out = pathlib.Path(spec["out"])
    if out.exists():
        return f"skip {out.name}"
    wf, sc = _load(spec["workflow"])
    agent = DQNAgent.load(spec["agent"])
    qs = []
    orig = agent.act

    def act(gid, s, t, rng, greedy=False):
        qs.append(agent.q_values(gid, s))
        return orig(gid, s, t, rng, greedy=greedy)
    agent.act = act
    run(wf, sc, RunConfig(pop_size=spec["pop"], generations=spec["generations"], seed=spec["seed"],
                          policy="dqn", greedy=True, learn=False), agent=agent, graph_key=spec["workflow"])
    np.save(out, np.asarray(qs))
    return f"done {out.name}"


# ------------------------------------------------------------ summary ---
def _rank(x):
    import numpy as np
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a, b) -> float:
    import numpy as np
    ra, rb = _rank(np.asarray(a)), _rank(np.asarray(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def two_way(R):
    """Variance shares of a (replicate x action) matrix."""
    import numpy as np
    m = R.mean()
    ss_t = ((R - m) ** 2).sum()
    ss_a = R.shape[0] * ((R.mean(0) - m) ** 2).sum()
    ss_j = R.shape[1] * ((R.mean(1) - m) ** 2).sum()
    if ss_t <= 0:
        return 0.0, 0.0, 0.0
    return float(ss_a / ss_t), float(ss_j / ss_t), float((ss_t - ss_a - ss_j) / ss_t)


def summarize(out: pathlib.Path, agents: dict) -> None:
    import numpy as np
    from gat_imoea_dqn.dqn import DQNAgent
    from gat_imoea_dqn.selection import selection_actions

    A = selection_actions()
    fav = int(np.argmin(np.abs(A - np.array([5 / 6, 1 / 6, 0])).sum(1)))
    S = {"A_q_gaps": {}, "B_immediate": [], "C_multistep": [], "D_q_vs_truth": {}}

    # A: Q gaps along greedy trajectories
    for name in agents:
        rows = []
        for p in sorted((out / "qtraces").glob(f"{name}__*.npy")):
            Q = np.load(p)
            srt = np.sort(Q, axis=1)
            gap = srt[:, -1] - srt[:, -2]
            spread = srt[:, -1] - srt[:, 0]
            arg = Q.argmax(1)
            rows.append({"gap_median": float(np.median(gap)), "spread_median": float(np.median(spread)),
                         "q_abs_mean": float(np.abs(Q).mean()),
                         "switch_rate": float(np.mean(arg[1:] != arg[:-1])),
                         "top_action_share": float(np.bincount(arg).max() / len(arg)),
                         "distinct_actions": int(len(np.unique(arg)))})
        if rows:
            S["A_q_gaps"][name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    # B / C / D: per snapshot
    snaps = sorted((out / "reward_matrices").glob("*.npz"))
    loaded_agents = {n: DQNAgent.load(p) for n, p in agents.items()}
    from dataset_tools.loader import load_manifest, load_workflow
    manifest = load_manifest(ROOT / "datasets")
    for p in snaps:
        d = np.load(p)
        wfname, t = p.stem.rsplit("_t", 1)
        R1, G, PIBI, HV = d["R1"], d["G"], d["PIBI"], d["HV"]
        ea, ej, er = two_way(R1)
        within_sd = float(np.sqrt(np.mean(R1.var(axis=0, ddof=1)))) if R1.shape[0] > 1 else float("nan")
        between_sd = float(R1.mean(0).std(ddof=1))
        best_per_rep = R1.argmax(1)
        S["B_immediate"].append({
            "snapshot": p.stem, "reward_mean": float(R1.mean()), "eta2_action": ea, "eta2_offspring": ej,
            "eta2_interaction": er, "action_sd": between_sd, "noise_sd": within_sd,
            "snr": between_sd / within_sd if within_sd > 0 else float("nan"),
            "best_action_agreement": float(np.mean(best_per_rep == np.bincount(best_per_rep).argmax())),
            "reward_fav_minus_mean": float(R1.mean(0)[fav] - R1.mean())})
        gm, pm, hm = G.mean(0), PIBI.mean(0), HV.mean(0)
        ga, gj, _ = two_way(G)
        pa, _, _ = two_way(PIBI)
        S["C_multistep"].append({
            "snapshot": p.stem, "eta2_action_return": ga, "eta2_action_finalPIBI": pa,
            "eta2_action_finalHV": two_way(HV)[0],
            "return_best_minus_worst": float(gm.max() - gm.min()), "return_noise_sd":
                float(np.sqrt(np.mean(G.var(axis=0, ddof=1)))) if G.shape[0] > 1 else float("nan"),
            "best_action_by_return": A[gm.argmax()].round(3).tolist(),
            "best_action_by_PIBI": A[pm.argmin()].round(3).tolist(),
            "fav_rank_by_PIBI": int(_rank(pm)[fav]) + 1,
            "corr_return_vs_PIBI": spearman(gm, -pm), "corr_immediate_vs_return": spearman(R1.mean(0), gm)})
        wf = load_workflow(ROOT / "datasets" / manifest[wfname]["path"])
        for name, ag in loaded_agents.items():
            gid = ag.register_graph(wfname, wf)
            q = ag.q_values(gid, d["state"])
            S["D_q_vs_truth"].setdefault(name, []).append({
                "snapshot": p.stem, "spearman_q_vs_return": spearman(q, gm),
                "spearman_q_vs_finalPIBI": spearman(q, -pm),
                "regret_return": float(gm.max() - gm[q.argmax()]),
                "regret_return_random": float(gm.max() - gm.mean()),
                "greedy_action": A[q.argmax()].round(3).tolist()})
    (out / "summary.json").write_text(json.dumps(S, indent=2))

    # ------------------------------------------------------------ markdown
    L = [f"# Learning diagnostics `{out.name}`", ""]
    L += ["## A. Q-value gaps along greedy runs (mean over workflows)", "",
          "| Agent | median gap (best − 2nd) | median spread (max − min) | mean abs Q | greedy action switches | "
          "top action share | distinct actions used |", "|---|---|---|---|---|---|---|"]
    for n, r in S["A_q_gaps"].items():
        L.append(f"| {n} | {r['gap_median']:.5f} | {r['spread_median']:.5f} | {r['q_abs_mean']:.4f} | "
                 f"{r['switch_rate']:.0%} | {r['top_action_share']:.0%} | {r['distinct_actions']:.1f} |")
    if S["B_immediate"]:
        L += ["", "## B. Immediate reward: how much does the action matter?", "",
              "η² = share of reward variance explained. Offspring = which children were drawn (not controllable).", "",
              "| Snapshot | mean reward | η² action | η² offspring | η² interaction | SNR (action sd / noise sd) | "
              "best action agrees across draws |", "|---|---|---|---|---|---|---|"]
        for r in S["B_immediate"]:
            L.append(f"| {r['snapshot']} | {r['reward_mean']:.5f} | {r['eta2_action']:.2f} | {r['eta2_offspring']:.2f} | "
                     f"{r['eta2_interaction']:.2f} | {r['snr']:.2f} | {r['best_action_agreement']:.0%} |")
        L += ["", "## C. Multi-step effect (each action followed for the horizon)", "",
              "| Snapshot | η² action (return) | η² action (final PIBI) | η² action (final HV) | return best − worst | "
              "return noise sd | best by return | best by PIBI | fixed_fav rank (PIBI) | corr(return, −PIBI) | "
              "corr(immediate, return) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in S["C_multistep"]:
            L.append(f"| {r['snapshot']} | {r['eta2_action_return']:.2f} | {r['eta2_action_finalPIBI']:.2f} | "
                     f"{r['eta2_action_finalHV']:.2f} | {r['return_best_minus_worst']:.4f} | {r['return_noise_sd']:.4f} | "
                     f"{r['best_action_by_return']} | {r['best_action_by_PIBI']} | {r['fav_rank_by_PIBI']}/28 | "
                     f"{r['corr_return_vs_PIBI']:.2f} | {r['corr_immediate_vs_return']:.2f} |")
        L += ["", "## D. Do the agents' Q-values rank actions like the measured returns?", "",
              "| Agent | Spearman Q vs return | Spearman Q vs final −PIBI | regret of greedy action | "
              "regret of a random action |", "|---|---|---|---|---|"]
        for n, rows in S["D_q_vs_truth"].items():
            f = lambda k: np.nanmean([r[k] for r in rows])  # noqa: E731
            L.append(f"| {n} | {f('spearman_q_vs_return'):.2f} | {f('spearman_q_vs_finalPIBI'):.2f} | "
                     f"{f('regret_return'):.4f} | {f('regret_return_random'):.4f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# --------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="main")
    ap.add_argument("--agents-from", default="hv", help="experiment whose agent_*.pt files are diagnosed")
    ap.add_argument("--workflows", nargs="*", default=TEST_SET)
    ap.add_argument("--snap-gens", nargs="*", type=int, default=[10, 100, 250])
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--pop", type=int, default=105)
    ap.add_argument("--replicates", type=int, default=8, help="offspring draws for the immediate reward")
    ap.add_argument("--return-replicates", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) * 3 // 4))
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out = ROOT / "results" / "diagnostics" / args.name
    for sub in ("snapshots", "reward_matrices", "qtraces"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    src = ROOT / "results" / "experiments" / args.agents_from
    agents = {p.stem.removeprefix("agent_"): str(p) for p in sorted(src.glob("agent_*_t*.pt"))}
    if not agents:
        print(f"no agents found in {src}")
        return 1
    if args.summary_only:
        summarize(out, agents)
        return 0

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    snap_specs = [{"workflow": w, "snap_gens": args.snap_gens, "generations": args.generations, "pop": args.pop,
                   "seed": 777, "out": str(out / "snapshots")} for w in args.workflows]
    q_specs = [{"workflow": w, "agent": p, "generations": args.generations, "pop": args.pop, "seed": 0,
                "out": str(out / "qtraces" / f"{n}__{w}.npy")} for n, p in agents.items() for w in args.workflows]
    with mp.Pool(args.workers) as pool:
        q_async = pool.map_async(q_trace, q_specs)                       # runs alongside the snapshots
        for m in pool.imap_unordered(make_snapshots, snap_specs):
            log(m)
        r_specs = [{"snapshot": str(p), "generations": args.generations, "replicates_immediate": args.replicates,
                    "replicates_return": args.return_replicates, "horizon": args.horizon,
                    "out": str(out / "reward_matrices" / f"{p.stem}.npz")}
                   for p in sorted((out / "snapshots").glob("*.pkl"))]
        for m in pool.imap_unordered(reward_matrices, r_specs):
            log(m)
        q_async.wait()
        log(f"{len(q_specs)} Q-value traces done")
    summarize(out, agents)
    return 0


if __name__ == "__main__":
    sys.exit(main())
