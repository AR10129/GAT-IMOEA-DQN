"""Train agents, then compare them with baselines and ablations on held-out workflows.

    python scripts/experiment.py --name main --policies trained fresh random fixed --train-seeds 0
    python scripts/experiment.py --name ablation --reuse-from main          # all policies, 3 training seeds
    python scripts/experiment.py --name smoke --train-episodes 2 --generations 20 --seeds 2 \
        --train-set montage_60 --test-set montage_100

Policies (same eval seed = same initial population for every policy):
  trained         greedy DQN with GAT, from each training seed; no exploration, no learning
  trained_nogat   greedy DQN trained and run with the graph embedding fixed at 0 (GAT ablation)
  trained_zeroed  the `trained` agents, but g_t = 0 at test time (does the policy use the graph?)
  trained_hv      like `trained`, but trained with the hypervolume-improvement reward
  trained_nogat_hv  like `trained_nogat`, with the hypervolume-improvement reward
  trained_fix     like `trained`, trained with running reward standardization and an epsilon floor of 0.1
  trained_nogat_fix like `trained_nogat`, with the same two learner fixes
  schedule        hand-made: linear from (1, 0, 0) to (1/6, 5/6, 0) over the run, snapped to the grid
  fresh           new DQN learning online from scratch (the PDF's single-run setting)
  random          uniform random action each generation
  fixed           always (1/3, 1/3, 1/3)
  fixed_fav       always (5/6, 1/6, 0), the action the first trained agent used most

Agents train in background processes while jobs that need no agent run in a
worker pool. Finished runs and agents are skipped, so an interrupted run
resumes; --reuse-from copies matching runs and agents from another experiment.

Outputs in results/experiments/<name>/: runs/*.json, agent_<variant>_t<seed>.pt,
train_log_<variant>_t<seed>.json, summary.json and summary.md. Indicators use
one shared normalization per test workflow (ideal = min lower bound, nadir =
max upper bound over every front of that workflow, across all policies):
  PIBI    interval PBI indicator (lower is better)
  HV_mid  hypervolume of front midpoints, reference 1.1 (higher is better)
  HV_hi   hypervolume of front upper bounds, i.e. worst case (higher is better)
  U1      mean normalized interval length sum (lower = less uncertain)
For agent-based policies, each eval seed's value is the mean over training seeds.
"""
from __future__ import annotations

import os

# One thread per process: without this, numpy's BLAS, torch and numba each
# start a thread per core in every worker and the pool thrashes (20x slower).
# Must be set before numpy / torch are imported; spawned workers inherit it.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
import pathlib
import re
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POLICIES = {
    "trained":        {"kind": "greedy", "variant": "gat"},
    "trained_nogat":  {"kind": "greedy", "variant": "nogat"},
    "trained_zeroed": {"kind": "greedy", "variant": "gat", "zero_graph": True},
    "trained_hv":     {"kind": "greedy", "variant": "gat_hv"},
    "trained_nogat_hv": {"kind": "greedy", "variant": "nogat_hv"},
    "trained_fix":    {"kind": "greedy", "variant": "gat_fix"},
    "trained_nogat_fix": {"kind": "greedy", "variant": "nogat_fix"},
    "schedule":       {"kind": "schedule"},
    "fresh":          {"kind": "fresh"},
    "random":         {"kind": "random"},
    "fixed":          {"kind": "fixed", "action": (1 / 3, 1 / 3, 1 / 3)},
    "fixed_fav":      {"kind": "fixed", "action": (5 / 6, 1 / 6, 0.0)},
}
TRAIN_SET = ["montage_100", "epigenomics_100", "cybershake_100", "inspiral_100", "sipht_100"]
TEST_SET = ["montage_200", "epigenomics_200", "cybershake_200", "inspiral_200", "sipht_200"]
RUN_NAME = re.compile(r"^(?P<wf>.+?)__(?P<pol>[a-z_]+?)(?:__t(?P<t>\d+))?__s(?P<s>\d+)\.json$")


def _limit_threads() -> None:
    import torch
    torch.set_num_threads(1)


def _load(name, env=None):
    """Workflow + scenario for one run. env: scenario name, data_multiplier (None = raw
    generator sizes; a number = rescale into Jin et al.'s ranges, times that), energy_scope."""
    from dataset_tools.loader import load_manifest, load_scenario, load_workflow
    env = env or {}
    manifest = load_manifest(ROOT / "datasets")
    wf = load_workflow(ROOT / "datasets" / manifest[name]["path"])
    if env.get("data_multiplier") is not None:
        from dataset_tools.rescale import rescale_workflow
        wf = rescale_workflow(wf, data_multiplier=float(env["data_multiplier"]))
    sc = load_scenario(ROOT / "datasets" / "scenarios" / f"{env.get('scenario', 'default')}.yaml")
    return wf, sc


def _ecfg(env=None):
    from gat_imoea_dqn.evaluator import EvaluatorConfig
    return EvaluatorConfig(energy_scope=(env or {}).get("energy_scope", "all"))


def agent_path(out: pathlib.Path, variant: str, tseed: int) -> pathlib.Path:
    return out / f"agent_{variant}_t{tseed}.pt"


# ---------------------------------------------------------------- train ---
def train(out_dir: str, variant: str, tseed: int, workflows, episodes: int, generations: int, pop: int,
          env=None) -> None:
    _limit_threads()
    import numpy as np
    from gat_imoea_dqn.algorithm import RunConfig, run
    from gat_imoea_dqn.dqn import DQNAgent, DQNConfig

    out = pathlib.Path(out_dir)
    fix = variant.endswith("_fix")
    agent = DQNAgent(DQNConfig(seed=tseed, use_graph=not variant.startswith("nogat"),
                               reward_scaling="running" if fix else "none", eps_floor=0.1 if fix else 0.0))
    reward = "hv" if variant.endswith("_hv") else "pibi"
    data = {w: _load(w, env) for w in workflows}
    log_path = out / f"train_log_{variant}_t{tseed}.json"
    tmp = out / f"agent_{variant}_t{tseed}.tmp.pt"
    log = []
    for ep in range(episodes):
        name = workflows[ep % len(workflows)]
        wf, sc = data[name]
        t0 = time.time()
        res = run(wf, sc, RunConfig(pop_size=pop, generations=generations, seed=10_000 + tseed * 1000 + ep,
                                    reward=reward, evaluator=_ecfg(env)),
                  agent=agent, graph_key=name)
        log.append({"episode": ep, "workflow": name, "seconds": round(time.time() - t0, 1),
                    "mean_reward": float(np.mean(res.history["reward"])),
                    "train_rounds": agent.train_rounds,
                    "last_loss": next((x for x in reversed(res.history["loss"]) if x is not None), None)})
        agent.save(tmp)
        log_path.write_text(json.dumps(log, indent=2))
    os.replace(tmp, agent_path(out, variant, tseed))


# ------------------------------------------------------------------ job ---
def job(spec) -> str:
    _limit_threads()
    from gat_imoea_dqn.algorithm import RunConfig, run
    from gat_imoea_dqn.dqn import DQNAgent

    path = pathlib.Path(spec["out"])
    if path.exists():
        return f"skip {path.name}"
    env = spec.get("env")
    wf, sc = _load(spec["workflow"], env)
    d = POLICIES[spec["policy"]]
    base = dict(pop_size=spec["pop"], generations=spec["generations"], seed=spec["seed"], evaluator=_ecfg(env))
    agent = None
    if d["kind"] == "greedy":
        agent = DQNAgent.load(spec["agent"])
        agent.zero_graph_at_inference = d.get("zero_graph", False)
        cfg = RunConfig(**base, policy="dqn", greedy=True, learn=False)
    elif d["kind"] == "fresh":
        cfg = RunConfig(**base, policy="dqn")
    elif d["kind"] == "random":
        cfg = RunConfig(**base, policy="random")
    elif d["kind"] == "schedule":
        cfg = RunConfig(**base, policy="schedule")
    else:
        cfg = RunConfig(**base, policy="fixed", fixed_action=tuple(d["action"]))
    res = run(wf, sc, cfg, agent=agent, graph_key=spec["workflow"])
    lo, hi = res.front_objectives()
    rec = {"workflow": spec["workflow"], "policy": spec["policy"], "seed": spec["seed"],
           "train_seed": spec.get("train_seed"), "seconds": round(res.seconds, 2),
           "front_lo": lo.tolist(), "front_hi": hi.tolist(),
           "actions": res.history["action"], "reward": res.history["reward"]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, path)
    return f"done {path.name} ({res.seconds:.0f}s)"


# -------------------------------------------------------------- summary ---
METRICS = ("IGD", "HV_jin", "Imprecision", "PIBI", "HV_mid", "HV_hi", "U1")
LOWER_BETTER = {"IGD": True, "HV_jin": False, "Imprecision": True, "PIBI": True, "HV_mid": False, "HV_hi": False,
                "U1": True}
SHOWN = ("IGD", "HV_jin", "Imprecision", "PIBI", "HV_hi")   # columns in the markdown tables


def _wilcoxon_symbol(a, b, lower_better: bool, alpha: float = 0.05) -> str:
    """Jin et al.'s convention, paired by seed: '+' = this policy significantly better than
    the reference, '-' = significantly worse, '=' = no significant difference."""
    import numpy as np
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = min(len(a), len(b))
    a, b = a[:k], b[:k]
    if k < 2 or np.allclose(a, b):
        return "="
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        return "="
    if p >= alpha:
        return "="
    better = a.mean() < b.mean() if lower_better else a.mean() > b.mean()
    return "+" if better else "-"


def summarize(out_dir: pathlib.Path, test_set, policies, seeds: int, reference: str = "trained",
              rank_metric: str = "PIBI") -> None:
    import numpy as np
    from gat_imoea_dqn.metrics import hv_jin, hypervolume_3d, igd, imprecision, pooled_reference
    from gat_imoea_dqn.selection import das_dennis, selection_actions
    from gat_imoea_dqn.state import pibi

    if rank_metric not in METRICS:
        raise ValueError(f"rank metric must be one of {METRICS}")
    W = das_dennis(6, 3)
    A = selection_actions()
    runs = []
    for p in sorted((out_dir / "runs").glob("*.json")):
        m = RUN_NAME.match(p.name)
        if not m or m["pol"] not in policies or m["wf"] not in test_set or int(m["s"]) >= seeds:
            continue
        r = json.loads(p.read_text())
        r.update(workflow=m["wf"], policy=m["pol"], seed=int(m["s"]),
                 train_seed=int(m["t"]) if m["t"] is not None else None)
        runs.append(r)

    for wfname in test_set:
        rs = [r for r in runs if r["workflow"] == wfname]
        if not rs:
            continue
        ideal = np.min([np.min(r["front_lo"], axis=0) for r in rs], axis=0)
        nadir = np.max([np.max(r["front_hi"], axis=0) for r in rs], axis=0)
        span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
        norm = [((np.asarray(r["front_lo"]) - ideal) / span, (np.asarray(r["front_hi"]) - ideal) / span) for r in rs]
        ref_front = pooled_reference([(lo + hi) / 2 for lo, hi in norm])
        for r, (lo, hi) in zip(rs, norm):
            mid = (lo + hi) / 2
            r["IGD"] = igd(mid, ref_front)
            r["HV_jin"] = hv_jin(lo, hi, [1.1] * 3)
            r["Imprecision"] = imprecision(lo, hi)
            r["PIBI"] = pibi(lo, hi, W)
            r["HV_mid"] = hypervolume_3d(mid, [1.1] * 3)
            r["HV_hi"] = hypervolume_3d(hi, [1.1] * 3)
            r["U1"] = float(np.mean((hi - lo).sum(axis=1)))
            r["front_size"] = len(lo)

    present = [p for p in policies if any(r["policy"] == p for r in runs)]
    wfs = [w for w in test_set if any(r["workflow"] == w for r in runs)]

    def value(pol, wf, s, m):
        v = [r[m] for r in runs if r["policy"] == pol and r["workflow"] == wf and r["seed"] == s]
        return float(np.mean(v)) if v else None

    summary = {"policies": present, "reference": reference, "rank_metric": rank_metric,
               "per_workflow": {}, "overall": {}, "per_training_seed": {}}
    ranks = {p: [] for p in present}
    for wf in wfs:
        per = {}
        for pol in present:
            per[pol] = {}
            for m in METRICS + ("front_size",):
                vals = [value(pol, wf, s, m) for s in range(seeds)]
                vals = [v for v in vals if v is not None]
                per[pol][m] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "values": vals}
        if reference in per:
            for pol in per:
                if pol == reference:
                    continue
                for m in METRICS:
                    a, b = per[pol][m]["values"], per[reference][m]["values"]
                    k = min(len(a), len(b))
                    wins = sum((x < y) if LOWER_BETTER[m] else (x > y) for x, y in zip(a[:k], b[:k]))
                    per[pol][m]["beats_ref"] = f"{wins}/{k}"
                    per[pol][m]["wilcoxon"] = _wilcoxon_symbol(a, b, LOWER_BETTER[m])
        wf_ranks = {p: [] for p in present}
        for s in range(seeds):
            vals = {p: value(p, wf, s, rank_metric) for p in present}
            vals = {p: v for p, v in vals.items() if v is not None}
            order = sorted(vals, key=vals.get, reverse=not LOWER_BETTER[rank_metric])
            for i, p in enumerate(order):
                wf_ranks[p].append(i + 1)
        for p in present:
            per[p]["mean_rank"] = float(np.mean(wf_ranks[p])) if wf_ranks[p] else None
            ranks[p] += wf_ranks[p]
        summary["per_workflow"][wf] = per

    for pol in present:
        o = {"mean_rank": float(np.mean(ranks[pol])),
             **{f"mean_{m}": float(np.mean([summary["per_workflow"][w][pol][m]["mean"] for w in wfs]))
                for m in METRICS}}
        for m in METRICS:
            if "wilcoxon" in summary["per_workflow"][wfs[0]][pol][m]:
                syms = [summary["per_workflow"][w][pol][m]["wilcoxon"] for w in wfs]
                o[f"wilcoxon_{m}"] = f"{syms.count('+')}/{syms.count('-')}/{syms.count('=')}"
                tot = [summary["per_workflow"][w][pol][m]["beats_ref"].split("/") for w in wfs]
                o[f"beats_ref_{m}"] = f"{sum(int(a) for a, _ in tot)}/{sum(int(b) for _, b in tot)}"
        summary["overall"][pol] = o

    for pol in present:
        tseeds = sorted({r["train_seed"] for r in runs if r["policy"] == pol and r["train_seed"] is not None})
        if not tseeds:
            continue
        rows = {}
        for t in tseeds:
            sel = [r for r in runs if r["policy"] == pol and r["train_seed"] == t]
            beats = {}
            for base in ("fixed_fav", "fixed", "random", "fresh"):
                if base not in present:
                    continue
                n = w = 0
                for r in sel:
                    b = value(base, r["workflow"], r["seed"], rank_metric)
                    if b is not None:
                        n += 1
                        w += (r[rank_metric] < b) if LOWER_BETTER[rank_metric] else (r[rank_metric] > b)
                beats[base] = f"{w}/{n}"
            rows[t] = {f"mean_{m}": float(np.mean([r[m] for r in sel])) for m in SHOWN}
            rows[t]["beats"] = beats
            acts = np.bincount([a for r in sel for a in r["actions"] if a >= 0], minlength=len(A))
            if acts.sum():
                top = np.argsort(acts)[::-1][:3]
                rows[t]["top_actions"] = [{"action": A[i].round(3).tolist(), "share": float(acts[i] / acts.sum())}
                                          for i in top if acts[i]]
            log = out_dir / f"train_log_{POLICIES[pol]['variant']}_t{t}.json"
            if log.exists():
                Lg = json.loads(log.read_text())
                rew = [e["mean_reward"] for e in Lg]
                q = max(1, len(rew) // 4)
                rows[t]["train_reward_first_vs_last_quarter"] = [float(np.mean(rew[:q])), float(np.mean(rew[-q:]))]
        summary["per_training_seed"][pol] = rows
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------ markdown
    arrow = {m: "↓" if LOWER_BETTER[m] else "↑" for m in METRICS}
    L = [f"# Experiment `{out_dir.name}`", "",
         "One shared normalization per workflow across all policies below. "
         + " · ".join(f"{m} {arrow[m]}" for m in SHOWN) + ". Agent policies: each eval seed is averaged over "
         f"training seeds. Symbols: Wilcoxon signed-rank test (paired by seed, 5%) against *{reference}* — "
         "`+` better, `−` worse, `=` no significant difference.", "",
         f"## Overall (mean over workflows; rank by {rank_metric})", "",
         f"| Policy | mean rank | " + " | ".join(SHOWN) + " | +/−/= vs " + reference + " (" + ", ".join(SHOWN[:3]) + ") |",
         "|---|---|" + "---|" * len(SHOWN) + "---|"]
    for pol in sorted(present, key=lambda p: summary["overall"][p]["mean_rank"]):
        o = summary["overall"][pol]
        sym = " · ".join(o.get(f"wilcoxon_{m}", "–") for m in SHOWN[:3]) if pol != reference else "(reference)"
        L.append(f"| {pol} | {o['mean_rank']:.2f} | " + " | ".join(f"{o[f'mean_{m}']:.4g}" for m in SHOWN)
                 + f" | {sym} |")
    if summary["per_training_seed"]:
        bases = [b for b in ("fixed_fav", "fixed", "random", "fresh") if b in present]
        L += ["", f"## Each training seed on its own ({rank_metric})", "",
              "| Policy | train seed | " + " | ".join(SHOWN[:3]) + " | "
              + " | ".join(f"beats {b}" for b in bases) + " | train reward (first → last quarter) | top action |",
              "|---|---|" + "---|" * (3 + len(bases)) + "---|---|"]
        for pol, rows in summary["per_training_seed"].items():
            for t, r in rows.items():
                tr = r.get("train_reward_first_vs_last_quarter")
                ta = r.get("top_actions", [{}])[0]
                L.append(f"| {pol} | {t} | " + " | ".join(f"{r[f'mean_{m}']:.4g}" for m in SHOWN[:3]) + " | "
                         + " | ".join(r["beats"].get(b, "–") for b in bases)
                         + f" | {'%.4f → %.4f' % tuple(tr) if tr else '–'} | "
                         f"{ta.get('action', '–')} ({ta.get('share', 0):.0%}) |")
    for wf in wfs:
        per = summary["per_workflow"][wf]
        L += ["", f"## {wf}", "", "| Policy | " + " | ".join(SHOWN) + " | front | rank |",
              "|---|" + "---|" * len(SHOWN) + "---|---|"]
        for pol in sorted(per, key=lambda p: per[p]["mean_rank"]):
            d = per[pol]
            cell = lambda m: (f"{d[m]['mean']:.4g} ± {d[m]['sd']:.2g}"  # noqa: E731
                              + (f" {d[m]['wilcoxon'].replace('-', '−')}" if "wilcoxon" in d[m] else ""))
            L.append(f"| {pol} | " + " | ".join(cell(m) for m in SHOWN)
                     + f" | {d['front_size']['mean']:.0f} | {d['mean_rank']:.2f} |")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# ----------------------------------------------------------------- main ---
def reuse(src: pathlib.Path, dst: pathlib.Path) -> int:
    """Copy runs and agents from an earlier experiment (old names are mapped)."""
    n = 0
    for p in (src / "runs").glob("*.json"):
        name = p.name
        m = re.match(r"^(.+?)__trained__s(\d+)\.json$", name)      # pre-ablation naming: GAT agent, seed 0
        if m:
            name = f"{m[1]}__trained__t0__s{m[2]}.json"
        if not (dst / "runs" / name).exists():
            shutil.copy2(p, dst / "runs" / name)
            n += 1
    for old, new in (("agent.pt", "agent_gat_t0.pt"), ("train_log.json", "train_log_gat_t0.json")):
        if (src / old).exists() and not (dst / new).exists():
            shutil.copy2(src / old, dst / new)
    for p in list(src.glob("agent_*_t*.pt")) + list(src.glob("train_log_*_t*.json")):
        if not (dst / p.name).exists():
            shutil.copy2(p, dst / p.name)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="ablation")
    ap.add_argument("--policies", nargs="*", default=list(POLICIES), choices=list(POLICIES))
    ap.add_argument("--train-seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--train-set", nargs="*", default=TRAIN_SET)
    ap.add_argument("--test-set", nargs="*", default=TEST_SET)
    ap.add_argument("--train-episodes", type=int, default=40)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--pop", type=int, default=105)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) * 3 // 4))
    ap.add_argument("--reuse-from", help="copy finished runs and agents from this experiment name")
    ap.add_argument("--reference", default="trained", help="policy that paired 'beats' counts compare against")
    ap.add_argument("--scenario", default="default", help="datasets/scenarios/<name>.yaml")
    ap.add_argument("--data-multiplier", type=float, default=None,
                    help="rescale workflows into Jin et al.'s ranges and multiply data by this (CCR level)")
    ap.add_argument("--energy-scope", choices=["all", "device"], default="all")
    ap.add_argument("--train-generations", type=int, default=None, help="default: --generations")
    ap.add_argument("--rank-metric", default="PIBI", help="metric used for mean ranks and table order")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    env = {"scenario": args.scenario, "data_multiplier": args.data_multiplier, "energy_scope": args.energy_scope}
    train_gens = args.train_generations or args.generations
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252

    out = ROOT / "results" / "experiments" / args.name
    (out / "runs").mkdir(parents=True, exist_ok=True)
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip())
    except OSError:
        commit, dirty = None, None
    (out / "config.json").write_text(json.dumps({**vars(args), "git_commit": commit, "git_dirty": dirty}, indent=2))
    if args.reuse_from:
        print(f"reused {reuse(ROOT / 'results' / 'experiments' / args.reuse_from, out)} runs from {args.reuse_from}")
    if args.summary_only:
        summarize(out, args.test_set, args.policies, args.seeds, args.reference, args.rank_metric)
        return 0

    variants = sorted({POLICIES[p]["variant"] for p in args.policies if POLICIES[p]["kind"] == "greedy"})
    trainers = []
    for v in variants:
        for t in args.train_seeds:
            if not agent_path(out, v, t).exists():
                pr = mp.Process(target=train, args=(str(out), v, t, args.train_set, args.train_episodes,
                                                    train_gens, args.pop, env))
                pr.start()
                trainers.append((v, t, pr))
                print(f"training {v} agent, seed {t} (pid {pr.pid})", flush=True)

    def specs(agent_jobs: bool):
        out_specs = []
        for w in args.test_set:
            for p in args.policies:
                d = POLICIES[p]
                if (d["kind"] == "greedy") != agent_jobs:
                    continue
                for s in range(args.seeds):
                    common = {"workflow": w, "policy": p, "seed": s, "generations": args.generations,
                              "pop": args.pop, "env": env}
                    if agent_jobs:
                        for t in args.train_seeds:
                            out_specs.append({**common, "train_seed": t,
                                              "agent": str(agent_path(out, d["variant"], t)),
                                              "out": str(out / "runs" / f"{w}__{p}__t{t}__s{s}.json")})
                    else:
                        out_specs.append({**common, "out": str(out / "runs" / f"{w}__{p}__s{s}.json")})
        return out_specs

    t0 = time.time()
    with mp.Pool(max(1, args.workers - len(trainers))) as pool:
        for msg in pool.imap_unordered(job, specs(False)):
            if not msg.startswith("skip"):
                print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    for v, t, pr in trainers:
        pr.join()
        if pr.exitcode != 0 or not agent_path(out, v, t).exists():
            print(f"training {v} seed {t} failed; see train_log_{v}_t{t}.json")
            return 1
        print(f"[{time.time() - t0:6.0f}s] {v} agent seed {t} trained", flush=True)
    with mp.Pool(args.workers) as pool:
        for msg in pool.imap_unordered(job, specs(True)):
            if not msg.startswith("skip"):
                print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    print(f"[{time.time() - t0:6.0f}s] all jobs finished", flush=True)
    summarize(out, args.test_set, args.policies, args.seeds, args.reference, args.rank_metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
