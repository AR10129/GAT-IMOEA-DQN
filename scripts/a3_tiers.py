"""A3: what the beta (dominance / diversity) and gamma (uncertainty) tiers buy, and the Rank C stall.

Part 1 (from the A2 sweep, results/experiments/action_sweep/summary.md): IGD ratio vs Imprecision
per solution for the 28 fixed actions; figure results/figures/a3_tradeoff.{png,pdf}.

Part 2 (new runs, --stall): records, every generation, how many of the alpha*N Rank C places are
decided by the tie-break rather than by the rank itself (the boundary tie group is larger than
the places left), and how many pool members share the best Rank C value. Compares Jin et al.'s
Rank C (R_IN, random tie-break; the `fixed_fav` policy) with final's Tchebycheff Rank C (ICD
tie-break). Writes results/stats/a3_stall.csv and results/figures/a3_stall.{png,pdf}.

  python scripts/a3_tiers.py                       # part 1
  python scripts/a3_tiers.py --stall --seeds 3     # part 2 (runs 2 policies x 2 instances x seeds)
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

STALL_CASES = [("medium_250_i0@x20", {"scenario": "default"}), ("montage_400@x20", {"scenario": "jin"})]
STALL_POLICIES = {"fixed_fav": ("Jin Rank C (R_IN, random ties)", "#c2477a"),
                  "final": ("Ours: Tchebycheff Rank C (ICD ties)", "#2a78d6")}


def tradeoff():
    rows = []
    pat = re.compile(r"^\| \(([\d.]+), ([\d.]+), ([\d.]+)\) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) / ([\d.]+) \| ([\d.]+) \|")
    for line in (ROOT / "results" / "experiments" / "action_sweep" / "summary.md").read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            a, b, g = (float(m[i]) for i in (1, 2, 3))
            rows.append((a, b, g, float(m[6]), float(m[8])))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cmap = {0.0: "#2a78d6", 0.167: "#1baf7a", 0.333: "#eda100", 0.5: "#eb6834", 0.667: "#c2477a",
            0.833: "#6250d6", 1.0: "#3d3d3a"}
    for gv in sorted({r[2] for r in rows}):
        pts = [r for r in rows if r[2] == gv]
        ax.scatter([p[3] for p in pts], [p[4] for p in pts], s=34, color=cmap.get(round(gv, 3), "#888"),
                   edgecolor="white", linewidth=0.8, label=f"γ = {gv:.2g}", zorder=3)
    for a, b, g, r, imp in rows:
        if (round(a, 3), round(b, 3), round(g, 3)) in {(0.833, 0.167, 0.0), (0.833, 0.0, 0.167), (0.667, 0.167, 0.167),
                                                       (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)}:
            ax.annotate(f"({a:.2g}, {b:.2g}, {g:.2g})", (r, imp), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("IGD / best IGD on the instance (mean, log scale)")
    ax.set_ylabel("Imprecision per solution")
    ax.grid(True, color="#e4e3dd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    figs = ROOT / "results" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"a3_tradeoff.{ext}", dpi=200)
    print("wrote", figs / "a3_tradeoff.png")
    return rows


def _stall_job(args):
    inst, env_extra, policy, seed, G = args
    import experiment as E
    from gat_imoea_dqn import algorithm as A
    from gat_imoea_dqn.algorithm import RunConfig
    env = {"energy_scope": "device", "external_io": "device", "makespan": "sum_apps", **env_extra}
    wf, sc = E._load(inst, env)
    d = E.POLICIES[policy]
    cfg = RunConfig(pop_size=105, generations=G, seed=seed, evaluator=E._ecfg(env), policy="fixed",
                    fixed_action=tuple(d["action"]), **E._backbone(d.get("backbone")))
    rec = []
    orig = A.environmental_selection

    def wrapped(*a, **k):
        keep, info = orig(*a, **k)
        R, n_C = np.asarray(info.R_C), info.counts[0]
        v = np.sort(R)
        c = v[n_C - 1]
        above, group = int((R < c).sum()), int((R == c).sum())
        slots = n_C - above
        rec.append((slots / n_C if group > slots else 0.0, int((R == v[0]).sum())))
        return keep, info

    A.environmental_selection = wrapped
    try:
        A.run(wf, sc, cfg)
    finally:
        A.environmental_selection = orig
    return inst, policy, seed, rec


def stall(seeds: int, G: int, workers: int):
    jobs = [(i, e, p, s, G) for i, e in STALL_CASES for p in STALL_POLICIES for s in range(seeds)]
    with mp.Pool(workers) as pool:
        res = pool.map(_stall_job, jobs)
    out = ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    agg = {}
    for inst, pol, s, rec in res:
        agg.setdefault((inst, pol), []).append(np.asarray(rec, float))
    with open(out / "a3_stall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "policy", "generation", "tie_decided_share", "n_best"])
        for (inst, pol), recs in agg.items():
            m = np.mean(recs, axis=0)
            for g, (t, nb) in enumerate(m, start=1):
                w.writerow([inst, pol, g, f"{t:.4f}", f"{nb:.2f}"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(STALL_CASES), figsize=(4.2 * len(STALL_CASES), 5.6), squeeze=False)
    for j, (inst, _) in enumerate(STALL_CASES):
        for pol, (lab, col) in STALL_POLICIES.items():
            m = np.mean(agg[(inst, pol)], axis=0)
            g = np.arange(1, len(m) + 1)
            axes[0, j].plot(g, 100 * m[:, 0], color=col, lw=1.6, label=lab)
            axes[1, j].plot(g, m[:, 1], color=col, lw=1.6, label=lab)
        axes[0, j].set_title(inst.replace("_i0", ""), fontsize=10)
        axes[1, j].set_xlabel("Generation")
        for ax in axes[:, j]:
            ax.grid(True, color="#e4e3dd", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("Rank C places decided\nby the tie-break (%)")
    axes[1, 0].set_ylabel("Pool members sharing\nthe best Rank C value")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    figs = ROOT / "results" / "figures"
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"a3_stall.{ext}", dpi=200)
    for (inst, pol), recs in agg.items():
        m = np.mean(recs, axis=0)
        print(f"{inst:20s} {pol:10s} tie-decided share: G1-100 {100 * m[:100, 0].mean():5.1f}%  "
              f"G101-500 {100 * m[100:500, 0].mean():5.1f}%  G501-1000 {100 * m[500:, 0].mean():5.1f}%  "
              f"n_best at G=1000 {m[-1, 1]:.0f}")
    print("wrote", figs / "a3_stall.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stall", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--generations", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.stall:
        stall(args.seeds, args.generations, args.workers)
    else:
        tradeoff()
    return 0


if __name__ == "__main__":
    sys.exit(main())
