"""E5: IGD and HV_jin against generation, from runs made with `experiment.py --trace`.

Per instance, the normalisation and the IGD reference front are the E3 ones (scripts/stats.py):
fixed over the final fronts of the paper policies in the pool experiments. So the generation-1000
points sit on the same scale as the E3 tables, and an early front outside the pool's nadir simply
scores worse (HV_jin counts nothing beyond the 1.1 reference point).

Each traced run's last front is also checked against the same policy and seed in the pool; it must
be identical (tracing does not touch any random stream).

  python scripts/convergence.py --name e5 --instances medium_250_i0@x20:e5_a:suite_a_main,suite_a_e2 \
      montage_400@x20:e5_b:suite_b,suite_b_e2 ...
Writes results/stats/<name>_convergence.csv, results/stats/<name>_convergence.md and
results/figures/<name>_convergence.{png,pdf}.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gat_imoea_dqn.metrics import hv_jin, igd, pooled_reference  # noqa: E402
from stats import LABEL, PAPER, RUN_NAME, load  # noqa: E402

# colour by method family, line style inside a family (10 series > 8 categorical slots)
STYLE = {"final": ("#2a78d6", "-", 2.4), "imoead_orig": ("#eb6834", "-", 1.6),
         "imoead_awn": ("#eb6834", "--", 1.6), "imea_ied": ("#1baf7a", "-", 1.6),
         "ip_moea": ("#6250d6", "-", 1.6), "nsga2_mid": ("#eda100", "-", 1.6),
         "nsga2_int": ("#eda100", "--", 1.6), "nsga3_mid": ("#008300", "-", 1.6),
         "fixed_fav": ("#c2477a", "--", 1.6), "trained_stable": ("#c2477a", "-", 1.6)}


def normalisation(pool):
    ideal = np.min([lo.min(0) for lo, _ in pool.values()], 0)
    nadir = np.max([hi.max(0) for _, hi in pool.values()], 0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    ref = pooled_reference([((lo - ideal) / span + (hi - ideal) / span) / 2 for lo, hi in pool.values()])
    return ideal, span, ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e5")
    ap.add_argument("--instances", nargs="+", required=True, help="instance:experiment:pool1,pool2")
    args = ap.parse_args()

    rows, checks, curves = [], [], {}
    for item in args.instances:
        inst, exp, pools = item.split(":")
        pool = load(pools.split(","), set(PAPER)).get(inst, {})
        ideal, span, ref = normalisation(pool)
        per = {}                                         # (policy, gen) -> list of (IGD, HV)
        for p in sorted((ROOT / "results" / "experiments" / exp / "runs").glob(f"{inst}__*.json")):
            m = RUN_NAME.match(p.name)
            if not m or m["pol"] not in PAPER:
                continue
            r = json.loads(p.read_text())
            key = (m["pol"], int(m["s"]), int(m["t"]) if m["t"] is not None else None)
            if key in pool:
                checks.append(np.array_equal(np.asarray(r["front_lo"]), pool[key][0])
                              and np.array_equal(np.asarray(r["front_hi"]), pool[key][1]))
            for g, (lo, hi) in r["trace"].items():
                lo, hi = (np.asarray(lo) - ideal) / span, (np.asarray(hi) - ideal) / span
                per.setdefault((m["pol"], int(g)), []).append((igd((lo + hi) / 2, ref), hv_jin(lo, hi, [1.1] * 3)))
        curves[inst] = per
        for (pol, g), v in sorted(per.items()):
            v = np.asarray(v)
            rows.append([inst, pol, g, len(v), *(f"{x:.6g}" for x in (v[:, 0].mean(), v[:, 0].std(),
                                                                       v[:, 1].mean(), v[:, 1].std()))])
        print(f"{inst}: {len({k[0] for k in per})} policies, {len({k[1] for k in per})} checkpoints", flush=True)

    out = ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.name}_convergence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "policy", "generation", "runs", "IGD_mean", "IGD_sd", "HV_jin_mean", "HV_jin_sd"])
        w.writerows(rows)

    # generation at which each policy first reaches final's generation-1000 IGD / within 1% of it
    L = [f"# E5 convergence: `{args.name}`", "",
         f"Final fronts identical to the pooled runs: {sum(checks)}/{len(checks)}.", "",
         "Mean IGD at selected generations (normalisation and reference front as in E3).", ""]
    for inst, per in curves.items():
        gens = sorted({g for _, g in per})
        show = [g for g in gens if g in (0, 50, 100, 250, 500, 750, 1000)]
        L += [f"## {inst}", "", "| Policy | " + " | ".join(f"G={g}" for g in show) + " | G to reach final's G=500 IGD |",
              "|---|" + "---|" * (len(show) + 1)]
        target = np.mean([v[0] for v in per[("final", 500)]])
        for pol in PAPER:
            if (pol, 0) not in per:
                continue
            mean = {g: np.mean([v[0] for v in per[(pol, g)]]) for g in gens}
            hit = next((g for g in gens if mean[g] <= target), None)
            L.append(f"| {LABEL[pol]} | " + " | ".join(f"{mean[g]:.4f}" for g in show) +
                     f" | {hit if hit is not None else 'not reached'} |")
        L.append("")
    (out / f"{args.name}_convergence.md").write_text("\n".join(L), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(curves)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.2), squeeze=False)
    for j, (inst, per) in enumerate(curves.items()):
        for pol in PAPER:
            gens = sorted(g for p, g in per if p == pol)
            if not gens:
                continue
            c, ls, lw = STYLE[pol]
            for i in (0, 1):
                y = [np.mean([v[i] for v in per[(pol, g)]]) for g in gens]
                axes[i, j].plot(gens, y, color=c, ls=ls, lw=lw, label=LABEL[pol], zorder=3 if pol == "final" else 2)
        axes[0, j].set_title(inst.replace("_i0", ""), fontsize=10)
        axes[0, j].set_yscale("log")
        axes[1, j].set_xlabel("Generation")
        for i in (0, 1):
            ax = axes[i, j]
            ax.grid(True, color="#e4e3dd", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("IGD (log scale)")
    axes[1, 0].set_ylabel("HV$_{jin}$")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    figs = ROOT / "results" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"{args.name}_convergence.{ext}", dpi=200)
    print(f"final fronts identical to pool: {sum(checks)}/{len(checks)}")
    print("wrote", out / f"{args.name}_convergence.md", "and", figs / f"{args.name}_convergence.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
