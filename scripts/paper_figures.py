"""Two paper figures (journal-format draft, 2026-11-19).

  python scripts/paper_figures.py

results/figures/fronts.{png,pdf}  Final fronts of four methods on one Suite A and one Suite B instance (data x20):
    for each method the run with the median IGD over its 10 runs (E3 scores), interval midpoints normalised
    over the four plotted fronts, shown as makespan-energy and makespan-cost projections. Runs come from
    results/experiments/e7_a / e7_b (the E7 runs keep the final fronts; identical to the suite runs).
results/figures/cd_diagram.{png,pdf}  Critical-difference diagrams (Demsar 2006) of the mean Friedman ranks on
    Suite B for IGD and HV, Nemenyi CD at alpha = 0.05; methods joined by a bar are not significantly different.

Colours: the method colours of the convergence figure; NSGA-II uses green here because the amber of the
convergence figure is too close to IMOEA/D's orange for normal vision (checked with the dataviz validator);
every method also has its own marker shape.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FIGS = ROOT / "results" / "figures"
STATS = ROOT / "results" / "stats"
INK, MUTED, GRID = "#2b2a27", "#6b6a64", "#e4e3dd"

FRONT_METHODS = [("final", "DR-IMOEA", "#2a78d6", "o"), ("imoead_orig", "IMOEA/D", "#eb6834", "s"),
                 ("nsga2_mid", "NSGA-II (mid)", "#1baf7a", "^"), ("fixed_fav", "IMOEA-DQN (fixed)", "#c2477a", "D")]
FRONT_CASES = [("medium_250_i0@x20", "e7_a", "paper_suite_a_runs.csv", "Suite A: Medium-250 (data ×20)"),
               ("cybershake_1000@x20", "e7_b", "paper_suite_b_runs.csv", "Suite B: Cybershake-1000 (data ×20)")]


def _median_seed(csv_name, inst, pol):
    igd = {}
    with open(STATS / csv_name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["instance"] == inst and r["policy"] == pol and int(r["seed"]) < 10:
                igd[int(r["seed"])] = float(r["IGD"])
    order = sorted(igd, key=igd.get)
    return order[len(order) // 2]


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def fronts(plt):
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9))
    for c, (inst, exp, scores, title) in enumerate(FRONT_CASES):
        mids = {}
        for pol, *_ in FRONT_METHODS:
            s = _median_seed(scores, inst, pol)
            r = json.loads((ROOT / "results" / "experiments" / exp / "runs" / f"{inst}__{pol}__s{s}.json").read_text())
            mids[pol] = (np.asarray(r["front_lo"]) + np.asarray(r["front_hi"])) / 2
        allm = np.vstack(list(mids.values()))
        lo, span = allm.min(0), np.ptp(allm, 0)
        span[span == 0] = 1
        for k, (j, ylab) in enumerate(((1, "Energy"), (2, "Cost"))):
            ax = axes[2 * c + k]
            for pol, lab, col, mk in FRONT_METHODS:
                m = (mids[pol] - lo) / span
                ax.scatter(m[:, 0], m[:, j], s=10, color=col, marker=mk, edgecolor="white", linewidth=0.3,
                           label=lab, zorder=3 if pol == "final" else 2, alpha=0.9)
            ax.set_xlabel("Makespan (normalised)", fontsize=8, color=INK)
            ax.set_ylabel(f"{ylab} (normalised)", fontsize=8, color=INK)
            ax.set_title(title, fontsize=8.5, color=INK)
            _style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False, markerscale=1.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fronts.{ext}", dpi=220)
    print("wrote", FIGS / "fronts.png")


def cd_diagram(plt):
    import build_paper as B
    S = B.read_scores("suite_b")
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 3.3))
    for ax, (m, name) in zip(axes, (("IGD", "IGD"), ("HV_jin", "HV"))):
        ranks, _, _, cd = B.friedman(S, B.PAPER, m)
        lab = {p: B.SHORT_FULL.get(p, B.LABEL[p]).replace("\\ours{}", "DR-IMOEA") for p in B.PAPER}
        order = np.argsort(ranks)
        k = len(ranks)
        ax.set_xlim(0.6, k + 0.4)
        ax.set_ylim(-0.45, 1.5)
        ax.axis("off")
        ax.plot([1, k], [1, 1], color=INK, lw=0.8)
        for t in range(1, k + 1):
            ax.plot([t, t], [1, 1.07], color=INK, lw=0.8)
            ax.text(t, 1.13, str(t), ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.plot([k - cd, k], [1.36, 1.36], color=INK, lw=1.2)
        ax.text(k - cd - 0.1, 1.36, f"CD = {cd:.2f}", ha="right", va="center", fontsize=7, color=INK)
        half = (k + 1) // 2
        for i, idx in enumerate(order):
            r = ranks[idx]
            left = i < half
            y = 0.75 - 0.27 * (i if left else (k - 1 - i))
            xt = 0.9 if left else k + 0.1
            ax.plot([r, r, xt], [1, y, y], color=MUTED, lw=0.6)
            ax.text(xt + (-0.05 if left else 0.05), y, f"{lab[B.PAPER[idx]]} ({r:.2f})", ha="right" if left else "left",
                    va="center", fontsize=6.5, color=INK, fontweight="bold" if B.PAPER[idx] == "final" else None)
        # bars joining methods whose rank difference is below CD (maximal cliques along the sorted ranks)
        rs = ranks[order]
        cliques = []
        for a in range(k):
            b = a
            while b + 1 < k and rs[b + 1] - rs[a] < cd:
                b += 1
            if b > a and not any(a >= c0 and b <= c1 for c0, c1 in cliques):
                cliques.append((a, b))
        for n, (a, b) in enumerate(cliques):
            yb = 0.92 - 0.06 * n
            ax.plot([rs[a] - 0.03, rs[b] + 0.03], [yb, yb], color=INK, lw=2.2, solid_capstyle="round")
        ax.text(0.6, 1.36, f"Suite B, {name}", ha="left", va="center", fontsize=8, color=INK, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"cd_diagram.{ext}", dpi=220)
    print("wrote", FIGS / "cd_diagram.png")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fronts(plt)
    cd_diagram(plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
