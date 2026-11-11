"""R2: system-model figure, results/figures/system_model.{png,pdf}.

  python scripts/system_figure.py

Left: the cloud-edge-end architecture of Jin et al.'s small scale (1 cloud, 3 edge servers, 6 mobile
devices; the origin device reaches all three edges), with the interval quantities of the model.
Right: an application DAG submitted by the origin device, its tasks coloured by the tier the
allocation string x assigns them to, and the priority string o, a topological order.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
INK, MUTED, GRID = "#2b2a27", "#6b6a64", "#e4e3dd"
TIER = {"dev": "#1baf7a", "edge": "#eb6834", "cloud": "#2a78d6"}
FILL = {"dev": "#e3f5ee", "edge": "#fdebe3", "cloud": "#e2eefb"}


def box(ax, x, y, w, h, tier, text, fs=8, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[tier], ec=TIER[tier], lw=lw, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, zorder=4,
            fontweight="bold" if bold else "normal")


def link(ax, a, b, color=MUTED, ls="-", lw=1.0, text=None, toff=(0, 0), fs=7, arrows=True):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>" if arrows else "-", mutation_scale=7, color=color,
                                 lw=lw, ls=ls, shrinkA=2, shrinkB=2, zorder=2))
    if text:
        ax.text((a[0] + b[0]) / 2 + toff[0], (a[1] + b[1]) / 2 + toff[1], text, fontsize=fs, color=color,
                ha="center", va="center", zorder=4, bbox=dict(fc="white", ec="none", pad=0.6))


def architecture(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    for y, name in ((8.6, "Cloud tier"), (5.4, "Edge tier"), (1.6, "End tier")):
        ax.text(0.05, y + 1.05, name, fontsize=8.5, color=MUTED, fontweight="bold", va="center")
        ax.plot([0.05, 9.95], [y - 1.25, y - 1.25], color=GRID, lw=0.8, zorder=0)
    box(ax, 5.0, 8.6, 3.6, 1.2, "cloud", "Cloud data centre\n" + r"$\tilde f_C$ per task", fs=8.5)
    edges = [(2.0, 5.4), (5.0, 5.4), (8.0, 5.4)]
    for i, (x, y) in enumerate(edges, 1):
        box(ax, x, y, 2.1, 1.0, "edge", f"Edge server {i}\n" + r"$\tilde f_u$, shared", fs=8)
    for i, (x, y) in enumerate(edges):
        link(ax, (x, y + 0.5), (5.0 + (x - 5.0) * 0.35, 8.0), color=TIER["cloud"],
             text=r"$\tilde W_C$" if i == 0 else None, toff=(-0.35, 0))
    link(ax, (3.05, 5.4), (3.95, 5.4), color=TIER["edge"])
    link(ax, (6.05, 5.4), (6.95, 5.4), color=TIER["edge"], text=r"$\tilde W_{E}$", toff=(0, 0.32))
    devs = [(0.9, 1.6), (2.45, 1.6), (4.0, 1.6), (5.6, 1.6), (7.15, 1.6), (8.7, 1.6)]
    for i, (x, y) in enumerate(devs):
        origin = i == 2
        box(ax, x, y, 1.25, 1.0, "dev", ("Origin\ndevice" if origin else f"Device {i + 1}") + "\n" + r"$\tilde f_i$, $\tilde e_i$",
            fs=7, bold=origin, lw=2.0 if origin else 1.0)
    for e in edges:                                           # origin device reaches all three edges
        link(ax, (4.0, 2.1), (e[0], 4.9), color=TIER["dev"], ls=(0, (3, 2)), lw=1.1)
    ax.text(3.0, 3.35, r"$\tilde R_{i,u} = \tilde B \log_2(1 + p_i g / v_0)$", fontsize=7, color=TIER["dev"],
            ha="center", bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
    for (x, y), e in zip(devs, (0, 0, None, 1, 2, 2)):
        if e is not None:
            link(ax, (x, y + 0.5), (edges[e][0] + (x - edges[e][0]) * 0.25, 4.9), color=TIER["dev"], ls=(0, (1, 2)),
                 lw=0.8)
    ax.text(5.0, 0.15, "Interval quantities: CPU frequencies, link rates, device energy per cycle "
            r"($\tilde a = [a^L, a^U]$)", fontsize=7, color=MUTED, ha="center")


def dag(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.2, 9.6, "Application DAG of the origin device", fontsize=8.5, color=MUTED, fontweight="bold")
    pos = {1: (1.3, 6.8), 2: (3.6, 8.3), 3: (3.6, 6.8), 4: (3.6, 5.3), 5: (6.0, 7.6), 6: (6.0, 5.9), 7: (8.4, 6.8)}
    tier = {1: "dev", 2: "edge", 3: "edge", 4: "cloud", 5: "edge", 6: "cloud", 7: "dev"}
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (3, 6), (4, 6), (5, 7), (6, 7)]
    crit = {(1, 3), (3, 6), (6, 7)}
    r = 0.55
    for a, b in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        on = (a, b) in crit
        ax.add_patch(FancyArrowPatch((x0 + r * ux, y0 + r * uy), (x1 - (r + 0.05) * ux, y1 - (r + 0.05) * uy),
                                     arrowstyle="-|>", mutation_scale=10, color=INK if on else MUTED,
                                     lw=1.8 if on else 0.9, shrinkA=0, shrinkB=0, zorder=2))
    ax.text(4.55, 5.85, r"$\tilde D_{36}$", fontsize=7.5, color=INK, ha="center")
    for k, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.55, fc=FILL[tier[k]], ec=TIER[tier[k]], lw=1.6, zorder=3))
        ax.text(x, y, f"$v_{k}$", ha="center", va="center", fontsize=9, color=INK, zorder=4)
    ax.text(1.3, 5.85, r"$\tilde C_1$", fontsize=7.5, color=INK, ha="center")
    for i, (t, lab) in enumerate((("dev", "device"), ("edge", "edge server"), ("cloud", "cloud"))):
        ax.add_patch(plt.Circle((0.6 + 2.7 * i, 3.9), 0.22, fc=FILL[t], ec=TIER[t], lw=1.4))
        ax.text(0.95 + 2.7 * i, 3.9, f"on {lab}", fontsize=7.5, va="center", color=INK)
    ax.plot([0.4, 1.4], [3.2, 3.2], color=INK, lw=1.8)
    ax.text(1.6, 3.2, "interval critical path (sets the makespan)", fontsize=7.5, va="center", color=INK)
    ax.text(0.3, 2.25, r"$\mathbf{x} = (0, 1, 2, 4, 1, 4, 0)$   allocation: 0 device, 1..3 edge, 4 cloud",
            fontsize=7.5, color=INK, family="monospace" if False else None)
    ax.text(0.3, 1.55, r"$\mathbf{o} = (v_1, v_3, v_2, v_4, v_6, v_5, v_7)$   priority: a topological order",
            fontsize=7.5, color=INK)
    ax.text(0.3, 0.6, r"Objectives (minimise): $\tilde F_1$ makespan, $\tilde F_2$ device energy, $\tilde F_3$ cost",
            fontsize=7.5, color=INK, bbox=dict(fc="#f4f3ef", ec=GRID, pad=3))


def main() -> int:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    architecture(a)
    dag(b)
    fig.tight_layout(w_pad=1.0)
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"system_model.{ext}", dpi=220)
    print("wrote", FIGS / "system_model.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
