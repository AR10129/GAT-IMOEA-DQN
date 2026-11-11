"""D1: figures for the learned-control analysis (C4).

  python scripts/d1_learning.py

Figure 1, results/figures/d1_reward_audit.{png,pdf} (data: results/experiments/reward_audit, every
one of the 28 actions held fixed for G = 1000, small_20_i0@x1 and basic_60_i0@x20, 2 seeds):
  (a) signal-to-noise of the per-generation reward the DQN learns from, per 50-generation window:
      standard deviation of the 28 actions' mean rewards / standard deviation of all per-step
      rewards in the window (as scripts/reward_audit.py);
  (b, c) per instance, the rank of each action's summed reward (what the DQN maximises) against
      its rank by final IGD, with the Spearman correlation.

Figure 2, results/figures/d1_training.{png,pdf} (training logs of the three learner designs on the
decomposition backbone): per episode, the mean reward divided by that workflow's mean over the
whole log (episodes cycle through workflows of different sizes, so raw rewards are not
comparable across episodes); 8-episode rolling mean over training seeds. A flat line at 1 means the
reward the agent earns does not rise with training. Panel (b): validation score of the stable
agents (the only design that logs one).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
FIGS = ROOT / "results" / "figures"
AUDIT_INST = [("small_20_i0@x1", "small-20 $\\times$1"), ("basic_60_i0@x20", "basic-60 $\\times$20")]
FAV, DIV = (5 / 6, 1 / 6, 0.0), (2 / 3, 0.0, 1 / 3)
DESIGNS = [("stable", "suite_a_tch", "Every generation, PIBI reward, 28 actions", "#2a78d6"),
           ("blk", "dqn_fix_test", "Every 50 generations, HV-gain reward, 28 actions", "#eb6834"),
           ("l8", "dqn_fix_test", "Every 25 generations, log-HV reward, 10 actions", "#1baf7a")]


def _style(ax):
    ax.grid(True, color="#e4e3dd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def reward_audit(plt):
    from scipy.stats import rankdata, spearmanr
    summ = json.loads((EXP / "reward_audit" / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    W = 50
    for (inst, lab), col in zip(AUDIT_INST, ("#2a78d6", "#eb6834")):
        acts = summ[inst]["actions"]
        R = []                                                  # (actions, seeds, G)
        for a in range(len(acts)):
            R.append([json.loads(p.read_text())["reward"]
                      for p in sorted((EXP / "reward_audit" / "runs").glob(f"{inst}__a{a:02d}__s*.json"))])
        R = np.asarray(R, float)
        G = R.shape[2]
        x, snr = [], []
        for lo in range(0, G, W):
            win = R[:, :, lo:lo + W]
            snr.append(win.mean(axis=(1, 2)).std() / max(win.std(), 1e-12))
            x.append(lo + W / 2)
        axes[0].plot(x, snr, color=col, lw=1.6, marker="o", ms=3, label=lab)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Generation (50-generation windows)")
    axes[0].set_ylabel("Reward signal-to-noise ratio")
    axes[0].set_title("(a) Per-step reward signal", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    _style(axes[0])
    for ax, (inst, lab), tag in zip(axes[1:], AUDIT_INST, ("(b)", "(c)")):
        d = summ[inst]
        acts = [tuple(a) for a in d["actions"]]
        R, I = np.asarray(d["R"]).mean(axis=1), np.asarray(d["IGD"]).mean(axis=1)      # mean over seeds
        r_rank, i_rank = rankdata(-R), rankdata(I)
        ax.plot([1, 28], [1, 28], color="#bdbbb3", lw=0.8, zorder=1)
        ax.scatter(i_rank, r_rank, s=22, color="#6b6a64", edgecolor="white", linewidth=0.6, zorder=2)
        for a, name, c in ((FAV, "$(5/6, 1/6, 0)$", "#2a78d6"), (DIV, "$(2/3, 0, 1/3)$", "#c2477a")):
            k = int(np.argmin([np.abs(np.subtract(a, b)).sum() for b in acts]))
            ax.scatter(i_rank[k], r_rank[k], s=46, color=c, edgecolor="white", linewidth=0.8, zorder=3, label=name)
        rho = spearmanr(R, -I).statistic
        ax.set_title(f"{tag} {lab}: Spearman {rho:.2f}", fontsize=9)
        ax.set_xlabel("Rank by final IGD (1 = best)")
        ax.set_ylabel("Rank by summed reward (1 = highest)")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        _style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"d1_reward_audit.{ext}", dpi=200)
    print("wrote", FIGS / "d1_reward_audit.png")


def _relative(log):
    by = {}
    for e in log:
        by.setdefault(e["workflow"], []).append(e["mean_reward"])
    mean = {w: np.mean(v) for w, v in by.items()}
    return np.array([e["mean_reward"] / mean[e["workflow"]] if mean[e["workflow"]] else np.nan for e in log])


def training(plt):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for design, exp, lab, col in DESIGNS:
        for gat, ls in (("gat", "-"), ("nogat", "--")):
            logs = sorted((EXP / exp).glob(f"train_log_tch_{gat}_{design}_t*.json"))
            if not logs:
                continue
            ys = [_relative(json.loads(p.read_text())) for p in logs]
            n = min(len(y) for y in ys)
            y = np.nanmean([y[:n] for y in ys], axis=0)
            k = 8
            sm = np.convolve(y, np.ones(k) / k, mode="valid")
            axes[0].plot(np.arange(k, n + 1), sm, color=col, ls=ls, lw=1.5,
                         label=f"{lab} ({'GAT' if gat == 'gat' else 'no GAT'})")
            if design == "stable":
                for p in logs:
                    v = [e["val_score"] for e in json.loads(p.read_text()) if e.get("val_score") is not None]
                    if v:
                        ep = [e["episode"] + 1 for e in json.loads(p.read_text()) if e.get("val_score") is not None]
                        axes[1].plot(ep, v, color=col, ls=ls, lw=1.2, marker="o", ms=3, alpha=0.8)
    axes[0].axhline(1.0, color="#bdbbb3", lw=0.8, zorder=0)
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Mean reward / workflow's mean\n(8-episode rolling mean)")
    axes[0].set_title("(a) Training reward", fontsize=9)
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Validation score")
    axes[1].set_title("(b) Validation score, every-generation design", fontsize=9)
    for ax in axes:
        _style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, frameon=False,
               title="Decision interval, reward, action set; solid = GAT, dashed = no GAT", title_fontsize=7)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"d1_training.{ext}", dpi=200)
    print("wrote", FIGS / "d1_training.png")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    reward_audit(plt)
    training(plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
