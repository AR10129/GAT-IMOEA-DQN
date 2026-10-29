"""Reward audit: does the DQN's reward rank the 28 actions the way the paper's metrics do?

    python scripts/reward_audit.py --workers 12            # run missing jobs, then score
    python scripts/reward_audit.py --score-only

Every one of the 28 Das-Dennis actions is held fixed for a whole run on the
decomposition (tch) backbone, with the tracker normalisation the stable agents
train with ("fixed": ideal 0, nadir from the initial population). Per run we keep
the per-generation reward (relative PIBI improvement), the PIBI trace and the
final front. Scoring:
  - final IGD / HV_jin per action (one normalisation per instance, pooled reference)
  - total reward per action (what the DQN maximises, gamma = 1)
  - Spearman correlation between the two rankings across actions
  - signal-to-noise of the per-step reward: spread of per-action means vs the
    per-step standard deviation, in early / middle / late windows
Output: results/experiments/reward_audit/summary.md and summary.json.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "experiments" / "reward_audit"
ENV = {"energy_scope": "device", "external_io": "device", "makespan": "sum_apps"}


def job(spec):
    from experiment import _backbone, _ecfg, _load
    from gat_imoea_dqn.algorithm import RunConfig, run
    from gat_imoea_dqn.selection import selection_actions
    out = pathlib.Path(spec["out"])
    if out.exists():
        return f"skip {out.name}"
    wf, sc = _load(spec["workflow"], ENV)
    action = selection_actions()[spec["action"]]
    cfg = RunConfig(pop_size=105, generations=spec["generations"], seed=spec["seed"], policy="fixed",
                    fixed_action=tuple(action), evaluator=_ecfg(ENV), tracker_normalization="fixed",
                    **_backbone("tch"))
    res = run(wf, sc, cfg)
    lo, hi = res.front_objectives()
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"front_lo": lo.tolist(), "front_hi": hi.tolist(), "seconds": res.seconds,
                               "reward": res.history["reward"], "pibi": res.history["pibi"]}))
    tmp.replace(out)
    return f"done {out.name} ({res.seconds:.0f}s)"


def score(instances, seeds, generations):
    from scipy.stats import spearmanr
    from gat_imoea_dqn.metrics import hv_jin, igd, pooled_reference
    from gat_imoea_dqn.selection import selection_actions
    acts = selection_actions()
    H = len(acts)
    lines = ["# Reward audit: 28 fixed actions on the tch backbone", "",
             f"G = {generations}, N = 105, {seeds} seeds, tracker normalisation 'fixed' (as the stable agents). "
             "Reward = relative PIBI improvement per generation; R = its sum over the run.", ""]
    summary = {}
    for w in instances:
        runs = {(a, s): json.loads((OUT / "runs" / f"{w}__a{a:02d}__s{s}.json").read_text())
                for a in range(H) for s in range(seeds)}
        allr = list(runs.values())
        ideal = np.min([np.min(r["front_lo"], 0) for r in allr], 0)
        nadir = np.max([np.max(r["front_hi"], 0) for r in allr], 0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        nz = lambda x: (np.asarray(x) - ideal) / span
        ref = pooled_reference([(nz(r["front_lo"]) + nz(r["front_hi"])) / 2 for r in allr])
        M = {k: np.zeros((H, seeds)) for k in ("IGD", "HV", "R", "PIBI_end")}
        rew = np.array([[runs[(a, s)]["reward"] for s in range(seeds)] for a in range(H)])   # (H, S, G)
        for (a, s), r in runs.items():
            lo, hi = nz(r["front_lo"]), nz(r["front_hi"])
            M["IGD"][a, s] = igd((lo + hi) / 2, ref)
            M["HV"][a, s] = hv_jin(lo, hi, [1.1] * 3)
            M["R"][a, s] = float(np.sum(r["reward"]))
            M["PIBI_end"][a, s] = r["pibi"][-1]
        mean = {k: v.mean(1) for k, v in M.items()}
        rho_igd = spearmanr(mean["R"], -mean["IGD"]).statistic
        rho_hv = spearmanr(mean["R"], mean["HV"]).statistic
        best_r, best_igd = int(np.argmax(mean["R"])), int(np.argmin(mean["IGD"]))
        G = rew.shape[2]
        snr = []
        for lo_t, hi_t in ((0, G // 10), (G // 10, G // 2), (G // 2, G)):
            win = rew[:, :, lo_t:hi_t]
            per_action = win.mean(axis=(1, 2))
            snr.append({"window": f"{lo_t}-{hi_t}", "spread_of_action_means": float(per_action.std()),
                        "per_step_sd": float(win.std()), "ratio": float(per_action.std() / max(win.std(), 1e-12)),
                        "mean_reward": float(win.mean())})
        summary[w] = {"actions": acts.tolist(), **{k: v.tolist() for k, v in M.items()},
                      "spearman_R_vs_IGD": rho_igd, "spearman_R_vs_HV": rho_hv, "snr": snr}
        lines += [f"## {w}", "",
                  f"Spearman(R, −IGD) = **{rho_igd:+.2f}**, Spearman(R, HV) = **{rho_hv:+.2f}** across the 28 actions. "
                  f"Highest R: {np.round(acts[best_r], 3).tolist()}; lowest IGD: {np.round(acts[best_igd], 3).tolist()}.",
                  "", "| action (α, β, γ) | IGD | HV_jin | R (sum of rewards) | final PIBI |", "|---|---|---|---|---|"]
        for a in np.argsort(mean["IGD"]):
            lines.append(f"| {np.round(acts[a], 3).tolist()} | {mean['IGD'][a]:.4f} | {mean['HV'][a]:.4f} | "
                         f"{mean['R'][a]:.3f} | {mean['PIBI_end'][a]:.4f} |")
        lines += ["", "Per-step reward signal-to-noise (spread of the 28 action means / per-step sd):", "",
                  "| generations | mean reward | spread of action means | per-step sd | ratio |", "|---|---|---|---|---|"]
        for d in snr:
            lines.append(f"| {d['window']} | {d['mean_reward']:.2e} | {d['spread_of_action_means']:.2e} | "
                         f"{d['per_step_sd']:.2e} | {d['ratio']:.3f} |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["small_20_i0@x1", "basic_60_i0@x20"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--generations", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if not a.score_only:
        specs = [{"workflow": w, "action": i, "seed": s, "generations": a.generations,
                  "out": str(OUT / "runs" / f"{w}__a{i:02d}__s{s}.json")}
                 for w in a.instances for i in range(28) for s in range(a.seeds)]
        with mp.Pool(a.workers) as pool:
            for msg in pool.imap_unordered(job, specs):
                print(msg, flush=True)
    score(a.instances, a.seeds, a.generations)


if __name__ == "__main__":
    main()
