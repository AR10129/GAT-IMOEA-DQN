"""A2: choose the default (alpha, beta, gamma) on the training instances only.

    python scripts/action_sweep.py --workers 12            # run missing jobs, then score
    python scripts/action_sweep.py --score-only

Each of the 28 Das-Dennis actions is held fixed for a whole run on the
decomposition (tch) backbone, on the Suite A training / validation instances
(i1-i3 of the 6 configurations, at CCR x1 and x20: 36 instances), G_max = 1000,
N = 105. The test instances (i0) are never used. Scoring, per instance: one
normalisation across the 28 runs, IGD against the pooled non-dominated
midpoints, HV_jin at reference 1.1, Imprecision; actions are then ranked within
the instance. The default action is the best mean IGD rank (ties: HV rank).
Output: results/experiments/action_sweep/summary.md and summary.json.
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

from reward_audit import job  # noqa: E402  (same run: fixed action, tch backbone, fixed tracker)

OUT = ROOT / "results" / "experiments" / "action_sweep"
CONFIGS = ["small_20", "small_50", "basic_60", "basic_100", "medium_100", "medium_250"]
INSTANCES = [f"{c}_i{i}@x{m}" for c in CONFIGS for i in (1, 2, 3) for m in (1, 20)]


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, 1 = smallest."""
    from scipy.stats import rankdata
    return rankdata(x)


def score(instances, seeds):
    from gat_imoea_dqn.metrics import hv_jin, igd, imprecision, pooled_reference
    from gat_imoea_dqn.selection import selection_actions
    A = selection_actions()
    H = len(A)
    per = {}
    for w in instances:
        runs = {(a, s): json.loads((OUT / "runs" / f"{w}__a{a:02d}__s{s}.json").read_text())
                for a in range(H) for s in range(seeds)}
        allr = list(runs.values())
        ideal = np.min([np.min(r["front_lo"], 0) for r in allr], 0)
        nadir = np.max([np.max(r["front_hi"], 0) for r in allr], 0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        nz = lambda x: (np.asarray(x) - ideal) / span
        ref = pooled_reference([(nz(r["front_lo"]) + nz(r["front_hi"])) / 2 for r in allr])
        M = {k: np.zeros(H) for k in ("IGD", "HV", "Imprecision")}
        for (a, s), r in runs.items():
            lo, hi = nz(r["front_lo"]), nz(r["front_hi"])
            M["IGD"][a] += igd((lo + hi) / 2, ref) / seeds
            M["HV"][a] += hv_jin(lo, hi, [1.1] * 3) / seeds
            M["Imprecision"][a] += imprecision(lo, hi) / len(lo) / seeds      # per solution
        per[w] = {k: v.tolist() for k, v in M.items()}
        per[w]["IGD_rank"] = _rank(M["IGD"]).tolist()
        per[w]["HV_rank"] = _rank(-M["HV"]).tolist()
        per[w]["IGD_ratio"] = (M["IGD"] / M["IGD"].min()).tolist()

    def agg(ws, key):
        return np.mean([per[w][key] for w in ws], axis=0)

    igd_rank, hv_rank = agg(instances, "IGD_rank"), agg(instances, "HV_rank")
    order = np.lexsort((hv_rank, igd_rank))
    best = int(order[0])
    groups = {"CCR x1": [w for w in instances if w.endswith("@x1")],
              "CCR x20": [w for w in instances if w.endswith("@x20")]}
    for c in CONFIGS:
        groups[c] = [w for w in instances if w.startswith(c + "_")]
    ratio = np.array([per[w]["IGD_ratio"] for w in instances])        # (W, H)
    fav = int(np.argmin(np.abs(A - np.array([5 / 6, 1 / 6, 0])).sum(1)))
    conv = int(np.argmin(np.abs(A - np.array([1.0, 0, 0])).sum(1)))

    fmt = lambda a: "(" + ", ".join(f"{v:.3g}" for v in A[a]) + ")"
    lines = ["# A2: action sweep on the training instances", "",
             f"28 fixed actions × {len(instances)} instances (Suite A i1–i3, CCR ×1 and ×20) × {seeds} seed(s); "
             "tch backbone, G_max = 1000, N = 105. Test instances (i0) are not used. Ranks are within each "
             "instance (1 = best), then averaged. IGD ratio = IGD / best IGD on that instance.", "",
             f"**Chosen default: {fmt(best)}** — best mean IGD rank ({igd_rank[best]:.2f}); mean HV rank "
             f"{hv_rank[best]:.2f}; mean IGD ratio {ratio[:, best].mean():.2f}, worst {ratio[:, best].max():.2f}.", "",
             f"For reference: (5/6, 1/6, 0) has mean IGD rank {igd_rank[fav]:.2f} (ratio {ratio[:, fav].mean():.2f}); "
             f"(1, 0, 0) has {igd_rank[conv]:.2f} (ratio {ratio[:, conv].mean():.2f}).", "",
             "## All actions (sorted by mean IGD rank)", "",
             "| action (α, β, γ) | IGD rank | HV rank | IGD ratio (mean / worst) | Imprecision per solution | "
             "best on # instances |", "|---|---|---|---|---|---|"]
    wins = np.bincount([int(np.argmin(per[w]["IGD"])) for w in instances], minlength=H)
    imp = agg(instances, "Imprecision")
    for a in order:
        lines.append(f"| {fmt(a)} | {igd_rank[a]:.2f} | {hv_rank[a]:.2f} | {ratio[:, a].mean():.2f} / "
                     f"{ratio[:, a].max():.2f} | {imp[a]:.3f} | {wins[a]} |")
    lines += ["", "## Mean IGD rank by α (rows) and β (columns); γ = 1 − α − β", "",
              "| α \\ β | " + " | ".join(f"{b:.3g}" for b in np.arange(7) / 6) + " |", "|---" * 8 + "|"]
    for i in range(6, -1, -1):
        cells = []
        for j in range(7):
            m = np.flatnonzero(np.all(np.isclose(A[:, :2], [i / 6, j / 6]), axis=1))
            cells.append(f"{igd_rank[m[0]]:.1f}" if len(m) else "")
        lines.append(f"| {i / 6:.3g} | " + " | ".join(cells) + " |")
    lines += ["", "## Top 5 by group (mean IGD rank)", "", "| group | 1st | 2nd | 3rd | 4th | 5th | rank of default |",
              "|---|---|---|---|---|---|---|"]
    for g, ws in groups.items():
        if not ws:
            continue
        r = agg(ws, "IGD_rank")
        top = np.argsort(r)[:5]
        lines.append(f"| {g} | " + " | ".join(f"{fmt(a)} {r[a]:.1f}" for a in top) +
                     f" | {int(np.sum(r < r[best])) + 1} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"actions": A.tolist(), "default": A[best].tolist(),
                                                  "igd_rank": igd_rank.tolist(), "hv_rank": hv_rank.tolist(),
                                                  "per_instance": per}, indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--generations", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if not a.score_only:
        # large instances first so the pool's tail is short
        specs = [{"workflow": w, "action": i, "seed": s, "generations": a.generations,
                  "out": str(OUT / "runs" / f"{w}__a{i:02d}__s{s}.json")}
                 for w in sorted(a.instances, key=lambda w: w.startswith("small")) for i in range(28)
                 for s in range(a.seeds)]
        with mp.Pool(a.workers) as pool:
            for msg in pool.imap_unordered(job, specs):
                print(msg, flush=True)
    score(a.instances, a.seeds)


if __name__ == "__main__":
    main()
