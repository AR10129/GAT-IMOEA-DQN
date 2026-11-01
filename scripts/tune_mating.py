"""A1b: tune the neighbourhood-mating parameters on the training instances only.

    python scripts/tune_mating.py --workers 12            # run missing jobs, then score
    python scripts/tune_mating.py --score-only

A1 showed the full method is sensitive to the neighbourhood size T and p_local,
but on the test cases. This re-selects them the way A2 selected the action:
T in {3, 5, 10, 20} x p_local in {0.9, 1.0}, action (5/6, 1/6, 0), tch backbone,
G_max = 1000, N = 105, on the 36 training instances (i1-i3 of the 6 configurations
at CCR x1 and x20), 2 seeds. Test instances (i0) are not used. Scoring per
instance: one normalisation across all settings and seeds, IGD against the pooled
non-dominated midpoints, HV_jin at 1.1; settings ranked by mean IGD within the
instance, then averaged. The choice is the best mean IGD rank (ties: HV rank).
Output: results/experiments/tune_mating/summary.md and summary.json.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from action_sweep import INSTANCES  # noqa: E402  (the 36 training instances)

OUT = ROOT / "results" / "experiments" / "tune_mating"
ENV = {"energy_scope": "device", "external_io": "device", "makespan": "sum_apps"}
FAV = (5 / 6, 1 / 6, 0.0)
SETTINGS = [(T, p) for T, p in itertools.product((3, 5, 10, 20), (0.9, 1.0))]
DEFAULT = (20, 0.9)


def _name(T, p):
    return f"T{T}_p{int(round(p * 100))}"


def job(spec):
    from experiment import _ecfg, _load
    from gat_imoea_dqn.algorithm import RunConfig, run
    from gat_imoea_dqn.operators import VariationConfig
    from gat_imoea_dqn.selection import SelectionConfig
    out = pathlib.Path(spec["out"])
    if out.exists():
        return f"skip {out.name}"
    wf, sc = _load(spec["workflow"], ENV)
    cfg = RunConfig(pop_size=105, generations=spec["generations"], seed=spec["seed"], policy="fixed",
                    fixed_action=FAV, evaluator=_ecfg(ENV),
                    selection=SelectionConfig(convergence="tchebycheff", tie_break="icd"),
                    variation=VariationConfig(partner="neighbours", neighbours=spec["T"], p_local=spec["p"]))
    res = run(wf, sc, cfg)
    lo, hi = res.front_objectives()
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"front_lo": lo.tolist(), "front_hi": hi.tolist(), "seconds": res.seconds}))
    tmp.replace(out)
    return f"done {out.name} ({res.seconds:.0f}s)"


def score(instances, seeds):
    from scipy.stats import rankdata
    from gat_imoea_dqn.metrics import hv_jin, igd, pooled_reference
    names = [_name(*st) for st in SETTINGS]
    H = len(names)
    per = {}
    for w in instances:
        runs = {(n, s): json.loads((OUT / "runs" / f"{w}__{n}__s{s}.json").read_text())
                for n in names for s in range(seeds)}
        allr = list(runs.values())
        ideal = np.min([np.min(r["front_lo"], 0) for r in allr], 0)
        nadir = np.max([np.max(r["front_hi"], 0) for r in allr], 0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        nz = lambda x: (np.asarray(x) - ideal) / span
        ref = pooled_reference([(nz(r["front_lo"]) + nz(r["front_hi"])) / 2 for r in allr])
        I, V = np.zeros(H), np.zeros(H)
        for (n, s), r in runs.items():
            lo, hi = nz(r["front_lo"]), nz(r["front_hi"])
            k = names.index(n)
            I[k] += igd((lo + hi) / 2, ref) / seeds
            V[k] += hv_jin(lo, hi, [1.1] * 3) / seeds
        per[w] = {"IGD": I.tolist(), "HV": V.tolist(), "IGD_rank": rankdata(I).tolist(),
                  "HV_rank": rankdata(-V).tolist(), "IGD_ratio": (I / I.min()).tolist()}

    def agg(ws, key):
        return np.mean([per[w][key] for w in ws], axis=0)

    ir, hr = agg(instances, "IGD_rank"), agg(instances, "HV_rank")
    ratio = np.array([per[w]["IGD_ratio"] for w in instances])
    order = np.lexsort((hr, ir))
    best = int(order[0])
    d = names.index(_name(*DEFAULT))
    beats = sum(per[w]["IGD"][best] < per[w]["IGD"][d] for w in instances)
    lines = ["# A1b: neighbourhood-mating parameters tuned on the training instances", "",
             f"T × p_local grid, action (5/6, 1/6, 0), tch backbone, {len(instances)} training instances "
             f"(i1–i3, CCR ×1 and ×20) × {seeds} seeds, G_max = 1000, N = 105. Test instances (i0) not used.", "",
             f"**Chosen: T = {SETTINGS[best][0]}, p_local = {SETTINGS[best][1]}** — mean IGD rank {ir[best]:.2f} of {H}; "
             f"lower mean IGD than the old default (T = 20, p_local = 0.9) on {beats}/{len(instances)} instances.", "",
             "| setting | mean IGD rank | mean HV rank | IGD ratio (mean / worst) | best on # instances |",
             "|---|---|---|---|---|"]
    wins = np.bincount([int(np.argmin(per[w]["IGD"])) for w in instances], minlength=H)
    for k in order:
        lines.append(f"| T = {SETTINGS[k][0]}, p_local = {SETTINGS[k][1]} | {ir[k]:.2f} | {hr[k]:.2f} | "
                     f"{ratio[:, k].mean():.3f} / {ratio[:, k].max():.3f} | {wins[k]} |")
    groups = {"CCR x1": [w for w in instances if w.endswith("@x1")],
              "CCR x20": [w for w in instances if w.endswith("@x20")]}
    for c in sorted({w.rsplit("_i", 1)[0] for w in instances}, key=lambda c: instances.index(next(
            w for w in instances if w.startswith(c + "_")))):
        groups[c] = [w for w in instances if w.startswith(c + "_")]
    lines += ["", "## Mean IGD rank by group", "", "| group | " + " | ".join(names) + " |", "|---" * (H + 1) + "|"]
    for g, ws in groups.items():
        if ws:
            r = agg(ws, "IGD_rank")
            lines.append(f"| {g} | " + " | ".join(f"{v:.2f}" for v in r) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"settings": SETTINGS, "chosen": SETTINGS[best],
                                                  "igd_rank": ir.tolist(), "hv_rank": hr.tolist(),
                                                  "per_instance": per}, indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--generations", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if not a.score_only:
        specs = [{"workflow": w, "T": T, "p": p, "seed": s, "generations": a.generations,
                  "out": str(OUT / "runs" / f"{w}__{_name(T, p)}__s{s}.json")}
                 for w in sorted(a.instances, key=lambda w: w.startswith("small")) for T, p in SETTINGS
                 for s in range(a.seeds)]
        with mp.Pool(a.workers) as pool:
            for msg in pool.imap_unordered(job, specs):
                print(msg, flush=True)
    score(a.instances, a.seeds)


if __name__ == "__main__":
    main()
