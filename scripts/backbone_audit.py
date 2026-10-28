"""Backbone audit: which IMOEA selection / mating choices close the gap to IMOEA/D?

    python scripts/backbone_audit.py --workers 12            # run missing jobs, then score
    python scripts/backbone_audit.py --score-only

Variants of the fixed-action IMOEA (no learning) at G_max = 1000, N = 105:
  base       fixed_fav (5/6, 1/6, 0), random tie-break in Rank C, no mating selection
  tie        + Rank C ties broken by descending ICD
  tour       + binary-tournament mating selection before DAEC
  both       tie + tour
  conv_tie   pure convergence (1, 0, 0) with the ICD tie-break
  near       fixed_fav with neighbourhood mating (partner among the 20 nearest, p = 0.9)
  near_tie   near + the ICD tie-break
  imoead_global  IMOEA/D with mating restriction off (delta = 0: partners from the whole population)
  imoead     the IMOEA/D baseline
base and imoead runs are copied from results/experiments/suite_a (identical code path).
Scoring matches scripts/experiment.py: one normalization per instance across all
variants, IGD against the pooled non-dominated midpoints, HV_jin at reference 1.1.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "experiments" / "backbone_audit"
SUITE = ROOT / "results" / "experiments" / "suite_a" / "runs"
ENV = {"energy_scope": "device", "external_io": "device", "makespan": "sum_apps"}
FAV = (5 / 6, 1 / 6, 0.0)
VARIANTS = {
    "tie": dict(action=FAV, tie_break="icd", mating="none"),
    "tour": dict(action=FAV, tie_break="random", mating="tournament"),
    "both": dict(action=FAV, tie_break="icd", mating="tournament"),
    "conv_tie": dict(action=(1.0, 0.0, 0.0), tie_break="icd", mating="none"),
    "near": dict(action=FAV, tie_break="random", mating="none", partner="neighbours"),
    "near_tie": dict(action=FAV, tie_break="icd", mating="none", partner="neighbours"),
    "imoead_global": dict(baseline="imoead", delta=0.0),
}
COPIED = {"base": "fixed_fav", "imoead": "imoead"}


def job(spec):
    from experiment import _ecfg, _load
    from gat_imoea_dqn.algorithm import RunConfig, run
    from gat_imoea_dqn.operators import VariationConfig
    from gat_imoea_dqn.selection import SelectionConfig
    out = pathlib.Path(spec["out"])
    if out.exists():
        return f"skip {out.name}"
    v = VARIANTS[spec["variant"]]
    wf, sc = _load(spec["workflow"], ENV)
    if "baseline" in v:
        from gat_imoea_dqn.baselines import BaselineConfig, run_baseline
        res = run_baseline(v["baseline"], wf, sc, BaselineConfig(generations=spec["generations"], seed=spec["seed"],
                                                                 evaluator=_ecfg(ENV), delta=v["delta"]))
        lo, hi = res.front_objectives()
        out.write_text(json.dumps({"front_lo": lo.tolist(), "front_hi": hi.tolist(), "seconds": res.seconds}))
        return f"done {out.name} ({res.seconds:.0f}s)"
    cfg = RunConfig(pop_size=105, generations=spec["generations"], seed=spec["seed"], policy="fixed",
                    fixed_action=v["action"], evaluator=_ecfg(ENV),
                    selection=SelectionConfig(tie_break=v["tie_break"]),
                    variation=VariationConfig(mating=v["mating"], partner=v.get("partner", "daec")))
    res = run(wf, sc, cfg)
    lo, hi = res.front_objectives()
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"front_lo": lo.tolist(), "front_hi": hi.tolist(), "seconds": res.seconds}))
    tmp.replace(out)
    return f"done {out.name} ({res.seconds:.0f}s)"


def score(instances, seeds):
    from gat_imoea_dqn.metrics import hv_jin, igd, imprecision, pooled_reference
    names = ["base", *VARIANTS, "imoead"]
    lines = ["# Backbone audit (G_max = 1000, N = 105, fixed actions, no learning)", "",
             "IGD ↓ · HV_jin ↑ · Imprecision ↓ · front size; mean ± std over seeds; "
             "`+`/`−`/`=` Wilcoxon vs base (5%); with 4 seeds the smallest possible p is 0.125, "
             "so read the means.", ""]
    from scipy.stats import wilcoxon
    summary = {}
    for w in instances:
        fronts = {n: [json.loads((OUT / "runs" / f"{w}__{n}__s{s}.json").read_text()) for s in range(seeds)]
                  for n in names}
        allr = [r for n in names for r in fronts[n]]
        ideal = np.min([np.min(r["front_lo"], 0) for r in allr], 0)
        nadir = np.max([np.max(r["front_hi"], 0) for r in allr], 0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        nz = lambda a: (np.asarray(a) - ideal) / span
        ref = pooled_reference([(nz(r["front_lo"]) + nz(r["front_hi"])) / 2 for r in allr])
        M = {}
        for n in names:
            M[n] = {"IGD": [], "HV_jin": [], "Imprecision": [], "size": []}
            for r in fronts[n]:
                lo, hi = nz(r["front_lo"]), nz(r["front_hi"])
                M[n]["IGD"].append(igd((lo + hi) / 2, ref))
                M[n]["HV_jin"].append(hv_jin(lo, hi, [1.1] * 3))
                M[n]["Imprecision"].append(imprecision(lo, hi))
                M[n]["size"].append(len(lo))
        summary[w] = M
        lines += [f"## {w}", "", "| Variant | IGD | HV_jin | Imprecision | front |", "|---|---|---|---|---|"]
        for n in names:
            cells = []
            for k, better_low in (("IGD", True), ("HV_jin", False), ("Imprecision", True)):
                a, b = np.array(M[n][k]), np.array(M["base"][k])
                mark = ""
                if n != "base":
                    try:
                        p = wilcoxon(a, b).pvalue
                    except ValueError:
                        p = 1.0
                    mark = " =" if p >= 0.05 else (" +" if (a.mean() < b.mean()) == better_low else " −")
                cells.append(f"{a.mean():.4g} ± {a.std():.2g}{mark}")
            lines.append(f"| {n} | " + " | ".join(cells) + f" | {np.mean(M[n]['size']):.0f} |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["small_20_i0@x1", "basic_60_i0@x20"])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--generations", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    for w in a.instances:
        for n, src in COPIED.items():
            for s in range(a.seeds):
                dst = OUT / "runs" / f"{w}__{n}__s{s}.json"
                if not dst.exists():
                    shutil.copy(SUITE / f"{w}__{src}__s{s}.json", dst)
    if not a.score_only:
        specs = [{"workflow": w, "variant": v, "seed": s, "generations": a.generations,
                  "out": str(OUT / "runs" / f"{w}__{v}__s{s}.json")}
                 for w in a.instances for v in VARIANTS for s in range(a.seeds)]
        with mp.Pool(a.workers) as pool:
            for msg in pool.imap_unordered(job, specs):
                print(msg, flush=True)
    score(a.instances, a.seeds)


if __name__ == "__main__":
    main()
