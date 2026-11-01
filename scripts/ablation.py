"""A1: backbone ablation of the decomposition-ranked selection on Suite A.

    python scripts/ablation.py --workers 12            # run missing jobs, then score
    python scripts/ablation.py --score-only

All variants hold the A2 action (5/6, 1/6, 0) fixed (no learning), G_max = 1000,
N = 105, 10 seeds, on the 12 Suite A test cases (i0 at CCR x1 and x20).

  Build-up (one component at a time, from Jin's backbone to the full method):
    jin        Jin et al. Rank C (R_IN), random tie-break, DAEC partner     [copied: suite_a fixed_fav]
    +tch       Tchebycheff Rank C
    +tch+icd   ... + ICD tie-break
    full       ... + neighbourhood mating (T = 20, p_local = 0.9)            [copied: suite_a_tch tch_fixed_fav]
  Leave one out of the full method:
    -tch       full with R_IN Rank C
    -icd       full with the random tie-break
    (-near is +tch+icd)
  Sensitivity of the full method:
    T10, T40           neighbourhood size
    p05, p10           p_local
    w53, w210          number of Tchebycheff weight vectors (default N = 105)
  References:
    final      full with T = 10, p_local = 1.0 (chosen on the training instances, scripts/tune_mating.py)
    conv       full with (1, 0, 0), the "no tuning" action                   [copied: suite_a_tch tch_conv]
    imoead     the IMOEA/D baseline                                         [copied: suite_a imoead]

Scoring matches scripts/experiment.py: one normalisation per instance across all
variants, IGD against the pooled non-dominated midpoints, HV_jin at reference 1.1;
Wilcoxon signed-rank (paired by seed, 5%) against `full`.
Output: results/experiments/ablation/summary.md and summary.json.
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

OUT = ROOT / "results" / "experiments" / "ablation"
EXP = ROOT / "results" / "experiments"
ENV = {"energy_scope": "device", "external_io": "device", "makespan": "sum_apps"}
FAV = (5 / 6, 1 / 6, 0.0)
CASES = [f"{c}_i0@x{m}" for c in ["small_20", "small_50", "basic_60", "basic_100", "medium_100", "medium_250"]
         for m in (1, 20)]

FULL = dict(convergence="tchebycheff", tie_break="icd", partner="neighbours", neighbours=20, p_local=0.9,
            n_weights=None)
VARIANTS = {
    "+tch": {**FULL, "tie_break": "random", "partner": "daec"},
    "+tch+icd": {**FULL, "partner": "daec"},
    "-tch": {**FULL, "convergence": "dominance"},
    "-icd": {**FULL, "tie_break": "random"},
    "T10": {**FULL, "neighbours": 10},
    "T40": {**FULL, "neighbours": 40},
    "p05": {**FULL, "p_local": 0.5},
    "p10": {**FULL, "p_local": 1.0},
    "w53": {**FULL, "n_weights": 53},
    "w210": {**FULL, "n_weights": 210},
    "final": {**FULL, "neighbours": 10, "p_local": 1.0},     # A1b: tuned on the training instances
}
COPIED = {"jin": ("suite_a", "fixed_fav"), "full": ("suite_a_tch", "tch_fixed_fav"),
          "conv": ("suite_a_tch", "tch_conv"), "imoead": ("suite_a", "imoead")}
ORDER = ["jin", "+tch", "+tch+icd", "full", "-tch", "-icd", "T10", "T40", "p05", "p10", "w53", "w210",
         "final", "conv", "imoead"]
GROUPS = [("Build-up", ["jin", "+tch", "+tch+icd", "full"]), ("Leave one out", ["full", "-tch", "-icd", "+tch+icd"]),
          ("Sensitivity", ["full", "T10", "T40", "p05", "p10", "w53", "w210"]), ("References", ["full", "final", "conv", "imoead"])]


def _file(w, v, s):
    return OUT / "runs" / f"{w}__{v.replace('+', 'p').replace('-', 'm')}__s{s}.json"


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
    cfg = RunConfig(pop_size=105, generations=spec["generations"], seed=spec["seed"], policy="fixed",
                    fixed_action=FAV, evaluator=_ecfg(ENV),
                    selection=SelectionConfig(convergence=v["convergence"], tie_break=v["tie_break"],
                                              n_weights=v["n_weights"]),
                    variation=VariationConfig(partner=v["partner"], neighbours=v["neighbours"], p_local=v["p_local"]))
    res = run(wf, sc, cfg)
    lo, hi = res.front_objectives()
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"front_lo": lo.tolist(), "front_hi": hi.tolist(), "seconds": res.seconds}))
    tmp.replace(out)
    return f"done {out.name} ({res.seconds:.0f}s)"


def score(instances, seeds):
    from scipy.stats import wilcoxon
    from gat_imoea_dqn.metrics import hv_jin, igd, imprecision, pooled_reference
    names = [n for n in ORDER if all(_file(w, n, s).exists() for w in instances for s in range(seeds))]
    keys = (("IGD", True), ("HV_jin", False), ("Imprecision", True))
    per = {}
    for w in instances:
        fronts = {n: [json.loads(_file(w, n, s).read_text()) for s in range(seeds)] for n in names}
        allr = [r for n in names for r in fronts[n]]
        ideal = np.min([np.min(r["front_lo"], 0) for r in allr], 0)
        nadir = np.max([np.max(r["front_hi"], 0) for r in allr], 0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        nz = lambda a: (np.asarray(a) - ideal) / span
        ref = pooled_reference([(nz(r["front_lo"]) + nz(r["front_hi"])) / 2 for r in allr])
        per[w] = {}
        for n in names:
            M = {"IGD": [], "HV_jin": [], "Imprecision": [], "seconds": []}
            for r in fronts[n]:
                lo, hi = nz(r["front_lo"]), nz(r["front_hi"])
                M["IGD"].append(igd((lo + hi) / 2, ref))
                M["HV_jin"].append(hv_jin(lo, hi, [1.1] * 3))
                M["Imprecision"].append(imprecision(lo, hi) / len(lo))
                M["seconds"].append(r.get("seconds", np.nan))
            per[w][n] = M

    def mark(w, n, k, low):
        a, b = np.array(per[w][n][k]), np.array(per[w]["full"][k])
        if n == "full" or np.allclose(a, b):
            return "="
        try:
            p = wilcoxon(a, b).pvalue
        except ValueError:
            p = 1.0
        return "=" if p >= 0.05 else ("+" if (a.mean() < b.mean()) == low else "−")

    lines = ["# A1: backbone ablation (Suite A test cases, fixed action (5/6, 1/6, 0), no learning)", "",
             f"{len(instances)} cases × {seeds} seeds, G_max = 1000, N = 105. One normalisation per case across "
             "all variants. Imprecision is per solution. W/L/T: Wilcoxon signed-rank (paired by seed, 5%) of each "
             "variant against **full** — W = the variant is significantly better than full, L = worse. "
             "IGD rank: variants ranked by mean IGD within each case, then averaged.", ""]
    igd_rank = {n: np.mean([1 + sum(np.mean(per[w][m]["IGD"]) < np.mean(per[w][n]["IGD"]) for m in names)
                            for w in instances]) for n in names}
    for title, group in GROUPS:
        group = [n for n in group if n in names]
        if len(group) < 2:
            continue
        lines += [f"## {title}", "", "| variant | mean IGD | IGD W/L/T vs full | mean HV_jin | HV W/L/T | "
                  "Imprecision | IGD rank (of all) | seconds |", "|---|---|---|---|---|---|---|---|"]
        for n in group:
            cells = []
            for k, low in keys[:2]:
                mk = [mark(w, n, k, low) for w in instances]
                cells.append((np.mean([np.mean(per[w][n][k]) for w in instances]),
                              f"{mk.count('+')}/{mk.count('−')}/{mk.count('=')}"))
            imp = np.mean([np.mean(per[w][n]["Imprecision"]) for w in instances])
            sec = np.nanmean([np.mean(per[w][n]["seconds"]) for w in instances])
            lines.append(f"| {n} | {cells[0][0]:.4f} | {cells[0][1] if n != 'full' else '—'} | {cells[1][0]:.4f} | "
                         f"{cells[1][1] if n != 'full' else '—'} | {imp:.3f} | {igd_rank[n]:.2f} | {sec:.0f} |")
        lines.append("")
    lines += ["## Mean IGD per case", "", "| case | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for w in instances:
        lines.append(f"| {w} | " + " | ".join(f"{np.mean(per[w][n]['IGD']):.4f}{'' if n == 'full' else ' ' + mark(w, n, 'IGD', True)}"
                                              for n in names) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"variants": {**VARIANTS, "full": FULL}, "per_case": per}, indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=CASES)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--generations", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    for w in a.instances:
        for n, (exp, pol) in COPIED.items():
            for s in range(a.seeds):
                src, dst = EXP / exp / "runs" / f"{w}__{pol}__s{s}.json", _file(w, n, s)
                if src.exists() and not dst.exists():
                    shutil.copy(src, dst)
    if not a.score_only:
        # large instances first so the pool's tail is short
        specs = [{"workflow": w, "variant": v, "seed": s, "generations": a.generations, "out": str(_file(w, v, s))}
                 for w in sorted(a.instances, key=lambda w: w.startswith("small")) for v in a.variants
                 for s in range(a.seeds)]
        with mp.Pool(a.workers) as pool:
            for msg in pool.imap_unordered(job, specs):
                print(msg, flush=True)
    score(a.instances, a.seeds)


if __name__ == "__main__":
    main()
