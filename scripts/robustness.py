"""Normalisation robustness of final vs IMOEA/D (E3 follow-up).

E3 normalises each instance once over the pooled final fronts of the paper policy set, and builds
the IGD reference front from the same pool. Swapping two weak baselines (ip_moea -> ip_moea_hv,
trained_stable -> imoea_dqn) flipped three 1000-task IGD results between final and IMOEA/D without
any of their runs changing. This script recomputes final vs IMOEA/D on every instance under
several pools, each defining the normalisation (ideal / nadir) and the reference front:

  paper10  the current paper set (as in results/stats/suite_*.md)
  union12  paper set + ip_moea + trained_stable (every policy either table version used)
  strong5  final, imoead_orig, imoead_awn, nsga2_mid, nsga2_int (no weak baselines)
  pair2    final and imoead_orig only

Per pool and instance: mean IGD and HV_jin of both, Wilcoxon rank-sum (two-sided, alpha 0.05,
single comparison so no Holm), and A12. Writes results/stats/robustness.{md,csv}.

  python scripts/robustness.py
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
import pathlib
import re
import sys

import numpy as np

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")               # one BLAS thread per worker (12 workers ran out of memory)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SUITES = {"A": ["suite_a_main", "suite_a_e2", "rerun_ipmoea_a", "rerun_imoeadqn_a"],
          "B": ["suite_b", "suite_b_e2", "rerun_ipmoea_b", "rerun_imoeadqn_b"]}
PAPER10 = ["final", "imoead_orig", "imoead_awn", "imea_ied", "ip_moea_hv", "nsga2_mid", "nsga2_int",
           "nsga3_mid", "fixed_fav", "imoea_dqn"]
POOLS = {"paper10": PAPER10, "union12": PAPER10 + ["ip_moea", "trained_stable"],
         "strong5": ["final", "imoead_orig", "imoead_awn", "nsga2_mid", "nsga2_int"],
         "pair2": ["final", "imoead_orig"]}
A, B = "final", "imoead_orig"


def _job(args):
    inst, exps = args
    runs = load_one(exps, inst)
    from scipy.stats import mannwhitneyu
    from gat_imoea_dqn.metrics import hv_jin, igd, pooled_reference
    from stats import a12
    out = []
    for name, pols in POOLS.items():
        pool = {k: v for k, v in runs.items() if k[0] in pols}
        ideal = np.min([lo.min(0) for lo, _ in pool.values()], 0)
        nadir = np.max([hi.max(0) for _, hi in pool.values()], 0)
        span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
        norm = {k: ((lo - ideal) / span, (hi - ideal) / span) for k, (lo, hi) in pool.items()}
        ref = pooled_reference([(lo + hi) / 2 for lo, hi in norm.values()])
        v = {}
        for k, (lo, hi) in norm.items():
            if k[0] in (A, B):
                v.setdefault(k[0], []).append((igd((lo + hi) / 2, ref), hv_jin(lo, hi, [1.1] * 3)))
        res = {"pool": name, "ref_points": len(ref)}
        for j, (m, lb) in enumerate((("IGD", True), ("HV_jin", False))):
            x, y = np.array([t[j] for t in v[A]]), np.array([t[j] for t in v[B]])
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            better = x.mean() < y.mean() if lb else x.mean() > y.mean()
            res[m] = (x.mean(), y.mean(), "=" if p >= 0.05 else ("+" if better else "-"), a12(x, y, lb))
        out.append(res)
    return inst, out


def load_one(exps, inst):
    """Like stats.load, for one instance only (each worker loads its own)."""
    import json
    from stats import RUN_NAME
    d = {}
    for e in exps:
        for p in sorted((ROOT / "results" / "experiments" / e / "runs").glob(f"{inst}__*.json")):
            m = RUN_NAME.match(p.name)
            if not m or m["wf"] != inst or m["pol"] not in POOLS["union12"]:
                continue
            key = (m["pol"], int(m["s"]), int(m["t"]) if m["t"] is not None else None)
            if key not in d:
                r = json.loads(p.read_text())
                d[key] = (np.asarray(r["front_lo"], float), np.asarray(r["front_hi"], float))
    return d


def main() -> int:
    from stats import RUN_NAME
    L = ["# Normalisation robustness: final vs IMOEA/D", "",
         "Each pool sets the per-instance normalisation and the IGD reference front. Symbol = final "
         "vs IMOEA/D, Wilcoxon rank-sum, two-sided, alpha 0.05 (`+` final better, `−` worse, `=` no "
         "significant difference); final 30 seeds, IMOEA/D 30 seeds.", "",
         "Pools: " + "; ".join(f"`{k}` = {', '.join(v)}" for k, v in POOLS.items()), ""]
    rows = []
    for suite, exps in SUITES.items():
        names = {RUN_NAME.match(p.name)["wf"] for e in exps
                 for p in (ROOT / "results" / "experiments" / e / "runs").glob("*.json") if RUN_NAME.match(p.name)}
        insts = sorted(names, key=lambda w: (re.sub(r"\d+", "", w), [int(x) for x in re.findall(r"\d+", w)]))
        with mp.Pool(8) as pool:
            res = dict(pool.map(_job, [(w, exps) for w in insts], chunksize=1))
        L += [f"## Suite {suite} ({len(insts)} instances)", "",
              "| Pool | IGD +/−/= | HV_jin +/−/= |", "|---|---|---|"]
        for name in POOLS:
            c = {m: [res[w][i][m][2] for w in insts for i in range(len(POOLS)) if res[w][i]["pool"] == name]
                 for m in ("IGD", "HV_jin")}
            L.append(f"| {name} | " + " | ".join(f"{c[m].count('+')}/{c[m].count('-')}/{c[m].count('=')}"
                                                 for m in ("IGD", "HV_jin")) + " |")
        L += ["", "Instances where the verdict is not `+` in every pool (IGD final / IMOEA/D, symbol):", "",
              "| Instance | " + " | ".join(POOLS) + " | HV_jin symbols |", "|---|" + "---|" * (len(POOLS) + 1)]
        for w in insts:
            r = {x["pool"]: x for x in res[w]}
            for name in POOLS:
                rows.append([suite, w, name, r[name]["ref_points"], *(f"{x:.6g}" if isinstance(x, float) else x
                                                                     for m in ("IGD", "HV_jin") for x in r[name][m])])
            if any(r[n][m][2] != "+" for n in POOLS for m in ("IGD", "HV_jin")):
                L.append(f"| {w} | " + " | ".join(f"{r[n]['IGD'][0]:.4f} / {r[n]['IGD'][1]:.4f} {r[n]['IGD'][2]}"
                                                  for n in POOLS) +
                         " | " + " ".join(r[n]["HV_jin"][2] for n in POOLS) + " |")
        L.append("")
        print(f"suite {suite} done", flush=True)
    out = ROOT / "results" / "stats"
    (out / "robustness.md").write_text("\n".join(L), encoding="utf-8")
    with open(out / "robustness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["suite", "instance", "pool", "ref_points", "IGD_final", "IGD_imoead", "IGD_sym", "IGD_A12",
                    "HV_final", "HV_imoead", "HV_sym", "HV_A12"])
        w.writerows(rows)
    print("wrote", out / "robustness.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
