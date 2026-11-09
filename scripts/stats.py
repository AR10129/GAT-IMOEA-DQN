"""E3 / E4: metrics on one fixed normalisation per instance, and the paper's statistics.

For each instance, every run of the listed policies from the listed experiments is pooled
(identical copies across experiments are counted once). The normalisation (ideal / nadir) and
the IGD reference front are computed once over that pool, so a policy's score does not depend
on which experiment it was run in.

Statistics (PLAN, "Statistics"):
  * Wilcoxon rank-sum, each policy vs the reference, Holm correction over the policies on each
    instance and metric; '+' = significantly better than the reference, '-' = worse, '=' = no
    significant difference (Jin et al.'s convention). Rank-sum allows unequal seed counts.
  * Vargha-Delaney A12 = P(policy better than reference) + 0.5 P(tie); 0.5 = no effect.
  * Friedman test over instances on the per-instance means, average ranks, Nemenyi critical
    difference at 5%.

Usage:
  python scripts/stats.py --name suite_a --experiments suite_a_main suite_a_e2
  python scripts/stats.py --name suite_b --experiments suite_b suite_b_e2
Writes results/stats/<name>.md and <name>_runs.csv (one row per run and metric set).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gat_imoea_dqn.metrics import hv_jin, hypervolume_3d, igd, imprecision, pooled_reference  # noqa: E402

RUN_NAME = re.compile(r"^(?P<wf>.+?)__(?P<pol>[a-z0-9_]+?)(?:__t(?P<t>\d+))?__s(?P<s>\d+)\.json$")
# the paper's comparison set; the old imoead schedule, fresh DQN and the heuristics are excluded
# (heuristics would stretch the normalisation; they are reported separately)
PAPER = ["final", "imoead_orig", "imoead_awn", "imea_ied", "ip_moea", "nsga2_mid", "nsga2_int",
         "nsga3_mid", "fixed_fav", "trained_stable"]
LABEL = {"final": "Ours (final)", "imoead_orig": "IMOEA/D", "imoead_awn": "IMOEA/D-AWN", "imea_ied": "IMEA-IED",
         "ip_moea": "IP-MOEA", "nsga2_mid": "NSGA-II (mid)", "nsga2_int": "NSGA-II (int)",
         "nsga3_mid": "NSGA-III (mid)", "fixed_fav": "Jin backbone (fixed)", "trained_stable": "IMOEA-DQN (ours model)"}
METRICS = {"IGD": True, "HV_jin": False, "Imprecision": True, "HV_hi": False}    # name -> lower is better
# Nemenyi q_0.05 for k = 2..10 (Demsar 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def load(experiments, policies):
    """{instance: {(policy, seed, tseed): (lo, hi)}} pooled over experiments, first copy wins."""
    out = {}
    for e in experiments:
        for p in sorted((ROOT / "results" / "experiments" / e / "runs").glob("*.json")):
            m = RUN_NAME.match(p.name)
            if not m or m["pol"] not in policies:
                continue
            key = (m["pol"], int(m["s"]), int(m["t"]) if m["t"] is not None else None)
            d = out.setdefault(m["wf"], {})
            if key in d:
                continue
            r = json.loads(p.read_text())
            d[key] = (np.asarray(r["front_lo"], float), np.asarray(r["front_hi"], float))
    return out


def score(runs):
    """Per-run metrics on the instance's fixed normalisation; agent runs averaged over training seeds."""
    keys = list(runs)
    ideal = np.min([lo.min(0) for lo, _ in runs.values()], 0)
    nadir = np.max([hi.max(0) for _, hi in runs.values()], 0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    norm = {k: ((lo - ideal) / span, (hi - ideal) / span) for k, (lo, hi) in runs.items()}
    ref = pooled_reference([(lo + hi) / 2 for lo, hi in norm.values()])
    per = {}
    for k in keys:
        lo, hi = norm[k]
        v = {"IGD": igd((lo + hi) / 2, ref), "HV_jin": hv_jin(lo, hi, [1.1] * 3),
             "Imprecision": imprecision(lo, hi), "HV_hi": hypervolume_3d(hi, [1.1] * 3)}
        per.setdefault((k[0], k[1]), []).append(v)
    return {pk: {m: float(np.mean([v[m] for v in vs])) for m in METRICS} for pk, vs in per.items()}


def a12(x, y, lower_better):
    """P(x better than y) + 0.5 P(tie)."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    better = (x < y) if lower_better else (x > y)
    return float(better.mean() + 0.5 * (x == y).mean())


def holm(pvals):
    """Holm-adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def magnitude(a):
    d = abs(a - 0.5)
    return "N" if d < 0.06 else "S" if d < 0.14 else "M" if d < 0.21 else "L"


def main() -> int:
    from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", default=PAPER)
    ap.add_argument("--reference", default="final")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    data = load(args.experiments, set(args.policies))
    insts = sorted(data, key=lambda w: (re.sub(r"\d+", "", w), [int(x) for x in re.findall(r"\d+", w)]))
    pols = [p for p in args.policies if any(k[0] == p for d in data.values() for k in d)]
    others = [p for p in pols if p != args.reference]
    res = {}
    for w in insts:
        res[w] = score(data[w])
        print(f"scored {w} ({len(data[w])} runs)", flush=True)

    vals = {w: {p: {m: [v[m] for (pp, s), v in sorted(res[w].items()) if pp == p] for m in METRICS} for p in pols}
            for w in insts}
    seeds = {p: sorted({len(vals[w][p]["IGD"]) for w in insts}) for p in pols}

    # Wilcoxon rank-sum + Holm, A12, per instance and metric
    sym, A = {}, {}
    for w in insts:
        for m, lb in METRICS.items():
            ref = vals[w][args.reference][m]
            ps, signs = [], []
            for p in others:
                x = vals[w][p][m]
                try:
                    ps.append(mannwhitneyu(x, ref, alternative="two-sided").pvalue if len(x) > 1 else 1.0)
                except ValueError:                       # all values identical
                    ps.append(1.0)
                signs.append(np.mean(x) < np.mean(ref) if lb else np.mean(x) > np.mean(ref))
                A[(w, m, p)] = a12(x, ref, lb)
            for p, padj, better in zip(others, holm(ps), signs):
                sym[(w, m, p)] = "=" if padj >= args.alpha else ("+" if better else "-")

    means = {(w, p, m): float(np.mean(vals[w][p][m])) for w in insts for p in pols for m in METRICS}
    sds = {(w, p, m): float(np.std(vals[w][p][m])) for w in insts for p in pols for m in METRICS}

    # Friedman over instances on the means
    fried = {}
    for m, lb in METRICS.items():
        M = np.array([[means[(w, p, m)] for p in pols] for w in insts])
        R = np.array([rankdata(row if lb else -row) for row in M])
        stat, pv = friedmanchisquare(*M.T) if len(pols) > 2 else (np.nan, np.nan)
        k, n = len(pols), len(insts)
        cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * n))
        fried[m] = (dict(zip(pols, R.mean(0))), stat, pv, cd)

    # ------------------------------------------------------------- report ---
    out = ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    L = [f"# Statistics: `{args.name}`", "",
         f"Experiments pooled: {', '.join(args.experiments)}. Instances: {len(insts)}. Policies and seeds per "
         "instance: " + ", ".join(f"{p} {'/'.join(map(str, seeds[p]))}" for p in pols) + ".",
         "One normalisation and one pooled reference front per instance, over every run of these policies.",
         f"Wilcoxon rank-sum vs `{args.reference}`, Holm-corrected over the {len(others)} policies per instance "
         f"and metric, alpha {args.alpha}: `+` better than {args.reference}, `−` worse, `=` no significant "
         "difference. A12 = P(policy better than reference); N/S/M/L = negligible / small / medium / large.", ""]
    L += ["## Overall", "",
          "| Policy | " + " | ".join(f"{m} mean" for m in METRICS) + " | " +
          " | ".join(f"{m} +/−/=" for m in METRICS) + " | " + " | ".join(f"{m} A12" for m in ("IGD", "HV_jin")) + " |",
          "|---|" + "---|" * (2 * len(METRICS) + 2)]
    for p in pols:
        row = [LABEL.get(p, p)]
        row += [f"{np.mean([means[(w, p, m)] for w in insts]):.4g}" for m in METRICS]
        if p == args.reference:
            row += ["(reference)"] * len(METRICS) + ["–", "–"]
        else:
            for m in METRICS:
                s = [sym[(w, m, p)] for w in insts]
                row.append(f"{s.count('+')}/{s.count('-')}/{s.count('=')}")
            for m in ("IGD", "HV_jin"):
                a = [A[(w, m, p)] for w in insts]
                mg = [magnitude(x) for x in a]
                row.append(f"{np.mean(a):.2f} (L{mg.count('L')})")
        L.append("| " + " | ".join(row).replace("-/", "−/") + " |")
    L += ["", "## Friedman ranks (over instances, on per-instance means; lower rank = better)", "",
          "| Metric | χ² | p | Nemenyi CD (5%) | " + " | ".join(LABEL.get(p, p) for p in pols) + " |",
          "|---|---|---|---|" + "---|" * len(pols)]
    for m, (rk, st, pv, cd) in fried.items():
        L.append(f"| {m} | {st:.1f} | {pv:.2g} | {cd:.2f} | " + " | ".join(f"{rk[p]:.2f}" for p in pols) + " |")
    for m in METRICS:
        L += ["", f"## {m} per instance (mean ± sd; symbol vs {args.reference})", "",
              "| Instance | " + " | ".join(LABEL.get(p, p) for p in pols) + " |", "|---|" + "---|" * len(pols)]
        for w in insts:
            best = (min if METRICS[m] else max)(means[(w, p, m)] for p in pols)
            cells = []
            for p in pols:
                c = f"{means[(w, p, m)]:.4g} ± {sds[(w, p, m)]:.2g}"
                if means[(w, p, m)] == best:
                    c = f"**{c}**"
                if p != args.reference:
                    c += " " + sym[(w, m, p)].replace("-", "−")
                cells.append(c)
            L.append(f"| {w} | " + " | ".join(cells) + " |")
    (out / f"{args.name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(out / f"{args.name}_runs.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["instance", "policy", "seed"] + list(METRICS))
        for w in insts:
            for (p, s), v in sorted(res[w].items()):
                wr.writerow([w, p, s] + [f"{v[m]:.6g}" for m in METRICS])
    print("wrote", out / f"{args.name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
