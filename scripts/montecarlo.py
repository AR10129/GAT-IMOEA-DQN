"""E7: Monte Carlo check of the final fronts under sampled crisp values.

Every uncertain input of the evaluator is an interval: task workloads C, edge data sizes D, node
CPU speeds f, link rates R and device energy per cycle e. A crisp scenario draws each of them
uniformly and independently inside its interval (one value per task / edge / node / link /
device, symmetric links stay symmetric) and evaluates every saved front solution with lo = hi =
the drawn value. All policies see the same scenarios (common random numbers).

Per instance this reports
  * containment: share of (solution, scenario) pairs whose crisp objectives lie inside the
    solution's predicted interval in all three objectives (should be 100%);
  * position: where inside the interval the crisp value falls (0 = lower bound, 1 = upper bound),
    and the 2.5-97.5% spread of the crisp values as a share of the interval width;
  * per scenario, each run's crisp front (non-dominated subset of its solutions' crisp values):
    HV (ref 1.1) and IGD against that scenario's pooled crisp front, normalised as in E3;
    per policy the mean over seeds, then mean / sd over scenarios, and the share of scenarios in
    which final is better than the policy;
  * Kendall tau between the policies' E3 interval IGD order and their Monte Carlo IGD order.

  python scripts/montecarlo.py --name e7 --scenarios 200 \
      --instances medium_250_i0@x20:e7_a:suite_a_main,suite_a_e2 montage_400@x20:e7_b:suite_b,suite_b_e2 ...
Writes results/stats/<name>_montecarlo.md and <name>_montecarlo.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_EV = None          # per worker: (interval evaluator, X, O)


def crisp_evaluator(ev, rng):
    """A copy of `ev` with every interval input replaced by one uniform draw (lo = hi)."""
    from dataset_tools.loader import KB_TO_MBIT
    c = copy.copy(ev)
    u = lambda lo, hi: lo + rng.random(np.shape(lo)) * (hi - lo)       # noqa: E731
    c.C_lo = c.C_hi = u(ev.C_lo, ev.C_hi)
    c.D_lo = c.D_hi = u(ev.D_lo, ev.D_hi)
    c.f_lo = c.f_hi = u(ev.f_lo, ev.f_hi)
    e_dev = u(ev.sc.dev_e[:, 0].astype(float), ev.sc.dev_e[:, 1].astype(float))
    c.e_lo_t = c.e_hi_t = e_dev[ev.owner]
    # link rates: draw R in [R_lo, R_hi]; seconds per KB = KB_TO_MBIT / R (co-located 0 and
    # unreachable inf stay as they are)
    live = (ev.spk_lo > 0) & np.isfinite(ev.spk_lo) & np.isfinite(ev.spk_hi)
    r_lo = np.where(live, KB_TO_MBIT / np.where(live, ev.spk_hi, 1.0), 0.0)
    r_hi = np.where(live, KB_TO_MBIT / np.where(live, ev.spk_lo, 1.0), 0.0)
    R = u(r_lo, r_hi)
    if np.allclose(ev.spk_lo, ev.spk_lo.T) and np.allclose(ev.spk_hi, ev.spk_hi.T):
        R = np.triu(R) + np.triu(R, 1).T
    spk = np.where(live, KB_TO_MBIT / np.where(live, R, 1.0), ev.spk_lo)
    c.spk_lo = c.spk_hi = np.ascontiguousarray(spk)
    for k in ("C_lo", "D_lo", "f_lo", "e_lo_t"):
        setattr(c, k, np.ascontiguousarray(getattr(c, k), dtype=float))
        setattr(c, k.replace("_lo", "_hi"), getattr(c, k))
    return c


def _init(workflow, env, X, O):
    global _EV
    import experiment as E
    from gat_imoea_dqn.evaluator import IntervalEvaluator
    wf, sc = E._load(workflow, env)
    _EV = (IntervalEvaluator(wf, sc, E._ecfg(env)), X, O)


def _scenario(seed):
    ev, X, O = _EV
    F_lo, F_hi = crisp_evaluator(ev, np.random.default_rng(seed)).evaluate_population(X, O)
    assert np.allclose(F_lo, F_hi)
    return F_lo


def kendall(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


def main() -> int:
    from convergence import normalisation
    from gat_imoea_dqn.metrics import hypervolume_3d, igd, nondominated
    from stats import LABEL, PAPER, RUN_NAME, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e7")
    ap.add_argument("--instances", nargs="+", required=True, help="instance:experiment:pool1,pool2")
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    L = [f"# E7 Monte Carlo: `{args.name}`", "",
         f"{args.scenarios} crisp scenarios per instance, every interval input drawn uniformly and independently; "
         "the same scenarios for every policy. HV (ref 1.1) and IGD on the E3 normalisation; IGD against each "
         "scenario's pooled crisp front. `final better` = share of scenarios where final's seed-mean beats the "
         "policy's.", ""]
    rows = []
    for item in args.instances:
        inst, exp, pools = item.split(":")
        cfg = json.loads((ROOT / "results" / "experiments" / exp / "config.json").read_text())
        env = {"scenario": cfg["scenario"], "data_multiplier": cfg.get("data_multiplier"),
               "energy_scope": cfg["energy_scope"], "external_io": cfg["external_io"], "makespan": cfg["makespan"],
               "width_scale": cfg.get("width_scale"), "edge_queue": cfg.get("edge_queue", False)}
        runs = []                                       # (policy, seed, start, end)
        Xs, Os, Lo, Hi = [], [], [], []
        for p in sorted((ROOT / "results" / "experiments" / exp / "runs").glob(f"{inst}__*.json")):
            m = RUN_NAME.match(p.name)
            if not m or m["pol"] not in PAPER:
                continue
            r = json.loads(p.read_text())
            n0 = sum(len(x) for x in Xs)
            Xs.append(np.asarray(r["front_X"])), Os.append(np.asarray(r["front_O"]))
            Lo.append(np.asarray(r["front_lo"])), Hi.append(np.asarray(r["front_hi"]))
            runs.append((m["pol"], int(m["s"]), n0, n0 + len(r["front_X"])))
        X, O = np.vstack(Xs).astype(np.int64), np.vstack(Os).astype(np.int64)
        F_lo, F_hi = np.vstack(Lo), np.vstack(Hi)
        print(f"{inst}: {len(runs)} runs, {len(X)} solutions, {args.scenarios} scenarios", flush=True)
        with mp.Pool(args.workers, initializer=_init, initargs=(inst, env, X, O)) as pool:
            crisp = np.stack(pool.map(_scenario, [10_000 + s for s in range(args.scenarios)]))   # (S, n, 3)

        width = F_hi - F_lo
        inside = np.all((crisp >= F_lo - 1e-9 * (1 + np.abs(F_lo))) & (crisp <= F_hi + 1e-9 * (1 + np.abs(F_hi))),
                        axis=2)
        pos = np.where(width > 0, (crisp - F_lo) / np.where(width > 0, width, 1), 0.5)
        spread = np.percentile(crisp, 97.5, axis=0) - np.percentile(crisp, 2.5, axis=0)
        share = np.where(width > 0, spread / np.where(width > 0, width, 1), np.nan)

        ideal, span, _ = normalisation(load(pools.split(","), set(PAPER)).get(inst, {}))
        pols = [p for p in PAPER if any(r[0] == p for r in runs)]
        HV = {p: np.zeros(args.scenarios) for p in pols}
        IG = {p: np.zeros(args.scenarios) for p in pols}
        cnt = {p: sum(r[0] == p for r in runs) for p in pols}
        for s in range(args.scenarios):
            Fn = (crisp[s] - ideal) / span
            fronts = [(pol, nondominated(Fn[a:b])) for pol, _, a, b in runs]
            ref = nondominated(np.vstack([f for _, f in fronts]))
            for pol, f in fronts:
                HV[pol][s] += hypervolume_3d(f, [1.1] * 3) / cnt[pol]
                IG[pol][s] += igd(f, ref) / cnt[pol]

        # E3 interval IGD (pooled stats) for the rank comparison
        e3 = {}
        stats_csv = ROOT / "results" / "stats" / ("suite_a_runs.csv" if exp.endswith("_a") else "suite_b_runs.csv")
        if stats_csv.exists():
            with open(stats_csv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["instance"] == inst:
                        e3.setdefault(row["policy"], []).append(float(row["IGD"]))
        tau = kendall([np.mean(e3[p]) for p in pols], [IG[p].mean() for p in pols]) if all(p in e3 for p in pols) else np.nan

        L += [f"## {inst}", "",
              f"Solutions: {len(X)}. Containment (all three objectives inside the predicted interval): "
              f"**{100 * inside.mean():.2f}%** of solution-scenario pairs. Mean position inside the interval "
              f"(makespan / energy / cost): {' / '.join(f'{x:.2f}' for x in pos.mean(axis=(0, 1)))}. "
              f"Crisp 2.5-97.5% spread as a share of the interval width: "
              f"{' / '.join(f'{x:.2f}' for x in np.nanmean(share, axis=0))}. "
              f"Kendall tau, E3 interval IGD order vs Monte Carlo IGD order: {tau:.2f}.", "",
              "| Policy | MC HV mean ± sd | MC IGD mean ± sd | final better (HV) | final better (IGD) | E3 IGD |",
              "|---|---|---|---|---|---|"]
        for p in pols:
            hb = "–" if p == "final" else f"{100 * np.mean(HV['final'] > HV[p]):.0f}%"
            ib = "–" if p == "final" else f"{100 * np.mean(IG['final'] < IG[p]):.0f}%"
            L.append(f"| {LABEL[p]} | {HV[p].mean():.4f} ± {HV[p].std():.4f} | {IG[p].mean():.4f} ± {IG[p].std():.4f} "
                     f"| {hb} | {ib} | {np.mean(e3[p]) if p in e3 else float('nan'):.4f} |")
            rows.append([inst, p, f"{HV[p].mean():.6g}", f"{HV[p].std():.6g}", f"{IG[p].mean():.6g}",
                         f"{IG[p].std():.6g}", hb, ib])
        L.append("")
        print(f"  containment {100 * inside.mean():.2f}%, tau {tau:.2f}", flush=True)

    out = ROOT / "results" / "stats"
    (out / f"{args.name}_montecarlo.md").write_text("\n".join(L), encoding="utf-8")
    with open(out / f"{args.name}_montecarlo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "policy", "HV_mean", "HV_sd", "IGD_mean", "IGD_sd", "final_better_HV", "final_better_IGD"])
        w.writerows(rows)
    print("wrote", out / f"{args.name}_montecarlo.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
