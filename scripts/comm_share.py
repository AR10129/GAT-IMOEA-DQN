"""How much does inter-task communication move the objectives?

    python scripts/comm_share.py --scenario jin --multipliers 1 5 20

For each workflow, random solutions and solutions evolved for --gens
generations are evaluated twice: with the real inter-task data, and with all
inter-task data set to zero. The median relative increase caused by
communication is reported per objective, along with typical magnitudes.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def one(spec):
    from dataset_tools.loader import load_manifest, load_scenario, load_workflow
    from dataset_tools.rescale import rescale_workflow
    from gat_imoea_dqn import DagIndex, EvaluatorConfig, IntervalEvaluator, init_population
    from gat_imoea_dqn.algorithm import RunConfig, run

    m = load_manifest(ROOT / "datasets")
    wf = load_workflow(ROOT / "datasets" / m[spec["wf"]]["path"])
    if spec["rescale"]:
        wf = rescale_workflow(wf, data_multiplier=spec["mult"])
    sc = load_scenario(ROOT / "datasets" / "scenarios" / f"{spec['scenario']}.yaml")
    ecfg = EvaluatorConfig(energy_scope=spec["energy"])
    ev = IntervalEvaluator(wf, sc, ecfg)
    wf0 = copy.deepcopy(wf)
    wf0.D_lo[:] = 0
    wf0.D_hi[:] = 0
    ev0 = IntervalEvaluator(wf0, sc, ecfg)
    pop = init_population(wf, sc, 105, np.random.default_rng(0), DagIndex(wf))
    res = run(wf, sc, RunConfig(generations=spec["gens"], seed=0, policy="fixed", fixed_action=(5 / 6, 1 / 6, 0),
                                evaluator=ecfg))
    rows = []
    for tag, P in (("random", pop), ("evolved", res.population)):
        F, F0 = ev.evaluate_population(P.X, P.O), ev0.evaluate_population(P.X, P.O)
        mid, mid0 = (F[0] + F[1]) / 2, (F0[0] + F0[1]) / 2
        rel = (mid - mid0) / np.maximum(mid0, 1e-12)
        inc = np.median(rel, axis=0)
        fast = int(np.argmin(mid[:, 0]))              # the fast end of the population
        rows.append((spec["wf"], spec["mult"], tag, inc, np.median(mid, axis=0), float((P.X != 0).mean()),
                     float(rel[fast, 0]), float(mid[fast, 0])))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="jin")
    ap.add_argument("--workflows", nargs="*", default=["montage_200", "epigenomics_200", "cybershake_200",
                                                        "inspiral_200", "sipht_200"])
    ap.add_argument("--multipliers", nargs="*", type=float, default=[1, 5, 20])
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--energy", choices=["all", "device"], default="device")
    ap.add_argument("--no-rescale", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    specs = [{"wf": w, "mult": x, "scenario": a.scenario, "gens": a.gens, "energy": a.energy,
              "rescale": not a.no_rescale} for x in a.multipliers for w in a.workflows]
    with mp.Pool(min(a.workers, len(specs))) as p:
        out = [r for rows in p.map(one, specs) for r in rows]
    print(f"{'workflow':16s} {'data x':>6s} {'set':8s} {'makespan +':>10s} {'energy +':>9s} {'cost +':>7s}"
          f" {'median makespan s':>18s} {'energy J':>9s} {'cost $':>7s} {'offloaded':>9s}"
          f" {'fastest: makespan +':>20s} {'fastest s':>10s}")
    for wfn, x, tag, inc, med, off, fast_inc, fast_mk in out:
        print(f"{wfn:16s} {x:6.0f} {tag:8s} {inc[0]:10.1%} {inc[1]:9.1%} {inc[2]:7.1%}"
              f" {med[0]:18.1f} {med[1]:9.2f} {med[2]:7.3f} {off:9.0%} {fast_inc:20.1%} {fast_mk:10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
