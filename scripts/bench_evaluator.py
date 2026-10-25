"""Time the evaluator on a merged population (2N = 210) for each family.

    python scripts/bench_evaluator.py [--size 2000] [--python]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_tools.loader import load_scenario, load_workflow  # noqa: E402
from gat_imoea_dqn import IntervalEvaluator  # noqa: E402
from gat_imoea_dqn import evaluator as evmod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=210)
    ap.add_argument("--python", action="store_true", help="disable numba for comparison")
    args = ap.parse_args()
    if args.python:
        evmod._schedule = evmod._schedule_py
    sc = load_scenario(ROOT / "datasets" / "scenarios" / "default.yaml")
    rng = np.random.default_rng(0)
    for fam in ("montage", "epigenomics", "cybershake", "inspiral", "sipht"):
        wf = load_workflow(ROOT / "datasets" / "workflows" / fam / f"{fam}_{args.size}.json")
        ev = IntervalEvaluator(wf, sc)
        Xs = rng.choice(sc.allowed_targets(), size=(args.pop, wf.num_tasks))
        Os = np.stack([wf.random_topological_order(rng) for _ in range(args.pop)])
        ev.evaluate(Xs[0], Os[0])                      # warm-up / JIT compile
        t0 = time.perf_counter()
        F_lo, F_hi = ev.evaluate_population(Xs, Os)
        dt = time.perf_counter() - t0
        print(f"{fam:12s} K={wf.num_tasks:5d} E={wf.num_edges:5d}  {dt * 1000:7.1f} ms / {args.pop} "
              f"({dt / args.pop * 1e3:.2f} ms each)  median F_mid={np.round(np.median((F_lo + F_hi) / 2, 0), 3)}")


if __name__ == "__main__":
    main()
