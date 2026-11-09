"""B4: count solution evaluations per run for every method (equal-budget check).

Runs each method for a few generations on one instance with the evaluator instrumented, and
prints evaluations, the expected N * (G + 1), and the ratio.
  python scripts/budget_check.py [--generations 10] [--instance cybershake_100@x1]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment as E  # noqa: E402
from gat_imoea_dqn import evaluator as EV  # noqa: E402

COUNT = [0]
_pop, _one = EV.IntervalEvaluator.evaluate_population, EV.IntervalEvaluator.evaluate


def counted_pop(self, Xs, Os, *a, **k):
    COUNT[0] += len(Xs)
    return _pop(self, Xs, Os, *a, **k)


def counted_one(self, X, O, *a, **k):
    COUNT[0] += 1
    return _one(self, X, O, *a, **k)


EV.IntervalEvaluator.evaluate_population = counted_pop
EV.IntervalEvaluator.evaluate = counted_one


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--pop", type=int, default=105)
    ap.add_argument("--instance", default="cybershake_100@x1")
    ap.add_argument("--policies", nargs="+", default=["final", "imoead_orig", "imoead_awn", "imea_ied", "ip_moea",
                                                     "nsga2_mid", "nsga2_int", "nsga3_mid", "fixed_fav",
                                                     "trained_stable"])
    args = ap.parse_args()
    env = {"scenario": "jin", "energy_scope": "device", "external_io": "device", "makespan": "sum_apps"}
    out = ROOT / "results" / "experiments" / "suite_b"
    expected = args.pop * (args.generations + 1)
    print(f"{args.instance}, N = {args.pop}, G = {args.generations}; expected N(G+1) = {expected}")
    for p in args.policies:
        COUNT[0] = 0
        spec = {"workflow": args.instance, "policy": p, "seed": 0, "generations": args.generations,
                "pop": args.pop, "env": env, "out": str(ROOT / "results" / "_budget_tmp.json")}
        if E.POLICIES[p]["kind"] == "greedy":
            spec["agent"] = str(E.agent_path(out, E.POLICIES[p]["variant"], 0))
        pathlib.Path(spec["out"]).unlink(missing_ok=True)
        E.job(spec)
        pathlib.Path(spec["out"]).unlink(missing_ok=True)
        print(f"  {p:15s} {COUNT[0]:7d}  ratio {COUNT[0] / expected:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
