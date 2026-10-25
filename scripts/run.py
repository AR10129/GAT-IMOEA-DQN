"""Run GAT-IMOEA-DQN on one workflow and save the final interval front.

    python scripts/run.py --workflow montage_200                          # fresh agent, online learning
    python scripts/run.py --workflow montage_200 --agent checkpoints/agent.pt --greedy
    python scripts/run.py --workflow montage_200 --policy random          # baseline
    python scripts/run.py --workflow montage_200 --policy fixed --action 0.5 0.5 0

Writes results/<workflow>_<policy>_s<seed>.json with the front (objective
intervals, allocation and priority strings) and the per-generation history.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_tools.loader import load_manifest, load_scenario, load_workflow  # noqa: E402
from gat_imoea_dqn.algorithm import RunConfig, run  # noqa: E402
from gat_imoea_dqn.dqn import DQNAgent  # noqa: E402
from gat_imoea_dqn.evaluator import EvaluatorConfig, OBJECTIVES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--scenario", default=str(ROOT / "datasets" / "scenarios" / "default.yaml"))
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--pop", type=int, default=105)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=["dqn", "random", "fixed"], default="dqn")
    ap.add_argument("--action", type=float, nargs=3, default=[1 / 3, 1 / 3, 1 / 3])
    ap.add_argument("--agent", help="saved agent (scripts/train.py); default: fresh agent")
    ap.add_argument("--greedy", action="store_true", help="dqn: no exploration")
    ap.add_argument("--no-learn", action="store_true", help="dqn: do not update the agent")
    ap.add_argument("--external-io", choices=["pdf", "device"], default="pdf")
    ap.add_argument("--edge-queue", action="store_true")
    ap.add_argument("--out", help="output JSON path")
    args = ap.parse_args()

    manifest = load_manifest(ROOT / "datasets")
    if args.workflow not in manifest:
        ap.error(f"unknown workflow {args.workflow!r}; see datasets/manifest.json")
    wf = load_workflow(ROOT / "datasets" / manifest[args.workflow]["path"])
    sc = load_scenario(args.scenario)
    agent = DQNAgent.load(args.agent) if (args.agent and args.policy == "dqn") else None
    cfg = RunConfig(pop_size=args.pop, generations=args.generations, seed=args.seed, policy=args.policy,
                    fixed_action=tuple(args.action), greedy=args.greedy, learn=not args.no_learn,
                    evaluator=EvaluatorConfig(external_io=args.external_io, edge_queue=args.edge_queue))

    def progress(t, rec):
        if (t + 1) % max(1, args.generations // 10) == 0:
            print(f"gen {t + 1:4d}  PIBI {rec['pibi']:.4f}  reward {rec['reward']:+.4f}  "
                  f"best mid {np.round(rec['best_mid'], 3).tolist()}")

    res = run(wf, sc, cfg, agent=agent, graph_key=args.workflow, callback=progress)
    lo, hi = res.front_objectives()
    out = pathlib.Path(args.out or ROOT / "results" / f"{args.workflow}_{args.policy}_s{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "workflow": args.workflow, "scenario": sc.raw.get("name"), "config": {
            "policy": args.policy, "generations": args.generations, "pop_size": args.pop, "seed": args.seed,
            "agent": args.agent, "greedy": args.greedy, "external_io": args.external_io,
            "edge_queue": args.edge_queue},
        "objectives": list(OBJECTIVES), "seconds": round(res.seconds, 2),
        "front": [{"F_lo": lo[i].tolist(), "F_hi": hi[i].tolist(),
                   "X": res.population.X[res.front[i]].tolist(), "O": res.population.O[res.front[i]].tolist()}
                  for i in range(len(res.front))],
        "history": res.history,
    }))
    print(f"{len(res.front)} solutions on the final interval front, {res.seconds:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
