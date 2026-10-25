"""Train one DQN agent (with its GAT encoder) across several workflows.

    python scripts/train.py                                   # default training set
    python scripts/train.py --workflows montage_100 sipht_100 --episodes 10 --generations 300

Each episode is one full GAT-IMOEA-DQN run on a workflow (round-robin over
--workflows); the agent, its replay buffer and update counter carry over, so
the PDF's online learning (buffer >= 250, train every 50 generations) spans
episodes. The agent is saved after every episode.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_tools.loader import load_manifest, load_scenario, load_workflow  # noqa: E402
from gat_imoea_dqn.algorithm import RunConfig, run  # noqa: E402
from gat_imoea_dqn.dqn import DQNAgent, DQNConfig  # noqa: E402

DEFAULT_SET = ["montage_100", "epigenomics_100", "cybershake_100", "inspiral_100", "sipht_100"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflows", nargs="*", default=DEFAULT_SET)
    ap.add_argument("--scenario", default=str(ROOT / "datasets" / "scenarios" / "default.yaml"))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--pop", type=int, default=105)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="continue training this saved agent")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "agent.pt"))
    args = ap.parse_args()

    manifest = load_manifest(ROOT / "datasets")
    missing = [w for w in args.workflows if w not in manifest]
    if missing:
        ap.error(f"unknown workflows {missing}; see datasets/manifest.json")
    sc = load_scenario(args.scenario)
    wfs = {w: load_workflow(ROOT / "datasets" / manifest[w]["path"]) for w in args.workflows}
    agent = DQNAgent.load(args.resume) if args.resume else DQNAgent(DQNConfig(seed=args.seed))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = []

    for ep in range(args.episodes):
        name = args.workflows[ep % len(args.workflows)]
        cfg = RunConfig(pop_size=args.pop, generations=args.generations, seed=args.seed + ep)
        t0 = time.time()
        res = run(wfs[name], sc, cfg, agent=agent, graph_key=name)
        h = res.history
        losses = [x for x in h["loss"] if x is not None]
        entry = {"episode": ep, "workflow": name, "seconds": round(time.time() - t0, 1),
                 "mean_reward": float(np.mean(h["reward"])), "final_pibi": h["pibi"][-1],
                 "front_size": int(len(res.front)), "best_mid": h["best_mid"][-1],
                 "train_rounds": agent.train_rounds, "last_loss": losses[-1] if losses else None,
                 "explore_share": float(np.mean(h["explored"]))}
        log.append(entry)
        print(f"ep {ep:3d} {name:16s} {entry['seconds']:6.1f}s  reward {entry['mean_reward']:+.4f}  "
              f"PIBI {entry['final_pibi']:.4f}  front {entry['front_size']:3d}  "
              f"rounds {agent.train_rounds:3d}  loss {entry['last_loss'] if entry['last_loss'] is None else round(entry['last_loss'], 5)}")
        agent.save(out)
    out.with_suffix(".log.json").write_text(json.dumps(log, indent=2))
    print(f"saved agent to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
