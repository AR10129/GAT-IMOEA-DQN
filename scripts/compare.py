"""Compare result files from scripts/run.py on one shared scale.

    python scripts/compare.py results/montage_200_*.json

Per-run PIBI values are normalized with each run's own bounds, so they are not
comparable. This pools every front, normalizes all of them with the same
ideal/nadir, and reports PIBI (lower is better), front size and the mean
normalized interval length U1 (lower = less uncertain).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gat_imoea_dqn.selection import das_dennis  # noqa: E402
from gat_imoea_dqn.state import pibi  # noqa: E402


def main(paths) -> int:
    if not paths:
        print(__doc__)
        return 1
    runs = []
    for p in paths:
        d = json.loads(pathlib.Path(p).read_text())
        lo = np.array([s["F_lo"] for s in d["front"]])
        hi = np.array([s["F_hi"] for s in d["front"]])
        runs.append((pathlib.Path(p).name, d, lo, hi))
    ideal = np.min([r[2].min(0) for r in runs], axis=0)
    nadir = np.max([r[3].max(0) for r in runs], axis=0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    W = das_dennis(6, 3)
    print(f"{'file':42s} {'PIBI':>8s} {'front':>6s} {'U1':>8s} {'best makespan':>14s} {'best energy':>12s} {'best cost':>10s}")
    for name, d, lo, hi in sorted(runs, key=lambda r: pibi((r[2] - ideal) / span, (r[3] - ideal) / span, W)):
        nlo, nhi = (lo - ideal) / span, (hi - ideal) / span
        mid = (lo + hi) / 2
        print(f"{name:42s} {pibi(nlo, nhi, W):8.4f} {len(lo):6d} {np.mean((nhi - nlo).sum(1)):8.4f} "
              f"{mid[:, 0].min():14.1f} {mid[:, 1].min():12.1f} {mid[:, 2].min():10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
