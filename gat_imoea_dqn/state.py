"""Population-dynamics features (PDF step 2) and the PIBI reward (step 5).

    tracker = PopulationTracker(g_max=300)
    tracker.observe(F_lo, F_hi)                  # every evaluated solution widens the bounds
    p = tracker.features(X, F_lo, F_hi, t)       # (7,) [c1, c2, d1, d2, u1, u2, alpha]
    r = tracker.reward(prev_F, new_F)            # relative PIBI improvement

The PDF names the seven features but not their formulas; definitions used here:
  Objective space is normalized with running bounds: ideal = min lower bound and
  nadir = max upper bound over every solution evaluated so far in the run.
  Values at t-1 are re-normalized with the current bounds before comparing.
  c1  distance between the population centroids (interval midpoints) of t-1 and t
  c2  mean distance from each current cluster centroid to the nearest t-1 cluster centroid
  d1  number of decision-space clusters / N; a cluster is a connected component of
      "allocation strings differ in at most cluster_threshold of their genes"
  d2  relative change of the mean interval crowding distance (ICD)
  u1  relative change of the population's summed interval length sum U1
  u2  relative change of the population's summed hyper-box volume U2
  alpha = t / G_max
  "Relative change" is (now - before) / (|now| + |before|), bounded in [-1, 1].
  All change features are 0 at t = 0.

PIBI (penalty-based interval boundary intersection), lower is better:
  for each reference direction w (Das-Dennis, q = 6) and each solution, the PBI
  value d1 + theta * d2 (MOEA/D) of its normalized lower and upper objective
  vectors, averaged over the two bounds; PIBI = mean over w of the best solution.
  Reward r_t = (PIBI_t - PIBI_{t+1}) / PIBI_t, clipped to [-1, 1]. The PDF writes
  (PIBI_{t+1} - PIBI_t) / PIBI_t; the sign is flipped so that an improvement
  (smaller PIBI) earns a positive reward, matching "relative improvement".

Normalization (normalization=...):
  "running" (default, as above): ideal and nadir follow every solution evaluated so
      far, so the same population can score differently later in the run.
  "fixed": ideal = 0 (every objective here is >= 0) and nadir = max upper bound of
      the FIRST population observed (the initial population), never updated. The
      reward and features then mean the same thing at every generation of a run.

Alternative reward (reward="hv", not in the PDF): relative improvement of
HV = (HV(midpoints) + HV(upper bounds)) / 2 on normalized objectives, with
reference point 1.1, r_t = (HV_{t+1} - HV_t) / HV_t clipped to [-1, 1].
Averaging in the upper-bound (worst-case) hypervolume keeps uncertainty in
the signal.

reward="hvjin": the same relative improvement, of HV_jin = (HV(lower bounds) +
HV(upper bounds)) / 2, the evaluation metric itself (Jin et al.'s interval HV).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .intervals import pareto_dominance_matrix
from .metrics import hypervolume_3d
from .selection import das_dennis, interval_crowding

STATE_POP_DIM = 7
REWARDS = ("pibi", "hv", "hvjin")
HV_REF = 1.1


def _hv_nondominated(F: np.ndarray) -> float:
    keep = ~pareto_dominance_matrix(F).any(axis=0)       # HV only depends on the non-dominated points
    return hypervolume_3d(F[keep], [HV_REF] * F.shape[1])


def _rel_change(now: float, before: float) -> float:
    den = abs(now) + abs(before)
    return float((now - before) / den) if den > 1e-12 else 0.0


def decision_clusters(X: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster labels: connected components of Hamming(X_a, X_b) / K <= threshold."""
    N = len(X)
    close = np.zeros((N, N), dtype=bool)
    for i in range(N):                                 # row-wise keeps memory at N*K
        close[i] = (X != X[i]).mean(axis=1) <= threshold
    labels = np.arange(N)
    while True:                                        # min-label propagation
        new = np.where(close, labels[None, :], N).min(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def pibi(F_lo_n: np.ndarray, F_hi_n: np.ndarray, weights: np.ndarray, theta: float = 5.0) -> float:
    """PIBI of a population whose objectives are already normalized (ideal at 0)."""
    W = weights / np.linalg.norm(weights, axis=1, keepdims=True)          # (H, M)
    vals = []
    for F in (F_lo_n, F_hi_n):
        d1 = F @ W.T                                                       # (P, H)
        d2 = np.linalg.norm(F[:, None, :] - d1[:, :, None] * W[None, :, :], axis=2)
        vals.append(d1 + theta * d2)
    per = (vals[0] + vals[1]) / 2.0
    return float(per.min(axis=0).mean())


class PopulationTracker:
    def __init__(self, g_max: int, cluster_threshold: float = 0.1, theta: float = 5.0,
                 k_neighbors: Optional[int] = None, reward: str = "pibi", normalization: str = "running"):
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {reward!r}")
        if normalization not in ("running", "fixed"):
            raise ValueError(f"normalization must be 'running' or 'fixed', got {normalization!r}")
        self.reward_kind = reward
        self.normalization = normalization
        self.g_max = max(int(g_max), 1)
        self.cluster_threshold = cluster_threshold
        self.theta = theta
        self.k = k_neighbors
        self.weights = das_dennis(6, 3)
        self.ideal: Optional[np.ndarray] = None
        self.nadir: Optional[np.ndarray] = None
        self._prev: Optional[dict] = None

    # ------------------------------------------------------------ bounds --
    def observe(self, F_lo: np.ndarray, F_hi: np.ndarray) -> None:
        lo, hi = np.asarray(F_lo).min(axis=0), np.asarray(F_hi).max(axis=0)
        if self.normalization == "fixed":
            if self.ideal is None:                       # set once, from the first population
                self.ideal = np.zeros_like(hi, dtype=float)
                self.nadir = hi.astype(float)
            return
        self.ideal = lo if self.ideal is None else np.minimum(self.ideal, lo)
        self.nadir = hi if self.nadir is None else np.maximum(self.nadir, hi)

    def normalize(self, F_lo, F_hi) -> Tuple[np.ndarray, np.ndarray]:
        span = np.where(self.nadir - self.ideal > 0, self.nadir - self.ideal, 1.0)
        return (np.asarray(F_lo) - self.ideal) / span, (np.asarray(F_hi) - self.ideal) / span

    # ---------------------------------------------------------- features --
    def _summary(self, X, F_lo, F_hi, labels: Optional[np.ndarray] = None) -> dict:
        lo, hi = self.normalize(F_lo, F_hi)
        mid = (lo + hi) / 2.0
        if labels is None:                               # decision-space clusters do not depend on bounds
            labels = decision_clusters(X, self.cluster_threshold)
        cents = np.stack([mid[labels == c].mean(axis=0) for c in range(labels.max() + 1)])
        w = hi - lo
        return {"labels": labels, "centroid": mid.mean(axis=0), "clusters": cents, "n_clusters": len(cents),
                "icd": float(np.mean(interval_crowding(lo, hi, self.k))) if len(lo) > 1 else 0.0,
                "u1": float(w.sum(axis=1).sum()), "u2": float(w.prod(axis=1).sum())}

    def features(self, X: np.ndarray, F_lo: np.ndarray, F_hi: np.ndarray, t: int) -> np.ndarray:
        """7 macro features of the population at generation t (vs. the last call)."""
        self.observe(F_lo, F_hi)
        now = self._summary(X, F_lo, F_hi)
        alpha = min(t / self.g_max, 1.0)
        if self._prev is None:
            feats = [0.0, 0.0, now["n_clusters"] / len(X), 0.0, 0.0, 0.0, alpha]
        else:
            before = self._summary(*self._prev["raw"], labels=self._prev["labels"])   # current bounds
            c1 = float(np.linalg.norm(now["centroid"] - before["centroid"]))
            dist = np.linalg.norm(now["clusters"][:, None, :] - before["clusters"][None, :, :], axis=2)
            c2 = float(dist.min(axis=1).mean())
            feats = [c1, c2, now["n_clusters"] / len(X), _rel_change(now["icd"], before["icd"]),
                     _rel_change(now["u1"], before["u1"]), _rel_change(now["u2"], before["u2"]), alpha]
        self._prev = {"raw": (X.copy(), np.asarray(F_lo).copy(), np.asarray(F_hi).copy()), "labels": now["labels"]}
        return np.asarray(feats, dtype=np.float32)

    # ------------------------------------------------------------ reward --
    def pibi(self, F_lo, F_hi) -> float:
        return pibi(*self.normalize(F_lo, F_hi), self.weights, self.theta)

    def hypervolume(self, F_lo, F_hi) -> float:
        lo, hi = self.normalize(F_lo, F_hi)
        return 0.5 * (_hv_nondominated((lo + hi) / 2.0) + _hv_nondominated(hi))

    def hv_jin(self, F_lo, F_hi) -> float:
        lo, hi = self.normalize(F_lo, F_hi)
        return 0.5 * (_hv_nondominated(lo) + _hv_nondominated(hi))

    def reward(self, F_before: Tuple[np.ndarray, np.ndarray], F_after: Tuple[np.ndarray, np.ndarray]) -> float:
        """Relative improvement (PIBI or HV) from P_t to P_{t+1}, both under the current bounds."""
        self.observe(*F_after)
        if self.reward_kind in ("hv", "hvjin"):
            f = self.hypervolume if self.reward_kind == "hv" else self.hv_jin
            h0, h1 = f(*F_before), f(*F_after)
            if h0 <= 1e-12:
                return 1.0 if h1 > 1e-12 else 0.0
            return float(np.clip((h1 - h0) / h0, -1.0, 1.0))
        p0, p1 = self.pibi(*F_before), self.pibi(*F_after)
        if p0 <= 1e-12:
            return 0.0
        return float(np.clip((p0 - p1) / p0, -1.0, 1.0))
