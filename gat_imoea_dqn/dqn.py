"""DQN action selector (PDF steps 3 and 5).

State s_t = [g_t (16, GAT embedding of the workflow) || 7 population features] (23).
Q(s, .) over the H = 28 Das-Dennis actions. epsilon-greedy with power decay
eps_t = 0.3 ** (t / 100). Transitions go to an experience replay buffer; once
it holds >= 250 transitions, the network is trained every 50 generations with
Double DQN targets r + gamma * Q_target(s', argmax_a' Q(s', a')).

The GAT encoder is part of the Q-network and is trained end to end by the TD
loss. The buffer stores only the 7 population features plus a graph key; the
graph embedding is recomputed from the (fixed) workflow graph at update time,
so one agent can learn across several workflows.

Choices beyond the PDF (DQNConfig): MLP 64-64 with ReLU, Adam lr 1e-3,
gamma 0.9, batch 32, 50 gradient steps per training round, hard target sync
after each round, gradient-norm clip 10, buffer capacity 10 000. The update
period counts generations over the agent's lifetime (so it carries over
between runs); epsilon restarts each run, as eps_t uses the run's generation t.

Learner fixes (off by default; see results/diagnostics for why):
  reward_scaling="running"  TD targets use (r - mean) / sd with the running mean
                            and sd of every reward stored so far; the buffer keeps
                            raw rewards so the scaling stays consistent.
  eps_floor=0.1             epsilon = max(floor, 0.3 ** (t / 100)).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .gat import GATEncoder, GraphTensors
from .state import STATE_POP_DIM


@dataclass(frozen=True)
class DQNConfig:
    n_actions: int = 28
    embed_dim: int = 16
    gat_hidden: int = 32
    gat_heads: int = 4
    hidden: Tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    gamma: float = 0.9
    batch_size: int = 32
    buffer_capacity: int = 10_000
    erb_threshold: int = 250
    update_period: int = 50
    grad_steps: int = 50
    grad_clip: float = 10.0
    eps_base: float = 0.3
    eps_period: float = 100.0
    seed: int = 0
    use_graph: bool = True              # False: ablation, g_t is always zero (GAT unused)
    reward_scaling: str = "none"        # "running": standardize rewards by running mean / sd when training
    eps_floor: float = 0.0              # epsilon never drops below this (keeps exploring)


class QNetwork(nn.Module):
    def __init__(self, cfg: DQNConfig):
        super().__init__()
        self.encoder = GATEncoder(hidden=cfg.gat_hidden, out_dim=cfg.embed_dim, heads=cfg.gat_heads)
        layers, d = [], cfg.embed_dim + STATE_POP_DIM
        for h in cfg.hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, cfg.n_actions))
        self.head = nn.Sequential(*layers)

    def q(self, g: torch.Tensor, pop: torch.Tensor) -> torch.Tensor:
        """g (B, 16), pop (B, 7) -> Q (B, H)."""
        return self.head(torch.cat([g, pop], dim=1))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.graph = np.zeros(capacity, np.int64)
        self.s = np.zeros((capacity, STATE_POP_DIM), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, STATE_POP_DIM), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.size = 0
        self._next = 0

    def add(self, graph_id, s, a, r, s2, done) -> None:
        i = self._next
        self.graph[i], self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = graph_id, s, a, r, s2, done
        self._next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=n)
        return self.graph[idx], self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    def __init__(self, cfg: DQNConfig = DQNConfig()):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.online = QNetwork(cfg)
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        self.graphs: Dict[int, GraphTensors] = {}
        self.graph_ids: Dict[str, int] = {}
        self._embed_cache: Dict[int, torch.Tensor] = {}
        self.lifetime_steps = 0          # transitions stored over the agent's life
        self.train_rounds = 0
        self.last_loss: Optional[float] = None
        # Ablation probe: act with g_t = 0 even though the agent was trained with
        # the graph, to test whether its decisions depend on the embedding.
        self.zero_graph_at_inference = False
        # running reward statistics (Welford), used when reward_scaling == "running"
        self.r_count, self.r_mean, self.r_m2 = 0, 0.0, 0.0

    # ------------------------------------------------------------ graphs --
    def register_graph(self, key: str, wf) -> int:
        if key not in self.graph_ids:
            gid = len(self.graph_ids)
            self.graph_ids[key] = gid
            self.graphs[gid] = GraphTensors.from_workflow(wf)
        return self.graph_ids[key]

    def embedding(self, gid: int) -> torch.Tensor:
        """g_t for acting (no grad, cached until the next training round)."""
        if not self.cfg.use_graph or self.zero_graph_at_inference:
            return torch.zeros(self.cfg.embed_dim)
        if gid not in self._embed_cache:
            with torch.no_grad():
                self._embed_cache[gid] = self.online.encoder(self.graphs[gid])
        return self._embed_cache[gid]

    def state(self, gid: int, pop_feats: np.ndarray) -> np.ndarray:
        """The full 23-dim state s_t, for inspection and logging."""
        return np.concatenate([self.embedding(gid).numpy(), pop_feats]).astype(np.float32)

    # ------------------------------------------------------------ acting --
    def epsilon(self, t: int) -> float:
        return float(max(self.cfg.eps_floor, self.cfg.eps_base ** (t / self.cfg.eps_period)))

    def q_values(self, gid: int, pop_feats: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            g = self.embedding(gid).unsqueeze(0)
            return self.online.q(g, torch.as_tensor(pop_feats, dtype=torch.float32).unsqueeze(0))[0].numpy()

    def act(self, gid: int, pop_feats: np.ndarray, t: int, rng: np.random.Generator,
            greedy: bool = False) -> Tuple[int, bool]:
        """Returns (action index, explored?)."""
        if not greedy and rng.random() <= self.epsilon(t):
            return int(rng.integers(self.cfg.n_actions)), True
        return int(np.argmax(self.q_values(gid, pop_feats))), False

    # ---------------------------------------------------------- learning --
    def store(self, gid, s, a, r, s2, done) -> None:
        self.buffer.add(gid, s, a, r, s2, float(done))
        self.lifetime_steps += 1
        self.r_count += 1
        delta = r - self.r_mean
        self.r_mean += delta / self.r_count
        self.r_m2 += delta * (r - self.r_mean)

    def reward_stats(self) -> Tuple[float, float]:
        sd = (self.r_m2 / (self.r_count - 1)) ** 0.5 if self.r_count > 1 else 1.0
        return self.r_mean, max(sd, 1e-8)

    def _scale(self, r: torch.Tensor) -> torch.Tensor:
        if self.cfg.reward_scaling != "running" or self.r_count < 2:
            return r
        mean, sd = self.reward_stats()
        return (r - mean) / sd

    def maybe_train(self) -> Optional[float]:
        """Train if the buffer is warm and an update period has elapsed."""
        if len(self.buffer) < self.cfg.erb_threshold or self.lifetime_steps % self.cfg.update_period != 0:
            return None
        return self.train_round()

    def _embed_batch(self, net: QNetwork, gids: np.ndarray) -> torch.Tensor:
        if not self.cfg.use_graph:
            return torch.zeros((len(gids), self.cfg.embed_dim))
        uniq, inv = np.unique(gids, return_inverse=True)
        embs = torch.stack([net.encoder(self.graphs[int(g)]) for g in uniq])
        return embs[torch.as_tensor(inv)]

    def train_round(self) -> float:
        c = self.cfg
        losses = []
        for _ in range(c.grad_steps):
            gid, s, a, r, s2, done = self.buffer.sample(c.batch_size, self.rng)
            s, s2 = torch.from_numpy(s), torch.from_numpy(s2)
            a, r, done = torch.from_numpy(a), self._scale(torch.from_numpy(r)), torch.from_numpy(done)
            g = self._embed_batch(self.online, gid)
            q = self.online.q(g, s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                a_star = self.online.q(g.detach(), s2).argmax(dim=1, keepdim=True)
                g_t = self._embed_batch(self.target, gid)
                q_next = self.target.q(g_t, s2).gather(1, a_star).squeeze(1)
                y = r + c.gamma * (1.0 - done) * q_next
            loss = torch.mean((y - q) ** 2)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
            self.opt.step()
            losses.append(loss.item())
        self.target.load_state_dict(self.online.state_dict())
        self._embed_cache.clear()
        self.train_rounds += 1
        self.last_loss = float(np.mean(losses))
        return self.last_loss

    # ------------------------------------------------------- persistence --
    def save(self, path) -> None:
        torch.save({"cfg": asdict(self.cfg), "online": self.online.state_dict(),
                    "target": self.target.state_dict(), "opt": self.opt.state_dict(),
                    "lifetime_steps": self.lifetime_steps, "train_rounds": self.train_rounds,
                    "reward_stats": (self.r_count, self.r_mean, self.r_m2)}, path)

    @classmethod
    def load(cls, path) -> "DQNAgent":
        ck = torch.load(path, weights_only=False)
        cfg = ck["cfg"]
        cfg["hidden"] = tuple(cfg["hidden"])
        agent = cls(DQNConfig(**cfg))
        agent.online.load_state_dict(ck["online"])
        agent.target.load_state_dict(ck["target"])
        agent.opt.load_state_dict(ck["opt"])
        agent.lifetime_steps, agent.train_rounds = ck["lifetime_steps"], ck["train_rounds"]
        agent.r_count, agent.r_mean, agent.r_m2 = ck.get("reward_stats", (0, 0.0, 0.0))
        return agent
