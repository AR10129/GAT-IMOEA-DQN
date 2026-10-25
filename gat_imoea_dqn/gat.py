"""Graph Attention Network structural encoder (PDF step 2), in plain PyTorch.

    graph = GraphTensors.from_workflow(wf)
    enc = GATEncoder()                 # 2 layers, node feats 5, edge feats 2, out 16
    g = enc(graph)                     # (16,) graph embedding g_t

Layer l, head h, receiver i, neighbour j:
    alpha^h_ij = softmax_j( LeakyReLU( a_h^T [W_h h_i || W_h h_j || W_e e_ij] ) )
    h_i^(l)    = ELU( 1/H_a * sum_h sum_j alpha^h_ij W_h h_j )
Pooling:  g = sum_k softmax_k(w_g^T h_k^(2)) h_k^(2)

Choice: N(i) is the undirected neighbourhood (parents and children) plus i
itself, so information flows both up and down the DAG in two layers. The
self-loop uses an all-zero edge feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as Fn


@dataclass
class GraphTensors:
    x: torch.Tensor            # (K, 5) node features
    edge_attr: torch.Tensor    # (E', 2) edge features, both directions + self-loops
    dst: torch.Tensor          # (E',) receiving node i
    src: torch.Tensor          # (E',) neighbour j
    num_nodes: int

    @staticmethod
    def from_workflow(wf, normalize: str = "log") -> "GraphTensors":
        x = wf.gat_node_features(normalize)
        e = wf.gat_edge_features(normalize)
        s, d = wf.edge_index
        K = wf.num_tasks
        loops = np.arange(K)
        src = np.concatenate([s, d, loops])
        dst = np.concatenate([d, s, loops])
        attr = np.concatenate([e, e, np.zeros((K, e.shape[1]), np.float32)])
        return GraphTensors(x=torch.from_numpy(x).float(), edge_attr=torch.from_numpy(attr).float(),
                            dst=torch.from_numpy(dst).long(), src=torch.from_numpy(src).long(), num_nodes=K)


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Softmax of scores (E, H) over entries that share the same index (E,)."""
    H = scores.shape[1]
    idx = index.unsqueeze(1).expand(-1, H)
    mx = torch.full((num_segments, H), -torch.inf, dtype=scores.dtype)
    mx = mx.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - mx[index])
    den = torch.zeros((num_segments, H), dtype=scores.dtype).index_add_(0, index, ex)
    return ex / den[index].clamp_min(1e-16)


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int = 2, slope: float = 0.2):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.We = nn.Linear(edge_dim, heads * out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(heads, 3 * out_dim))
        nn.init.xavier_uniform_(self.a)
        self.slope = slope

    def forward(self, h: torch.Tensor, g: GraphTensors) -> torch.Tensor:
        H, D = self.heads, self.out_dim
        Wh = self.W(h).view(-1, H, D)                         # (K, H, D)
        We = self.We(g.edge_attr).view(-1, H, D)              # (E', H, D)
        a_i, a_j, a_e = self.a[:, :D], self.a[:, D:2 * D], self.a[:, 2 * D:]
        score = ((Wh * a_i).sum(-1)[g.dst] + (Wh * a_j).sum(-1)[g.src] + (We * a_e).sum(-1))
        alpha = segment_softmax(Fn.leaky_relu(score, self.slope), g.dst, g.num_nodes)   # (E', H)
        msg = alpha.unsqueeze(-1) * Wh[g.src]                 # (E', H, D)
        out = torch.zeros((g.num_nodes, H, D), dtype=h.dtype).index_add_(0, g.dst, msg)
        return Fn.elu(out.mean(dim=1))                        # average heads


class GATEncoder(nn.Module):
    def __init__(self, node_dim: int = 5, edge_dim: int = 2, hidden: int = 32, out_dim: int = 16,
                 heads: int = 4):
        super().__init__()
        self.layer1 = GATLayer(node_dim, hidden, heads, edge_dim)
        self.layer2 = GATLayer(hidden, out_dim, heads, edge_dim)
        self.gate = nn.Linear(out_dim, 1, bias=False)
        self.out_dim = out_dim

    def node_embeddings(self, g: GraphTensors) -> torch.Tensor:
        return self.layer2(self.layer1(g.x, g), g)

    def forward(self, g: GraphTensors) -> torch.Tensor:
        h = self.node_embeddings(g)                           # (K, 16)
        w = torch.softmax(self.gate(h).squeeze(-1), dim=0)    # (K,)
        return (w.unsqueeze(-1) * h).sum(0)                   # (16,)
