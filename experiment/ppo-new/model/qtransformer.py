import math

import dgl
import torch
import torch.nn as nn


def _sinusoidal(length: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(length, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : dim // 2]
    return pe


class QTransformer(nn.Module):
    """Plain-sequence encoder. Uses ONLY ndata['gate_type'] and per-graph node
    counts; edges and adjacency are never read. Node order within each graph is
    taken as the sequence order (assumed to be circuit order; unverified)."""

    def __init__(self, num_layers, num_gate_types, embed_dim, d_model, out_dim,
                 num_heads, dim_feedforward=None, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0 and d_model % 2 == 0
        self.embedding = nn.Embedding(num_gate_types, embed_dim)
        self.in_proj = nn.Linear(embed_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward or 2 * d_model,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        counts = g.batch_num_nodes().tolist()
        x = self.in_proj(self.embedding(g.ndata['gate_type'].long()))
        padded = nn.utils.rnn.pad_sequence(torch.split(x, counts), batch_first=True)
        length = padded.shape[1]
        padded = padded + _sinusoidal(length, padded.shape[2], padded.device).unsqueeze(0)
        lens = torch.tensor(counts, device=padded.device)
        pad_mask = torch.arange(length, device=padded.device).unsqueeze(0) >= lens.unsqueeze(1)
        out = self.out_proj(self.encoder(padded, src_key_padding_mask=pad_mask))
        return out[~pad_mask]  # (num_nodes, out_dim), same order as batched nodes
