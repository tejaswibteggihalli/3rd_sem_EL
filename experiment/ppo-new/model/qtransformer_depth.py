import dgl
import torch
import torch.nn as nn

from model.qtransformer import _sinusoidal


class QTransformerDepth(nn.Module):
    """Transformer over gate types; position = longest-path depth computed from
    forward edges (reversed == 0). Uses edges ONLY to compute depth. Independent
    gates at equal depth share a position, so BFS tie-breaking cannot matter."""

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

    @staticmethod
    def _depth(g):
        n = g.num_nodes()
        depth = torch.zeros(n, dtype=torch.long, device=g.device)
        if 'reversed' not in g.edata or g.num_edges() == 0:
            return depth
        fwd = g.edata['reversed'] == 0
        s, d = g.edges()
        s, d = s[fwd].long(), d[fwd].long()
        for _ in range(n):
            new = depth.clone().scatter_reduce(0, d, depth[s] + 1, reduce='amax')
            if torch.equal(new, depth):
                break
            depth = new
        return depth

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        counts = g.batch_num_nodes().tolist()
        x = self.in_proj(self.embedding(g.ndata['gate_type'].long()))
        depth = self._depth(g)
        pe = _sinusoidal(int(depth.max().item()) + 1 if depth.numel() else 1, x.shape[1], x.device)
        x = x + pe[depth]
        padded = nn.utils.rnn.pad_sequence(torch.split(x, counts), batch_first=True)
        length = padded.shape[1]
        lens = torch.tensor(counts, device=padded.device)
        pad_mask = torch.arange(length, device=padded.device).unsqueeze(0) >= lens.unsqueeze(1)
        out = self.out_proj(self.encoder(padded, src_key_padding_mask=pad_mask))
        return out[~pad_mask]
