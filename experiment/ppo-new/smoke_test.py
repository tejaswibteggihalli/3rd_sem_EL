import sys
import torch
import dgl

sys.path.insert(0, '.')
from model.actor_critic import ActorCritic


def toy_circuit(gate_types):
    # chain 0->1->2->3 plus reversed copies, mimicking to_dgl_graph's layout
    src = [0, 1, 2, 1, 2, 3]
    dst = [1, 2, 3, 0, 1, 2]
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=4)
    g.edata['src_idx'] = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.int32)
    g.edata['dst_idx'] = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.int32)
    g.edata['reversed'] = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    g.ndata['gate_type'] = torch.tensor(gate_types, dtype=torch.int32)
    return g


batch = dgl.batch([toy_circuit([0, 1, 2, 1]), toy_circuit([3, 3, 0, 2])])
print('batched nodes:', batch.num_nodes(), 'per-graph:', batch.batch_num_nodes().tolist())

for gnn_type in ['QGNN', 'QGIN', 'QGAT', 'QTransformer']:
    model = ActorCritic(gnn_type=gnn_type, gnn_num_layers=4, gin_graph_pooling_type='none')
    model.eval()
    with torch.no_grad():
        emb = model(batch, 'gnn')
        logits = model(emb[[0, 5]], 'actor')
        values = model(emb, 'critic')
    print(gnn_type, '-> embeddings', tuple(emb.shape),
          '| actor logits', tuple(logits.shape),
          '| critic values', tuple(values.shape))
    print('  params:', sum(p.numel() for p in model.gnn.parameters()))