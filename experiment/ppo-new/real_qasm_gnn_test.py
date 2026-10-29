import os
import sys
import types
import torch
import dgl

# 1. Path & DLL Search Setup
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'python'))
sys.path.insert(0, os.path.join(repo_root, 'experiment', 'ppo-new'))

if hasattr(os, 'add_dll_directory'):
    build_win = os.path.join(repo_root, 'build_win')
    py_quartz = os.path.join(repo_root, 'python', 'quartz')
    if os.path.isdir(build_win):
        os.add_dll_directory(build_win)
    if os.path.isdir(py_quartz):
        os.add_dll_directory(py_quartz)

# Ensure qiskit stub if qiskit not installed (needed by top-level quartz/__init__.py)
if 'qiskit' not in sys.modules:
    try:
        import qiskit
    except ImportError:
        sys.modules['qiskit'] = types.ModuleType('qiskit')

# 2. Import Quartz Python binding
from quartz.core import QuartzContext, PyQASMParser, PyGraph
from model.actor_critic import ActorCritic

def main():
    print("=" * 60)
    print("REAL QASM -> QUARTZ -> DGL -> QGNN/QGIN VERIFICATION")
    print("=" * 60)

    # 3. Target Circuit and ECC Set
    qasm_rel_path = os.path.join('circuit', 'example-circuits', 't_tdg_t.qasm')
    qasm_path = os.path.join(repo_root, qasm_rel_path)
    ecc_path = os.path.join(repo_root, 'experiment', 'ecc_set', 't_tdg_ecc.json')

    print(f"\n[1] Loading Circuit: {qasm_rel_path}")
    if not os.path.exists(qasm_path):
        raise FileNotFoundError(f"QASM circuit not found at: {qasm_path}")
    if not os.path.exists(ecc_path):
        raise FileNotFoundError(f"ECC file not found at: {ecc_path}")

    with open(qasm_path, 'r') as f:
        qasm_str = f.read()
    print("QASM Content:")
    print(qasm_str.strip())

    # 4. Quartz Context & Parsing
    print("\n[2] Initializing QuartzContext & Parsing QASM -> PyGraph")
    ctx = QuartzContext(
        gate_set=['h', 'x', 'cx', 't', 'tdg'],
        filename=ecc_path,
        no_increase=False,
        include_nop=True,
    )
    parser = PyQASMParser(context=ctx)
    dag = parser.load_qasm(filename=qasm_path)
    pygraph = PyGraph(context=ctx, dag=dag)

    num_gates = pygraph.gate_count
    nodes_list = pygraph.all_nodes()
    edges_list = pygraph.all_edges()

    print(f"Quartz PyGraph created successfully:")
    print(f"  - Gate count: {num_gates}")
    print(f"  - Total nodes: {len(nodes_list)}")
    print(f"  - Total edges: {len(edges_list)}")

    # 5. Convert PyGraph -> DGL Graph
    print("\n[3] Converting PyGraph -> DGL Graph")
    g: dgl.DGLGraph = pygraph.to_dgl_graph()

    print("DGL Graph created successfully:")
    print(f"  - Graph type: {type(g)}")
    print(f"  - Number of nodes: {g.num_nodes()}")
    print(f"  - Number of edges: {g.num_edges()}")
    print(f"  - Node feature names: {list(g.ndata.keys())}")
    for k, v in g.ndata.items():
        print(f"      * ndata['{k}']: shape={tuple(v.shape)}, dtype={v.dtype}, values={v.tolist()}")
    print(f"  - Edge feature names: {list(g.edata.keys())}")
    for k, v in g.edata.items():
        print(f"      * edata['{k}']: shape={tuple(v.shape)}, dtype={v.dtype}, values={v.tolist()}")

    # 6. Pass Real DGL Graph through Model Architecture (QGNN & QGIN)
    print("\n[4] Passing Real DGL Graph into Existing ActorCritic Models")

    # Wrap in DGL batch (single graph batch, standard for PPO inference / episode evaluation)
    batched_g = dgl.batch([g])

    for gnn_type in ['QGNN', 'QGIN']:
        print(f"\n--- Testing Architecture: {gnn_type} ---")
        model = ActorCritic(
            gnn_type=gnn_type,
            num_gate_types=40,
            gate_type_embed_dim=16,
            gnn_num_layers=4,
            gnn_hidden_dim=32,
            gnn_output_dim=32,
            gin_graph_pooling_type='none',
            actor_hidden_size=32,
            critic_hidden_size=32,
            action_dim=32,
            device='cpu',
        )
        model.eval()

        # Compute parameter count
        gnn_params = sum(p.numel() for p in model.gnn.parameters())
        actor_params = sum(p.numel() for p in model.actor.parameters())
        critic_params = sum(p.numel() for p in model.critic.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameter Counts:")
        print(f"  - GNN ({gnn_type}): {gnn_params}")
        print(f"  - Actor head: {actor_params}")
        print(f"  - Critic head: {critic_params}")
        print(f"  - Total parameters: {total_params}")

        with torch.no_grad():
            # 1. Forward pass through GNN to produce node embeddings
            node_embs = model(batched_g, 'gnn')
            
            # 2. Forward pass through Actor head
            actor_out = model(node_embs, 'actor')
            
            # 3. Forward pass through Critic head
            critic_out = model(node_embs, 'critic')

        print(f"Forward Pass Output Shapes:")
        print(f"  - Node embeddings shape: {tuple(node_embs.shape)} (Expected: [num_nodes, gnn_output_dim])")
        print(f"  - Actor output shape:    {tuple(actor_out.shape)} (Expected: [num_nodes, action_dim])")
        print(f"  - Critic output shape:   {tuple(critic_out.shape)} (Expected: [num_nodes, 1])")

        assert node_embs.shape == (g.num_nodes(), 32), f"Unexpected embedding shape: {node_embs.shape}"
        assert actor_out.shape == (g.num_nodes(), 32), f"Unexpected actor output shape: {actor_out.shape}"
        assert critic_out.shape == (g.num_nodes(), 1), f"Unexpected critic output shape: {critic_out.shape}"
        print(f"  -> {gnn_type} forward pass fully verified!")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED: Real QASM -> PyGraph -> DGL -> GNN -> ActorCritic")
    print("=" * 60)

if __name__ == '__main__':
    main()
