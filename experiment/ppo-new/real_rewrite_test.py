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

# Ensure qiskit & IPython stubs if not installed
if 'qiskit' not in sys.modules:
    try:
        import qiskit
    except ImportError:
        sys.modules['qiskit'] = types.ModuleType('qiskit')

if 'IPython' not in sys.modules:
    try:
        import IPython
    except ImportError:
        ip_mod = types.ModuleType('IPython')
        ip_mod.embed = lambda *a, **kw: None
        sys.modules['IPython'] = ip_mod

# 2. Imports from Quartz and Active Repo Architecture
from quartz.core import QuartzContext, PyQASMParser, PyGraph, PyXfer, PyNode
from model.actor_critic import ActorCritic
from enum import Enum

class CostType(Enum):
    gate_count = 0
    cx_count = 1
    cx_gate = 2
    depth = 3
    depth_gc = 4
    depth_2_gc = 5

def get_cost(graph: PyGraph, tp: CostType) -> int:
    if tp is CostType.gate_count:
        return graph.gate_count
    elif tp is CostType.cx_count:
        return graph.cx_count
    elif tp is CostType.cx_gate:
        return 2 * graph.cx_count + graph.gate_count
    elif tp is CostType.depth:
        return graph.depth
    elif tp is CostType.depth_gc:
        return graph.depth + graph.gate_count
    elif tp is CostType.depth_2_gc:
        return 2 * graph.depth + graph.gate_count
    else:
        raise NotImplementedError(f'Unexpected CostType {tp}')

def main():
    print("=" * 65)
    print("REAL QUARTZ CIRCUIT REWRITE VERIFICATION")
    print("=" * 65)

    # 3. Circuit and ECC Configuration
    circuit_rel_path = os.path.join('circuit', 'example-circuits', 't_tdg_t.qasm')
    circuit_path = os.path.join(repo_root, circuit_rel_path)
    ecc_path = os.path.join(repo_root, 'experiment', 'ecc_set', 't_tdg_ecc.json')

    print(f"\n[1] Target Circuit: {circuit_rel_path}")
    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"Circuit file missing: {circuit_path}")
    if not os.path.exists(ecc_path):
        raise FileNotFoundError(f"ECC set missing: {ecc_path}")

    with open(circuit_path, 'r') as f:
        qasm_str = f.read()
    print("QASM Content:")
    print(qasm_str.strip())

    # 4. Initialize Quartz Context
    print("\n[2] Initializing Quartz Context & Loading Circuit")
    ctx = QuartzContext(
        gate_set=['h', 'x', 'cx', 't', 'tdg'],
        filename=ecc_path,
        no_increase=False,
        include_nop=True,
    )
    parser = PyQASMParser(context=ctx)
    dag = parser.load_qasm(filename=circuit_path)
    graph_before: PyGraph = PyGraph(context=ctx, dag=dag)

    # 5. Measure Cost Before Rewrite
    cost_before = get_cost(graph_before, CostType.gate_count)
    gate_count_before = graph_before.gate_count
    nodes_before = graph_before.all_nodes()
    edges_before = graph_before.all_edges()

    print(f"Graph BEFORE rewrite:")
    print(f"  - Cost (gate_count): {cost_before}")
    print(f"  - Gate count: {gate_count_before}")
    print(f"  - PyGraph nodes: {len(nodes_before)}")
    print(f"  - PyGraph edges: {len(edges_before)}")

    # Convert Before Graph to DGL
    dgl_before: dgl.DGLGraph = graph_before.to_dgl_graph()
    print(f"  - DGL before nodes: {dgl_before.num_nodes()}, edges: {dgl_before.num_edges()}")

    # 6. Discover Available Valid Rewrites
    print("\n[3] Discovering Available Valid Rewrite Rules")
    valid_candidates = []
    total_xfers_queried = 0

    for node_idx, node in enumerate(nodes_before):
        available_xfer_ids = graph_before.available_xfers(context=ctx, node=node)
        total_xfers_queried += len(available_xfer_ids)
        for xfer_id in available_xfer_ids:
            xfer: PyXfer = ctx.get_xfer_from_id(id=xfer_id)
            if xfer.is_nop:
                continue
            # Probe apply_xfer
            candidate_graph = graph_before.apply_xfer(xfer=xfer, node=node)
            if candidate_graph is not None:
                valid_candidates.append({
                    'node_idx': node_idx,
                    'node': node,
                    'xfer_id': xfer_id,
                    'xfer': xfer,
                    'graph': candidate_graph,
                })

    print(f"Discovery Summary:")
    print(f"  - Total nodes inspected: {len(nodes_before)}")
    print(f"  - Total applicable xfer instances considered: {total_xfers_queried}")
    print(f"  - Total valid applicable rewrites found: {len(valid_candidates)}")

    if len(valid_candidates) == 0:
        raise RuntimeError("No valid rewrite rules found for the circuit in this context.")

    # 7. Deterministically Select ONE Valid Rewrite
    # We find candidates:
    # 1) A rewrite that preserves graph connectivity for DGL conversion (e.g., commutation / replacement)
    # 2) A cancellation rewrite that strictly reduces gate count (e.g., T * Tdg -> identity)
    
    selected = None
    cancellation_candidate = None

    for c in valid_candidates:
        if len(c['graph'].all_edges()) > 0 and selected is None:
            selected = c
        if c['graph'].gate_count < graph_before.gate_count and cancellation_candidate is None:
            cancellation_candidate = c

    if selected is None:
        selected = valid_candidates[0]

    selected_node_idx = selected['node_idx']
    selected_node = selected['node']
    selected_xfer_id = selected['xfer_id']
    selected_xfer: PyXfer = selected['xfer']

    print(f"\n[4] Deterministically Selected Rewrite (Primary Test):")
    print(f"  - Target node index: {selected_node_idx} (Gate type: {selected_node.gate.tp})")
    print(f"  - Transfer ID (xfer_id): {selected_xfer_id}")
    print(f"  - Rule source gates ({selected_xfer.src_gate_count}): '{selected_xfer.src_str.strip()}'")
    print(f"  - Rule destination gates ({selected_xfer.dst_gate_count}): '{selected_xfer.dst_str.strip()}'")

    # 8. Apply Rewrite via Quartz's Real Rewrite API
    print("\n[5] Applying Quartz Rewrite to Circuit")
    graph_after: PyGraph = graph_before.apply_xfer(xfer=selected_xfer, node=selected_node)
    assert graph_after is not None, "Quartz rewrite failed to produce a valid graph."

    # Also test the tracking API used by actor.py
    graph_tracked, tracked_nodes = graph_before.apply_xfer_with_local_state_tracking(
        xfer=selected_xfer,
        node=selected_node,
    )
    print(f"Rewrite applied successfully via Quartz engine.")
    print(f"  - Local state tracked nodes: {tracked_nodes}")

    # 9. Measure Cost After Rewrite
    cost_after = get_cost(graph_after, CostType.gate_count)
    gate_count_after = graph_after.gate_count
    nodes_after = graph_after.all_nodes()
    edges_after = graph_after.all_edges()
    cost_diff = cost_after - cost_before

    print(f"\nGraph AFTER rewrite:")
    print(f"  - Cost (gate_count): {cost_after}")
    print(f"  - Gate count: {gate_count_after}")
    print(f"  - PyGraph nodes: {len(nodes_after)}")
    print(f"  - PyGraph edges: {len(edges_after)}")
    print(f"  - Cost change (after - before): {cost_diff}")
    improved = cost_after < cost_before
    print(f"  - Did cost improve? {improved} ({cost_before} -> {cost_after})")

    # 10. Convert After Graph to DGL
    print("\n[6] Converting Modified PyGraph -> DGL Graph")
    dgl_after: dgl.DGLGraph = graph_after.to_dgl_graph()
    print(f"DGL Graph after rewrite:")
    print(f"  - DGL after nodes: {dgl_after.num_nodes()}")
    print(f"  - DGL after edges: {dgl_after.num_edges()}")
    print(f"  - ndata['gate_type']: {dgl_after.ndata['gate_type'].tolist()}")

    # 11. Secondary Check: QGNN Forward Pass on After Graph
    print("\n[7] Secondary Check: QGNN Forward Pass on AFTER Graph")
    model = ActorCritic(
        gnn_type='QGNN',
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
    with torch.no_grad():
        batched_after = dgl.batch([dgl_after])
        embs_after = model(batched_after, 'gnn')
        actor_after = model(embs_after, 'actor')
        critic_after = model(embs_after, 'critic')
    print(f"  - QGNN embeddings shape on after graph: {tuple(embs_after.shape)}")
    print(f"  - Actor output shape on after graph:    {tuple(actor_after.shape)}")
    print(f"  - Critic output shape on after graph:   {tuple(critic_after.shape)}")
    print("  -> Secondary QGNN forward pass completed successfully!")

    # 12. Also Report Cancellation / Reduction Rewrite
    if cancellation_candidate is not None:
        print("\n[8] Gate Cancellation / Reduction Rewrite Verification:")
        canc_xfer = cancellation_candidate['xfer']
        canc_graph = cancellation_candidate['graph']
        canc_cost = get_cost(canc_graph, CostType.gate_count)
        print(f"  - Cancellation Xfer ID: {cancellation_candidate['xfer_id']}")
        print(f"  - Rule: '{canc_xfer.src_str.strip()}' -> '{canc_xfer.dst_str.strip()}'")
        print(f"  - Gate count reduction: {cost_before} -> {canc_cost} (diff: {canc_cost - cost_before})")
        print(f"  - Did cost improve? {canc_cost < cost_before}")

    print("\n" + "=" * 65)
    print("REWRITE VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 65)

if __name__ == '__main__':
    main()
