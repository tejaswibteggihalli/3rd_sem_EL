# Encoder Guide: Quartz / Quarl RL

A working guide to the GNN encoder in this repo: how to get the code running on Windows, where the encoder lives, what interface a replacement encoder must follow, and the theory behind it.

Everything marked **(verified)** was run or read directly from the code. Everything marked **(unverified)** is a reasoned guess that should be checked before relying on it.

---

## 1. What this project does

A quantum circuit is a sequence of gates applied to qubits. Fewer gates generally means a faster, more reliable circuit. This repo optimizes circuits with reinforcement learning (RL):

1. An agent looks at a circuit.
2. It picks a **gate (node)** and a **rewrite rule (xfer)** to apply there.
3. It is rewarded when the circuit gets cheaper (for example, fewer gates).

The circuit is represented as a **graph**: nodes are gates, edges are qubit wires. A neural network called the **encoder** turns every gate into a vector summarizing "what is this gate and what is around it?". The decision networks (actor and critic) work from those vectors.

The repo is a fork of Quarl, which is built on Quartz. The root `README.md` and `INSTALL.md` describe the Quartz C++ runtime only. The RL and GNN part lives under `experiment/`.

---

## 2. Code map

All paths are relative to the repo root.

| Path | What it is |
|---|---|
| `experiment/ppo-new/model/qgnn.py` | `QGNN`, the baseline encoder used in training (verified) |
| `experiment/ppo-new/model/qgin.py` | `QGIN`, a GIN-based alternative (verified) |
| `experiment/ppo-new/model/qgat.py` | GAT-based alternative (not read in detail) |
| `experiment/ppo-new/model/actor_critic.py` | `ActorCritic`: wraps an encoder with actor and critic MLPs. New encoders are registered here (verified) |
| `experiment/ppo-new/config/base_config.py` | Default hyperparameters (verified: `gnn_type='QGNN'`, `gnn_num_layers=4`, `subgraph_opt=True`) |
| `experiment/ppo-new/ppo.py` | PPO training loop (PyTorch RPC + DDP) |
| `experiment/ppo-new/actor.py` | Rollout workers; contains the k-hop subgraph logic |
| `experiment/ppo-new/scripts/` | Original multi-GPU launch scripts |
| `python/quartz/_cython/core.pyx` | Python bindings; `to_dgl_graph()` (around line 531) builds the encoder's input graph |
| `experiment/deprecated/` | Older attempts, including attention and transformer variants (`rl_a2c_rewrite_attn.py`, `rl_a2c_transformer_xl.py`, `rl/QBert.py`) |
| `experiment/ppo-nonhir/` | Variant using a global-attention readout over node embeddings |

Note that `ActorCritic` defaults to `gnn_type='QGIN'` with 6 layers, but `config/base_config.py` overrides this with `QGNN` and 4 layers, and the training scripts pass `c.gnn_type=qgnn`. **`QGNN` with 4 layers is the real baseline.**

---

## 3. The encoder interface (the contract)

Any replacement encoder must satisfy this. It was checked by running both `QGNN` and `QGIN` (see section 5).

**Input:** a `dgl.DGLGraph`, possibly a batch built with `dgl.batch`, with:
- `g.ndata['gate_type']`: integer gate type per node
- `g.edata['src_idx']`, `g.edata['dst_idx']`, `g.edata['reversed']`: integer edge features. Both existing encoders stack these into `g.edata['w']` with shape `(num_edges, 3)`.

**Output:** a tensor of shape `(total_num_nodes, gnn_output_dim)`: one row per node, **in the same node order as the input**. Callers split the output with `batch_num_nodes()`, so reordering nodes silently breaks things.

**Heads:** the actor takes one node embedding and returns `action_dim` logits (one per rewrite rule). The critic takes node embeddings and returns one scalar each.

**Registering a new encoder:** add a branch to the `gnn_type` if/elif chain in `ActorCritic.__init__` (`experiment/ppo-new/model/actor_critic.py`), and set `gnn_output_dim` accordingly.

**Gotcha with `QGIN`:** construct it with `gin_graph_pooling_type='none'` (the string, not Python `None`). With `None`, the constructor raises `NotImplementedError`. With `'sum'`, `'mean'` or `'max'`, `forward` returns a tuple `(node_embeddings, graph_feature)`, which breaks the contract above.

### Baseline sizes (4 layers, 32-dimensional output)

| Encoder | Parameters |
|---|---|
| `QGNN` | 12,096 |
| `QGIN` | 8,896 |

Keep new encoders in a similar range, or report size next to results. `QGIN` contains batch norm and `QGNN` does not, so train and eval behavior can differ.

---

## 4. Important constraint: the encoder must be local

`config/base_config.py` sets `subgraph_opt: bool = True` (verified). In `actor.py`, when this is on:

- During **rollout**, the actor encodes the **full** circuit graph.
- For the **training data**, it stores only the **k-hop subgraph** around the node acted on, using `dgl.khop_out_subgraph(..., k=self.ac_net.gnn_num_layers)`, and remaps the action's node index into that subgraph.

A k-layer message-passing network gives a node the same embedding from the full graph or from its k-hop neighborhood, because the node only ever depends on nodes within k hops. So for the baseline, the shortcut is safe.

A **global-attention** encoder (for example a graph transformer where every node attends to every node) does **not** have this property. Rollout would use the full graph and training would use the subgraph, so the embeddings, values and log-probabilities would disagree. PPO compares new and old action probabilities, so this produces a corrupted ratio **without raising any error**.

Options for a transformer-style encoder:

1. **Restrict attention to the k-hop neighborhood.** Safest; keeps the existing logic valid.
2. **Set `subgraph_opt=False`.** Consistent, but costs more memory and time.
3. **Use a larger k than the model depth.** Cheaper than option 2, but only approximate.

Also, k is read from `ac_net.gnn_num_layers`. A new encoder must keep that attribute meaningful, or the truncation will not match the encoder's real receptive field.

**(unverified)** The "out-neighborhood equals receptive field" argument assumes every edge also exists in the reverse direction, which the `reversed` edge feature suggests. Confirm by reading `to_dgl_graph()` in `python/quartz/_cython/core.pyx` (around line 531).

---

## 5. Smoke test

This script checks the contract on a toy circuit. Save it as `experiment/ppo-new/smoke_test.py` and run it from `experiment/ppo-new`.

```python
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

for gnn_type in ['QGNN', 'QGIN']:
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
```

Expected output (verified on the Windows CPU setup below):

```
batched nodes: 8 per-graph: [4, 4]
QGNN -> embeddings (8, 32) | actor logits (2, 32) | critic values (8, 1)
  params: 12096
QGIN -> embeddings (8, 32) | actor logits (2, 32) | critic values (8, 1)
  params: 8896
```

To test a new encoder, add it to `ActorCritic` and add its name to the loop.

---

## 6. Setting up on Windows (CPU, for development and unit tests)

This is the sequence that worked, including fixes for problems that the stock instructions do not cover. Use the **x64 Native Tools Command Prompt for VS** (Visual Studio Build Tools with the "Desktop development with C++" workload).

### 6.1 Clone outside OneDrive

Building inside a synced folder causes file-lock and path problems.

```
mkdir C:\dev
cd C:\dev
git clone --recursive https://github.com/tejaswibteggihalli/3rd_sem_EL.git
cd 3rd_sem_EL
```

If you cloned without `--recursive`: `git submodule update --init --recursive`.

### 6.2 Conda environment

```
conda env create --name quartz --file env.yml
conda activate quartz
```

### 6.3 Build the C++ runtime

Two fixes were needed: CMake 4.x rejects old minimum-version declarations, and `CMakeLists.txt` calls pybind11 with a flag (`--cmake`) that pybind11 3.x removed, so pybind11 must be located explicitly.

```
for /f "delims=" %i in ('python -m pybind11 --cmakedir') do set PYBIND11_DIR=%i
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPython_EXECUTABLE="%CONDA_PREFIX%\python.exe" -DPython3_EXECUTABLE="%CONDA_PREFIX%\python.exe" -Dpybind11_DIR="%PYBIND11_DIR%"
cmake --build . --config Debug --parallel
cd ..
build\Debug\test_optimize.exe
```

Success looks like `number of xfers: 130` followed by `Best cost` lines (press Ctrl+C after a few). If the first `cmake` fails on pybind11, `pip install "pybind11<3"` is the fallback. Run `test_optimize.exe` from the repo root, not from `build\Debug`.

### 6.4 Build the Python bindings

`python/setup.py` assumes Linux paths and a GCC flag, so supply the Windows equivalents through environment variables instead of editing the file:

```
cd python
set INCLUDE=C:\dev\3rd_sem_EL\src;%INCLUDE%
set LIB=C:\dev\3rd_sem_EL\build\Debug;%LIB%
set CL=/std:c++17
python setup.py build_ext --inplace install
cd ..
```

Python 3.8+ does not search `PATH` for extension-module DLLs, so copy the runtime DLL next to the module, both in the source tree and in the installed package:

```
copy build\Debug\quartz_runtime.dll python\quartz\
copy build\Debug\quartz_runtime.dll "%CONDA_PREFIX%\Lib\site-packages\quartz\"
```

Repeat the copy whenever the C++ runtime is rebuilt.

### 6.5 PyTorch and DGL (CPU)

`import quartz` imports `dgl` and `torch` (`core.pyx`, lines 25-26). The versions below were chosen because DGL 2.2.1 bundles a `graphbolt` library only for torch up to 2.3.0:

```
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==2.2.1 torchdata==0.7.1 "numpy<2" pyyaml pydantic
conda env config vars set DGLBACKEND=pytorch
conda deactivate
conda activate quartz
pip install hydra-core icecream wandb natsort sortedcontainers tqdm matplotlib pylatexenc
python -c "import torch, dgl, quartz; print('ok', torch.__version__, dgl.__version__)"
```

`numpy<2` makes pip warn that `qiskit` wants NumPy 2; this was ignored because qiskit is not needed to import `quartz`.

### 6.6 Limits of the Windows setup

- Good for: reading code, unit-testing encoders, short forward passes, debugging.
- **Full training probably does not work on Windows.** `ppo.py` and `actor.py` use `torch.distributed.rpc` and DDP, and the shipped scripts (`scripts/pret_rig5_rm.sh`) assume 4 GPUs (`'c.gpus=[0,1,2,3]'`, `mini_batch_size=4800`). **(unverified)** Check with `python -c "import torch.distributed.rpc as rpc; print(rpc.is_available())"`.
- For training, use Linux (a Linux machine, WSL2, Colab or Kaggle). Reduce `gpus`, `mini_batch_size` and the number of parallel agents for a single GPU. The rollout work is CPU-bound, and free cloud tiers have only a few CPU cores, so expect slower training than the original setup.
- The `experiment/ppo-new/env_ppo.yml` file pins an old CUDA 11.7 stack. Do not recreate it as-is on a new machine; pick a matching modern torch and DGL pair instead.

---

## 7. Theory

### 7.1 Graphs and the encoder

A circuit is a graph: nodes are gates, edges are qubit wires. A neural network cannot read a graph directly, so the encoder converts each node into a fixed-size vector (32 numbers in the baseline) that captures the gate and its surroundings.

### 7.2 Message passing, using `QGNN` as the example

Each `QConv` layer (`qgnn.py`) does four things:

1. **Message.** Every node sends a message along each edge: the sender's vector joined with the edge features `[src_idx, dst_idx, reversed]`. Edge features matter because, for example, a CX gate has distinct control and target qubits.
2. **Transform.** Each message goes through a Linear layer and ReLU.
3. **Aggregate.** Each node combines the messages it received into one vector (sum by default).
4. **Update.** The node joins its own vector with the aggregate and passes the result through another Linear and ReLU.

`QGNN` first embeds each gate type with `nn.Embedding`, then applies `num_layers` of these layers. After layer 1 a node knows its direct neighbors; after layer 2, nodes two steps away; and so on. **Number of layers equals the number of hops a node can see (its receptive field).** That is why `gnn_num_layers` appears in the k-hop code in `actor.py`.

The weights are shared across all nodes, so the same network handles circuits of any size.

### 7.3 Aggregation choices

A node's neighbors have no natural order, so aggregation must be order-invariant. `QConv` supports:

- **sum:** preserves counts (three neighbors differ from one). Default.
- **mean:** keeps the average pattern, ignoring counts.
- **max:** keeps the strongest signal per feature.

For circuits, counts plausibly matter (for example, how many T gates are nearby), which may be why sum is the default. **(unverified: this is an interpretation, not something documented in the repo.)**

### 7.4 The alternatives in this repo

- **`QGNN`:** a GraphSAGE-style "pooling aggregator" (the code comment says so), with a transform before aggregation and an update afterward.
- **`QGIN`:** Graph Isomorphism Network. Sum aggregation followed by an MLP, with batch normalization. Adapted from the DGL GIN example, per the comments in `qgin.py`. GIN is known for being as expressive as standard message passing can be at distinguishing graphs.
- **`qgat.py`:** Graph Attention Network. Learns an attention weight per neighbor so more relevant neighbors count more. **(Not read in detail; verify before relying on this description.)**

All three are **local**: a node only communicates with its neighbors.

### 7.5 Where transformers fit

A transformer's core operation is **self-attention**: each item computes how much to attend to every other item and takes a weighted mix.

1. **It is global.** Every node can see every other node in one layer. Cost grows with the square of the node count.
2. **It has no built-in sense of structure.** A plain transformer sees an unordered set. For circuits, wire connectivity is essential, so a **graph transformer** injects structure back in, for example through graph-derived positional encodings or an attention bias based on edges.

One way to think of it: GAT is attention restricted to neighbors; a graph transformer is attention over everyone, guided by the graph. The locality constraint in section 4 is the main design issue when trying a transformer here.

This repository includes two experimental transformer encoders:

- **`QTransformer`** uses node order as the sequence position and ignores graph edges.
- **`QTransformerDepth`** uses the longest-path depth of each node over the forward circuit edges
  (`reversed == 0`) as its sinusoidal position. Nodes at the same depth share a position,
  while self-attention remains global.

`QTransformerDepth` is therefore structure-aware positional encoding, not local message
passing. It does not make `subgraph_opt=True` safe: the full-graph rollout and the
subgraph used during PPO training can still produce different embeddings. Use
`subgraph_opt=False` when evaluating either transformer, or implement an explicitly
local attention mask before comparing transformer results with the local GNN baselines.

### 7.6 How the encoder is used in RL (actor-critic with PPO)

- The **critic** scores every node (how promising is acting at this gate?). Scores are used to choose which node to act on.
- The **actor** takes the chosen node's embedding and scores each rewrite rule. This chooses which rule to apply.
- The **reward** reflects how much the circuit cost decreases.
- **PPO** updates the policy in small, safe steps. It compares the probability of an action under the old policy (when it was taken) with the probability under the current policy, and clips the ratio so one update cannot change the policy too much. In `ppo.py` this appears as `torch.exp(xfer_logprobs - exps.xfer_logprob)`.

Encoder consistency matters because of that ratio: if "old" came from the full graph and "new" from a subgraph, the ratio contains noise that is not learning.

### 7.7 Suggested study order

1. Neural network basics: Linear layers, ReLU, embeddings, backpropagation.
2. Message passing: re-read `qgnn.py` and map each line to the four steps in 7.2.
3. GIN versus GAT: read `qgin.py` and `qgat.py` next to `qgnn.py`.
4. RL basics: policy, reward, actor-critic, then PPO's clipping idea (Schulman et al., 2017).
5. Transformers: only if pursuing a graph transformer. Start with "Attention Is All You Need", then a survey of graph transformers.

A good self-test: explain why a 4-layer encoder makes `subgraph_opt` safe. (Answer: each node's embedding depends only on nodes within 4 hops, so cutting out that neighborhood loses nothing.)

---

## 8. Open items

- Verify that `to_dgl_graph()` adds reversed copies of every edge (section 4).
- Check whether `torch.distributed.rpc` is available on the Windows setup (section 6.6).
- Read `qgat.py` and confirm its attention mechanism.
- Choose single-GPU settings (`gpus`, `mini_batch_size`, number of agents) for Colab and Kaggle runs, starting from `config/base_config.py`.
- Time one baseline training iteration on a free T4 to see whether the CPU limit makes cloud training workable.
- For any transformer-style encoder, decide how to handle the locality constraint (section 4) before comparing results with the baseline.
