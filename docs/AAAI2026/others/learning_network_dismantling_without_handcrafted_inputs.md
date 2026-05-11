---
title: >-
  [Paper Note] Learning Network Dismantling Without Handcrafted Inputs
description: >-
  [AAAI 2026][Network dismantling] This paper proposes MIND (Message Iteration Network Dismantler), which eliminates the dependence of GNNs on handcrafted features through a novel All-to-One attention mechanism and Message…
tags:
  - "AAAI 2026"
  - "Network dismantling"
  - "graph neural networks"
  - "attention mechanism"
  - "reinforcement learning"
  - "vital node identification"
date: 2026-05-08
content_hash: 5f857a1176bb11e1
---

# Learning Network Dismantling Without Handcrafted Inputs

**Conference**: AAAI 2026
**arXiv**: [2508.00706](https://arxiv.org/abs/2508.00706)
**Code**: [GitHub](https://github.com/HaozheTian/MIND-ND)
**Area**: Other
**Keywords**: Network dismantling, graph neural networks, attention mechanism, reinforcement learning, vital node identification

## TL;DR

This paper proposes MIND (Message Iteration Network Dismantler), which eliminates the dependence of GNNs on handcrafted features through a novel All-to-One attention mechanism and Message Iteration Profiles. Using only raw adjacency information, MIND achieves state-of-the-art network dismantling performance on real-world networks with up to millions of nodes, while maintaining the lowest computational complexity of $O(|V|+|E|)$.

## Background & Motivation

**Network Dismantling Problem**: Given a network, find a node removal sequence that fragments the network into isolated small components as rapidly as possible. This is equivalent to vital node identification and has broad real-world applications, including dismantling criminal networks, targeted vaccination to interrupt infectious disease spread, assessing healthcare system resilience, and preventing wildfire propagation. The problem is NP-hard, and only approximate solutions can be sought.

**Three Major Approaches**:

**Centrality heuristics** (degree centrality, betweenness centrality, PageRank): computationally simple but with fixed strategies that cannot adapt to different network topologies.

**Approximate theoretical methods** (optimal percolation, graph decycling, generalized network dismantling): theoretically grounded but computationally expensive.

**Machine learning methods** (FINDER, GDM): employ GNNs to learn node representations, but **heavily rely on handcrafted input features** (degree, neighborhood degree statistics, k-core number, clustering coefficient, etc.).

**Key Challenge**: Although existing ML methods (FINDER, GDM) leverage GNNs to achieve learnable dismantling strategies, they require precomputed handcrafted node features as GNN inputs. This introduces two problems:

**High computational overhead**: Computing features such as betweenness centrality on networks with millions of nodes is prohibitively expensive.

**Introduced bias**: Handcrafted features constrain the model's "field of view." Experiments demonstrate that GDM's dismantling sequences are highly correlated (significant Spearman $R$) with the PCA ranking of its input features, indicating that the model is bound by the input features and cannot discover superior structural information.

**Key Insight**: This work completely eliminates handcrafted features and initializes GNN node embeddings with all-ones vectors. Two novel mechanisms enable the GNN to learn sufficiently rich structural representations from pure adjacency information: (1) an All-to-One attention mechanism replacing softmax normalization, and (2) Message Iteration Profiles that aggregate embeddings across all layers. Combined with structurally diverse synthetic training networks, this achieves generalization from small networks to large real-world networks.

## Method

### Overall Architecture

MIND adopts a GNN-based Actor-Critic reinforcement learning framework:

- **State**: current network $G_t = (V_t, E_t)$
- **Action**: select a node $v_t$ for removal
- **Reward**: $r_t = -\text{LCC}(G_t \setminus \{v_t\}) / |V_0|$ (negative relative size of the largest connected component)
- **Objective**: minimize the area under the dismantling curve (AUC), i.e., minimize $\sum_t \text{LCC}(G_t) / |V_0|$
- **Encoder**: GNN extracts structural representation $z_i$ for each node
- **Decoder**: two MLPs outputting Q-values and policy probabilities respectively

The Actor (policy $\pi$) and Critic (Q-function) share the GNN encoder's node embeddings, augmented by a synthetic global node $v_o$ that provides network-level state information. Training uses the Soft Actor-Critic (SAC) algorithm, which offers high sample efficiency and entropy regularization to encourage exploration.

### Key Designs

1. **All-to-One Attention Mechanism (MIND-AM)**:

    - **Function**: Replaces softmax-normalized attention in standard GAT to address the failure of GNN learning under constant initialization.
    - **Mechanism**: For head $h$, attention coefficients depend not only on the features of the current head but also on the concatenated features of all heads. The self-attention coefficient is $\alpha_i^h = \text{MLP}_\sigma^h([\|_{h=1}^H W_\sigma^h e_i^h])$, and the neighbor attention coefficient is $\alpha_{i,j}^h = \text{MLP}_\nu^h([\|_{h=1}^H W_\sigma^h e_i^h] + [\|_{h=1}^H W_\nu^h e_j^h])$; outputs are squashed to $(0,1)$ via sigmoid.
    - **Design Motivation**: Standard GAT uses softmax-normalized attention coefficients. When all node embeddings are initialized identically (all-ones vectors), softmax assigns equal weight to every neighbor, causing all node embeddings to remain identical throughout iterations and rendering learning impossible. MIND-AM replaces softmax with sigmoid, avoiding explicit normalization while achieving implicit adaptive normalization through cross-head information. This simultaneously preserves injectivity over neighbor multisets and prevents embedding value explosion.

2. **Message Iteration Profiles (MIND-MP)**:

    - **Function**: Concatenates node embeddings from all message-passing iterations (layers) as the final representation, rather than using only the last layer.
    - **Mechanism**: $z_i = \text{MLP}_\zeta([\|_{k=1}^K e_i^{(k)}])$, where $e_i^{(k)}$ is the embedding of node $i$ after the $k$-th iteration.
    - **Design Motivation**: Two aspects. (1) Mitigating over-smoothing: as the number of iterations increases, all node embeddings converge to the principal eigenvector of the message-passing operator (Theorem 1 in the paper); early iterations retain the diversity of local structure. (2) Capturing convergence rate information: nodes with different centralities converge at different rates (more central nodes begin aggregating global information earlier), and these convergence rate differences themselves encode critical structural role information. The paper theoretically proves that MIND can approximate the Fiedler vector—a spectral heuristic widely used in the dismantling literature.

3. **Systematically Diversified Training Networks**:

    - **Function**: Systematically generates structurally diverse training networks through degree-preserving edge rewiring.
    - **Mechanism**: Generates 10,000 small networks (100–200 nodes) from three models—LPA, Copying Model, and ER—then applies edge rewiring to introduce varying levels of degree assortativity (assortative/uncorrelated/disassortative) and modularity (modular/random/multipartite).
    - **Design Motivation**: Standard random graph models generate structurally homogeneous networks that fail to cover the structural diversity of real-world networks (e.g., high modularity, strong negative assortativity). Edge rewiring preserves the degree sequence while altering connection patterns, effectively expanding the training distribution.

4. **Omni-node**:

    - **Function**: Adds a synthetic node $v_o$ to which all other nodes are connected via directed edges.
    - **Mechanism**: The embedding $z_o$ of $v_o$ aggregates network-wide information and is concatenated with node embedding $z_i$ before being passed to the decoder: $Q(G_t, v_i) = \text{MLP}_\theta([z_i \| z_o])$.
    - **Design Motivation**: Dismantling decisions require not only local information (how important is the node itself) but also global state (the current overall fragmentation of the network); the omni-node provides awareness of the global network state.

### Loss & Training

The discrete-action variant of Soft Actor-Critic (SAC) is adopted:
- **Q-function update**: minimize TD error $Q(G_t, v_t) \approx r_t + Q(G_{t+1}, \pi(G_{t+1}))$
- **Policy update**: maximize $\sum_{v_i \in V_t} \pi(v_i|G_t) Q(G_t, v_i)$ with entropy regularization
- Experience replay improves sample efficiency; training spans a total of 8 million dismantling episodes.

## Key Experimental Results

### Main Results (Real-World Network Dismantling, Relative AUC, MIND = 100)

| Method | Biological | Social | Information | Technological | Overall |
|--------|-----------|--------|-------------|---------------|---------|
| MIND | **100.0** | **100.0** | **100.0** | 100.0 | **100.0** |
| GDM | - | - | - | - | 104.13 |
| EI | - | - | - | - | 107.96 |
| FINDER | - | - | - | - | >107.96 |

MIND ranks first overall across dozens of real-world networks spanning four categories, achieving an overall AUC 4.13% lower than the second-best method GDM.

### Ablation Study

| Configuration | Validation AUC | Notes |
|---------------|----------------|-------|
| Full MIND | Best and stable | All components work synergistically |
| w/o MIND-AM | Performance degrades, $p=8.1\times10^{-4}$ | Attention mechanism is critical for learning under constant initialization |
| w/o MIND-MP | Performance degrades, $p=4.5\times10^{-2}$ | Multi-layer embedding aggregation provides essential structural information |
| GATv2 substitute | Completely fails to learn | Softmax normalization keeps all embeddings identical under constant initialization |
| GCN substitute | Suboptimal with high variance | Capable of learning but with insufficient expressiveness |

### Effect of Training Network Diversification

| Network Type | AUC with rewiring / without rewiring | Notes |
|--------------|--------------------------------------|-------|
| High-modularity network (roads-california, modularity 0.975) | <20% | Modularity introduced by rewiring leads to substantial strategy improvement |
| Strong negative assortativity network (munmun_twitter, assortativity −0.878) | ~60% | Disassortative mixing created by rewiring enhances generalization |
| Most networks | <100% | Overall positive effect |

### Key Findings

- MIND has the lowest computational complexity among all ML methods: $O(|V|+|E|)$, as no handcrafted feature precomputation is required.
- GDM's dismantling sequences are highly correlated (large Spearman $R$) with the PCA ranking of its input features, demonstrating that handcrafted features introduce strategy bias.
- MIND's dismantling sequences are nearly uncorrelated with standard heuristics, indicating that it has learned novel structural features beyond known heuristics.
- On ER random graphs, simple degree centrality outperforms GDM with handcrafted features, further confirming the detrimental effect of handcrafted features.
- MIND scales to networks with up to 1.4 million nodes, despite being trained exclusively on small networks of 100–200 nodes.

## Highlights & Insights

- The core contribution extends beyond "removing handcrafted features": the paper demonstrates that handcrafted features actually **constrain** the learning capacity of GNNs, overturning the conventional wisdom that handcrafted feature initialization is always beneficial.
- The design of MIND-AM is elegant: replacing softmax with cross-head sigmoid attention simultaneously resolves (1) the degeneration problem under constant initialization and (2) embedding value explosion—achieving two goals with a single design.
- The theoretical analysis of MIND-MP is insightful: the differences in convergence rates across nodes during iteration inherently encode centrality information, and explicitly preserving this "process information" is a key insight.
- The training network diversification method is simple and effective: degree-preserving edge rewiring to modulate assortativity and modularity can be directly transferred to other graph learning tasks.

## Limitations & Future Work

- Performance on technological networks (large-diameter networks such as power grids) is slightly inferior to EI and GND, as the receptive field of GNN message passing is limited by the number of layers.
- The method addresses only node dismantling and has not been extended to link dismantling.
- Training requires 8 million episodes, incurring non-trivial training costs.
- Validation is currently limited to graphs without attributes; whether the approach remains superior to handcrafted feature initialization when node/edge attributes are present has not been explored.

## Related Work & Insights

- The Message Iteration Profiles concept is similar to JK-Net (Jumping Knowledge Networks), but with a more theoretically grounded motivation based on convergence rate analysis.
- The All-to-One attention mechanism resolves the degeneration of GAT on feature-free graphs, a finding with implications for all tasks requiring representation learning on graphs without node features.
- The structural diversification method for training networks is generalizable to other combinatorial optimization problems on graphs (e.g., minimum vertex cover, maximum cut).
- The success of this work suggests that allowing models to autonomously discover features may be a more effective paradigm than manually engineering features in graph learning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Optimal Welfare in Noncooperative Network Formation under Attack](optimal_welfare_in_noncooperative_network_formation_under_attack.md)
- [\[AAAI 2026\] Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[AAAI 2026\] DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes](detonation_decoupled_torch_network-aware_training_on_interlinked_online_nodes.md)
- [\[NeurIPS 2025\] Optimism Without Regularization: Constant Regret in Zero-Sum Games](../../NeurIPS2025/others/optimism_without_regularization_constant_regret_in_zero-sum_games.md)

</div>

<!-- RELATED:END -->
