---
title: >-
  [Paper Note] GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning
description: >-
  [ICLR 2026][Graph Learning][heterophilic graph] This paper proposes GRAPHITE, a learning-free graph transformation method that **directly boosts graph homophily** by introducing "feature nodes" as hubs to indirectly conn…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "heterophilic graph"
  - "homophily boosting"
  - "graph transformation"
  - "feature nodes"
  - "GNN"
date: 2026-05-08
content_hash: 407711f57e5ead7d
---

# GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning

**Conference**: ICLR 2026
**arXiv**: [2602.07256](https://arxiv.org/abs/2602.07256)  
**Code**: [https://github.com/q-rz/ICLR26-GRAPHITE](https://github.com/q-rz/ICLR26-GRAPHITE)  
**Area**: Graph Learning
**Keywords**: heterophilic graph, homophily boosting, graph transformation, feature nodes, GNN

## TL;DR
This paper proposes GRAPHITE, a learning-free graph transformation method that **directly boosts graph homophily** by introducing "feature nodes" as hubs to indirectly connect nodes sharing common features. It is the first approach to address heterophilic graph learning by modifying graph structure rather than redesigning GNN architectures, achieving substantial improvements over 27 state-of-the-art methods on challenging benchmarks such as Actor.

## Background & Motivation

1. **The heterophilic graph dilemma**: The core operation of GNNs—neighborhood aggregation—breaks down on heterophilic graphs, where connected nodes are inherently dissimilar in features/labels, causing aggregation to dilute rather than enrich useful signals.
2. **Architecture-level fixes have plateaued**: A large body of heterophilic GNNs (H2GCN, GPR-GNN, FAGCN, OrderedGNN, etc.) have tackled this problem via separated ego/neighbor embeddings, multi-hop information, frequency-based filtering, and similar strategies. Yet experiments show that 23 recent GNNs still underperform the simplest MLP baseline on the Actor dataset.
3. **Root cause observation**: The fundamental issue lies in the **low homophily of the graph structure itself**, not insufficient model capacity. The appropriate remedy is therefore to directly modify the graph structure to be more homophilic, rather than engineering more complex models.
4. **Key insight**: The definition of homophily implies that nodes sharing features are more likely to be adjacent. A naïve solution—directly adding edges between all pairs of nodes sharing at least one feature—can be proven to increase homophily, but the number of added edges grows as $O(|V|^2)$, making it computationally infeasible.

## Core Problem
How can one directly boost homophily in heterophilic graphs through a deterministic graph transformation, without substantially increasing graph size, so that standard GNNs can operate effectively?

## Method

### Step 1: Naïve Homophily Booster (NHB) — Theoretical Motivation
For every pair of nodes $(v_i, v_j)$ sharing at least one feature, a direct edge is added. Theorem 1 proves this strictly increases homophily: $\text{hom}(\mathcal{G}^\dagger) > \text{hom}(\mathcal{G})$. However, the number of new edges is $O(|V|^2)$ (a graph with 2,000 nodes may incur nearly 2 million additional edges), rendering the approach impractical for real training.

### Step 2: Efficient Design of GRAPHITE
The core idea replaces direct connections with indirect connections mediated by "feature nodes" acting as hubs:

1. **Create feature nodes**: For each feature $k$ in the feature set $\mathcal{X}$, introduce a feature node $x_k$, yielding $|\mathcal{X}|$ new nodes in total.
2. **Add feature edges**: For each graph node $v_i$ possessing feature $k$ (i.e., $\mathbf{X}[v_i, k] = 1$), add the edge $(v_i, x_k)$.
3. **Adjacency matrix**: The transformed graph has a block-structured adjacency matrix $\mathbf{A}^* = \begin{bmatrix} \mathbf{A} & \mathbf{X} \\ \mathbf{X}^T & \mathbf{0} \end{bmatrix}$.
4. **Feature node attribute vectors**: Each feature node is assigned the mean of the attribute vectors of all graph nodes connected to it.

**Key property**: Any pair of nodes $(v_i, v_j)$ sharing feature $k$ are connected within 2 hops via $v_i \to x_k \to v_j$ (Observation 2), enabling homophilic message passing.

### Step 3: Theoretical Guarantees (Theorem 3)
- Homophily is strictly improved: $\text{hom}(\mathcal{G}^*) > \text{hom}(\mathcal{G})$
- Graph size grows only linearly: $|\mathcal{V}^*| \leq O(|\mathcal{V}|)$, $|\mathcal{E}^*| \leq O(|\mathcal{E}|)$
- Compared to NHB's $O(|V|^2)$ edge growth, GRAPHITE is substantially more efficient

### Step 4: Dedicated GNN Architecture
To fully exploit the structural properties of the transformed graph, a specialized aggregation mechanism is designed to differentiate graph edges from feature edges:

- **Typed edge weights**: Graph edge weight $w_\mathcal{E} = 1$; feature edge weight $w_\mathcal{X}$ and self-loop weight $w_0$ are tunable
- **Self-gating mechanism** (inspired by FAGCN): $\alpha_{u,u'} = \tanh(\frac{\mathbf{a}^T(\mathbf{h}_u \| \mathbf{h}_{u'}) + b}{\tau})$, adaptively leveraging homophilic and heterophilic signals
- Aggregation is followed by an MLP with residual connections and GELU activation

### Loss & Training
- The graph transformation is a **learning-free deterministic preprocessing step**, compatible as a plug-and-play module with any existing GNN
- Standard cross-entropy loss is used for node classification

## Key Experimental Results

### Main Results: Comparison with 27 Methods (6 Datasets)

| Method | Actor | Squirrel-F | Chameleon-F | Minesweeper | Cora | CiteSeer |
|--------|-------|-----------|------------|------------|------|----------|
| MLP | 35.04 | 33.91 | 38.44 | 50.99 | 75.45 | 71.53 |
| GCN | 30.21 | 35.57 | 40.06 | 72.32 | 87.50 | 75.77 |
| FAGCN | 36.18 | 36.52 | 39.83 | 84.69 | 88.66 | 76.82 |
| JKNet | 28.63 | 40.81 | 40.39 | 81.00 | 86.24 | 73.11 |
| SGFormer | 25.89 | 34.54 | 42.79 | 52.06 | 86.24 | 70.74 |
| **GRAPHITE** | **37.69** | **43.06** | **45.08** | **94.78** | 88.23 | 76.41 |

GRAPHITE ranks first by a clear margin on all 4 heterophilic graphs, outperforming the best baseline by +4.17%/+5.23%/+5.35%/+3.47%; performance on homophilic graphs is on par with state of the art.

### Homophily Improvement from GRAPHITE Transformation (Before → After)

| Dataset | Feature Homophily Gain | Adjusted Homophily Gain |
|---------|----------------------|------------------------|
| Actor | +179% | +2767% |
| Squirrel-F | +961% | +215% |
| Chameleon-F | +1739% | +402% |
| Minesweeper | +41% | +1023% |

### Gain from GRAPHITE Transformation Applied to Homophilic GNNs (Plug-and-Play)

| Method | Actor (Original → Transformed) | Minesweeper (Original → Transformed) |
|--------|-------------------------------|--------------------------------------|
| GCN | 30.21 → 34.83 | 72.32 → 75.38 |
| GAT | 28.86 → 32.09 | 87.59 → 88.66 |
| JKNet | 28.63 → 35.96 | 81.00 → 85.56 |
| GIN | 28.29 → 33.75 | 75.89 → 87.07 |

### Efficiency Analysis

| Dataset | Training Time (No Transform) | Training Time (With Transform) |
|---------|------------------------------|-------------------------------|
| Minesweeper (10k nodes) | 1.9 min | 2.3 min |
| Actor (7.6k nodes) | 1.5 min | 2.0 min |
| Squirrel-F (2.2k nodes) | 0.7 min | 1.1 min |

Runtime overhead is modest and far outweighed by accuracy gains.

### Ablation Study: Feature Node Aggregation Strategy
Mean aggregation, learnable embeddings, learnable attention, and majority voting yield nearly identical performance; the simplest option—mean aggregation—is therefore adopted.

## Highlights & Insights
- **A new paradigm**: GRAPHITE is the first to propose directly boosting graph homophily rather than designing more powerful GNNs—a previously unexplored direction.
- **Minimalist yet powerful**: The core operation simply adds feature nodes and edges—no learning, no training, fully deterministic, and plug-and-play compatible with any GNN.
- **Theoretically grounded**: Both homophily improvement and linear graph size growth are backed by rigorous mathematical proofs.
- **Compelling empirical evidence**: A comprehensive comparison against 27 baselines demonstrates state-of-the-art performance on all heterophilic benchmarks; the graph transformation alone substantially benefits standard homophilic GNNs.

## Limitations & Future Work
- **Reliance on discrete/binary features**: Continuous features must be discretized (e.g., via one-hot encoding), and no theoretical guidance is provided for choosing discretization granularity.
- **Sensitivity to feature dimensionality**: The number of feature nodes equals the feature dimension $|\mathcal{X}|$; very high-dimensional sparse features may produce an excessive number of hub nodes.
- **False-positive homophilic connections**: Nodes sharing a feature but belonging to different classes are also connected within 2 hops; self-gating mitigates but does not eliminate this issue.
- **Scope of theoretical assumptions**: Theorems 1 and 3 rely on "mild and realistic assumptions," but conditions under which these assumptions fail are not thoroughly discussed.

## Related Work & Insights
- **vs. H2GCN / GPR-GNN / ACM-GNN**: These methods address heterophily at the architecture level (separated ego/neighbor representations, polynomial filters, etc.); GRAPHITE modifies the graph structure directly—the two approaches are orthogonal and complementary.
- **vs. FAGCN**: FAGCN employs self-gating to adaptively handle homophilic and heterophilic signals; the GNN component of GRAPHITE also uses self-gating, but the key innovation lies in the upstream graph transformation.
- **vs. Graph augmentation / graph rewiring methods**: Most such methods learn edge weights or add/remove edges (e.g., DropEdge, IDGL); GRAPHITE performs a **deterministic, feature-structure-based transformation** involving no learning.
- **vs. Virtual Node**: Virtual Node adds a single global node connected to all nodes, lacking feature-level discrimination; GRAPHITE's feature nodes establish fine-grained connections by feature category.
- **vs. Graph Transformers (NodeFormer, SGFormer, DIFFormer)**: Transformers capture long-range dependencies via global attention but do not necessarily perform well on small heterophilic graphs; GRAPHITE achieves more direct improvements through structural transformation.

### Additional Insights
- **"Transform the data before improving the model"**: GRAPHITE demonstrates that in the heterophilic graph setting, data-level transformation can be more effective than model-level innovation—a principle that may generalize to other scenarios where data properties are unfavorable to standard models.
- **Feature nodes as structured explicit memory**: Each feature node aggregates information from all nodes possessing that feature, functioning as a cluster centroid defined by discrete attributes.
- **Integration with self-supervised contrastive learning**: Feature nodes provide natural positive sample groups for contrastive learning (nodes sharing a feature node neighbor form positive pairs), which is a direction worth exploring.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A genuinely new paradigm—directly transforming graph structure to boost homophily, offering a uniquely distinct perspective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison against 27 baselines + quantified homophily gains + plug-and-play verification + ablation + efficiency analysis
- Writing Quality: ⭐⭐⭐⭐⭐ The progressive exposition from the naïve approach (NHB) to the efficient solution (GRAPHITE) is logically clear and well-structured
- Value: ⭐⭐⭐⭐⭐ Simple yet effective method, pioneering new paradigm, complete theoretical guarantees—strong potential for follow-up research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs](../../AAAI2026/graph_learning/gcl-ot_graph_contrastive_learning_with_optimal_transport_for_heterophilic_text-a.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](../../ICML2026/graph_learning/generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](../../AAAI2026/graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[ICLR 2026\] A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)

</div>

<!-- RELATED:END -->
