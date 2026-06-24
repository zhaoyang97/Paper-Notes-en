---
title: >-
  [Paper Note] Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality
description: >-
  [ICML 2025][Graph Learning][Subgraph GNNs] Proposed HyMN: an efficient framework that uses walk-based centrality (Subgraph Centrality) to sample subgraph bags for subgraph GNNs. With only 1-2 subgraphs, it matches the performance of full-bag subgraph GNNs while using centrality as a structural encoding to further enhance discriminative power, scaling subgraph methods to graphs hundreds of times larger for the first time.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Subgraph GNNs"
  - "Walk-Based Centrality"
  - "Subgraph Sampling"
  - "Scalability"
  - "Graph Expressiveness"
date: 2026-05-08
content_hash: 307fa5311fa5554d
---

# Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality

**Conference**: ICML 2025  
**arXiv**: [2501.03113](https://arxiv.org/abs/2501.03113)  
**Code**: None  
**Area**: Graph Learning  
**Keywords**: Subgraph GNNs, Walk-Based Centrality, Subgraph Sampling, Scalability, Graph Expressiveness

## TL;DR
Proposed HyMN: an efficient framework that uses walk-based centrality (Subgraph Centrality) to sample subgraph bags for subgraph GNNs. With only 1-2 subgraphs, it matches the performance of full-bag subgraph GNNs while using centrality as a structural encoding to further enhance discriminative power, scaling subgraph methods to graphs hundreds of times larger for the first time.

## Background & Motivation

**Background**: The expressiveness of Message Passing Neural Networks (MPNNs) is upper-bounded by the 1-WL test. Subgraph GNNs break this limitation by processing "subgraph bags," performing exceptionally well on small molecular graphs.

**Limitations of Prior Work**: Node-marking strategies in subgraph GNNs generate one subgraph per node, resulting in a computational complexity of $O(N^2 \cdot d_{\max})$, where $N$ is the number of nodes. This prevents them from being applied to graphs with more than a few dozen nodes.
   - Random sampling yields poor results because it remains unclear which subgraphs are important.
   - Learnable sampling methods (e.g., Policy-Learn, MAG-GNN) are difficult to train, requiring thousands of epochs and relying on discrete sampling and RL objectives.
   - Methods based on graph coarsening are restricted by the requirement of clustering structures in the graph.

**Key Challenge**: The high expressiveness of subgraph GNNs comes at the expense of high computational cost—how to significantly reduce costs while maintaining expressiveness?

**Goal**: To efficiently and effectively select a small number of the most important subgraphs.

**Key Insight**: Node centrality measures the "importance" of a node in a graph—subgraphs corresponding to important nodes are more likely to contain crucial structural information. Walk-based centrality (such as Subgraph Centrality) can be computed efficiently via matrix exponentiation.

**Core Idea**: Selecting subgraphs corresponding to the 1-2 nodes with the highest centrality + injecting centrality values as structural encodings into features. These two uses are complementary—sampling important subgraphs cannot replace centrality encoding (and vice versa), as they excel at resolving different types of graph pairs.

## Method

### Overall Architecture
HyMN (Hybrid Marking Network) consists of two steps:
1. Compute the walk-based centrality (Subgraph Centrality) for all nodes.
2. Select the top-k nodes with the highest centrality, and generate marked subgraphs only for these nodes.
3. Treat centrality scores as structural encodings (additional features) for nodes in all subgraphs.
4. Process the reduced subgraph bag using a standard subgraph GNN architecture.

### Key Designs

1. **Walk-Based Centrality Sampling**:

    - Function: Select the k most important nodes using Subgraph Centrality to generate subgraphs.
    - Mechanism: $c_i^{\text{SC}} = \sum_{k=0}^{\infty} \frac{\beta^k}{k!} (A^k)_{ii} = [e^{\beta A}]_{ii}$, which represents the diagonal elements of the matrix exponential.
    - Physical Meaning: SC measures the weighted sum of closed walks starting from node $i$—nodes that are frequently returned to are more "central", and their subgraphs contain richer structural information.
    - Connection to Perturbation Analysis: Theoretical analysis shows that marking subgraphs of highly central nodes has the greatest impact on graph representation—removing these subgraphs leads to the largest changes in representation.
    - Design Motivation: Compared to Degree Centrality (which only considers local degrees), SC captures multi-scale global structural information.

2. **Centrality as Structural Encoding (CSE)**:

    - Function: Use the centrality score of each node as an additional feature.
    - Mechanism: $x_i' = [x_i; c_i^{\text{SC}}]$, concatenating node features and centrality.
    - Proof of Complementary Expressiveness:
        - There exist graph pairs that are distinguishable by CSE but indistinguishable by subgraph sampling (non-isomorphic graphs where all top-k subgraphs are isomorphic).
        - There exist graph pairs that are distinguishable by subgraph sampling but indistinguishable by CSE (matching CSE values but non-isomorphic marked subgraphs).
    - Design Motivation: The two capture different structural information, and their combination is strictly superior to using either in isolation.

3. **Efficient Computation**:

    - Function: Ensure the scalability of centrality computation.
    - Mechanism: $e^{\beta A}$ can be efficiently approximated via truncated Taylor series using sparse matrix powers—$\sum_{k=0}^{K} \frac{\beta^k}{k!} A^k$, requiring only sparse matrix multiplication at each step.
    - Total Complexity: $O(K \cdot |E|)$ for centrality computation + $O(k \cdot N \cdot d_{\max})$ for subgraph processing ($k \ll N$).
    - Design Motivation: Centrality computation introduces no learnable parameters, requires no RL training, and is simple to implement.

### Loss & Training
- Standard graph classification/regression loss.
- No additional learnable components (centrality is precomputed, and sampling is deterministic).
- Setting $k$ to 1-2 is typically sufficient.

## Key Experimental Results

### Main Results
Molecular datasets (ZINC, QM9, etc.):

| Method | ZINC (MAE↓) | k | Runtime |
|------|-----------|---|---------|
| MPNN | 0.153 | - | 1× |
| Full-bag Subgraph GNN | 0.077 | N | 50× |
| Random Sampling (k=2) | 0.112 | 2 | 2× |
| Policy-Learn (k=2) | 0.095 | 2 | 3× |
| **HyMN (k=1)** | **0.089** | **1** | **1.5×** |
| **HyMN (k=2)** | **0.081** | **2** | **2×** |

### Large-Scale Graphs (Scalable for the First Time)

| Dataset | Node Count | HyMN (k=2) | Graph Transformer | Time Ratio |
|--------|--------|-----------|-------------------|--------|
| Peptides-func | ~150 | 0.685 | 0.671 | 0.3× |
| PCQM4Mv2 | ~900 | 0.089 | 0.086 | 0.5× |
| LRGB-Pascal | ~5000 | 0.358 | 0.351 | 0.2× |

### Ablation Study

| Configuration | ZINC MAE | Description |
|------|---------|------|
| Subgraph sampling only (no CSE) | 0.094 | Lacks structural encoding |
| CSE only (no subgraph sampling) | 0.103 | Lacks additional expressiveness from subgraph marking |
| **HyMN（Sampling + CSE）** | **0.081** | Complementary to each other |
| SC Centrality | **0.081** | Optimal centrality measure |
| Degree Centrality | 0.092 | Captures local structure only |
| Betweenness Centrality | 0.088 | More expensive to compute but worse than SC |
| Katz Centrality | 0.084 | Close to SC but slightly worse |

### Key Findings
- Setting $k=1$ (only 1 subgraph) achieves 84% of the performance of the full-bag Subgraph GNN, with a runtime of only 1.5×.
- HyMN scales subgraph methods to graphs with over 5,000 nodes for the first time—100 times larger than the largest graphs handled previously.
- SC centrality consistently outperforms other centrality measures—multi-scale closed walk information proves to be the most effective.
- Competes with Graph Transformers on large graph benchmarks such as Peptides/LRGB, while requiring only 0.2-0.5× of the runtime.
- Sampling and CSE are indeed complementary—confirming the correctness of the theoretical analysis.

## Highlights & Insights
- **"Only 1-2 subgraphs are sufficient"**—This finding challenges the prior assumption that all $N$ subgraphs must be processed, indicating that structural information is highly concentrated on a small set of key nodes.
- Strong physical intuition of SC centrality: "A high number of closed walks returning to the node" implies that the node is embedded in rich local structures, making it a naturally suitable candidate for marking.
- The formal proof of complementary expressiveness solidifies the theoretical foundation—showing that the combination is not a heuristic ensemble but has mathematical guarantees for capturing distinct information.
- A parameter-free sampling component makes the method exceptionally simple and reliable—unlike RL-based sampling methods, HyMN avoids tuning hyperparameters for sampling policies.
- Inspiring for the graph expressiveness community: The connection between node centrality and subgraph importance is highly worthy of further exploration.

## Limitations & Future Work
- SC centrality assumes that denser subgraph structures are more important—this may not be optimal for sparse tree-like graphs.
- The choice of $k$ remains a hyperparameter (though 1-2 is usually sufficient).
- Dynamic/adaptive selection of $k$ (determining the required number of subgraphs based on graph structure) remains unexplored.
- The number of large-scale graph experiments is limited (only 3 datasets).
- Potential integration with newer architectures (e.g., Mamba, SSM) has not been investigated.

## Related Work & Insights
- **vs. Full-bag Subgraph GNNs**: Achieves similar performance but reduces computation by 50×, significantly advancing the Pareto frontier of efficiency and expressiveness.
- **vs. Policy-Learn/MAG-GNN**: Learnable sampling requires thousands of epochs of RL training, whereas HyMN introduces zero training overhead.
- **vs. Graph Transformers**: GTs require global attention ($O(N^2)$), whereas HyMN is more efficient ($O(k \cdot N)$).
- **Insight**: The role of node centrality in graph learning may be far from fully exploited—it can be used not only for sampling but also as positional encodings, structural priors, etc.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of walk-based centrality and subgraph sampling is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive evaluation across small/large graphs, ablations, and various centrality measures.
- Writing Quality: ⭐⭐⭐⭐⭐ An excellent combination of theory and experiments, with an elegant proof of complementary expressiveness.
- Value: ⭐⭐⭐⭐⭐ Makes subgraph methods practically viable—transforming them from theoretical concepts into practical tools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](../../AAAI2026/graph_learning/enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [\[ICML 2025\] TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation](tined_gnns-to-mlps_by_teacher_injection_and_dirichlet_energy_distillation.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](../../NeurIPS2025/graph_learning/logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [\[ICML 2025\] Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes](machines_and_mathematical_mutations_using_gnns_to_characterize_quiver_mutation_c.md)
- [\[ICML 2025\] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)

</div>

<!-- RELATED:END -->
