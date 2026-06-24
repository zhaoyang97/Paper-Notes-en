---
title: >-
  [Paper Note] Diss-l-ECT: Dissecting Graph Data with Local Euler Characteristic Transforms
description: >-
  [ICML2025][Graph Learning][Euler Characteristic Transform] This paper proposes the Local Euler Characteristic Transform ($\ell$-ECT), extending classical ECT topological invariants to the local neighborhoods of graphs to generate lossless topological-geometric fingerprints for each node. It outperforms standard GNNs on node classification tasks, particularly on highly heterophilous graphs, while providing theoretical invertibility guarantees and interpretability.
tags:
  - "ICML2025"
  - "Graph Learning"
  - "Euler Characteristic Transform"
  - "Local Topological Invariants"
  - "Node Classification"
  - "Heterophilous Graphs"
  - "Interpretability"
date: 2026-05-08
content_hash: cae005d8410f5750
---

# Diss-l-ECT: Dissecting Graph Data with Local Euler Characteristic Transforms

**Conference**: ICML2025  
**arXiv**: [2410.02622](https://arxiv.org/abs/2410.02622)  
**Area**: Graph Topology / Graph Representation Learning  
**Keywords**: Euler Characteristic Transform, Local Topological Invariants, Node Classification, Heterophilous Graphs, Interpretability

## TL;DR

This paper proposes the Local Euler Characteristic Transform ($\ell$-ECT), extending classical ECT topological invariants to the local neighborhoods of graphs to generate lossless topological-geometric fingerprints for each node. It outperforms standard GNNs on node classification tasks, particularly on highly heterophilous graphs, while providing theoretical invertibility guarantees and interpretability.

## Background & Motivation

**Core Problem**: Traditional GNNs rely on message passing to aggregate neighbor features, which loses critical local structural details during the information diffusion process, and performs poorly especially on highly heterophilous graphs.

**Limitations of Prior Work**:

- The expressive power of classical GNNs such as GCN, GAT, and GIN is bounded by the 1-WL test, rendering them unable to count induced subgraphs with $\geq 3$ nodes.
- Architectures specifically designed for heterophily (e.g., H2GCN) are effective in specific scenarios but lack generality.
- Although Persistent Homology is highly expressive, it suffers from prohibitive computational costs.

**Design Motivation**: The Euler Characteristic Transform (ECT) is an efficiently computable geometric-topological invariant that has been proven to possess invertibility for data embedded in $\mathbb{R}^n$ (i.e., the original data can be reconstructed from the ECT). The authors generalize it to a **local version** to characterize the complete topological-geometric information of each node's neighborhood in the graph, thereby bypassing the inherent limitations of message passing.

## Method

### 1. Euler Characteristic and ECT Foundations

For a simplicial complex $K$, the Euler characteristic is defined as the alternating sum of the number of simplices in each dimension:

$$\chi(K) = \sum_{k=0}^{d} (-1)^k \sigma_k(K)$$

where $\sigma_k(K)$ is the number of $k$-dimensional simplices. The ECT generalizes this to a directional sweep function:

$$\text{ECT}(X)(v, t) := \chi(\{x \in X \mid x \cdot v \leq t\})$$

which filters the sub-level set along direction $v \in S^{n-1}$ with a threshold $t$ and records its Euler characteristic. The ECT possesses **invertibility** for constructible sets embedded in $\mathbb{R}^n$—the original data can be fully reconstructed from the ECT.

### 2. Core Definition of Local ECT (ℓ-ECT)

Given a vertex $x$ in a geometric simplicial complex $X \subset \mathbb{R}^n$, its $k$-local ECT is defined as:

$$\ell\text{-ECT}_k(x; X) := \text{ECT}(N_k(x; X))$$

where $N_k(x; X)$ is the full subcomplex induced by the $k$-hop neighborhood of $x$. In practical computation, by uniformly sampling $m$ directions $v_1, \dots, v_m$ on $S^{n-1}$ and $l$ filtration thresholds $t_1, \dots, t_l$, the ℓ-ECT is approximated as an $m \cdot l$-dimensional vector. The paper defaults to $m = l = 64$, meaning each node obtains a 4096-dimensional topological fingerprint.

### 3. Theoretical Guarantees

**Theorem 1 (Inclusion of Message Passing Information)**: For an attributed graph $\mathcal{G}$, the set of 1-hop local ECTs $\{\ell\text{-ECT}_1(x; \mathcal{G})\}_x$ contains all (non-learnable) information required to perform one step of message passing—the feature vectors of each node's 1-hop neighbors can be reconstructed from it.

**Theorem 2 (Graph Isomorphism Distinction)**: Two attributed graphs $\mathcal{G}_1, \mathcal{G}_2$ are isomorphic if and only if $\text{ECT}(\mathcal{G}_1) = \text{ECT}(\mathcal{G}_2)$.

**Corollary 1 (Subgraph Counting)**: The ECT method can perform subgraph counting, whereas message-passing GNNs cannot do so for connected substructures with $\geq 3$ nodes.

### 4. Rotation-Invariant Metric

The ECT is invariant under translation and scaling, but sensitive to rotation. To address this, a rotation-invariant metric is constructed:

$$d_{\text{ECT}}(X, Y) := \inf_{\rho \in \text{SO}(n)} \|\text{ECT}(X) - \text{ECT}(\rho Y)\|_\infty$$

**Theorem 3** proves that $d_{\text{ECT}}$ constitutes a metric on the rotation equivalence classes of finite simplicial complexes. This metric can be used for spatial alignment of graph data.

### 5. Overall Workflow

1. For each node $x$ in the graph, extract the $k$-hop neighborhood subgraph $N_k(x; \mathcal{G})$.
2. Compute the approximate ECT of this subgraph to obtain an $m \cdot l$-dimensional vector.
3. Concatenate the ℓ-ECT vector with the original node features to serve as the input for the downstream classifier.
4. Use XGBoost for node classification (which can also be replaced by any other classifier).

## Key Experimental Results

Compared with GCN, GAT, GIN, GraphSAGE, and H2GCN on 10 node classification benchmarks, averaged over 5 runs.

### Main Results

#### WebKB Heterophilous Graphs (Highly Heterophilous, Core Results)

| Model | Cornell | Wisconsin | Texas |
|------|---------|-----------|-------|
| GCN | 45.0 ± 2.2 | 44.2 ± 2.6 | 47.3 ± 1.5 |
| GAT | 44.7 ± 2.9 | 48.2 ± 2.0 | 51.7 ± 3.2 |
| GraphSAGE | 76.0 ± 3.5 | 72.9 ± 1.9 | 71.8 ± 2.4 |
| H2GCN | 66.2 ± 3.5 | 70.2 ± 2.3 | 72.3 ± 3.0 |
| **ℓ-ECT₁+ℓ-ECT₂** | **67.1 ± 4.1** | **78.5 ± 2.6** | **74.8 ± 3.1** |

#### Heterophilous Graphs Benchmark

| Model | Amazon Ratings | Roman Empire |
|------|---------------|--------------|
| GAT | 44.6 ± 0.9 | 76.4 ± 1.2 |
| H2GCN | 40.1 ± 0.7 | 64.2 ± 0.9 |
| **ℓ-ECT₁+ℓ-ECT₂** | **49.8 ± 0.3** | **81.1 ± 0.4** |

#### Planetoid Homophilous Graphs

| Model | Cora | CiteSeer | PubMed |
|------|------|----------|--------|
| GAT | 88.3 ± 1.1 | 75.3 ± 1.5 | 85.7 ± 4.2 |
| GCN | 88.1 ± 1.2 | 74.6 ± 1.5 | 85.3 ± 4.7 |
| **ℓ-ECT₁+ℓ-ECT₂** | 87.8 ± 0.6 | 72.5 ± 0.7 | **90.3 ± 0.5** |

**Key Findings**: ℓ-ECT₁+ℓ-ECT₂ ranks 2nd on average in the critical difference diagram, and even the weakest ℓ-ECT₂ outperforms all non-ℓ-ECT methods (including GAT). It achieves performance close to GNNs on homophilous graphs and substantially leads on heterophilous graphs.

## Highlights & Insights

1. **Theoretical Elegance**: Generalizes the ECT invertibility theorem from algebraic topology to the local neighborhoods of graphs, proving that ℓ-ECT strictly contains the information of one-step message passing and can perform subgraph counting that standard GNNs cannot achieve.
2. **Architecture-Agnostic**: ℓ-ECT generates fixed-dimensional vector representations that can be fed into any downstream model (e.g., XGBoost, MLP, SVM), eliminating the need for GNN-specific training techniques.
3. **Heterophily-Friendly**: Does not rely on neighbor feature aggregation, naturally fitting heterophilous graph scenarios where node labels differ from neighbor labels.
4. **Interpretability**: Feature importance based on XGBoost can be traced back to specific directions and thresholds of the ECT, providing topological-level explanations for decisions.
5. **Rotation-Invariant Metric**: Proposes the $d_{\text{ECT}}$ metric, which can be efficiently approximated, providing a geometric tool for the spatial alignment of graph data.

## Limitations & Future Work

1. **Computational Complexity**: ℓ-ECT requires extracting $k$-hop subgraphs and computing ECTs for every node. The overhead increases significantly when $k$ is large or when the graph is dense. The paper acknowledges scalability challenges on large-scale graphs.
2. **Suboptimal performance on Homophilous Graphs**: It is slightly inferior to GCN/GAT on homophilous graphs such as Cora/CiteSeer, where the global information aggregation of message passing still holds advantages.
3. **Limitations in Specific Heterophilous Scenarios**: It is outperformed by H2GCN on Chameleon/Squirrel, indicating that ℓ-ECT is not a silver bullet for all heterophilous graphs.
4. **Dependence on Downstream Models**: The paper primarily utilizes XGBoost, lacking deep integration with end-to-end differentiable architectures (e.g., embedding the ℓ-ECT layer directly within GNNs).
5. **High-Dimensional Feature Graphs**: When the node feature dimension is extremely high (e.g., the Actor dataset), the incremental information provided by the 4096-dimensional topological vector of ℓ-ECT relative to the original features may be limited.

## Related Work & Insights

- **Foundations of ECT Invertibility**: Ghrist et al. (2018), Curry et al. (2022) prove the invertibility of ECT on constructible sets.
- **ECT and Deep Learning**: Röell & Rieck (2024) first integrated ECT into depth learning frameworks.
- **Persistent Homology Methods**: Rieck et al. (2019), Zhao et al. (2020) utilize persistent homology for graph learning, albeit with higher computational costs.
- **Heterophilous Graph Methods**: H2GCN (Zhu et al., 2020), GloGNN (Luan et al., 2022).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to generalize ECT to a local version for graph learning, featuring an elegant theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 10 datasets and various baselines, but lacks large-scale graph experiments.
- Writing Quality: ⭐⭐⭐⭐ — Mathematically rigorous, with clearly organized theory and experiments.
- Value: ⭐⭐⭐⭐ — Opens new directions for incorporating topological methods into graph learning, though scalability remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LEAP: Local ECT-Based Learnable Positional Encodings for Graphs](../../ICLR2026/graph_learning/leap_local_ect-based_learnable_positional_encodings_for_graphs.md)
- [\[ICML 2025\] Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis](does_graph_prompt_work_a_data_operation_perspective_with_theoretical_analysis.md)
- [\[ICML 2025\] GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers](grokformer_graph_fourier_kolmogorov-arnold_transformers.md)
- [\[ICML 2025\] Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification](mitigating_over-squashing_in_graph_neural_networks_by_spectrum-preserving_sparsi.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](../../NeurIPS2025/graph_learning/duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)

</div>

<!-- RELATED:END -->
