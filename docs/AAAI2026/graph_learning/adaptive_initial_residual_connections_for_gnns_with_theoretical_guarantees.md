---
title: >-
  [Paper Note] Adaptive Initial Residual Connections for GNNs with Theoretical Guarantees
description: >-
  [AAAI 2026][Graph Learning][Adaptive residual connections] This paper proposes Adaptive Initial Residual Connections (Adaptive IRC), which allows each node to have a personalized residual strength learned from its initial features. It provides the first theoretical proof of a positive lower bound on the Dirichlet energy of initial residual connections with activation functions (guaranteeing the absence of over-smoothing), and introduces a PageRank-based heuristic variant that…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Adaptive residual connections"
  - "over-smoothing"
  - "Dirichlet energy"
  - "PageRank"
  - "heterophilic graphs"
  - "GNN depth"
date: 2026-05-08
content_hash: 0394794441cbd343
---

# Adaptive Initial Residual Connections for GNNs with Theoretical Guarantees

**Conference**: AAAI 2026
**arXiv**: [2511.06598](https://arxiv.org/abs/2511.06598)  
**Code**: [Adaptive-IRC](https://rb.gy/dlgx42)  
**Area**: Graph Neural Networks
**Keywords**: Adaptive residual connections, over-smoothing, Dirichlet energy, PageRank, heterophilic graphs, GNN depth

## TL;DR

This paper proposes Adaptive Initial Residual Connections (Adaptive IRC), which allows each node to have a personalized residual strength learned from its initial features. It provides the first theoretical proof of a positive lower bound on the Dirichlet energy of initial residual connections with activation functions (guaranteeing the absence of over-smoothing), and introduces a PageRank-based heuristic variant that achieves comparable or superior performance without learning additional parameters.

## Background & Motivation

**Background**: The core of GNNs is message passing—nodes update their embeddings by aggregating neighborhood information. However, deep GNNs suffer from over-smoothing: repeated neighborhood averaging causes all node embeddings to converge to an indistinguishable state.

**Limitations of Prior Work**: (1) Static IRC methods (e.g., GCNII) use a shared, fixed residual strength and cannot differentiate across nodes; (2) existing theoretical guarantees are restricted to the linear (activation-free) setting; (3) existing adaptive residual methods are complex and lack theoretical guarantees.

**Key Challenge**: A mechanism is needed that is both theoretically grounded and capable of adaptively modulating residual strength.

**Key Insight**: Inspired by the Friedkin–Johnsen opinion dynamics model, in which individuals differ in their susceptibility to external information.

**Core Idea**: Node-level personalized residual strength + theoretical guarantee via a positive lower bound on Dirichlet energy + a PageRank-based heuristic requiring zero additional parameters.

## Method

### Overall Architecture

The Adaptive IRC message-passing update is:
$$H^{(\ell+1)} = \sigma\!\left(\Lambda \mathcal{A} H^{(\ell)} W^{(\ell)} + (I - \Lambda) H^{(0)} \Theta^{(\ell)}\right),$$
where $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$ denotes node-level residual strengths.

### Key Designs

1. **Parameterization of Adaptive Residual Strength**
    - **Function**: Generates personalized residual weights for each node.
    - **Mechanism**: $\Lambda = \text{diag}(\sigma(H^{(0)} W_{\text{att}}))$, where sigmoid ensures outputs lie in $(0,1)$.
    - **Design Motivation**: Weights derived from initial features generalize to unseen nodes; sharing weights across layers reduces parameter count.

2. **Proof of a Positive Lower Bound on Dirichlet Energy (Theorem 2)**
    - **Core Result**: $\mathcal{E}(H^{(\ell+1)}) \geq \dfrac{\zeta \bar{\sigma}_r(\Theta)}{1 - \eta \bar{\sigma}_r} \mathcal{E}(H^{(0)}) > 0$
    - **Key Quantities**: $\eta = \alpha^2 \lambda_{\min}^2 \sigma_r^2(\mathcal{A})$, $\zeta = \alpha^2 (1 - \lambda_{\max})^2$
    - **Proof Outline**: Lemma 1 (energy lower bound for weight matrices) + Lemma 2 (energy lower bound for the adjacency operation) → Corollary 1 → energy preservation under Leaky ReLU → recursive unrolling to obtain the convergence lower bound.
    - **Significance**: First theoretical guarantee of over-smoothing mitigation for nonlinear IRC.

3. **PageRank-Based Heuristic Variant**
    - **Function**: Replaces learned residual strengths with PageRank scores.
    - **Mechanism**: The top-$k$% of nodes by PageRank are assigned $\lambda_{\max}$; the remainder are assigned $\lambda_{\min}$.
    - **Design Motivation**: Node centrality is positively correlated with optimal residual strength.
    - **Advantage**: Requires no additional learned parameters while achieving comparable or superior performance.

### Complexity

$O(|E|d + nd^2)$ per layer, identical to vanilla GCN.

## Key Experimental Results

### Node Classification — Comparison with SOTA

| Method | Cora (H:0.83) | Texas (H:0.11) | Wisconsin (H:0.21) | Chameleon (H:0.23) | Squirrel (H:0.22) |
|--------|---------------|----------------|--------------------|--------------------|-------------------|
| GCN | 79.2±0.4 | 55.9±6.4 | 47.1±8.5 | 33.4±2.2 | 27.2±0.7 |
| GCNII | 79.9±0.5 | 59.5±5.3 | 60.4±7.4 | 36.2±2.7 | 28.8±1.0 |
| DirGNN | 77.5±1.2 | 84.6±6.1 | 82.2±2.3 | 60.6±2.2 | 45.3±1.5 |
| **IRC (Learned)** | **80.1±1.0** | 73.0±5.8 | **82.4±4.7** | **64.1±1.1** | **47.7±2.2** |
| **IRC (PageRank)** | **80.7±0.4** | **77.0±6.8** | 79.0±3.3 | **65.0±2.0** | **49.0±2.2** |

### Improvements on Heterophilic Graphs (vs. GCNII)

| Dataset | Texas | Wisconsin | Cornell | Chameleon | Squirrel |
|---------|-------|-----------|---------|-----------|----------|
| Gain | +17.5% | +18.6% | +25.4% | +28.8% | +20.2% |

### Over-Smoothing Mitigation

- The Dirichlet energy of GCN/GAT/GraphSAGE decays exponentially with depth, approaching zero.
- Adaptive IRC maintains positive energy, remaining stable at 16 layers.
- Both variants sustain consistently high performance as depth increases beyond 6 layers.

### Key Findings

- The PageRank variant performs comparably to or better than the learned variant, indicating that the heuristic is sufficient.
- Improvements are largest on heterophilic graphs (+17–29%), as adaptive residuals can distinguish between similar and dissimilar neighbors.
- The proposed method outperforms all baselines on all datasets except Actor.

## Highlights & Insights

1. **Theorem 2 is rigorous**: the first proof of a positive lower bound on Dirichlet energy for nonlinear IRC.
2. **The PageRank variant is surprisingly effective**: centrality-based heuristics provide adaptive capability almost for free.
3. **Opinion dynamics analogy**: GNN message passing ↔ opinion propagation in social networks.
4. **Rank preservation (Theorem 1)**: in the simplified setting, the rank of the embedding matrix is fully preserved.

## Limitations & Future Work

1. The PageRank threshold and $\lambda$ values still require tuning.
2. Theorem 2 relies on a cross-layer positive alignment assumption (Property 2).
3. Only node classification is evaluated; graph classification remains untested.
4. Comparison with more recent GNN methods (e.g., GREAD) is absent.
5. Leaky ReLU is a necessary condition, restricting the choice of activation function.

## Related Work & Insights

- The Friedkin–Johnsen model demonstrates that sociological models can provide useful inductive biases for GNN architecture design.
- The success of the PageRank variant suggests that graph-topological priors are underutilized.
- Dirichlet energy analysis is broadly applicable for evaluating any novel message-passing mechanism.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: The theoretical proof and the PageRank variant represent significant contributions.
- **Experimental Thoroughness** ⭐⭐⭐⭐: 9 datasets, in-depth analysis, and energy visualization.
- **Writing Quality** ⭐⭐⭐⭐: Theoretical derivations are clearly presented.
- **Value** ⭐⭐⭐⭐: Provides a theoretically grounded, lightweight solution for deep GNNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [\[AAAI 2026\] Logical Characterizations of GNNs with Mean Aggregation](logical_characterizations_of_gnns_with_mean_aggregation.md)
- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [\[ACL 2026\] Overcoming the Impedance Mismatch: A Theoretical Roadmap for Fusing Foundation Models and Knowledge Graphs](../../ACL2026/graph_learning/overcoming_the_impedance_mismatch_a_theoretical_roadmap_for_fusing_foundation_mo.md)

</div>

<!-- RELATED:END -->
