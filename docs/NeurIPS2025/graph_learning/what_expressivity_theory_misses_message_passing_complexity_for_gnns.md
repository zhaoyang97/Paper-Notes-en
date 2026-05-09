---
title: >-
  [Paper Note] What Expressivity Theory Misses: Message Passing Complexity for GNNs
description: >-
  [NeurIPS 2025][Graph Learning][message passing complexity] This paper critiques the binary expressivity theory of GNNs for its inability to explain practical performance differences, and proposes MPC—a continuous, task-specific complexity measure grounded in probabilistic lossyWL—achieving a Spearman correlation of -1 with accuracy (versus ρ = 0 for classical WLC), and successfully explaining why GCN with virtual nodes outperforms higher-expressivity higher-order models on long-range tasks.
tags:
  - NeurIPS 2025
  - Graph Learning
  - message passing complexity
  - lossyWL
  - GNN expressivity
  - over-squashing
  - continuous metric
date: 2026-05-08
content_hash: 3b3e2294b4e49bff
---

# What Expressivity Theory Misses: Message Passing Complexity for GNNs

**Conference**: NeurIPS 2025
**arXiv**: [2509.01254](https://arxiv.org/abs/2509.01254)
**Code**: [https://www.cs.cit.tum.de/daml/message-passing-complexity/](https://www.cs.cit.tum.de/daml/message-passing-complexity/)
**Area**: GNN Theory / Expressivity
**Keywords**: message passing complexity, lossyWL, GNN expressivity, over-squashing, continuous metric

## TL;DR
This paper critiques the binary expressivity theory of GNNs for its inability to explain practical performance differences, and proposes MPC—a continuous, task-specific complexity measure grounded in probabilistic lossyWL—achieving a Spearman correlation of -1 with accuracy (versus ρ = 0 for classical WLC), and successfully explaining why GCN with virtual nodes outperforms higher-expressivity higher-order models on long-range tasks.

## Background & Motivation

### State of the Field

**Background**: GNN expressivity theory centers on the WL isomorphism test, with a large body of work pursuing expressivity beyond WL.

**Limitations of Prior Work**: Expressivity theory is binary (can/cannot distinguish), and thus cannot explain performance differences among models of equal expressivity; nearly all graph pairs in standard benchmarks are already distinguished by WL, so higher expressivity should yield no additional benefit.

**Key Challenge**: GCN with virtual nodes (which does not increase WL expressivity) outperforms higher-order models that surpass WL on long-range tasks—a phenomenon expressivity theory cannot explain.

**Goal**: Propose a continuous complexity framework that preserves impossibility conclusions while explaining practical performance differences.

**Key Insight**: Probabilistically relax the WL test—each message survives independently with a random-walk probability (lossyWL).

**Core Idea**: The negative log of the information retention probability under lossyWL = MPC = a task-specific, architecture-specific, continuous measure of learning difficulty.

## Method

### Overall Architecture
lossyWL is a probabilistic variant of standard WL in which each message $m_{u \to v}$ survives with probability $I_{vu}$. MPC is defined as $-\log P[\text{lossyWL}_v^L(G) \vDash f_v(G)]$.

### Key Designs

1. **lossyWL**:

    - **Function**: Models the lossy nature of message passing in practical MPNNs.
    - **Mechanism**: At each layer, each message survives independently with probability $I_{vu}$, making node colorings random variables.
    - **Design Motivation**: Standard WL assumes lossless propagation, whereas over-squashing, over-smoothing, and under-reaching cause information loss in practice.

2. **Theoretical Properties of MPC**:

    - Impossibility preservation: $MPC = \infty$ if and only if WL also fails to distinguish.
    - Function refinement: More fine-grained tasks yield higher complexity.
    - Over-squashing lower bound: $MPC \geq -\log(I^L_{vu})$.

3. **Generalization to Arbitrary Architectures**:

    - VN, rewiring, higher-order graphs, etc., are handled by executing lossyWL on the transformed message-passing graph.

## Key Experimental Results

### Feature Retention Task (Over-Smoothing Proxy)


### Main Results

| Architecture | MPC vs. Accuracy Spearman ρ | WLC ρ |
|---|---|---|
| GCN / GIN / GCN-VN / GSN | **ρ = −1** (perfect negative correlation) | **ρ = 0** |

### Long-Range Information Propagation Task


### Ablation Study

| Architecture | Expressivity | MPC | Performance |
|---|---|---|---|
| GCN | WL-equivalent | High | Poor |
| GCN-VN | WL-equivalent | **Low** | **Good** |
| CIN | Beyond WL | High | Poor |
| FragNet | Beyond WL | High | Poor |

### Key Findings
- MPC is perfectly negatively correlated with accuracy; WLC yields ρ = 0.
- GCN+VN surpasses higher-expressivity models on long-range tasks.

## Highlights & Insights
- "Pursuing higher expressivity may be misleading"—the key to success is minimizing task-specific MPC.
- The design of lossyWL is elegant: modifying a single assumption transforms a binary theory into a continuous one.

## Limitations & Future Work
- Does not model learning dynamics (gradients / optimization difficulty).
- Monte Carlo simulation may be computationally expensive for large graphs.
- Currently limited to node-level tasks.

## Related Work & Insights
- **vs. WL Expressivity Theory**: The binary theory is shown to have limited explanatory power for practical performance.
- **vs. Over-Squashing Literature**: Over-squashing provides a lower bound on MPC.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ lossyWL + MPC represents a significant advance in GNN theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across multiple tasks × architectures × ZINC.
- Writing Quality: ⭐⭐⭐⭐⭐ Arguments are sharp and theory is rigorous.
- Value: ⭐⭐⭐⭐⭐ Redefines how GNNs are understood and improved.

### Supplementary Technical Details
- The Bernoulli survival probability $I_{vu}$ in lossyWL corresponds to elements of the normalized adjacency matrix, i.e., random-walk transition probabilities.
- Impossibility theorem for MPC: $MPC = \infty$ if and only if there exist $G', w$ such that $f_v(G) \neq f_w(G')$ but $M_v(G) = M_w(G')$ for all $M \in \mathcal{M}$.
- Function refinement theorem: If $f \vDash g$ (f is more fine-grained than g), then $MPC(f_v, G) \geq MPC(g_v, G)$.
- Task triangle inequality: $MPC(f_v \| g_v, G) \leq MPC(f_v, G) + MPC(g_v, G)$.
- Validated on 3-regular random graphs; results transfer to ZINC and the Long Range Graph Benchmark.
- Complexity of the feature retention task grows at least linearly: $\mathbb{E}[MPC] \in \Omega(L)$ (Lemma retaining).
- Analysis covers 8 architectures: MLP, GCN, GIN, GraphSAGE, GCN-VN, GSN, FragNet, and CIN.
- CIN and GraphSAGE achieve perfect accuracy due to explicit residual connections—the MPC framework abstracts away such implementation details.
- Table 1 shows that WL already distinguishes nearly all graph pairs in popular benchmarks, casting doubt on the practical necessity of higher expressivity.
- MPC can be computed via Monte Carlo simulation, yielding a complexity value per (graph, node, task) tuple.
- Virtual nodes do not increase WL expressivity but reduce MPC (all nodes become reachable via VN in one hop), explaining their empirical effectiveness.
- Per-tuple MPC computation is fast; large-scale evaluation requires sampling.
- Transferability is further validated on ZINC and the Long Range Graph Benchmark.
- The MPC framework naturally generalizes to arbitrary MP graph transformations (including rewiring and higher-order graphs); see Appendix B.1.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs](../../AAAI2026/graph_learning/are_graph_transformers_necessary_efficient_long-range_messag.md)
- [\[NeurIPS 2025\] Moscat: Mixture of Scope Experts at Test for Generalizing Deeper GNNs](mixture_of_scope_experts_at_test_generalizing_deeper_graph_neural_networks_with_.md)
- [\[NeurIPS 2025\] Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Graph Persistence goes Spectral](graph_persistence_goes_spectral.md)

</div>

<!-- RELATED:END -->
