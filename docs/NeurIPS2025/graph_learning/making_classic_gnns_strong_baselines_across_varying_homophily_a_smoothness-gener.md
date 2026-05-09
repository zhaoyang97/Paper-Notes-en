---
title: >-
  [Paper Note] Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective
description: >-
  [NeurIPS 2025][Graph Learning][GNN] This paper theoretically reveals the smoothness-generalization dilemma inherent in GNN message passing, and proposes the IGNN framework with three minimal design principles — separative neighborhood transformation, inceptive aggregation, and neighborhood relationship learning — to systematically alleviate this dilemma. IGNN achieves top performance among 30 baselines and demonstrates universality across both homophilic and heterophilic graphs.
tags:
  - NeurIPS 2025
  - Graph Learning
  - GNN
  - Homophily
  - Heterophily
  - Smoothness-Generalization Dilemma
  - IGNN
date: 2026-05-08
content_hash: 470e24662fb4ae73
---

# Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective

**Conference**: NeurIPS 2025
**arXiv**: [2412.09805](https://arxiv.org/abs/2412.09805)
**Code**: [https://github.com/galogm/IGNN](https://github.com/galogm/IGNN)
**Area**: Graph Learning / Node Classification
**Keywords**: GNN, Homophily, Heterophily, Smoothness-Generalization Dilemma, IGNN

## TL;DR
This paper theoretically reveals the smoothness-generalization dilemma inherent in GNN message passing, and proposes the IGNN framework with three minimal design principles — separative neighborhood transformation, inceptive aggregation, and neighborhood relationship learning — to systematically alleviate this dilemma. IGNN achieves top performance among 30 baselines and demonstrates universality across both homophilic and heterophilic graphs.

## Background & Motivation

**Background**: GNNs are broadly categorized into homophilic GNNs (suited for graphs where connected nodes share similar labels) and heterophilic GNNs (suited for graphs where connected nodes differ in labels). In practice, graph homophily lies on a continuous spectrum rather than a binary distinction — homophily varies substantially across different hops and different nodes within the same graph.

**Limitations of Prior Work**: (a) Empirical observations suggest that homophilic GNNs can perform competitively on heterophilic graphs with tuning, yet no theoretical explanation exists; (b) existing heterophilic GNN designs employ complex modules to handle homophilic and heterophilic components separately, but separating the two inherently requires label information — creating a paradox; (c) the relationships among oversmoothing, heterophily, and generalization have been studied pairwise, but a unified theoretical framework is lacking.

**Key Challenge**: As the number of message-passing layers increases, smoothness (representation convergence) inevitably strengthens while generalization (ability to handle distributional shift) correspondingly degrades. This is particularly detrimental in high-order homophilic neighborhoods and all heterophilic neighborhoods.

**Goal**: (1) Provide a unified theoretical understanding of the common root cause underlying oversmoothing, poor generalization, and heterophily failure; (2) design minimal modifications that elevate classic GCN to a universal strong baseline.

**Key Insight**: The smoothness-generalization dilemma is formalized via Lipschitz constants and the distance to a subspace $\mathcal{M}$, from which principled design guidelines are derived.

**Core Idea**: Smoothness and generalization represent an inevitable trade-off in GNN message passing; this trade-off can be systematically alleviated through separative hop-wise transformation, inceptive aggregation, and neighborhood relationship learning.

## Method

### Overall Architecture
IGNN (Inceptive GNN) is built upon classic GCN using three minimal design principles:
- **SN** (Separative Neighborhood Transformation): applies independent transformation matrices for each hop
- **IN** (Inceptive Neighborhood Aggregation): learns multiple receptive fields in parallel
- **NR** (Neighborhood Relationship Learning): learns an adaptive weighted combination of per-hop outputs

### Key Designs

1. **Smoothness-Generalization Dilemma (Theorem 4.1)**:

   - Function: Theoretically analyzes the upper bound on representation distances produced by $k$-layer GCN message passing
   - Core formula: $d_{\mathcal{M}}(\mathbf{H}_G^{(k)}) \leq \hat{L}_G \lambda^k \mathcal{D}$
   - Here $\hat{L}_G$ is the Lipschitz constant (larger values imply worse generalization), $\lambda < 1$ is the second-largest eigenvalue of the normalized adjacency matrix, and $k$ is the number of layers
   - Key insight: As $\lambda^k \to 0$, $\hat{L}_G$ must increase to prevent representational collapse, yet a large $\hat{L}_G$ implies poor generalization — constituting the dilemma

2. **Separative Neighborhood Transformation (SN)**:

   - Function: Assigns independent weight matrices $\mathbf{W}^{(k)}$ to neighborhoods at different hops
   - Design Motivation: Shared transformation matrices couple the generalization capacities of different hops; separation allows each hop to independently control its Lipschitz constant, enabling hop-wise generalization

3. **Inceptive Neighborhood Aggregation (IN)**:

   - Function: Processes aggregation results from different hops in parallel, analogous to the Inception architecture
   - Mechanism: Per-hop outputs $\mathbf{H}^{(1)}, \mathbf{H}^{(2)}, \ldots, \mathbf{H}^{(K)}$ are computed independently from the input features via $k$-step aggregation, rather than through serial stacking
   - Design Motivation: Avoids the cumulative smoothness introduced by sequential layer stacking

4. **Neighborhood Relationship Learning (NR)**:

   - Function: Learns an adaptive weighted combination of per-hop outputs
   - Mechanism: Per-hop representations are weighted and summed via learnable coefficients $\alpha_k$, reflecting the importance of each hop
   - Design Motivation: IN combined with NR can approximate arbitrary graph filters, enabling adaptive smoothness control

### Loss & Training
- Standard cross-entropy loss for node classification
- SN is theoretically shown to confer independent hop-wise generalization capacity
- IN + NR is shown to be equivalent to learning polynomial graph filter coefficients

## Key Experimental Results

### Main Results

| Dataset Type | IGNN vs. 30 Baselines |
|---|---|
| Homophilic graphs (Cora, CiteSeer, PubMed, …) | SOTA or near-SOTA |
| Heterophilic graphs (Roman-Empire, Amazon-Ratings, …) | SOTA |
| Large-scale graphs (ogbn-arxiv, ogbn-proteins) | Strongly competitive |

### Ablation Study

| Configuration | Performance |
|---|---|
| GCN + SN only | Significant generalization improvement |
| GCN + IN only | Adaptive smoothness improvement |
| GCN + NR only | Increased filter flexibility |
| GCN + SN + IN + NR (IGNN) | Best overall performance |

### Key Findings
- Each of the three design principles contributes independently, with their combination yielding the best results
- Several existing homophilic GNNs (e.g., GCN+JK) are shown to implicitly alleviate parts of the dilemma, explaining their competitiveness on heterophilic graphs
- IGNN achieves universality without any heterophily-specific modules

## Highlights & Insights
- **Elegant unified theoretical framework**: The smoothness-generalization dilemma unifies three seemingly independent phenomena — oversmoothing, heterophily failure, and generalization gap — under a single conceptual lens.
- **Minimal modification principle**: All three design principles constitute lightweight modifications to classic GCN without introducing complex structures, embodying the spirit of Occam's razor.
- **A notable finding**: Certain homophilic GNNs inherently possess cross-graph universality (e.g., JK-Net implicitly realizes IN), providing a theoretical explanation for why well-tuned homophilic GNNs can generalize to heterophilic settings.

## Limitations & Future Work
- **Theory based on linear GCN**: Theorem 4.1 is derived for linear GCN; its applicability to GNNs with nonlinear activations requires further validation
- **Computational overhead**: IN requires parallel computation of multi-hop outputs, and independent weight matrices per hop increase the parameter count
- **Future directions**: Applying IGNN principles to other architectures such as GAT and GraphSAGE; integration with graph structure learning

## Related Work & Insights
- **vs. HeteroGNNs (e.g., H2GCN, LINKX)**: These methods design heterophily-specific modules, whereas IGNN achieves universality without any such specialized components
- **vs. JK-Net**: JK-Net's jumping knowledge mechanism implicitly realizes IN; this work provides a theoretical justification for its cross-graph generality
- **vs. Oversmoothing research**: Prior oversmoothing studies focus solely on smoothness; this work introduces the complementary generalization perspective

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified theoretical framework connecting smoothness, generalization, and homophily
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive benchmarking against 30 baselines
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations and concise design principles
- Value: ⭐⭐⭐⭐⭐ Significant implications for GNN design; unifies multiple long-standing debates

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](../../AAAI2026/graph_learning/beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[NeurIPS 2025\] What Expressivity Theory Misses: Message Passing Complexity for GNNs](what_expressivity_theory_misses_message_passing_complexity_for_gnns.md)
- [\[NeurIPS 2025\] Moscat: Mixture of Scope Experts at Test for Generalizing Deeper GNNs](mixture_of_scope_experts_at_test_generalizing_deeper_graph_neural_networks_with_.md)
- [\[ICLR 2026\] A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](../../ICLR2026/graph_learning/a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)
- [\[ICLR 2026\] GraphUniverse: Synthetic Graph Generation for Evaluating Inductive Generalization](../../ICLR2026/graph_learning/graphuniverse_synthetic_graph_generation_for_evaluating_inductive_generalization.md)

</div>

<!-- RELATED:END -->
