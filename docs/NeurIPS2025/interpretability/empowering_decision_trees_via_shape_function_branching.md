---
title: >-
  [Paper Note] Empowering Decision Trees via Shape Function Branching
description: >-
  [NeurIPS 2025][Interpretability][Decision Trees] This paper proposes the Shape Generalized Tree (SGT), which replaces the conventional linear threshold split at each internal node of a decision tree with a learnable axis…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Decision Trees"
  - "Shape Functions"
  - "Tabular Data"
  - "Nonlinear Splitting"
date: 2026-05-08
content_hash: 20bcf5009655694c
---

# Empowering Decision Trees via Shape Function Branching

**Conference**: NeurIPS 2025
**arXiv**: [2510.19040](https://arxiv.org/abs/2510.19040)
**Code**: Not provided
**Area**: Interpretable Machine Learning / Decision Trees
**Keywords**: Decision Trees, Shape Functions, Interpretability, Tabular Data, Nonlinear Splitting

## TL;DR

This paper proposes the Shape Generalized Tree (SGT), which replaces the conventional linear threshold split at each internal node of a decision tree with a learnable axis-aligned shape function, enabling the capture of nonlinear feature effects within more compact tree structures while preserving interpretability.

## Background & Motivation

- **Background**: Conventional axis-aligned linear decision trees perform simple $x_d \leq \theta$ splits at each node; when faced with nonlinear feature–target relationships, repeated splits on the same feature lead to deep, large trees.
- **Limitations of Prior Work**: Tree comprehensibility is highly sensitive to path depth and the number of leaf nodes; deep trees directly undermine interpretability. Generalized Additive Models (GAMs) model single-feature contributions via nonlinear shape functions with good interpretability, but lack the hierarchical structure of decision trees. Oblique trees allow splits based on linear combinations of multiple features, but high-dimensional cuts are difficult for humans to interpret.
- **Key Challenge**: There is a need for a method that captures nonlinear decision boundaries within a single node while maintaining axis-aligned interpretability.

## Core Problem

How to enhance the expressive capacity of individual decision tree nodes without sacrificing interpretability, so that shallower and more compact trees can achieve better predictive performance.

## Method

### Shape Generalized Tree (SGT) Definition

A conventional linear tree performs the following split at each node:

$$\mathcal{D}^l = \{(\mathbf{x}_n, y_n) \in \mathcal{D} \mid \mathbf{w}^\top \mathbf{x}_n \leq \theta\}$$

SGT replaces the threshold split with a shape function split:

$$\mathcal{D}^l = \{(\mathbf{x}_n, y_n) \in \mathcal{D} \mid f_\Theta(\mathbf{w}^\top \mathbf{x}_n) \leq 0.0\}$$

where $\mathbf{w}$ is a one-hot feature selection vector and $f_\Theta: \text{dom}(\mathcal{X}_d) \to \mathbb{R}$ is a learnable shape function. Since $f$ operates on a single feature, it remains directly visualizable.

### Extended Variants

- **S2GT (Bivariate Shape Function Tree)**: Each node allows splitting on a joint shape function of two features, $f^2_\Theta(x_{d_1}, x_{d_2}) \leq 0$, which can be visualized as a heatmap.
- **SGT$^K$ (Multi-way Split Tree)**: Generalizes binary splits to $K$-way splits using a vector-valued shape function $f^{(K)}: \text{dom}(\mathcal{X}_d) \to \mathbb{R}^K$, with branch selection via argmax. Experiments are restricted to $K=3$.

### Expressiveness Guarantees

**Theorem 1**: SGT is at least as expressive as a linear tree with the same number of nodes (linear trees are a special case of SGT).

**Theorem 2**: For any $B \in \mathbb{N}$, there exist functions for which a linear tree requires at least $B$ more decision nodes than an SGT.

### ShapeCART Algorithm

A CART-like top-down greedy construction is adopted, solving a bi-level optimization at each node:

$$\min_{\mathbf{w}} \sum_{d=1}^{D} w_d \min_{\Theta_d} \left[\mathcal{L}(\{\mathcal{D}^l_d, \mathcal{D}^r_d\})\right]$$

where the weighted impurity is defined as:

$$\mathcal{L}(\mathbf{D}) = \sum_{\mathcal{D} \in \mathbf{D}} |\mathcal{D}| \cdot \mathcal{H}(\Pi(\mathcal{D}))$$

**Shape function learning proceeds in two stages:**

1. **Binning**: An internal CART tree maps samples to $L$ mutually exclusive bins based on single-feature values, with each bin storing the empirical class distribution $\pi_\ell$ and weight $W_\ell$.
2. **Bin-to-Branch Mapping**: Coordinate descent solves the discrete optimization that assigns each bin to a left/right (or $K$-way) branch, minimizing weighted impurity:

$$\min_{\mathbf{a}} \sum_{k} W_k \cdot \mathcal{H}(\Pi_k)$$

The initialization strategy selects the better of weighted K-Means clustering and the root-node assignment from the internal CART tree.

### Bivariate Candidate Pair Screening

To avoid $O(D^2)$ enumeration of feature pairs, the Cartesian product of branch assignments from univariate shape functions is used to rapidly estimate interaction gain:

$$\delta_{(d_1, d_2)} = \min(\mathcal{L}(\mathbf{D}_{d_1}), \mathcal{L}(\mathbf{D}_{d_2})) - \mathcal{L}(\{\mathcal{D}^i \cap \mathcal{D}^j\})$$

Only the top-$P$ candidate pairs by $\delta$ value are retained, and a regularization penalty $\gamma$ is added to suppress unnecessary bivariate splits.

### Post-Processing

Tree Alternating Optimization (TAO) is applied to globally optimize the greedily constructed tree, refitting shape functions node by node and performing pruning.

## Key Experimental Results

Evaluated on 26 real-world classification datasets at depths 2–6.

### Mean Test Accuracy (%) — Axis-Aligned Methods

| Method | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Best |
|--------|---------|---------|---------|---------|---------|------|
| CART | 83.9 | 81.6 | 82.3 | 84.0 | 85.1 | 85.2 |
| SERDT | 83.7 | 80.9 | 81.7 | 83.7 | 85.1 | 85.1 |
| **SGT-C** | **84.5** | **82.5** | **83.7** | **84.9** | **86.2** | **86.3** |
| **SGT3-C** | **85.7** | **84.6** | **85.9** | **87.3** | **87.8** | **87.8** |
| AxTAO | 83.9 | 82.1 | 82.8 | 84.5 | 85.5 | 85.4 |
| DPDT | 84.9 | 83.4 | 83.8 | 85.2 | 86.2 | 86.2 |
| **SGT-T** | **85.0** | **83.5** | **84.6** | **85.9** | **86.8** | **86.8** |
| **SGT3-T** | **86.4** | **85.1** | **86.2** | **87.5** | **88.0** | **88.0** |

### Mean Test Accuracy (%) — Bivariate Methods

| Method | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Best |
|--------|---------|---------|---------|---------|---------|------|
| BiCART | 87.3 | 87.6 | 87.9 | 88.7 | 89.6 | 89.6 |
| BiTAO | 87.9 | 88.1 | 88.4 | 89.3 | 90.1 | 90.0 |
| **S2GT-C** | **89.3** | **89.1** | **89.5** | **90.5** | **91.3** | **91.4** |
| **S2GT3-T** | **90.2** | **90.6** | **91.2** | **91.7** | **92.4** | **91.9** |

**Key Findings**:
- SGT-C at depth 2 frequently matches or exceeds CART at depth 6 (e.g., on the *eye-movements* and *electricity* datasets).
- S2GT-C at depth 2 achieves performance comparable to BiCART/BiTAO at depth 6.
- Three-way splits (SGT3) outperform binary splits at all depths.

## Highlights & Insights

- ⭐ Shape function splitting introduces the nonlinear modeling capability of GAMs into decision tree nodes while preserving visualizability and interpretability.
- ⭐ The two-stage shape function learning procedure (CART-based binning followed by coordinate descent for branch assignment) is efficient and theoretically grounded, with an information gain lower bound no worse than CART.
- ⭐ The bivariate candidate pair screening heuristic substantially reduces computational overhead, lowering complexity from $O(D^2 \cdot NC\log N)$ to $O(P \cdot NC\log N)$.
- The performance advantage at shallow depths is particularly pronounced, directly improving interpretability in practical deployments.

## Limitations & Future Work

- As a tree-based model, SGT is primarily suited to tabular data and has limited applicability to unstructured data such as images or text.
- Understanding shape function splits imposes a greater cognitive burden than simple threshold splits.
- No human subject studies are conducted to systematically evaluate the practical comprehensibility of SGT.
- Post-processing relies on TAO global optimization, which increases training complexity.

## Related Work & Insights

| Model Category | Representative Methods | Node Expressiveness | Interpretability | Performance |
|---------------|----------------------|--------------------|--------------------|-------------|
| Axis-aligned linear trees | CART | Single-feature threshold | Highest | Baseline |
| Oblique trees | TAO-Oblique | Linear combination of multiple features | Low (high-dimensional cuts) | High |
| Bivariate oblique trees | BiCART | Linear combination of two features | Medium (visualizable) | High |
| **SGT (Ours)** | ShapeCART | Nonlinear function of a single feature | High (shape function plots) | Highest |

**Insights**:
- The combination of shape functions and decision trees yields superior interpretable expressiveness; this idea generalizes naturally to other tree-based models such as random forests and gradient-boosted trees.
- The application of coordinate descent to discrete optimization is instructive: a continuous relaxation (K-Means initialization) followed by discrete optimization (coordinate descent) offers a broadly applicable strategy.
- Parameterizing shape functions with neural networks (analogous to NAMs) is a promising direction for further performance gains on more complex feature relationships.

## Rating

- ⭐ **Novelty**: 8/10 — The idea of incorporating shape functions into decision tree nodes is natural yet innovative, with complete theoretical guarantees.
- ⭐ **Experimental Thoroughness**: 8/10 — Comprehensive evaluation across 26 datasets, multiple depths, multiple variants (SGT/S2GT/SGT$^K$), and strong baselines.
- ⭐ **Writing Quality**: 8/10 — Definitions are precise, algorithmic pseudocode is complete, and visualization examples are intuitive.
- ⭐ **Value**: 8/10 — Has direct implications for the deployment of interpretable AI in high-stakes domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [\[ICLR 2026\] Uncovering Grounding IDs: How External Cues Shape Multimodal Binding](../../ICLR2026/interpretability/uncovering_grounding_ids_how_external_cues_shape_multimodal_binding.md)
- [\[ACL 2026\] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](../../ACL2026/interpretability/forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](../../AAAI2026/interpretability/shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)

</div>

<!-- RELATED:END -->
