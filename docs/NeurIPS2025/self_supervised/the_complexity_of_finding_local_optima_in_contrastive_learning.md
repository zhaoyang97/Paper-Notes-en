---
title: >-
  [Paper Note] The Complexity of Finding Local Optima in Contrastive Learning
description: >-
  [NeurIPS 2025][Self-Supervised Learning][contrastive learning] This paper proves that finding local optima in contrastive learning is computationally hard: the discrete triplet maximization problem is PLS-hard (even when…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "contrastive learning"
  - "local optima"
  - "PLS-hard"
  - "CLS-hard"
  - "triplet loss"
  - "computational complexity"
date: 2026-05-08
content_hash: 16457dbc7e5e5e7b
---

# The Complexity of Finding Local Optima in Contrastive Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.16898](https://arxiv.org/abs/2509.16898)
**Code**: Available (supplementary)
**Area**: Computational Complexity / Contrastive Learning
**Keywords**: contrastive learning, local optima, PLS-hard, CLS-hard, triplet loss, computational complexity

## TL;DR
This paper proves that finding local optima in contrastive learning is computationally hard: the discrete triplet maximization problem is PLS-hard (even when $d=1$), and continuous triplet loss minimization is CLS-hard, implying that (under standard assumptions) no polynomial-time algorithm exists for finding local optima.

## Background & Motivation
**Background**: Contrastive learning learns an embedding space via triplet constraints $(x, y, z)$ (requiring $x$ to be closer to $y$ than to $z$). In practice, gradient descent and local search find good solutions with consistent effectiveness.

**Limitations of Prior Work**: Despite the empirical success of contrastive learning, the computational complexity of its optimization landscape is entirely unexplored. While finding a global optimum is intuitively NP-hard, whether finding a *local* optimum is tractable remains an open question. The PLS (Polynomial Local Search) and CLS (Continuous Local Search) frameworks provide the appropriate complexity-theoretic characterization.

**Key Challenge**: Local search is the practical optimization strategy for contrastive learning. If finding a local optimum is itself intractable, why does local search succeed in practice? Do hard instances exist that require exponentially many steps?

**Goal**: To determine the precise complexity class (PLS/CLS-hardness) of local search in contrastive learning.

**Key Insight**: Starting from classical combinatorial optimization problems (LocalMaxCut, QuadraticProgram-KKT), the paper constructs polynomial reductions to contrastive learning problems using geometric gadgets (boundary points, isosceles triangles, regular simplices) to encode graph structure.

**Core Idea**: A carefully designed geometric encoding reduces MaxCut local search to contrastive triplet optimization, proving that local search may require exponential time even in one-dimensional embeddings.

## Method

### Overall Architecture
Three classes of contrastive learning problems are defined, and PLS/CLS-hardness is established for each:
- **Triplet Maximization (discrete)**: Given a triplet set $\{(x_i, y_i, z_i)\}$, find an embedding maximizing the total weight of satisfied triplets.
- **Tree Triplet Consistency**: Map points to leaves of a tree such that triplet constraints are satisfied.
- **Triplet Loss Minimization (continuous)**: Minimize $\sum w_i \max\{\|f(x_i)-f(y_i)\|^2 - \|f(x_i)-f(z_i)\|^2 + \alpha, 0\}$.

### Key Designs

1. **PLS Reduction (Discrete Triplet Maximization)**:

    - **Function**: Reduction from LocalMaxCut to LocalContrastive-Euclidean.
    - **Mechanism**: Each edge $(u,v)$ in the graph is encoded as a triplet constraint. For $d=1$: vertices are placed on a line segment, and a "boundary point" $X$ forces vertices to one of two sides (encoding the two sets of a MaxCut partition). "Heavy" constraints fix the positions of special vertices.
    - For $d=2$: isosceles triangles force vertices onto two rays.
    - For higher $d$: special vertices $X_1,\ldots,X_n$ of a regular simplex are used.
    - **Key Finding**: **The problem is PLS-hard even for $d=1$ (line embeddings).**

2. **CLS Reduction (Continuous Triplet Loss)**:

    - **Function**: Reduction from QuadraticProgram-KKT to LocalTripletLoss-Euclidean.
    - **Mechanism**: Weighted triplet constraints encode a quadratic function such that KKT points correspond to local optima of the triplet loss.
    - **Key Finding**: **The continuous case is CLS-hard ($d \geq 1$).**

3. **Tree Triplet Consistency**:

    - **Function**: Proves that local search for mapping points to tree leaves satisfying triplet constraints is also PLS-hard.
    - Connects to practical problems in phylogenetic tree inference and hierarchical clustering.

### Loss & Training
This is a purely theoretical work; no training is involved.

## Key Experimental Results

### Main Results (Hard Instance Verification)

| Problem Type | Complexity | Dimension | Key Finding |
|---|---|---|---|
| Discrete Triplet Maximization | PLS-hard | $d \geq 1$ | Hard even for $d=1$ |
| Tree Triplet Consistency | PLS-hard | Tree embedding | Related to phylogenetic tree inference |
| Continuous Triplet Loss | CLS-hard | $d \geq 1$ | Covers mainstream triplet loss |

### Ablation Study (Algorithm Verification)

| Instance Source | Local Search Steps | Notes |
|---|---|---|
| Monien & Tscheuschner MaxCut hard instances | Exponential | Exponential behavior preserved after reduction |
| Random initialization | Typically fast convergence | Worst-case vs. average-case gap |

### Key Findings
- PLS-hardness holds even in the simplest $d=1$ setting—dimensionality reduction does not circumvent the difficulty.
- Constructed hard instances exist for which local search requires exponentially many steps.
- However, random initialization typically converges quickly, explaining the empirical success of contrastive learning.
- Results extend to approximate local optima.

## Highlights & Insights
- **Surprising result for $d=1$**: Finding local optima in contrastive learning with one-dimensional line embeddings is already PLS-hard, a significantly stronger result than intuition suggests (low-dimensional problems are generally expected to be easier). The geometric gadget design (boundary points + heavy constraints) is particularly elegant.
- **Reconciling theory and practice**: PLS-hardness is a worst-case result. Random initialization converges quickly even on constructed hard instances, indicating that the optimization landscape encountered in practice differs substantially from the worst case. This explains why contrastive learning succeeds empirically.
- **Applicability to practical loss functions**: The CLS-hardness of continuous triplet loss applies directly to the widely used triplet loss with margin formulation.

## Limitations & Future Work
- Only worst-case analysis is provided; complexity under average-case and smoothed analysis settings remains open.
- The existence of efficient algorithms on practical data distributions is not ruled out.
- The *quality* of local optima is not analyzed—even if a local optimum is found, how good is it?
- The effect of initialization strategies (e.g., informed initialization exploiting data structure) on complexity is unexplored.
- Triplet weights in the reduction constructions can be very large, differing from the uniform weights used in practice.

## Related Work & Insights
- **vs. MaxCut local search (Schäffer & Yannakakis, 1991)**: The PLS-completeness of MaxCut local search is a classical result; this paper extends it to contrastive learning embedding spaces.
- **vs. CLS (Daskalakis et al., 2011)**: This work constitutes the first application of CLS-hardness to a practical ML problem.
- **vs. NP-hardness of contrastive learning (Eppstein, 2004)**: Prior work established only that finding a global optimum is NP-hard; this paper resolves the finer question of *local* search complexity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First hardness result for local search in contrastive learning; $d=1$ finding is unexpected.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical with hard-instance verification; no experiments on real ML datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Proof intuitions are clear and reduction constructions are elegant.
- Value: ⭐⭐⭐⭐ A foundational theoretical contribution explaining the optimization landscape of contrastive learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](contrastive_representations_for_temporal_reasoning.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)
- [\[CVPR 2026\] UniGeoCLIP: Unified Geospatial Contrastive Learning](../../CVPR2026/self_supervised/unigeoclip_geospatial_contrastive.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](../../ICLR2026/self_supervised/maximizing_incremental_information_entropy_for_contrastive_learning.md)
- [\[ICLR 2026\] Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective](../../ICLR2026/self_supervised/difficult_examples_hurt_unsupervised_contrastive_learning_a_theoretical_perspect.md)

</div>

<!-- RELATED:END -->
