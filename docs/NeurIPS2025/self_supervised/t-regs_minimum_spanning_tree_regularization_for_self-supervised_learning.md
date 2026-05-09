---
title: >-
  [Paper Note] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning
description: >-
  [NeurIPS 2025][Self-Supervised Learning] This paper proposes T-REGS — a self-supervised learning regularization framework based on maximizing the length of the minimum spanning tree (MST). The authors theoretically prove that the method simultaneously prevents dimensional collapse and promotes uniform distribution of representations on compact Riemannian manifolds, with empirical validation on standard JE-SSL benchmarks.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - minimum spanning tree
  - dimensional collapse
  - uniformity
  - regularization
date: 2026-05-08
content_hash: ffe541629a474db7
---

# T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.23484](https://arxiv.org/abs/2510.23484)
**Code**: [GitHub](https://github.com/) (available)
**Area**: Self-Supervised Learning / Representation Learning / Regularization
**Keywords**: self-supervised learning, minimum spanning tree, dimensional collapse, uniformity, regularization

## TL;DR
This paper proposes T-REGS — a self-supervised learning regularization framework based on maximizing the length of the minimum spanning tree (MST). The authors theoretically prove that the method simultaneously prevents dimensional collapse and promotes uniform distribution of representations on compact Riemannian manifolds, with empirical validation on standard JE-SSL benchmarks.

## Background & Motivation

**Background**: Joint-embedding self-supervised learning (JE-SSL) learns representations by encouraging embeddings of different views of the same image to be similar. The central challenge is preventing representation collapse (all inputs mapped to the same vector) and dimensional collapse (representations confined to a low-dimensional subspace).

**Limitations of Prior Work**:
- Contrastive methods (SimCLR, MoCo) require large numbers of negative samples and large batch sizes, incurring substantial computational overhead.
- Redundancy-reduction methods (BarlowTwins, VICReg) rely solely on the covariance matrix (second-order statistics), ignoring higher-order information such as distribution mass concentration, and cannot guarantee convergence to a uniform distribution.
- Asymmetric methods (BYOL, DINO) lack theoretical explanations for why asymmetric architectures prevent collapse.
- The optimal transport regularization proposed by Fang et al. is computationally expensive, and its closed-form acceleration is only valid on the hypersphere while suffering from numerical instability.

**Key Challenge**: A good SSL regularizer must simultaneously satisfy four properties (instance permutation invariance, instance cloning, feature cloning, and feature constraint). Existing methods either fail to satisfy all four or incur high computational cost.

**Goal**: Design an SSL regularization method that is conceptually simple, computationally efficient, and theoretically provable in terms of preventing dimensional collapse and promoting uniformity.

**Key Insight**: MST length is a classical estimator of distributional entropy (Steele's theorem); maximizing MST length is equivalent to maximizing the Rényi entropy of the representation distribution, which naturally promotes uniformity. Moreover, MST length is sensitive to the intrinsic dimensionality of a point cloud, enabling natural detection and avoidance of dimensional collapse.

**Core Idea**: By maximizing the length of the minimum spanning tree in the embedding space subject to a hyperspherical constraint, a single simple regularization term simultaneously addresses dimensional collapse and non-uniformity.

## Method

### Overall Architecture
The T-REGS framework applies T-REG regularization to both branches of a JE-SSL model. Each branch independently computes the MST length and maximizes it. The overall loss is the view-invariance loss (e.g., MSE or the loss of an existing SSL method) plus the T-REG regularization term.

### Key Designs

1. **MST Length Maximization Loss $\mathcal{L}_E$**:

    - **Function**: Maximizes the MST length of the embedding point cloud $Z$.
    - **Mechanism**: $\mathcal{L}_E(Z) = -\frac{1}{n} E(\text{MST}(Z))$. Each edge in the MST induces a repulsive force between its two endpoints (the gradient has a clean closed form: $\nabla_x E = \sum_{(x,z) \in \text{MST}} \frac{x-z}{\|x-z\|_2}$). Computation can be performed efficiently using GPU-parallel MST algorithms.
    - **Design Motivation**: MST length is related to the Rényi $\frac{d-1}{d}$-entropy (Steele's theorem); maximizing MST length → maximizing entropy → convergence toward a uniform distribution.

2. **Hyperspherical Constraint $\mathcal{L}_S$**:

    - **Function**: Soft constraint that projects embedding points onto the unit hypersphere.
    - **Mechanism**: $\mathcal{L}_S(Z) = \frac{1}{n} \sum_i (\|z_i\|_2 - 1)^2$, preventing points from diverging to infinity when MST length is naively maximized.
    - Combined as T-REG: $\mathcal{L}_{\text{T-REG}}(Z) = \gamma \mathcal{L}_E(Z) + \lambda \mathcal{L}_S(Z)$.

3. **Theoretical Guarantees**:

    - **Small-sample behavior (Theorem 4.1)**: When $n \leq d+1$, the MST length on the hypersphere is maximized at the vertices of a regular simplex — points naturally spread to the maximum-margin configuration on the sphere.
    - **Large-sample asymptotics (Theorem 4.4 + Proposition 4.5)**: As $n \to \infty$, on a compact Riemannian manifold, maximizing MST length is equivalent to maximizing an estimator of the Rényi entropy, whose unique maximum is attained at the uniform distribution. Furthermore, the uniform distribution on a higher-dimensional manifold yields a larger upper bound, naturally encouraging the use of all available dimensions.

### Usage
T-REGS can serve as: (a) a standalone regularizer — replacing an entire SSL method with MSE + T-REG; or (b) an auxiliary loss — stacked on top of existing methods (VICReg, BarlowTwins, etc.) to improve performance.

## Key Experimental Results

### Main Results
Standard SSL evaluation (linear probing, KNN) on CIFAR-10, CIFAR-100, STL-10, and Tiny-ImageNet using ResNet-18/50.

| Method | Key Metric | Note |
|--------|-----------|------|
| T-REGS (standalone) | Comparable to VICReg/BarlowTwins | Uses only MSE + T-REG, without negative samples or covariance regularization |
| VICReg + T-REG | Outperforms VICReg | Improves existing method as an auxiliary loss |
| BarlowTwins + T-REG | Outperforms BarlowTwins | Same as above |

### Ablation Study

| Configuration | Result |
|--------------|--------|
| $\mathcal{L}_E$ only (no hyperspherical constraint) | Points diverge to infinity; no convergence |
| $\mathcal{L}_S$ only | No regularization effect |
| Full T-REG | Stable convergence to uniform distribution on hypersphere |
| High-dimensional (256-dim) synthetic data | Successfully converges to regular simplex |

### Key Findings
- **MST regularization effectively prevents dimensional collapse**: The effective rank of the embeddings is substantially higher than that of baselines without T-REG.
- **Improved uniformity**: T-REG yields more uniform representation distributions on the hypersphere compared to covariance regularization alone.
- **Universally effective as an auxiliary loss**: Adding T-REG on top of multiple existing methods consistently yields performance gains.
- **Satisfies all four properties of Fang et al.**: T-REG is theoretically proven to satisfy instance permutation invariance, instance cloning, feature cloning, and the feature constraint property.

## Highlights & Insights
- **An elegant connection from topological data analysis to SSL**: MST length is directly linked to uniformity via Steele's theorem and Rényi entropy — a connection that is both natural and powerful.
- **The conceptually simplest SSL regularizer**: Compared to VICReg's multi-term loss or the negative-sample mechanism of contrastive learning, "maximize MST length + hyperspherical constraint" is remarkably concise.
- **General theory on Riemannian manifolds**: The theoretical results are not restricted to Euclidean space or the hypersphere, but hold for arbitrary compact Riemannian manifolds.

## Limitations & Future Work
- **MST computational cost**: Despite GPU-parallel algorithms, MST computation remains at least $O(n^2)$, which may become a bottleneck for very large batch sizes.
- **Hyperparameter tuning of $\gamma$ and $\lambda$**: The two weight parameters governing the balance between repulsion and constraint require empirical tuning.
- **Limited large-scale experiments**: The primary experiments are conducted on CIFAR/STL/Tiny-ImageNet; validation at the ImageNet-1K scale is absent.
- **Comparison with state-of-the-art methods**: No direct comparison with the strongest current self-supervised methods such as DINO/DINOv2.

## Related Work & Insights
- **vs. VICReg/BarlowTwins**: These methods apply covariance regularization (second-order statistics), whereas T-REG implicitly captures higher-order distributional information through the MST.
- **vs. Fang et al. OT method**: Both satisfy the four required properties, but T-REG is computationally simpler, requiring neither SVD nor square-root operations.
- **vs. Contrastive learning**: T-REG requires no negative samples and naturally achieves uniformity through the repulsive forces induced by MST edges.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying MST regularization to SSL represents a novel cross-domain transfer with elegant theoretical foundations.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic and medium-scale experiments are sufficient, but large-scale validation is lacking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](m-grpo_stabilizing_self-supervised_reinforcement_learning_for_large_language_mod.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)

</div>

<!-- RELATED:END -->
