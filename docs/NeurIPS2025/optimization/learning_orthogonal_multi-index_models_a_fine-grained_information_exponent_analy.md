---
title: >-
  [Paper Note] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis
description: >-
  [NeurIPS 2025][Optimization][multi-index models] This paper proves that orthogonal multi-index models $f_*(\mathbf{x}) = \sum_{k=1}^P \phi(\mathbf{v}_k^* \cdot \mathbf{x})$ can be learned via a two-phase online SGD with…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "multi-index models"
  - "information exponent"
  - "Hermite expansion"
  - "SGD sample complexity"
  - "subspace recovery"
date: 2026-05-08
content_hash: d7968548c0bdb8b5
---

# Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis

**Conference**: NeurIPS 2025
**arXiv**: [2410.09678](https://arxiv.org/abs/2410.09678)  
**Code**: To be confirmed  
**Area**: Learning Theory / Optimization
**Keywords**: multi-index models, information exponent, Hermite expansion, SGD sample complexity, subspace recovery

## TL;DR
This paper proves that orthogonal multi-index models $f_*(\mathbf{x}) = \sum_{k=1}^P \phi(\mathbf{v}_k^* \cdot \mathbf{x})$ can be learned via a two-phase online SGD with sample complexity $\tilde{O}(dP^{L-1})$ (where $L$ is the lowest higher-order Hermite degree of the link function), significantly improving upon $\tilde{O}(Pd^{L-1})$ obtained by using only the lowest-order information. The key insight is to first recover the subspace using second-order terms and then recover the directions using $L$-th order terms, jointly exploiting Hermite components of different orders.

## Background & Motivation

**Background**: Information exponent (IE) theory predicts that the sample complexity of online SGD for learning single-index models is $\tilde{O}(d^{\text{IE}-1})$, where IE is the lowest nonzero degree in the Hermite expansion of the link function. This result has been extended to multi-index models, where the complexity is typically $\tilde{O}(Pd^{L-1})$ ($P$ directions, $L$ being the IE). The generative exponent (GE) extends IE by reducing the effective degree through label transformations.

**Limitations of Prior Work**:
- Both IE and GE focus solely on the "lowest order," ignoring structural information contained in higher-order terms of the link function.
- When IE = 2 (second-order Hermite term), rotational invariance allows only subspace recovery rather than exact direction recovery — a fundamental limitation for all frameworks that focus exclusively on the lowest order.
- However, if the link function contains both second-order and higher-order terms (e.g., $\phi = h_2 + h_4$), the higher-order terms can in principle break rotational symmetry and enable direction recovery — an intuition that had not been formally established.

**Key Challenge**: For link functions of the form $\phi = h_2 + h_L$, IE = 2, and existing theory predicts that only subspace recovery is achievable. Yet intuitively, first recovering the subspace via second-order terms (dimension $P$, cost $\tilde{O}(dP)$), then recovering the directions within the $P$-dimensional subspace via $L$-th order terms (cost $\tilde{O}(P^{L-1})$), should yield a total complexity of $\tilde{O}(dP^{L-1})$ rather than $\tilde{O}(Pd^{L-1})$ — a gap of up to $(d/P)^{L-2}$.

**Goal**: To rigorously prove the above intuition: a two-phase SGD that jointly exploits Hermite components of different orders achieves sample complexity $\tilde{O}(dP^{L-1})$.

**Key Insight**: The learning process is decomposed into a subspace search phase (driven by second-order terms) and a direction recovery phase (driven by $L$-th order terms), with gradient flow dynamics analyzed in each phase.

**Core Idea**: Use second-order Hermite terms to recover the subspace at $O(d)$ cost, then use $L$-th order terms within the low-dimensional subspace to recover the directions at $O(P^{L-1})$ cost, yielding a total complexity of $\tilde{O}(dP^{L-1})$.

## Method

### Overall Architecture
Learning objective: $f_*(\mathbf{x}) = \sum_{k=1}^P \phi(\mathbf{v}_k^* \cdot \mathbf{x})$, $\mathbf{x} \sim \mathcal{N}(0, \mathbf{I}_d)$, $\phi = \hat\phi_2 h_2 + \sum_{l \geq L} \hat\phi_l h_l$ (IE = 2 but containing terms of degree $L \geq 3$). A two-layer network of width $O(P)$ is trained via two-phase online SGD.

### Key Designs

1. **Phase 1: Subspace Recovery (driven by second-order terms)**

    - **Function**: Use $\tilde{O}(dP)$ samples to align the $P$ neuron directions of the network with the target subspace $\text{span}\{\mathbf{v}_k^*\}$.
    - **Mechanism**: The gradient information contributed by the second-order Hermite term $\hat\phi_2 h_2$ causes each neuron's direction to grow toward the projection matrix of the target subspace. Gradient flow dynamics analysis shows that starting from random initialization, each neuron achieves constant correlation with the subspace after $\tilde{O}(d)$ steps.
    - **Design Motivation**: Second-order terms cannot distinguish different directions within the subspace (rotational invariance), but they can efficiently locate the subspace — this is exploited for dimensionality reduction as a first step.

2. **Phase 2: Direction Recovery (driven by $L$-th order terms)**

    - **Function**: Recover each direction $\mathbf{v}_k^*$ within the already-recovered $P$-dimensional subspace using $\tilde{O}(P^{L-1})$ samples.
    - **Mechanism**: The $L$-th order Hermite term $\hat\phi_L h_L$ breaks the rotational symmetry of the second-order terms. Within the $P$-dimensional subspace, direction recovery is equivalent to $P$ independent single-index problems with effective dimension $P$, each requiring $\tilde{O}(P^{L-2})$ samples.
    - **Design Motivation**: The subspace information from Phase 1 is jointly leveraged — without first locating the subspace, recovering each direction would require $\tilde{O}(d^{L-2})$ samples.

3. **Two-Phase SGD Implementation**

    - **Function**: Concrete algorithm design.
    - **Mechanism**: Phase 1 uses population gradient flow for theoretical analysis, approximated by online SGD in practice. Phase 2 also uses online SGD, operating in the low-dimensional space after subspace projection. An optional ridge regression fine-tuning step can be appended.
    - **Design Motivation**: Online SGD is the most natural and information-theoretically efficient learning algorithm — each sample is used only once.

### Loss & Training
- Squared loss $\ell = (f(\mathbf{x}) - y)^2$
- Two-layer network $f(\mathbf{x}) = \sum_{j=1}^m a_j \sigma(\mathbf{w}_j \cdot \mathbf{x})$, width $m = \tilde{O}(P)$

## Key Experimental Results

### Main Results (Theoretical Theorems)

| Setting | Lowest-order IE only | Two-phase (Ours) | Improvement |
|---------|---------------------|------------------|-------------|
| $P=5, d=1000, L=4$ | $\tilde{O}(5 \times 10^9)$ | $\tilde{O}(1000 \times 125)$ | ~40,000× |
| $P=10, d=10^4, L=3$ | $\tilde{O}(10 \times 10^8)$ | $\tilde{O}(10^4 \times 100)$ | ~10,000× |
| General | $\tilde{O}(Pd^{L-1})$ | $\tilde{O}(dP^{L-1})$ | $\sim (d/P)^{L-2}$ |

### Ablation Study: Contribution of Each Hermite Order

| Component | Cost | Recoverable Content |
|-----------|------|---------------------|
| $h_2$ term only | $\tilde{O}(dP)$ samples | Subspace (directions not recoverable) |
| $h_L$ term only | $\tilde{O}(Pd^{L-1})$ samples | Directions (cost scales as $d^{L-1}$) |
| $h_2 + h_L$ jointly | $\tilde{O}(dP^{L-1})$ samples | Directions (cost scales as $P^{L-1}$) |

### Key Findings
- **Focusing solely on the lowest-order information exponent misses critical structure**: At IE = 2, existing theory suggests only subspace recovery is possible; however, jointly exploiting higher-order terms enables exact direction recovery.
- **$\tilde{O}(dP^{L-1})$ matches the complexity of $P$ independent single-index problems with known subspace**: This indicates the two-phase approach is optimal (up to log factors).
- **Dimensional dependence reduced from $d^{L-1}$ to $d$**: The improvement is exponential in high dimensions.
- **Only a two-layer network of width $\tilde{O}(P)$ is required**: Network complexity matches the effective dimensionality without requiring excessive width.

## Highlights & Insights
- The perspective that **"each Hermite component serves its own purpose"** is highly insightful — second-order terms handle subspace search while higher-order terms handle direction recovery, analogous to two rulers of different precision used for coarse and fine measurement respectively.
- The transition from $Pd^{L-1}$ to $dP^{L-1}$ is a **qualitative leap** — an exponential improvement in the typical regime where $P \ll d$.
- This work challenges the prevailing learning-theoretic consensus that "examining only the lowest order suffices" — multi-order joint analysis is necessary.
- The results provide theoretical guidance for understanding feature learning in neural networks.

## Limitations & Future Work
- Assumes Gaussian inputs $\mathbf{x} \sim \mathcal{N}(0, \mathbf{I}_d)$ — Hermite analysis breaks down for non-Gaussian distributions.
- Assumes orthogonal directions — when directions are non-orthogonal, the interaction between subspace recovery and direction recovery becomes more complex.
- This is a purely theoretical work with no large-scale experimental validation.
- The link function $\phi$ must simultaneously contain $h_2$ and $h_L$ terms — if the second-order coefficient is zero, the result degrades to the general $Pd^{L-1}$ bound.

## Related Work & Insights
- **vs. Ben Arous et al. (2021)**: Foundational work on IE theory establishing sample complexity $\tilde{O}(d^{\text{IE}-1})$. This paper demonstrates the insufficiency of IE in the multi-index setting.
- **vs. Damian et al. (2024) GE**: GE reduces IE via label transformations but still considers only the lowest order. This paper considers joint multi-order analysis.
- **vs. Tensor decomposition methods**: Higher-order tensor decomposition can recover directions but does not exploit lower-order structure. The two-phase SGD proposed here jointly exploits information from all orders more efficiently.
- The results provide a theoretical explanation for the "staged" learning behavior of neural networks — different features are learned at different stages of training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Challenges the "lowest-order-suffices" consensus; the theoretical perspective of joint multi-order analysis is entirely novel.
- Experimental Thoroughness: ⭐⭐ Purely theoretical; no experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Theorems are clearly stated; the progression from intuition to formalization is smooth.
- Value: ⭐⭐⭐⭐⭐ Makes an important foundational contribution to sample complexity analysis in learning theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Single-Index Models via Harmonic Decomposition](learning_single-index_models_via_harmonic_decomposition.md)
- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](../../ICLR2026/optimization/neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)
- [\[NeurIPS 2025\] MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions](mobo-osd_batch_multi-objective_bayesian_optimization_via_orthogonal_search_direc.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[NeurIPS 2025\] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)

</div>

<!-- RELATED:END -->
