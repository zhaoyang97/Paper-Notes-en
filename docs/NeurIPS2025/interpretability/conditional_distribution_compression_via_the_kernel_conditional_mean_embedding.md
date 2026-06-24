---
title: >-
  [Paper Note] Conditional Distribution Compression via the Kernel Conditional Mean Embedding
description: >-
  [NeurIPS 2025][Interpretability][distribution compression] This work presents the first compression algorithm targeting **conditional distributions** (rather than joint distributions), introducing a novel metric AMCMD based on the kernel conditional mean embedding (KCME) and a linear-time algorithm ACKIP for constructing compressed datasets that preserve the statistical properties of conditional distributions.
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "distribution compression"
  - "kernel methods"
  - "conditional mean embedding"
  - "RKHS"
  - "data compression"
date: 2026-05-08
content_hash: 5f84b556b189cc63
---

# Conditional Distribution Compression via the Kernel Conditional Mean Embedding

**Conference**: NeurIPS 2025
**arXiv**: [2504.10139](https://arxiv.org/abs/2504.10139)  
**Code**: None  
**Area**: Interpretability
**Keywords**: distribution compression, kernel methods, conditional mean embedding, RKHS, data compression

## TL;DR

This work presents the first compression algorithm targeting **conditional distributions** (rather than joint distributions), introducing a novel metric AMCMD based on the kernel conditional mean embedding (KCME) and a linear-time algorithm ACKIP for constructing compressed datasets that preserve the statistical properties of conditional distributions.

## Background & Motivation

**Background**: Existing distribution compression methods (e.g., Kernel Herding, Kernel Thinning) focus on unlabeled data and construct compressed sets by minimizing MMD to approximate the marginal distribution $\mathbb{P}_X$. These methods have achieved notable success in reducing data volume while maintaining statistical fidelity.

**Limitations of Prior Work**: For labeled data $\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n$, no existing method directly targets compression of the conditional distribution family $\mathbb{P}_{Y|X}$. Prior approaches either ignore label information or can only indirectly approximate the conditional distribution by compressing the joint distribution $\mathbb{P}_{X,Y}$.

**Key Challenge**: Estimating the kernel conditional mean embedding (KCME) incurs $\mathcal{O}(n^3)$ cost (due to matrix inversion), rendering conditional distribution compression computationally infeasible at scale. While directly compressing the conditional distribution should intuitively outperform indirect compression via the joint distribution, the prohibitive computational cost has hindered practical adoption.

**Goal**: To construct a compressed set $\mathcal{C}$ in linear time $\mathcal{O}(n)$ such that the compressed conditional distribution satisfies $\tilde{\mathbb{P}}_{Y|X=\mathbf{x}} \approx \mathbb{P}_{Y|X=\mathbf{x}}$ for almost all $\mathbf{x}$.

**Key Insight**: A key observation based on the tower property — the objective of conditional distribution compression satisfies $\mathbb{E}_{\mathbf{x} \sim \mathbb{P}_X}[\langle \mu_{Y|X=\mathbf{x}}, h(\mathbf{x}) \rangle] = \mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim \mathbb{P}_{X,Y}}[h(\mathbf{x})(\mathbf{y})]$, thereby avoiding explicit KCME estimation and reducing complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(n)$.

**Core Idea**: By defining the AMCMD metric for conditional distributions and leveraging the tower property to simplify estimation, the paper designs the first linear-time conditional distribution compression algorithm.

## Method

### Overall Architecture

Four methods are proposed for labeled-data distribution compression, organized into two categories:
- **Joint distribution compression**: JKH (greedy) + JKIP (joint optimization)
- **Conditional distribution compression**: ACKH (greedy) + ACKIP (joint optimization)

The core pipeline constructs a small compressed set $\mathcal{C} = \{(\tilde{\mathbf{x}}_j, \tilde{\mathbf{y}}_j)\}_{j=1}^m$ ($m \ll n$) from the original dataset $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n$, with the optimization objective of minimizing a discrepancy measure between the original and compressed distributions.

### Key Designs

#### 1. The AMCMD Metric

**Function**: Defines a distance measure between families of conditional distributions.

**Core Formula**:
$$\text{AMCMD}[\mathbb{P}_{X^*}, \mathbb{P}_{Y|X}, \mathbb{P}_{Y'|X'}] = \sqrt{\mathbb{E}_{\mathbf{x} \sim \mathbb{P}_{X^*}} \left[\|\mu_{Y|X=\mathbf{x}} - \mu_{Y'|X'=\mathbf{x}}\|_{\mathcal{H}_l}^2\right]}$$

**Design Motivation**: (1) The introduction of an independent probability measure $\mathbb{P}_{X^*}$ renders AMCMD more general than the existing KCD metric; (2) it satisfies positive definiteness and the triangle inequality (Theorem 4.1); (3) it admits a closed-form estimator (Lemma 4.3).

#### 2. Complexity Reduction via the Tower Property (Lemma 4.7)

**Function**: Eliminates the need for explicit KCME estimation.

**Mechanism**: For any vector-valued function $h: \mathcal{X} \to \mathcal{H}_l$:
$$\mathbb{E}_{\mathbf{x} \sim \mathbb{P}_X}[\langle \mu_{Y|X=\mathbf{x}}, h(\mathbf{x}) \rangle_{\mathcal{H}_l}] = \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \mathbb{P}_{X,Y}}[h(\mathbf{x})(\mathbf{y})]$$

**Design Motivation**: KCME estimation requires $\mathcal{O}(n^3)$ operations (matrix inversion), but by setting $h(\mathbf{x}) = \tilde{\mu}_{Y|X=\mathbf{x}}^{\mathcal{C}}$, the expectation can be computed directly via an empirical average, reducing complexity to $\mathcal{O}(n)$.

#### 3. The ACKIP Algorithm

**Function**: Jointly optimizes all point pairs in the compressed set to minimize $\text{AMCMD}^2[\hat{\mathbb{P}}_X, \hat{\mathbb{P}}_{Y|X}, \tilde{\mathbb{P}}_{Y|X}]$.

**Objective**:
$$\mathcal{J}^{\mathcal{D}}(\tilde{\mathbf{X}}, \tilde{\mathbf{Y}}) = \frac{1}{n}\text{Tr}(K_{\mathbf{X}\tilde{\mathbf{X}}} W_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} L_{\tilde{\mathbf{Y}}\tilde{\mathbf{Y}}} W_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} K_{\tilde{\mathbf{X}}\mathbf{X}}) - \frac{2}{n}\text{Tr}(L_{\mathbf{Y}\tilde{\mathbf{Y}}} W_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} K_{\tilde{\mathbf{X}}\mathbf{X}})$$

where $W_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} = (K_{\tilde{\mathbf{X}}\tilde{\mathbf{X}}} + \lambda I)^{-1}$.

**Design Motivation**: ACKH (greedy) cannot revise previously selected points and incurs quartic complexity $\mathcal{O}(m^4 + m^3 n)$. ACKIP reduces this to $\mathcal{O}(m^3 + m^2 n)$ via joint optimization, achieving an $m$-fold speedup.

### Loss & Training

- Gaussian kernels are used for continuous responses; indicator kernels for discrete responses
- All points in the compressed set are jointly optimized via gradient descent
- Initialization: $m$ points are sampled uniformly at random from the original dataset
- The regularization parameter $\lambda$ controls the smoothness of the KCME estimate

## Key Experimental Results

### Main Results

**True Conditional Distribution Matching ($m=500$)**: ACKIP outperforms all methods on both AMCMD² and RMSE across all test functions.

**Superconductivity Dataset ($n=10000, m=250$)**:

| Method | $h(y)=y$ RMSE | $h(y)=y^2$ RMSE | Rank |
|--------|--------------|-----------------|------|
| ACKIP | **Lowest** | **Lowest** | 1 |
| ACKH | 2nd lowest | 2nd lowest | 2 |
| JKIP | Medium | Medium | 3 |
| JKH | High | High | 4 |
| Random | Highest | Highest | 5 |

### Ablation Study

**Conditional vs. Joint Compression**:

| Comparison | Finding |
|------------|---------|
| ACKIP vs. JKIP | Direct conditional compression > indirect compression via joint distribution |
| JKIP vs. JKH | Joint optimization > greedy selection |
| ACKIP vs. ACKH | Joint optimization > greedy selection (with lower complexity) |

**Imbalanced Classification Dataset ($m=250$)**: ACKIP achieves the lowest RMSE for estimating conditional probabilities $\mathbb{P}(Y=c|X)$ in 3 out of 4 classes, approaching full-data performance using only 3% of the data. ACKH even underperforms random sampling in 3 classes.

### Key Findings

1. ACKIP consistently achieves the best performance across all experiments (continuous/discrete, synthetic/real data)
2. Greedy methods (JKH, ACKH) are notably weaker than joint optimization due to their inability to revise previously selected points
3. Conditional distribution compression methods systematically outperform joint distribution compression on conditional expectation estimation
4. At compressed set size $m = 250$, ACKIP already approaches the performance of the full dataset with $n = 10000$

## Highlights & Insights

1. **Strong theoretical contributions**: AMCMD satisfies rigorous metric properties (positive definiteness + triangle inequality), admits a closed-form estimator, and is provably consistent
2. **Complexity breakthrough**: The tower property reduces KCME-related computation from $\mathcal{O}(n^3)$ to $\mathcal{O}(n)$, which is the key to making conditional distribution compression feasible
3. **Unified framework**: A complete family of methods for both joint and conditional distribution compression is proposed, enabling systematic comparison
4. **Practical value**: KCME is widely used in conditional density estimation, Bayesian optimization, and reinforcement learning; the proposed compression method can substantially broaden its applicability

## Limitations & Future Work

1. **No convergence guarantees**: ACKIP lacks rigorous theoretical convergence proofs; convergence is supported only by empirical observation
2. **Gradient dependency**: Kernel gradients are inapplicable or difficult to interpret for discrete structured data such as graphs and text
3. **Kernel selection**: Algorithm performance is sensitive to the choice of kernel and hyperparameters (bandwidth, regularization)
4. **Scalability**: Although complexity is linear in $n$, the $\mathcal{O}(m^3)$ term limits the feasible compressed set size
5. Incorporating subset-selection strategies from Kernel Thinning could be explored for scenarios where gradients are unavailable

## Related Work & Insights

- **Kernel Herding** [Chen et al. 2010]: A classical unlabeled distribution compression method; this work extends it to joint and conditional distributions
- **Kernel Thinning** [Dwivedi & Mackey 2021]: Restricts the compressed set to a subset of the original data; may be more suitable when gradients are unavailable
- **KCME** [Song et al. 2009]: The theoretical foundation of kernel conditional mean embeddings; this work makes KCME practically viable for large-scale data via distribution compression

## Rating

⭐⭐⭐⭐ (4/5)

The work presents strong theoretical innovations (AMCMD + complexity reduction via the tower property) and systematic experiments, but lacks convergence guarantees and validation on high-dimensional complex data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement](far_from_the_shallow_brain-predictive_reasoning_embedding_through_residual_disen.md)
- [\[ICML 2026\] MAAT: Heterogeneous Partial Observation State Reconstruction Based on Knowledge-Guided Kernel Regression](../../ICML2026/interpretability/knowledge-informed_kernel_state_reconstruction_from_heterogeneous_partial_observ.md)
- [\[ICLR 2026\] Learning is Forgetting: LLM Training As Lossy Compression](../../ICLR2026/interpretability/learning_is_forgetting_llm_training_as_lossy_compression.md)
- [\[CVPR 2026\] Hierarchical Concept Embedding & Pursuit for Interpretable Image Classification](../../CVPR2026/interpretability/hierarchical_concept_embedding_pursuit_for_interpretable_image_classification.md)
- [\[ICML 2025\] SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression](../../ICML2025/interpretability/slim_one-shot_quantization_and_sparsity_with_low-rank_approximation_for_llm_weig.md)

</div>

<!-- RELATED:END -->
