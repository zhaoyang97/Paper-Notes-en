---
title: >-
  [Paper Note] Rethinking Consistent Multi-Label Classification Under Inexact Supervision
description: >-
  [ICLR 2026][Optimization][Multi-label classification] The COMES framework is proposed to provide consistent risk estimators for multi-label classification under inexact supervision through first-order (Hamming loss) and second-order (Ranking loss) strategies, eliminating the need for label generation process estimation or uniform distribution assumptions.
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Multi-label classification"
  - "weakly supervised learning"
  - "partial multi-label learning"
  - "complementary multi-label learning"
  - "risk consistency"
date: 2026-05-08
content_hash: ea88c27bb4f8ab9e
---

# Rethinking Consistent Multi-Label Classification Under Inexact Supervision

**Conference**: ICLR 2026  
**arXiv**: [2510.04091](https://arxiv.org/abs/2510.04091)  
**Area**: Optimization  
**Keywords**: Multi-label classification, weakly supervised learning, partial multi-label learning, complementary multi-label learning, risk consistency  

## TL;DR

The COMES framework is proposed to provide consistent risk estimators for multi-label classification under inexact supervision through first-order (Hamming loss) and second-order (Ranking loss) strategies, eliminating the need for label generation process estimation or uniform distribution assumptions.

## Background & Motivation

Multi-label classification (MLC) requires each instance to be associated with multiple relevant labels, leading to significantly higher annotation costs than single-label tasks. To reduce annotation pressure, two weakly supervised paradigms have been proposed:

- **Partial Multi-Label learning (PML)**: Each instance is annotated with a candidate label set containing all ground-truth relevant labels and some irrelevant "false positive" labels.
- **Complementary Multi-Label learning (CML)**: Each instance is annotated with complementary labels indicating which classes the instance does not belong to.

Core Observation: PML and CML are mathematically equivalent—the complement of the candidate label set is the complementary label set.

**Limitations of Prior Work**:

1. Accurate estimation of the generation process (i.e., transition matrix) for candidate/complementary labels is required, but the overconfidence of deep neural networks makes such estimation unreliable.
2. Uniform distribution assumptions are often used to bypass estimation problems, but these are oversimplified and fail to handle real-world class imbalance.
3. Many methods model different labels independently, ignoring semantic correlations between labels.

## Method

### Overall Architecture

COMES (COnsistent Multi-label classification under inExact Supervision) first utilizes a "class-wise query" data generation hypothesis that closely aligns with real-world annotation processes. It decomposes the multi-label risk under inexact supervision into forms that can be estimated directly from weak labels, further providing first-order (based on Hamming loss) and second-order (based on Ranking loss) consistent risk estimators. This approach requires neither the estimation of transition matrices for candidate/complementary labels nor uniform distribution assumptions. PML and CML are unified within this framework due to their mathematical equivalence.

### Key Designs

**1. Class-wise Query Hypothesis: Replacing Uniform Assumptions with Weaker Conditions**

Prior works either estimate transition matrices that are difficult to fit reliably or crudely assume candidate labels follow a uniform distribution, failing to address class imbalance. COMES assumes that annotation is performed independently for each class: if the $j$-th class is irrelevant to instance $\boldsymbol{x}$, it is labeled as a non-candidate label with a constant probability $p_j$, i.e., $p(j \notin S \mid \boldsymbol{x}, j \notin Y) = p_j$. The value of this assumption lies in Lemma 1—the conditional density of non-candidate instances is exactly equal to the conditional density of irrelevant samples for that class: $p(\boldsymbol{x} \mid s_j = 0) = p(\boldsymbol{x} \mid y_j = 0)$. With this equivalence, "non-candidate" samples observed under weak supervision can be treated as clean negative samples, enabling risk reformulation; furthermore, different labels can have distinct $p_j$ values, which is much more flexible than the uniform assumption.

**2. COMES-HL: Decomposing Multi-Label Risk into Estimable Binary Risks**

The first-order strategy treats MLC as $q$ independent binary classification problems, aiming for Hamming loss consistency. Via Theorem 1, the Hamming risk is rewritten using only observable distributions:

$$R_H^\ell(\boldsymbol{g}) = \mathbb{E}_{p(\boldsymbol{x})}\left[\frac{1}{q}\sum_{j=1}^q \ell(g_j(\boldsymbol{x}), 1)\right] + \sum_{j=1}^q \mathbb{E}_{p(\boldsymbol{x}|s_j=0)}\left[\frac{1-\pi_j}{q}\big(\ell(g_j(\boldsymbol{x}), 0) - \ell(g_j(\boldsymbol{x}), 1)\big)\right]$$

The first term is estimated on the entire (unlabeled) dataset $\mathcal{D}_U$, and the second term is estimated on the non-candidate conditional datasets $\mathcal{D}_j$ for each class, with class priors $\pi_j$ serving as weights. This yields an unbiased risk estimator. However, the negative weight term $1-\pi_j$ can cause deep networks to push empirical risk to negative values, leading to overfitting. Therefore, an absolute value wrapper is used to bound the negative term, resulting in a corrected estimator $\tilde{R}_H^\ell$ that constrains the risk within a reasonable range without sacrificing consistency.

**3. COMES-RL: Introducing Label Correlation via Ranking Relationships**

While the Hamming strategy is class-independent and ignores semantic correlations, the second-order strategy optimizes Ranking loss by modeling the relative order of label pairs $(j, k)$. It requires the surrogate loss to satisfy a symmetry condition $\ell(z, \cdot) + \ell(-z, \cdot) = M$, allowing the Ranking risk to be rewritten based only on non-candidate conditional distributions:

$$R_R^\ell(\boldsymbol{g}) = \sum_{1 \leq j < k \leq q}\Big((1-\pi_j)\mathbb{E}_{p(\boldsymbol{x}|s_j=0)}[\ell(g_j - g_k, 0)] + (1-\pi_k)\mathbb{E}_{p(\boldsymbol{x}|s_k=0)}[\ell(g_j - g_k, 1)]\Big)$$

Symmetry is key, as it allows positive label terms to be absorbed into the constant $M$, leaving only terms estimable from weak labels. To mitigate overfitting, flooding regularization is used to pull the empirical risk back to a set lower bound $\beta$: $\tilde{R}_R^\ell = |\hat{R}_R^\ell - \beta| + \beta$. The trade-off is the $O(q^2)$ complexity required to iterate through all label pairs. Consequently, RL is suitable for scenarios with strong label correlations and moderate label spaces, complementing the efficient but correlation-agnostic HL.

### Loss & Training

COMES-HL uses binary cross-entropy as the surrogate loss, while COMES-RL requires symmetric losses such as sigmoid to satisfy the symmetry condition. The class priors $\pi_j$ required by both can be directly estimated from candidate labels using existing class prior estimation methods. Building on this, both estimators provide complete theoretical guarantees:

| Property | COMES-HL | COMES-RL |
|------|----------|----------|
| Bounded Bias | $0 \leq \text{bias} \leq O(\Delta_j)$, $\Delta_j \to 0$ as $n \to \infty$ | $0 \leq \text{bias} \leq O(\Delta')$, $\Delta' \to 0$ as $n \to \infty$ |
| Estimation Error Conv. | $O(\mathfrak{R}_n(\mathcal{G}) + \sqrt{\ln(1/\delta)/n})$ | $O(\mathfrak{R}_n(\mathcal{G}) + \sqrt{\ln(1/\delta)/n})$ |
| Consistency | Bayes optimal w.r.t. Hamming loss | Bayes optimal w.r.t. Ranking loss |

The bias introduced by correction terms (absolute value / flooding) vanishes as the sample size increases, and estimation errors converge at standard statistical learning rates. Thus, both are asymptotically consistent with respect to Hamming loss and Ranking loss, respectively.

## Key Experimental Results

### Main Results: Real-World Datasets (Ranking Loss ↓)

| Method | mirflickr | music_emotion | yeastBP | yeastCC | yeastMF |
|------|-----------|---------------|---------|---------|---------|
| BCE | 0.106 | 0.244 | 0.328 | 0.206 | 0.251 |
| CCMN | 0.106 | 0.224 | 0.328 | 0.210 | 0.245 |
| GDF | 0.159 | 0.278 | 0.501 | 0.504 | 0.495 |
| CTL | 0.130 | 0.266 | 0.498 | 0.467 | 0.471 |
| **COMES-HL** | **0.095** | **0.214** | **0.154** | **0.124** | 0.173 |
| **COMES-RL** | 0.106 | **0.213** | 0.166 | **0.117** | **0.151** |

### Feature Comparison

| Method | No Uniform Dist. Assumption | No Generation Process Est. | Label Correlation Aware | Multiple Complementary Labels |
|------|:---:|:---:|:---:|:---:|
| CCMN | ✓ | ✗ | ✓ | ✓ |
| CTL | ✗ | ✓ | ✗ | ✗ |
| GDF | ✗ | ✓ | ✗ | ✓ |
| **COMES-HL** | **✓** | **✓** | ✗ | **✓** |
| **COMES-RL** | **✓** | **✓** | **✓** | **✓** |

### Key Findings

1. COMES-HL and COMES-RL consistently outperform SOTA methods across 5 evaluation metrics on 6 real-world datasets.
2. The advantage is particularly significant on yeast series datasets, with Ranking Loss reduced by approximately 50% (e.g., yeastBP: 0.154 vs 0.328).
3. COMES-RL performs better on datasets with strong label correlations (e.g., yeastMF: 0.151 vs 0.173).
4. Both methods demonstrate robustness on synthetic data with different label generation processes (uniform/non-uniform).

## Highlights & Insights

1. **Significant Theoretical Contribution**: Proves consistency for MLC under inexact supervision for the first time without relying on transition matrix estimation or uniform distribution assumptions.
2. **Unified Treatment of PML and CML**: Leverages their mathematical equivalence to solve both problems within a single framework.
3. **Complementary First and Second-Order Strategies**: COMES-HL is efficient but ignores label correlation, while COMES-RL utilizes ranking relationships at a higher computational cost, catering to different scenarios.
4. **Practical Data Generation Hypothesis**: The "class-wise query irrelevance" hypothesis aligns better with actual annotation workflows.
5. **Ingenious Risk Estimator Correction**: Absolute value wrapping and flooding regularization effectively address overfitting issues in first-order and second-order strategies.

## Limitations

1. Focuses only on Hamming loss and Ranking loss, not covering other MLC metrics (e.g., F1-measure).
2. The quality of class prior $\pi_j$ estimation affects final performance, but error propagation is not analyzed in depth.
3. The assumption of constant $p_j$ in the data generation process may not fully capture complex annotation behaviors.
4. The $O(q^2)$ computational complexity of the second-order strategy may become a bottleneck in large label spaces.
5. Datasets used in experiments are relatively small; performance on large-scale datasets remains to be verified.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Novel framework design that unifies PML/CML and removes strong assumptions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensively surpasses SOTA across multiple datasets and metrics, though dataset scales are small.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous and complete theoretical derivations with a clear structure.
- **Value**: ⭐⭐⭐⭐ — Provides a solid theoretical foundation and practical methodology for weakly supervised MLC.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](../../ICML2026/optimization/a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[ICLR 2026\] In-Context Multi-Objective Optimization](in-context_multi-objective_optimization.md)
- [\[ICLR 2026\] Hierarchical Multi-Stage Recovery Framework for Kronecker Compressed Sensing](hierarchical_multi-stage_recovery_framework_for_kronecker_compressed_sensing.md)
- [\[ICLR 2026\] From Gradient Volume to Shapley Fairness: Towards Fair Multi-Task Learning](from_gradient_volume_to_shapley_fairness_towards_fair_multi-task_learning.md)

</div>

<!-- RELATED:END -->
