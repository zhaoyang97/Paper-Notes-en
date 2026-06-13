---
title: >-
  [Paper Note] Rethinking Consistent Multi-Label Classification Under Inexact Supervision
description: >-
  [ICLR 2026][Optimization][multi-label classification] The paper proposes the COMES framework, which provides consistent risk estimators for multi-label classification under inexact supervision via first-order (Hamming lo…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "multi-label classification"
  - "weakly supervised learning"
  - "partial multi-label learning"
  - "complementary multi-label learning"
  - "risk consistency"
date: 2026-05-08
content_hash: b2fc5652e064d799
---

# Rethinking Consistent Multi-Label Classification Under Inexact Supervision

**Conference**: ICLR 2026
**arXiv**: [2510.04091](https://arxiv.org/abs/2510.04091)  
**Area**: Optimization
**Keywords**: multi-label classification, weakly supervised learning, partial multi-label learning, complementary multi-label learning, risk consistency

## TL;DR

The paper proposes the COMES framework, which provides consistent risk estimators for multi-label classification under inexact supervision via first-order (Hamming loss) and second-order (Ranking loss) strategies, without requiring estimation of the label generation process or uniform distribution assumptions.

## Background & Motivation

Multi-label classification (MLC) requires associating each instance with multiple relevant labels, making annotation far more costly than single-label tasks. To reduce annotation burden, two weakly supervised paradigms have been proposed:

- **Partial Multi-Label Learning (PML)**: Each instance is annotated with a candidate label set containing all truly relevant labels and some irrelevant "false positive" labels.
- **Complementary Multi-Label Learning (CML)**: Each instance is annotated with complementary labels indicating the classes to which it does not belong.

A core observation is that PML and CML are mathematically equivalent — the complement of a candidate label set is precisely the complementary label set.

**Limitations of Prior Work**:

1. Existing methods require accurate estimation of the candidate/complementary label generation process (i.e., the transition matrix), but the overconfidence of deep neural networks makes such estimation unreliable.
2. Some methods assume a uniform distribution to bypass estimation, but this assumption is overly simplistic and cannot handle real-world class imbalance.
3. Many approaches model different labels independently, ignoring semantic correlations among labels.

## Method

### Overall Architecture: COMES

COMES (COnsistent Multi-label classification under inExact Supervision) introduces a new data generation process assumption and two risk estimation strategies.

### Data Generation Process

It is assumed that candidate labels are generated via per-class queries: if class $j$ is irrelevant to instance $\boldsymbol{x}$, it is annotated as a non-candidate label with a constant probability $p_j$:

$$p(j \notin S | \boldsymbol{x}, j \notin Y) = p_j$$

**Key Lemma (Lemma 1)**: Under this assumption, the conditional density of instances with non-candidate label $j$ is equivalent to the conditional density of instances irrelevant to class $j$:

$$p(\boldsymbol{x} | s_j = 0) = p(\boldsymbol{x} | y_j = 0)$$

This assumption is more general than the uniform distribution assumption, as the conditional probabilities for different candidate label sets need not be identical.

### First-Order Strategy: COMES-HL (Based on Hamming Loss)

MLC is decomposed into multiple independent binary classification problems. By Theorem 1, the $\ell$-risk of the Hamming loss can be equivalently expressed as:

$$R_H^\ell(\boldsymbol{g}) = \mathbb{E}_{p(\boldsymbol{x})}\left[\frac{1}{q}\sum_{j=1}^q \ell(g_j(\boldsymbol{x}), 1)\right] + \sum_{j=1}^q \mathbb{E}_{p(\boldsymbol{x}|s_j=0)}\left[\frac{1-\pi_j}{q}(\ell(g_j(\boldsymbol{x}), 0) - \ell(g_j(\boldsymbol{x}), 1))\right]$$

An unbiased risk estimator is obtained by constructing an unlabeled dataset $\mathcal{D}_U$ and conditional datasets $\mathcal{D}_j$. To prevent overfitting in deep networks, an absolute value function is applied to wrap the negative terms, yielding the corrected risk estimator $\tilde{R}_H^\ell$.

### Second-Order Strategy: COMES-RL (Based on Ranking Loss)

This strategy considers the pairwise ranking relationships among labels and exploits the symmetric loss assumption $\ell(z, \cdot) + \ell(-z, \cdot) = M$:

$$R_R^\ell(\boldsymbol{g}) = \sum_{1 \leq j < k \leq q}\left((1-\pi_j)\mathbb{E}_{p(\boldsymbol{x}|s_j=0)}[\ell(g_j - g_k, 0)] + (1-\pi_k)\mathbb{E}_{p(\boldsymbol{x}|s_k=0)}[\ell(g_j - g_k, 1)]\right)$$

Flooding regularization is applied to mitigate overfitting: $\tilde{R}_R^\ell = |\hat{R}_R^\ell - \beta| + \beta$.

### Loss & Training

- COMES-HL uses binary cross-entropy as the surrogate loss.
- COMES-RL uses a symmetric loss function (e.g., sigmoid loss).
- The class prior $\pi_j$ can be estimated from candidate labels using existing class prior estimation methods.

### Theoretical Guarantees

| Property | COMES-HL | COMES-RL |
|----------|----------|----------|
| Bias bound | $0 \leq \text{bias} \leq O(\Delta_j)$, $\Delta_j \to 0$ as $n \to \infty$ | $0 \leq \text{bias} \leq O(\Delta')$, $\Delta' \to 0$ as $n \to \infty$ |
| Estimation error convergence | $O(\mathfrak{R}_n(\mathcal{G}) + \sqrt{\ln(1/\delta)/n})$ | $O(\mathfrak{R}_n(\mathcal{G}) + \sqrt{\ln(1/\delta)/n})$ |
| Consistency | Bayes-optimal w.r.t. Hamming loss | Bayes-optimal w.r.t. Ranking loss |

## Key Experimental Results

### Main Results: Real-World Datasets (Ranking Loss ↓)

| Method | mirflickr | music_emotion | yeastBP | yeastCC | yeastMF |
|--------|-----------|---------------|---------|---------|---------|
| BCE | 0.106 | 0.244 | 0.328 | 0.206 | 0.251 |
| CCMN | 0.106 | 0.224 | 0.328 | 0.210 | 0.245 |
| GDF | 0.159 | 0.278 | 0.501 | 0.504 | 0.495 |
| CTL | 0.130 | 0.266 | 0.498 | 0.467 | 0.471 |
| **COMES-HL** | **0.095** | **0.214** | **0.154** | **0.124** | 0.173 |
| **COMES-RL** | 0.106 | **0.213** | 0.166 | **0.117** | **0.151** |

### Method Comparison

| Method | No Uniform Assumption | No Generation Process Estimation | Label Correlation-Aware | Multiple Complementary Labels |
|--------|-----------------------|----------------------------------|------------------------|-------------------------------|
| CCMN | ✓ | ✗ | ✓ | ✓ |
| CTL | ✗ | ✓ | ✗ | ✗ |
| GDF | ✗ | ✓ | ✗ | ✓ |
| **COMES-HL** | **✓** | **✓** | ✗ | **✓** |
| **COMES-RL** | **✓** | **✓** | **✓** | **✓** |

### Key Findings

1. COMES-HL and COMES-RL consistently outperform state-of-the-art methods across 5 evaluation metrics on 6 real-world datasets.
2. The advantage is particularly pronounced on the yeast benchmark series, with Ranking Loss reduced by approximately 50% (e.g., yeastBP: 0.154 vs. 0.328).
3. COMES-RL achieves superior performance on datasets with strong label correlations (e.g., yeastMF: 0.151 vs. 0.173).
4. Both methods demonstrate robustness on synthetic data generated under both uniform and non-uniform label generation processes.

## Highlights & Insights

1. **Strong Theoretical Contributions**: This work is the first to establish consistency for MLC under inexact supervision without relying on transition matrix estimation or uniform distribution assumptions.
2. **Unified Treatment of PML and CML**: The mathematical equivalence between the two paradigms is exploited to address both problems within a single framework.
3. **Complementary First- and Second-Order Strategies**: COMES-HL is computationally efficient but ignores label correlations, while COMES-RL leverages pairwise ranking relationships at higher computational cost, making the two strategies suitable for different scenarios.
4. **Practical Data Generation Assumption**: The per-class query independence assumption more faithfully reflects realistic annotation workflows.
5. **Elegant Corrected Risk Estimator Design**: Absolute value wrapping and flooding regularization respectively resolve overfitting issues in the first-order and second-order strategies.

## Limitations & Future Work

1. The framework focuses solely on Hamming loss and Ranking loss, leaving other MLC evaluation metrics (e.g., F1-measure) unaddressed.
2. The quality of class prior estimation $\pi_j$ affects final performance, yet the propagation of estimation error is not thoroughly analyzed.
3. The assumption that $p_j$ is constant in the data generation process may not fully capture complex annotation behaviors.
4. The computational complexity of the second-order strategy is $O(q^2)$, which may become a bottleneck when the label space is large.
5. The datasets used in experiments are relatively small-scale; performance on large-scale benchmarks remains to be validated.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The framework design unifying PML/CML while removing strong assumptions is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive outperformance of state-of-the-art across multiple datasets and metrics, though dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and complete; the structure is clear.
- **Value**: ⭐⭐⭐⭐ — Provides a solid theoretical foundation and practical methodology for weakly supervised MLC.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](../../ICML2026/optimization/a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)
- [\[AAAI 2026\] Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment](../../AAAI2026/optimization/cost-minimized_label-flipping_poisoning_attack_to_llm_alignment.md)
- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)

</div>

<!-- RELATED:END -->
