---
title: >-
  [Paper Note] Principled Algorithms for Optimizing Generalized Metrics in Binary Classification
description: >-
  [ICML 2025][Learning Theory][generalized metrics] This paper proposes METRO, a principled algorithm for optimizing generalized classification metrics (such as $F_\beta$, Jaccard, weighted accuracy, etc.). Based on $H$-consistency bounds and surrogate loss theory, it reformulates metric optimization as a generalized cost-sensitive learning problem, providing finite-sample generalization guarantees.
tags:
  - "ICML 2025"
  - "Learning Theory"
  - "Classification Optimization"
  - "generalized metrics"
  - "F-measure"
  - "H-consistency"
  - "surrogate loss"
  - "cost-sensitive learning"
date: 2026-05-08
content_hash: 1972ec281805c755
---

# Principled Algorithms for Optimizing Generalized Metrics in Binary Classification

**Conference**: ICML 2025  
**arXiv**: [2512.23133](https://arxiv.org/abs/2512.23133)  
**Code**: None  
**Area**: Learning Theory / Classification Optimization  
**Keywords**: generalized metrics, F-measure, H-consistency, surrogate loss, cost-sensitive learning

## TL;DR
This paper proposes METRO, a principled algorithm for optimizing generalized classification metrics (such as $F_\beta$, Jaccard, weighted accuracy, etc.). Based on $H$-consistency bounds and surrogate loss theory, it reformulates metric optimization as a generalized cost-sensitive learning problem, providing finite-sample generalization guarantees.

## Background & Motivation
**Background**: In scenarios with class imbalance or asymmetric costs, generalized metrics such as the $F_\beta$-score, AM measure, and Jaccard index are more appropriate than the standard 0-1 loss. However, these metrics are typically non-decomposable and cannot be written as a sum of single-sample losses.

**Limitations of Prior Work**: Existing methods often rely on characterizations of the Bayes optimal classifier, estimating class probabilities first and then searching for the optimal threshold. This leads to: (1) algorithms not tailored to restricted hypothesis sets; (2) a lack of finite-sample performance guarantees.

**Key Challenge**: Minimizing the target metrics directly is theoretically required, but these metrics are non-convex and non-decomposable, making direct optimization difficult.

**Goal**: Provide principled algorithms for optimizing generalized metrics with both $H$-consistency and finite-sample generalization bounds.

**Key Insight**: Reformulate metric optimization as generalized cost-sensitive learning and design surrogate losses with provable $H$-consistency.

**Core Idea**: For each generalized metric, there exists an equivalent cost-sensitive learning problem that can be efficiently optimized using tailored surrogate losses.

## Method

### Overall Architecture
Input: Training data $\{(x_i, y_i)\}_{i=1}^n$, target generalized metric $\Psi$ (e.g., $F_\beta$)  
Output: Classifier $h \in H$ optimizing this metric  

Pipeline:
1. Decompose the generalized metric $\Psi$ into a function of the confusion matrix elements (TP, FP, FN, TN).
2. Construct an equivalent cost-sensitive learning problem where the cost parameters depend on the unknown prior distribution.
3. Design surrogate loss functions with $H$-consistency.
4. Alternately optimize the cost parameters and the classifier using the METRO algorithm.

### Key Designs

1. **Reduction from Generalized Metrics to Cost-Sensitive Learning**:
    - **Function**: Prove that optimizing $\Psi$ is equivalent to solving a specific cost-sensitive classification problem.
    - **Mechanism**: For metrics like $F_\beta = \frac{(1+\beta^2) \text{TP}}{(1+\beta^2)\text{TP} + \beta^2 \text{FN} + \text{FP}}$, the Bayes optimal solution can be represented as a threshold classifier $h(x) = \mathbb{1}[\eta(x) > c^*]$, where the threshold $c^*$ is a function of the cost parameters. This is generalized to cost-sensitive learning: $\min_h \mathbb{E}[c \cdot \mathbb{1}[h(x) \neq y]]$, where the cost $c$ depends on the metric and class priors.
    - **Design Motivation**: Cost-sensitive learning has a mature theoretical foundation; after reduction, existing surrogate loss theories can be leveraged.

2. **$H$-Consistent Surrogate Loss**:
    - **Function**: Design new surrogate loss functions that satisfy consistency for restricted hypothesis sets $H$.
    - **Mechanism**: $H$-consistency requires that the classifier obtained by minimizing the surrogate loss also converges to the optimum of the target metric. This paper proves that a class of convex surrogate losses parameterized by cost possesses this property:
    $$\ell_c(h(x), y) = c \cdot \phi(y \cdot h(x))$$
    where $\phi$ is an appropriately designed convex decreasing function.
    - **Design Motivation**: Standard cross-entropy loss is not necessarily $H$-consistent for generalized metrics, necessitating specially designed losses.

3. **METRO Algorithm**:
    - **Function**: Alternately optimize the cost parameter $c$ and the classifier $h$.
    - **Mechanism**:
        - Step 1: Fix the classifier $h$, and update the cost parameter $c$ to match the optimal cost under the current classifier.
        - Step 2: Fix the cost $c$, and train the classifier $h$ using the surrogate loss.
        - Repeat until convergence.
    - **Design Motivation**: Cost parameters depend on distributional information (such as $P(Y=1)$) and need to be estimated from data. Alternating optimization naturally addresses this mutual dependence.

### Loss & Training
- Surrogate loss: $\hat{R}_\ell(h) = \frac{1}{n} \sum_{i=1}^n c_i \cdot \phi(y_i h(x_i))$
- Finite-sample generalization bound: $\Psi(h) \leq \Psi(h^*) + O(\sqrt{\text{complexity}(H)/n})$

## Key Experimental Results

### Main Results

| Dataset | Metric | METRO | Threshold Search | Direct Opt. Baseline | Standard CE |
|---|---|---|---|---|---|
| Credit Default | $F_1$ | **0.523** | 0.498 | 0.486 | 0.451 |
| Medical Diagnosis | $F_2$ | **0.714** | 0.688 | 0.670 | 0.632 |
| Fraud Detection | Jaccard | **0.381** | 0.355 | 0.342 | 0.310 |
| Synthetic Imbalanced | AM | **0.862** | 0.841 | 0.830 | 0.795 |

### Ablation Study

| Configuration | $F_1$ | Description |
|---|---|---|
| METRO (Full) | **0.523** | Cost-sensitive + consistent loss |
| No cost update (fixed $c$) | 0.495 | Cost needs to be adaptive |
| Standard CE + Threshold Search | 0.498 | Traditional approach |
| METRO + Non-consistent surrogate | 0.507 | Consistency guarantee is beneficial |

### Key Findings
- METRO outperforms baseline methods across all metrics and datasets, showing more pronounced advantages in highly imbalanced scenarios.
- The $H$-consistent surrogate loss yields a significant improvement over standard surrogate losses, validating the theoretical analysis.
- Adaptive updating of the cost parameter is crucial (ablation shows that fixed costs perform significantly worse than adaptive ones).
- Finite-sample generalization bounds are tight, with practical performance matching the theoretical bounds.

## Highlights & Insights
- Theoretical completeness: Concurrently achieves both $H$-consistency and finite-sample bounds, a first among existing methods.
- Practicality: The METRO algorithm is simple and efficient, requiring only an added cost-updating step within standard training loops.
- Broad applicability: Unifies the handling of various metrics, including $F_\beta$, Jaccard, AM measure, and weighted accuracy.

## Limitations & Future Work
- Currently restricted to binary classification; extension to multi-class classification is an important direction for future work.
- Cost parameter estimation may be unstable under extreme class imbalance.
- Integration with deep learning (e.g., joint fine-tuning with pre-trained models) warrants further exploration.

## Related Work & Insights
- Complements the thresholding methods of Koyejo et al. (2014) and Narasimhan et al. (2014).
- The concept of $H$-consistency originates from Awasthi et al. (2022); this work is the first to apply it to generalized metric optimization.
- Offers direct guiding value for imbalanced classification in practice.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The reduction framework and consistency analysis are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple metrics and datasets.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear theoretical exposition and concise algorithm description.
- **Value**: ⭐⭐⭐⭐ Addresses an important practical problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Generalized Geometric Theoretical Framework of Centroid Discriminant Analysis for Linear Classification of Multi-dimensional Data](../../ICLR2026/learning_theory/a_generalized_geometric_theoretical_framework_of_centroid_discriminant_analysis_.md)
- [\[ICLR 2026\] Optimizing Data Augmentation through Bayesian Model Selection](../../ICLR2026/learning_theory/optimizing_data_augmentation_through_bayesian_model_selection.md)
- [\[ICML 2025\] Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)
- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](../../ICLR2026/learning_theory/conformal_prediction_for_long-tailed_classification.md)
- [\[NeurIPS 2025\] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](../../NeurIPS2025/learning_theory/a_highdimensional_statistical_method_for_optimizing_transfer.md)

</div>

<!-- RELATED:END -->
