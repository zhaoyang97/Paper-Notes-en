---
title: >-
  [Paper Note] Gradient Extrapolation for Debiased Representation Learning
description: >-
  [ICCV 2025][Social Computing][debiasing] This paper proposes GERNE, a method that constructs two batches with different degrees of spurious correlation and performs linear extrapolation on their gradients to guide the model toward learning debiased representations, outperforming state-of-the-art methods under both known and unknown attribute settings.
tags:
  - ICCV 2025
  - Social Computing
  - debiasing
  - spurious correlations
  - gradient extrapolation
  - robustness
  - representation learning
date: 2026-05-08
content_hash: 2ce86bc3d4388539
---

# Gradient Extrapolation for Debiased Representation Learning

**Conference**: ICCV 2025
**arXiv**: [2503.13236](https://arxiv.org/abs/2503.13236)
**Code**: [Project Page](https://gerne-debias.github.io/)
**Area**: Social Computing
**Keywords**: debiasing, spurious correlations, gradient extrapolation, robustness, representation learning

## TL;DR

This paper proposes GERNE, a method that constructs two batches with different degrees of spurious correlation and performs linear extrapolation on their gradients to guide the model toward learning debiased representations, outperforming state-of-the-art methods under both known and unknown attribute settings.

## Background & Motivation

Deep learning classifiers trained with empirical risk minimization (ERM) often inadvertently rely on **spurious correlations**. For example, on the Waterbirds dataset, models may classify birds based on background (water/land) rather than the birds' intrinsic features. When such spurious associations are absent at test time, generalization performance degrades sharply.

**Limitations of Prior Work**:

**Methods requiring full attribute annotations** (e.g., Group DRO): directly minimize worst-group loss, but annotation costs are prohibitive.

**Methods using attributes only on validation sets** (e.g., DFR, JTT): infer pseudo-attributes from ERM-pretrained models, but precision is limited.

**Resampling/reweighting methods**: simple and effective, yet performance is constrained — models still preferentially learn "shortcut features" under strong spurious correlations.

**Key Challenge**: ERM optimizes average performance and naturally favors shortcut features predictive for the majority of samples. Even with balanced resampling, models tend to learn spurious features first.

**Core Idea**: From a model optimization perspective, the gradient difference between two batches with differing degrees of spurious correlation defines a "debiasing direction." The target gradient is set as a **linear extrapolation** along this direction. Tuning the extrapolation factor $\beta$ flexibly optimizes either Group-Balanced Accuracy or Worst-Group Accuracy.

## Method

### Overall Architecture

1. Construct two batch types: a biased batch $B_b$ (preserving the original spurious distribution) and a less-biased batch $B_{lb}$ (with more balanced attribute distribution).
2. Compute losses and gradients for each batch separately.
3. Define the target gradient as a linear extrapolation of the two gradients.
4. Update model parameters using the extrapolated gradient.

### Key Designs

1. **Batch Sampling Strategy**:

    - Biased batch: $p_b(a|y) = \alpha_{ya} = \frac{|\mathcal{X}_{y,a}|}{|\mathcal{X}_y|}$, reflecting the inherent data bias.
    - Less-biased batch: $p_{lb}(a|y) = \alpha_{ya} + c \cdot (\frac{1}{A} - \alpha_{ya})$, where $c \in (0,1]$ controls the degree of bias reduction.
    - Both batch types ensure uniform inter-class sampling and uniform intra-group sampling.

2. **Gradient Extrapolation**:

    - Target loss: $\mathcal{L}_{ext} = \mathcal{L}_{lb} + \beta \cdot (\mathcal{L}_{lb} - \mathcal{L}_b)$
    - Target gradient: $\nabla_\theta \mathcal{L}_{ext} = \nabla_\theta \mathcal{L}_{lb} + \beta \cdot (\nabla_\theta \mathcal{L}_{lb} - \nabla_\theta \mathcal{L}_b)$
    - Equivalent to simulating a conditional attribute distribution $p_{ext}(a|y) = \alpha_{ya} + c \cdot (\beta + 1) \cdot (\frac{1}{A} - \alpha_{ya})$.

3. **GERNE as a General Framework**:

    - $\beta = -1$: reduces to ERM.
    - $c = 1, \beta = 0$: equivalent to Resampling.
    - $c \cdot (\beta + 1) = 1$: expected equivalence to balanced sampling, but with different loss variance.
    - $c \cdot (\beta + 1) > 1$: **oversamples minority groups**, yielding stronger debiasing.

4. **Theoretical Bounds on $\beta$**:

    - Lower bound $\beta_{\min} = -1$ (degenerates to ERM).
    - Upper bound $\beta_{\max}$ determined by the maximum group proportion $\alpha_{y''a''}$ and $c$.
    - Increasing $\beta$ beyond $\frac{1}{c} - 1$ assigns higher weight to minority groups, optimizing worst-group risk.

5. **Unknown Attribute Setting**:

    - An ERM model $\tilde{f}$ is first trained; easy/hard samples are separated by prediction confidence to generate pseudo-attributes.
    - Gradient extrapolation can simulate conditional attribute distributions beyond the range of pseudo-group distributions (Proposition 1).

### Loss & Training

- Cross-entropy loss is used across all experiments.
- SGD optimizer (vision tasks); AdamW (NLP tasks).
- Hyperparameters $c$ and $\beta$ are tuned via grid search.
- In the unknown attribute setting, threshold $t$ serves as an additional hyperparameter.

## Key Experimental Results

### Main Results

C-MNIST and C-CIFAR-10 datasets (GBA %, known attributes):

| Method | C-MNIST 0.5% | C-MNIST 1% | C-CIFAR-10 0.5% | C-CIFAR-10 1% |
|------|-------------|------------|----------------|---------------|
| Group DRO | 63.12 | 68.78 | 33.44 | 38.30 |
| Resampling | 77.68 | 84.36 | 45.10 | 50.08 |
| **GERNE** | **77.79** | **84.47** | **45.34** | **50.84** |

Waterbirds / CelebA / CivilComments (WGA %, known attributes):

| Method | Waterbirds | CelebA | CivilComments |
|------|-----------|--------|---------------|
| Group DRO | 78.60 | 89.00 | 70.60 |
| DFR | 91.00 | 90.40 | 69.60 |
| **GERNE** | **90.20** | **91.98** | **74.65** |

### Ablation Study

Effect of $\beta$ on debiasing performance (C-MNIST 0.5%, $c=0.5$):

| $\beta$ | Equivalent Distribution | Minority Group Train Acc. | Unbiased Test Acc. | Stability |
|---------|---------|-------------|------------|--------|
| -1 | ERM | Low | ~35% | Stable |
| 0 | Weak debiasing | ~100% | ~70% | Stable |
| 1 | Strong debiasing | ~100% | ~77% | Stable |
| 1.2 | Near upper bound | Fluctuating | ~74% | High variance |
| >1.22 | Beyond bound | — | Diverges | Unusable |

Variance analysis comparing GERNE and Resampling shows that GERNE escapes sharp minima via controlled loss variance, whereas equivalent sampling with weighting approaches near-zero variance under extreme debiasing settings, making it susceptible to local optima.

### Key Findings

- GERNE outperforms Resampling by over 13% on bFFHQ, demonstrating that the gradient extrapolation direction is more effective than simple balancing.
- GERNE's advantage is most pronounced when minority samples are extremely scarce (0.5% minority ratio).
- In the unknown attribute setting, GERNE remains highly competitive, validating the effectiveness of pseudo-attribute generation combined with extrapolation.
- The choice of threshold $t$ influences the optimal $\beta$: higher-quality pseudo-attributes allow for a lower $\beta$.

## Highlights & Insights

- The unified framework that subsumes ERM and Resampling as special cases is an elegant design.
- The theoretical analysis is complete: derivations of upper and lower bounds on the extrapolation factor, and a direct connection to worst-group risk.
- Beyond simple balancing: extrapolation can simulate sampling with "reversed bias," which is unachievable by Resampling alone.
- The theoretical analysis of controllable loss variance provides a new perspective for understanding why extrapolation outperforms equivalent sampling.

## Limitations & Future Work

- The optimal value of $\beta$ is dataset-sensitive, and the feasible range narrows when $c$ is large.
- In the unknown attribute setting, performance depends on the quality of the ERM-pretrained model used to generate pseudo-attributes.
- There is no mechanism for dynamically adjusting $\beta$ — ideally it should adapt as training progresses.
- Performance on CelebA degrades substantially without validation attributes, indicating a strong dependence on validation set quality.
- Computational overhead: two batch gradients must be computed at each step.

## Related Work & Insights

- Group DRO directly optimizes worst-group risk but requires full annotations; GERNE can be viewed as a "softened" variant.
- The two-stage training idea in JTT is analogous to the pseudo-attribute generation in GERNE's unknown attribute approach.
- DFR's strategy of retraining the last layer on a validation set is complementary to GERNE.
- The gradient extrapolation idea can inspire robustness optimization in other domains.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel perspective of gradient extrapolation for debiasing.
- Technical Depth: ⭐⭐⭐⭐ Complete theoretical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across 6 benchmarks.
- Practical Value: ⭐⭐⭐⭐ Simple, effective, and easy to integrate.
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](../../CVPR2026/social_computing/learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[ICCV 2025\] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection](no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](../../ACL2026/social_computing/toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)
- [\[NeurIPS 2025\] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](../../NeurIPS2025/social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)

</div>

<!-- RELATED:END -->
