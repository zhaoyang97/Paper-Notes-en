---
title: >-
  [Paper Note] Improved Balanced Classification with Theoretically Grounded Loss Functions
description: >-
  [NeurIPS 2025][AI Safety][Balanced classification loss] Two theory-driven surrogate loss families are proposed—Generalized Logit-Adjusted (GLA) loss and Generalized Class-Aware weighted (GCA) loss—providing stronger theoretical guarantees and improved empirical performance for multi-class classification under class imbalance.
tags:
  - NeurIPS 2025
  - AI Safety
  - Balanced classification loss
  - surrogate loss
  - H-consistency
  - logit adjustment
  - class-aware weighting
date: 2026-05-08
content_hash: 19c6278822ac00a3
---

# Improved Balanced Classification with Theoretically Grounded Loss Functions

**Conference**: NeurIPS 2025
**arXiv**: [2512.23947](https://arxiv.org/abs/2512.23947)
**Authors**: Corinna Cortes, Mehryar Mohri, Yutao Zhong
**Code**: None
**Area**: Machine Learning Theory / Class-Imbalanced Classification
**Keywords**: Balanced classification loss, surrogate loss, H-consistency, logit adjustment, class-aware weighting

## TL;DR

Two theory-driven surrogate loss families are proposed—Generalized Logit-Adjusted (GLA) loss and Generalized Class-Aware weighted (GCA) loss—providing stronger theoretical guarantees and improved empirical performance for multi-class classification under class imbalance.

## Background & Motivation

Class imbalance is a pervasive challenge in multi-class classification. Balanced classification losses promote fairness by assigning equal importance to all classes, ensuring that minority classes are not neglected. However, directly minimizing the balanced classification loss is generally intractable, making the design of effective surrogate losses a central problem.

Existing surrogate loss methods suffer from the following limitations:

**Standard class-weighted loss**: Scales losses by the inverse of class frequencies, with limited theoretical guarantees.

**Logit-Adjusted (LA) loss**: Shifts logits according to class prior probabilities and is effective within the standard cross-entropy family, but H-consistency holds only for the complete (unbounded) hypothesis class.

**Insufficient theoretical guarantees**: The H-consistency bounds of existing methods depend on the inverse of the minimum class probability $p_{\min}$, which degrades severely in highly imbalanced settings.

The core motivation of this paper is: can surrogate losses be designed with stronger theoretical guarantees under broader conditions while maintaining strong empirical performance?

## Method

### Overall Architecture

Two surrogate loss families are proposed, both extended to the generalized cross-entropy loss family:

1. **GLA (Generalized Logit-Adjusted)**: Extends logit adjustment to the generalized cross-entropy family.
2. **GCA (Generalized Class-Aware weighted)**: Introduces class-dependent confidence margins, extending the standard class-weighted loss.

### Key Designs

#### 1. Generalized Logit-Adjusted Loss (GLA)

The standard LA loss adjusts logits by adding an offset $\log p_c$ for each class $c$:

$$\ell_{\text{LA}}(h, x, y) = -\log \frac{e^{h_y(x) + \log p_y}}{\sum_c e^{h_c(x) + \log p_c}}$$

GLA generalizes this to the generalized cross-entropy loss family $\Phi$, allowing a more general convex function $\Phi$ to replace $\log$:

$$\ell_{\text{GLA}}(h, x, y) = \Phi\left(\frac{e^{h_y(x) + \tau_y}}{\sum_c e^{h_c(x) + \tau_c}}\right)$$

where the offsets $\tau_c$ may take values other than $\log p_c$ and can be analyzed uniformly within the generalized cross-entropy framework.

**Theoretical properties**:
- Bayes consistency: GLA loss is Bayes consistent.
- H-consistency: Holds only for the complete (unbounded) hypothesis class.
- H-consistency bound: Depends on $1/p_{\min}$; insufficiently tight in imbalanced settings.

#### 2. Generalized Class-Aware Weighted Loss (GCA)

GCA introduces two key innovations over the standard class-weighted loss:

**a. Class-dependent confidence margins**: A distinct confidence margin $m_c$ is assigned to each class rather than a uniform threshold:

$$\ell_{\text{GCA}}(h, x, y) = \frac{1}{p_y} \Phi\left(\frac{e^{h_y(x) - m_y}}{\sum_c e^{h_c(x) - m_c}}\right)$$

These margins can be calibrated via theoretical analysis, assigning larger margins to minority classes and stricter margins to majority classes.

**b. Generalized cross-entropy extension**: Analogously to GLA, the loss is extended to a broader family of convex functions.

**Theoretical properties**:
- H-consistency: Holds for any bounded or complete hypothesis class (broader than GLA).
- H-consistency bound: Depends on $1/\sqrt{p_{\min}}$, which is superior to GLA's $1/p_{\min}$.
- Provides significantly stronger theoretical guarantees in highly imbalanced settings.

#### 3. Theoretical Analysis Framework

The core theoretical tool is H-consistency bounds, which quantify the gap between the minimizer of the surrogate loss and the balanced classification error:

| Loss Type | Bayes Consistent | H-consistency Condition | Bound Dependence |
|-----------|-----------------|------------------------|-----------------|
| Standard class-weighted | Yes | Bounded/Complete | Baseline |
| LA (original) | Yes | Complete only | $1/p_{\min}$ |
| GLA (Ours) | Yes | Complete only | $\geq 1/p_{\min}$ |
| GCA (Ours) | Yes | Bounded/Complete | $1/\sqrt{p_{\min}}$ |

### Loss & Training

Margin calibration strategy for GCA:
- Margins $m_c$ are set according to class frequencies $p_c$.
- Minority classes receive larger margins (lower confidence requirements).
- Margin selection is designed to optimize H-consistency bounds.
- Fine-tuning via cross-validation on a validation set is supported.

## Key Experimental Results

### Main Results

#### Standard Imbalanced Classification Benchmarks

| Method | CIFAR-10-LT (IF=100) | CIFAR-100-LT (IF=100) | ImageNet-LT | Theoretical Guarantee |
|--------|---------------------|----------------------|-------------|----------------------|
| Standard class-weighted | Baseline | Baseline | Baseline | Bounded/Complete |
| LA (original) | Above baseline | Above baseline | Above baseline | Complete only |
| **GLA (Ours)** | **Typically best** | **Typically best** | **Typically best** | Complete only |
| **GCA (Ours)** | Near-best | Near-best | Near-best | Bounded/Complete |

IF = Imbalance Factor, denoting the ratio of sample counts between the largest and smallest class.

#### Performance Under Extreme Imbalance

| Method | IF=10 | IF=50 | IF=100 | IF=200 |
|--------|-------|-------|--------|--------|
| Standard class-weighted | Baseline | Baseline | Baseline | Baseline |
| LA loss | +small | +moderate | +moderate | +moderate |
| **GLA** | +large | +large | **Best** | Near-best |
| **GCA** | +moderate | +large | Near-best | **Best** |

Key observation: GLA tends to perform marginally better on common benchmarks, while GCA exhibits a marginal advantage under extreme imbalance (IF≥100), consistent with theoretical analysis—the $1/\sqrt{p_{\min}}$ bound of GCA is more favorable under severe imbalance.

### Ablation Study

#### Effect of Margin Calibration

| GCA Variant | No margin | Uniform margin | Calibrated (theory) | Calibrated (val set) |
|-------------|-----------|---------------|--------------------|--------------------|
| Balanced accuracy | Baseline | +small | +moderate | **+largest** |

#### Choice of Generalized Cross-Entropy Function $\Phi$

| $\Phi$ Choice | GLA Performance | GCA Performance | Characteristics |
|---------------|----------------|----------------|----------------|
| Standard log | Baseline | Baseline | Classical cross-entropy |
| Polynomial | Slightly higher | Slightly higher | Smooth gradients |
| Exponential | Similar | Similar | Emphasizes hard samples |

### Key Findings

1. **Complementarity of GLA and GCA**: GLA performs slightly better on common benchmarks; GCA excels under extreme imbalance.
2. **Theory–empirical alignment**: The tightness of H-consistency bounds corresponds to observed performance differences.
3. **Strength of class-weighted baseline**: Simple class weighting is already a strong baseline, but GLA/GCA yield further gains.
4. **Importance of margin calibration**: GCA performance is substantially dependent on correct margin specification.

## Highlights & Insights

1. **Theoretical rigor**: The team from Google Research (Cortes is a co-inventor of SVMs) provides complete theoretical analysis.
2. **Improvement in H-consistency bounds**: GCA's $1/\sqrt{p_{\min}}$ bound represents a fundamental improvement over LA's $1/p_{\min}$.
3. **Practical contribution**: GLA and GCA serve as direct drop-in replacements for existing loss functions.
4. **Importance of hypothesis class**: The paper reveals the critical distinction between bounded and complete hypothesis classes in loss consistency analysis.
5. **Adaptive to imbalance degree**: GCA adapts to varying imbalance levels through margin calibration.

## Limitations & Future Work

1. **Extreme long-tail scenarios**: Performance with more than 1,000 classes in extreme long-tail settings remains untested.
2. **Combination with other long-tail methods**: Whether GLA/GCA can complement decoupled training, data augmentation, and similar approaches is unexplored.
3. **Fine-tuning large models**: Performance under pretrain-then-finetune paradigms has not been validated.
4. **Computational overhead**: Margin calibration increases the cost of hyperparameter tuning.
5. **Theory-practice gap**: The theoretically optimal choice of $\Phi$ does not fully align with the empirically optimal choice.

## Related Work & Insights

- **Logit Adjustment** (Menon et al., 2021): The original LA loss, generalized in this work.
- **Class-Balanced Loss** (Cui et al., 2019): A classical class-weighted loss; GCA is its theoretically enhanced counterpart.
- **H-consistency bounds** (Awasthi et al., 2022): The core theoretical tool, applied in depth to the imbalanced classification setting.
- **Focal Loss** (Lin et al., 2017): An alternative approach to imbalance, orthogonal to the proposed methods.
- **Prior work** (Mao, Mohri, Zhong, 2023–2024): Theoretical contributions from the same team on problems such as multi-class abstention.

## Rating

- **Novelty**: ★★★★☆ — Class-dependent margin design in GCA and stronger theoretical guarantees.
- **Theoretical Depth**: ★★★★★ — Rigorous and complete H-consistency analysis.
- **Experimental Thoroughness**: ★★★★☆ — Validated across multiple imbalance scales and datasets.
- **Value**: ★★★★☆ — Direct replacement for existing loss functions.
- **Writing Quality**: ★★★★★ — From a leading theory group; presentation is clear and well-structured.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] Robust Graph Condensation via Classification Complexity Mitigation](robust_graph_condensation_via_classification_complexity_mitigation.md)
- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[NeurIPS 2025\] Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)

</div>

<!-- RELATED:END -->
