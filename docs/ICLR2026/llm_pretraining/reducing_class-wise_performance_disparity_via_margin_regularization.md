---
title: >-
  [Paper Note] Reducing Class-Wise Performance Disparity via Margin Regularization
description: >-
  [LLM Pretraining] This paper proposes MR2 (Margin Regularization for performance disparity Reduction), which dynamically adjusts class-dependent margins in both the logit and representation spaces. Grounded in theoretically derived generalization bounds, MR2 reduces class-wise performance disparity while simultaneously improving overall accuracy.
tags:
  - LLM Pretraining
date: 2026-05-08
content_hash: 0000ff3fbfe5b94c
---

# Reducing Class-Wise Performance Disparity via Margin Regularization

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2602.00205](https://arxiv.org/abs/2602.00205)
- **Code**: [https://github.com/BeierZhu/MR2](https://github.com/BeierZhu/MR2)
- **Area**: LLM Pre-training
- **Keywords**: class-wise disparity, margin regularization, generalization bound, Rademacher complexity, representation learning

## TL;DR
This paper proposes MR2 (Margin Regularization for performance disparity Reduction), which dynamically adjusts class-dependent margins in both the logit and representation spaces. Grounded in theoretically derived generalization bounds, MR2 reduces class-wise performance disparity while simultaneously improving overall accuracy.

## Background & Motivation
- Deep networks exhibit severe class-wise accuracy disparity even when trained on class-balanced data. For instance, ResNet-50 on ImageNet achieves 100% accuracy on the best class but only 16% on the worst.
- Prior work has identified that "hard" classes (with low accuracy) tend to exhibit greater feature variability (Figure 1b), yet proposed solutions remain largely empirical (e.g., data augmentation, representation learning) and lack theoretical grounding.
- Existing margin-based methods (LDAM, Logit Adjustment, etc.) are designed for imbalanced classification and degenerate to standard cross-entropy under class balance, failing to address performance disparity.

## Method

### Overall Architecture
MR2 applies margin regularization at two levels:

### Key Design 1: Logit Margin Loss
$$\ell_{\bm{\gamma},\mathsf{ce}}(f, \mathbf{x}, y) = -\mathbf{1}_y^\top \ln[\text{softmax}(\mathbf{z} / \gamma_y)]$$

The class-dependent margin is defined as:
$$\gamma_y = \frac{\bar{c} \cdot K (\|\hat{\bm{\mu}}_y\|_2^2 + \|\hat{\mathbf{s}}_y\|_2^2)^{1/3}}{\sum_{k=1}^K (\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2)^{1/3}}$$

- $\hat{\bm{\mu}}_k$: feature mean of class $k$
- $\|\hat{\mathbf{s}}_k\|_2^2$: mean squared deviation of class $k$
- "Hard" classes with greater feature variability receive larger margins, leading to better generalization.

### Key Design 2: Representation Margin Loss
$$\ell_{\bar{s}}(f, \mathbf{x}, y) = \ln\left[1 + \sum_{\mathbf{x}^+ \in \mathcal{D}_y \setminus \{\mathbf{x}\}} \exp(\|\phi(\mathbf{x}) - \phi(\mathbf{x}^+)\|_2^2 - 2\bar{s})\right]$$

The average mean squared deviation $2\bar{s}$ serves as the margin, encouraging intra-class compactness. This is equivalent to minimizing intra-class mean squared deviation.

### Overall Objective
$$\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{\mathbf{x},y \in \mathcal{D}} [\ell_{\bm{\gamma},\mathsf{ce}}(f, \mathbf{x}, y) + \lambda \cdot \ell_{\bar{s}}(f, \mathbf{x}, y)]$$

### Theoretical Foundation
**Proposition 1 (Class-Sensitive Generalization Bound)**:
$$\mathcal{R}(f) \leq \frac{1}{\ln 2} \hat{\mathcal{R}}_{\mathcal{D}}^{\bm{\gamma},\mathsf{ce}}(f) + \frac{4\sqrt{2}\Lambda K}{\sqrt{N}} \sqrt{\sum_{k=1}^K \frac{\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2}{\gamma_k^2}} + \mathcal{O}(1/\sqrt{N})$$

**Corollary 1**: Under a fixed average margin budget, $\gamma_k \propto (\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2)^{1/3}$ minimizes the complexity term.

## Key Experimental Results

### Main Results: CIFAR-100 & ImageNet

| Method | Overall Acc. | Easy | Medium | Hard |
|--------|-------------|------|--------|------|
| ERM (standard training) | 70.9 | 84.5 | 71.0 | 56.7 |
| LfF | 69.1 | 83.6 (−0.9) | 70.1 (−0.9) | 53.7 (−3.0) |
| JTT | 70.6 | 84.3 (−0.2) | 70.8 (−0.2) | 56.2 (−0.5) |
| DRO | ~70.0 | decreased | ~71.0 | ~56.0 |
| **MR2 (Ours)** | **71.8** | **85.0 (+0.5)** | **72.0 (+1.0)** | **58.5 (+1.8)** |

> MR2 substantially improves performance on "hard" classes (+1.8) while also boosting "easy" classes (+0.5), requiring no trade-off between the two.

### Ablation Study: Pre-trained Backbone + Fine-tuning Paradigm

| Backbone + Paradigm | ERM | MR2 | Hard Gain |
|--------------------|-----|-----|-----------|
| MAE (end-to-end) | baseline | improved | significant |
| MoCov2 (linear probe) | baseline | improved | significant |
| CLIP (linear probe) | baseline | improved | significant |
| ResNet-50 (from scratch) | 70.9 | **71.8** | +1.8 |
| ViT-B/16 (from scratch) | baseline | improved | significant |

> MR2 is effective across all pre-training methods (MAE/MoCov2/CLIP) and training paradigms (end-to-end / linear probe).

### Key Findings
1. Existing debiasing methods (LfF, JTT, DRO) typically sacrifice "easy" class performance when improving "hard" classes — MR2 eliminates this trade-off.
2. Logit margin and representation margin are complementary: the former allocates larger generalization budgets to hard classes, while the latter reduces intra-class variability.
3. The theoretically derived $\gamma_k$ closely aligns with the empirically optimal values obtained via grid search.
4. Even on L2-normalized CLIP features, class-sensitive margins can be recovered using $L_p$ norms with $p \neq 2$.

## Highlights & Insights
- **Theory-driven design**: Margin formulations are derived from generalization bounds rather than empirical heuristics.
- **No accuracy trade-off**: MR2 simultaneously improves both hard and easy classes, a property rarely observed in fairness or debiasing methods.
- **Broad applicability**: Consistent gains are demonstrated across 7 datasets, CNN/ViT architectures, and diverse pre-training paradigms.
- **Orthogonal to long-tail methods**: MR2 remains meaningful under class-balanced settings, addressing a theoretical gap regarding performance disparity in balanced data.

## Limitations & Future Work
- Maintaining class statistics via EMA introduces modest computational overhead.
- The representation margin loss requires intra-class sample pairing, which may be unstable for classes with very few samples.
- Theoretical analysis assumes uniformly bounded classifier weight norms ($\Lambda$), which may not hold strictly in all models.
- Hyperparameters $\bar{c}$ and $\lambda$ still require tuning.

## Related Work & Insights
- **Long-tail classification margins**: LDAM (Cao et al., 2019), Logit Adjustment (Menon et al., 2021), Balanced Softmax (Ren et al., 2020)
- **Performance disparity analysis**: Cui et al. (2024) attribute disparity to representation rather than classifier bias
- **Neural Collapse**: The idealized assumptions of Papyan et al. (2020) do not hold on large-scale datasets
- **Contrastive learning**: SupCon (Khosla et al., 2020) does not incorporate margin constraints

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Margin regularization under class-balanced data, unifying theoretical derivation with empirical insight
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Complete Rademacher complexity analysis with rigorous generalization bounds
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 7 datasets, multiple architectures, diverse pre-training paradigms, and detailed ablations
- **Value**: ⭐⭐⭐⭐ — Plug-and-play, open-source implementation, broadly applicable to classification models

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](understanding_and_improving_shampoo_and_soap_via_kullback-leibler_minimization.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](moma_a_modular_deep_learning_framework_for_material_property_prediction.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](intrinsic_training_dynamics_of_deep_neural_networks.md)

<!-- RELATED:END -->
