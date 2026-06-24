---
title: >-
  [Paper Note] KAC: Kolmogorov-Arnold Classifier for Continual Learning
description: >-
  [CVPR 2025][Physics & Scientific Computing][KAN] First to apply Kolmogorov-Arnold Networks (KAN) to continual learning. By replacing B-splines with Radial Basis Functions (RBF) to construct the classifier KAC, consistent and significant performance gains are achieved across multiple continual learning methods (up to +20.70% on CUB200 40-step) with only 0.23M additional parameters.
tags:
  - "CVPR 2025"
  - "Physics & Scientific Computing"
  - "KAN"
  - "continual learning"
  - "RBF"
  - "Kolmogorov-Arnold"
  - "classifier"
date: 2026-05-08
content_hash: 7bc564c3b69088a0
---

# KAC: Kolmogorov-Arnold Classifier for Continual Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.21076](https://arxiv.org/abs/2503.21076)  
**Code**: [https://github.com/Ethanhuhuhu/KAC](https://github.com/Ethanhuhuhu/KAC)  
**Area**: LLM Evaluation  
**Keywords**: KAN, continual learning, RBF, Kolmogorov-Arnold, classifier

## TL;DR
First to apply Kolmogorov-Arnold Networks (KAN) to continual learning. By replacing B-splines with Radial Basis Functions (RBF) to construct the classifier KAC, consistent and significant performance gains are achieved across multiple continual learning methods (up to +20.70% on CUB200 40-step) with only 0.23M additional parameters.

## Background & Motivation

**Background**: Continual learning methods based on pre-trained ViT (L2P, DualPrompt, CODAPrompt, CPrompt) have achieved significant progress in incremental learning, but they typically use simple linear classification heads, which limits the discriminative capability of the model between old and new tasks.

**Limitations of Prior Work**: Linear classifiers have limited representation capacity and cannot fully model complex class boundaries. This issue becomes particularly severe as the number of tasks increases—performance drops sharply in the 40-step incremental learning scenario. Replacing them with MLPs yields even worse results (accuracy decreased by 0.36% under the same parameter budget), indicating that simply increasing non-linear layers is not the solution.

**Key Challenge**: Continual learning requires classifiers to possess strong representation capacity to distinguish a large number of accumulated classes, while remaining lightweight enough to avoid overfitting on limited data. Linear classifiers are too weak, whereas MLPs are prone to overfitting.

**Goal**: Design a lightweight yet powerful classifier that can improve existing continual learning methods in a plug-and-play manner.

**Key Insight**: The Kolmogorov-Arnold representation theorem indicates that any continuous multivariate function can be represented as a combination and addition of data-efficient univariate functions. KAN provides more flexible function approximation capability by placing learnable activation functions on the edges.

**Core Idea**: Replace B-splines in KAN with RBFs, leveraging the Gaussian mixture interpretation and localization characteristics of RBFs to construct a highly expressive continual learning classifier with minimal parameters.

## Method

### Overall Architecture
KAC replaces the linear classification head in existing continual learning methods in a plug-and-play manner. The input consists of features extracted by the ViT backbone, which pass through Layer Normalization and then enter the KAC classifier to output class predictions. Only the classification head is modified, keeping the backbone and prompt mechanisms unchanged.

### Key Designs

1. **RBF-based KAN Layer**

    - Function: Replace B-spline activation functions in KAN with radial basis functions (RBF).
    - Mechanism: paper_notes/docs/CVPR2025/ai_safety/deal_data-efficient_adversarial_learning_for_high-quality_infrared_imaging.md(x) = \sum_p \Phi_p \sum_{i=1}^N \omega_{p,i} \varphi(||x_p - c_i||)$, where $\varphi(r) = \exp(-r^2/(2\sigma_i^2))$ is the Gaussian RBF
    - Distribution Interpretation: $\sum_i \omega_{p,i} \mathcal{N}(c_i, \sigma_i^2)$ is equivalent to a Gaussian Mixture Model
    - Design Motivation: B-splines require frequent grid updates in continual learning, which is unstable; RBF centers are fixed (uniformly distributed from -2 to 2) and widths are fixed ($\sigma=1$), requiring only the weights $\omega$ to be learned

2. **KAC Architecture Design**

    - Function: Instantiate RBF-KAN specifically as a classifier
    - Mechanism: Input goes through LayerNorm $\rightarrow$ $N=4$ Gaussian RBF mappings $\rightarrow$ $\text{Diag}(W_C \cdot \Phi(\text{LN}(F(x))) \cdot W_q)$
    - $W_q \in \mathbb{R}^{C \times n}$ is the output weight, and $W_C \in \mathbb{R}^{N \times C}$ represents the class-specific RBF weights
    - Design Motivation: Diagonalization ensures that each class has an independent activation pattern, adding only 0.23M parameters (vs. 86M for ViT-B/16)

### Loss & Training
- Directly employ the loss functions of the original continual learning methods; KAC only replaces the classification head
- No additional regularization or distillation losses are introduced
- Compatible with L2P, DualPrompt, CODAPrompt, and CPrompt

## Key Experimental Results

### Main Results

**ImageNet-R (40-step):**

| Method | Baseline | +KAC | Gain |
|------|---------|------|------|
| L2P | 74.28 | 76.34 | +2.06 |
| DualPrompt | 74.51 | 76.87 | +2.36 |
| CODAPrompt | 76.80 | 79.79 | +2.99 |
| CPrompt | 78.98 | 80.89 | +1.91 |

**CUB200 (40-step, most significant improvement):**

| Method | Baseline | +KAC | Gain |
|------|---------|------|------|
| L2P | 46.84 | 66.08 | **+19.24** |
| DualPrompt | 50.61 | 71.31 | **+20.70** |
| CODAPrompt | 52.57 | 71.36 | +18.79 |
| CPrompt | 77.34 | 85.11 | +7.77 |

### Ablation Study

| Configuration | ImageNet-R 20-step |
|------|-------------------|
| Linear classifier | 80.92% |
| MLP (same params) | 80.56% |
| MLP (fixed) | 65.87% |
| **KAC (RBF)** | **83.59%** |

| N (Number of RBFs) | Performance |
|-----------|------|
| 2 | Lowest |
| 4 | Optimal ✓ |
| 8 | Comparable to N=4 |
| 16 | Slight decrease |

### Key Findings
- **More task steps lead to larger gains for KAC**: For 5-step, performance increases by 0.2–4.4%, whereas for 40-step, it increases by 2–20%
- **Largest gains on fine-grained datasets**: Improvement on CUB200 far exceeds that on ImageNet-R; the localized characteristics of RBF excel at building fine class boundaries
- **Significant improvement in stability**: The standard error of L2P on CUB200 dropped from 5.06 to 0.57
- MLP performs worse than the linear classifier (-0.36%), proving that the problem lies in the learnable activation design unique to KAN

## Highlights & Insights
- **Minimalist yet highly effective**: Consistently improves all baseline methods with merely a 0.23M parameter modification
- **Clever Gaussian mixture interpretation of RBF**: Converts classification into feature localization within a Gaussian mixture space
- **Larger improvements with more steps**: In practical applications, the number of incremental steps is often large, which is the weakest link of existing methods
- Transferable to scenarios requiring lightweight and effective classifiers, such as few-shot learning and open-set recognition

## Limitations & Future Work
- All experiments are based on the ViT-B/16 backbone; compatibility with other backbones has not been validated
- RBF centers and widths are manually set; adaptive determination might yield further improvements
- Integration with exemplar-based methods (e.g., DER, BiC) remains unverified
- Smaller improvement on DomainNet (+1-2%), indicating that domain incremental learning might require additional mechanisms

## Related Work & Insights
- **vs Linear Classifier**: Lacks representation capacity, especially for long-sequence and fine-grained tasks
- **vs MLP**: Performs worse under the same parameter budget, proving that KAN's advantage is not simply about deepening the network
- **vs Original KAN (B-spline)**: B-spline requires grid updates, while RBF has fixed centers and only learns weights, making it more suitable for incremental scenarios

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce KAN to continual learning, with a clear motivation to replace B-splines with RBF
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across four methods, four datasets, and multiple step settings
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rich experiments
- Value: ⭐⭐⭐⭐ Plug-and-play nature makes it highly adoptable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Catastrophic Forgetting in Kolmogorov-Arnold Networks](../../AAAI2026/physics/catastrophic_forgetting_in_kolmogorov-arnold_networks.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](../../AAAI2026/physics/learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [\[ICLR 2026\] KANO: Kolmogorov–Arnold Neural Operator](../../ICLR2026/physics/kano_kolmogorov-arnold_neural_operator.md)
- [\[ICLR 2026\] Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study](../../ICLR2026/physics/initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)
- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](../../AAAI2026/physics/flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)

</div>

<!-- RELATED:END -->
