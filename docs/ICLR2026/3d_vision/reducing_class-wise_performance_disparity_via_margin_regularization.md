---
title: >-
  [Paper Note] Reducing Class-Wise Performance Disparity via Margin Regularization
description: >-
  [ICLR 2026][3D Vision][class-wise disparity] The paper proposes MR2 (Margin Regularization for performance disparity Reduction), which dynamically adjusts class-dependent margins in logit and representation spaces. Based on a theoretically derived generalization bound, it reduces inter-class performance disparity while simultaneously improving overall accuracy.
tags:
  - ICLR 2026
  - 3D Vision
  - class-wise disparity
  - margin regularization
  - generalization bound
  - Rademacher complexity
  - representation learning
date: 2026-05-08
content_hash: 10709f5c6b8cae8e
---
# Reducing Class-Wise Performance Disparity via Margin Regularization

## Meta Information
- **Conference**: ICLR 2026
- **arXiv**: [2602.00205](https://arxiv.org/abs/2602.00205)
- **Code**: [https://github.com/BeierZhu/MR2](https://github.com/BeierZhu/MR2)
- **Area**: LLM Pretraining
- **Keywords**: class-wise disparity, margin regularization, generalization bound, Rademacher complexity, representation learning

## TL;DR
The paper proposes MR2 (Margin Regularization for performance disparity Reduction), which dynamically adjusts class-dependent margins in logit and representation spaces. Based on a theoretically derived generalization bound, it reduces inter-class performance disparity while simultaneously improving overall accuracy.

## Background & Motivation
- Deep networks exhibit significant inter-class accuracy disparity even when trained on balanced datasets. For instance, ResNet-50 on ImageNet achieves 100% accuracy on some classes but only 16% on others.
- Prior work observed that "hard" classes (low accuracy) exhibit greater feature variability (Figure 1b), but existing solutions are largely empirical (data augmentation, representation learning) and lack theoretical foundations.
- Existing margin-based methods (e.g., LDAM, Logit Adjustment) are designed for imbalanced classification and degenerate to standard cross-entropy in balanced scenarios, failing to address performance disparity.

## Method

### Overall Architecture
The starting point of MR2 is a class-sensitive generalization bound: it decomposes the generalization risk of each class into an "empirical risk" plus a complexity term. This term is proportional to intra-class feature variability and inversely proportional to the square of the class margin. By assigning larger margins to "hard" classes with more dispersed features while compressing their intra-class variability, the method reduces inter-class disparity without sacrificing "easy" class performance. Based on this insight, the method introduces two regularization terms in the logit and representation spaces respectively, jointly optimizing an objective derived from standard cross-entropy.

### Key Designs

**1. Class-Sensitive Generalization Bound: Theoretical Foundation Linking "Hard Classes" and "Margins"**

The foundation of the method is a class-sensitive generalization upper bound provided by Proposition 1:

$$\mathcal{R}(f) \leq \frac{1}{\ln 2} \hat{\mathcal{R}}_{\mathcal{D}}^{\bm{\gamma},\mathsf{ce}}(f) + \frac{4\sqrt{2}\Lambda K}{\sqrt{N}} \sqrt{\sum_{k=1}^K \frac{\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2}{\gamma_k^2}} + \mathcal{O}(1/\sqrt{N})$$

It decomposes the generalization risk into an "empirical risk" and a complexity term, where each class contributes $(\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2)/\gamma_k^2$. The numerator represents the feature energy and variability of the $k$-th class, while the denominator is its margin $\gamma_k$. This bound aligns with the empirical observation in Figure 1(b)—"hard" classes with lower accuracy have more dispersed features (larger numerator), leading to poorer generalization. To lower the overall bound, one must either decrease the numerator or increase $\gamma_k$ proportionally to the numerator. The two subsequent regularization terms address these paths respectively. Since they optimize the same global upper bound, improving hard classes does not penalize easy classes, avoiding the zero-sum trade-off common in debiasing methods.

**2. Logit Margin Loss: Allocating Margin Budgets based on Feature Dispersion**

The first strategy is to increase the margin for hard classes. MR2 replaces the uniform treatment of standard cross-entropy by assigning a unique temperature $\gamma_y$ as a margin for each class:

$$\ell_{\bm{\gamma},\mathsf{ce}}(f, \mathbf{x}, y) = -\mathbf{1}_y^\top \ln[\text{softmax}(\mathbf{z} / \gamma_y)]$$

Instead of being a manually tuned hyperparameter, $\gamma_y$ is a closed-form solution (Corollary 1) obtained by minimizing the complexity term under the constraint of a "fixed average budget ($\bar{\gamma}=\bar{c}$)":

$$\gamma_y = \frac{\bar{c} \cdot K (\|\hat{\bm{\mu}}_y\|_2^2 + \|\hat{\mathbf{s}}_y\|_2^2)^{1/3}}{\sum_{k=1}^K (\|\hat{\bm{\mu}}_k\|_2^2 + \|\hat{\mathbf{s}}_k\|_2^2)^{1/3}}$$

Here, $\hat{\bm{\mu}}_k$ is the feature mean of class $k$, and $\|\hat{\mathbf{s}}_k\|_2^2$ is its mean squared deviation, both maintained online during training via EMA. $\bar{c}$ controls the overall margin strength. When all $\gamma_k=1$, the loss degenerates to standard cross-entropy. The $1/3$ power distribution law allows hard classes with more dispersed features to receive a larger $\gamma_y$, essentially tilting more generalization budget toward them. This narrows inter-class gaps at the source rather than compensating for hard classes at the expense of easy ones.

**3. Representation Margin Loss: Directly Compressing Intra-class Variability**

The second strategy is to decrease the numerator. Simply increasing the margin treats the symptoms; the root cause is the dispersed nature of hard class features. This term uses $2\bar{s}$ (where $\bar{s}=\frac{1}{K}\sum_k\|\hat{\mathbf{s}}_k\|_2^2$) as a margin for global average mean squared deviation, applied to intra-class sample pairs:

$$\ell_{\bar{s}}(f, \mathbf{x}, y) = \ln\Big[1 + \sum_{\mathbf{x}^+ \in \mathcal{D}_y \setminus \{\mathbf{x}\}} \exp(\|\phi(\mathbf{x}) - \phi(\mathbf{x}^+)\|_2^2 - 2\bar{s})\Big]$$

Classes are penalized when their intra-class distance exceeds this margin, thereby tightening the representation. This is equivalent to minimizing intra-class mean squared deviation. It complements the logit term: while the logit term allocates the budget, the representation term reduces the amount of budget required. Both paths work to suppress the same global bound. The final training objective is a weighted combination with coefficient $\lambda$:

$$\min_{f} \frac{1}{N} \sum_{\mathbf{x},y \in \mathcal{D}} \big[\ell_{\bm{\gamma},\mathsf{ce}}(f, \mathbf{x}, y) + \lambda \cdot \ell_{\bar{s}}(f, \mathbf{x}, y)\big]$$

## Key Experimental Results

### Main Results: CIFAR-100 & ImageNet

| Method | Overall Acc | Easy | Medium | Hard |
|------|----------|------|--------|------|
| ERM (Standard) | 70.9 | 84.5 | 71.0 | 56.7 |
| LfF | 69.1 | 83.6 (-0.9) | 70.1 (-0.9) | 53.7 (-3.0) |
| JTT | 70.6 | 84.3 (-0.2) | 70.8 (-0.2) | 56.2 (-0.5) |
| DRO | ~70.0 | Decrease | ~71.0 | ~56.0 |
| **MR2 (Ours)** | **71.8** | **85.0 (+0.5)** | **72.0 (+1.0)** | **58.5 (+1.8)** |

> MR2 significantly improves "hard" class performance (+1.8) while also improving "easy" classes (+0.5), requiring no trade-off.

### Ablation Study: Pre-trained Backbones + Fine-tuning Methods

| Backbone + Method | ERM | MR2 | Hard Gain |
|-----------|-----|-----|---------|
| MAE (end-to-end) | Baseline | +Gain | Significant |
| MoCov2 (linear probe) | Baseline | +Gain | Significant |
| CLIP (linear probe) | Baseline | +Gain | Significant |
| ResNet-50 (scratch) | 70.9 | **71.8** | +1.8 |
| ViT-B/16 (scratch) | Baseline | +Gain | Significant |

> MR2 is applicable across various pre-training methods (MAE/MoCov2/CLIP) and training paradigms (end-to-end/linear probe).

### Key Findings
1. Existing debiasing methods (LfF, JTT, DRO) often sacrifice "easy" class performance to improve "hard" classes—MR2 avoids this trade-off.
2. Logit margin and representation margin are complementary: the former allocates more generalization budget to hard classes, while the latter reduces intra-class variability.
3. The theoretically derived $\gamma_k$ is highly consistent with optimal values selected via grid search in practice.
4. Even on L2-normalized CLIP features, using $L_p (p \neq 2)$ norms can still recover class-sensitive margins.

## Highlights & Insights
- **Theory-Driven Method**: Margin design is derived from generalization bounds rather than empirical tuning.
- **No-Trade-off Improvement**: Simultaneously improves both hard and easy classes, which is rare in fairness/debiasing literature.
- **Wide Applicability**: Consistent improvements across 7 datasets, CNN/ViT architectures, and multiple pre-training paradigms.
- **Non-conflicting with Long-tail Methods**: Still meaningful in balanced scenarios, filling the theoretical gap regarding performance disparity in balanced data.

## Limitations & Future Work
- Maintaining class statistics via EMA introduces minor computational overhead.
- Representation margin loss requires intra-class sample pairing, which may be unstable for classes with very few samples.
- Theoretical analysis assumes uniformly bounded classifier weight norms ($\Lambda$), which may not hold strictly in all models.
- Hyperparameters $\bar{c}$ and $\lambda$ still require tuning.

## Related Work
- **Margin for Long-tail Classification**: LDAM (Cao et al., 2019), Logit Adjustment (Menon et al., 2021), Balanced Softmax (Ren et al., 2020).
- **Performance Disparity Research**: Cui et al. (2024) found that disparity stems from representation rather than classifier bias.
- **Neural Collapse**: Idealized assumptions from Papyan et al. (2020) do not hold on large datasets.
- **Contrastive Learning**: SupCon (Khosla et al., 2020) lacks margin constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ — Margin regularization for balanced data, unifying theoretical derivation and empirical insights.
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Complete Rademacher complexity analysis and generalization bounds.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7 datasets, multiple architectures, multiple pre-training methods, and detailed ablations.
- Value: ⭐⭐⭐⭐ — Plug-and-play, open-source implementation, applicable to various classification models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Universal Beta Splatting](universal_beta_splatting.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/3d_vision/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [\[CVPR 2026\] No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency](../../CVPR2026/3d_vision/no_calibration_no_depth_no_problem_cross-sensor_view_synthesis_with_3d_consisten.md)

</div>

<!-- RELATED:END -->
