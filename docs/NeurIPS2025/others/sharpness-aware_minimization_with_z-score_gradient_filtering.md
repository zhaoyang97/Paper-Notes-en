---
title: >-
  [Paper Note] Sharpness-Aware Minimization with Z-Score Gradient Filtering
description: >-
  [NeurIPS 2025 (OPT Workshop)][Sharpness-Aware Minimization] This paper proposes Z-Score Filtered SAM (ZSAM), which applies per-layer Z-Score statistical filtering to gradient vectors…
tags:
  - "NeurIPS 2025 (OPT Workshop)"
  - "Sharpness-Aware Minimization"
  - "Gradient Filtering"
  - "Z-Score"
  - "Generalization"
  - "Flat Minima"
date: 2026-05-08
content_hash: 4362af9593213250
---

# Sharpness-Aware Minimization with Z-Score Gradient Filtering

**Conference**: NeurIPS 2025 (OPT Workshop)
**arXiv**: [2505.02369](https://arxiv.org/abs/2505.02369)  
**Code**: [Available](https://github.com/YUNBLAK/Sharpness-Aware-Minimization-with-Z-Score-Gradient-Filtering)  
**Area**: Other
**Keywords**: Sharpness-Aware Minimization, Gradient Filtering, Z-Score, Generalization, Flat Minima

## TL;DR

This paper proposes Z-Score Filtered SAM (ZSAM), which applies per-layer Z-Score statistical filtering to gradient vectors, retaining only the most statistically significant gradient components for the perturbation ascent step. This guides the optimizer toward flat minima more effectively, achieving consistent improvements in test accuracy across multiple datasets and architectures.

## Background & Motivation

Sharpness-Aware Minimization (SAM) improves generalization by perturbing parameters along high-curvature directions in the parameter space, and has become an important tool in deep learning optimization. However, SAM computes the perturbation direction using the full gradient vector, which contains many small or noisy gradient components. These unimportant components can distort the direction of the ascent step, causing the optimizer to deviate from the optimal trajectory and fail to effectively reach flat minima regions.

Existing SAM variants (e.g., ASAM, GSAM) improve upon SAM from different perspectives—such as adaptive perturbation and gradient decomposition—but none addresses the question of which gradient components are truly worth perturbing from the perspective of the statistical distribution of gradients. The authors observe that within each layer, the absolute values of gradient components exhibit significant variation: a small number of components have abnormally large absolute values (high Z-Score), while the majority cluster near the mean. Focusing on statistically prominent gradient components therefore provides a more effective signal toward high-curvature directions.

## Method

### Overall Architecture

ZSAM augments the standard two-step SAM optimization pipeline with a gradient filtering step:

1. **Gradient Computation**: Compute the gradient $\nabla L(w)$ of the loss at the current parameters $w$.
2. **Z-Score Filtering**: Compute per-layer Z-Scores independently and construct a mask to retain the most important components.
3. **Perturbation Ascent**: Compute the perturbation direction $\hat{\epsilon}$ using the filtered gradients.
4. **Gradient Descent**: Compute new gradients at the perturbed parameters $w + \hat{\epsilon}$ and perform the update.

### Key Designs

**Per-Layer Z-Score Computation**: For the gradient vector $g^{(l)}$ of the $l$-th layer, the Z-Score of each component $i$ is computed as:

$$z_i^{(l)} = \frac{|g_i^{(l)}| - \mu^{(l)}}{\sigma^{(l)}}$$

where $\mu^{(l)}$ and $\sigma^{(l)}$ are the mean and standard deviation of the absolute gradient values in that layer.

**Percentile Mask Construction**: A percentile threshold $Q_p$ is set, retaining only the gradient components whose Z-Scores fall in the top $(1-Q_p)$ percentile:

$$m_i^{(l)} = \mathbb{1}(|z_i^{(l)}| \geq z_{Q_p}^{(l)})$$

The filtered gradient is $\tilde{g}^{(l)} = m^{(l)} \odot g^{(l)}$, and only these filtered gradients are used to compute the perturbation direction.

**Rationale for Per-Layer Independent Filtering**: Gradient distributions vary substantially across layers—gradients in shallow layers are typically smaller than those in deep layers. Global filtering would cause nearly all shallow-layer gradients to be discarded. Per-layer processing ensures a reasonable proportion of gradients is retained at each layer.

### Loss & Training

The loss function follows the standard SAM minimax objective:

$$\min_w \max_{\|\epsilon\| \leq \rho} L(w + \epsilon)$$

The only difference lies in the computation of the perturbation direction. During training, $Q_p$ is a fixed hyperparameter; the authors recommend searching in the range 0.5–0.9. The additional computational overhead over SAM is negligible, consisting only of Z-Score computation and masking operations.

## Key Experimental Results

### Main Results

Experiments are conducted on CIFAR-10, CIFAR-100, and Tiny-ImageNet using three architectures: ResNet-18, VGG-16, and ViT-Small.

| Method | CIFAR-10 (ResNet-18) | CIFAR-100 (ResNet-18) | Tiny-ImageNet (ResNet-18) |
|--------|---------------------|-----------------------|--------------------------|
| SGD | 95.03 | 77.52 | 62.14 |
| SAM | 95.68 | 79.31 | 64.37 |
| ASAM | 95.72 | 79.45 | 64.52 |
| GSAM | 95.75 | 79.58 | 64.61 |
| **ZSAM** | **96.01** | **80.12** | **65.28** |

| Method | CIFAR-10 (VGG-16) | CIFAR-100 (VGG-16) | CIFAR-10 (ViT-S) | CIFAR-100 (ViT-S) |
|--------|-------------------|-------------------|-------------------|-------------------|
| SGD | 93.21 | 73.85 | 94.56 | 76.82 |
| SAM | 93.89 | 75.23 | 95.12 | 78.15 |
| ASAM | 93.95 | 75.38 | 95.18 | 78.29 |
| **ZSAM** | **94.28** | **76.01** | **95.52** | **79.03** |

### Ablation Study

**Effect of the Percentile Threshold $Q_p$ (CIFAR-100, ResNet-18)**:

| $Q_p$ | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 |
|-------|------|------|------|------|------|-------|
| Accuracy | 79.42 | 79.78 | 80.12 | 79.95 | 79.63 | 79.15 |

- Performance peaks around $Q_p = 0.7$, indicating that retaining approximately the top 30% most significant gradient components is optimal.
- Too low a threshold (retaining too many components) degrades to standard SAM; too high a threshold (retaining too few) leads to severe information loss.

**Global vs. Per-Layer Filtering**:

| Filtering Strategy | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|-------------------|----------|-----------|--------------|
| Global Z-Score | 95.71 | 79.52 | 64.55 |
| **Per-Layer Z-Score** | **96.01** | **80.12** | **65.28** |

Per-layer filtering consistently outperforms global filtering, validating the design motivation that different layers require independent processing due to their differing gradient distributions.

### Key Findings

1. **Consistent Improvement**: ZSAM outperforms SAM and its major variants across all tested dataset–architecture combinations.
2. **Effectiveness on Both CNNs and ViTs**: The method is not limited to CNN architectures and proves effective on Vision Transformers as well.
3. **Flatness Verification**: Loss landscape visualizations confirm that ZSAM converges to flatter minima than SAM.
4. **Low Overhead**: The method introduces negligible additional computational cost over SAM.

## Highlights & Insights

- **Elegant Statistical Perspective**: Using Z-Score to measure the "importance" of gradient components is a concise and effective idea that avoids complex adaptive mechanisms.
- **Per-Layer Independent Processing**: Accounting for the distributional differences across layers is a principled design choice.
- **Plug-and-Play**: The method can be readily integrated into any SAM-based optimizer by inserting the filtering step before the perturbation computation.
- **Clear Theoretical Intuition**: Filtering out unimportant gradient components focuses the perturbation on directions of genuinely high curvature.

## Limitations & Future Work

1. **Workshop-Level Scope**: The experimental scale is relatively limited, lacking validation on large-scale datasets (ImageNet-1K) and large models.
2. **Adaptive Adjustment of $Q_p$**: A fixed percentile threshold may be suboptimal; dynamically adjusting it according to training progress is a natural extension.
3. **Absence of Theoretical Analysis**: There is no formal proof explaining why Z-Score filtering leads to better convergence to flat minima.
4. **Insufficient Downstream Task Validation**: Experiments are limited to classification; validation on detection, segmentation, and other tasks is lacking.
5. **Comparison with Alternative Filtering Methods**: Baselines such as Top-K filtering and random masking are not included to rigorously demonstrate the superiority of Z-Score filtering.

## Related Work & Insights

- **SAM Family**: Original SAM (Foret et al., 2021) → ASAM → GSAM → LookSAM → Fisher-SAM.
- **Gradient Compression/Sparsification**: Extensive gradient sparsification techniques (Top-K, Random-K) exist in the distributed training literature; this paper applies analogous ideas to perturbation computation in optimization.
- **Flat Minima Theory**: Theoretical foundations on the relationship between flat/sharp minima and generalization, as established by Keskar et al. (2017) and related works.

## Rating

- **Novelty**: 3/5 — The core idea is clean but technically incremental.
- **Technical Quality**: 3/5 — Experimental design is sound but limited in scale.
- **Writing Quality**: 4/5 — The paper is clearly written and the method is easy to follow.
- **Value**: 4/5 — Plug-and-play with low implementation cost.
- **Overall**: 3.5/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation](../../ICLR2026/others/revisiting_sharpness-aware_minimization_a_more_faithful_and_effective_implementa.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Manipulating Feature Visualizations with Gradient Slingshots](manipulating_feature_visualizations_with_gradient_slingshots.md)
- [\[NeurIPS 2025\] Hessian-guided Perturbed Wasserstein Gradient Flows for Escaping Saddle Points](hessian-guided_perturbed_wasserstein_gradient_flows_for_escaping_saddle_points.md)
- [\[NeurIPS 2025\] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)

</div>

<!-- RELATED:END -->
