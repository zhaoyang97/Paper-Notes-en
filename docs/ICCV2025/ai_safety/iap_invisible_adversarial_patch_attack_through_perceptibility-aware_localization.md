---
title: >-
  [Paper Note] IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization
description: >-
  [ICCV 2025][Adversarial patch] This paper proposes the IAP framework, which achieves — for the first time in targeted attack settings — truly invisible adversarial patches via **perceptibility-aware patch localization**…
tags:
  - "ICCV 2025"
  - "Adversarial patch"
  - "imperceptible attack"
  - "perceptual sensitivity"
  - "targeted attack"
  - "patch defense"
date: 2026-05-08
content_hash: 5e60ba64b50e8232
---

# IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization

**Conference**: ICCV 2025
**arXiv**: [2507.06856](https://arxiv.org/abs/2507.06856)  
**Code**: [https://github.com/subratkishoredutta/IAP](https://github.com/subratkishoredutta/IAP)  
**Area**: Other
**Keywords**: Adversarial patch, imperceptible attack, perceptual sensitivity, targeted attack, patch defense

## TL;DR

This paper proposes the IAP framework, which achieves — for the first time in targeted attack settings — truly invisible adversarial patches via **perceptibility-aware patch localization** and **color-preserving gradient updates**, while simultaneously bypassing multiple SOTA patch defenses.

## Background & Motivation

Adversarial patch attacks fool classifiers by modifying only a local region of an image, yet existing methods exhibit two fundamental tensions:

**Imperceptibility vs. attack effectiveness**: Targeted attacks require large perturbations, which render patches visually conspicuous. Consequently, existing imperceptible methods (e.g., VRAP, Bai et al.) are limited to untargeted attacks.

**Saliency vs. defense detectability**: Existing defenses (e.g., SAC, Jedi, DIFFender) exploit the high saliency of patch regions for localization and restoration, rendering conventional high-saliency patch attacks ineffective against them.

**Core Problem**: Can targeted attacks be realized with visually imperceptible adversarial patches?

The authors' key insight is that rather than restricting perturbation magnitude to achieve imperceptibility (the conventional approach), **large perturbations can instead be strategically placed in regions to which the human eye is insensitive**. This motivates an entirely new perceptibility-aware optimization paradigm.

## Method

### Overall Architecture

IAP consists of two stages: (1) patch placement optimization — identifying the optimal location; and (2) perturbation update optimization — generating imperceptible perturbations.

### Key Designs

#### 1. Perceptibility-Aware Patch Localization (Perturbation Priority Index)

A Perturbation Priority Index $G(\mathbf{x}; i,j)$ is defined to balance model vulnerability against human visual insensitivity:

$$G(\mathbf{x}; i,j) = \sum_{k=0}^{w}\sum_{l=0}^{h} \frac{J_y(\mathbf{x}; i+k, j+l)}{\text{Sens}(\mathbf{x}; i+k, j+l)}$$

- **Numerator $J_y$ (class localization map)**: A Grad-CAM attention heatmap reflecting the model's focus on the current class; high values indicate regions with greater influence on model predictions and thus higher attack susceptibility.
- **Denominator Sens (sensitivity map)**: The reciprocal of the per-pixel standard deviation along horizontal and vertical directions; low-sensitivity regions (high textural complexity) can absorb larger perturbations without perceptual detection.

The optimal placement is $(i', j') = \arg\max_{i,j} G(\mathbf{x}; i,j)$. This ratio elegantly achieves the balance between high attack potency and low visibility.

#### 2. Perceptually Regularized Perturbation Optimization

The total loss function is:
$$\mathcal{L}_T = w_1 \cdot \mathcal{L}_{CE}(\hat{\mathbf{x}}, y_{targ}) - w_2 \cdot \mathcal{L}_{CE}(\hat{\mathbf{x}}, y) + w_3 \cdot D(\mathbf{x}, \hat{\mathbf{x}})$$

where the perceptual distance $D(\mathbf{x}, \hat{\mathbf{x}})$ weights by human visual system sensitivity:
$$D(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{h \times w} \sum \text{Sens}(\mathbf{x}; k,l) \cdot |x_{kl} - \hat{x}_{kl}|$$

This encourages large perturbations in low-sensitivity regions while suppressing perturbations in high-sensitivity regions.

#### 3. Color-Preserving Gradient Update Rule

$$\delta_{t+1} = \delta_t - \eta \cdot \overline{\nabla_\delta} \mathcal{L}_T \odot (\delta_t \oslash \text{Sens}(\mathbf{x}))$$

- $\overline{\nabla_\delta}$: Gradients are averaged across the three RGB channels, ensuring **identical update magnitudes across channels** and thereby preserving the base hue of each pixel.
- $\delta_t \oslash \text{Sens}(\mathbf{x})$: Step sizes are scaled inversely by sensitivity, permitting larger steps in high-texture regions.

This design is grounded in the psychophysical observation that the human eye is far less sensitive to luminance/saturation changes within a base hue than to changes in hue itself.

### Loss & Training

- Patches are initialized to original pixel values (rather than random noise) to minimize initial visual discrepancy.
- Optimization proceeds until target-class confidence ≥ 0.9 or 1,000 iterations are reached.
- Patch size is 84×84 (14% of the image area); arbitrary shapes (square, circular) are supported.
- Upon failure, the step size is re-initialized, with up to 3 restarts.

## Key Experimental Results

### Main Results (ImageNet + VGG Face, Targeted Attack)

| Dataset | Method | ASR(%) | LPIPS_L(↓) | SSIM_L(↑) |
|--------|------|:---:|:---:|:---:|
| ImageNet (ResNet-50) | Google Patch | 99.10 | 0.74 | 0.010 |
| ImageNet (ResNet-50) | GDPA | 93.70 | 0.57 | 0.350 |
| ImageNet (ResNet-50) | MPGD | 97.80 | 0.24 | 0.790 |
| ImageNet (ResNet-50) | **IAP** | **99.50** | **0.12** | **0.940** |
| ImageNet (Swin-B) | Google Patch | 97.90 | 0.77 | 0.003 |
| ImageNet (Swin-B) | MPGD | 70.50 | 0.20 | 0.800 |
| ImageNet (Swin-B) | **IAP** | **99.40** | **0.07** | **0.970** |

### Ablation Study (Defense Bypass, ImageNet ResNet-50)

| Method | Jedi | Jujutsu | SAC | DW | DIFFender | DiffPAD |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Google Patch | 46.8 | 0.0 | 2.7 | 1.4 | 35.5 | 33.2 |
| GDPA | 67.1 | 94.0 | 7.4 | 1.3 | 57.0 | 52.1 |
| MPGD | 68.2 | 95.1 | 11.6 | 79.0 | 95.7 | 92.1 |
| **IAP** | **78.6** | **99.8** | **100** | **89.8** | **99.8** | **98.6** |

### Key Findings

1. IAP achieves **ASR ≥ 94.5% across all models and datasets**, while reducing LPIPS by more than 50% relative to the second-best method.
2. Human perceptual study: IAP patches are detected by only **4.2%** of participants, compared to 94.5% for MPGD.
3. Grad-CAM analysis reveals that approximately **70% of IAP samples have their highest attention region outside the attack patch area**, which is the fundamental reason the method bypasses saliency-based defenses.
4. ASR against the SAC defense increases from a baseline maximum of 11.6% to **100%**, demonstrating the substantial threat posed by imperceptible attacks to existing defense frameworks.
5. In black-box settings, using a surrogate model combined with NES query optimization still achieves ASR > 89%.

## Highlights & Insights

- **Reverse thinking**: Rather than restricting perturbation magnitude, large perturbations are concealed in regions to which the human eye is insensitive. This challenges the conventional assumption that "small perturbation = imperceptibility."
- **The color-preserving update rule is simple yet effective**: Preserving base hue through channel-averaged gradients alone incurs virtually zero computational overhead.
- **Implications for defense**: All six SOTA defenses are nearly completely defeated by IAP, indicating that **saliency-based defense paradigms require fundamental reconsideration**.

## Limitations & Future Work

- Local pixel context is not considered, and individual pixels may occasionally appear anomalously bright or dark.
- Perceptibility-aware localization introduces additional computational overhead due to sliding-window search.
- Physical-world attacks are only preliminarily validated (70% ASR) without domain-specific adaptation.
- Attack performance degrades as patch size decreases.

## Related Work & Insights

- Unlike generative imperceptible patch methods such as PS-GAN and GDPA, IAP achieves imperceptibility directly at the optimization level without relying on generative models.
- The color-preserving update strategy is generalizable to other adversarial attack settings (e.g., video, 3D).
- The results highlight an urgent need to develop novel defense strategies grounded in **alignment between machine perception and human perception**.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of perceptibility-aware localization and color-preserving updates is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 architectures, 2 datasets, 6 defenses, human study, black-box, and physical attacks)
- Writing Quality: ⭐⭐⭐⭐ (Formulations are clear; visualizations are excellent)
- Value: ⭐⭐⭐⭐ (Carries significant cautionary implications for the security research community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] C4D: 4D Made from 3D through Dual Correspondences](c4d_4d_made_from_3d_through_dual_correspondences.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)
- [\[ICCV 2025\] LaCoOT: Layer Collapse through Optimal Transport](lacoot_layer_collapse_through_optimal_transport.md)
- [\[NeurIPS 2025\] Rethinking PCA Through Duality](../../NeurIPS2025/others/rethinking_pca_through_duality.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)

</div>

<!-- RELATED:END -->
