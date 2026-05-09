---
title: >-
  [Paper Note] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer
description: >-
  [CVPR 2026][Medical Imaging][Zero-shot anomaly detection] This paper revisits the necessity of the text branch in zero-shot anomaly detection (ZSAD) and proposes VisualAD, a purely vision-based framework. Two learnable tokens (anomaly/normal) are inserted into a frozen ViT, enhanced by Spatial-Aware Cross-Attention (SCA) and a Self-Alignment Function (SAF). Without a text encoder, VisualAD achieves state-of-the-art performance across 13 industrial and medical benchmarks.
tags:
  - CVPR 2026
  - Medical Imaging
  - Zero-shot anomaly detection
  - Vision Transformer
  - language-free
  - learnable token
  - industrial + medical
date: 2026-05-08
content_hash: 98da7a041014bf07
---

# VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer

**Conference**: CVPR 2026
**arXiv**: [2603.07952](https://arxiv.org/abs/2603.07952)
**arXiv**: [2603.07952](https://arxiv.org/abs/2603.07952)
**Code**: None
**Area**: Medical Imaging
**Keywords**: Zero-shot anomaly detection, Vision Transformer, language-free, learnable token, industrial + medical

## TL;DR

This paper revisits the necessity of the text branch in zero-shot anomaly detection (ZSAD) and proposes VisualAD, a purely vision-based framework. Two learnable tokens (anomaly/normal) are inserted into a frozen ViT, enhanced by Spatial-Aware Cross-Attention (SCA) and a Self-Alignment Function (SAF). Without a text encoder, VisualAD achieves state-of-the-art performance across 13 industrial and medical benchmarks.

## Background & Motivation

- **ZSAD challenge**: Anomalies must be detected on unseen categories without per-class normal training samples.
- **Mainstream methods rely on CLIP's text branch**: Methods such as AnomalyCLIP generate normal/abnormal prototypes via learnable text prompts and perform image-text similarity scoring.
- **Core question**: If the final decision depends solely on two prototype vectors (normal and abnormal), is the text modality truly indispensable?
- **Exploratory experiment**: Removing the text encoder from AnomalyCLIP and directly optimizing two visual vectors yields:
    - No significant drop in detection performance
    - A reduction of 99%+ in trainable parameters
    - More stable training curves (the text-branch variant exhibits severe oscillation)
- **Conclusion**: Text prompts may serve merely as an indirect channel for shaping visual prototypes, rather than being essential.

## Method

### Overall Architecture

Two learnable tokens are inserted into the token sequence of a frozen ViT:

$$z_0 = [t_a, t_n, t_c, p_1, \ldots, p_N]$$

where $t_a$ is the anomaly token, $t_n$ is the normal token, and $t_c$ is the original class token.

Features are extracted from intermediate layers $\mathcal{L} = \{6, 12, 18, 24\}$, tokens are enhanced via SCA, patches are calibrated via SAF, and per-layer anomaly maps are computed and fused.

### Spatial-Aware Cross-Attention (SCA)

Global tokens lack spatial localization capability. SCA aggregates local spatial evidence through a small set of anchor queries $Q_{\text{anchor}} \in \mathbb{R}^{m \times d}$ ($m=4$):

$$A_\ell = \text{softmax}\left(\frac{Q_{\text{anchor}} (P_\ell^{\text{pos}})^\top}{\sqrt{d}}\right), \quad U_\ell = A_\ell P_\ell$$

A token-guided gating mechanism adaptively modulates the output:

$$g(t) = \sigma(W_g t) \in \mathbb{R}^m$$
$$\tilde{t}_\ell = t + \alpha \sum_{i=1}^{m} g_i(t) \cdot a_i$$

SCA is instantiated independently at each layer, dynamically adjusting the spatial sensitivity of tokens for each input image.

### Self-Alignment Function (SAF)

A single-hidden-layer MLP at each layer calibrates the patch features: $\hat{P}_\ell = \mathcal{F}_\ell(P_\ell)$

### Anomaly Scoring

After L2 normalization, anomaly scores are computed as cosine contrastive differences:

$$s_i^{(\ell)} = \langle \bar{\hat{p}}_i^{(\ell)}, \bar{t}_a^{(\ell)} \rangle - \langle \bar{\hat{p}}_i^{(\ell)}, \bar{t}_n^{(\ell)} \rangle$$

Multi-layer fusion: $H = \sum_{\ell \in \mathcal{L}} H_\ell$; image-level scores are computed as the mean of the top-1% pixel values.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{seg}} + \mathcal{L}_{\text{ctr}}$$

- $\mathcal{L}_{\text{cls}}$: image-level BCE loss
- $\mathcal{L}_{\text{seg}}$: per-layer Focal + Dice loss
- $\mathcal{L}_{\text{ctr}}$: cosine margin penalty ensuring an angular distance > 120° between $t_a$ and $t_n$

Only $t_a$, $t_n$, SCA, and SAF are updated; the ViT backbone remains frozen.

## Key Experimental Results

### Image-Level AUROC on Industrial Domains

| Method | MVTec-AD | VisA | BTAD | KSDD2 | DAGM |
|--------|----------|------|------|-------|------|
| WinCLIP | 90.4 | 75.6 | 68.2 | 93.5 | 91.8 |
| AnomalyCLIP | 91.6 | 81.0 | 88.7 | 91.9 | 98.0 |
| AdaCLIP | 92.0 | 79.7 | 90.0 | 94.9 | 98.3 |
| **VisualAD(CLIP)** | **92.2** | **84.7** | **94.9** | **98.0** | **99.5** |

### Image-Level AUROC on Medical Domains

| Method | OCT17 | BrainMR1 | Brain_AD | HIS |
|--------|-------|----------|----------|-----|
| AnomalyCLIP | 63.7 | 96.4 | 69.0 | 55.2 |
| **VisualAD(CLIP)** | **88.9** | **96.7** | **80.8** | **60.1** |
| **VisualAD(DINOv2)** | **91.2** | 93.8 | **87.1** | 60.1 |

Gains on medical domains are particularly substantial: AUROC on OCT17 improves from 63.7 to 91.2 (+27.5).

### Ablation Study

| Module | Image AUROC | Pixel AP |
|--------|------------|----------|
| w/o SCA | 82.3 | 27.4 |
| w/o SAF | 50.5 | 3.5 |
| w/o SCA & SAF | 48.0 | 0.8 |
| **Full model** | **84.7** | **28.4** |

SAF is the critical component; its removal causes catastrophic performance degradation.

### Backbone Flexibility

The same framework seamlessly accommodates both CLIP and DINOv2 backbones. The DINOv2 variant performs stronger on pixel-level segmentation, while the CLIP variant is superior on image-level classification.

## Highlights & Insights

1. **Challenging the necessity of text**: Experiments demonstrate that the CLIP text branch in ZSAD may serve merely as an indirect channel for shaping visual prototypes, with a 99% reduction in parameters.
2. **Minimalist and elegant design**: Only two learnable tokens plus lightweight SCA/SAF modules are required, with a unified training and inference pipeline.
3. **Cross-domain zero-shot generalization**: The framework generalizes strongly under the industrial-training → medical-inference setting.
4. **Backbone agnosticism**: Compatible with both CLIP and DINOv2, offering strong extensibility.

## Limitations & Future Work

- Gains on certain medical datasets (e.g., HIS) are limited, likely due to weak visual priors for histopathological anomalies.
- The choice of anchor query count $m=4$ lacks in-depth analysis.
- An auxiliary training set (normal and abnormal samples from VisA) is required; the method is not fully training-free.
- Pixel-level segmentation still lags behind some specialized methods on certain datasets.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversionbased_reconstructionfree_anomaly_de.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/medical_imaging/dual_distillation_for_few-shot_anomaly_detection.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)

</div>

<!-- RELATED:END -->
