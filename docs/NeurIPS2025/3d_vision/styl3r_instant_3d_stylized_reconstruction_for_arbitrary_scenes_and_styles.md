---
title: >-
  [Paper Note] Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles
description: >-
  [NeurIPS 2025][3D Vision][3D stylization] This paper proposes Styl3R, a feed-forward network that decouples 3D reconstruction from stylization via a structure–appearance dual-branch architecture, enabling stylized 3D reconstruction from uncalibrated sparse-view images and arbitrary style images in 0.15 seconds.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D stylization
  - Gaussian splatting
  - feed-forward reconstruction
  - multi-view consistency
  - dual-branch architecture
date: 2026-05-08
content_hash: 6549d0c1096e5b24
---

# Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles

**Conference**: NeurIPS 2025  
**arXiv**: [2505.21060](https://arxiv.org/abs/2505.21060)  
**Code**: [Project Page](https://styl3r.github.io/)  
**Area**: 3D Vision  
**Keywords**: 3D stylization, Gaussian splatting, feed-forward reconstruction, multi-view consistency, dual-branch architecture

## TL;DR

This paper proposes Styl3R, a feed-forward network that decouples 3D reconstruction from stylization via a structure–appearance dual-branch architecture, enabling stylized 3D reconstruction from uncalibrated sparse-view images and arbitrary style images in 0.15 seconds.

## Background & Motivation

Extending 2D style transfer to 3D scenes has long been a significant challenge. The core difficulty lies in the inherent tension between appearance stylization and 3D structural consistency: modifying visual appearance to reflect a target style tends to compromise multi-view consistency and structural coherence.

Limitations of existing methods:

**2D style transfer methods** (e.g., AdaIN, StyTr2) achieve strong visual results but are entirely geometry-unaware and lack multi-view consistency; directly extending them to 3D leads to inconsistent stylization across viewpoints.

**3D stylization methods** (e.g., StyleRF, StyleGaussian, ARF) provide a degree of view consistency but require dense multi-view inputs, known camera poses, and costly per-scene or even per-style optimization. ARF requires per-scene per-style optimization; StyleRF and StyleGaussian support zero-shot style transfer but still require per-scene optimization, with training times ranging from 12 to 132 minutes.

No existing method is capable of directly generating stylized 3D reconstructions from sparse uncalibrated images and arbitrary style images.

The central problem of this paper is: how to achieve fast, multi-view-consistent 3D stylization from a small number of uncalibrated content images and an arbitrary style image, without any test-time optimization?

## Method

### Overall Architecture

Styl3R builds upon feed-forward 3D Gaussian splatting reconstruction and adopts a dual-branch architecture: the structure branch predicts geometric parameters of 3D Gaussians (position, orientation, scale, opacity), while the appearance branch predicts stylized colors. The framework leverages dense geometric priors from DUSt3R/MASt3R and is trained via a two-stage curriculum (NVS pre-training → stylization fine-tuning).

### Key Designs

1. **Structure Branch**: Leverages dense geometric priors from DUSt3R via a ViT encoder–decoder architecture to estimate scene structure.

    - Sparse uncalibrated images are encoded by a shared ViT encoder into content tokens $\mathcal{T}^c = \{\mathbf{t}_k^c\}$
    - A ViT decoder fuses multi-view information via cross-view cross-attention
    - Two DPT heads separately predict Gaussian center positions $\boldsymbol{\mu}_j$ and other structural attributes (orientation $\mathbf{r}_j$, scale $\mathbf{s}_j$, opacity $\alpha_j$)
    - The structure branch is frozen during stylization fine-tuning to prevent stylization losses from corrupting the geometric prior

2. **Appearance Branch (Stylization Decoder)**: The core innovation lies in the fusion mechanism between multi-view content tokens and style tokens.

    - A style encoder shares the same ViT architecture as the content encoder but with different weights; the style image $\mathbf{I}^s$ is encoded into style tokens $\mathcal{T}^s$
    - Global self-attention is first applied across multi-view content tokens to ensure multi-view consistency, yielding $\hat{\mathcal{T}}^c$
    - Cross-attention is then applied using the self-attended content tokens as queries and style tokens as keys/values:
    $\mathcal{T}^{cs} = \text{CrossAttention}(\hat{\mathcal{T}}^c W^Q, \mathcal{T}^s W^K, \mathcal{T}^s W^V)$
    - A DPT head predicts stylized colors $\mathbf{c}_j^s$ from the mixed tokens
    - **Content as Style insight**: content images can be treated as a special case of style images; feeding content images into the appearance branch yields a non-stylized reconstruction $\mathcal{G}^c$, allowing the model to simultaneously support stylized and standard reconstruction

3. **Two-Stage Training Curriculum**:

    - **NVS Pre-training**: The full model is trained end-to-end for novel view synthesis; content images are randomly fed into the appearance branch as "style," training the branch to preserve the original appearance.
    - **Stylization Fine-tuning**: The structure branch is frozen; only the appearance branch is trained. An **Identity Loss** is introduced — in addition to the style image, a content image is randomly fed in simultaneously, enabling the model to retain NVS capability while learning stylization.

### Loss & Training

Training losses are organized by stage:
$$\mathcal{L} = \begin{cases} \mathcal{L}_{photo}(\mathcal{G}^c), & \text{NVS pre-training} \\ \lambda \mathcal{L}_{style}(\mathcal{G}^s) + \mathcal{L}_{content}(\mathcal{G}^s) + \mathcal{L}_{photo}(\mathcal{G}^c), & \text{stylization fine-tuning} \end{cases}$$

- $\mathcal{L}_{photo}$: MSE + 0.05×LPIPS
- $\mathcal{L}_{style}$: mean and variance discrepancy computed at VGG19 feature layers relu1_1/2_1/3_1/4_1
- $\mathcal{L}_{content}$: computed jointly at relu3_1 and relu4_1 (experiments show this better preserves structure than using a single layer)
- $\lambda = 10$; progressive multi-view training is adopted (2-view NVS pre-training → initialization of 4-view training + stylization fine-tuning)

## Key Experimental Results

### Main Results

**Multi-view consistency and stylization time comparison on RE10K:**

| Method | Short LPIPS↓ | Short RMSE↓ | Long LPIPS↓ | Long RMSE↓ | Time |
|--------|-------------|-------------|------------|------------|------|
| AdaIN (2D) | 0.163 | 0.063 | 0.323 | 0.111 | 0.004s |
| StyTr2 (2D) | 0.167 | 0.059 | 0.315 | 0.098 | 0.029s |
| StyleRF (3D) | 0.062 | 0.021 | 0.172 | 0.042 | 90min |
| StyleGS (3D) | 0.048 | 0.022 | 0.137 | 0.043 | 132min |
| ARF (3D) | 0.093 | 0.038 | 0.217 | 0.070 | 12min |
| **Styl3R** | **0.044** | **0.022** | **0.107** | **0.038** | **0.147s** |

**User Study (preference votes):**

| Dataset | Styl3R | ARF | StyleGS | StyleRF | StyTr2 |
|---------|--------|-----|---------|---------|--------|
| RE10K | **53.29%** | 14.13% | 6.67% | 1.81% | 24.10% |
| TnT | **38.78%** | 22.11% | 11.56% | 3.06% | 24.49% |

### Ablation Study

| Configuration | ArtFID↓ | Histogram↓ | Vote Preference | Notes |
|---------------|---------|-----------|-----------------|-------|
| Content loss with relu3_1 only | 42.01 | 0.277 | 35.20% | Style over-dominates structure |
| **Content loss with relu3_1+relu4_1** | **35.83** | **0.239** | **64.80%** | Better balance of style and structure |

**NVS performance comparison (RE10K 2-view):**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| NoPoSplat | 25.033 | 0.838 | 0.160 |
| Ours (w/o stylization) | 24.871 | 0.837 | 0.165 |
| Ours (stylization mode) | 24.055 | 0.820 | 0.179 |

### Key Findings

- Styl3R surpasses all 3D baselines in multi-view consistency while being 3–5 orders of magnitude faster (0.147s vs. minutes/hours)
- Identity Loss is critical for preserving NVS capability: removing it prevents the model from recovering the original appearance
- Using dual content loss layers (relu3_1+relu4_1) better balances style expression and structural fidelity than a single layer
- Although trained with 4 views, the model flexibly handles 2–8 view inputs at inference
- Strong cross-dataset generalization: the model performs well on Tanks and Temples without any fine-tuning

## Highlights & Insights

- **Feed-forward + zero-shot dual advantages**: Styl3R is the first to achieve feed-forward inference from sparse uncalibrated images to stylized 3D reconstruction, requiring no test-time optimization, no known poses, and no scene- or style-specific fine-tuning.
- **Structure–appearance decoupling**: Freezing the structure branch prevents stylization losses from corrupting the geometry — a simple yet critical design choice.
- **Content as Style insight**: Treating content images as a special case of style images naturally enables a single model to support both stylization and standard reconstruction modes.
- **Style interpolation**: Smooth style transitions can be achieved by interpolating the style tokens of two style images.

## Limitations & Future Work

- NVS metrics degrade slightly after stylization (approximately 1 dB PSNR), indicating a remaining trade-off between stylization and reconstruction quality.
- The current training resolution is 256×256; stylization quality at higher resolutions remains to be validated.
- Zeroth-order spherical harmonics (degree-0 SH) are used, limiting view-dependent appearance effects.
- Future directions include stylization of video sequences and integration with text-driven style guidance.

## Related Work & Insights

The method builds upon NoPoSplat (pose-free reconstruction) and MASt3R (dense geometric priors). Compared to per-scene optimization methods such as StyleRF and StyleGaussian, the feed-forward paradigm yields a qualitative leap in efficiency. A key insight is that dense geometric priors from pre-trained 3D reconstruction models can be effectively "frozen" for downstream tasks, and the independent learning of an appearance branch is a pattern worth generalizing.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-branch decoupling combined with the feed-forward paradigm is innovative for 3D stylization, though the overall approach is a clever composition of existing modules.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage via quantitative evaluation, user study, cross-dataset generalization, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-illustrated figures, clearly articulated methodology, and thorough comparison with prior work.
- Value: ⭐⭐⭐⭐ Sub-second 3D stylization offers strong practical value and is directly applicable to interactive applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting](../../ICCV2025/3d_vision/a3gs_arbitrary_artistic_style_into_arbitrary_3d_gaussian_spl.md)
- [\[NeurIPS 2025\] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](../../ICCV2025/3d_vision/instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[AAAI 2026\] FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting](../../AAAI2026/3d_vision/fantasystyle_controllable_stylized_distillation_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
