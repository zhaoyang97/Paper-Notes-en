---
title: >-
  [Paper Note] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing
description: >-
  [CVPR 2025][Image Generation][Flow matching inversion] To achieve tuning-free image editing based on Flow Transformer (MM-DiT), this paper proposes a two-stage flow inversion method (fixed-point iteration + velocity field compensation) and an Adaptive Layer Normalization (AdaLN)-based invariance control mechanism to uniformly support both rigid and non-rigid editing operations.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Flow matching inversion"
  - "image editing"
  - "MM-DiT"
  - "invariance control"
  - "AdaLN"
date: 2026-05-08
content_hash: 6a8989c4783570fc
---

# Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing

**Conference**: CVPR 2025  
**arXiv**: [2411.15843](https://arxiv.org/abs/2411.15843)  
**Code**: None  
**Area**: Image Generation/Editing  
**Keywords**: Flow matching inversion, image editing, MM-DiT, invariance control, AdaLN

## TL;DR

To achieve tuning-free image editing based on Flow Transformer (MM-DiT), this paper proposes a two-stage flow inversion method (fixed-point iteration + velocity field compensation) and an Adaptive Layer Normalization (AdaLN)-based invariance control mechanism to uniformly support both rigid and non-rigid editing operations.

## Background & Motivation

- Text-to-Image (T2I) models based on Flow Transformer (such as Stable Diffusion 3.5) possess stronger generative priors and superior text-image alignment capabilities, yet their editing potential remains under-explored.
- Classically successful DDIM inversion methods in diffusion models perform poorly on the Euler sampler of Rectified Flow due to larger approximation errors, resulting in degraded quality when directly transferred to flow models.
- Experiments reveal that even when running Euler inversion on the stronger SD3.5 model, the reconstruction quality is far inferior to running DDIM inversion on the weaker SD1.5.
- Attention-based invariance control methods in U-Net architectures (e.g., cross-attention injection, KV-injection) cannot be directly duplicated in MM-DiT, as MM-DiT does not have individual cross-attention modules.
- Existing attention injection techniques struggle to balance rigid editing (e.g., object replacement) and non-rigid editing (e.g., changing pose or quantity).
- Therefore, there is a strong demand for dedicated inversion and invariance control schemes tailored specifically for flow models and Transformer architectures.

## Method

### Overall Architecture

The overall framework is divided into two phases: inversion and editing. The inversion phase employs a two-stage strategy to project the input image into the noise domain of the flow model: the first stage optimizes the velocity estimation via fixed-point iteration to obtain an inversion trajectory close to the actual generation process; the second stage computes a velocity compensation during editing to achieve precise reconstruction of the original image. The editing phase introduces an AdaLN-based invariance control, keeping non-target regions unchanged by replacing features corresponding to unedited text tokens, without hindering the editing effects in target regions.

### Key Designs

**1. Two-stage Flow Inversion**

- **Function**: To project real images precisely into the noise domain of the flow model while maintaining edit-friendliness.
- **Mechanism**: The first stage utilizes fixed-point iteration to improve the velocity field estimation. Standard Euler inversion approximates $v_\theta(\mathbf{x}_t, t)$ using $v_\theta(\mathbf{x}_{t+1}, t)$, which carries a substantial approximation error. Fixed-point iteration starts from $\mathbf{x}_t^0 = \mathbf{x}_{t+1}$, iteratively calculates $\mathbf{x}_t^{i+1} = \mathbf{x}_{t+1} + (\sigma_t - \sigma_{t+1})v_\theta(\mathbf{x}_t^i, t)$, and averages the results. This leverages the theoretically constant nature of the velocity field $\mathbf{x}_1 - \mathbf{x}_0$ to obtain more stable estimations. In the second stage, a compensation $\epsilon_t = \mathbf{x}_{t+1} - \hat{\mathbf{x}}_{t+1}$ is calculated during editing and added to the velocity.
- **Design Motivation**: An ideal inversion trajectory should stay close to the trajectory of the actual generation process, which both accurately reconstructs the original image and preserves the text-image alignment prior distribution for easier editing. This avoids overfitting to a single image that would otherwise degrade the model's editing capability.

**2. AdaLN-based Invariance Control**

- **Function**: To preserve non-target content during the editing process while supporting both rigid and non-rigid editing.
- **Mechanism**: In MM-DiT, the text features within the Adaptive Layer Normalization (AdaLN) correspond closely to the image semantics (e.g., pose, count, object type). A Map function is defined to keep edited token features unchanged while replacing features of unchanged tokens in the target text feature $M^a$ with corresponding token features from the original text feature $M^b$. This operation is executed from the initial timestep until timestep $S$.
- **Design Motivation**: MM-DiT processes text and image features jointly in self-attention blocks, and lacks the cross-attention control mechanism found in U-Net. However, AdaLN directly reflects text variations, enabling precise distinction between edited and non-edited semantics, which avoids the issue of attention injection limiting non-rigid editing capabilities.

**3. Velocity Averaging Strategy in Fixed-Point Iteration**

- **Function**: To improve the quality of velocity estimation at every step of the inversion trajectory.
- **Mechanism**: The ideal velocity of Rectified Flow $v_\theta = \mathbf{x}_1 - \mathbf{x}_0$ is constant. Consequently, the average of the fixed-point iteration sequence $\{\mathbf{x}_t^i\}_{i=1}^I$ yields a more accurate estimation than any single iteration result, as the iteration might oscillate around the true fixed point rather than converging monotonically.
- **Design Motivation**: Experimental visualization validates that non-averaged latents deviate significantly at different numbers of iterations, whereas they stay closer to the original image after averaging.

### Loss & Training

- The proposed approach is a tuning-free method, requiring no additional training losses.
- Built on top of Stable Diffusion 3.5 using the Euler sampler.
- The number of inversion steps is set to 30. CFG is set to 1 for inversion and 2 for editing.
- The number of fixed-point iterations is set to 3.
- The invariance control threshold timestep $S$ is set uniformly across all editing types without requiring per-type hyperparameter tuning.

## Key Experimental Results

### Main Results

Quantitative comparison on the PIE benchmark (700 natural/synthetic images):

| Method | Structure Dist↓ | PSNR↑ | LPIPS↓ | MSE↓ | SSIM↑ | CLIP Whole↑ | CLIP Edited↑ |
|------|----------------|-------|--------|------|-------|------------|-------------|
| P2P | **13.44** | 27.03 | 60.67 | 35.86 | 84.11 | 24.75 | 21.86 |
| PnP | 24.29 | 22.46 | 106.06 | 80.45 | 79.68 | **25.41** | **22.62** |
| MasaCtrl | 24.70 | 22.64 | 88.79 | 81.09 | 80.76 | 24.38 | 21.35 |
| InfEdit | 24.70 | 26.31 | 87.94 | 75.19 | 81.33 | 23.67 | 21.86 |
| RFinv | 32.62 | 22.03 | 159.62 | 96.01 | 73.26 | 24.89 | 21.89 |
| **Ours** | 18.17 | **26.62** | **80.55** | **40.24** | **91.50** | 25.74 | 22.27 |

### Ablation Study

Ablation on the number of fixed-point iterations (150 random subsets):

| Iteration Count | Structure Dist↓ | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP Whole↑ |
|---------|----------------|-------|--------|-------|------------|
| Iter 0 (plain Euler) | 48.49 | 20.78 | 199.93 | 80.59 | 24.14 |
| Iter 1 | 15.29 | 26.98 | 67.80 | 93.89 | 24.31 |
| Iter 3 | **14.92** | **27.51** | **66.48** | **94.40** | **24.48** |

### Key Findings

1. Plain Euler inversion (Iter 0) exhibits a high structure distance of 48.49. However, even 1 fixed-point iteration drops this metric to 15.29, indicating the severity of the approximation error in plain Euler inversion.
2. The proposed method demonstrates the best balance between background preservation (SSIM=91.50) and editing capability (CLIP Edited=22.27).
3. The AdaLN replacement timestep $S$ is robust across a wide range of values and does not hinder non-rigid editing, outperforming attention-injection baselines.
4. If self-attention injection exceeds 20% of the total timesteps, non-rigid editing operations like "sit $\rightarrow$ stand" are severely hindered.

## Highlights & Insights

- This work systematically uncovers the mathematical relationship between Euler inversion and DDIM inversion, demonstrating that while Euler inversion appears similar, it is significantly more prone to approximation errors.
- It leverages the constant property of the velocity field in Rectified Flow to formulate a theoretically grounded fixed-point iteration and averaging strategy.
- The AdaLN invariance control mechanism elegantly solves the challenge of unifying both rigid and non-rigid editing within a single design, bypassing the need for per-type hyperparameter tuning.
- Represents the first successful attempt to achieve high-quality tuning-free image editing on the MM-DiT architecture.

## Limitations & Future Work

- When the input image is out-of-distribution (OOD) relative to the model, the inversion error remains large, and the AdaLN control becomes less effective.
- Fixed-point iteration introduces high computational overhead during inference (requiring multiple Transformer forward passes per step).
- Editing capabilities are fundamentally bounded by the generative priors of the base model, preventing edits that the model cannot natively generate.
- Future work may explore the integration of this method with instruction-tuning approaches to further extend the limits of editing capabilities.

## Related Work & Insights

- **DDIM Inversion + Null-text Optimization**: A classic inversion-editing framework in diffusion models. This paper demonstrates its failure modes in flow models and puts forward a targeted solution.
- **Prompt-to-Prompt (P2P)**: Achieves editing through cross-attention replacement, but is inapplicable to MM-DiT due to the lack of dedicated cross-attention blocks.
- **MasaCtrl / InfEdit**: Attention manipulation approaches tailored for non-rigid editing, but they incur a trade-off with rigid consistency.
- **Insight**: For generative models featuring the next-generation Transformer architecture, exploring alternative modulation points besides attention (such as normalization layers) represents a highly prospective direction for design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Highly detailed and systematic analysis of flow model inversion; the AdaLN control scheme is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative evaluations on the PIE benchmark along with sufficient ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear mathematical derivations and systematic problem analysis.
- **Value**: ⭐⭐⭐⭐ — Lays an important foundation for image editing in flow models with a simple and practical approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](../../ICML2025/image_generation/taming_rectified_flow_for_inversion_and_editing.md)
- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[NeurIPS 2025\] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](../../NeurIPS2025/image_generation/splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)

</div>

<!-- RELATED:END -->
