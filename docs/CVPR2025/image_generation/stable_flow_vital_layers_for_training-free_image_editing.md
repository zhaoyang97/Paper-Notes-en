---
title: >-
  [Paper Note] Stable Flow: Vital Layers for Training-Free Image Editing
description: >-
  [CVPR 2025][Image Generation][Image Editing] Stable Flow proposes to automatically detect "vital layers" in DiT (FLUX) and inject attention features of the reference image only into these layers to achieve various training-free image editing operations, while introducing a latent nudging technique to improve the quality of flow model inversion for real images.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Editing"
  - "DiT"
  - "Flow Matching"
  - "Attention Injection"
  - "Training-Free Editing"
date: 2026-05-08
content_hash: f53cf1972d736252
---

# Stable Flow: Vital Layers for Training-Free Image Editing

**Conference**: CVPR 2025  
**arXiv**: [2411.14430](https://arxiv.org/abs/2411.14430)  
**Code**: [Project Page](https://omriavrahami.com/stable-flow)  
**Area**: Image Generation / Image Editing  
**Keywords**: Image Editing, DiT, Flow Matching, Attention Injection, Training-Free Editing

## TL;DR
Stable Flow proposes to automatically detect "vital layers" in DiT (FLUX) and inject attention features of the reference image only into these layers to achieve various training-free image editing operations, while introducing a latent nudging technique to improve the quality of flow model inversion for real images.

## Background & Motivation
- Next-generation T2I models like FLUX and SD3 adopt the DiT architecture combined with flow matching, significantly improving generation quality.
- Flow-based models sample along straight trajectories, leading to reduced diversity—however, this work views this limitation as an **editing advantage**.
- Editing methods in the UNet era (such as P2P, MasaCtrl) leveraged the coarse-to-fine-to-coarse hierarchical structure of UNet for attention injection.
- The DiT architecture **lacks** the hierarchical structure of UNet, with unclear roles for each layer, making it impossible to directly migrate former injection strategies.
- Standard inverse Euler ODE inversion performs poorly on FLUX, leading to reconstruction failures.
- Two core problems need to be solved: (1) **Which layers** in DiT are suitable for injecting features? (2) How can real images be effectively inverted?

## Method

### Overall Architecture
A three-step framework: (1) **Vital Layers Detection**—removes each layer of DiT one by one to measure its perceptual impact on the generated image, defining high-impact layers as vital layers; (2) **Vital Layers Injection Editing**—generates the reference and edited images in parallel, replacing the image embeddings of the edited image with those of the reference image only within the vital layers; (3) **Latent Nudging**—multiplies the initial latent by a small coefficient $\lambda=1.15$ during real image inversion to improve reconstruction quality.

### Key Designs

**Design 1: Automatic Vital Layers Detection**
- **Function**: Identification of the subset of layers in DiT that are critical for image formation.
- **Mechanism**: A set of reference images $G_{ref}$ is generated using 64 diverse text prompts; for each layer $\ell$, the layer is bypassed via residual connection to generate $G_\ell$; DINOv2 is used to calculate the perceptual similarity between $G_{ref}$ and $G_\ell$; the vitality score is defined as $vitality(\ell) = 1 - \frac{1}{k} \sum d(M_{full}, M_{-\ell})$; layers that exceed a threshold $\tau_{vit}$ are defined as vital layers.
- **Design Motivation**: DiT lacks the structured hierarchy of UNet; experiments show that vital layers are scattered throughout the Transformer (not concentrated in one specific region); bypass experiments quantify the actual contribution of each layer to the image content.

**Design 2: Vital Layers Attention Injection Editing**
- **Function**: Replacement of reference image embeddings only in vital layers to enable diverse editing operations (non-rigid deformation, object addition, global modification).
- **Mechanism**: Parallel generation of the source image $x$ (source prompt + seed) and the edited image $\hat{x}$ (edit prompt + same seed); in the vital layers $V$, the image embeddings of $\hat{x}$ are replaced with those of $x$; non-vital layers retain the original embeddings of $\hat{x}$.
- **Design Motivation**: Analysis reveals that multimodal attention in vital layers achieves a favorable balance between preserving source content and responding to text edits—unchanged regions primarily focus on visual features, while edited regions focus on text tokens (e.g., "avocado"); non-vital layers focus almost entirely on the image, which is unfavorable for editing.

**Design 3: Latent Nudging Real Image Inversion**
- **Function**: Improving the inversion reconstruction quality of real images in the FLUX model.
- **Mechanism**: Prior to inversion, the clean latent $z_0$ is multiplied by $\lambda=1.15$ to slightly shift it out of the training distribution; subsequently, the standard inverse Euler ODE is applied: $z_t = z_{t-1} + (\sigma_t - \sigma_{t+1}) \cdot u_t(z_{t-1})$.
- **Design Motivation**: Standard inversion assumes that $u(z_t) \approx u(z_{t-1})$, but the straight trajectories in FLUX make this assumption invalid; a slight shift makes the model less likely to alter the image content during the forward process.

### Loss & Training
- A training-free method requiring no loss function.
- DINOv2 perceptual similarity is used only during the vital layers detection stage.

## Key Experimental Results

### Main Results: Quantitative Comparison

| Method | CLIP_txt↑ | CLIP_img↑ | CLIP_dir↑ |
|------|------|------|------|
| SDEdit | 0.24 | 0.71 | 0.07 |
| P2P+NTI | 0.21 | 0.76 | 0.08 |
| Instruct-P2P | 0.22 | 0.87 | 0.07 |
| MagicBrush | 0.24 | 0.88 | 0.11 |
| MasaCTRL | 0.20 | 0.76 | 0.03 |
| **Stable Flow** | **0.23** | **0.92** | **0.14** |

### Ablation Study

| Configuration | CLIP_txt↑ | CLIP_img↑ | CLIP_dir↑ |
|------|------|------|------|
| **Stable Flow** | **0.23** | **0.92** | **0.14** |
| All-layer injection | 0.17 | 0.98 | 0.00 |
| Non-vital layer injection | 0.25 | 0.72 | 0.09 |
| No latent nudging | 0.22 | 0.62 | 0.05 |

### User Study (Two-Alternative Forced Choice, Win Rate)

| vs Method | Prompt Alignment | Image Preservation | Realism | Overall |
|------|------|------|------|------|
| vs SDEdit | 69% | 68% | 64% | 71% |
| vs MasaCTRL | 82% | 80% | 80% | 72% |
| vs MagicBrush | 61% | 67% | 77% | 74% |

### Key Findings
- All-layer injection -> CLIP_dir drops to 0 (completely copying the source image, resulting in no editing effect).
- Non-vital layer injection -> significant drop in image preservation (CLIP_img 0.72), resulting in over-editing.
- Approximately 20 vital layers are selected (out of 57 layers in FLUX), scattered across the Transformer.
- Latent nudging improves CLIP_img from 0.62 to 0.92.

## Highlights & Insights
- **Turning the low diversity drawback of flow models into an editing advantage** is a very clever perspective.
- The automatic detection method for vital layers is highly generalizable and can be applied to the analysis of other DiT models.
- The same injection mechanism can handle non-rigid editing, object addition, and scene modification—achieving high unification.
- Latent nudging is extremely simple (multiplying by 1.15) yet highly effective.

## Limitations & Future Work
- Validated only on the FLUX model; applicability to other DiT models such as SD3 requires testing.
- The set of vital layers is fixed across all editing types, which may not be optimal for every edit.
- The coefficient $\lambda=1.15$ for latent nudging is empirical, lacking a theoretical explanation.
- Future directions could explore adaptive vital layer selection, controllable editing strength, and extension to video editing.

## Related Work & Insights
- P2P/MasaCTRL: Attention injection editing methods from the UNet era.
- FLUX/SD3: Latest DiT + flow matching T2I models.
- SDEdit: A classic method to achieve image editing through adding noise and denoising.
- Insight: Understanding the internal **functional division of layers** is a key prerequisite for designing training-free editing methods.

## Rating
⭐⭐⭐⭐ — The perspective of converting the low diversity of DiT into an editing advantage is highly novel; the vital layers detection method is generalizable and insightful; latent nudging is simple yet highly effective; the comprehensive quantitative and user study validation is very convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing](unveil_inversion_and_invariance_in_flow_transformer_for_versatile_image_editing.md)
- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](../../NeurIPS2025/image_generation/training-free_constrained_generation_with_stable_diffusion_models.md)
- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)
- [\[CVPR 2025\] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](diff2flow_training_flow_matching_models_via_diffusion_model_alignment.md)
- [\[ICLR 2026\] FlowAlign: Trajectory-Regularized, Inversion-Free Flow-based Image Editing](../../ICLR2026/image_generation/flowalign_trajectory-regularized_inversion-free_flow-based_image_editing.md)

</div>

<!-- RELATED:END -->
