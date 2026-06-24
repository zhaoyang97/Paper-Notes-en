---
title: >-
  [Paper Note] MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion
description: >-
  [ICCV2025][Model Compression][video motion editing] This paper proposes MotionFollower, which achieves video motion editing via two lightweight convolutional controllers (pose + appearance) and a consistency guidance mechanism based on score function regularization, surpassing strong baselines such as MotionEditor while reducing GPU memory consumption by approximately 80%.
tags:
  - "ICCV2025"
  - "Model Compression"
  - "video motion editing"
  - "diffusion models"
  - "score guidance"
  - "lightweight controller"
  - "pose transfer"
date: 2026-05-08
content_hash: 7982216129cf9613
---

# MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion

**Conference**: ICCV2025
**arXiv**: [2405.20325](https://arxiv.org/abs/2405.20325)  
**Code**: [francis-rings/MotionFollower](https://francis-rings.github.io/MotionFollower/)  
**Area**: Model Compression
**Keywords**: video motion editing, diffusion models, score guidance, lightweight controller, pose transfer

## TL;DR

This paper proposes MotionFollower, which achieves video motion editing via two lightweight convolutional controllers (pose + appearance) and a consistency guidance mechanism based on score function regularization, surpassing strong baselines such as MotionEditor while reducing GPU memory consumption by approximately 80%.

## Background & Motivation

- **Limitations of existing video editing**: Current diffusion model-driven video editing primarily focuses on attribute-level editing (style transfer, appearance/background modification), with very limited attention to motion information—the most distinctive and complex characteristic of video.
- **Shortcomings of MotionEditor**: As the sole pioneering work capable of motion editing, MotionEditor relies on ControlNet combined with attention injection, suffering from three major issues: (1) poor temporal consistency, (2) prohibitive computational overhead due to the attention-heavy ControlNet, and (3) performance degradation under large camera motion and complex backgrounds.
- **Human motion transfer vs. motion editing**: Existing human motion transfer methods (MagicAnimate, AnimateAnyone, Champ) aim at generating animations from a single image, which fundamentally differs from video motion editing—the latter must simultaneously preserve camera motion, per-frame background variation, and subject appearance.
- **Core motivation**: To design a lightweight, efficient motion editing framework that preserves source video details under complex scenarios involving large camera motion and complex backgrounds.

## Method

### Overall Architecture

MotionFollower is built upon a T2I diffusion model (LDM/SD1.5), inflating the 2D U-Net into a 3D U-Net by inserting temporal layers. The overall framework comprises two core designs:

1. **Two lightweight conditional controllers**: Pose Controller (PoCtr) + Reference Controller (ReCtr)
2. **Score-guided consistency regularization at inference**: dual-branch architecture (reconstruction branch + editing branch) + multiple loss functions

### Lightweight Conditional Controllers

#### Pose Controller (PoCtr)
- **Function**: Encodes target pose signals to control motion modification.
- **Architecture**: Consists of only 4 convolutional blocks, each containing 2 convolutional layers, with **no attention computation whatsoever**.
- **Mechanism**: Encodes pose signals into a representation of the same dimensionality as the initial latent, which is directly added to the latent prior to denoising.
- **Initialization strategy**: The final projection layer weights are initialized to zero to avoid introducing excessive perturbation at the early stage of training.
- Compared to ControlNet: parameter count and computational cost are substantially reduced, and the robustness degradation caused by directly generating modified motion from random noise is avoided.

#### Reference Controller (ReCtr)
- **Function**: Preserves appearance information from the source video, replacing time-consuming DDIM inversion.
- **Architecture**: Likewise 4 convolutional blocks (2 convolutional layers each), downsampling source frame features to the same dimensionality as the latent.
- **Multi-scale injection**: Multi-scale source features are directly added to the features at each encoder block of the U-Net.
- **CLIP embedding**: Source frames are additionally projected via a CLIPProjector into CLIP embeddings, which interact with the latent through cross-attention to preserve precise low-level details.
- Compared to MagicAnimate/AnimateAnyone: The latter employ a full U-Net as the reference network with numerous attention operations and require channel concatenation, incurring approximately twice the memory overhead of ReCtr.

### Loss & Training (Two-Stage)

| Stage | Data | Trainable Modules | Objective |
|-------|------|-------------------|-----------|
| Stage 1 (image) | Single frames; source/target randomly sampled from the same video | PoCtr + ReCtr + U-Net (temporal layers removed) | Learn image editing capability given source image and target pose |
| Stage 2 (video) | Video clips + 40% probability of degrading to image training | Temporal layers only (rest frozen) | Enhance temporal consistency; temporal layers initialized from AnimateDiff pretrained weights |

- Training data: 3K internet videos (60–90 seconds), cropped to 512×512.
- Each stage is trained for 100K steps with learning rate 1e-5.
- DWPose is used for pose extraction; a proprietary lightweight segmentation model is used for mask extraction.

### Score-Guided Consistency Regularization (Core Inference Design)

**Core Idea**: The denoising process of a diffusion model can be viewed as a continuous process guided by the score function (the gradient of the log probability density with respect to the data). By adding a regularization term to the score function, the model can be forced to preserve semantic details from the source video without updating its weights.

**Dual-Branch Architecture**:
- **Reconstruction branch**: random noise + source pose + source frames → reconstruct source video
- **Editing branch**: random noise + target pose + source frames → generate edited video

**Score Function Decomposition**:

$$\nabla_{z_t^e} \log q(z_t^e, F_t^e, F_t^r) = \underbrace{\nabla_{z_t^e} \log q(z_t^e)}_{\text{original denoising score}} + \underbrace{\nabla_{z_t^e} \log q(F_t^e, F_t^r | z_t^e)}_{\text{consistency regularization}}$$

**Region-wise Loss Design**: Segmentation masks are used to decompose the loss into foreground and background components.

#### Foreground Loss $\mathcal{L}_{fg}$
- Maximizes the similarity between foreground features of the editing branch and the reconstruction branch.
- Employs mask pooling and spatially averaged cosine similarity.
- Ensures subject appearance remains consistent after motion modification.

#### Background Loss $\mathcal{L}_{bg}$ (Three Sub-losses)

| Loss | Mask Definition | Function |
|------|-----------------|----------|
| $\mathcal{L}_{over}$ (overlapping background) | $(1-M^e) \odot (1-M^r)$ | Enforces consistency between overlapping background regions of the edited and reconstructed outputs |
| $\mathcal{L}_{body}$ (non-overlapping human body) | $M^r \odot (1-M^e)$ | Mitigates ghosting/blurring caused by source pose bias |
| $\mathcal{L}_{com}$ (complementary) | non-overlapping body region vs. source background | Guides the model to inpaint non-overlapping body regions with background information |

**Final Guidance**: Foreground and background gradients are combined via masks to update the latent in a region-wise manner:

$$\nabla_{z_t^e} \log q(F_t^e, F_t^r | z_t^e) = \frac{d\mathcal{L}_{fg}}{dz_t^e} \odot M^e + \frac{d\mathcal{L}_{bg}}{dz_t^e} \odot (1-M^e)$$

Key: **only the latent is optimized; model weights are not updated**.

## Key Experimental Results

### Main Results (100 in-the-wild videos)

| Model | L1↓ | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FID-VID↓ | FVD↓ | Memory↓ |
|-------|-----|-------|-------|--------|------|----------|------|---------|
| MotionEditor | 9.13E-5 | 17.34 | 0.68 | 0.34 | 31.98 | 20.57 | 395.43 | 42.6G |
| MagicAnimate | 1.09E-4 | 16.22 | 0.62 | 0.35 | 33.04 | 26.59 | 477.65 | 21.8G |
| AnimateAnyone | 9.45E-5 | 16.18 | 0.65 | 0.32 | 35.81 | 28.31 | 515.57 | 16.1G |
| Champ | 9.94E-5 | 16.12 | 0.58 | 0.36 | 36.55 | 25.89 | 452.65 | 17.4G |
| **MotionFollower** | **6.31E-5** | **20.85** | **0.75** | **0.22** | **26.30** | **12.42** | **276.04** | **9.8G** |

- Leads across all metrics; FVD improves from 395→276 (↓30%); memory reduces from 42.6G→9.8G (↓77%).
- On a single A100, MotionEditor can only edit 16 frames, while MotionFollower can edit 56 frames.

### Human Motion Transfer (TikTok Benchmark)

| Model | SSIM↑ | LPIPS↓ | FVD↓ |
|-------|-------|--------|------|
| Champ | 0.773 | 0.235 | 170.20 |
| **MotionFollower** | **0.793** | **0.230** | **159.88** |

Achieves state-of-the-art results on the simpler motion transfer task as well.

### Ablation Study

| Configuration | FID↓ | FID-VID↓ | FVD↓ | Memory |
|---------------|------|----------|------|--------|
| w/o ReCtr | 64.82 | 47.28 | 545.39 | 8.3G |
| Replace ReCtr→RNet | 26.35 | 13.12 | 288.57 | **17.8G** |
| Replace PoCtr→CNet | 30.57 | 23.75 | 381.52 | **15.2G** |
| w/o score guidance | 35.91 | 28.10 | 437.69 | 7.1G |
| **Full model** | **26.30** | **12.42** | **276.04** | **9.8G** |

- Removing ReCtr causes a catastrophic performance drop, highlighting the critical role of appearance control.
- Replacing ReCtr with RNet doubles memory while yielding slightly worse performance, validating the superiority of the pure convolutional design.
- Score guidance contributes significantly to background consistency.

## Highlights & Insights

1. **Extreme lightweight design**: Replacing ControlNet and Reference Net with purely convolutional controllers reduces GPU memory by 80%—an excellent demonstration of "less is more." Complex attention mechanisms in fact disrupt the distributional alignment between the reference network and the U-Net.
2. **Elegance of score guidance**: Model weights are left untouched; the latent is optimized at inference time solely through loss gradients. This "test-time optimization" paradigm can be seamlessly transferred to other diffusion models.
3. **Fine-grained region-wise loss design**: Distinct constraints are imposed on foreground, background, overlapping, and non-overlapping regions via segmentation masks, effectively addressing ghosting, blurring, and background inconsistency in motion editing.
4. **Long video and large camera motion**: MotionEditor suffers severe degradation on 600-frame long videos and under large camera motion, while MotionFollower handles these scenarios robustly through consistency guidance.
5. **Two-stage training + mixed training strategy**: In the second stage, a 40% probability of degrading to image training effectively prevents temporal layer training from compromising single-frame editing capability.

## Limitations & Future Work

1. **Dependence on pose detection quality**: PoCtr relies on DWPose for pose extraction; editing quality may degrade when pose detection fails (severe occlusion, unconventional poses).
2. **Resolution limitation**: Training and testing are conducted at 512×512; scalability to high-resolution (1024+) scenarios remains unexplored.
3. **Based on SD1.5 architecture**: The more recent SDXL or DiT architectures are not leveraged, potentially limiting the upper bound of generation quality.
4. **Inference speed**: Processing 24 frames requires 50 seconds on an A100, leaving a substantial gap for real-time applications; score guidance requires dual-branch forward passes, increasing inference overhead.
5. **Human motion only**: The current approach is limited to pose-driven human motion editing; object-level or scene-level motion editing is not addressed.
6. **Dependence on segmentation model**: Score guidance requires accurate foreground/background segmentation masks; segmentation errors propagate into the editing results.

## Related Work & Insights

- **MotionEditor** (ICLR 2024): The pioneering motion editing work using ControlNet + attention injection; the primary baseline for this paper.
- **AnimateDiff** (ICLR 2024): Provides pretrained temporal layer weights used for initialization in Stage 2 of MotionFollower's training.
- **MagicAnimate / AnimateAnyone / Champ**: Human motion transfer methods employing a full U-Net as the reference network; this paper demonstrates that a pure convolutional alternative is superior.
- **Score guidance**: Conceptually rooted in SDE theory and classifier guidance, but here innovatively applied to region-level consistency constraints rather than global guidance.
- **Inspiration**: The paradigm of lightweight controllers + test-time latent optimization is generalizable to other conditional video generation tasks (e.g., video inpainting, video super-resolution).

## Rating
- Novelty: ⭐⭐⭐⭐ — Score-guided consistency regularization and pure convolutional controller design are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage of quantitative, qualitative, ablation, human evaluation, long video, and camera motion experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ — The 80% memory reduction carries significant practical value; the lightweight design philosophy is broadly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[ICLR 2026\] DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing](../../ICLR2026/model_compression/diffvax_optimization-free_image_immunization_against_diffusion-based_editing.md)
- [\[CVPR 2026\] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](../../CVPR2026/model_compression/prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)
- [\[ICCV 2025\] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning](multi-object_sketch_animation_by_scene_decomposition_and_motion_planning.md)
- [\[CVPR 2026\] F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling](../../CVPR2026/model_compression/f2hdr_two-stage_hdr_video_reconstruction_via_flow_adapter_and_physical_motion_mo.md)

</div>

<!-- RELATED:END -->
