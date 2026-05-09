---
title: >-
  [Paper Note] DIVE: Taming DINO for Subject-Driven Video Editing
description: >-
  [Video Generation] This paper proposes DIVE, a framework that leverages semantic features from the pretrained DINOv2 model as implicit correspondences to guide subject-driven video editing. DINO features are used for temporal motion modeling and target subject identity registration, enabling high-quality subject replacement while preserving motion consistency.
tags:
  - Video Generation
date: 2026-05-08
content_hash: 5458511c2646b614
---

# DIVE: Taming DINO for Subject-Driven Video Editing

## Basic Information
- **Conference**: ICCV 2025
- **arXiv**: 2412.03347
- **Code**: Not released
- **Area**: Image/Video Generation / Video Editing
- **Keywords**: Video editing, DINO features, subject-driven, motion consistency, LoRA, diffusion models

## TL;DR

This paper proposes DIVE, a framework that leverages semantic features from the pretrained DINOv2 model as implicit correspondences to guide subject-driven video editing. DINO features are used for temporal motion modeling and target subject identity registration, enabling high-quality subject replacement while preserving motion consistency.

## Background & Motivation

- **Problem Definition**: Subject-driven video editing requires replacing subjects (animals, objects, etc.) in source videos such that the edited video preserves the original motion trajectory and temporal consistency while faithfully rendering the target subject's identity.
- **Limitations of Prior Work**:
    - Attention-based methods (Tune-A-Video, FateZero, etc.): Transfer motion information via attention maps/feature injection, but stored features may retain source subject appearance, causing blending between source and target appearances.
    - Dense correspondence-based methods (optical flow, depth maps, edge maps): High density leads to visual incoherence and flickering.
    - VideoSwap uses sparse semantic points as correspondences: Can precisely align motion trajectories, but requires manual annotation.
- **Key Finding**: DINOv2 features exhibit high semantic consistency across video frames while containing minimal appearance information, making them naturally suitable as robust video correspondences (Fig. 2 demonstrates three key advantages of DINO features: motion tracking, appearance sparsity, and semantic consistency with intra-frame discriminability).

## Method

### Overall Architecture

DIVE consists of three stages: (1) Temporal Motion Modeling → (2) Subject Identity Registration → (3) Inference.

### Stage 1: Temporal Motion Modeling

**Objective**: Extract motion information from the source video to serve as motion guidance during editing.

- DINOv2 (ViT-g/14) semantic features $\mathbf{F}_d \in \mathbb{R}^{N \times h \times w \times c}$ are extracted from each frame of the source video.
- PCA dimensionality reduction with thresholding is applied to automatically generate foreground masks and isolate foreground subject features.
- A set of learnable MLPs $\boldsymbol{\psi} = \{\boldsymbol{\psi}_l | l \in \{1,2,3,4\}\}$ projects DINO features into the diffusion model feature space.
- The projected features are injected into the intermediate features of each downsampling block in the U-Net encoder via element-wise addition:

$$\mathbf{F}_l^t \leftarrow \mathbf{F}_l^t + \lambda \mathbf{F}_l^s$$

- **Optimization objective**: Only MLP parameters $\boldsymbol{\psi}$ are optimized, enhancing the diffusion model's denoising capability in foreground regions:

$$\min_{\boldsymbol{\psi}} \mathbb{E}_{\epsilon,t} \|[\epsilon - \epsilon_{\theta'}(\mathbf{Z}^t, t, \mathbf{c}, \boldsymbol{\psi}(\mathbf{F}_d))] \odot \mathbf{M}\|_2^2$$

- Training is restricted to higher timesteps $t \in [T/2, T]$ to prevent overfitting to low-level subject details.

### Stage 2: Subject Identity Registration

**Objective**: Learn the target subject's identity from reference images.

- Three to five reference images of the target subject are used.
- DINOv2 foreground features $\tilde{\mathbf{F}}_d$ are extracted from the reference images and projected into the diffusion feature space via a separate MLP set $\boldsymbol{\phi}$.
- The projected features are fused with Stable Diffusion intermediate features to provide accurate identity guidance.
- A LoRA $\Delta\theta$ is trained to register the target identity:

$$\min_{\boldsymbol{\phi}, \Delta\theta} \mathbb{E}_{\epsilon,t} \|[\epsilon - \epsilon_{\theta+\Delta\theta}(\mathbf{I}^t, t, \tilde{\mathbf{c}}, \boldsymbol{\phi}(\tilde{\mathbf{F}}_d))] \odot \tilde{\mathbf{M}}\|_2^2$$

- DINO features provide high-level semantic matching (e.g., part-level correspondences), compensating for SD features' limitation of capturing only low-level spatial information.

### Stage 3: Inference

- DDIM inversion is applied to obtain the initial noise of the source video.
- Motion guidance features learned in Stage 1 are injected during denoising steps from $T$ to $T/2$.
- The pretrained LoRA from Stage 2 provides identity guidance.
- The source subject token in the text prompt is replaced (e.g., "cat" → "dog").
- Latent blending with foreground masks preserves the background.

## Key Experimental Results

### Quantitative Evaluation (Reference Image-Guided)

| Method | Text Align↑ | Image Align↑ | Temporal Consist↑ | Video Quality↑ | User Preference↑ |
|------|------------|-------------|-------------------|---------------|----------|
| Slicedit | 28.21 | 64.57 | 91.09 | 0.592 | 6.73% |
| AnyV2V | 28.13 | 78.26 | 90.52 | 0.439 | 13.2% |
| FLATTEN | 28.79 | 69.32 | 92.09 | 0.683 | 8.67% |
| RAVE | 28.26 | 66.25 | 91.71 | 0.646 | 5.80% |
| **DIVE** | **29.43** | **84.27** | **92.33** | **0.775** | **65.6%** |

DIVE outperforms all competing methods on every metric, with a user preference rate of 65.6%—far exceeding the runner-up at 13.2%.

### Ablation Study

| Configuration | Text Align↑ | Image Align↑ | Temporal Consist↑ | Video Quality↑ |
|------|------------|-------------|-------------------|---------------|
| w/o DINO, w/ learnable motion | 29.91 | 67.49 | 92.18 | 0.737 |
| w/ DINO λ=0 (LoRA only) | - | - | - | Motion inconsistency |
| w/ DINO λ=0.5 | - | - | - | Partial motion loss |
| **w/ DINO λ=1.0 (full)** | **29.43** | **84.27** | **92.33** | **0.775** |

**Motion guidance ablation**: Without DINO features, relying solely on AnimateDiff's temporal modeling results in residual source subject appearance and misalignment. The value of λ controls the strength of motion guidance.

**Identity guidance ablation**: Identity registration without DINO guidance produces semantic errors (e.g., a Corgi with vs. without a tail); DINO provides part-level semantic guidance that improves faithfulness to the reference images.

### Text-Guided Editing Results

| Method | Text Align↑ | Temporal Consist↑ | Video Quality↑ | User Preference↑ |
|------|------------|-------------------|---------------|----------|
| Slicedit | 31.24 | 92.95 | 0.562 | 5.50% |
| AnyV2V | 31.05 | 93.73 | 0.533 | 19.9% |
| FLATTEN | 31.55 | 95.35 | 0.567 | 11.9% |
| RAVE | 31.57 | 95.12 | 0.588 | 9.90% |
| **DIVE** | **32.29** | **95.89** | **0.614** | **52.8%** |

## Highlights & Insights

1. **Dual utility of DINO features**: Used simultaneously for motion modeling and identity registration, grounded in three key properties—motion tracking capability, appearance sparsity, and semantic consistency.
2. **Motion–appearance disentanglement**: The appearance sparsity of DINO features naturally decouples motion from appearance, avoiding the appearance leakage observed in attention-based methods.
3. **No manual annotation required**: Unlike VideoSwap, which requires manually defined semantic points, DINO features automatically provide robust implicit correspondences.
4. **Automatic foreground segmentation via PCA**: PCA dimensionality reduction with thresholding over DINO features automatically generates foreground masks without requiring an additional segmentation model.
5. **Dominant user preference margin**: A 65.6% user preference rate substantially surpasses competing methods, indicating a significant perceptual quality advantage.

## Limitations & Future Work

- Built upon Stable Diffusion 1.5; generation quality is constrained by the base model.
- Each video requires independent training of Stage 1 and Stage 2 (50–100 and 800–1000 iterations, respectively), limiting efficiency.
- Validated only on 16-frame short videos; performance on longer videos remains unexplored.
- Reference image-guided editing requires 3–5 target subject images; single-image scenarios are not investigated.
- Relies on DDIM inversion quality, which may be unstable for complex backgrounds or large motions.

## Related Work & Insights

- **VideoSwap**: Uses sparse semantic points but requires manual annotation; DIVE automates this process with DINO features.
- **TokenFlow**: Propagates diffusion features via inter-frame correspondences but still suffers from appearance leakage.
- **RAVE**: Uses depth-map-based correspondences; high density introduces flickering.
- **Insight**: DINO features hold substantial potential in the video domain; the simple PCA + thresholding foreground segmentation strategy is particularly practical.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The motivation for using DINO features as video correspondences is clear and the application is elegant.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Comprehensive quantitative comparisons, ablations, and a convincing user study.)
- **Writing Quality**: ⭐⭐⭐⭐ (PCA visualizations and other analyses are intuitive; the pipeline diagram is clear.)
- **Value**: ⭐⭐⭐⭐ (First systematic exploration of DINO's potential in video editing, opening a new research direction.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)

</div>

<!-- RELATED:END -->
