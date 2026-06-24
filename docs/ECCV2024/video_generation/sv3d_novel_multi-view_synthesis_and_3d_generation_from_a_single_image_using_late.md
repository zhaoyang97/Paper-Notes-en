---
title: >-
  [Paper Note] SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion
description: >-
  [ECCV 2024][Video Generation][Multi-view Synthesis] SV3D is proposed to adapt image-to-video diffusion models for multi-view synthesis and 3D generation, leveraging the generalization capability and multi-view consistency of video models while introducing explicit camera control.
tags:
  - "ECCV 2024"
  - "Video Generation"
  - "Multi-view Synthesis"
  - "3D Generation"
  - "Video Diffusion Models"
  - "Single-image 3D Reconstruction"
  - "Camera-controllable"
date: 2026-05-08
content_hash: 733298d26a3a5299
---

# SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion

**Conference**: ECCV 2024  
**arXiv**: [2403.12008](https://arxiv.org/abs/2403.12008)  
**Code**: Yes (Project Page)  
**Area**: Video Generation  
**Keywords**: Multi-view Synthesis, 3D Generation, Video Diffusion Models, Single-image 3D Reconstruction, Camera-controllable

## TL;DR

SV3D is proposed to adapt image-to-video diffusion models for multi-view synthesis and 3D generation, leveraging the generalization capability and multi-view consistency of video models while introducing explicit camera control.

## Background & Motivation

Generating 3D models from a single image is a long-standing challenge in computer vision and computer graphics. The core difficulty lies in the fact that a single image only provides viewpoint information from a single perspective, requiring the model to "hallucinate" unobserved views.

In recent years, two categories of methods have made progress: (1) **SDS-based optimization methods** (e.g., DreamFusion, Zero-1-to-3), which use gradients from 2D diffusion models to guide 3D optimization, but suffer from slow generation speeds and are prone to the Janus problem (multi-faced issue where every side of the object looks like the front); (2) **Multi-view generation methods** (e.g., Zero-1-to-3), which directly generate target viewpoint images, but struggle to guarantee consistency across views.

Stability AI's Stable Video Diffusion (SVD) model has demonstrated strong capabilities in image-to-video generation—it inherently possesses temporal consistency, which can be re-interpreted as **multi-view consistency**. The core insight of this work is to treat the orbital video of an object as a special type of video, leveraging the temporal consistency of SVD to guarantee multi-view consistency.

SV3D adapts SVD by introducing explicit camera parameter control to achieve controllable multi-view synthesis and high-quality 3D generation.

## Method

### Overall Architecture

The pipeline of SV3D consists of two stages: (1) Multi-view synthesis stage: Given a single input image, SV3D generates a sequence of multi-view images (orbital video) around the object; (2) 3D reconstruction stage: Using the generated multi-view images as supervision, a 3D model (such as NeRF or mesh) is reconstructed through an improved 3D optimization pipeline.

### Key Designs

1. **Multi-view Adaptation of Video Diffusion Models**:
    - Function: Adapting SVD from general video generation to multi-view synthesis.
    - Mechanism: Re-interpreting the time axis of SVD as the viewpoint axis—the sequence of video frames corresponds to a sequence of camera viewpoints rotating around the object. SVD is fine-tuned to learn "orbital motion" patterns rather than free motion. The temporal attention mechanism of SVD is retained to ensure multi-view consistency.
    - Design Motivation: SVD has already learned strong temporal consistency priors, which can be directly transferred to multi-view consistency, avoiding training from scratch.

2. **Explicit Camera Condition Control**:
    - Function: Allowing users to precisely control the camera parameters (azimuth, elevation) for each output view.
    - Mechanism: Encoding the target camera pose into conditioning vectors injected into the denoising process of the U-Net. SV3D offers two variants: SV3D_u, which generates orbital videos with fixed elevation, and SV3D_p, which supports more flexible camera paths (including varying elevations). Camera conditions are embedded using positional encodings + MLPs and then added to the features of each frame.
    - Design Motivation: Uncontrolled video generation cannot precisely match target viewpoints; explicit camera control is a necessary improvement for adapting video models to 3D tasks.

3. **Improved 3D Optimization Pipeline**:
    - Function: Reconstructing high-quality 3D models from multi-view images.
    - Mechanism: Proposing several improvements to better utilize SV3D's outputs: (a) using multi-view images as direct pixel-level supervision (rather than SDS); (b) introducing an occlusion-aware loss to handle self-occlusion regions; (c) designing a progressive training scheme to optimize the 3D representation from coarse to fine. Compared to pure SDS methods, direct multi-view supervision provides more stable gradients.
    - Design Motivation: The quality of multi-view images generated by SV3D is sufficiently high that they can be directly used as reconstruction targets, eliminating the reliance on unstable SDS losses.

### Loss & Training

- Diffusion Training Loss: Standard denoising loss on multi-view datasets, conditioned on camera parameters.
- Noise Schedule: A noise schedule optimized for multi-view data.
- 3D Optimization Loss: Multi-view RGB reconstruction loss + perceptual loss + regularization loss.
- Training Data: Multi-view images rendered from large-scale 3D asset datasets such as Objaverse.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Google Scanned Objects | PSNR $\uparrow$ | SOTA | Zero-1-to-3 XL | +2-4dB |
| Google Scanned Objects | SSIM $\uparrow$ | SOTA | One-2-3-45 | +0.05-0.10 |
| Objaverse | LPIPS $\downarrow$ | SOTA | SyncDreamer | Significantly reduced |
| User Study | 3D Quality Preference | Best | All baselines | >70% preference rate |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| No camera control | Random view | Cannot specify target viewpoint |
| SDS optimization vs Multi-view supervision | Multi-view is better | More stable, fewer artifacts |
| SV3D_u vs SV3D_p | _p is more flexible | Supports paths with varying elevations |
| Different number of views | More is better | But with diminishing marginal returns |

### Key Findings

- The temporal consistency prior of video diffusion models can be effectively transferred to multi-view consistency.
- Explicit camera control is a crucial enhancement for adapting video models to 3D tasks.
- Direct multi-view supervision yields better 3D reconstruction results than SDS optimization.
- SV3D achieves state-of-the-art (SOTA) performance on both novel view synthesis (NVS) and 3D reconstruction.

## Highlights & Insights

- Elegant Core Insight: An orbital video is essentially a sequence of multi-view images, making video models naturally suited for this task.
- Stability AI's industrial-grade implementation ensures model quality and scalability.
- SV3D_p supports flexible camera paths, making it more practical than fixed orbits.
- The paradigm shift from 2D video diffusion to 3D generation has a profound impact.

## Limitations & Future Work

- The generation quality is ultimately limited by the capability of the base SVD model.
- For heavily occluded or geometrically complex objects, multi-view consistency may degrade.
- The 3D optimization phase still takes considerable time, and feed-forward 3D generation is not yet realized.
- It primarily targets object-level 3D generation, while scene-level 3D generation remains a more challenging direction.
- Combining with 3D Gaussian Splatting could achieve faster reconstruction.

## Related Work & Insights

- **Zero-1-to-3**: Pioneering work using diffusion models for single-view to novel-view synthesis.
- **DreamFusion**: The pioneer of SDS-based 3D generation, but limited in efficiency and quality.
- **Stable Video Diffusion**: The base model for SV3D, providing the temporal consistency prior.
- Insight: Large-scale pretrained generative models contain rich 3D prior knowledge; the key lies in how to correctly extract and utilize it.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of adapting video diffusion to multi-view synthesis is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 2D/3D metrics, user studies, and multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured with detailed methodological descriptions.
- Value: ⭐⭐⭐⭐⭐ Released by Stability AI, high-impact, driving the advancement of single-image 3D generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](../../ICCV2025/video_generation/fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)
- [\[ECCV 2024\] MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)
- [\[ECCV 2024\] MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)
- [\[ECCV 2024\] VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)
- [\[CVPR 2025\] Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency](../../CVPR2025/video_generation/geometry-guided_online_3d_video_synthesis_with_multi-view_temporal_consistency.md)

</div>

<!-- RELATED:END -->
