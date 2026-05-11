---
title: >-
  [Paper Note] Reangle-A-Video: 4D Video Generation as Video-to-Video Translation
description: >-
  [ICCV 2025][Video Generation][Multi-view video] Reangle-A-Video reformulates multi-view video generation as a video-to-video translation problem. It learns view-invariant motion via self-supervised fine-tuning of a video…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Multi-view video"
  - "video translation"
  - "view transfer"
  - "camera control"
  - "diffusion models"
  - "LoRA"
  - "DUSt3R"
date: 2026-05-08
content_hash: 77e6af847a52e472
---

# Reangle-A-Video: 4D Video Generation as Video-to-Video Translation

**Conference**: ICCV 2025
**arXiv**: [2503.09151](https://arxiv.org/abs/2503.09151)
**Area**: Video Generation · 4D Generation
**Keywords**: Multi-view video, video translation, view transfer, camera control, diffusion models, LoRA, DUSt3R

## TL;DR

Reangle-A-Video reformulates multi-view video generation as a video-to-video translation problem. It learns view-invariant motion via self-supervised fine-tuning of a video diffusion model, and combines DUSt3R-guided multi-view consistent inpainting to generate synchronized multi-view videos from a monocular input video.

## Background & Motivation

Generating synchronized multi-view videos from a single input video is a core requirement for 4D content creation. Dominant approaches train multi-view video diffusion models on large-scale 4D datasets, but suffer from the following issues:

**Data scarcity**: High-quality multi-view dynamic video data is extremely rare, and synthetic data exhibits large domain gaps.

**Domain limitation**: Models trained on synthetic assets fail to generalize to real-world scenes.

**Non-video input**: Most existing methods generate from text or images rather than from user-provided videos.

**Closed source**: The majority of methods do not release code.

Core Idea: **Decompose view change into view-dependent appearance (starting image) and view-invariant motion (image-to-video)**, handled separately by off-the-shelf image and video diffusion priors.

## Method

### Stage I: Point-Based Video Warping Data Augmentation

Given an input video $\mathbf{x}^{1:N}$:
1. Estimate per-frame depth maps $\mathbf{D}^i$ using Depth Anything V2.
2. Lift RGBD images to point clouds $\mathcal{P}^i = \phi_{2\to3}([\mathbf{x}^i, \mathbf{D}^i], \mathbf{K}, \mathbf{P}^i_{\text{src}})$.
3. Define $M$ target camera extrinsic trajectories $\Phi_j = \{\mathbf{P}^1_j, ..., \mathbf{P}^N_j\}$.
4. Re-project to obtain warped videos and visibility masks: $(\hat{\mathbf{x}}^i_j, \mathbf{m}^i_j) = \phi_{3\to2}(\mathcal{P}^i, \mathbf{K}, \mathbf{P}^i_j)$.

**Static view transfer**: the target trajectory remains constant across all frames. **Dynamic camera control**: the target pose changes incrementally per frame.

### Stage II: Multi-View Motion Learning

LoRA (rank=128) is applied to fine-tune the 3D full-attention layers of CogVideoX-5b (MM-DiT architecture), optimizing approximately 2% of the parameters.

Key design — **Masked Diffusion Loss**:

$$\mathbb{E}[\|\boldsymbol{\epsilon} \odot \mathbf{m}_{\text{down}}^{1:N} - \epsilon_\theta(\mathbf{z}_t^{1:N}, t, c) \odot \mathbf{m}_{\text{down}}^{1:N}\|_2^2]$$

Loss is computed only on visible pixels, preventing black regions from corrupting the original model prior. Warped and original videos are trained jointly, enabling the model to learn view-invariant scene motion.

**Dynamic camera control** requires explicit specification of camera motion type in the text prompt (e.g., "horizontal orbit left"), since all warped videos share the same starting frame.

### Stage III: Multi-View Consistent Image Inpainting

For static view transfer, a starting image from the target viewpoint is required:

1. Warp the first frame to the target viewpoint.
2. Fill invisible regions using FLUX + inpainting ControlNet.
3. **Stochastic control guidance** (core design): at each step, $S=25$ candidates are generated; DUSt3R computes a multi-view consistency score (DINO feature similarity) and the optimal path is selected to continue denoising.

This inference-time compute scaling strategy ensures cross-view consistency.

## Key Experimental Results

### Main Results

| Method | Subject↑ | Temporal↑ | Dynamic↑ | MEt3R↓ | FID↓ | FVD↓ |
|------|----------|-----------|----------|--------|------|------|
| **Static View Transfer** |
| GCD | 0.885 | 0.873 | 0.761 | 0.124 | 155.2 | 5264.7 |
| Vanilla CogVideoX | **0.945** | **0.974** | 0.729 | 0.054 | 79.6 | 3664.2 |
| **Reangle-A-Video** | 0.952 | 0.976 | **0.766** | **0.041** | **53.4** | **2690.9** |
| **Dynamic Camera Control** |
| NVS-Solver | 0.904 | 0.905 | **0.881** | 0.109 | 95.8 | 3516.5 |
| Trajectory Attn | 0.898 | 0.934 | 0.889 | 0.097 | 109.2 | 3624.9 |
| **Reangle-A-Video** | **0.914** | **0.939** | 0.888 | **0.065** | **74.2** | **3019.7** |

The proposed method achieves comprehensive superiority on MEt3R (multi-view consistency), FID, and FVD.

### Ablation Study

**Multi-view inpainting ablation**:

| Configuration | MEt3R↓ | SED↓ | TSED↑ |
|------|--------|------|-------|
| w/o stochastic control guidance | 0.143 | 1.197 | 0.524 |
| **w/ stochastic control guidance** | **0.118** | **1.184** | **0.559** |

Stochastic control guidance significantly improves multi-view consistency.

**Data augmentation ablation**: Fine-tuning on original videos only fails to accurately capture motion (e.g., a rhinoceros moving in front of trees); incorporating warped videos substantially improves motion fidelity. A user study further confirms this finding.

## Highlights & Insights

1. **Video translation paradigm**: Recasting 4D generation as video-to-video translation entirely circumvents the need for expensive 4D training data.
2. **Self-supervised training**: Only a single video is required; multi-view training data is created via depth-based warping.
3. **Masked diffusion loss**: An elegant design that learns motion on visible pixels while preserving the model prior in invisible regions.
4. **Inference-time compute scaling**: DUSt3R serves as a reward function to enforce multi-view consistency during inpainting.
5. Unified support for both **static view transfer** and **dynamic camera control** with full 6DoF.

## Limitations & Future Work

- Fine-tuning requires approximately 1 hour per video (400-step LoRA), precluding real-time use.
- Depth estimation errors propagate into the warped videos.
- Artifacts may appear under fast motion or large viewpoint changes.
- Evaluation is currently limited to 28 videos; large-scale performance remains unknown.

## Related Work & Insights

- Multi-view video generation: CAT4D, Generative Camera Dolly
- Camera-controlled video generation: CameraCtrl, MotionCtrl, Recapture
- Video diffusion models: CogVideoX, Wan, Sora

## Rating

| Dimension | Score (1–5) |
|------|-----------|
| Novelty | 5 |
| Technical Depth | 5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| **Overall** | **4.5** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction](geo4d_leveraging_video_generators_for_geometric_4d_scene_reconstruction.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)
- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](../../ICLR2026/video_generation/geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[AAAI 2026\] 3D4D: An Interactive Editable 4D World Model via 3D Video Generation](../../AAAI2026/video_generation/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)
- [\[ICCV 2025\] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](recammaster_camera-controlled_generative_rendering_from_a_single_video.md)

</div>

<!-- RELATED:END -->
