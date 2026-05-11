---
title: >-
  [Paper Note] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering
description: >-
  [ICCV 2025][Video Generation][3D scene generation] SteerX proposes a zero-shot inference-time guidance method that integrates scene reconstruction into the video generation process. By designing geometric reward function…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "3D scene generation"
  - "4D scene generation"
  - "geometric steering"
  - "inference-time guidance"
date: 2026-05-08
content_hash: cc4ee3ddeac6119f
---

# SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering

**Conference**: ICCV 2025
**arXiv**: [2503.12024](https://arxiv.org/abs/2503.12024)
**Code**: [https://github.com/byeongjun-park/SteerX](https://github.com/byeongjun-park/SteerX)
**Area**: Video Generation
**Keywords**: 3D scene generation, 4D scene generation, geometric steering, inference-time guidance, video generation

## TL;DR

SteerX proposes a zero-shot inference-time guidance method that integrates scene reconstruction into the video generation process. By designing geometric reward functions using camera-free feed-forward reconstruction models, SteerX steers the generation distribution toward geometrically consistent samples, enabling high-quality camera-free 3D/4D scene generation.

## Background & Motivation

**Background**: 3D/4D scene generation is a prominent direction in computer vision. The prevailing approach first employs a video generation model to synthesize multi-view videos, then applies a reconstruction model to recover the 3D/4D scene. These two stages—video generation and scene reconstruction—are typically optimized independently.

**Limitations of Prior Work**: Existing methods improve geometric consistency within each stage separately, but subtle misalignments in one stage are difficult to correct in the other. For instance, a video generation model may introduce minor multi-view inconsistencies that the reconstruction model cannot compensate for; conversely, optimizations in the reconstruction stage cannot feed back into the generation stage. This siloed strategy limits the geometric quality of the final 3D/4D scenes.

**Key Challenge**: Video generation and scene reconstruction are treated as two independent optimization problems, lacking a unified framework to jointly enforce geometric alignment across both stages.

**Goal**: To design a method that directly incorporates geometric constraints from scene reconstruction into the inference process of video generation, so that the generated videos are inherently more geometrically consistent.

**Key Insight**: The authors observe that if the geometric quality of a candidate frame sequence can be assessed in real time during video generation—via fast feed-forward reconstruction—this assessment signal can serve as a reward to guide the sampling process. This idea has precedents in the LLM domain (e.g., RLHF) but has not been sufficiently explored in 3D generation.

**Core Idea**: Geometric reward functions are designed using camera-free feed-forward 3D reconstruction models (e.g., MEt3R). Under the Feynman-Kac guidance framework, the data distribution is tilted at inference time toward samples with better geometric alignment, thereby unifying generation and reconstruction.

## Method

### Overall Architecture

SteerX takes text descriptions or a single image as input and produces geometrically consistent multi-view videos that can be further reconstructed into 3D/4D scenes. The entire pipeline is built upon the denoising process of diffusion models: at each denoising step, SteerX generates multiple candidate samples (particles), evaluates each particle's geometric quality using reward functions, and resamples to retain high-reward particles for subsequent denoising. This is a zero-shot method that requires no additional training and is applicable to arbitrary video generation models.

### Key Designs

1. **GS-MEt3R Geometric Reward (Static 3D Scenes)**:

    - **Function**: Evaluates 3D geometric consistency across generated video frames.
    - **Mechanism**: Intermediate generated video frames are fed into MEt3R (a camera-free feed-forward 3D Gaussian reconstruction model) to obtain a 3D Gaussian representation, which is then rendered back to each viewpoint. The DINO feature similarity between the original generated frames and the rendered frames is computed as the reward. Specifically, DINO feature maps are extracted from both sets of frames, cosine similarity is computed between corresponding feature maps, and the upsampled average similarity serves as the final reward score.
    - **Design Motivation**: If the generated multi-view video is geometrically consistent, images rendered from the reconstructed 3D Gaussians should closely resemble the original frames. DINO features are sensitive to semantic information and more robust than pixel-level comparisons.

2. **Dyn-MEt3R Geometric Reward (Dynamic 4D Scenes)**:

    - **Function**: Evaluates geometric consistency of background regions in dynamic scenes.
    - **Mechanism**: The video frames are divided into two halves. MEt3R reconstructs a 3D point cloud from the first half; DINO features of the background region are extracted and back-projected into 3D space. These 3D features are then projected onto the second half of the frames, and the similarity between the projected features and the original features of the second half is computed as the reward.
    - **Design Motivation**: In dynamic scenes, foreground objects are in motion, making global consistency checks inappropriate. However, the background should remain consistent across frames; thus, cross-frame consistency of background features serves as a measure of geometric quality.

3. **Feynman-Kac Geometric Guidance**:

    - **Function**: Leverages reward signals to guide sampling during diffusion model inference.
    - **Mechanism**: Under the Feynman-Kac particle guidance framework, multiple particles (candidate samples) are maintained throughout the denoising process. Geometric rewards are computed at regular intervals, and particles are resampled with weights proportional to their rewards—high-reward particles are duplicated while low-reward ones are discarded. Through repeated resampling, the data distribution progressively shifts toward geometrically consistent outputs.
    - **Design Motivation**: This approach requires no modification of the generative model's weights and operates entirely at inference time, making it zero-shot and plug-and-play. Compared to gradient-based guidance, particle guidance avoids the high computational cost and instability associated with gradient computation.

### Loss & Training

SteerX is an inference-time method and involves no additional training. The core "loss" is the geometric reward function: $R = \frac{1}{N}\sum_{i=1}^{N} \text{cos\_sim}(\phi(I_i), \phi(\hat{I}_i))$, where $\phi$ denotes the DINO feature extractor, and $I_i$ and $\hat{I}_i$ are the original and rendered frames, respectively.

## Key Experimental Results

### Main Results

SteerX achieves significant improvements in both 3D and 4D scene generation and supports multiple video generation backbones.

| Method | CLIP-Score ↑ | 3D Consistency ↑ | User Preference ↑ | Backbone |
|--------|-------------|-----------------|------------------|---------|
| Wan2.1 (baseline) | 0.287 | 0.71 | 23% | Wan2.1 |
| SteerX + Wan2.1 | **0.301** | **0.82** | **77%** | Wan2.1 |
| CogVideoX (baseline) | 0.275 | 0.68 | 19% | CogVideoX |
| SteerX + CogVideoX | **0.293** | **0.79** | **81%** | CogVideoX |

### Ablation Study

| Configuration | Geometric Consistency ↑ | Visual Quality ↑ | Note |
|--------------|------------------------|-----------------|------|
| Full SteerX | 0.82 | 0.94 | Complete model (GS-MEt3R + FK guidance) |
| w/o geometric reward | 0.71 | 0.93 | Degenerates to vanilla generation |
| w/o resampling | 0.74 | 0.92 | Reward computed but no particle resampling |
| Pixel-level reward (MSE) vs. DINO | 0.76 | 0.88 | MSE replaces DINO feature similarity |
| \# particles = 2 | 0.77 | 0.93 | Reduced particle count |
| \# particles = 8 | 0.83 | 0.94 | Increased particles with diminishing returns |

### Key Findings

- The geometric reward function is the core contribution; removing it causes a substantial drop in geometric consistency (0.82→0.71), confirming that vanilla video generation models suffer from significant geometric inconsistency.
- DINO features are more suitable as reward signals than pixel-level MSE, as DINO is more robust to minor appearance variations while being more sensitive to geometric errors.
- A particle count of 4 offers the best cost-effectiveness; further increases yield marginal improvements at linearly growing computational cost.
- SteerX is effective across different video generation backbones (Wan2.1, CogVideoX, etc.), demonstrating the generality of the framework.

## Highlights & Insights

- **Inference-Time Guidance Paradigm**: Incorporating geometric rewards into the sampling process of diffusion models is an elegant design. Without retraining any model, arbitrary video generators can produce geometrically consistent outputs. This plug-and-play philosophy is transferable to other generation tasks that require satisfying specific constraints.
- **Feed-Forward Reconstruction as an Evaluator**: Camera-free feed-forward reconstruction models such as MEt3R are fast and well-suited as core components of reward functions. This "evaluate generation quality via reconstruction quality" paradigm avoids the need for ground-truth data.
- **Unified Framework for 3D and 4D**: By designing two distinct reward functions—GS-MEt3R for static scenes and Dyn-MEt3R for dynamic scenes—SteerX achieves unified handling of both 3D and 4D scene generation.

## Limitations & Future Work

- Inference cost scales linearly with the number of particles; using 4 particles roughly quadruples generation time.
- The geometric reward relies on the reconstruction quality of MEt3R; if the reconstruction model fails on certain scene types, the reward signal becomes unreliable.
- Validation is currently limited to relatively short videos (tens of frames); the effectiveness of particle guidance on long videos remains unclear.
- The background consistency reward (Dyn-MEt3R) assumes a static background and may not be applicable to scenes with dynamic backgrounds.

## Related Work & Insights

- **vs. DreamFusion / Score Jacobian Chaining**: These methods distill 3D content from 2D diffusion models via SDS loss but require per-scene optimization. SteerX is zero-shot and considerably faster.
- **vs. ViewCrafter**: ViewCrafter fine-tunes video models to incorporate 3D understanding, at the cost of generality. SteerX does not modify model weights, preserving generality.
- **vs. Feynman-Kac Steering (SVDD)**: SteerX's guidance framework builds upon this prior work; the key contribution lies in designing geometric reward functions tailored to 3D/4D scene generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing inference-time guidance into 3D/4D generation is novel, though the Feynman-Kac framework itself has prior precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3D and 4D scenes, multiple backbone networks, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and motivation is well-articulated.
- **Value**: ⭐⭐⭐⭐ The plug-and-play framework design is highly practical, though the multiplicative inference cost increase limits real-world applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction](geo4d_leveraging_video_generators_for_geometric_4d_scene_reconstruction.md)
- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [\[ICCV 2025\] DACoN: DINO for Anime Paint Bucket Colorization with Any Number of Reference Images](dacon_dino_for_anime_paint_bucket_colorization_with_any_number_of_reference_imag.md)
- [\[ICCV 2025\] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](recammaster_camera-controlled_generative_rendering_from_a_single_video.md)
- [\[AAAI 2026\] 3D4D: An Interactive Editable 4D World Model via 3D Video Generation](../../AAAI2026/video_generation/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)

</div>

<!-- RELATED:END -->
