---
title: >-
  [Paper Note] SceneCrafter: Controllable Multi-View Driving Scene Editing
description: >-
  [CVPR 2025][Autonomous Driving][Multi-View Consistency] SceneCrafter proposes a driving scene editing framework based on multi-view diffusion models. Through a teacher-student two-stage training paradigm, it generates high-quality synthetic paired data, supporting global editing of weather/time and local editing of foreground object addition/deletion while maintaining 3D geometric consistency across cameras.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Multi-View Consistency"
  - "Driving Scene Editing"
  - "Diffusion Models"
  - "Synthetic Data Generation"
  - "Sensor Simulation"
date: 2026-05-08
content_hash: a433b316b25ce732
---

# SceneCrafter: Controllable Multi-View Driving Scene Editing

**Conference**: CVPR 2025  
**arXiv**: [2506.19488](https://arxiv.org/abs/2506.19488)  
**Code**: None  
**Area**: Autonomous Driving / Scene Editing  
**Keywords**: Multi-View Consistency, Driving Scene Editing, Diffusion Models, Synthetic Data Generation, Sensor Simulation

## TL;DR

SceneCrafter proposes a driving scene editing framework based on multi-view diffusion models. Through a teacher-student two-stage training paradigm, it generates high-quality synthetic paired data, supporting global editing of weather/time and local editing of foreground object addition/deletion while maintaining 3D geometric consistency across cameras.

## Background & Motivation

**Background**: Autonomous driving simulation requires realistic sensor data to evaluate full-stack systems. Existing methods are divided into reconstruction-based (neural fields/primitives) and generation-based (image/video generation models). Reconstruction methods are faithful to the real scene but lack flexible editing capabilities, while pure generation methods lack correlation with the real scene.

**Limitations of Prior Work**: Image editing in driving simulation faces three unique challenges: (1) 3D consistency must be maintained across multiple cameras; (2) training data almost exclusively consists of "busy streets with cars," making it difficult for models to learn the prior of "empty streets"; (3) acquiring paired training data for global/local editing is extremely difficult.

**Key Challenge**: Editing models require paired data for supervised training, but real paired data (e.g., image pairs of the same scene under different weather conditions or with different vehicle layouts) is extremely hard to obtain. Directly applying existing methods like Prompt-to-Prompt or RePaint yields poor results in multi-view driving scenes.

**Goal**: Construct a unified multi-view driving scene editor that simultaneously supports global editing (weather/time) and local editing (object addition/deletion) while maintaining geometric consistency across views.

**Key Insight**: The authors adopt a teacher-student architecture—first training a teacher model to generate high-quality synthetic paired data, and then training a unified student editing model using this generated data. This indirect approach bypasses the difficulty of directly acquiring paired data.

**Core Idea**: Generate paired data for global editing via an improved Prompt-to-Prompt (replacing self-attention instead of cross-attention), generate paired data for local editing via masked training + multi-view repaint + alpha blending, and finally distill them into a unified scene editing model.

## Method

### Overall Architecture

The pipeline of SceneCrafter is divided into two main stages: the first stage trains two teacher models to generate synthetic paired data for global and local editing, respectively; the second stage trains a unified student editing model using the 1 million generated synthetic pairs. The inputs are multi-view source images (8 cameras) and editing conditions, and the outputs are edited multi-view images.

### Key Designs

1. **Multimodal Conditional Control of the Teacher Model**:

    - **Function**: Supports scene generation under various conditions, such as weather, time, HD maps, and object bounding boxes.
    - **Mechanism**: Global conditions (weather encoded using CLIP text encoder, time encoded using sun angle position encoding) and local conditions (HD maps reduced to 512 tokens using PerceiverIO, object bounding boxes encoded using MLP) are injected into the U-Net through cross-attention in a unified manner. The foreground mask and raymap are concatenated along the channel dimension of the input. During training, each condition is randomly dropped out with a 10% probability to improve robustness.
    - **Design Motivation**: Rich conditional signals anchor the scene geometry within the generation process, allowing fine geometric details to be preserved when modifying high-level attributes (such as weather).

2. **Improved Prompt-to-Prompt for Global Editing Paired Data Generation**:

    - **Function**: Generates geometrically consistent paired data under varying weather and time.
    - **Mechanism**: Unlike the original P2P which freezes cross-attention weights, SceneCrafter freezes the weights of all self-attention layers because global editing should affect all regions of the image while preserving pixel-level layout. In addition, more conditional signals (object bounding boxes, HD maps) are introduced to enhance geometric consistency, and only daytime is used as the source image time to achieve optimal generation quality. The generated source-target pairs are randomly flipped in order to serve as training data.
    - **Design Motivation**: There are no text tokens to manipulate in multi-view driving scenes; thus, replacing self-attention weights can better maintain geometric consistency at the pixel level.

3. **Masked Training + Multi-View Repaint + Alpha Blending for Local Editing Paired Data Generation**:

    - **Function**: Generates multi-view consistent paired data for object addition and deletion.
    - **Mechanism**: Masked training sets the noise of the foreground region to zero during training, denoises only the background, and computes the loss solely on background pixels: $\mathbf{z}_t = (1-\mathbf{m}) \odot (\alpha_t \mathbf{z}_0 + \sigma_t \epsilon) + \mathbf{m} \odot \mathbf{z}_0$. This enables the model to learn the "empty street" prior. Multi-view repaint processes the foreground regions of all views simultaneously in each reverse step, ensuring multi-view consistency. Alpha blending mixes the "empty street" and "busy street" according to sampled masks to generate paired data with arbitrary numbers of objects.
    - **Design Motivation**: Directly erasing objects using RePaint yields poor results because the model is trained on "busy street" data and lacks an empty street prior. Masked training cleverly learns the vehicle-free prior from vehicle-containing data in a self-supervised manner.

### Loss & Training

The student model directly concatenates the latent of the source image into the denoising latent (instead of using cross-attention); experiments demonstrate that the concatenation method is more effective for pixel-level conditions (reducing FID by 13.1). The model's weights are initialized from the global editing teacher, and it is trained for 100K iterations on 128 TPU v5 chips with a learning rate of $1e^{-5}$ and a batch size of 128. Inference uses 50-step denoising with classifier-free guidance.

## Key Experimental Results

### Main Results

| Method | Time Editing FID↓ | Time Editing CLIP↑ | Time Editing User Pref.↑ | Weather Editing FID↓ | Weather Editing CLIP↑ | Weather Editing User Pref.↑ |
|------|-------------|--------------|----------------|-------------|--------------|----------------|
| SDEdit | 60.4 | 0.204 | 2.7% | 78.3 | 0.203 | 1.8% |
| P2P* | 46.8 | 0.223 | 13.6% | 55.4 | 0.207 | 12.7% |
| **SceneCrafter** | **37.2** | **0.220** | **83.6%** | **38.9** | **0.221** | **85.5%** |

Local editing (FID↓): 2D-RePaint removal/insertion 30.6/31.9, MV-RePaint 26.0/28.5, SceneCrafter **23.5/21.7**.

### Ablation Study

| Replace Self-Attn | Add Conditions | Daytime Source | FID↓ | CLIP↑ |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 57.1 | 0.204 |
| ✓ | ✗ | ✗ | 41.5 | 0.202 |
| ✓ | ✓ | ✗ | 39.9 | 0.214 |
| ✓ | ✓ | ✓ | **36.2** | **0.223** |

### Key Findings

- Replacing self-attention weights preserves pixel-level geometric consistency better than cross-attention.
- Introducing more conditional signals (object bounding boxes + HD maps) significantly improves controllability.
- Using only daytime as the source image significantly enhances generation quality.
- Injecting source images via concatenation is superior to the cross-attention method (FID 50.3 vs 37.2).
- The 3D LPIPS of the complete model reaches 0.187, which is comparable to the 0.186 of real data.

## Highlights & Insights

- **Masked training** is highly ingenious: under circumstances where almost all training data contains foreground objects, it allows the model to learn the empty street prior self-supervisedly via a differentiated noise strategy.
- The discovery of replacing self-attention instead of cross-attention is inspiring, as the conditions in driving scenes are not text tokens.
- Using box conditions instead of mask conditions for object editing performs better on small objects, avoiding boundary issues caused by imprecise segmentation.
- The teacher-student training paradigm effectively addresses the difficulty of obtaining paired data.

## Limitations & Future Work

- It relies on private Waymo data (around 14 million driving video segments), making it difficult to reproduce.
- The editing capability is limited by the generation quality of the teacher model.
- It only handles static scene editing and has not been extended to temporal video consistency.
- Although using box conditions is beneficial, it fails to achieve precise control over the texture and appearance details of objects.
- Extending to a broader range of editing types (e.g., changes in road structures) can be explored.

## Related Work & Insights

- The multi-view extension strategy of **Prompt-to-Prompt** is worth referencing: replacing self-attention to achieve geometric preservation.
- The limitations of **RePaint** in multi-view scenarios inspired the masked training approach.
- The teacher-student synthetic data paradigm can be generalized to other editing tasks lacking paired data.

## Rating

- **Novelty**: 8/10 — Masked training and self-attention replacement strategies are novel.
- **Experimental Thoroughness**: 8/10 — Comprehensive ablations, introduces a new 3D consistency metric, and includes a user study.
- **Writing Quality**: 8/10 — Well-structured with a natural motivation development.
- **Value**: 8/10 — Possesses direct engineering value for autonomous driving simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] IntrinsicWeather: Controllable Weather Editing in Intrinsic Space](../../CVPR2026/autonomous_driving/intrinsicweather_controllable_weather_editing_in_intrinsic_space.md)
- [\[CVPR 2026\] HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles](../../CVPR2026/autonomous_driving/horizonforge_driving_scene_editing_with_any_trajectories_and_any_vehicles.md)
- [\[CVPR 2025\] MITracker: Multi-View Integration for Visual Object Tracking](mitracker_multi-view_integration_for_visual_object_tracking.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](../../ICCV2025/autonomous_driving/controllable_3d_outdoor_scene_generation_via_scene_graphs.md)
- [\[CVPR 2025\] UniScene: Unified Occupancy-centric Driving Scene Generation](uniscene_unified_occupancy-centric_driving_scene_generation.md)

</div>

<!-- RELATED:END -->
