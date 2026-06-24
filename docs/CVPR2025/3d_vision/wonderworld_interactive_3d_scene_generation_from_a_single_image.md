---
title: >-
  [Paper Note] WonderWorld: Interactive 3D Scene Generation from a Single Image
description: >-
  [CVPR 2025][3D Vision][Interactive 3D Scene Generation] WonderWorld is proposed as the first framework to support interactive 3D scene generation, allowing users to control scene content and layout in real time via camera movement and text prompts. Each scene is generated in less than 10 seconds on a single A6000 GPU, which is ~80x faster than existing methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Interactive 3D Scene Generation"
  - "Gaussian Splatting"
  - "Layered Representation"
  - "Guided Depth Diffusion"
  - "Real-time Generation"
date: 2026-05-08
content_hash: 23d894c42c7b59f7
---

# WonderWorld: Interactive 3D Scene Generation from a Single Image

**Conference**: CVPR 2025  
**arXiv**: [2406.09394](https://arxiv.org/abs/2406.09394)  
**Code**: [Project Page](https://kovenyu.com/WonderWorld/)  
**Area**: 3D Vision / 3D Scene Generation  
**Keywords**: Interactive 3D Scene Generation, Gaussian Splatting, Layered Representation, Guided Depth Diffusion, Real-time Generation

## TL;DR

WonderWorld is proposed as the first framework to support interactive 3D scene generation, allowing users to control scene content and layout in real time via camera movement and text prompts. Each scene is generated in less than 10 seconds on a single A6000 GPU, which is ~80x faster than existing methods.

## Background & Motivation

3D scene generation has flourished in recent years, but existing methods (such as WonderJourney, LucidDreamer, and Text2Room) operate in an **offline** mode—once a user provides an image or text, the system takes tens of minutes to hours to build a fixed scene. This offline mode cannot meet the demands of applications requiring **interactive control**, such as game world prototyping and real-time VR exploration.

Existing methods are slow due to two bottlenecks: (1) they require **step-by-step generation of dense multi-view images and aligned depth maps** to fill occluded areas, and (2) they require **long optimization times** for 3D scene representations (NeRF, 3DGS, etc.). Furthermore, when connecting multiple scenes, inconsistent depth estimation leads to **geometric distortions and seams**.

WonderWorld achieves interactive generation through two core technical breakthroughs: (1) **FLAGS (Fast LAyered Gaussian Surfels)**—a rapid layered Gaussian surfel representation that eliminates the need for dense multi-view generation, whose geometry-based initialization shortens optimization time from minutes to seconds; (2) **Guided Depth Diffusion**—which leverages the partially visible depth of existing scenes to guide the depth estimation of new scenes, reducing geometric distortions.

## Method

### Overall Architecture

Given an initial image, WonderWorld enters an interactive control loop: the user moves the camera to select a generation location and inputs text prompts to specify content, after which the system generates a new scene and connects it to the existing world in <10 seconds. Each iteration consists of: (1) using an LLM to generate a structured scene description (foreground/background/style), (2) generating/outpainting scene images with a diffusion model, (3) generating the FLAGS three-layer representation (foreground/background/sky), and (4) aligning geometry via guided depth diffusion.

### Key Designs

1. **FLAGS (Fast LAyered Gaussian Surfels) Representation**:
    - Function: Fast 3D scene layered representation supporting real-time rendering and rapid generation.
    - Mechanism: Each scene $\mathcal{E} = \{\mathcal{L}_{fg}, \mathcal{L}_{bg}, \mathcal{L}_{sky}\}$ consists of three radiance field layers. Each layer contains a set of surfels parameterized by position $\mathbf{p}$, orientation quaternion $\mathbf{q}$, scale $\mathbf{s} = [s_x, s_y]$ (with the z-axis set to a minimal value $\epsilon$), opacity $o$, and RGB color $\mathbf{c}$. Foreground, background, and sky layers are separated using depth edges and object segmentation, and occluded foreground areas are filled using diffusion inpainting. The key innovation is **geometry-based initialization**—utilizing estimated depth and normals to directly initialize the position, orientation, and scale of the surfels ($s_x = d/(kf_x\cos\theta_x)$). This turns optimization into "fine-tuning" rather than training from scratch, requiring only 100 iterations per layer (<1 second).
    - Design Motivation: Optimizing 3DGS from scratch requires massive amounts of time and dense views. The surfel representation possesses an explicit concept of normals, making geometry-based initialization natural and effective.

2. **Guided Depth Diffusion**:
    - Function: Ensures that the depth of the new scene is geometrically consistent with the existing scene during outpainting.
    - Mechanism: A guidance term is injected into the denoising process of a standard depth diffusion model. The noise prediction is modified to $\hat{\boldsymbol{\epsilon}}_t = \text{UNet}(\mathbf{d}_t, \mathbf{I}_{scene}, t) - s_t \mathbf{g}_t$, where the guidance gradient is $\mathbf{g}_t = \nabla_{\mathbf{d}_t}\|\mathbf{D}_{t-1} \odot \mathbf{M}_{guide} - \mathbf{D}_{guide} \odot \mathbf{M}_{guide}\|^2$. This encourages the generated depth to be consistent with the pre-existing depth in the visible overlapping areas. This is a training-free approach that can be directly applied to pre-trained Marigold depth models.
    - Design Motivation: Performing global shift/scale alignment after independent depth estimation is insufficient because depth estimation itself is inherently ambiguous. Guided diffusion constraints the process probabilistically, which is a more principled approach.

3. **Single-View Layer Generation**:
    - Function: Generates three-layer representations from a single scene image without requiring dense multi-views.
    - Mechanism: A depth gradient threshold is used to detect salient depth edges $\mathbf{E}$, and OneFormer is applied to segment object masks $\{\mathbf{O}_k\}$. Objects overlapping with depth edges are assigned to the foreground mask $\mathbf{M}_{fg}$. The sky is detected via a segmentation network. The background region occluded by the foreground is filled using diffusion inpainting (conditioned on the background prompt), and the sky layer is fully covered and inpainted in non-sky areas. Surfels are generated independently for each layer.
    - Design Motivation: This avoids the time-consuming step of progressive multi-view generation, which is a key design for achieving sub-10 second generation times.

### Loss & Training

- **Photometric Loss**: $L = 0.8L_1 + 0.2L_{\text{D-SSIM}}$, with masking.
- **Back-to-Front Optimization**: Sky layer first $\to$ freeze and optimize background layer $\to$ freeze and optimize foreground layer.
- **Optimization Parameters**: Only opacity, orientation, and scale are optimized, while color and position are fixed.
- **Optimization Iterations**: Only 100 Adam iterations per layer, without densification.
- **LLM Assistance**: Large language models are used to generate a structured description for each scene (foreground objects/background/style).

## Key Experimental Results

### Main Results

Scene generation speed comparison (single A6000 GPU):

| Method | Generation Time / Scene | Scene Representation |
|------|-------------|---------|
| WonderJourney | 749.5 s | Point Cloud |
| LucidDreamer | 798.1 s | 3DGS |
| Text2Room | 766.9 s | Mesh |
| **WonderWorld** | **9.5 s** | FLAGS |

New view rendering quality:

| Method | CLIP Score↑ | CLIP Consistency↑ | CLIP-IQA+↑ | Q-Align↑ |
|------|------------|------------|-----------|---------|
| WonderJourney | 27.34 | 0.9544 | 0.6443 | 2.717 |
| LucidDreamer | 26.72 | 0.8972 | 0.5260 | 2.736 |
| Text2Room | 24.50 | 0.9035 | 0.5620 | 2.650 |
| **WonderWorld** | **29.47** | **0.9948** | **0.6512** | **3.641** |

### Ablation Study

| Configuration | Description |
|------|------|
| W/o Guided Depth Diffusion | Severe seams and geometric distortions appear at scene junctions |
| W/o Layered Representation | Foreground/background cannot be processed independently, leaving holes in occluded regions |
| W/o Geometry-Based Initialization | Optimization time increases significantly and still requires intensive iterations |

### Key Findings

- In human 2AFC preference tests, WonderWorld dominates all baselines with a preference rate $>98\%$.
- Speed is accelerated by approximately 80x (9.5 seconds vs. ~750 seconds) with superior quality.
- CLIP Consistency (CC) reaches 0.9948, indicating extremely high semantic consistency across multiple views.
- It supports mixing different styles within the same world (e.g., Minecraft, painting, Lego).

## Highlights & Insights

- **Paradigm shift from "offline generation" to "interactive generation"**: Demonstrates for the first time that 3D scenes can be generated interactively with second-level latency, unlocking a new application paradigm for 3D scene generation (such as game world prototyping, instant VR exploration, etc.).
- **Exquisite geometry-based initialization**: Derives the surfel scale initialization formula $s_x = d/(kf_x\cos\theta_x)$ using the Nyquist sampling theorem. This allows initialized surfels to seamlessly cover the visible surfaces, converting optimization from "training" into "fine-tuning".
- **Generality of guided depth diffusion**: Can be applied to any pre-trained depth diffusion model without training, achieving partial depth conditioning via gradient guidance.

## Limitations & Future Work

- Scene quality is limited by the image generation/inpainting capabilities of the diffusion model.
- Single-view depth estimation still has intrinsic limitations; fine-grained geometries can be inaccurate.
- Currently only supports static scenes; future work could extend this to dynamic objects and interactions.
- Diffusion inference dominates the execution time (~8 seconds), which would benefit from future acceleration techniques.

## Related Work & Insights

- **vs. WonderJourney**: WonderJourney also generates connected diverse scenes but requires dense multi-view synthesis, causing each scene to take ~750 seconds; WonderWorld reduces this to ~10 seconds via single-view layered generation.
- **vs. LucidDreamer**: LucidDreamer generates a fixed single-scene 3DGS with severe boundary distortion and does not support interactive outpainting.
- **vs. Text2Room**: Text2Room's depth inpainting model is trained solely on indoor data and does not generalize well to outdoor scenes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to achieve interactive 3D scene generation, with elegant designs for FLAGS representation and guided depth diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Uses multiple evaluation metrics and human preference testing, though evaluated on a limited number of test scenes (28 scenes).
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed technical descriptions, and excellent figures.
- Value: ⭐⭐⭐⭐⭐ 80x speedup with better quality, carrying immense potential for games, VR, and creative design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PhysGen3D: Crafting a Miniature Interactive World from a Single Image](physgen3d_crafting_a_miniature_interactive_world_from_a_single_image.md)
- [\[CVPR 2025\] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)
- [\[CVPR 2025\] Disco4D: Disentangled 4D Human Generation and Animation from a Single Image](disco4d_disentangled_4d_human_generation_and_animation_from_a_single_image.md)
- [\[CVPR 2025\] Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation](symmetry_strikes_back_from_single-image_symmetry_detection_to_3d_generation.md)
- [\[ICCV 2025\] WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions](../../ICCV2025/3d_vision/wonderplay_dynamic_3d_scene_generation_from_a_single_image_and_actions.md)

</div>

<!-- RELATED:END -->
