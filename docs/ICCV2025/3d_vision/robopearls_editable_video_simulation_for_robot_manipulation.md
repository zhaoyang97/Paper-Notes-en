---
title: >-
  [Paper Note] RoboPearls: Editable Video Simulation for Robot Manipulation
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper presents RoboPearls, an editable video simulation framework built upon 3D Gaussian Splatting (3DGS) that constructs photorealistic simulation environments from demonstration videos. It supports rich scene editing operations via Incremental Semantic Distillation (ISD) and a 3D-regularized NNFM loss, and employs a multi-LLM agent system to automate the simulation generation pipeline, forming a VLM-in-the-loop robot learning augmentation system.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Editable Simulation
  - Robot Manipulation
  - LLM Agent
  - Sim-to-Real
date: 2026-05-08
content_hash: 8a46b20dbba0f207
---

# RoboPearls: Editable Video Simulation for Robot Manipulation

**Conference**: ICCV 2025
**arXiv**: [2506.22756](https://arxiv.org/abs/2506.22756)
**Code**: [Project Page](https://robopearls.github.io/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Editable Simulation, Robot Manipulation, LLM Agent, Sim-to-Real

## TL;DR

This paper presents RoboPearls, an editable video simulation framework built upon 3D Gaussian Splatting (3DGS) that constructs photorealistic simulation environments from demonstration videos. It supports rich scene editing operations via Incremental Semantic Distillation (ISD) and a 3D-regularized NNFM loss, and employs a multi-LLM agent system to automate the simulation generation pipeline, forming a VLM-in-the-loop robot learning augmentation system.

## Background & Motivation

The development of generalist robot manipulation policies relies on large-scale demonstration data, yet faces two core bottlenecks:

**Difficulty of real-world data collection**: Large-scale real demonstrations executed by human experts are both costly and inefficient, making large-scale expansion impractical.

**Sim-to-real gap**: Although physics-based simulation platforms (Isaac Sim, PyBullet, etc.) provide controlled environments, the sim-to-real gap remains a significant obstacle. Modifying scenes in traditional simulators (e.g., changing the color of a cup) requires reprogramming the simulation environment, which is highly inefficient.

The emergence of 3DGS opens new opportunities—its explicit representation, high-quality reconstruction, and real-time rendering capabilities make it feasible to construct photorealistic simulations from demonstration videos. However, **prior work has explored 3DGS reconstruction and editing capabilities in isolation, without forming a systematic pipeline for robotics**. The authors draw an analogy to a pearl necklace: previous works found individual "pearls" (standalone operators), but had not yet strung them into a complete "necklace" (a systematic solution).

## Method

### Overall Architecture

RoboPearls consists of four functional layers: (1) dynamic semantic-enhanced Gaussian reconstruction → (2) diverse editable simulation operators → (3) LLM agent-automated simulation generation → (4) VLM-in-the-loop feedback for robot learning enhancement.

### Key Designs

1. **Dynamic Semantic-enhanced Gaussians**:

    - **Dynamic reconstruction**: Extends 3DGS to 4D by expanding the position from $(μ_x, μ_y, μ_z)$ to $(μ_x, μ_y, μ_z, μ_t)$ and the covariance matrix to a 4D ellipsoid, enabling each dynamic frame to be viewed as a 3D snapshot at a specific timestamp $t$:
    $C = \sum_{i=1}^{N} p_i(t) p_i(x'|t) \alpha_i c_i \prod_{j=1}^{i-1}(1 - p_j(t) p_j(x'|t) \alpha_j)$
    - **Semantic Gaussians**: Attaches a low-dimensional learnable identity code $e_i$ to each Gaussian primitive, supervised by 2D masks from SAM; 2D identity features $E$ are rendered via $\alpha$-blending and classified with softmax.
    - **Optimization objective**: $L = \lambda_{2d} L_{2d} + \lambda_{sem} L_{sem} + \lambda_{3d} L_{3d}$, where $L_{3d}$ applies KL divergence to enforce identity code consistency among K-nearest-neighbor Gaussians, mitigating occlusion issues.

2. **Incremental Semantic Distillation (ISD)**: Addresses the core challenge of fine-grained object retrieval. SAM's 2D masks may not cover all granularities (e.g., small buttons on a stove). The ISD pipeline:

    - Localizes the target object using G-DINO to obtain a mask ID.
    - Renders a 2D object mask and verifies whether the correct target is retrieved (e.g., whether the entire stove is retrieved instead of the button).
    - If the target is not correctly identified, prompts SAM with the bounding box for finer-grained segmentation.
    - **Only fine-tunes** the identity codes $e$ of the relevant Gaussian primitives, maintaining overall efficiency.

3. **3D-Regularized NNFM Loss (3D-NNFM Loss)**: Addresses multi-view consistency in object texture/style editing. The original NNFM loss transfers high-frequency visual details but is limited to holistic scene stylization. The improvements of 3D-NNFM:

    - Optimizes only the spherical harmonics (SH) parameters of the target object's Gaussians, preserving the background.
    - Regularizes with the original reconstruction loss to prevent SH optimization from producing artifacts at object boundaries:
    $L_{\text{3D-NNFM}} = L_{\text{NNFM}}^{M_{3d}} + L_{\text{gs}}^{\overline{M_{3d}}}$

4. **Multi-LLM Agent Collaboration System**: Six specialized agents connected in series form an automated pipeline:

    - **Simulation Manager Agent**: Team leader; decomposes user commands into specific instructions.
    - **Grounding Agent**: Handles object localization requests (supports complex spatial relationship queries).
    - **Scene Operation Agent**: Executes editing operations (color/texture/removal/insertion, etc.).
    - **3D Asset Management Agent**: Manages and generates 3D assets (ShapeSplat/uCO3D databases + GRM/LGM generation).
    - **Scene Refiner Agent**: Performs global quality refinement.
    - **Scene Renderer Agent**: Controls viewpoints and rendering.
    - **VLM-in-the-loop**: Uses a VLM to analyze robot learning failure cases, automatically generates simulation requirement instructions, and drives the simulation loop.

### Loss & Training

Scene reconstruction: MSE rendering loss $L_{2d}$ + semantic cross-entropy $L_{sem}$ + KL divergence 3D consistency loss $L_{3d}$

Texture editing: 3D-NNFM loss = target region NNFM + background reconstruction loss

Color modification uses the CIELAB color space to alter hue while preserving original luminance variation.

After object insertion, the libcom image compositing library is applied for color harmonization, followed by fine-tuning the SH coefficients of the inserted object's Gaussians.

## Key Experimental Results

### Main Results

| Benchmark | Method | Avg Success↑ | Key Task Gain |
|-----------|--------|-------------|---------------|
| Colosseum | RVT | 51.7 | - |
| Colosseum | RVT-2 | 64.6 | - |
| Colosseum | **RoboPearls-RVT** | **69.2** | +17.5 vs RVT |
| Colosseum | **RoboPearls-RVT2** | **75.4** | +10.8 vs RVT-2 |
| RLBench | RVT | 62.4 | Stack Cups: 17.6 |
| RLBench | SAM2Act | 83.8 | Stack Cups: 63.2 |
| RLBench | **RoboPearls-SAM2Act** | **88.5** | **Stack Cups: 68.0 (+4.8)** |
| RLBench | **RoboPearls-RVT2** | **78.0** | **Put in Cupboard: 75.5 (+23.0)** |

Real robot experiments (Kinova Gen3, 20 trials per task):

| Task | RDT (Seen/Unseen) | RoboPearls (Seen/Unseen) |
|------|-------------------|--------------------------|
| Pick up | 10/4 | **15/14** |
| Put on | 7/0 | **10/9** |
| Place in | 8/1 | **12/12** |

### Ablation Study

| Method | Stack Cups | Put in Cupboard | Insert Peg |
|--------|-----------|----------------|------------|
| RVT (baseline) | 14.5 | 40.4 | 11.0 |
| + IP2P (2D editing) | 18.4 | 44.8 | 10.7 |
| RoboPearls (w/o VLM) | 24.7 | 45.0 | 16.5 |
| **RoboPearls (full)** | **37.7** | **55.5** | **17.1** |

### Key Findings

- **3D video simulation substantially outperforms 2D image editing**: IP2P yields only marginal improvement, whereas RoboPearls' 3D-consistent editing delivers significant gains.
- **VLM-in-the-loop feedback is highly effective**: Stack Cups improves from 24.7 to 37.7 (+52.6% recovery gain), demonstrating the value of automated failure analysis and targeted data augmentation.
- RoboPearls exhibits strong generalization to unseen objects in real-robot settings (Place in: 1/20 → 12/20).
- All 13 perturbation types on Colosseum (color/texture/size/lighting/background/camera/distractors) show consistent improvement, indicating strong overall robustness.

## Highlights & Insights

- **Systematic design**: Integrates disparate 3DGS editing capabilities into a complete pipeline oriented toward robot learning, with high practical value.
- **Elegant ISD design**: Incremental semantic distillation updates only the identity codes of relevant Gaussians, maintaining efficiency while supporting arbitrary-granularity retrieval.
- **VLM-in-the-loop as a key innovation**: Beyond generating simulation data, it automatically analyzes failure causes and generates targeted simulation requirements.
- **Multi-LLM agent architecture**: Six specialized agents collaborate in a division-of-labor fashion, enabling complex simulation workflows to be driven by natural language.

## Limitations & Future Work

- The physical realism of simulated videos remains limited and does not fully replace the precise interaction simulation of physics engines.
- 4DGS reconstruction quality may degrade in highly dynamic scenes.
- LLM agent command parsing carries a risk of error accumulation.
- Current physics simulation (MPM) requires manual configuration of physical parameters, limiting automation.
- No comparison with other data augmentation approaches such as GAN- or diffusion-based methods.

## Related Work & Insights

- Unlike GaussianGrasper, GraspSplats, and related works, RoboPearls is a system-level solution targeting robot simulation.
- The concept of the 3D-NNFM loss is generalizable to other 3DGS editing tasks requiring local editing with global consistency.
- The VLM-in-the-loop feedback paradigm can be extended to other data-driven robot learning scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic engineering innovation with well-designed ISD and 3D-NNFM components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensively validated on RLBench, Colosseum, real robots, Ego4D, and Open X.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with an apt "pearl necklace" analogy; some technical details could be elaborated further.
- Value: ⭐⭐⭐⭐ Provides a practical photorealistic data augmentation solution for robot learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 4D Visual Pre-training for Robot Learning](4d_visual_pre-training_for_robot_learning.md)
- [\[ICLR 2026\] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](../../ICLR2026/3d_vision/generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)

</div>

<!-- RELATED:END -->
