---
title: >-
  [Paper Note] Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images
description: >-
  [ICLR 2026][3D Vision][Unified Understanding and Generation] This paper presents Omni-View, a unified 3D scene understanding and generation model that enhances understanding performance through a texture module (novel view synthesis) and a geometry module (depth/pose estimation), achieving a score of 55.4 on VSI-Bench and surpassing all existing specialized 3D understanding models.
tags:
  - ICLR 2026
  - 3D Vision
  - Unified Understanding and Generation
  - 3D Scene Understanding
  - Novel View Synthesis
  - Spatial Reasoning
  - Multi-view
date: 2026-05-08
content_hash: 4f9ac1b19ebf8631
---

# Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images

**Conference**: ICLR 2026
**arXiv**: [2511.07222](https://arxiv.org/abs/2511.07222)
**Code**: [https://github.com/AIDC-AI/Omni-View](https://github.com/AIDC-AI/Omni-View)
**Area**: 3D Vision / Multi-view Understanding
**Keywords**: Unified Understanding and Generation, 3D Scene Understanding, Novel View Synthesis, Spatial Reasoning, Multi-view

## TL;DR
This paper presents Omni-View, a unified 3D scene understanding and generation model that enhances understanding performance through a texture module (novel view synthesis) and a geometry module (depth/pose estimation), achieving a score of 55.4 on VSI-Bench and surpassing all existing specialized 3D understanding models.

## Background & Motivation

**Background**: Unified multimodal understanding and generation (UMM) has achieved remarkable progress in the 2D domain (e.g., BAGEL, Janus), yet unified models for 3D scenes remain largely unexplored. Existing 3D understanding methods (LLaVA-3D, GPT4Scene, etc.) rely on explicit 3D inputs (voxels, BEV), limiting their practical applicability.

**Limitations of Prior Work**: (a) 2D UMM research has only explored "understanding facilitating generation," while the reverse direction—"generation facilitating understanding"—has not been sufficiently validated; (b) 3D understanding tasks require geometric measurement and spatiotemporal modeling capabilities, yet existing models lack mechanisms to acquire them; (c) methods dependent on 3D inputs are difficult to deploy in real-world scenarios.

**Key Challenge**: 3D scene understanding (distance estimation, directional reasoning, appearance ordering) inherently requires geometric and spatiotemporal modeling capabilities, but purely understanding-oriented models learn only from a semantic perspective and cannot acquire these capabilities.

**Goal**: To endow an understanding model with geometric and spatiotemporal modeling capabilities via 3D generation tasks (geometric estimation + novel view synthesis), constructing the first unified understanding and generation model for general 3D scenes.

**Key Insight**: Drawing on neuroscientific evidence that human understanding of 3D environments relies on the ability to "generate and imagine" future sensory and geometric data, which directly motivates the applicability of the "generation facilitates understanding" paradigm in 3D scenes.

**Core Idea**: Novel view synthesis is leveraged to learn spatiotemporal modeling, while depth/pose estimation is leveraged to learn geometric measurement; the two generative capabilities jointly enhance 3D understanding.

## Method

### Overall Architecture
Built upon BAGEL-7B, the model comprises an understanding module and a generation module. The generation module is further divided into a texture module (flow matching for novel view synthesis) and a geometry module (depth map + camera pose estimation). Training proceeds in two stages: Stage 1 jointly trains all three components (generation facilitates understanding); Stage 2 freezes the understanding model and fine-tunes the generation modules.

### Key Designs

1. **Texture Module (Spatiotemporal Modeling)**:

    - **Function**: Generates novel views from a reference image and a target camera pose.
    - **Mechanism**: The reference image is encoded via FLUX-VAE; camera poses are encoded as Plücker rays and used as positional encodings; denoising is performed via flow matching. Autoregressive generation is adopted—when generating frame $n$, the model attends to the preceding $n-1$ frames, compelling the model to understand temporal sequential relationships.
    - **Design Motivation**: The autoregressive scheme enables the acquisition of spatiotemporal modeling capability, directly addressing the core demands of understanding tasks such as appearance ordering.

2. **Geometry Module (Geometric Measurement)**:

    - **Function**: Estimates depth maps and camera poses from the latent features of the texture module.
    - **Mechanism**: Receives the last-layer latent features from the texture module, concatenates depth noise and learnable pose queries, and fuses intermediate features from the understanding model via cross-attention. Depth is estimated via flow matching; pose is estimated via a VGGT decoder with Huber loss.
    - **Design Motivation**: Depth estimation enables the model to understand relative positional relationships between objects (corresponding to relative distance/direction tasks), and gradients can be back-propagated to the understanding model.

3. **Dense-to-Sparse (D2S) Training Strategy**:

    - **Function**: Progressively reduces the number of reference images.
    - **Mechanism**: At the beginning of training, reference images include all inputs (dense); they are progressively reduced to only the first frame (sparse), implementing a curriculum from easy to hard.
    - **Design Motivation**: Forcing the model to generate under increasingly limited information enhances its deep understanding of scene structure.

### Loss & Training

Stage 1: $L_{s1} = \lambda_{und} L_{und} + \lambda_{tex} L_{tex} + \lambda_{geo} L_{geo}$ (default weights 1:1:0.1). The understanding loss uses next-token prediction; the texture loss uses MSE (predicted vs. actual noise); the geometry loss uses depth MSE + pose Huber loss. Diffusion forcing is adopted during training to improve 3D consistency. Stage 2: The understanding model is frozen, and RGBDP joint learning is used to optimize generation quality.

## Key Experimental Results

### Main Results

VSI-Bench spatial reasoning (without 3D inputs):

| Method | Object Counting | Absolute Distance | Relative Distance | Appearance Order | Average |
|--------|----------------|-------------------|-------------------|-----------------|---------|
| SpatialMLLM-4B | 65.3 | 34.8 | 41.3 | 46.3 | 48.4 |
| VG-LLM-4B | 66.4 | 36.6 | 40.8 | 39.5 | 46.1 |
| BAGEL-7B-FT | 62.8 | 36.3 | 46.1 | 43.1 | 46.3 |
| **Omni-View-7B** | **70.3** | **46.4** | **65.9** | **49.0** | **55.4** |

Novel view synthesis (Re10k): PSNR = 23.22 (surpassing Voyager-13B at 23.12), LPIPS = 0.114 (substantially leading).

### Ablation Study

| Configuration | VSI-Bench Avg. | Note |
|---------------|---------------|------|
| Understanding only (BAGEL-FT) | 46.3 | Baseline |
| + Texture module | ~50 | Spatiotemporal modeling → Appearance order +4.1 |
| + Geometry module | ~49 | Geometry → Relative distance significantly improved |
| + Texture + Geometry (unified arch.) | ~52 | Unified inferior to separated |
| **+ Texture + Geometry (separated arch.)** | **55.4** | Separated design is optimal |
| w/o D2S strategy | Drops | Curriculum learning is effective |
| w/o autoregressive generation | Drops | Forced temporal understanding is effective |

### Key Findings
- Generation genuinely facilitates understanding: relative distance improves from 46.1→65.9 (+19.8), absolute distance from 36.3→46.4 (+10.1), appearance order from 43.1→49.0 (+5.9).
- The texture and geometry modules contribute distinct capabilities: texture → spatiotemporal modeling; geometry → spatial measurement.
- The separated dual-module design outperforms the unified architecture, avoiding gradient conflicts between the two generative objectives.
- The model surpasses most methods requiring 3D inputs without using any 3D input.
- Understanding and generation training data are completely non-overlapping, ruling out data memorization effects.

## Highlights & Insights
- **Systematic validation of "generation facilitates understanding"**: This intuition is validated at scale in 3D scenes for the first time, with ablation studies clearly disentangling the contributions of the texture module (spatiotemporal) and the geometry module (metric).
- **Separated dual-module design**: Treating texture and geometry separately outperforms a unified architecture by avoiding multi-task gradient conflicts—a generalizable design principle for other unified model architectures.
- **D2S curriculum learning**: Progressively reducing reference images is a simple yet effective curriculum strategy; the core logic is that "less information → harder → forces deeper understanding."
- **Dramatic gain in relative distance (+19.8)**: The most striking ablation result, clearly demonstrating the critical role of geometric estimation capability in spatial reasoning.

## Limitations & Future Work
- Camera pose control precision is insufficient—novel view synthesis only marginally outperforms specialized models at the pixel-fidelity level.
- Gains in absolute metric understanding (e.g., room size, absolute distance) are limited, as synthesized depth maps lack absolute scale.
- The model is at the 7B scale and has not been validated at larger scales.
- Data conditions are constrained: ScanNet/Re10k cover a limited range of scene types, with no validation on large-scale outdoor scenes.
- In Stage 2, the geometry module no longer depends on the understanding model's features, potentially causing inconsistency in generative capabilities across the two training stages.

## Related Work & Insights
- **vs. LLaVA-3D / GPT4Scene**: These methods require 3D inputs (voxels/BEV), yet Omni-View approaches or surpasses their performance using only multi-view images (ScanQA CIDEr: 103.0 vs. 102.1).
- **vs. SpatialMLLM / VG-LLM**: These methods embed 3D priors via VGGT features; Omni-View internalizes such priors through generative tasks, achieving superior results.
- **vs. BAGEL**: Direct fine-tuning of the BAGEL baseline yields only 46.3; incorporating the generation modules raises this to 55.4 (+9.1), confirming that the gain stems from generation rather than data.
- **Implications for unified models**: Generation and understanding are not merely two independent tasks—capabilities acquired during generation (spatiotemporal modeling, geometric measurement) can directly enhance understanding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified understanding and generation model for general 3D scenes; systematic validation of "generation facilitates understanding."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-benchmark evaluation across VSI-Bench/SQA3D/ScanQA/ScanRefer/Re10k with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, though some passages could be more concise.
- Value: ⭐⭐⭐⭐⭐ Pioneering contribution to unified 3D understanding and generation with state-of-the-art performance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding](../../CVPR2026/3d_vision/masking_matters_unlocking_the_spatial_reasoning_capabilities_of_llms_for_3d_scen.md)
- [\[ICLR 2026\] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)
- [\[ICCV 2025\] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing](../../ICCV2025/3d_vision/univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)
- [\[ICCV 2025\] UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation](../../ICCV2025/3d_vision/uniegomotion_a_unified_model_for_egocentric_motion_reconstruction_forecasting_an.md)
- [\[ICLR 2026\] Learning Unified Representation of 3D Gaussian Splatting](learning_unified_representation_of_3d_gaussian_splatting.md)

<!-- RELATED:END -->
