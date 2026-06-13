---
title: >-
  [Paper Note] Stepper: Stepwise Immersive Scene Generation with Multiview Panoramas
description: >-
  [CVPR 2026][3D Vision][Panorama Generation] This paper proposes Stepper, a framework that generates immersive 3D scenes driven by text input by progressively synthesizing multi-view panoramas and feeding them into a feed…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Panorama Generation"
  - "3D Scene Synthesis"
  - "Diffusion Models"
  - "Multi-view Consistency"
  - "Immersive Scenes"
date: 2026-05-08
content_hash: e0a0eebe6180864e
---

# Stepper: Stepwise Immersive Scene Generation with Multiview Panoramas

**Conference**: CVPR 2026 Findings  
**arXiv**: [2603.28980](https://arxiv.org/abs/2603.28980)  
**Code**: [Project Page](https://fwmb.github.io/stepper)  
**Area**: 3D Vision / Scene Generation
**Keywords**: Panorama Generation, 3D Scene Synthesis, Diffusion Models, Multi-view Consistency, Immersive Scenes

## TL;DR

This paper proposes Stepper, a framework that generates immersive 3D scenes driven by text input by progressively synthesizing multi-view panoramas and feeding them into a feed-forward 3D reconstruction pipeline, achieving an average PSNR improvement of 3.3 dB over existing methods.

## Background & Motivation

**Background**: Synthesizing explorable immersive 3D scenes from text or images is a core task in computer vision, with broad applications in AR/VR and spatial computing. Current mainstream approaches fall into two categories: autoregressive outpainting methods (e.g., DiffDreamer, Text2Room), which progressively fill in novel viewpoints using image/video models; and panorama-lifting methods (e.g., HoloDreamer, Matrix-3D), which directly lift 360° panoramas into 3D space.

**Limitations of Prior Work**: Autoregressive methods rely on perspective images with limited fields of view, causing context drift—geometric errors accumulate and visual fidelity degrades as the number of extension steps increases. Panorama-lifting methods yield good quality near the projection center but fail to handle disoccluded regions, producing blurring and stretching when rendering far from the origin. Panoramic video generation methods (e.g., Matrix-3D) offer better consistency but are constrained by the computational cost of video generation models, achieving only 1440×720 resolution with insufficient detail.

**Key Challenge**: There is a fundamental trade-off between visual fidelity and explorability—high quality but limited range vs. wide range but poor quality.

**Goal**: How can large-baseline scene exploration be achieved while maintaining high resolution and high fidelity?

**Key Insight**: The authors observe that panoramas are powerful scene context representations (covering 360° information), and that the cubemap representation decomposes a panorama into standard perspective images, enabling direct reuse of pretrained 2D image diffusion models for high-resolution panorama generation without the resolution bottleneck of video models.

**Core Idea**: Redefine scene extension as a multi-view cubemap generation problem—at each step, a complete novel-view panorama is generated at a fixed forward distance, reconciling high resolution with global consistency.

## Method

### Overall Architecture

Stepper consists of three core components: (1) a multi-view panorama diffusion model that generates a novel-view panorama at a fixed distance ahead from an input panorama; (2) a geometry reconstruction pipeline based on the feed-forward reconstruction model MapAnything, which lifts multiple panoramas into a consistent 3D point cloud; and (3) 3D Gaussian Splatting (3DGS) optimization, which converts the point cloud into a real-time renderable scene representation. The overall pipeline is: text → CubeDiff generates the initial panorama → multi-step autoregressive extension → MapAnything reconstructs the point cloud → 3DGS optimization → real-time exploration.

### Key Designs

1. **Multi-view Panorama Diffusion**:

    - **Function**: Generates a novel-view panorama $P_{nv}$ at a fixed step size $d=0.25\text{m}$ ahead from an input panorama $P_{in}$.
    - **Mechanism**: Each of the two panoramas is decomposed into 6 cubemap faces (12 perspective images total), which are fed directly into a pretrained LDM with batch size $t=12$. The key innovation is inflating the self-attention layers in the deeper layers of the LDM, extending the token sequence from $(bt) \times (hw) \times l$ to $b \times (thw) \times l$, so that each cubemap face token can attend to all other faces—including all faces of both the input panorama and the novel-view panorama—ensuring cross-view and cross-panorama consistency. UV coordinate positional encodings and panorama-source identifiers are concatenated as additional conditioning signals.
    - **Design Motivation**: The cubemap representation eliminates polar distortions of equirectangular projection, and each face is a standard perspective image (90° FOV) consistent with the pretraining data distribution, avoiding the need for training from scratch. Panorama-level context covers the complete scene, fundamentally reducing context drift.

2. **Feed-Forward Geometry Reconstruction Pipeline**:

    - **Function**: Converts multiple generated panoramas into a consistent 3D point cloud.
    - **Mechanism**: Rather than relying on error-prone monocular depth estimation alignment, the feed-forward SfM model MapAnything is applied directly to perspective views extracted from panoramas. To match MapAnything's training data distribution, a dedicated view extraction scheme is designed—views are sampled from horizontal cubemap faces with ±45° vertical rotations to ensure sufficient overlap. To control point cloud size, an iterative construction strategy is adopted: PyTorch3D's point cloud renderer checks whether points introduced by a new panorama are already visible in previous panoramas, and only newly unobserved points are added.
    - **Design Motivation**: This avoids the distortion of monocular depth estimators on spherical data, and the end-to-end SfM approach provides more robust multi-view geometric consistency than depth alignment.

3. **3DGS Optimization and Stepwise Scene Exploration**:

    - **Function**: Converts the point cloud into a real-time renderable 3DGS representation and supports autoregressive multi-step exploration.
    - **Mechanism**: 3DGS is initialized with the accurate point cloud from MapAnything, and a simplified MCMC-GS optimization strategy is adopted—Gaussian positions are fixed, and only the color of each Gaussian is optimized as an appearance representation. Training views include 6 cubemap faces and 8 additional perspective views. During scene exploration, starting from the initial panorama, $n$ steps are taken in each of four directions, yielding $1+4n$ panoramic views covering a large scene extent.
    - **Design Motivation**: Fixing Gaussian positions leverages the accurate initialization from the feed-forward model, reducing the complexity of the under-constrained problem.

### Loss & Training

For training data, a large-scale synthetic multi-view panorama dataset is constructed using the Infinigen procedural generation framework, comprising approximately 230,000 panorama pairs (resolution 4096×2048) across 5,000 indoor and outdoor scenes. The diffusion model is fine-tuned using the standard diffusion loss on cubemap faces of panorama pairs for 90,000 steps with a batch size of 1 (12 cubemap faces), trained on 4 ViperFish TPUs with sharding (64 TPUs total), giving an effective batch size of 16. The step size is fixed at $d=0.25\text{m}$, as experiments show that a fixed step size yields more stable training than a variable one.

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| WorldExplorer | 13.145 | 0.624 | 0.648 |
| LayerPano3D | 17.931 | 0.688 | 0.503 |
| Matrix-3D | 18.133 | 0.665 | 0.515 |
| **Stepper (Ours)** | **21.426** | **0.735** | **0.385** |

(Averaged results across three subsets: Infinigen indoor/outdoor + Blender scenes)

### Ablation Study

| Setting | Effect |
|---|---|
| Single panorama vs. multi-panorama input to 3DGS | Multi-panorama input significantly reduces holes while preserving quality at the initial viewpoint |
| Variable step direction | Increased geometric errors and texture artifacts |
| Step size $d=0.5\text{m}$ vs. $d=0.25\text{m}$ | $0.5\text{m}$ still generates high-quality panoramas but with slightly weaker detail preservation |

### Key Findings

- Stepper comprehensively outperforms all baselines across all datasets and metrics, with an average PSNR improvement of at least 3.3 dB.
- SSIM of 0.735 vs. 0.688 for the second-best LayerPano3D; LPIPS of 0.385 vs. 0.503 for the second-best.
- Fixed step size outperforms variable step size—the fixed step simplifies the learning task and stabilizes generation quality.
- Panorama-level context is the key to reducing drift: covering the complete scene at 360°, as opposed to the limited field of view of perspective images, fundamentally suppresses semantic and geometric inconsistency at the source.

## Highlights & Insights

- **Paradigm Innovation**: The problem of scene extension is recast from "frame-by-frame video generation" to "multi-view cubemap image generation," elegantly circumventing the resolution bottleneck of video models while retaining the global context advantage of panoramas.
- **Dataset Contribution**: A large-scale dataset of 230,000 multi-view panorama pairs is constructed, filling a critical data gap in the field and providing a unified quantitative evaluation benchmark.
- **Refined Engineering Design**: The view extraction scheme for MapAnything (45°-rotated sampling), the iterative deduplication strategy for point cloud construction, and the simplified 3DGS optimization with fixed Gaussian positions are all practical and well-validated engineering contributions.

## Limitations & Future Work

- The fixed step size of 0.25 m may lack flexibility for scenes of varying scales.
- Only four horizontal directions are currently supported; exploration in the vertical direction (e.g., staircase scenes) is limited.
- Reliance on Infinigen synthetic data for training may limit generalization to real-world scenes.
- The autoregressive generation process still accumulates some bias, which is mitigated rather than fundamentally resolved by panorama-level context.
- Fixing Gaussian positions during 3DGS optimization simplifies the problem but may lack flexibility in regions with complex occlusions.

## Related Work & Insights

- CubeDiff [Uy et al.] establishes the cubemap paradigm that underpins this work—transforming panorama generation into multi-view image generation.
- MapAnything [Hong et al.] provides powerful feed-forward SfM capabilities, enabling a seamless pipeline from generated images to 3D reconstruction.
- The stepwise extension strategy proposed in this paper can be generalized to other tasks requiring large-scale scene generation (e.g., autonomous driving simulation, game world generation).
- The contrast between panorama-level context and perspective-image local context offers important insights for other autoregressive generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of redefining scene extension as multi-view cubemap generation is novel, though individual components (cubemap diffusion, MapAnything, 3DGS) are integrations of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative and qualitative comparisons are thorough, with multi-subset evaluation and ablation studies, though user studies and real-scene evaluation are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Logically clear, with polished figures and detailed method descriptions.
- **Value**: ⭐⭐⭐⭐ — The dataset and unified benchmark offer long-term value to the community, and the method has clear prospects for practical applications in AR/VR scene generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Geometry: Artistic Disparity Synthesis for Immersive 2D-to-3D](beyond_geometry_artistic_disparity_synthesis_for_immersive_2d-to-3d.md)
- [\[CVPR 2026\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp3d_joint_open_vocabulary_semantic_segmentation.md)
- [\[CVPR 2026\] HyperMVP: Hyperbolic Multiview Pretraining for Robotic Manipulation](hyperbolic_multiview_pretraining_for_robotic_manipulation.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)

</div>

<!-- RELATED:END -->
