---
title: >-
  [Paper Note] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation
description: >-
  [CVPR 2025][3D Vision][Multi-view Reconstruction] Proposes Murre, a novel multi-view 3D reconstruction framework. By injecting sparse SfM point clouds into a diffusion model to guide monocular depth estimation, it bypasses the multi-view matching step of traditional MVS and outperforms the SOTA on various real-world scenes (indoor, street, aerial).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-view Reconstruction"
  - "SfM Guidance"
  - "Monocular Depth Estimation"
  - "Diffusion Models"
  - "Depth Completion"
date: 2026-05-08
content_hash: deba132c53d599a8
---

# Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.14483](https://arxiv.org/abs/2503.14483)  
**Code**: [https://zju3dv.github.io/murre/](https://zju3dv.github.io/murre/)  
**Area**: 3D Vision / Multi-view Reconstruction  
**Keywords**: Multi-view Reconstruction, SfM Guidance, Monocular Depth Estimation, Diffusion Models, Depth Completion

## TL;DR

Proposes Murre, a novel multi-view 3D reconstruction framework. By injecting sparse SfM point clouds into a diffusion model to guide monocular depth estimation, it bypasses the multi-view matching step of traditional MVS and outperforms the SOTA on various real-world scenes (indoor, street, aerial).

## Background & Motivation

**Background**: Learning-based MVS methods perform poorly in low-texture areas and sparse views, and 3D cost volumes consume a significant amount of GPU memory.

**Limitations of Prior Work**: MVS implicitly relies on multi-view consistency, which leads to matching failures under sparse views; monocular depth estimation does not require matching but lacks multi-view consistency and metric information.

**Key Challenge**: Multi-view consistency requires matching, but matching is unreliable in challenging scenes; monocular prediction does not require matching but lacks consistency.

**Core Idea**: Use SfM point clouds as an explicit intermediate representation to inject multi-view information into a monocular depth diffusion model, combining the advantages of both.

## Method

### Overall Architecture

Given multi-view images: (1) SfM reconstructs a sparse point cloud; (2) the point cloud is projected onto each view to form sparse depth maps, which are then densified; (3) the densified depth maps and RGB images are used as conditional inputs to a diffusion model to predict metric depth; (4) TSDF fusion is applied to obtain the final geometry.

### Key Designs

1. **SfM Prior Injection into Diffusion Model**:

    - **Function**: Provides multi-view consistent metric information for monocular depth estimation.
    - **Mechanism**: Densifies SfM sparse depth maps using KNN interpolation and computes a distance map (measuring the distance of each pixel to the nearest valid point) as a confidence indicator. The densified depth map and the distance map are fed together as conditions into a depth diffusion model based on Stable Diffusion V2.
    - **Design Motivation**: SfM point clouds are a condensed representation of multi-view information, naturally providing metric scale and prominent scene structures.

2. **Depth Normalization and Scale Alignment**:

    - **Function**: Handles depth-range discrepancies across different scenes and viewpoints.
    - **Mechanism**: First filters out the top and bottom 2% outliers of SfM depths, then expands the range to $0.8 \times \min$ and $1.2 \times \max$, and uses this range to normalize ground-truth depths for training. During inference, RANSAC linear regression is used to align the predicted depths with SfM depths.
    - **Design Motivation**: SfM depths contain outliers and cover only a fraction of pixels, requiring a robust normalization strategy.

3. **Stable Diffusion-Based Depth Estimation**:

    - **Function**: Leverages the strong priors of 2D foundation models to achieve generalization.
    - **Mechanism**: Initialized from SD V2, freezing the VAE and only fine-tuning the UNet. The depth is duplicated into three channels and mapped to the latent space via the VAE encoder, where denoising and diffusion actions take place.
    - **Design Motivation**: Fine-tuning on a small amount of synthetic data enables generalization to diverse real-world scenes.

### Loss & Training

Standard diffusion denoising loss. Detector-free SfM is utilized to handle low-texture regions. The training data includes synthetic scenes.

## Key Experimental Results

### Main Results

| Dataset | Ours vs Prev. SOTA |
|--------|-------------|
| DTU | Outperforms existing MVS and neural implicit methods |
| ScanNet | Competitive performance |
| Waymo | Outperforms monocular methods |
| UrbanScene3D | Outperforms MVS |

### Key Findings

- SfM guidance significantly improves depth consistency and metric accuracy.
- KNN densification combined with the distance map is more effective than directly using sparse depths.
- Training on synthetic data is sufficient to generalize to various real-world scenarios.

## Highlights & Insights

- Injecting SfM as a "condensed" representation of multi-view information into a monocular model is an elegant idea.
- Bypassing the 3D cost volume effectively resolves the twin bottlenecks of GPU memory and sparse viewpoints.
- The generalization ability of 2D foundation models is effectively unleashed through SfM guidance.

## Limitations & Future Work

- Dependent on the quality of SfM; the method is affected when SfM fails.
- Multi-step inference of diffusion models introduces additional computational overhead.
- Dynamic scenes require extra handling.

## Rating

- Novelty: 8/10 — The combination of SfM and diffusion-based depth is novel.
- Technical Depth: 8/10 — The normalization and alignment strategies are detailed and well-designed.
- Experimental Thoroughness: 9/10 — Validated across five scene types.
- Writing Quality: 8/10 — The methodology description is clear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] MEt3R: Measuring Multi-View Consistency in Generated Images](met3r_measuring_multi-view_consistency_in_generated_images.md)

</div>

<!-- RELATED:END -->
