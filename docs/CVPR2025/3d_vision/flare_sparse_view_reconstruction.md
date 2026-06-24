---
title: >-
  [Paper Note] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views
description: >-
  [CVPR 2025][3D Vision][Sparse-view Reconstruction] Proposed is FLARE, a feed-forward differentiable system that simultaneously infers high-quality camera poses, 3D geometry, and appearance from uncalibrated sparse-view images (2-8 views) in 0.5 seconds, progressively simplifying the complex 3D learning task by employing a cascaded learning paradigm with camera poses acting as a bridge.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Sparse-view Reconstruction"
  - "Camera Estimation"
  - "3D Gaussians"
  - "Feed-forward"
  - "Pointmaps"
date: 2026-05-08
content_hash: 970595d4d843de3a
---

# FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views

**Conference**: CVPR 2025  
**arXiv**: [2502.12138](https://arxiv.org/abs/2502.12138)  
**Code**: [https://zhanghe3z.github.io/FLARE/](https://zhanghe3z.github.io/FLARE/)  
**Area**: 3D Vision / Sparse-view Reconstruction  
**Keywords**: Sparse-view Reconstruction, Camera Estimation, 3D Gaussians, Feed-forward, Pointmaps

## TL;DR

Proposed is FLARE, a feed-forward differentiable system that simultaneously infers high-quality camera poses, 3D geometry, and appearance from uncalibrated sparse-view images (2-8 views) in 0.5 seconds, progressively simplifying the complex 3D learning task by employing a cascaded learning paradigm with camera poses acting as a bridge.

## Background & Motivation

**Background**: SfM+MVS is the classic two-stage reconstruction pipeline, but feature matching is difficult under sparse views; DUSt3R/MASt3R predict pointmaps but rely on post-optimization for global registration.

**Limitations of Prior Work**: DUSt3R only supports pairwise matching followed by global optimization, which is time-consuming and yields sub-optimal results; the triplane representation in PF-LRM limits its performance in large-scale scenes; existing methods fail to resolve camera estimation, geometric reconstruction, and appearance modeling simultaneously and efficiently.

**Key Challenge**: Jointly optimizing pose, geometry, and appearance directly from images is highly prone to falling into local optima.

**Goal**: To design a cascaded learning paradigm that progressively lowers learning difficulty by utilizing camera poses as intermediate proxies.

**Core Idea**: Estimate coarse poses first → guide local geometry in camera coordinates → project to global coordinates → generate 3D Gaussians for rendering.

## Method

### Overall Architecture

A four-step cascade: (1) Neural Pose Predictor estimates coarse poses; (2) Camera-centric geometry estimation predicts local pointmaps under each camera coordinate system; (3) Global Geometry Projector unifies local pointmaps into global coordinates; (4) 3D Gaussian regression head is used for novel view synthesis.

### Key Designs

1. **Neural Pose Predictor**:

    - **Function**: Directly regress camera poses from sparse-view images
    - **Mechanism**: Concatenate image patches and learnable camera latents into a 1D sequence, which is then fed into a small decoder-only transformer to predict 7D poses (translation + normalized quaternion)
    - **Design Motivation**: Skip feature matching and regress poses directly; even imperfect poses provide valuable spatial initialization

2. **Two-stage Geometry Estimation**:

    - **Function**: Progressively learn geometry from local to global
    - **Mechanism**: Map local pointmaps under each camera coordinate system first (consistent with the imaging process, simplifying learning), then use a learnable geometry projector to transform local pointmaps into global coordinates. Pose perturbations with noise are added during training to enhance robustness.
    - **Design Motivation**: Local prediction avoids directly reasoning about complex global spatial relationships, decomposing the learning difficulty.

3. **3D Gaussian Appearance Modeling**:

    - **Function**: Generate renderable 3D Gaussians from estimated geometry
    - **Mechanism**: Use global pointmaps as Gaussian centers, and predict opacity, rotation, scale, and spherical harmonics after fusing VGG features with appearance features. To address scale inconsistency, both predicted and ground-truth pointmaps are normalized to a unit space.
    - **Design Motivation**: Decouple geometry and appearance, where geometry serves as the geometric scaffold for 3D Gaussians.

### Loss & Training

Total loss = pose loss (Huber) + geometry loss (confidence-aware L2) + Gaussian rendering loss (L2 + VGG perceptual + depth). Jointly trained on a mixture of large-scale public datasets.

## Key Experimental Results

### Main Results

On RealEstate10K and multiple datasets:
- Pose estimation: Outperforms DUSt3R and MASt3R
- Novel view synthesis: Outperforms existing pose-free methods
- Inference speed: < 0.5 seconds

### Key Findings

- Cascaded learning is significantly better than direct joint learning
- Two-stage geometry (local → global) converges faster than direct global prediction
- Noisy pose augmentation is crucial for inference-time robustness

## Highlights & Insights

- Core insight of the cascaded learning paradigm: Pose as a bridge from 2D to 3D reduces learning complexity
- Inference speed of 0.5 seconds is orders of magnitude faster than optimization-based methods
- Can handle an arbitrary number of input images

## Limitations & Future Work

- GPU memory limits the number of images processed simultaneously
- Still partially dependent on the quality of camera pose estimation
- Performance may degrade in textureless regions

## Rating

- Novelty: 8/10 — Well-designed cascaded learning paradigm
- Technical Depth: 8/10 — Complete multi-task joint learning framework
- Experimental Thoroughness: 8/10 — Multi-dataset validation
- Writing Quality: 8/10 — Clear structure

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[CVPR 2025\] Light3R-SfM: Towards Feed-forward Structure-from-Motion](light3r-sfm_towards_feed-forward_structure-from-motion.md)
- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[CVPR 2025\] UniK3D: Universal Camera Monocular 3D Estimation](unik3d_universal_camera_monocular_3d_estimation.md)

</div>

<!-- RELATED:END -->
