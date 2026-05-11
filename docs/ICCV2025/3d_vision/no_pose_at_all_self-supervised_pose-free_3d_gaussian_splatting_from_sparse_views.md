---
title: >-
  [Paper Note] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes SPFSplat, the first self-supervised 3DGS framework that requires no ground-truth poses at either training or inference time. By sharing a ViT backbone to…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "pose-free training"
  - "self-supervised"
  - "sparse views"
  - "pose estimation"
date: 2026-05-08
content_hash: c58cc11df73e98f8
---

# No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views

**Conference**: ICCV 2025
**arXiv**: [2508.01171](https://arxiv.org/abs/2508.01171)
**Code**: [Project Page](https://ranrhuang.github.io/spfsplat/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, pose-free training, self-supervised, sparse views, pose estimation

## TL;DR

This paper proposes SPFSplat, the first self-supervised 3DGS framework that requires no ground-truth poses at either training or inference time. By sharing a ViT backbone to jointly predict Gaussian primitives and camera poses, SPFSplat surpasses pose-dependent state-of-the-art methods under extreme viewpoint changes.

## Background & Motivation

Existing sparse-view NVS methods can be categorized into three groups based on pose dependency:

**Pose-required**: pixelSplat, MVSplat — rely on SfM, which is unreliable under sparse views.

**Supervised pose-free**: NoPoSplat, Splatt3R — pose-free at inference but require ground-truth poses during training, limiting data scalability.

**Self-supervised pose-free**: PF3plat, SelfSplat — use separate modules for pose estimation and reconstruction, leading to feature inconsistency and unstable feedback loops.

The core challenge is that **rendering loss inherently couples the learning of 3D geometry and camera poses**. Pose errors degrade reconstruction quality, which in turn further impairs pose estimation, forming an unstable feedback loop.

Existing self-supervised methods employ **independent modules** for pose estimation and scene reconstruction, operating in different feature spaces without sharing geometric knowledge, resulting in training instability.

## Method

### Shared Backbone Architecture

A ViT-based encoder-decoder:
- **Encoder**: Shared weights, processes each view independently.
- **Decoder**: Aggregates multi-view information via cross-attention.
- **Gaussian prediction head**: Two DPT heads predict center positions and other attributes.
- **Pose head**: A lightweight 3-layer MLP built on the same decoder, outputting a 10-dimensional pose representation.

Pose representation: 4D homogeneous translation + 6D rotation (two unnormalized basis vectors; the full rotation matrix is constructed via cross-product).

### Dual-Branch Design During Training

- **Context-only branch**: Predicts Gaussians from context views only (used at inference).
- **Context-with-target branch**: Includes both context and target views to estimate target poses (used during training only).

Crucially, Gaussian reconstruction and target pose prediction are **decoupled**, preventing target-view information from leaking into the 3D representation.

### Reprojection Loss

$$\mathcal{L}_{reproj} = \sum_{v=1}^N \sum_{j=1}^{H \times W} \|\mathbf{p}_j^v - \pi(\boldsymbol{K}^v, \boldsymbol{P}^{v \to 1}, \boldsymbol{\mu}_j^{v \to 1})\|$$

The reprojection loss is applied to poses from both branches, enforcing alignment between Gaussian centers and image pixels.

**Why not replace this with rendering loss?** Including context views in the rendering loss causes training collapse — the model preferentially optimizes the first view (whose rendering is pose-independent), suppressing Gaussians from other views.

### Total Loss

$$\mathcal{L} = \mathcal{L}_{render} + \mathcal{L}_{reproj}$$

where $\mathcal{L}_{render} = \|I^t - \hat{I^t}\|_2 + \gamma\text{LPIPS}(I^t, \hat{I^t})$

## Key Experimental Results

### Novel View Synthesis on RealEstate10K

| Method | Category | Small PSNR | Medium PSNR | Large PSNR | Avg PSNR |
|--------|----------|-----------|------------|-----------|---------|
| pixelSplat | Pose-required | 20.28 | 23.73 | 27.15 | 23.86 |
| MVSplat | Pose-required | 20.37 | 23.81 | 27.47 | 24.01 |
| NoPoSplat | Supervised pose-free | 22.51 | 24.90 | 27.41 | 25.03 |
| SelfSplat | Self-supervised | 14.83 | 18.86 | 23.34 | 19.15 |
| PF3plat | Self-supervised | 18.36 | 20.95 | 23.49 | 21.04 |
| **SPFSplat** | **Self-supervised** | **22.90** | **25.10** | **27.65** | **25.27** |

### Pose Estimation Performance

SPFSplat surpasses state-of-the-art methods that rely on geometric priors, without requiring any pose supervision.

### Key Findings

1. **First self-supervised method to outperform pose-required methods** — SPFSplat achieves Avg PSNR of 25.27, surpassing NoPoSplat (25.03) and MVSplat (24.01).
2. The advantage is particularly pronounced under **large viewpoint changes** (Small category).
3. Inference speed is comparable to NoPoSplat (0.042s), far faster than PF3plat (1.171s).
4. The reprojection loss is critical to training stability.

## Highlights & Insights

1. **Truly pose-free** — No ground-truth poses are required at training or inference, greatly expanding the range of trainable data.
2. **Mutual reinforcement through shared backbone** — Pose estimation benefits from scene geometry, while Gaussian prediction benefits from accurate alignment, forming a positive feedback loop.
3. **Critical role of reprojection loss** — Resolves the training collapse caused by pure rendering loss.
4. **Pose-agnostic canonical space** — Predicting Gaussians in the first-view coordinate frame reduces the impact of pose errors on geometry.

## Limitations & Future Work

- Requires known camera intrinsics.
- Currently supports sparse-view (2–3 views) input.
- Although the method shows clear advantages at large baselines, absolute reconstruction quality still has room for improvement.

## Related Work & Insights

- **Pose-required**: pixelSplat, MVSplat, GRM
- **Supervised pose-free**: NoPoSplat, Splatt3R, LEAP, PF-LRM
- **Self-supervised pose-free**: PF3plat, SelfSplat, Nope-NeRF
- **SfM**: DUSt3R, MASt3R, VGGSfM

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First pose-free self-supervised method to surpass pose-required approaches)
- Technical Depth: ⭐⭐⭐⭐⭐ (Insightful feedback-loop analysis and elegant reprojection loss design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive comparison against all three method categories)
- Value: ⭐⭐⭐⭐⭐ (Truly pose-free; scalable to unannotated data)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)
- [\[ICCV 2025\] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations](pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](../../NeurIPS2025/3d_vision/onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](../../CVPR2026/3d_vision/e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)

</div>

<!-- RELATED:END -->
