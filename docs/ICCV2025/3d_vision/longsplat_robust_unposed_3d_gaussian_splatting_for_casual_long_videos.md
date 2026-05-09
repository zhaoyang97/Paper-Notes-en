---
title: >-
  [Paper Note] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos
description: >-
  [3D Vision] LongSplat targets casually captured long videos without known camera poses. It proposes an incremental joint optimization framework that simultaneously optimizes camera poses and 3DGS, introduces a robust pose estimation module based on MASt3R priors, and designs an adaptive octree anchor formation mechanism, collectively addressing pose drift, inaccurate geometry initialization, and memory constraints.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: f3215b0c3e5418ce
---

# LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2508.14041](https://arxiv.org/abs/2508.14041)
- **Code**: [Project Page](https://linjohnss.github.io/longsplat/)
- **Area**: 3D Vision
- **Keywords**: 3D Gaussian Splatting, unposed long video reconstruction, octree anchor, incremental joint optimization, pose estimation

## TL;DR
LongSplat targets casually captured long videos without known camera poses. It proposes an incremental joint optimization framework that simultaneously optimizes camera poses and 3DGS, introduces a robust pose estimation module based on MASt3R priors, and designs an adaptive octree anchor formation mechanism, collectively addressing pose drift, inaccurate geometry initialization, and memory constraints.

## Background & Motivation

Casually captured long videos are an important source of 3D content, yet they pose unique challenges for novel view synthesis:

**Pose estimation difficulty**: COLMAP frequently fails on casual videos; foundation models such as MASt3R accumulate errors and drift over long sequences.

**Memory constraints**: COLMAP-free methods such as CF-3DGS encounter out-of-memory (OOM) issues at large scales.

**Complex trajectories**: LocalRF produces fragmented reconstructions under irregular camera motion.

**Lack of global consistency**: Incremental methods are prone to local optima.

## Method

### Overall Architecture

LongSplat adopts a fully incremental pipeline:
1. Initialization: MASt3R global alignment point cloud → octree anchor 3DGS
2. Global optimization: joint optimization of all poses and Gaussians
3. Per-frame insertion: PnP pose estimation + photometric refinement + anchor expansion
4. Alternating local–global optimization

### Key Design 1: Octree Anchor Formation

Unlike the fixed-resolution voxels in Scaffold-GS, LongSplat adaptively subdivides voxels based on point cloud density:

$$\epsilon_{l+1} = \frac{1}{2}\epsilon_l$$

Voxels with density below $\tau_{\text{prune}}$ are pruned, while those exceeding $\tau_{\text{split}}$ are recursively subdivided up to a maximum level $L$. The spatial scale of each anchor is proportional to its voxel size: $s_v \propto \epsilon_v$. An overlap check against existing anchors prevents duplication.

### Key Design 2: Robust Pose Estimation Module

For each incoming frame $t$:

1. **PnP Initialization**: 2D–3D correspondences are established using MASt3R 2D matches and back-projected depth from the previous rendered frame; pose is estimated via PnP+RANSAC:

$$X_i = D_{t-1}(x_i) \cdot K^{-1}\tilde{x}_i$$

2. **Photometric Refinement**: Minimizes the discrepancy between the rendered image and the actual frame:

$$\mathcal{L}_{\text{photo}} = \sum_{p \in \Omega}\|I_t(p) - \hat{I}_t(p)\|^2$$

3. **Depth Scale Correction**: Aligns the scale between MASt3R depth and rendered depth:

$$\hat{s}_t = \frac{\langle D_{t-1}, D_t^{\text{align}}\rangle}{\langle D_t^{\text{align}}, D_t^{\text{align}}\rangle}$$

4. **Occlusion-Aware Expansion**: Newly exposed regions are detected via forward warping, back-projected into 3D points, and converted into octree anchors.

5. **Fallback Mechanism**: When PnP fails, a global re-optimization is triggered before retrying.

### Key Design 3: Visibility-Adaptive Local Window

Co-visibility between frames is measured via IoU over anchor visibility sets:

$$\text{IoU}(t, t') = \frac{|\mathcal{V}(t) \cap \mathcal{V}(t')|}{|\mathcal{V}(t) \cup \mathcal{V}(t')|}$$

Frames falling below threshold $\tau$ are excluded from the local optimization window, ensuring Gaussians receive balanced multi-view supervision.

### Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{photo}} + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}} + \lambda_{\text{reprojection}}\mathcal{L}_{\text{reprojection}}$$

## Key Experimental Results

### Main Results: Free Dataset

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Notes |
|--------|-------|-------|--------|-------|
| COLMAP + Scaffold-GS | Failed | - | - | Pose estimation failure |
| CF-3DGS | - | - | - | OOM |
| LocalRF | Low | Low | High | Fragmented reconstruction |
| MASt3R + Scaffold-GS | Medium | Medium | Medium | Inaccurate poses |
| **LongSplat** | **Best** | **Best** | **Best** | Robust |

LongSplat consistently outperforms all baselines across all scenes.

### Ablation Study: Key Components

| Octree Anchor | Pose Refinement | Incremental Opt. | PSNR↑ |
|---------------|-----------------|------------------|-------|
| ✗ (fixed voxel) | ✗ | ✗ | Baseline |
| ✓ | ✗ | ✗ | Gain + memory saving |
| ✓ | ✓ | ✗ | Significant gain |
| ✓ | ✓ | ✓ | **Best** |

Octree anchors substantially reduce memory usage while maintaining quality; pose refinement is the largest contributor to performance improvement.

### Key Findings
- COLMAP completely fails on 14 out of 19 Free scenes, whereas LongSplat succeeds on all.
- Octree anchors reduce memory by 30–50% compared to fixed voxels.
- The PnP fallback mechanism is triggered on approximately 5–10% of frames, significantly improving robustness.
- LongSplat also achieves state-of-the-art results on Tanks and Temples and the Hike dataset.

## Highlights & Insights

1. **End-to-end unposed reconstruction**: No dependency on COLMAP or accurate camera calibration.
2. **Incremental design**: Frame-by-frame processing with controllable memory footprint, suitable for long sequences.
3. **MASt3R as a soft prior**: Used as a flexible initialization rather than a hard constraint; errors are progressively corrected via joint optimization.
4. **Comprehensive robustness mechanisms**: The combination of PnP fallback, photometric refinement, and scale correction ensures stable reconstruction.

## Limitations & Future Work
- The quality of MASt3R estimates for the initial frames has a substantial impact on the overall reconstruction.
- PnP may be unstable under extremely fast motion or pure rotation.
- Global optimization becomes slower as the number of frames grows.
- When camera intrinsics are unknown, the method relies on focal length estimates from MASt3R.

## Related Work & Insights
- CF-3DGS: Progressive unposed 3DGS optimization
- LocalRF: Localized radiance field construction
- MASt3R / DUSt3R: 3D foundation models
- Scaffold-GS / Octree-GS: Anchor-based / octree-based 3DGS

## Rating
- **Novelty**: ⭐⭐⭐⭐ — A complete system integrating octree anchors, incremental joint optimization, and PnP fallback
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly handles casually captured smartphone videos with high practical value
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset validation with comprehensive ablation studies
- **Writing Quality**: ⭐⭐⭐⭐ — System description is clear with intuitive pipeline diagrams

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)

<!-- RELATED:END -->
