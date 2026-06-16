---
title: >-
  [Paper Note] Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos
description: >-
  [AAAI 2026][3D Vision][4D Gaussian Splatting] This paper proposes a coarse-to-fine temporal alignment module that can be plugged into existing 4D Gaussian Splatting frameworks to address reconstruction quality degradatio…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "temporal alignment"
  - "dynamic scene reconstruction"
  - "multi-view video"
  - "unsynchronized cameras"
date: 2026-05-08
content_hash: d6e102485fa3c605
---

# Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos

**Conference**: AAAI 2026
**arXiv**: [2511.11175](https://arxiv.org/abs/2511.11175)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: 4D Gaussian Splatting, temporal alignment, dynamic scene reconstruction, multi-view video, unsynchronized cameras

## TL;DR
This paper proposes a coarse-to-fine temporal alignment module that can be plugged into existing 4D Gaussian Splatting frameworks to address reconstruction quality degradation caused by temporal misalignment across multi-view videos. The method achieves consistent improvements in PSNR/SSIM/LPIPS over multiple baselines on the DyNeRF dataset.

## Background & Motivation

**Background**: 4D Gaussian Splatting (4DGS) is the dominant approach for dynamic scene reconstruction, achieving real-time rendering and high-fidelity reconstruction via explicit Gaussian primitives. Representative methods include 4DGaussians, SC-GS, and RT4DGS.

**Limitations of Prior Work**: These methods uniformly assume that multi-view cameras are strictly time-synchronized—i.e., all cameras trigger simultaneously at each timestamp. In practice, however, independently operated cameras (consumer cameras, smartphones, GoPros, etc.) inevitably exhibit temporal offsets ranging from milliseconds to seconds due to the absence of a unified clock, network latency, and manual operation.

**Key Challenge**: When a model attempts to fuse views captured at physically different moments to reconstruct a single logical timestamp, rapidly moving objects exhibit significant positional discrepancies across views, producing severe artifacts such as ghosting and motion blur. These inconsistent observations mislead the 4DGS optimization process, causing it to incorrectly attribute temporal errors to spatial geometry or appearance deficiencies.

**Goal**: To achieve high-quality 4D dynamic scene reconstruction from unsynchronized multi-view videos without requiring dedicated synchronization hardware.

**Key Insight**: Explicitly incorporating the unknown temporal offset of each camera into the optimization objective, and estimating temporal misalignment through a two-stage decomposition into coarse frame-level offsets and fine sub-frame offsets.

**Core Idea**: Design a coarse-to-fine plug-and-play temporal alignment module that combines LoFTR-based feature matching for coarse alignment with learnable parameters for fine alignment, jointly optimized end-to-end with the 4DGS framework.

## Method

### Overall Architecture
The input consists of multi-view unsynchronized videos, and the output is a high-quality 4D Gaussian scene representation. The method proceeds in two stages:
1. **Coarse Temporal Alignment**: Uses feature matching to identify an integer frame-level offset for each camera relative to a reference.
2. **Fine Temporal Refinement**: Jointly optimizes a learnable sub-frame offset parameter during training.

The entire module integrates seamlessly into existing 4DGS frameworks without modifying the core architecture of the baseline methods.

### Key Designs

1. **Coarse Temporal Alignment**:

    - **Function**: Estimates the integer frame-level offset $\Delta t_j^*$ of each non-reference video relative to the reference video.
    - **Mechanism**: When two views capture the scene at the same moment, dynamic foreground objects are effectively "instantaneously static," and the number of cross-view feature matches reaches its peak. The method employs the LoFTR dense feature matcher to generate candidate correspondences, then applies RANSAC to fit a fundamental matrix and counts geometrically consistent inliers as the alignment score.
    - **Formula**: $\Delta t_j^* = \arg\max_{\Delta t_j \in [-k,k]} \sum_{t_i} N_{\text{inlier-fg}}(I_{\text{ref}}^{t_i}, I_j^{t_i + \Delta t})$
    - **Design Motivation**: All candidate offsets within the search range $[-k, k]$ are exhaustively evaluated, and the offset yielding the maximum number of foreground inliers is selected as the coarse alignment result. A video segmentation model is used to extract foreground masks in advance, restricting matching to dynamic foreground regions to avoid interference from the static background.

2. **Fine Temporal Refinement**:

    - **Function**: Learns a continuous sub-frame residual offset $\tau_j$ on top of the coarse alignment.
    - **Mechanism**: A learnable parameter $\tau_j$ is introduced for each camera $j$, so the final query time becomes $t' = t + \Delta t_j^* + \tau_j$. The parameter $\tau_j$ is jointly trained with the 4DGS model and optimized through gradients backpropagated from the photometric reconstruction loss.
    - **Design Motivation**: Coarse alignment achieves only frame-level precision, whereas fast-motion scenes require sub-frame accuracy. End-to-end differentiable optimization is used to discover and correct residual sub-frame temporal discrepancies.

3. **Integration with Different 4D Representations**:

    - **Neural 4D representations** (e.g., 4DGaussians, SC-GS): The temporal input to the deformation network $\mathcal{D}_\theta(\gamma(\boldsymbol{\mu}_k), \gamma(t))$ is replaced from $t$ with $t + \Delta_j^* + \tau_j$. Since the output is differentiable with respect to the temporal input, gradients for $\tau_j$ are naturally computed via backpropagation.
    - **Direct 4D representations** (e.g., RT4DGS): As the original implementation does not include gradient computation with respect to timestamp $t$, finite differences are used to approximate: $\frac{\partial \mathcal{L}}{\partial t} \approx \frac{\mathcal{L}(t+h) - \mathcal{L}(t)}{h}$, where $h$ is set to 1/30 of the inter-frame interval.

### Loss & Training
The original photometric reconstruction losses of each baseline method (e.g., L1 + SSIM) are used without modification; only the temporal offset is added to the time input. Hyperparameters are kept consistent with the respective baselines, and no additional loss terms are introduced.

## Key Experimental Results

### Main Results
Evaluated on the DyNeRF dataset, comprising 6 dynamic scenes with approximately 20 viewpoints, downsampled to 15 FPS with random temporal offsets of up to 10 frames applied.

| Method | Coffee Martini PSNR | Cook Spinach PSNR | Flame Steak PSNR | Sear Steak PSNR |
|------|-----|-----|-----|-----|
| 4DGaussians | 26.44 | 31.44 | 30.68 | 29.67 |
| 4DGaussians+Ours | **28.01** | **32.57** | **32.63** | **32.51** |
| RT4DGS* | 27.92 | 31.15 | 31.13 | 32.94 |
| RT4DGS*+Ours | **28.35** | **33.15** | **33.34** | **33.51** |

All baseline methods show consistent improvements upon incorporating the proposed module, with RT4DGS+Ours achieving the best performance in most scenes.

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 4DGaussians (no alignment) | 29.56 | 0.935 | 0.099 |
| +Coarse only | 30.92 | 0.943 | 0.092 |
| +Fine only | 30.87 | 0.941 | 0.091 |
| +Full (Coarse+Fine) | **31.16** | **0.942** | **0.091** |

### Key Findings
- Both coarse and fine alignment independently reduce artifacts significantly, but their combination yields the best overall performance.
- As the random temporal offset increases from 3 to 10 frames, baseline performance degrades sharply (PSNR: 30.69→29.60), whereas the proposed module maintains nearly stable performance (31.25→31.16).
- Foreground mask filtering is critical for coarse alignment—it prevents static background matches from interfering with temporal alignment of dynamic objects.

## Highlights & Insights
- **The "instantaneous static" assumption is elegant**: When multiple views capture the scene at the same moment, dynamic objects appear static across views; thus, maximizing cross-view feature match counts is equivalent to temporal alignment. This is an intuitive yet highly effective insight.
- **Plug-and-play design**: The module does not alter the core architecture of baseline methods—it only modifies the temporal input—achieving true drop-in compatibility. This design philosophy is transferable to other scenarios involving input misalignment.
- **Finite differences for non-differentiable cases**: For methods such as RT4DGS that do not expose temporal gradients, finite difference approximation is employed, offering strong practical applicability.

## Limitations & Future Work
- Only temporal translation offsets are handled; frame rate mismatches (speed discrepancies) are not considered.
- Experiments are conducted solely on the DyNeRF dataset; evaluation on real-world outdoor scenes is absent.
- The computational overhead of LoFTR + RANSAC in the coarse alignment stage is not analyzed in detail.
- The method assumes constant temporal offsets between cameras (i.e., no clock drift over time), which may not hold in practice.

## Related Work & Insights
- **vs. 4DGaussians**: 4DGaussians assumes synchronized inputs; this work extends its applicability via the temporal alignment module.
- **vs. NeRF-based dynamic reconstruction**: The proposed approach could in principle be extended to NeRF frameworks, though it is currently validated only within the 3DGS family.
- **vs. optical flow–constrained methods** (GaussianFlow, MotionGS): These methods impose motion constraints via optical flow, which is complementary to the temporal alignment strategy presented here.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic treatment of temporal asynchrony in 4DGS, with a clear and compelling insight.
- Experimental Thoroughness: ⭐⭐⭐ Validation limited to a single dataset (DyNeRF); scene diversity is insufficient.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and the method is described in detail.
- Value: ⭐⭐⭐⭐ Highly practical; lowers the hardware barrier for dynamic scene capture.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](../../ICCV2025/3d_vision/beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[ICLR 2026\] Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos](../../ICLR2026/3d_vision/mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)

</div>

<!-- RELATED:END -->
