---
title: >-
  [Paper Note] Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos
description: >-
  [AAAI 2026][3D Vision][4D Gaussian Splatting] Proposes a coarse-to-fine temporal alignment module that can be plugged into existing 4D Gaussian Splatting frameworks. It addresses the degradation of dynamic scene reconstruction quality caused by temporal desynchronization in multi-view videos, significantly improving PSNR/SSIM/LPIPS of multiple baseline methods on the DyNeRF dataset.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "temporal alignment"
  - "dynamic scene reconstruction"
  - "multi-view videos"
  - "unsynchronized cameras"
date: 2026-05-08
content_hash: 33358f8c83252c22
---

# Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos

**Conference**: AAAI 2026  
**arXiv**: [2511.11175](https://arxiv.org/abs/2511.11175)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 4D Gaussian Splatting, temporal alignment, dynamic scene reconstruction, multi-view videos, unsynchronized cameras

## TL;DR
Proposes a coarse-to-fine temporal alignment module that can be plugged into existing 4D Gaussian Splatting frameworks. It addresses the degradation of dynamic scene reconstruction quality caused by temporal desynchronization in multi-view videos, significantly improving PSNR/SSIM/LPIPS of multiple baseline methods on the DyNeRF dataset.

## Background & Motivation

**Background**: 4D Gaussian Splatting (4DGS) is a predominant method for dynamic scene reconstruction, achieving real-time rendering and high-fidelity reconstruction by representing scenes with explicit Gaussian primitives. Representative methods include 4DGaussians, SC-GS, RT4DGS, etc.

**Limitations of Prior Work**: These methods assume strict temporal synchronization among multi-view cameras—i.e., all cameras trigger simultaneously at the same timestamp. However, in practical scenarios (independent recording via consumer-grade cameras, mobile phones, GoPros, etc.), millisecond-to-second-level temporal offsets are almost inevitable due to the lack of a unified clock, network latency, manual operations, and other factors.

**Key Challenge**: When models attempt to fuse views captured at physically different timestamps to reconstruct a scene at a single logical timestamp, fast-moving objects exhibit significant positional discrepancies across views, leading to severe artifacts such as ghosting and motion blur. Such inconsistent observations mislead the 4DGS optimization process, incorrectly attributing temporal errors to spatial geometry or appearance deficiencies.

**Goal**: To perform high-quality 4D dynamic scene reconstruction from unsynchronized multi-view videos without requiring specialized synchronization hardware.

**Key Insight**: Explicitly incorporate the unknown temporal offset of each camera into the optimization objective, accurately estimating the temporal misalignment by decomposing it into two stages: coarse-grained frame-level offset and fine-grained sub-frame offset.

**Core Idea**: Design a plug-and-play coarse-to-fine temporal alignment module that utilizes LoFTR feature matching for coarse alignment and learnable parameters for fine refinement, optimized jointly with 4DGS in an end-to-end manner.

## Method

### Overall Architecture
The input consists of unsynchronized multi-view videos, and the output is a high-quality 4D Gaussian scene representation. The method consists of two stages:
1. **Coarse Temporal Alignment**: Employs feature matching to identify the integer frame-level offset of each camera.
2. **Fine Temporal Refinement**: Jointly optimizes a learnable sub-frame offset parameter during training.

The entire module can be seamlessly integrated into existing 4DGS frameworks without architectural modifications to the baselines.

### Key Designs

1. **Coarse Temporal Alignment**:

    - **Function**: Estimates the frame-level integer offset $\Delta t_j^*$ of each non-reference video with respect to the reference video.
    - **Mechanism**: When two views capture the scene at the exact same moment, the dynamic foreground objects are effectively "instantaneously static," causing the number of cross-view feature matches to peak. A LoFTR dense feature matcher is employed to generate candidate correspondences, and RANSAC is used to fit the fundamental matrix, where the number of geometrically consistent inliers serves as the alignment score.
    - **Formula**: $\Delta t_j^* = \arg\max_{\Delta t_j \in [-k,k]} \sum_{t_i} N_{\text{inlier-fg}}(I_{\text{ref}}^{t_i}, I_j^{t_i + \Delta t})$
    - **Design Motivation**: Exhaustively search all candidate offsets within the search range $[-k, k]$, selecting the offset with the maximum number of foreground inliers as the coarse alignment result. A video segmentation model is used beforehand to extract foreground masks, limiting the matching to dynamic foreground regions to prevent interference from static backgrounds.

2. **Fine Temporal Refinement**:

    - **Function**: Learns a continuous sub-frame residual offset $\tau_j$ on top of the coarse alignment.
    - **Mechanism**: Introduces a learnable parameter $\tau_j$ for each camera $j$, leading to the final query time $t' = t + \Delta t_j^* + \tau_j$. $\tau_j$ is jointly trained with the 4DGS model, optimized via the gradient backpropagation of the photometric reconstruction loss.
    - **Design Motivation**: Coarse alignment only achieves frame-level accuracy, whereas fast-moving scenes require sub-frame precision. End-to-end differentiable optimization is used to discover and correct residual sub-frame temporal discrepancies.

3. **Integration with Different 4D Representations**:

    - **Neural 4D Representations** (e.g., 4DGaussians, SC-GS): The temporal input to the deformation network $\mathcal{D}_\theta(\gamma(\boldsymbol{\mu}_k), \gamma(t))$ is modified from $t$ to $t + \Delta_j^* + \tau_j$. Since the output is differentiable with respect to the temporal input, the gradient of $\tau_j$ can be naturally computed via backpropagation.
    - **Direct 4D Representations** (e.g., RT4DGS): The original implementation does not support gradient computation with respect to timestamp $t$. Therefore, finite difference is used for approximation: $\frac{\partial \mathcal{L}}{\partial t} \approx \frac{\mathcal{L}(t+h) - \mathcal{L}(t)}{h}$, where $h$ is set to 1/30 of the frame interval.

### Loss & Training
Uses the original photometric reconstruction loss (e.g., L1 + SSIM) of each baseline method, only adding offsets to the temporal input. Hyperparameters remain consistent with the baselines, and no additional loss terms are introduced.

## Key Experimental Results

### Main Results
Evaluated on the DyNeRF dataset across 6 dynamic scenes with approximately 20 views, downsampled to 15 FPS, with a random temporal offset of up to 10 frames applied.

| Method | Coffee Martini PSNR | Cook Spinach PSNR | Flame Steak PSNR | Sear Steak PSNR |
|------|-----|-----|-----|-----|
| 4DGaussians | 26.44 | 31.44 | 30.68 | 29.67 |
| 4DGaussians+Ours | **28.01** | **32.57** | **32.63** | **32.51** |
| RT4DGS* | 27.92 | 31.15 | 31.13 | 32.94 |
| RT4DGS*+Ours | **28.35** | **33.15** | **33.34** | **33.51** |

All baseline methods achieve consistent improvements after incorporating the proposed module, with RT4DGS+Ours achieving optimal performance in most scenes.

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 4DGaussians (w/o alignment) | 29.56 | 0.935 | 0.099 |
| +Coarse only | 30.92 | 0.943 | 0.092 |
| +Fine only | 30.87 | 0.941 | 0.091 |
| +Full (Coarse+Fine) | **31.16** | **0.942** | **0.091** |

### Key Findings
- Both coarse and fine alignments individually reduce artifacts significantly, but their combination achieves the best performance.
- As the random temporal offset increases from 3 to 10 frames, baseline performance drops sharply (PSNR from 30.69 → 29.60). However, the performance is barely affected after incorporating the proposed module (31.25 → 31.16).
- Foreground mask filtering is crucial for coarse alignment—it prevents match points belonging to static background from interfering with the temporal alignment of dynamic objects.

## Highlights & Insights
- **The "instantaneous static" assumption is elegant**: When multi-view cameras capture the scene at the same moment, the dynamic object becomes static relative to multiple views. Therefore, maximizing the number of cross-view feature matches is equivalent to temporal alignment. This is an intuitively simple yet highly effective insight.
- **Plug-and-play design**: The module does not modify the core architecture of the baseline. It only alters the temporal input, truly achieving "plug-and-play" capability. This design philosophy can be transferred to other scenarios addressing input misalignment issues.
- **Handling non-differentiable cases via finite difference**: For methods like RT4DGS that do not provide temporal gradients, utilizing finite difference approximation proves highly practical.

## Limitations & Future Work
- Only addresses temporal offsets (translation) and does not consider cases with different frame rates (rate discrepancies).
- The experiments are verified only on the single DyNeRF dataset, lacking evaluation on real-world outdoor scenes.
- The computational overhead of LoFTR + RANSAC during the coarse alignment stage is not analyzed in detail.
- Assumes that the temporal offset between all cameras is constant (time-invariant), whereas clock drift might occur in practice.

## Related Work & Insights
- **vs 4DGaussians**: 4DGaussians assumes synchronized inputs; this work extends its applicability via the temporal alignment module.
- **vs NeRF-based Dynamic Reconstruction**: The proposed method can similarly be extended to NeRF frameworks, though it is currently validated only on the 3DGS series.
- **vs Flow-constrained Methods (GaussianFlow, MotionGS)**: These methods introduce motion constraints via optical flow, which is complementary to the temporal alignment approach proposed in this paper.

## Rating
- Novelty: ⭐⭐⭐⭐ The first to systematically address the temporal desynchronization issue in 4DGS with clear insights.
- Experimental Thoroughness: ⭐⭐⭐ Validated only on a single dataset (DyNeRF), lacking scene diversity.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and detailed methodology description.
- Value: ⭐⭐⭐⭐ High practicality, lowering the hardware threshold for dynamic scene capture.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](../../ICCV2025/3d_vision/beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[CVPR 2026\] MAPo: Motion-Aware Partitioning of Deformable 3D Gaussian Splatting for High-Fidelity Dynamic Scene Reconstruction](../../CVPR2026/3d_vision/mapo_motion-aware_partitioning_of_deformable_3d_gaussian_splatting_for_high-fide.md)
- [\[CVPR 2026\] Point4Cast: Streaming Dynamic Scene Reconstruction and Forecasting](../../CVPR2026/3d_vision/point4cast_streaming_dynamic_scene_reconstruction_and_forecasting.md)

</div>

<!-- RELATED:END -->
