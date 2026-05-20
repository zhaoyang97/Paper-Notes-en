---
title: >-
  [Paper Note] StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams
description: >-
  [ICLR 2026][3D Vision][Dynamic 3D Reconstruction] StreamSplat proposes a fully feed-forward online dynamic 3D reconstruction framework that enables instant generation of dynamic 3DGS representations from uncalibrated vid…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Dynamic 3D Reconstruction"
  - "3D Gaussian Splatting"
  - "Online Reconstruction"
  - "Feed-Forward Model"
  - "Video Streams"
date: 2026-05-08
content_hash: 7f0a7ec0a6e0ac45
---

# StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams

**Conference**: ICLR 2026
**arXiv**: [2506.08862](https://arxiv.org/abs/2506.08862)  
**Code**: [https://streamsplat3d.github.io/](https://streamsplat3d.github.io/)  
**Area**: 3D Vision
**Keywords**: Dynamic 3D Reconstruction, 3D Gaussian Splatting, Online Reconstruction, Feed-Forward Model, Video Streams

## TL;DR

StreamSplat proposes a fully feed-forward online dynamic 3D reconstruction framework that enables instant generation of dynamic 3DGS representations from uncalibrated video streams, achieving 1200× speedup over optimization-based methods through three key innovations: probabilistic position sampling, bidirectional deformation fields, and adaptive Gaussian fusion.

## Background & Motivation

Real-time dynamic 3D reconstruction (4D reconstruction) is critical for robotics, AR/VR, and autonomous driving. However, existing methods suffer from fundamental limitations:

**Offline Dependency**: Mainstream dynamic 3DGS methods (e.g., 4DGS, DGMarbles) require access to complete video sequences and undergo hours of per-scene iterative optimization, including multi-stage pipelines of camera calibration → static Gaussian optimization → deformation field learning → temporal fusion.

**Poor Real-Time Performance**: Even state-of-the-art methods require 30 minutes to 24 hours to process a single scene, precluding deployment in real-time applications.

**Calibration Requirements**: Nearly all methods require pre-calibrated camera parameters.

**Limitations of Feed-Forward Methods**: Existing feed-forward 3DGS methods (pixelSplat, NoPoSplat, StreamGS) support only static scenes; dynamic variants still require calibration and full-sequence access.

The authors pose a core research question: **Can offline-quality reconstruction be achieved under fully online conditions using uncalibrated video streams?**

## Method

### Overall Architecture

StreamSplat maintains a canonical Gaussian set $\tilde{\mathcal{G}}(t)$ and, for each input frame, executes a pipeline of encoding → bidirectional deformation prediction → adaptive fusion → rendering. Training follows a two-stage strategy: a static encoder is trained first, after which the encoder is frozen for dynamic decoder training.

### Key Designs

1. **Probabilistic Position Sampling**

   To address the sensitivity of 3DGS to position initialization and the tendency of feed-forward models to fall into local optima, the method predicts a truncated normal distribution over each 3D offset rather than directly regressing it:

    $\boldsymbol{o} \sim \mathcal{N}_{[-1,1]}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$

   The final 3D position is obtained via pixel-aligned prediction: $\boldsymbol{\mu}_i = (u_i + o_{i,0},\; v_i + o_{i,1},\; g(o_{i,2}))$, where $g(z) = 2/(1+z)$ is a depth mapping function. This strategy promotes spatial exploration during early training and stabilizes convergence to optimal positions in later stages. Ablation results show that probabilistic sampling improves PSNR by 6.36 dB over deterministic prediction.

2. **Bidirectional Deformation Field**

   Traditional methods instantiate new Gaussians per frame and optimize iteratively, which is ill-suited for feed-forward models. This work jointly models forward and backward motion: the forward field deforms Gaussians from the previous frame $\mathcal{G}_{t-1}$ to the current time $t$, while the backward field deforms current-frame Gaussians $\mathcal{G}_t$ back to $t-1$. This symmetric design:
    - Provides robust cross-frame correspondence
    - Naturally handles appearing/disappearing Gaussians
    - Simplifies prediction and supervision for end-to-end training

3. **Adaptive Gaussian Fusion**

   Soft-matching fusion is achieved through temporally dependent opacity modulation, allowing each Gaussian to persist across two consecutive frames:

    $\alpha(t) = \alpha \cdot \frac{\sigma(-\gamma_0(|t - t_0| - \gamma_1))}{\sigma(\gamma_0 \cdot \gamma_1)}$

   where $t_0$ is the frame at which the Gaussian was created, $\gamma_0$ controls the transition rate, and $\gamma_1$ controls the fade-out window. This mechanism implicitly fuses forward and backward Gaussians: the reconstruction loss induces soft matching, propagating persistent Gaussians while handling appearing/disappearing ones, maintaining temporal consistency without hard assignment or iterative fusion.

### Loss & Training

**Stage 1 — Static Encoder**:
$$\mathcal{L}_{\text{static}} = \mathcal{L}_{\text{recon}}(\hat{I}_t, I_t) + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}}(\hat{D}_t, D_t)$$
The depth loss adopts a scale-and-shift invariant formulation, with an adaptive decay factor $\hat{\lambda}_{\text{depth}}$ introduced to reduce the influence of noisy pseudo-depth.

**Stage 2 — Dynamic Decoder** (encoder frozen):
$$\mathcal{L}_{\text{dynamic}} = \mathbb{E}_t[\mathcal{L}_{\text{recon}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}]$$
An auxiliary reconstruction loss on moving foreground regions is added, supervised by segmentation masks from DAVIS/YouTube-VOS.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (StreamSplat) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| DAVIS Key Frame | PSNR↑ | 37.83 | 42.33 (MonST3R) | Competitive |
| DAVIS Key Frame | LPIPS↓ | 0.016 | 0.012 (MonST3R) | Close |
| DAVIS Middle-4 | PSNR↑ | **23.66** | 21.33 (DGMarbles) | +2.33 |
| DAVIS Middle-4 | LPIPS↓ | **0.193** | 0.313 (DGMarbles) | −0.12 |
| RE10K Average | PSNR↑ | **29.51** | 23.73 (DGMarbles) | +5.78 |
| 8-frame Interpolation | PSNR↑ | **22.10** | 21.09 (AMT) | +1.01 |

### Ablation Study

| Configuration | PSNR (Key)↑ | PSNR (Mid)↑ | Note |
|------|---------|---------|------|
| w/o Probabilistic Sampling | 31.47 | — | Deterministic prediction, −6.36 dB |
| w/o Depth Supervision | 36.68 | — | Spatial structure degraded |
| w/o Bidirectional Deformation | — | 18.89 | Pixel-aligned structure lost |
| Full (Ours) | **37.83** | **23.66** | Complete model |

### Key Findings

- StreamSplat is the only method supporting near-real-time dynamic 3D reconstruction at 0.049 seconds per frame, 1200× faster than optimization-based methods.
- It achieves competitive key-frame reconstruction quality against MonST3R, which requires post-optimization and is limited to key frames.
- It surpasses all baselines on intermediate-frame reconstruction, including 2D video interpolation methods.
- It supports online reconstruction of arbitrarily long video streams.

## Highlights & Insights

1. **Online Processing Paradigm**: StreamSplat is the first to realize feed-forward online dynamic 3D reconstruction on uncalibrated video streams, fundamentally departing from the traditional offline multi-stage pipeline.
2. **Probabilistic Position Sampling**: A simple yet effective solution to the local-optima problem in feed-forward 3DGS, yielding substantial improvement (+6.36 dB).
3. **Adaptive Opacity Fusion**: Temporal opacity modulation enables soft matching, elegantly circumventing the hard assignment and iterative fusion required by conventional methods.
4. **Canonical Space Design**: An orthographic canonical space bypasses per-scene camera calibration, absorbing camera motion into Gaussian dynamics.

## Limitations & Future Work

- Key-frame reconstruction quality remains slightly below that of MonST3R (point cloud representation), though MonST3R does not support online processing.
- Input resolution is limited to 512×288; high-resolution scenes may suffer from loss of detail.
- Evaluation is conducted only on short-to-medium-length videos; error accumulation over very long sequences requires further validation.
- The orthographic projection assumption may be limiting in scenes with strong perspective effects.

## Related Work & Insights

- NoPoSplat (Ye et al., 2024) and StreamGS (Li et al., 2025) address the pose-free and online settings, respectively, but are both restricted to static scenes.
- The bidirectional deformation concept is generalizable to other temporal modeling tasks (video generation, autonomous driving prediction).
- The lifecycle management idea underlying adaptive Gaussian fusion is inspired by Zhao et al. (2024).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to achieve online feed-forward dynamic 3D reconstruction from uncalibrated video streams, with three synergistically designed technical innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple dynamic/static benchmarks with detailed ablations, but lacks evaluation on longer video sequences.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem motivation is clear, methodological exposition is logically coherent, and figures are well-crafted.
- Value: ⭐⭐⭐⭐⭐ — The 1200× speedup carries significant practical value and opens a new paradigm for online dynamic reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](../../AAAI2026/3d_vision/mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)
- [\[CVPR 2026\] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](../../CVPR2026/3d_vision/onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction](../../ICCV2025/3d_vision/dynamic_point_maps_a_versatile_representation_for_dynamic_3d_reconstruction.md)
- [\[CVPR 2026\] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator](../../CVPR2026/3d_vision/gaussfusion_improving_3d_reconstruction_in_the_wild_with_a_geometry-informed_vid.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](../../NeurIPS2025/3d_vision/onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)

</div>

<!-- RELATED:END -->
