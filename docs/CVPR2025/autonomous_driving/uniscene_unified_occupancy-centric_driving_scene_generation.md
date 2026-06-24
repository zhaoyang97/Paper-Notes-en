---
title: >-
  [Paper Note] UniScene: Unified Occupancy-centric Driving Scene Generation
description: >-
  [CVPR 2025][Autonomous Driving][Occupancy Grid Generation] UniScene is proposed, a two-stage driving scene generation framework with occupancy grids as the unified intermediate representation. An Occupancy Diffusion Transformer generates semantic occupancy from BEV layouts, which is then rendered into semantic and depth maps via Gaussian Splatting to condition dual diffusion models for generating video and LiDAR. UniScene achieves an FVD of 71.94 (compared to the previous SOT…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Occupancy Grid Generation"
  - "Gaussian Rendering"
  - "Video Generation"
  - "LiDAR Generation"
  - "Diffusion Transformer"
date: 2026-05-08
content_hash: 799433fca6122aa5
---

# UniScene: Unified Occupancy-centric Driving Scene Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.05435](https://arxiv.org/abs/2412.05435)  
**Code**: [https://arlo0o.github.io/uniscene/](https://arlo0o.github.io/uniscene/) (Project Page)  
**Area**: Autonomous Driving / Scene Generation  
**Keywords**: Occupancy Grid Generation, Gaussian Rendering, Video Generation, LiDAR Generation, Diffusion Transformer

## TL;DR

UniScene is proposed, a two-stage driving scene generation framework with occupancy grids as the unified intermediate representation. An Occupancy Diffusion Transformer generates semantic occupancy from BEV layouts, which is then rendered into semantic and depth maps via Gaussian Splatting to condition dual diffusion models for generating video and LiDAR. UniScene achieves an FVD of 71.94 (compared to the previous SOTA Drive-WM of 122.70) and improves downstream 3D detection by 3.62% mAP via data augmentation.

## Background & Motivation

### Background

**Background**: Autonomous driving data collection is expensive and struggles to cover corner cases. Scene generation can provide synthetic training data. Existing methods generate videos (e.g., DriveDreamer) or LiDAR (e.g., LiDARDM) separately, but they suffer from inconsistency—videos and LiDAR for the same scene may describe different geometries.

**Limitations of Prior Work**: (1) Independent video and LiDAR generation leads to multi-modal inconsistency; (2) directly generating pixels or point clouds involves extremely high-dimensional spaces; (3) there is a lack of a unified intermediate representation to bridge different modalities.

**Key Challenge**: Generating multi-modal data (video + LiDAR + semantics) with consistent structure simultaneously, despite the drastically different data spaces of each modality.

**Key Insight**: Occupancy grids serve as a natural intermediate layer connecting 2D videos and 3D point clouds—they contain both spatial structure and semantic information, and can be deterministically rendered into any modality.

**Core Idea**: BEV $\rightarrow$ Occupancy Grid (DiT) $\rightarrow$ Gaussian Rendering $\rightarrow$ Conditional Video & LiDAR Diffusion = Structurally consistent multi-modal scene generation.

## Method

### Key Designs

1. **Occupancy DiT**: A VAE compresses the occupancy grid (mIoU 72.9%), and a DiT generates it in the latent space conditioned on the BEV layout. 3D axial attention (attention applied along H/W/Z dimensions sequentially) is utilized to reduce computational complexity.

2. **Gaussian Joint Rendering**: Semantic occupancy voxels are converted to 3D Gaussians to render semantic and depth maps. Depth maps provide geometric priors, and semantic maps provide content priors.

3. **Translation Strategy**: Videos are generated using a depth-aware noise prior (encoding depth information into the initial noise of diffusion), while LiDAR is generated using a sparse UNet combined with prior-guided sparse ray sampling (achieving a 96× speedup compared to LiDARDM).

### Loss & Training

Occupancy VAE: $\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{LS} + \lambda_2 \mathcal{L}_{KL}$. Standard denoising loss is used for DiT. Respective diffusion losses and conditionings are used for video and LiDAR.

## Key Experimental Results

| Task | Ours | Prev. SOTA | Gain |
|------|---------|---------|------|
| Video FVD↓ | **71.94** | 122.70 (Drive-WM) | 41% |
| Video FID↓ | **6.45** | 13.97 (Vista) | 54% |
| LiDAR MMD↓ | **2.40e-4** | 3.51e-4 (LiDARDM) | 32% |
| LiDAR Speed | **0.47s** | 45.12s | **96×** |
| Downstream 3D Detection mAP | +3.62% | — | Effective Augmentation |

### Ablation Study
- 3D axial attention contributes 38% of the performance.
- Rendering semantic and depth maps contributes 30-35% of the FVD improvement.
- Sparse ray sampling reduces computation by 59% with only a 4% JSD degradation.

### Key Findings
- **Occupancy grids serve as an excellent unified intermediate representation**, ensuring geometric consistency between video and LiDAR.
- **96× LiDAR speedup**—sparse prior guidance reduces generation time from minutes to seconds.
- **Effective downstream augmentation**: Occupancy prediction improves by +8.5% IoU, and BEV segmentation improves by +7.39% mIoU.

## Highlights & Insights
- **Occupancy-centric unified framework**—a single occupancy grid drives both video and LiDAR generation simultaneously.
- **Gaussian rendering as a bridge**—efficiently transforms 3D occupancy into 2D conditional signals.

## Limitations & Future Work
- Requires occupancy annotations for training.
- Supported raw data rate is limited to 2Hz (12Hz via interpolation).
- Calibration of dynamic objects remains unresolved.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Occupancy-centric multi-modal unified generation is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across four dimensions: video, LiDAR, occupancy, and downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear and complete.
- Value: ⭐⭐⭐⭐⭐ Provides a complete generation pipeline for autonomous driving data augmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](../../ICCV2025/autonomous_driving/hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[ICLR 2026\] WorldSplat: Gaussian-Centric Feed-Forward 4D Scene Generation for Autonomous Driving](../../ICLR2026/autonomous_driving/worldsplat_gaussian-centric_feed-forward_4d_scene_generation_for_autonomous_driv.md)
- [\[CVPR 2025\] VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving](visionpad_a_vision-centric_pre-training_paradigm_for_autonomous_driving.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](rethinking_temporal_fusion_with_a_unified_gradient_descent_view_for_3d_semantic_.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](../../ICCV2025/autonomous_driving/uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)

</div>

<!-- RELATED:END -->
