---
title: >-
  [Paper Note] Distilling Monocular Foundation Model for Fine-grained Depth Completion
description: >-
  [CVPR 2025][Autonomous Driving][Depth Completion] This paper proposes DMD3C, a two-stage knowledge distillation framework that transfers geometric knowledge from monocular depth foundation models (such as Depth Anything V2) to depth completion networks. The first stage performs pre-training using synthesized training data, and the second stage fine-tunes on real-world data utilizing a scale-shift invariant loss (SSI Loss), achieving first place on the KITTI depth completion l…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Depth Completion"
  - "Knowledge Distillation"
  - "Monocular Depth Estimation"
  - "Foundation Models"
  - "Scale-invariant Loss"
date: 2026-05-08
content_hash: 19de28836a21defa
---

# Distilling Monocular Foundation Model for Fine-grained Depth Completion

**Conference**: CVPR 2025  
**arXiv**: [2503.16970](https://arxiv.org/abs/2503.16970)  
**Code**: [Sharpiless/DMD3C](https://github.com/Sharpiless/DMD3C)  
**Area**: Autonomous Driving  
**Keywords**: Depth Completion, Knowledge Distillation, Monocular Depth Estimation, Foundation Models, Scale-invariant Loss

## TL;DR
This paper proposes DMD3C, a two-stage knowledge distillation framework that transfers geometric knowledge from monocular depth foundation models (such as Depth Anything V2) to depth completion networks. The first stage performs pre-training using synthesized training data, and the second stage fine-tunes on real-world data utilizing a scale-shift invariant loss (SSI Loss), achieving first place on the KITTI depth completion leaderboard.

## Background & Motivation

The objective of depth completion is to predict dense depth maps from sparse LiDAR inputs, which is widely applied in fields like autonomous driving, robotics, and augmented reality. A core challenge for current methods lies in the sparsity of training data, which severely limits the model's capacity to learn fine-grained geometric features. For instance, in the KITTI dataset, sparse LiDAR depth only covers about 5% of image pixels. Even after post-processing like multi-frame fusion, the ground truth coverage is only around 20%, leaving distant and dynamic object areas typically without annotations.

On the other hand, monocular depth estimation foundation models (such as Depth Anything V2, MiDaS) can predict dense depth maps with rich geometric details and relative depth relationships from a single RGB image. However, these models suffer from inherent scale ambiguity and cannot directly provide real-world metric depth.

The core idea of this work is to leverage monocular foundation models as dense supervision signals to compensate for the lack of sparse annotations, while addressing the scale ambiguity problem through a carefully designed two-stage distillation strategy.

## Method

### Overall Architecture
DMD3C adopts a two-stage distillation framework: the first stage pre-trains the depth completion model using unlabeled natural images via a data generation strategy to learn diverse geometric priors; the second stage fine-tunes the model on labeled real-world datasets, utilizing the SSI Loss to combine sparse supervision with monocular depth distillation to learn real-world scale.

### Key Designs

1. **Stage 1: Data Generation and Pre-training**:

    - Goal: Learn geometric features from massive unlabeled natural images without any ground truth depth.
    - Data sources: A mixture of approximately 360K diverse images, including COCO (118K), Google Landmarks (117K), nuScenes (93K), Cityscapes (20K), and DAVIS (10K).
    - Data generation pipeline:
      (a) Predict a monocular depth map for each image using Depth Anything V2.
      (b) Randomly sample a camera intrinsic matrix $K$.
      (c) Back-project the depth map to a 3D point cloud.
      (d) Generate continuous 3D meshes using Poisson surface reconstruction.
      (e) Simulate the LiDAR ray-casting process to generate sparse depth points.
    - Pre-training utilizes $L_1$ loss supervised by the synthesized dense depth.
    - Key novelty: Through mesh reconstruction and ray-casting simulation, the generated sparse depth highly aligns with real LiDAR scanning patterns.

2. **Stage 2: Fine-tuning with SSI Loss**:

    - Fine-tune on real-world datasets with sparse ground truth (KITTI, NYUv2).
    - Sparse $L_1$ supervision: Use sparse ground truth to provide real-world metric scale.
    - Scale-Shift Invariant Loss (SSI Loss): Resolves the scale ambiguity of monocular depth.
        - Goal: Find the optimal scale parameter $s$ and shift parameter $b$ that minimize $|D_f - (s \cdot D_m + b)|$.
        - By aligning the global scale and shift of the predicted depth and monocular depth, the model learns to preserve the relative depth structure.
    - Gradient matching regularization term: Maintains sharpness of depth edges, calculated over $K=4$ scale levels.
    - Final Loss = $L_{\text{sup}} + L_{\text{SSIL}} + L_{\text{reg}}$.

3. **Network Architecture Compatibility**:

    - The method is an innovation at the training strategy level and does not rely on a specific network architecture.
    - Main experiments use BP-Net (based on ResNet blocks), while validation is also performed on LRRU and CFormer.
    - Consistent performance gains are achieved across all architectures.

### Loss & Training
- Stage 1: $L_1$ pre-training loss, purely synthetic data, trained from scratch for 600K iterations.
- Stage 2: $L_{\text{sup}}$ (sparse $L_1$) + $L_{\text{SSIL}}$ (scale-invariant distillation) + $L_{\text{reg}}$ (gradient regularization), fine-tuned for 300K iterations.
- Optimizer: AdamW, weight decay 0.05, gradient clipping threshold 0.1.
- Use EMA (decay=0.9999) for training stabilization.
- Learning rate scheduler: OneCycle.
- Hardware: 4 $\times$ A100 GPUs, batch size 16.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| KITTI | RMSE (mm) ↓ | 678.12 | 684.90 (BP-Net) | -6.78 |
| KITTI | MAE (mm) ↓ | 194.46 | 187.95 (ImprovingDC) | - |
| KITTI | iRMSE ↓ | 1.82 | 1.82 (BP-Net/TPVD) | On par |
| NYUv2 | RMSE ↓ | 0.085 | 0.086 (TPVD) | -0.001 |
| NYUv2 | $\delta_{1.25}$ ↑ | 99.7 | 99.7 (TPVD) | On par |

### Ablation Study

| Configuration | RMSE ↓ | Description |
|------|---------|------|
| DMD3C (Full) | 678.12 | Two-stage distillation |
| w/o Pre-training | 682.34 (+4.22) | Remove first-stage pre-training |
| w/o SSI Loss | 684.54 (+6.42) | Remove SSI Loss |
| BP-Net baseline | 684.90 (+6.78) | Original BP-Net |
| LRRU baseline | 696.51 | Original LRRU |
| LRRU + Ours | 693.17 (-3.34) | Generalization validation |
| CFormer baseline | 764.87 | Original CFormer |
| CFormer + Ours | 760.29 (-4.58) | Generalization validation |

### Cross-Domain Zero-Shot Generalization

| Dataset | Ours (RMSE) | BP-Net (RMSE) | Gain |
|--------|-----------|-------------|------|
| ScanNet | 0.101 | 0.122 | -17.2% |
| DDAD | 7.766 | 8.903 | -12.8% |
| VOID1500 | 0.676 | 0.704 | -4.0% |

### Key Findings
- Pre-training and SSI Loss each contribute approximately 4-6 mm of RMSE improvement, showing complementary effects.
- The method is generalizable to different network architectures, bringing performance gains to BP-Net, LRRU, and CFormer.
- Zero-shot cross-domain performance significantly outperforms the baseline, particularly achieving a 12.8% RMSE reduction on DDAD, which demonstrates that pre-training significantly enhances generalization.
- High-quality depth maps are generated even in regions without ground truth, presenting more complete and coherent 3D point cloud visualizations.

## Highlights & Insights
- Simple and elegant concept: Utilizing existing monocular foundation models as "knowledge teachers" to supplement the shortage of sparse annotations.
- Ingenious design of the stage-one data generation pipeline: Synthesizing LiDAR via mesh reconstruction and ray-casting simulation effectively bypasses the domain gap.
- The design of the SSI Loss resolves the fundamental issue of scale ambiguity in monocular depth, rendering dense distillation feasible in real-world scenarios.
- The method represents an innovation at the training strategy level, offering plug-and-play capability to boost any depth completion architecture.
- The top ranking on the competitive KITTI leaderboard is highly convincing.

## Limitations & Future Work
- Although achieving first place in the RMSE metric, the MAE is not optimal, indicating potential large errors in certain regions.
- The first-stage pre-training requires 600K iterations, presenting a high computational cost.
- The quality of the monocular foundation model directly impacts distillation performance; distillation effectiveness may be limited in scenarios where foundation models underperform (e.g., highly reflective or transparent objects).
- Quantitative evaluations are limited to KITTI and NYUv2, whereas other depth completion benchmarks (e.g., Waymo) are not explored.
- The gradient matching regularization hyperparameter (number of scale levels $K = 4$) lacks sensitivity analysis.
- The performance of depth completion on fast-moving objects in dynamic scenes has not been specifically analyzed.

## Related Work & Insights
- Compared to competing methods on the KITTI leaderboard such as TPVD and ImprovingDC, this work achieves a decisive advantage by incorporating external knowledge from monocular foundation models.
- The concept of SSI Loss is inspired by G2-MonoDepth and MiDaS, but this work is the first to systematically apply it inside a depth completion task.
- The pre-training strategy is inspired by the Depth Anything series, yet it innovatively generates training data tailored for depth completion through mesh reconstruction and LiDAR simulation.
- The philosophy of this distillation framework can be extended to other sparsely supervised tasks (e.g., sparse-to-dense optical flow estimation, point cloud completion).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The two-stage distillation framework is logically designed, and the application of SSI Loss to depth completion is novel, though the core components are mostly combinations of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across multiple datasets, validated on multiple architectures, and with comprehensive zero-shot generalization experiments, although confined to two main benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, though mathematical notations are slightly redundant in some sections.
- **Value**: ⭐⭐⭐⭐⭐ Top-1 placement on the KITTI leaderboard provides the strongest validation, and the generality and plug-and-play characteristics of the method make it highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[CVPR 2025\] Distilling Multi-modal Large Language Models for Autonomous Driving](distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [\[CVPR 2026\] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](../../CVPR2026/autonomous_driving/ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [\[ICCV 2025\] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion](../../ICCV2025/autonomous_driving/distilling_diffusion_models_to_efficient_3d_lidar_scene_completion.md)
- [\[CVPR 2025\] SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization](superpc_a_single_diffusion_model_for_point_cloud_completion_upsampling_denoising.md)

</div>

<!-- RELATED:END -->
