---
title: >-
  [Paper Note] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM
description: >-
  [ECCV 2024][3D Vision][SLAM] Ours proposes SGS-SLAM, the first semantic visual SLAM system based on Gaussian Splatting. By optimizing multiple channels to integrate appearance, geometry, and semantic features, it achieves state-of-the-art (SOTA) performance in camera pose estimation, map reconstruction, and semantic segmentation.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "SLAM"
  - "3D Gaussian Splatting"
  - "Semantic Segmentation"
  - "Dense Reconstruction"
  - "Real-time Rendering"
date: 2026-05-08
content_hash: 4435e5b5585f5efc
---

# SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM

**Conference**: ECCV 2024  
**arXiv**: [2402.03246](https://arxiv.org/abs/2402.03246)  
**Code**: [GitHub](https://github.com/ShuhongLL/SGS-SLAM)  
**Area**: 3D Vision  
**Keywords**: SLAM, 3D Gaussian Splatting, Semantic Segmentation, Dense Reconstruction, Real-time Rendering

## TL;DR

Ours proposes SGS-SLAM, the first semantic visual SLAM system based on Gaussian Splatting. By optimizing multiple channels to integrate appearance, geometry, and semantic features, it achieves state-of-the-art (SOTA) performance in camera pose estimation, map reconstruction, and semantic segmentation.

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: Existing NeRF-based SLAM methods (such as NICE-SLAM and Co-SLAM) utilize MLPs as implicit representations, which suffer from three core issues: (1) over-smoothed object edges and a lack of fine details; (2) difficulty in decoupling object representations, hindering scene editing; (3) catastrophic forgetting, where new scenes damage the learned model. Concurrently, existing Gaussian SLAM methods (such as SplaTAM) lack semantic understanding capabilities.

## Method

### Overall Architecture

SGS-SLAM represents scenes using isotropic 3D Gaussians, where each Gaussian carries position, radius, opacity, RGB color, and semantic color across three channels. The system consists of two core processes: tracking and mapping.

### Key Designs

**Multi-Channel Gaussian Representation**: Based on standard Gaussian parameters, a semantic color channel $s_i = [r_i, b_i, g_i]^T$ is added. The 2D semantic map is rendered using the same volume rendering formula as color and depth.

**Semantic-Guided Keyframe Selection**: A two-level screening strategy: (1) Geometric overlap ratio filtering: projected sampled Gaussians onto the keyframe view to calculate the overlap ratio $\eta$, filtering out those below the threshold; (2) Semantic filtering: filtering out keyframes with excessively high semantic map mIoU, prioritizing keyframes from different perspectives. An uncertainty weight based on timestamps $\mathcal{U}(t) = e^{-\tau t}$ is introduced.

**Multi-Channel Joint Optimization**: The tracking loss simultaneously contains three channels: depth L1, color L1, and semantic L1. The mapping loss utilizes a weighted SSIM loss to process color and semantic images.

### Loss & Training

- **Tracking Loss**: $\mathcal{L}_{tracking} = \lambda_D|D^{GT} - D| + \lambda_C|C^{GT} - C| + \lambda_S|S^{GT} - S|$
- **Mapping Loss**: $\mathcal{L}_{mapping} = \mathcal{U}_t(\lambda_D|D^{GT} - D| + \lambda_C\mathcal{L}_C + \lambda_S\mathcal{L}_S)$
- where both $\mathcal{L}_C$ and $\mathcal{L}_S$ adopt a hybrid loss of L1+SSIM.

## Key Experimental Results

### Main Results

Comparison of rendering quality on the Replica dataset (average of 8 scenes):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Depth L1 (cm)↓ | ATE RMSE (cm)↓ |
|------|-------|-------|--------|----------------|----------------|
| NICE-SLAM | 24.42 | 0.809 | 0.233 | 1.903 | 2.503 |
| Co-SLAM | 30.24 | 0.939 | 0.252 | 1.513 | 1.059 |
| ESLAM | 29.08 | 0.929 | 0.336 | 1.180 | 0.630 |
| SplaTAM | 33.98 | 0.969 | 0.099 | 0.525 | 0.454 |
| **SGS-SLAM** | **34.66** | **0.973** | **0.096** | **0.356** | **0.412** |

### Semantic Segmentation Results

Semantic segmentation accuracy on the Replica dataset (mIoU%):

| Method | Average mIoU↑ | Room0 | Room1 | Room2 | Office0 |
|------|-----------|-------|-------|-------|---------|
| NIDS-SLAM | 82.37 | 82.45 | 84.08 | 76.99 | 85.94 |
| DNS-SLAM | 84.77 | 88.32 | 84.90 | 81.20 | 84.66 |
| SNI-SLAM | 87.41 | 88.42 | 87.43 | 86.16 | 87.63 |
| **SGS-SLAM** | **>90** | - | - | - | - |

SGS-SLAM outperforms all NeRF-based semantic SLAM methods by more than 10% in semantic segmentation.

### Key Findings

- The depth L1 error is reduced by approximately 32% compared to SplaTAM (0.356 vs 0.525), showing that semantic information facilitates geometric reconstruction.
- The multi-channel optimization strategy allows tracking and mapping to mutually benefit each other.
- The explicit Gaussian representation naturally supports object-level scene editing operations.

## Highlights & Insights

1. **Semantic feature loss** effectively addresses the deficiencies of traditional depth and color losses in object optimization.
2. Appending semantic channels directly to Gaussians is highly elegant, avoiding the complex multi-stage model designs in NeRF methods.
3. The semantic-guided keyframe selection strategy effectively prevents reconstruction errors caused by accumulated errors.

## Limitations & Future Work

- Reliance on 2D semantic priors (provided by datasets or off-the-shelf models).
- The assumption of isotropic Gaussians may limit the representative capability of complex scenes.
- Scalability has not been validated in large-scale scenes.

## Related Work & Insights

Integrating semantic information into Gaussian SLAM in this manner is simple yet effective, laying the foundation for future work (such as dynamic-scene semantic SLAM). The two-level keyframe selection strategy is highly instructive.

## Rating

- Novelty: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM](i2-slam_inverting_imaging_process_for_robust_photorealistic_dense_slam.md)
- [\[ECCV 2024\] CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field](cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)
- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [\[CVPR 2026\] ODGS-SLAM: Omnidirectional Gaussian Splatting SLAM](../../CVPR2026/3d_vision/odgs-slam_omnidirectional_gaussian_splatting_slam.md)
- [\[CVPR 2025\] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](../../CVPR2025/3d_vision/wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)

</div>

<!-- RELATED:END -->
