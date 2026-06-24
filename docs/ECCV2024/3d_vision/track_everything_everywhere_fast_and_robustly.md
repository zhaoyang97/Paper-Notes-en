---
title: >-
  [Paper Note] Track Everything Everywhere Fast and Robustly
description: >-
  [ECCV 2024][3D Vision] This paper proposes an efficient and robust test-time optimization method for pixel tracking. By introducing the CaDeX++ invertible deformation network, monocular depth priors, and DINOv2 long-term semantic consistency, the method accelerates the training speed by over 10 times while significantly improving tracking accuracy and robustness.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: a41b17e82b80860d
---

# Track Everything Everywhere Fast and Robustly

**Conference**: ECCV 2024  
**arXiv**: [2403.17931](https://arxiv.org/abs/2403.17931)  
**Code**: [Project Page](https://timsong412.github.io/FastOmniTrack/)  
**Area**: 3D Vision

## TL;DR

This paper proposes an efficient and robust test-time optimization method for pixel tracking. By introducing the CaDeX++ invertible deformation network, monocular depth priors, and DINOv2 long-term semantic consistency, the method accelerates the training speed by over 10 times while significantly improving tracking accuracy and robustness.

## Background & Motivation

### Background

**Background**: OmniMotion is the current state-of-the-art (SOTA) optimization-based tracking method. However, it suffers from three main limitations: excessively long training times, instability in convergence due to high sensitivity to random seeds, and a lack of long-term association as it only fits short-term optical flow.

### Limitations of Prior Work

**Limitations of Prior Work**: Feature-based methods (such as SIFT) produce sparse matches, while optical flow methods struggle to handle long-range motion and occlusions.

### Key Challenge

**Key Challenge**: Although feed-forward methods (such as TAPIR and CoTracker) are fast, they generalize poorly in textureless scenes.

### Mechanism

**Mechanism**: **Core Problem**: OmniMotion relies on volume rendering for geometric reconstruction, which is computationally expensive and yields poor triangulation accuracy in small-baseline videos.

## Method

### Overall Architecture

The query pixel is first unprojected (lifted) to 3D space using an optimizable depth map, mapped to a shared canonical space via the CaDeX++ invertible deformation field, and then mapped to the target frame to complete the tracking. Short-term RAFT optical flow and long-term DINOv2 semantic correspondences are employed as the optimization targets.

### Key Designs

**CaDeX++ Invertible Deformation Network**:
- The global MLP latent code is decomposed into local spatio-temporal feature grids (multi-resolution lookup), inspired by Instant-NGP and TensoRF.
- Monotonic piecewise-linear functions (with $B$ control points) are used to replace the original affine transformations, enhancing single-step expressiveness while preserving invertibility.
- The network is heavily lightweighted, significantly accelerating training.

**Depth Prior**: Monocular metric depth from ZoeDepth is utilized to initialize the optimizable depth map for each frame, eliminating the inefficient NeRF volume rendering process. The tracking function is simplified to: back-projection $\rightarrow$ deformation $\rightarrow$ projection.

**DINOv2 Long-term Semantic Correspondence**: Sparse yet reliable long-range correspondences are extracted via mutual nearest neighbor matching and self-similarity filtering, compensating for the limitations of short-term optical flow.

### Loss & Training

$\text{Total Loss} = \text{Pixel Position Loss} (L_1) + \text{Depth Consistency Loss} + \text{Depth Regularization Loss}$

Depth consistency constrains the depth of the deformed 3D points to be consistent with the depth map of the target frame; depth regularization ensures that the optimized depth does not deviate too far from the initial prediction of ZoeDepth.

## Key Experimental Results

### Main Results

| Method | Category | AJ↑ | $\delta_{\text{avg}}$↑ | OA↑ | TC↓ |
|------|------|-----|--------|-----|-----|
| CoTracker | Feed-forward | 65.1 | 79.0 | 89.4 | 0.93 |
| TAPIR | Feed-forward | 59.8 | 72.3 | 87.6 | - |
| OmniMotion | Optimization | 51.7 | 67.5 | 85.3 | 0.74 |
| **Ours** | Optimization | **59.4** | **77.4** | **85.9** | **0.68** |

On the RGB-Stacking dataset, the proposed method achieves an OA of 93.6%, outperforming CoTracker's 85.4%.

### Ablation Study

| Configuration | AJ↑ | $\delta_{\text{avg}}$↑ | OA↑ | TC↓ |
|------|-----|--------|-----|-----|
| No depth | 42.0 | 56.8 | 73.3 | 1.42 |
| No long-term | 45.6 | 61.3 | 75.5 | 1.32 |
| No CaDeX++ | 48.2 | 65.4 | 80.1 | 0.97 |
| **Full** | **48.6** | **65.7** | **80.1** | 1.14 |

### Key Findings

- Training speed is accelerated by over 10 times (on DAVIS) and 5 times (on RGB-Stacking), with more stable convergence.
- The depth prior contributes the most: removing it leads to a 6.6 drop in AJ and a 6.8 drop in OA.
- Long-term semantic supervision significantly improves trajectory accuracy, especially when dealing with frequent occlusions.
- It outperforms feed-forward methods on textureless synthetic videos, as optimization-based methods do not heavily rely on visual texture features.
- The consistency between tracking trajectories and optical flow (DAG metric) is significantly superior to CoTracker.

## Highlights & Insights

- Introducing the local representation concept of Instant-NGP into the invertible deformation field represents a significant improvement over the NVP architecture.
- Replacing NeRF volume rendering with an optimizable depth map achieves a win-win in both efficiency and accuracy.
- The long-range semantic correspondences provided by DINOv2 fill the information gap left by short-term optical flow.
- The convergence robustness is far superior to OmniMotion, with a drastically reduced variance of results across different random seeds.
- The design of the method reflects the engineering wisdom of "correct inductive bias > more optimization time": the depth prior provides a good initialization, DINOv2 provides global constraints, and CaDeX++ provides highly efficient parameterization.

## Further Comparison with CoTracker

CoTracker has the best performance among feed-forward methods, utilizing cross-trajectory attention to achieve a global receptive field. However, on textureless synthetic videos (RGB-Stacking), CoTracker's OA is only 85.4%, much lower than the 93.6% of this method. The DAG metric (the degree of inconsistency between trajectory and optical flow) shows that this method achieves 14.9 vs. CoTracker's 40.3 in the *car-turn* scene, and 12.8 vs. 32.5 in the *plane* scene, indicating that trajectories generated by optimization methods are more consistent with local optical flow.

In terms of convergence robustness, OmniMotion exhibits high variance across different random seeds and can even diverge completely. In contrast, the proposed method significantly reduces sensitivity to initialization through depth prior initialization and DINOv2 semantic constraints.

## Limitations & Future Work

- Test-time optimization is still required, preventing real-time applications.
- Occlusion reasoning relies on depth thresholds, which may fail in extreme scenarios.
- The final tracking performance is sensitive to the quality of the depth prior.
- Although the piecewise-linear approximation in CaDeX++ enhances expressiveness, the number of control points $B$ requires hyperparameter tuning.

## Rating

- Novelty: ⭐⭐⭐⭐
- Effectiveness: ⭐⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐
- Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)

</div>

<!-- RELATED:END -->
