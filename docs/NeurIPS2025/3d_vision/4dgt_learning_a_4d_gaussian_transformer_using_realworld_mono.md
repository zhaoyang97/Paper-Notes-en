---
title: >-
  [Paper Note] 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos
description: >-
  [NeurIPS 2025 (Spotlight)][3D Vision][4D Gaussian Splatting] This paper proposes 4DGT — a 4D Gaussian-based Transformer model trained entirely on real-world monocular posed videos that reconstructs dynamic scenes in seconds via feed-forward inference, significantly outperforming comparable feed-forward networks while achieving accuracy on par with optimization-based methods.
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "Transformer"
  - "Dynamic Scene Reconstruction"
  - "Feed-Forward Inference"
  - "Monocular Video"
date: 2026-05-08
content_hash: cb2050b19fe8f70a
---

# 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos

**Conference**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2506.08015](https://arxiv.org/abs/2506.08015)  
**Code**: [https://github.com/facebookresearch/4dgt](https://github.com/facebookresearch/4dgt)  
**Area**: Others  
**Keywords**: 4D Gaussian Splatting, Transformer, Dynamic Scene Reconstruction, Feed-Forward Inference, Monocular Video  

## TL;DR
This paper proposes 4DGT — a 4D Gaussian-based Transformer model trained entirely on real-world monocular posed videos that reconstructs dynamic scenes in seconds via feed-forward inference, significantly outperforming comparable feed-forward networks while achieving accuracy on par with optimization-based methods.

## Background & Motivation

### Root Cause

**Background**: Dynamic 3D scene reconstruction is a core task in computer vision. Existing approaches fall into two main categories: (1) **Optimization-based methods** such as Shape-of-Motion, which require hours of per-scene optimization and cannot scale to long videos or real-time applications; (2) **Feed-forward methods** such as L4GM and StaticLRM, which offer fast inference but typically require multi-view inputs or synthetic training data, limiting their effectiveness on complex real-world dynamic scenes.

**Limitations of Prior Work**: Existing feed-forward methods do not adequately model the temporal dimension of scene elements — static backgrounds and dynamic foregrounds have different temporal lifespans, objects may appear and disappear, and traditional 3D Gaussian representations lack temporal modeling capacity. Furthermore, as the number of input frames grows, the spatial-temporal token count explodes, creating memory and efficiency bottlenecks during training and inference.

### Paper Goals

**Goal**: How to design a feed-forward dynamic scene reconstruction model that: (1) can be trained on real-world monocular videos without relying on synthetic data; (2) uniformly models static and dynamic components with their varying temporal lifespans; and (3) remains efficient when processing long video sequences?

## Method

### Overall Architecture
4DGT takes as input a sequence of monocular RGB frames with camera poses and timestamps (default: 64 frames) and outputs a set of 4D Gaussian primitives that can be rendered at arbitrary time steps and viewpoints.

The overall pipeline consists of two training stages and a sliding-window inference mechanism:
1. **Stage 1**: Trained on EgoExo4D data at lower spatial resolution to predict pixel-aligned 4D Gaussian parameters.
2. **Stage 2**: Using the opacity histogram from the Stage 1 model to prune most inactive Gaussians, then increasing spatial and temporal token sampling density for training at higher resolution.
3. **Inference**: Processes long videos using a sliding window of 64 frames, producing consistent 4D Gaussians via feed-forward prediction.

### Key Designs

1. **4D Gaussian Representation**: Each Gaussian primitive extends beyond traditional 3DGS attributes (position xyz, scale, rotation, opacity, rgb) with additional temporal parameters:

    - `t`: temporal position (at which time point the Gaussian is most active)
    - `cov_t`: temporal covariance (duration of the Gaussian's temporal lifespan)
    - `ms3`: marginal velocity (describes the Gaussian's spatial motion direction and rate over time; supports multi-degree modeling)
    - `omega`: angular velocity (rotational change of the Gaussian over time)
    - `dxyzt`: fine-grained residual correction in position and time

   During rendering, the contribution weight of each Gaussian to the current frame is determined by computing the marginal probability $p(t|\mu_t, \sigma_t)$ between the target time $t$ and the Gaussian's temporal center $\mu_t$ and temporal covariance $\sigma_t$. Static objects have large $\sigma_t$ (always visible), while dynamic objects have small $\sigma_t$ (visible only during specific time intervals), naturally unifying static and dynamic modeling.

2. **DINOv2 + Transformer Encoder (TLoD)**:

    - Uses a frozen DINOv2 ViT-B/14 as the visual feature extraction backbone.
    - RGB images, Plücker ray coordinates (encoding camera poses and pixel directions), and timestamps are concatenated and patchified into spatial-temporal tokens.
    - DINOv2 features are concatenated with Plücker and timestamp features and fed into a Transformer.
    - 12-layer Self-Attention Blocks perform global spatial-temporal feature fusion.
    - A decoding head maps tokens to 4D Gaussian parameters via MLP.

3. **Density Control Strategy (Magic Filter)**:
   This is a core training technique. After Stage 1 training, each 14×14 patch predicts a Gaussian for each of its 196 pixels, resulting in a massive total count. To handle larger spatial-temporal inputs in Stage 2 while maintaining rendering efficiency, an opacity-based adaptive pruning scheme is proposed:
    - **Patch Sorting**: Within each 14×14 patch, Gaussians are sorted by predicted opacity and only the top-k (approximately 10) most active Gaussians are retained, pruning roughly 95% of inactive ones.
    - This substantially reduces the number of Gaussians during rendering, enabling increased spatial and temporal token resolution in Stage 2.
    - The strategy is applied during the forward pass at training time, ensuring only high-contribution Gaussians participate in rendering and gradient computation.

4. **Temporal Level of Detail (TLoD)**:
   Supports multi-level processing (when n_levels > 1), divided into global, regional, and detail levels:
    - **Global level**: Processes temporally and spatially downsampled inputs to capture overall scene structure.
    - **Detail level**: Processes single or few frames at original resolution to capture high-frequency details.
    - 4D Gaussian parameters from all levels are concatenated and jointly used for rendering.

### Loss & Training
- Training data comes from large-scale real-world posed monocular video datasets (e.g., EgoExo4D); frames are sampled at varying temporal granularities during training, with all images used for supervision.
- Standard photometric loss (rendering loss) is used, including RGB reconstruction loss.
- Stage 1 trains on EgoExo4D using all pixel-aligned Gaussians.
- Stage 2 continues training at higher resolution after pruning, with increased spatial and temporal token sampling density.
- Total model parameter size is approximately 14.5 GB (full model); the Stage 1 model is approximately 4.85 GB.
- Inference uses bfloat16 precision and requires at least 16 GB of GPU memory.

## Key Experimental Results

| Dataset | Baseline | Advantage of Ours |
|--------|----------|----------|
| DyCheck (cross-domain) | vs Shape-of-Motion (optimization-based) | Achieves comparable accuracy; inference time reduced from hours to seconds |
| DyCheck (cross-domain) | vs L4GM, StaticLRM (feed-forward) | Significantly outperforms these feed-forward methods |
| Ego-Exo4D | vs other methods | Best performance on real-world video |
| AEA, ADT, HOT3D, Nymeria | Qualitative evaluation | High-quality dynamic reconstruction |
| TUM Dynamics | vs other methods | Strong cross-domain generalization |

Note: The HTML version of the paper is unavailable; specific quantitative metrics (PSNR/SSIM/LPIPS) could not be retrieved from current sources. Based on the abstract and code, the core conclusion is that 4DGT "significantly outperforms feed-forward methods and is comparable to optimization-based methods."

### Ablation Study
- **Magic Filter (density control)**: Removing density control causes the number of Gaussians to explode, making it infeasible to handle long spatial-temporal inputs.
- **4D temporal parameters**: `cov_t` (temporal covariance) and `ms3` (motion velocity) are critical for modeling dynamic scenes; removing them leads to confusion between static and dynamic objects.
- **Two-stage training**: Using only Stage 1 dense Gaussians yields inferior rendering quality and efficiency compared to the two-stage scheme.
- **TLoD multi-level processing**: Multi-level processing helps balance global consistency with local detail.

## Highlights & Insights
- **Unified modeling via 4D Gaussian representation**: Through temporal covariance `cov_t` and temporal position `t`, static and dynamic objects are naturally unified — static objects have long temporal lifespans while dynamic ones have short lifespans — without requiring explicit separation.
- **Density control as a key innovation**: The Magic Filter's opacity-based pruning is conceptually simple yet highly effective; it reduces rendering overhead and frees memory for larger spatial-temporal inputs, serving as the core enabling technique for feed-forward inference over 64-frame sequences.
- **Real-world training data**: The model is trained entirely on real-world monocular videos without relying on synthetic data, resulting in strong generalization ability.
- **Systematic engineering**: The complete system — from data processing and model architecture to the rendering pipeline — is fully open-sourced and accompanied by an interactive viewer.

## Limitations & Future Work
- **Geometric accuracy ceiling with monocular input**: Monocular video lacks multi-view constraints, making it difficult to recover precise geometric depth, especially in occluded regions.
- **Limitations of the 64-frame window**: Although sliding-window inference is supported, no explicit consistency constraint is imposed between 4D Gaussians across windows, potentially causing inter-window discontinuities in long videos.
- **Training data dependency**: Large-scale monocular video with accurate poses is required; errors in pose estimation degrade reconstruction quality.
- **Non-rigid motion modeling**: The linear motion assumption in 4D Gaussians (`ms3` velocity + `omega` angular velocity) has limited capacity to model nonlinear or highly complex motions.
- **Rendering efficiency**: While substantially faster than optimization-based methods, the 14.5 GB model still has room for optimization in real-time applications.

## Related Work & Insights
- **vs Shape-of-Motion**: SoM is a per-scene optimization method with high accuracy but slow speed (hours). 4DGT achieves second-level feed-forward inference with better generalization, though it may trade some peak per-scene accuracy.
- **vs L4GM**: L4GM is also a feed-forward 3DGS model but primarily targets synthetic multi-view inputs. 4DGT is substantially stronger on real-world monocular video.
- **vs StaticLRM**: StaticLRM handles only static scenes without temporal modeling. 4DGT extends support to dynamic scenes through 4D Gaussians.
- **vs 4DSTR/4DGC and other 4D methods**: 4DGT's core advantage lies in feed-forward inference combined with real-world training, rather than relying on synthetic data or per-scene optimization.

## Related Work & Insights
- **The density control idea of Magic Filter** can be transferred to other token-heavy vision tasks — pruning inactive tokens based on activation magnitude or importance is a general efficiency-enhancing strategy.
- The architectural pattern of frozen DINOv2 backbone combined with lightweight Transformer fusion continues to demonstrate effectiveness in feed-forward 3D generation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of 4D Gaussian Transformer and density control strategy is innovative, though individual components (4DGS, Vision Transformer, DINOv2 backbone) are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple real-world datasets with comparisons to several baselines and ablation studies; specific quantitative results could not be fully retrieved due to unavailability of the HTML version.
- Writing Quality: ⭐⭐⭐⭐ A Spotlight paper from Meta RL Research and Zhejiang University, with open-sourced code and a complete project page demonstrating systematic presentation.
- Value: ⭐⭐⭐⭐⭐ Feed-forward reconstruction of dynamic scenes is a high-value direction; reducing inference time from hours to seconds represents a significant practical advancement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos](../../CVPR2026/3d_vision/neoverse_enhancing_4d_world_model_with_in-the-wild_monocular_videos.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[ICLR 2026\] Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos](../../ICLR2026/3d_vision/mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)

</div>

<!-- RELATED:END -->
