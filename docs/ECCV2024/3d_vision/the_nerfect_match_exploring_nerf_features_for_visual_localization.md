---
title: >-
  [Paper Note] The NeRFect Match: Exploring NeRF Features for Visual Localization
description: >-
  [ECCV 2024][3D Vision][Visual Localization] Proposes NeRFMatch, which explores the potential of internal NeRF features as 3D descriptors and establishes an attention-based 2D-3D matching network. It achieves competitive localization performance on Cambridge Landmarks, validating the feasibility of NeRF as a scene representation for localization.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Visual Localization"
  - "NeRF Features"
  - "2D-3D Matching"
  - "Pose Estimation"
  - "Scene Representation"
date: 2026-05-08
content_hash: ca1656a9527b3f65
---

# The NeRFect Match: Exploring NeRF Features for Visual Localization

**Conference**: ECCV 2024  
**arXiv**: [2403.09577](https://arxiv.org/abs/2403.09577)  
**Code**: [Project Page](https://nerfmatch.github.io)  
**Area**: 3D Vision  
**Keywords**: Visual Localization, NeRF Features, 2D-3D Matching, Pose Estimation, Scene Representation

## TL;DR

Proposes NeRFMatch, which explores the potential of internal NeRF features as 3D descriptors and establishes an attention-based 2D-3D matching network. It achieves competitive localization performance on Cambridge Landmarks, validating the feasibility of NeRF as a scene representation for localization.

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: Visual localization aims to determine the camera pose of a query image in a 3D environment. Mainstream representations include: point clouds + descriptors (HLoc), meshes (MeshLoc), and learned implicit representations (APR/SCR). As a compact 3D scene representation (Mip-NeRF is only 5.28MB), NeRF has been used for data augmentation, auxiliary supervision, and pose refinement, but the potential of its internal features for direct 2D-3D matching remains under-explored. Prior works (CrossFire, NeRF-Loc) require joint training with the matching task and cannot utilize pre-trained NeRFs.

## Method

### Overall Architecture

A three-step pipeline: (1) Image retrieval to find the nearest reference pose; (2) Render 3D points and features from the reference view using NeRF, and establish 2D-3D correspondences using NeRFMatch to compute the pose; (3) Optional iterative pose refinement.

### Key Designs

**NeRF Feature Rendering**: Given a 3D encoder $\Theta_x$ (with $L$ layers) of NeRF, the $j$-th layer feature is extracted as $f^j = \Theta_x^j \circ \cdots \circ \Theta_x^1(P_x(X))$. The surface points $\hat{X}(r) = \sum w_i X_i$ and corresponding descriptors $\hat{F}^j(r) = \sum w_i f_i^j$ are aggregated through volume rendering. **Key Insight**: This feature only depends on the 3D coordinates and is independent of the viewpoint, making it naturally suitable for cross-view matching.

**NeRFMatch-Mini**: A lightweight version where a CNN encoder extracts $8\times$ downsampled image features and directly performs dual-softmax matching with NeRF features, without requiring a learnable matching module.

**NeRFMatch (Full)**: Incorporates coarse-to-fine matching with self-attention and cross-attention modules. The coarse matching layer uses a shared self-attention module (to pull the feature spaces of the two domains closer) and explicitly concatenates the positional encodings of 3D coordinates to enhance spatial awareness, followed by cross-attention for cross-domain interaction. The fine matching layer performs sub-pixel level matching via heatmap regression within a high-resolution local window.

**Pose Refinement**: Two schemes: (1) Iterative matching refinement: re-matching using the estimated pose as the new reference; (2) iNeRF-style photometric optimization followed by re-matching.

### Loss & Training

- **Coarse Matching Loss**: $L_c = -\frac{1}{M_{gt}} \sum \log(S(i,j))$ (log loss to maximize the dual-softmax probability of the GT positions)
- **Fine Matching Loss**: $L_f = \frac{1}{M_f} \sum \frac{1}{\sigma^2(i)} \|\tilde{x}_j - x_j\|_2$ (variance-weighted pixel distance loss)

## Key Experimental Results

### Outdoor Localization on Cambridge Landmarks

Median pose error (cm / °):

### Main Results

| Method | Scene Representation | Kings | Hospital | Shop | StMary | Court | Average |
|------|----------|-------|----------|------|--------|-------|------|
| DSAC* | SCR Network | 15/0.3 | 21/0.4 | 5/0.3 | 13/0.4 | 49/0.3 | 20.6/0.3 |
| ACE | SCR Network | 28/0.4 | 31/0.6 | 5/0.3 | 18/0.6 | 43/0.2 | 25/0.4 |
| HLoc | 3D+RGB | 12/0.2 | 15/0.3 | 4/0.2 | 7/0.2 | 16/0.1 | 10.8/0.2 |
| NeRFLoc | NeRF+RGBD | 11/0.2 | 18/0.4 | 4/0.2 | 7/0.2 | 25/0.1 | 13/0.2 |
| NeRFMatch-Mini | NeRF+RGB | 19.0/0.3 | 30.2/0.6 | 10.3/0.5 | 11.3/0.4 | 29.1/0.2 | 20.0/0.4 |
| **NeRFMatch** | **NeRF+RGB** | **13.0/0.2** | **19.4/0.4** | **8.5/0.4** | **7.9/0.3** | **17.5/0.1** | **13.3/0.3** |

### Model Efficiency


### Ablation Study

| Component | Size | Inference Time |
|------|------|----------|
| Mip-NeRF Scene | 5.28 MB | - |
| NeRF Feature Rendering (3600 points)| - | 141 ms |
| NeRFMatch-Mini | 42.8 MB | 37 ms |
| NeRFMatch | 50.4 MB | 157 ms |

### Key Findings

- Under the condition of only using NeRF+RGB (without depth), NeRFMatch achieves an average error of 13.3cm, which is close to NeRFLoc (13.0cm) that requires RGBD.
- Using pure NeRF features (without RGB from image retrieval) can still achieve 14.6cm, proving that the internal features of NeRF themselves are high-quality 3D descriptors.
- Significant improvement is obtained on the first iteration after iterative refinement, with diminishing returns for further iterations.
- Performance on the indoor 7-Scenes is relatively weak, pointing out directions for future improvement.

## Highlights & Insights

1. **Core Discovery**: The internal features learned by NeRF through view synthesis are naturally viewpoint-invariant, allowing them to serve as 3D descriptors without additional training.
2. Allows utilizing the NeRF model for localization without modifications or retraining, thereby directly benefiting from continuous advancements in NeRF research.
3. The Mini version demonstrates that reasonable localization can be achieved without learning matching functions, but solely by learning good feature representations.

## Limitations & Future Work

- Poor performance in indoor scenes due to low texture and motion blur.
- Dependent on image retrieval to provide the initial pose range.
- The 141ms latency of NeRF feature rendering limits real-time applications.

## Related Work & Insights

Unlike CrossFire and NeRF-Loc, this work treats NeRF as a black-box scene representation rather than a module that needs joint training, making it more versatile. The discovery of internal NeRF features provides significant inspiration for understanding what NeRF learns.

## Rating

- Novelty: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Invertible Neural Warp for NeRF](invertible_neural_warp_for_nerf.md)
- [\[ECCV 2024\] SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs](scenegraphloc_cross-modal_coarse_visual_localization_on_3d_scene_graphs.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)
- [\[ICLR 2026\] A Scene is Worth a Thousand Features: Feed-Forward Camera Localization from a Collection of Image Features](../../ICLR2026/3d_vision/a_scene_is_worth_a_thousand_features_feed-forward_camera_localization_from_a_col.md)
- [\[ECCV 2024\] Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions](deblur_e-nerf_nerf_from_motion-blurred_events_under_high-speed_or_low-light_cond.md)

</div>

<!-- RELATED:END -->
