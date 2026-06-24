---
title: >-
  [Paper Note] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments
description: >-
  [CVPR 2025][3D Vision][articulated objects] Constructs a hierarchical semantic feature field based on 3DGS, integrating semantic information from CLIP, SAM, and DINOv2 to achieve interactive affordance prediction and cross-state motion parameter recovery for articulated objects, supporting complex indoor scenes with arbitrary categories and multiple movable parts.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "articulated objects"
  - "affordance"
  - "3D Gaussian Splatting"
  - "foundation models"
  - "motion recovery"
  - "scene understanding"
date: 2026-05-08
content_hash: 9521ecdebdabeaf4
---

# IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments

**Conference**: CVPR 2025  
**arXiv**: [2504.06827](https://arxiv.org/abs/2504.06827)  
**Code**: [https://lulusindazc.github.io/IAAOproject/](https://lulusindazc.github.io/IAAOproject/)  
**Area**: 3D Vision  
**Keywords**: articulated objects, affordance, 3D Gaussian Splatting, foundation models, motion recovery, scene understanding

## TL;DR

Constructs a hierarchical semantic feature field based on 3DGS, integrating semantic information from CLIP, SAM, and DINOv2 to achieve interactive affordance prediction and cross-state motion parameter recovery for articulated objects, supporting complex indoor scenes with arbitrary categories and multiple movable parts.

## Background & Motivation

**Background**: Understanding the shape, pose, and joint motion of articulated objects (such as cabinet doors, scissors, etc.) is a core requirement for robotics and AR/VR. Methods like Ditto and PARIS construct digital twins through observations of two joint states.

**Limitations of Prior Work**: (1) Ditto requires category-specific pre-training and has poor generalization capability; (2) PARIS is based on NeRF implicit representation, making it sensitive to initialization and unstable; (3) Existing methods assume that the static parts are aligned and camera poses are known, which is impractical for real-world scenes; (4) There is a lack of fine-grained affordance detection capability (e.g., identifying small functional elements like door handles).

**Key Challenge**: Agents need to simultaneously understand the "what" (semantics), "where" (localization), and "how" (interaction method) of objects, whereas existing methods typically focus on only a single aspect and are restricted to simple two-part objects.

**Key Insight**: Utilizing explicit 3DGS representation and semantic distillation from large foundation models (CLIP, SAM, DINOv2) to construct interactive 3D semantic fields.

## Method

### Overall Architecture

Three-stage pipeline:
1. **Semantic Scene Reconstruction**: Reconstructs 3DGS with hierarchical features for two joint states separately.
2. **Affordance and Motion Prediction**: Performs object/part-level queries directly on the 3D Gaussian primitives to estimate global transformations and local joint parameters.
3. **Scene Fusion**: Merges the 3DGS models of the two states based on estimated transformations to fill in occluded areas.

### Key Designs

**1. View-Consistent Mask Clustering and Hierarchical Feature Field**
- **Function**: Generates view-wise class-agnostic masks using SAM, and associates 2D masks into 3D consistent instances through mask graph clustering; simultaneously distills MaskCLIP and DINOv2 features using a low-dimensional feature field and a decoder.
- **Mechanism**: Constructs a mask graph $\mathcal{G}^t_0 = (\mathcal{V}^t_0, \mathcal{E}^t_0)$, where each node is a 2D mask, and clusters them through feature similarity to obtain cross-view consistent 3D instances. The decoder $\mathcal{D}$ projects the rendered features into three branches (instance/part-level CLIP + DINO) using a small MLP.
- **Design Motivation**: Directly embedding high-dimensional 2D features incurs significant memory overhead; mask graph clustering resolves issues such as cross-view inconsistency and over-segmentation in SAM.
- **Loss Function**: Feature distillation loss $\mathcal{L}_{feat} = \|\mathcal{D}(\hat{F}^t(I^t_i)) - F^t(I^t_i)\|_2^2$ + cross-entropy loss of the label field $\mathcal{L}_{label}$.

**2. Global-Local Motion Recovery**
- **Function**: First uses GeoTransformer to estimate a coarse global alignment $\xi_g^t = (s_g^t, R_g^t, T_g^t)$ from the 3D Gaussians of static parts, then estimates local transformations $\xi_o^t = (R_o^t, T_o^t)$ for each articulated part.
- **Mechanism**: Computes 3D-to-2D correspondences via DINO features: computes the feature similarity matrix $\alpha_{p \to o}$ between sampled Gaussian points and target-state mask pixels, applies softmax to obtain weights $\beta_{p \to o}$, and takes a weighted sum to obtain the corresponding 2D pixel positions.
- **Key Losses**:
    - Matching loss: $\mathcal{L}_{match} = \|\pi^{t'}_n(p^{t \to t'}) - s^{t'}_{p \to n}(I^{t'}_n)\|_2^2$
    - Mask feature loss: $\mathcal{L}_{mask}$ (coarse guided transformation)
    - RGB consistency loss: $\mathcal{L}_{rgb}$
- **Design Motivation**: Point clouds from 3DGS are sparse and noisy, making direct 3D-3D registration difficult; indirectly establishing correspondences via 3D-to-2D projection and feature matching is more robust.

**3. Functional Affordance Prediction**
- **Function**: Given a task description (e.g., "open the drawer"), encodes it with the CLIP text encoder and computes similarity with the feature field to localize the corresponding functional region (e.g., the drawer handle).
- **Mechanism**: Explicit 3DGS primitives allow direct feature queries in 3D space without rendering. Supports multiple prompt modalities, such as language, points, and masks.
- **Design Motivation**: Unlike NeRF, which requires rendering the entire image to extract features, explicit Gaussian primitives can be operated on directly, achieving efficient object/part-level localization.

### Loss & Training

Total loss: $\mathcal{L} = \lambda_{cons}(\mathcal{L}_{rgb} + \mathcal{L}_{mask} + \mathcal{L}_{label}) + \lambda_{match}\mathcal{L}_{match}$

## Key Experimental Results

### Main Results — Articulation Parameter Estimation

**PARIS Two-part Dataset (10 synthetic + 2 real objects)**:

| Method | Axis Ang↓ | Axis Pos↓ | Part Motion↓ |
|---|---|---|---|
| PARIS* | 11.14 | 1.79 | 50.52 |
| DigitalTwinArt | 0.14 | 0.01 | 0.12 |
| **IAAO** | **0.11** | **0.01** | **0.10** |

Real-world scenes:

| Method | Axis Ang↓ |
|---|---|
| PARIS* | 16.00 |
| DigitalTwinArt | 10.11 |
| **IAAO** | **8.86** |

### Ablation Study

| Setting | Axis Ang↓ | Axis Pos↓ | Part Motion↓ |
|---|---|---|---|
| w/o semantic association | 0.25 | 0.03 | 0.20 |
| w/o mask loss | 0.18 | 0.02 | 0.15 |
| w/o match loss | 0.32 | 0.05 | 0.28 |
| Full IAAO | **0.11** | **0.01** | **0.10** |

### Key Findings

1. **Fully Outperforms SOTA**: Outperforms DigitalTwinArt (the current strongest baseline) on all metrics, with a joint axis angle error of only 0.11°.
2. **Strong Generalization**: Can handle unseen object categories without relying on pre-training, and is equally effective for multi-part objects.
3. **Indoor Scene Extension**: Successfully works in OmniSim indoor scenes, supporting complex backgrounds and multiple articulated objects.
4. **DINOv2 Features are Crucial for Motion Recovery**: DINO's dense features provide fine-grained 3D-2D correspondences, whereas CLIP features are used for coarse mask-level association.

## Highlights & Insights

- The paradigm of explicit 3DGS combined with foundation model distillation is highly versatile, allowing direct queries on 3D primitives instead of post-processing after rendering.
- The global-to-local hierarchical transformation estimation strategy elegantly bypasses the common challenge of misaligned states.
- Supports arbitrary categories and any number of movable parts, which is far superior to methods requiring category-specific pre-training.
- The idea of using mask graph clustering to solve the cross-view inconsistency of SAM is highly inspiring.

## Limitations & Future Work

- Currently, only revolute and prismatic joint types are supported, failing to cover more complex motions (such as non-rigid deformation).
- Relies on multi-view images from two known joint states, which is costly to acquire.
- The global alignment of GeoTransformer may fail under severe partial occlusion.
- The 3DGS reconstruction quality is sensitive to SfM initialization, and may degrade in low-texture regions.
- Inference speed and real-time interaction capabilities have not yet been discussed.

## Related Work & Insights

- **PARIS**: NeRF-based reconstruction of two-part articulated objects, which does not require pre-training but is sensitive to initialization.
- **DigitalTwinArt**: A two-stage method (reconstructing shape first, then recovering joints), serving as the strongest baseline.
- **GaussianGrouping / SAGA**: 3DGS-based segmentation methods, providing the foundation for mask clustering.
- **Insight**: The explicit point cloud representation of 3DGS is naturally suited for feature fusion and interactive control with foundation models.

## Rating

⭐⭐⭐⭐ — Complete method design, thorough experiments, and achieves SOTA in interactive understanding of articulated objects, although the experimental scale is still dominated by small synthetic scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[CVPR 2025\] GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency](geal_generalizable_3d_affordance_learning_with_cross-modal_consistency.md)
- [\[CVPR 2025\] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](isegman_interactive_segment-and-manipulate_3d_gaussians.md)
- [\[CVPR 2026\] Part$^{2}$GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting](../../CVPR2026/3d_vision/part2gs_part-aware_modeling_of_articulated_objects_using_3d_gaussian_splatting.md)
- [\[CVPR 2025\] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)

</div>

<!-- RELATED:END -->
