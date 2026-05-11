---
title: >-
  [Paper Note] DROID-W: DROID-SLAM in the Wild
description: >-
  [CVPR 2026][3D Vision][SLAM] This paper proposes DROID-W, which introduces uncertainty estimation into differentiable Bundle Adjustment (Uncertainty-aware BA)…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "SLAM"
  - "dynamic scenes"
  - "uncertainty estimation"
  - "bundle adjustment"
  - "DINOv2"
date: 2026-05-08
content_hash: 7acf5f16e56ff763
---

# DROID-W: DROID-SLAM in the Wild

**Conference**: CVPR 2026
**arXiv**: [2603.19076](https://arxiv.org/abs/2603.19076)
**Code**: [MoyangLi00/DROID-W](https://github.com/MoyangLi00/DROID-W.git)
**Area**: 3D Vision
**Keywords**: SLAM, dynamic scenes, uncertainty estimation, bundle adjustment, DINOv2

## TL;DR

This paper proposes DROID-W, which introduces uncertainty estimation into differentiable Bundle Adjustment (Uncertainty-aware BA), combined with a DINOv2-feature-driven dynamic uncertainty update mechanism and monocular depth regularization, enabling robust camera pose estimation and scene reconstruction for DROID-SLAM in highly dynamic in-the-wild scenarios at approximately 10 FPS in real time.

## Background & Motivation

Visual SLAM (Simultaneous Localization and Mapping) is a core technology for robotics, AR/VR, and autonomous driving, with the goal of simultaneously estimating camera poses and constructing the 3D structure of a scene from continuous video frames.

DROID-SLAM is one of the current state-of-the-art deep learning SLAM systems, whose core strength lies in its differentiable Dense Bundle Adjustment (DBA) layer, which achieves excellent accuracy through end-to-end training. However, DROID-SLAM and nearly all classical SLAM methods are built upon a critical assumption:

**The Static World Assumption**: all observable points in the scene are static across different time frames, and their 3D positions do not change over time.

This assumption is severely violated in real-world in-the-wild scenarios:

1. **Pedestrians and vehicles**: large numbers of moving objects in urban scenes break geometric consistency.
2. **Wind-blown foliage and flowing water**: non-rigid motion is ubiquitous in natural environments.
3. **YouTube videos**: internet videos are filled with various dynamic elements that traditional SLAM cannot handle.

Existing methods for handling dynamic scenes fall into two main categories:

- **Semantic segmentation-based methods**: detect and mask object categories likely to move (e.g., pedestrians, vehicles) in advance, but rely on predefined dynamic category priors and cannot handle unexpectedly moving objects.
- **Neural implicit map-based methods** (e.g., RoDynRF, DynaMoN): jointly model static and dynamic regions using NeRF, achieving high accuracy but at extreme computational cost, precluding real-time operation.

Core motivation: **Can BA be made to adaptively reduce the influence of dynamic regions without relying on predefined dynamic priors?** The authors observe that if a per-pixel uncertainty weight can be assigned—high uncertainty for dynamic regions and low uncertainty for static regions—the BA optimization will naturally "ignore" the contributions of dynamic pixels.

## Core Problem

- Dynamic objects violate the static world assumption, causing large numbers of outliers in the reprojection residuals of dynamic regions within BA, severely disrupting pose estimation.
- Predefined semantic priors cannot cover all dynamic categories, and "potentially moving categories" are not always in motion.
- Neural implicit methods can handle dynamic scenes but have prohibitive computational costs that prevent real-time deployment.

## Method

### Overall Architecture

DROID-W introduces three key improvements over DROID-SLAM:

1. **Uncertainty-aware BA (UBA)**: integrates per-pixel uncertainty weights into Bundle Adjustment optimization.
2. **Dynamic Uncertainty Update**: detects dynamic regions and assigns uncertainty using semantic features from the DINOv2 visual foundation model.
3. **Monocular Depth Regularization**: incorporates monocular depth prior constraints into BA to enhance stability in extremely dynamic scenes.

### Uncertainty-aware Bundle Adjustment (UBA)

The standard DROID-SLAM DBA layer jointly optimizes camera poses $\{G_i\}$ and inverse depth maps $\{d_i\}$ by minimizing a weighted reprojection error:

$$E = \sum_{(i,j) \in \mathcal{E}} \| p_{ij}^* - \Pi_c(G_{ij} \cdot \Pi_c^{-1}(p_i, d_i)) \|_{\Sigma_{ij}}^2$$

where $p_{ij}^*$ is the correspondence coordinate obtained via correlation lookup, and $\Sigma_{ij}$ is the predicted confidence weight.

DROID-W's UBA further introduces per-pixel uncertainty $u_{ij}$:

$$E_{UBA} = \sum_{(i,j) \in \mathcal{E}} \frac{1}{u_{ij}} \| p_{ij}^* - \Pi_c(G_{ij} \cdot \Pi_c^{-1}(p_i, d_i)) \|_{\Sigma_{ij}}^2 + \log u_{ij}$$

Key design choices:

- A larger uncertainty $u_{ij}$ reduces the contribution of that pixel to the reprojection error, effectively down-weighting it automatically.
- The $\log u_{ij}$ regularization term prevents uncertainty from growing unboundedly (avoiding the trivial solution).
- The uncertainty $u_{ij}$ participates in the iterative BA optimization and is jointly updated with poses and depths.

### Dynamic Uncertainty Update

The initialization and update of uncertainty $u_{ij}$ rely on dense visual semantic features extracted by DINOv2:

**Step 1 — Feature Extraction**: For each frame $I_i$, a pretrained DINOv2 is used to extract a dense feature map $F_i \in \mathbb{R}^{H \times W \times C}$.

**Step 2 — Rigid Motion Correspondence**: The poses and depths estimated by the current BA are used to compute the rigid reprojection coordinates $p_{ij}$ from frame $i$ to frame $j$.

**Step 3 — Feature Cosine Similarity**: The feature $F_i(p)$ at pixel $p$ in frame $i$ is compared with the feature $F_j(p_{ij})$ at the corresponding location $p_{ij}$ in frame $j$:

$$s_{ij}(p) = \frac{F_i(p) \cdot F_j(p_{ij})}{\|F_i(p)\| \cdot \|F_j(p_{ij})\|}$$

**Step 4 — Uncertainty Assignment**:

$$u_{ij}(p) = 1 - s_{ij}(p)$$

The core intuition is as follows:

- **Static regions**: features are highly consistent after rigid reprojection; $s_{ij}$ approaches 1 and $u_{ij}$ approaches 0 (low uncertainty).
- **Dynamic regions**: after object motion, rigid reprojection points to an incorrect location; $F_j(p_{ij})$ mismatches $F_i(p)$, resulting in low $s_{ij}$ and high $u_{ij}$ (high uncertainty).

DINOv2 semantic features are more robust than raw pixel values—even under illumination changes or slight viewpoint variations, feature similarity remains high in static regions, while geometric inconsistency amplifies feature discrepancies in dynamic regions.

### Monocular Depth Regularization

In extremely dynamic scenes (e.g., where 80%+ of the scene is in motion), most pixels are assigned high uncertainty, leaving too few constraints for BA and causing optimization instability. The solution is to incorporate a monocular depth prior as regularization:

$$E_{depth} = \lambda \sum_i \| d_i - d_i^{mono} \|^2$$

where $d_i^{mono}$ is the depth predicted by a pretrained monocular depth estimation model (e.g., DPT/ZoeDepth). A scale- and shift-invariant loss formulation is used, since monocular depth lacks absolute scale.

### Iterative Optimization Pipeline

1. Initialization: standard DROID-SLAM pipeline, iteratively updating optical flow and confidence using ConvGRU.
2. After every $K$ BA iterations, the Dynamic Uncertainty Update is invoked to recompute $u_{ij}$.
3. The updated $u_{ij}$ is injected into the next round of UBA optimization.
4. The process repeats until convergence.

## Key Experimental Results

### TUM RGB-D Dynamic Sequences

| Method | ATE RMSE (cm)↓ | Dynamic Ratio |
|--------|----------------|---------------|
| ORB-SLAM3 | 36.5 | High |
| DROID-SLAM | 28.3 | High |
| DynaSLAM | 3.8 | High |
| **DROID-W** | **2.1** | **High** |

On high-dynamic TUM sequences (e.g., the *walking* series), the ATE is reduced to 2.1 cm, representing a 13× improvement over the original DROID-SLAM.

### DROID-W Dataset (In-the-Wild Data)

The authors construct a dedicated evaluation dataset containing diverse outdoor dynamic scenes (street pedestrians, cyclists, runners, etc.) and YouTube video clips. Qualitative evaluation shows:

- DROID-SLAM exhibits severe trajectory drift in dynamic scenes.
- DROID-W maintains stable trajectory estimation, with camera paths closely aligned with ground truth.

### KITTI Dynamic Scenes

| Method | Translation Error↓ | Rotation Error↓ |
|--------|-------------------|----------------|
| DROID-SLAM | Failure / Drift | Failure / Drift |
| **DROID-W** | **Significant Improvement** | **Significant Improvement** |

On KITTI sequences with dense vehicle traffic, DROID-SLAM frequently fails, while DROID-W tracks continuously and stably.

### Ablation Study

- **Removing UBA (standard BA only)**: ATE increases substantially, degrading to DROID-SLAM performance.
- **Removing DINOv2 features (using raw pixel similarity instead)**: uncertainty estimation becomes less robust, ATE increases by approximately 40%.
- **Removing Monocular Depth Regularization**: BA occasionally diverges in scenes with extremely high dynamic ratios.
- **Different backbone models**: DINOv2 > DINO > CLIP > ResNet50 features, validating the semantic robustness of DINOv2.

### Runtime Efficiency

- Approximately 10 FPS real-time operation.
- 100× faster than neural implicit methods (e.g., RoDynRF at ~0.1 FPS).
- DINOv2 feature extraction can be optimized via caching and downsampling, adding approximately 15% overhead.

## Highlights & Insights

- **Elegant uncertainty modeling**: the dynamic detection problem is reformulated as uncertainty weighting and seamlessly integrated into the existing BA framework without modifying the underlying optimizer architecture.
- **No predefined dynamic priors required**: dynamic regions are detected adaptively via feature similarity, enabling the handling of any type of moving object, including categories unseen during training.
- **Leveraging visual foundation models**: the strong semantic features of DINOv2 make dynamic detection robust under challenging conditions such as illumination changes and insufficient texture.
- **Real-time capability preserved**: the ~10 FPS operating speed enables practical deployment, far exceeding neural implicit approaches.
- **Minimal modifications**: only an uncertainty module is added on top of DROID-SLAM, resulting in small code changes and strong generalizability.

## Limitations & Future Work

- The computational overhead of DINOv2 itself is non-negligible, posing challenges for deployment on embedded devices.
- In extreme scenarios where the static background is almost entirely occluded (e.g., in-vehicle footage with entirely moving content outside the window), the constraint from monocular depth regularization is limited.
- No comparison is made against recent 3D Gaussian Splatting dynamic scene methods (e.g., DynGaussian).
- The uncertainty update frequency (every $K$ steps) is a hyperparameter whose optimal value may vary across scenes.
- Validation is limited to monocular settings; extensions to stereo or RGB-D inputs are not explored.

## Related Work & Insights

| Dimension | DROID-SLAM | DynaSLAM | RoDynRF | DROID-W |
|-----------|-----------|----------|---------|---------|
| Dynamic Handling | None | Semantic masking | Neural implicit | Uncertainty weighting |
| Dynamic Prior | None | Required (predefined categories) | None | None |
| Real-time | ~15 FPS | ~5 FPS | ~0.1 FPS | ~10 FPS |
| Scene Reconstruction | Sparse / semi-dense | Sparse | Dense | Sparse / semi-dense |
| Robustness | Fails on dynamics | OK for known categories | General | General |

DROID-W achieves the best balance between general dynamic robustness and real-time performance, representing a solution with both engineering practicality and academic novelty.

The uncertainty-weighting idea is a general robust estimation technique transferable to all vision geometry tasks based on BA/least squares, including optical flow estimation, stereo matching, and SfM. The use of DINOv2 features as universal semantic descriptors is noteworthy; future work could explore applying them to SLAM sub-modules such as loop closure detection and place recognition. An interesting contrast can be drawn with DMAligner: both address the challenge of dynamic scenes, but DMAligner uses generative methods to "bypass" the problem while DROID-W uses uncertainty weighting to "tolerate" it—entirely different philosophies, each with its own strengths. Future work could explore combining uncertainty estimation with 3DGS dynamic reconstruction for real-time high-quality dynamic scene reconstruction.

## Rating

- **Novelty**: 7/10 — Uncertainty-weighted BA is not novel in itself, but its combination with DINOv2 features and seamless integration into DROID-SLAM constitutes a practical contribution.
- **Experimental Thoroughness**: 8/10 — Multi-dataset evaluation, detailed ablation studies, and a self-collected in-the-wild dataset; however, quantitative comparisons with some recent methods are missing.
- **Writing Quality**: 8/10 — Problem motivation is clear, method description is concise, and experimental organization is well-structured.
- **Value**: 8/10 — Directly improves the dynamic robustness of a classical SLAM system with high engineering deployment value; the ~10 FPS real-time performance is a major selling point.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VGGT-SLAM++: Visual SLAM with DEM-Based Covisibility and Local Bundle Adjustment](vggt-slam.md)
- [\[CVPR 2026\] Unblur-SLAM: Dense Neural SLAM for Blurry Inputs](unblur-slam_dense_neural_slam_for_blurry_inputs.md)
- [\[CVPR 2026\] Scene Grounding In the Wild](scene_grounding_in_the_wild.md)
- [\[CVPR 2026\] SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM](sgad-slam_splatting_gaussians_at_adjusted_depth_for_better_radiance_fields_in_rg.md)
- [\[CVPR 2026\] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator](gaussfusion_improving_3d_reconstruction_in_the_wild_with_a_geometry-informed_vid.md)

</div>

<!-- RELATED:END -->
