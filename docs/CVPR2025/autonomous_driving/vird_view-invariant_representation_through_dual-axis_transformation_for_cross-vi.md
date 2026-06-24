---
title: >-
  [Paper Note] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation
description: >-
  [CVPR2025][Autonomous Driving][cross-view pose estimation] VIRD constructs view-invariant representations through a dual-axis transformation (polar transformation + context-enhanced positional attention) to achieve omnidirectional cross-view pose estimation without orientation priors, reducing position and orientation errors on KITTI by 50.7% and 76.5%, respectively.
tags:
  - "CVPR2025"
  - "Autonomous Driving"
  - "cross-view pose estimation"
  - "geo-localization"
  - "view-invariant representation"
  - "polar transformation"
  - "positional attention"
date: 2026-05-08
content_hash: 47859c0362e770dd
---

# VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation

**Conference**: CVPR2025  
**arXiv**: [2603.12918](https://arxiv.org/abs/2603.12918)  
**Code**: To be confirmed  
**Area**: Autonomous Driving  
**Keywords**: cross-view pose estimation, geo-localization, view-invariant representation, polar transformation, positional attention

## TL;DR
VIRD constructs view-invariant representations through a dual-axis transformation (polar transformation + context-enhanced positional attention) to achieve omnidirectional cross-view pose estimation without orientation priors, reducing position and orientation errors on KITTI by 50.7% and 76.5%, respectively.

## Background & Motivation
1. Global localization is crucial for autonomous driving and robotics, but the reliability of GNSS decreases in dense urban areas due to signal occlusion and multipath effects.
2. Cross-view pose estimation (CVPE) achieves fine-grained 3-DoF localization by matching ground images with satellite images, serving as a promising alternative to GNSS.
3. Early CVPE methods relied on coarse orientation priors, which are often inaccurate or unavailable in practice.
4. Existing omnidirectional CVPE methods ignore the huge perspective gap between ground and satellite views, where relying solely on semantic similarity is insufficient to establish spatial correspondences.
5. Polar transformation solves horizontal alignment but ignores vertical axis misalignment; projection-based transformation is sensitive to camera calibration and introduces severe artifacts at vertical structures such as buildings.
6. Effectively resolving vertical axis misalignment remains an open challenge.

## Method

### Overall Architecture
VIRD constructs view-invariant descriptors for omnidirectional cross-view pose estimation. The workflow includes: (1) descriptor construction via dual-axis transformation; (2) view-reconstruction loss to enhance view invariance; (3) descriptor matching + residual regression to predict the final pose.

### Key Designs

**1. Dual-Axis Transformation**

*Horizontal Axis — Polar Transformation*:
- Performs polar transformation on the satellite feature map $F_s$ centered at the candidate position.
- Maps azimuth to the horizontal axis and radial distance to the vertical axis.
- Set the transformed width to $W_s = \frac{2\pi}{\text{HFoV}} \cdot W_g$ to ensure FoV consistency.

*Vertical Axis — Positional Attention (PA)*:
- Defines three positional encodings: shared virtual $P_a \in \mathbb{R}^{H_Q \times d_p}$, ground $P_g$, and satellite $P_{s2p}$.
- Attention weights $\mathcal{A}_v = \text{Softmax}\left(\frac{(P_a W_v^Q)(P_v W_v^K)^\top}{\sqrt{d_k}}\right)$.
- Establishes cross-view consistent vertical correspondences via the shared virtual vertical axis.

*Context-Enhanced Positional Attention (CEPA)*:
- Standard PA assumes identical vertical transformation for all horizontal directions, lacking adaptability to vertical structures.
- CEPA refines attention using ground feature context: $\mathcal{A}_{g'} = \mathcal{A}_g + \text{Softmax}(\Phi(\mathcal{A}_g \oplus F_g))$.
- Allows the model to adaptively transform ground features in different horizontal directions based on the scene context.

**2. View-Reconstruction Loss**
- Trains descriptors to reconstruct both original views and cross-views simultaneously.
- Four decoders: $G_{g \to g}$, $G_{s \to s}$, $G_{s \to g}$, $G_{g \to s}$.
- Original-view reconstruction $\mathcal{L}_{\text{origin}}$ + cross-view reconstruction $\mathcal{L}_{\text{cross}}$.
- Guides descriptors to encode vertical structural information, resolving ambiguities in visually similar road scenes.

**3. Matching and Regression**
- Descriptor matching: Computes cosine similarity on candidate-pose grids, trained with InfoNCE loss.
- Pose regression: Predicts the residual $\Delta \mathbf{p} = (\Delta x, \Delta y, \Delta \theta)$ from the coarsely matched pose.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{match}} + \mathcal{L}_{\text{reg}}$

## Key Experimental Results

### KITTI Dataset (No Orientation Prior, Same-Area)

| Method | Backbone | Med. Pos.(m)↓ | Med. Ori.(°)↓ | R@1m Lat.↑ |
|------|----------|---------------|---------------|------------|
| SliceMatch | VGG16 | 5.41 | 4.42 | 39.73% |
| CCVPE | EffNet-B0 | 3.47 | 6.12 | 53.30% |
| DenseFlow | ResNet18 | 4.26 | 0.99 | 73.87% |
| **VIRD** | **VGG16** | **2.07** | **1.02** | **79.46%** |

### VIGOR Dataset (Same-Area, Unaligned)

| Method | Med. Pos.(m)↓ | Med. Ori.(°)↓ |
|------|---------------|---------------|
| SliceMatch | 5.77 | 67.37 |
| CCVPE | 4.56 | 75.86 |
| **VIRD** | **3.74** | **2.15** |

### Key Findings
- Median position error decreases from 4.26m to 2.07m (a 50.7% reduction) on KITTI, and orientation error decreases from 4.42° to 1.02° (a 76.5% reduction).
- Cross-Area generalization is also significantly superior to existing methods.
- The view-reconstruction loss significantly contributes to encoding vertical structural information.
- CEPA achieves greater improvement compared to standard PA in complex urban scenes.

## Highlights & Insights
1. **Dual-Axis Transformation Innovation**: For the first time, explicitly resolves the cross-view gap along both horizontal and vertical axes separately, establishing consistent correspondences via a shared virtual axis.
2. **CEPA Adaptability**: Dynamically adjusts vertical attention weights through ground context, capturing orientation changes in vertical structures such as buildings.
3. **No Camera Parameters Required**: Positional attention learns vertical transformations without relying on internal or external camera parameters, avoiding calibration sensitivity associated with perspective projection.
4. **View-Reconstruction Regularization**: Enhances the view-invariance of descriptors through reconstruction tasks, serving as an elegant self-supervised signal.

## Limitations & Future Work
1. The polar transformation assumes a flat ground approximation, which may fail in hilly or highly uneven terrains.
2. Validation is only performed on KITTI and VIGOR; scenes with larger discrepancies in satellite image resolution or timestamps have not been tested.
3. View reconstruction requires additional decoders, which increases training overhead; although omitted during inference, this adds complexity to the training phase.
4. Negligible pitch/roll is assumed, which might not hold for real-world robotics (especially on slopes).

## Related Work & Insights
- Polar transformation originates from the works of Shi & Li, and VIRD complements this with a vertical axis solution.
- CEPA can be generalized to other tasks requiring cross-domain vertical alignment (e.g., remote sensing-to-ground matching, floor-level localization).
- The concept of view-reconstruction loss can inspire regularization designs for other cross-view or cross-modal matching tasks.
- It complements LiDAR-based localization and is highly suitable for scenarios with only monocular cameras.

## Rating
- Novelty: ⭐⭐⭐⭐ (Dual-Axis Transformation + CEPA + View Reconstruction)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two datasets, Same/Cross-Area, multiple backbones)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear diagrams, thorough analysis of problems)
- Value: ⭐⭐⭐⭐ (Significantly advances prior-free cross-view localization SOTA)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [\[CVPR 2025\] Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models](closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.md)
- [\[CVPR 2025\] PIDLoc: Cross-View Pose Optimization Network Inspired by PID Controllers](pidloc_cross-view_pose_optimization_network_inspired_by_pid_controllers.md)
- [\[CVPR 2025\] ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)
- [\[CVPR 2025\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)

</div>

<!-- RELATED:END -->
