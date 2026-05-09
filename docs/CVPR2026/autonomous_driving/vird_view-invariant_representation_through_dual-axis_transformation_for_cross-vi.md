---
title: >-
  [Paper Note] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation
description: >-
  [CVPR2026][Autonomous Driving][cross-view pose estimation] This paper proposes VIRD, which constructs view-invariant representations via dual-axis transformation (polar transformation + context-enhanced positional attention) to achieve state-of-the-art cross-view pose estimation without orientation priors, reducing position and orientation errors on KITTI by 50.7% and 76.5%, respectively.
tags:
  - CVPR2026
  - Autonomous Driving
  - cross-view pose estimation
  - view-invariant representation
  - polar transformation
  - positional attention
  - autonomous driving localization
date: 2026-05-08
content_hash: 247211ee5a04a471
---

# VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation

**Conference**: CVPR2026
**arXiv**: [2603.12918](https://arxiv.org/abs/2603.12918)
**Code**: To be confirmed
**Area**: Autonomous Driving
**Keywords**: cross-view pose estimation, view-invariant representation, polar transformation, positional attention, autonomous driving localization

## TL;DR

This paper proposes VIRD, which constructs view-invariant representations via dual-axis transformation (polar transformation + context-enhanced positional attention) to achieve state-of-the-art cross-view pose estimation without orientation priors, reducing position and orientation errors on KITTI by 50.7% and 76.5%, respectively.

## Background & Motivation

**Global localization is a critical requirement for autonomous driving**: Accurate global localization is essential for autonomous driving and mobile robotics, serving as a foundational capability for real-world navigation.

**GNSS is unreliable in urban scenarios**: In dense urban areas, GNSS signals degrade severely due to occlusion and multipath effects, causing significant drops in localization accuracy.

**Cross-view pose estimation as an alternative**: Estimating the 3-DoF pose of a ground camera using geo-referenced satellite imagery is a promising alternative, but the large viewpoint discrepancy between ground and satellite views poses a fundamental challenge.

**Existing methods rely on orientation priors**: Early methods assume a known coarse orientation and iteratively refine within a narrow search space; however, orientation priors are often inaccurate or unavailable in practice, leading to convergence to suboptimal solutions.

**Semantic methods neglect spatial correspondence**: Recent omnidirectional CVPE methods reduce the viewpoint gap via semantic similarity (cross-attention, contrastive learning), but overlook spatial correspondences, leaving the viewpoint discrepancy fundamentally unresolved.

**Geometric transformation methods each have drawbacks**: Polar transformation addresses only horizontal alignment while neglecting the vertical axis; projective transformation depends on camera parameters and introduces severe artifacts around vertical structures such as buildings.

## Method

### Overall Architecture

VIRD is an omnidirectional cross-view pose estimation framework that builds view-invariant descriptors through dual-axis transformation. The overall pipeline is as follows:
1. **Feature extraction**: A pretrained CNN (VGG16 / EfficientNet-B0) extracts ground features $F_g \in \mathbb{R}^{C \times H \times W_g}$ and satellite features $F_s \in \mathbb{R}^{C \times A \times A}$ separately.
2. **Horizontal-axis alignment**: Polar transformation is applied to satellite features, mapping the azimuth angle to the horizontal axis.
3. **Vertical-axis alignment**: A Context-Enhanced Positional Attention (CEPA) module eliminates vertical-axis misalignment.
4. **Descriptor generation**: Features are compressed along the vertical direction and flattened into orientation-aware 1D descriptors $D_g$ and $D_{s2p}$.
5. **Coarse matching + fine regression**: A coarse pose is obtained via cosine similarity matching, then refined by a regression module that predicts residual corrections.

### Key Designs

**Polar Transformation (horizontal alignment)**: Satellite feature maps are resampled in polar coordinates centered at a candidate location, mapping azimuth and radial distance to the horizontal and vertical axes, respectively. The transformation width $W_s = \frac{2\pi}{\text{HFoV}} \cdot W_g$ ensures consistency across different fields of view.

**Positional Attention (PA)**: Three sets of sinusoidal positional encodings are defined — a shared virtual encoding $P_a$, a ground encoding $P_g$, and a satellite encoding $P_{s2p}$ — and an attention mechanism learns the mapping between vertical coordinates across views:
$$\mathcal{A}_v = \text{Softmax}\left(\frac{(P_a W_v^Q)(P_v W_v^K)^\top}{\sqrt{d_k}}\right)$$
The core insight is to reinterpret positional attention as learning a cross-view vertical coordinate transformation through a shared virtual axis, without requiring camera parameters.

**Context-Enhanced Positional Attention (CEPA)**: PA assumes a consistent vertical transformation across all horizontal directions and cannot adapt to direction-dependent variations in vertical structures. CEPA leverages local context from ground features to enhance attention weights:
$$\mathcal{A}_{g'} = \mathcal{A}_g + \text{Softmax}\left(\Phi(\mathcal{A}_g \oplus F_g)\right)$$
A convolutional layer $\Phi$ processes the concatenated attention weights and features, enabling the model to adaptively transform ground features according to scene context.

**View Reconstruction Loss**: Each descriptor is trained to reconstruct both the original and the cross-view image, comprising an original reconstruction loss $\mathcal{L}_{\text{origin}}$ and a cross-reconstruction loss $\mathcal{L}_{\text{cross}}$, guiding descriptors to encode vertical structural information and improve view invariance.

### Loss & Training

The total loss consists of three terms:
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{match}} + \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{recon}} = \alpha_1 \mathcal{L}_{\text{origin}} + \alpha_2 \mathcal{L}_{\text{cross}}$: view reconstruction loss ($\ell_1$-loss)
- $\mathcal{L}_{\text{match}}$: InfoNCE matching loss, encouraging high similarity at ground-truth poses
- $\mathcal{L}_{\text{reg}}$: $\ell_2$-loss on pose residuals

## Key Experimental Results

### Main Results

**KITTI dataset** (no orientation prior, cross-area setting, EfficientNet-B0):

| Method | Position Mean (m)↓ | Position Median (m)↓ | Orientation Mean (°)↓ | Orientation Median (°)↓ |
|------|:---:|:---:|:---:|:---:|
| HighlyAccurate | 15.50 | 16.02 | 89.84 | 89.85 |
| SliceMatch | 14.85 | 11.85 | 23.64 | 7.96 |
| CCVPE | 13.94 | 10.98 | 77.84 | 63.84 |
| FG2 | 13.58 | 11.72 | 90.12 | 90.42 |
| **VIRD (Ours)** | **11.12** | **5.41** | **22.03** | **1.87** |

**VIGOR dataset** (no orientation alignment, cross-area setting, EfficientNet-B0):

| Method | Position Mean (m)↓ | Position Median (m)↓ | Orientation Mean (°)↓ | Orientation Median (°)↓ |
|------|:---:|:---:|:---:|:---:|
| CCVPE | 5.41 | 1.89 | 27.78 | 13.58 |
| DenseFlow | 7.67 | 3.67 | 17.63 | 2.94 |
| FG2† | 5.95 | 2.40 | 28.41 | 2.20 |
| **VIRD (Ours)** | **4.61** | **1.55** | **16.50** | **1.17** |

### Ablation Study

**Dual-axis transformation ablation** (KITTI cross-area, VGG16):

| Transformation Strategy | Position Median (m) | Orientation Median (°) |
|----------|:---:|:---:|
| Projective S2G | 10.59 | 3.84 |
| Projective G2S | 15.20 | 5.44 |
| Polar only | 11.75 | 4.00 |
| Polar + PA | 9.76 | 3.44 |
| Polar + CEPA | **8.88** | **3.36** |

**Model component ablation** (KITTI cross-area, VGG16):

| Configuration | Position Median (m) | Orientation Median (°) |
|------|:---:|:---:|
| Polar + CEPA | 8.88 | 3.36 |
| + $\mathcal{L}_{\text{origin}}$ | 8.29 | 3.31 |
| + $\mathcal{L}_{\text{cross}}$ | 8.10 | 3.21 |
| + Both | 7.90 | 3.05 |
| + Both + Regression | **7.05** | **2.22** |

### Key Findings

- Dual-axis transformation significantly outperforms all single geometric transformation baselines; the incremental introduction of PA and CEPA yields consistent improvements.
- The view reconstruction loss contributes most to orientation estimation, reducing mean orientation error by 39.1%; cross-reconstruction proves more effective than original reconstruction alone.
- The regression module reduces position median by 0.85 m and orientation median by 0.83°, though mean orientation error slightly increases due to opposite-direction predictions being refined further away.
- VIRD maintains low errors across varying levels of orientation noise (±10° to ±180°), demonstrating substantially better robustness than CCVPE and FG2.

## Highlights & Insights

- **Dual-axis transformation strategy is elegantly designed**: The cross-view matching problem is decomposed into two tractable sub-problems — horizontal (polar transformation) and vertical (positional attention) — avoiding the dependence of projective transformation on camera parameters and depth information.
- **CEPA module is intuitively motivated and effective**: Positional attention is reinterpreted as a coordinate transformation through a shared virtual axis, and context enhancement enables adaptation to direction-dependent vertical structural variations.
- **View reconstruction loss is cleverly designed**: Requiring descriptors to reconstruct both original and cross-view images enforces view invariance while encouraging the model to focus on structures shared across views.
- **No orientation prior required**: VIRD achieves state-of-the-art performance over a full 360° search space, yielding high practical value for real-world autonomous driving localization.

## Limitations & Future Work

- Only 3-DoF pose (x, y, yaw) is addressed; pitch and roll are assumed negligible, which may not hold in complex terrain.
- In the VIGOR setting with aligned orientations, position accuracy falls short of methods such as FG2 that exploit 3D structural information.
- The regression module may amplify errors when the coarse matching orientation is incorrect (opposite-direction issue).
- Validation is limited to the KITTI and VIGOR datasets; generalization to a broader range of urban environments and regions remains unknown.
- Polar transformation must be applied at each candidate pose during training, and computational overhead may be significant as the number of candidates increases.

## Related Work & Insights

- **Limited-angle CVPE**: Methods such as HighlyAccurate and PIDLoc use the LM algorithm or neural pose optimizers for iterative refinement within a narrow angular range, constrained by orientation priors.
- **Omnidirectional CVPE**: SliceMatch employs content-based cross-attention; CCVPE uses contrastive learning; both emphasize semantic similarity while neglecting spatial correspondence.
- **Geometric transformations**: Polar transformation (Shi, Li, et al.) addresses horizontal alignment; projective transformation (HighlyAccurate, et al.) attempts to handle both axes simultaneously but depends on camera parameters and introduces artifacts.
- **FG2**: Exploits height-aware 3D point selection to narrow the viewpoint gap, but sparse point matching leads to degraded orientation estimation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-axis transformation strategy and CEPA module are novel; the reinterpretation of positional attention is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparisons on two mainstream datasets with complete ablations, including robustness analysis and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated problem formulation, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — State-of-the-art without orientation priors, with direct applicability to real-world autonomous driving localization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CycleBEV: Regularizing View Transformation Networks via View Cycle Consistency for Bird's-Eye-View Semantic Segmentation](cyclebev_regularizing_view_transformation_networks_via_view_cycle_consistency_fo.md)
- [\[CVPR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)
- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[CVPR 2026\] x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)

</div>

<!-- RELATED:END -->
