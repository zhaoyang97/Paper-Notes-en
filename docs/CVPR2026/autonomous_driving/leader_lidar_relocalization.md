---
title: >-
  [Paper Note] LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization
description: >-
  [CVPR 2026][Autonomous Driving][LiDAR relocalization] LEADER achieves 24.1% and 73.9% relative reductions in position error on LiDAR relocalization benchmarks via a robust projection-based geometric encoder (yaw-invarian…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "LiDAR relocalization"
  - "scene coordinate regression"
  - "yaw invariance"
  - "reliability estimation"
  - "point cloud"
date: 2026-05-08
content_hash: e82f5aeb5e0a4dcc
---

# LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization

**Conference**: CVPR 2026 Highlight  
**arXiv**: [2604.11355](https://arxiv.org/abs/2604.11355)  
**Code**: [https://github.com/JiansW/LEADER](https://github.com/JiansW/LEADER)  
**Area**: Autonomous Driving
**Keywords**: LiDAR relocalization, scene coordinate regression, yaw invariance, reliability estimation, point cloud

## TL;DR
LEADER achieves 24.1% and 73.9% relative reductions in position error on LiDAR relocalization benchmarks via a robust projection-based geometric encoder (yaw-invariant) and a truncated relative reliability loss (suppressing unreliable points).

## Background & Motivation

**Background**: LiDAR relocalization is critical for autonomous driving. Dominant approaches fall into two categories: retrieval-and-registration (requiring storage of dense point cloud maps) and learning-based regression, the latter further divided into absolute pose regression (APR) and scene coordinate regression (SCR).

**Limitations of Prior Work**: (1) Retrieval-and-registration methods incur significant storage and communication overhead; (2) APR achieves limited accuracy; (3) existing SCR network architectures lack yaw invariance, causing performance degradation during vehicle turns; (4) all predicted points are treated equally, and erroneous correspondences from degenerate regions (textureless areas, dynamic objects) severely corrupt pose estimation.

**Key Challenge**: Yaw rotations are frequent in autonomous driving scenarios, and a large proportion of scene points are inherently unsuitable for relocalization (dynamic objects, repetitive textures), yet existing methods neither handle rotations nor distinguish point-level reliability.

**Core Idea**: Design a yaw-invariant geometric encoder combined with point-level reliability quantification to jointly improve the robustness of SCR.

## Method

### Overall Architecture
Raw point cloud → ground plane estimation and plane rectification → cylindrical projection (yaw dimension converted to translation) → voxelization → circular sparse convolution for multi-scale feature extraction → multi-head max regressor outputting scene coordinates and reliability scores → Cartesian recovery → training with truncated relative reliability loss → inference-time RANSAC pose estimation driven by high-reliability points.

### Key Designs

1. **Robust Projection-based Geometric Encoder (RPGE)**:

    - **Function**: Extract yaw-invariant multi-scale geometric features.
    - **Mechanism**: The point cloud is transformed via cylindrical projection as $(x^p = s \cdot \arctan2(y', x'), y^p = \sqrt{x'^2 + y'^2}, z^p = z')$, converting yaw rotation into a translation along the $x$ direction in the projected space. After voxelization, circular sparse convolution handles discontinuities at yaw boundaries (features at boundaries are circularly padded), and initial features are constructed from range, height, and reflectance intensity only (excluding yaw-related coordinates).
    - **Design Motivation**: Standard convolution produces discontinuous features at yaw boundaries; circular convolution ensures ring-wise continuity. Excluding yaw-related coordinates guarantees rotation invariance of the initial features.

2. **Truncated Relative Reliability Loss (TRR)**:

    - **Function**: Model point-level reliability and suppress interference from degenerate regions.
    - **Mechanism**: The network simultaneously predicts scene coordinates and reliability scores $u_i$. Scores are transformed into self-normalized weights $w_i$ via arctan scaling and truncation, assigning large weights to high-reliability points and suppressing low-reliability ones. The loss is $\mathcal{L}_{TRR} = \sum w_i \mathcal{L}_{raw,i}$.
    - **Design Motivation**: Not all scene points are suitable for relocalization—predictions from dynamic objects and textureless regions are inherently unreliable. The network is trained to "abandon" such points and focus on reliable features.

3. **Multi-Head Max Regressor**:

    - **Function**: Regress scene coordinates from encoded features.
    - **Mechanism**: The 512-dimensional feature is projected to $k \times 512$ dimensions ($k=4$ heads); element-wise maximum is taken across heads. Five such operations are stacked, followed by fully connected layers outputting 3D coordinates and reliability scores.
    - **Design Motivation**: Multi-head max pooling enhances feature robustness and expressiveness.

### Loss & Training
End-to-end training with the truncated relative reliability loss (TRR). At inference, the top-$k$ points by reliability score are selected for RANSAC-based 6-DoF pose estimation.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|--------|----------|---------|
| Oxford RobotCar | Position error (m) | 0.63 | 0.83 (LightLoc) | −24.1% |
| NCLT | Position error (m) | 0.19 | 0.72 (SGLoc→LiSA) | −73.9% |
| Oxford | Orientation error (°) | 1.11 | 1.12 | −0.9% |

### Ablation Study

| Configuration | Oxford Pos. Error | NCLT Pos. Error | Note |
|------|--------------|-------------|------|
| Full LEADER | 0.63 | 0.19 | Complete model |
| w/o circular convolution | Increases | Increases | Yaw boundary discontinuity |
| w/o TRR | Increases | Increases | Interference from unreliable points |
| w/o projection transform | Increases | Increases | Loss of yaw invariance |

### Key Findings
- The improvement is most pronounced on NCLT (73.9%), as NCLT contains greater yaw variation and more degenerate regions.
- The reliability scores learned by TRR are intuitively consistent—stable structures such as building facades receive high scores, while ground and vegetation receive low scores.
- SCR methods consistently outperform APR methods by explicitly leveraging geometric constraints.

## Highlights & Insights
- **Cylindrical projection + circular convolution**: Elegantly reformulates the yaw rotation problem as a translation equivariance problem—computationally efficient and theoretically well-motivated.
- **Self-learned reliability**: The network automatically learns to distinguish reliable from unreliable regions without manual annotation or semantic priors.

## Limitations & Future Work
- Only yaw rotation is explicitly handled; pitch and roll variations are not modeled.
- Adaptive selection of the reliability threshold remains an open problem.
- Future work could incorporate semantic information to further improve reliability estimation.

## Related Work & Insights
- **vs. LiSA**: LiSA uses semantic priors to differentiate point contributions, whereas LEADER employs learned reliability scores without requiring additional semantic annotations.
- **vs. RALoc**: RALoc also addresses rotation but via a different mechanism; LEADER's projection-based approach is more principled and natural.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of projection transform and reliability loss is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Substantial margins over prior art on two authoritative benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and well-organized.
- Value: ⭐⭐⭐⭐ Practical significance for LiDAR-based localization in autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images](bev-sld_self-supervised_scene_landmark_detection_for_global_localization_with_li.md)
- [\[AAAI 2026\] Beta Distribution Learning for Reliable Roadway Crash Risk Assessment](../../AAAI2026/autonomous_driving/beta_distribution_learning_for_reliable_roadway_crash_risk_a.md)
- [\[CVPR 2026\] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)
- [\[CVPR 2026\] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](neural_distribution_prior_for_lidar_ood_detection.md)

</div>

<!-- RELATED:END -->
