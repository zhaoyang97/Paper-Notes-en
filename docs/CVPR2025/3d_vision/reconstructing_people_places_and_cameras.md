---
title: >-
  [Paper Note] Reconstructing People, Places, and Cameras
description: >-
  [CVPR 2025][3D Vision][Multi-view Reconstruction] HSfM unifies human mesh estimation with the traditional SfM framework. By jointly optimizing humans, scene point clouds, and camera parameters, it achieves metric-scale world coordinate reconstruction from uncalibrated sparse multi-view images, reducing human localization error from 3.59m to 0.50m.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-view Reconstruction"
  - "Human Pose Estimation"
  - "Structure from Motion"
  - "Metric Scale Recovery"
  - "Joint Optimization"
date: 2026-05-08
content_hash: f9aeb0c15c4a581d
---

# Reconstructing People, Places, and Cameras

**Conference**: CVPR 2025  
**arXiv**: [2412.17806](https://arxiv.org/abs/2412.17806)  
**Code**: [https://muelea.github.io/hsfm](https://muelea.github.io/hsfm)  
**Area**: 3D Vision  
**Keywords**: Multi-view Reconstruction, Human Pose Estimation, Structure from Motion, Metric Scale Recovery, Joint Optimization

## TL;DR
HSfM unifies human mesh estimation with the traditional SfM framework. By jointly optimizing humans, scene point clouds, and camera parameters, it achieves metric-scale world coordinate reconstruction from uncalibrated sparse multi-view images, reducing human localization error from 3.59m to 0.50m.

## Background & Motivation

**Background**: 3D human reconstruction and scene reconstruction (SfM) are two rapidly developing fields that have evolved independently for a long time. Data-driven SfM methods like DUSt3R can estimate dense scene point clouds and camera parameters, while methods like HMR2 can estimate human meshes from single images, each with its own strengths.

**Limitations of Prior Work**: Scene reconstruction methods (e.g., DUSt3R, MASt3R) do not reconstruct humans and lack metric scale information—their output camera poses and point clouds have only an arbitrary, up-to-scale relative relationship. Conversely, human reconstruction methods (e.g., HMR2, UnCaliPose) lack scene context, preventing them from placing human bodies in a world coordinate system consistent with the environment.

**Key Challenge**: SfM reconstruction naturally lacks absolute scale, while human estimation inherently lacks global scene anchoring. If resolved separately, neither can yield a complete, unified "human-scene-camera" representation.

**Goal**: Starting from sparse, uncalibrated multi-view images, simultaneously recover multiple human meshes, scene point clouds, and camera parameters, with all elements aligned in a unified metric world coordinate system.

**Key Insight**: The authors observe that human mesh estimation methods implicitly contain metric scale information (the statistical height of humans in the training data), which can be used to constrain the scene scale; meanwhile, 2D keypoint detection provides reliable cross-view correspondences that can offer strong constraints for Bundle Adjustment (BA).

**Core Idea**: Embed the human statistical model as a scale prior into the SfM framework, and perform joint optimization through human-keypoint-based BA and global scene alignment to achieve collaborative human-scene-camera reconstruction.

## Method

### Overall Architecture
The input to HSfM is a set of synchronized multi-view images (uncalibrated, with known cross-view human correspondences). It first leverages pretrained models (DUSt3R for scene point clouds and camera initialization, HMR2 for 3D human mesh initialization, and ViTPose for 2D keypoints) to obtain initial estimates, and then aligns all elements into a unified metric world coordinate system through a two-stage joint optimization. Outputs include: (1) SMPL-X mesh parameters for all humans, (2) scene point clouds for each view, (3) intrinsic and extrinsic parameters for each camera and the metric scale factor $\alpha$.

### Key Designs

1. **World Initialization (Metric Scale Recovery)**:

    - **Function**: Aligning human and scene estimates from different networks into the same coordinate system
    - **Mechanism**: Leveraging human orientation consistency constraints to estimate camera rotation $\hat{R}^c$, using similarity triangle relationships and predicted focal lengths to estimate human positions $\gamma$ in world coordinates, and finally solving for the scale factor $\hat{\alpha}$ via least squares to align SfM-predicted camera positions with human-derived camera positions. The key formula is $\hat{T}^c = \tilde{\gamma}^{c_1} - (\hat{R}^c)^\top \tilde{\gamma}^c$
    - **Design Motivation**: SfM reconstruction is up-to-scale. If the initial value of $\alpha$ is unreasonable (e.g., the scene is too small, causing the camera to be initialized inside the human body), the optimization can easily get trapped in local optima. A reasonable initialization is provided via the human height prior.

2. **Human-Keypoint-Based Bundle Adjustment**:

    - **Function**: Jointly optimizing human parameters and camera parameters via 2D keypoint reprojection errors
    - **Mechanism**: Defining the reprojection loss as $L_J^{ch} = \frac{1}{b_{2D}^{ch}} \| c_{2D}^{ch}(J_{2D}^{ch} - K^c(R^c J_{3D}^h + \alpha t^c)) \|_2$, normalized by the bounding box height and weighted by keypoint confidence. Meanwhile, body shape regularization $L_\beta^h = \|\beta^h\|_2$ is added to prevent overfitting. The BA process simultaneously updates $\{\alpha, \gamma, \beta, \phi, \theta, R, t, K\}$
    - **Design Motivation**: 2D human keypoints are natural cross-view corresponding points, which are more robust than traditional feature matching in wide-baseline and crowded scenarios. Furthermore, the 3D human mesh provides a reliable initial 3D structure.

3. **Global Scene Alignment Optimization**:

    - **Function**: Fusing multi-view point clouds into a unified world coordinate system
    - **Mechanism**: Following DUSt3R's global alignment loss, cross-view point cloud pairs $X^{c_i,c_j}$ are aligned to world coordinates weighted by projection matrices $P^{c_i,c_j \to w}$ and confidence $Q_i^{c_i,c_j}$. Unlike DUSt3R, scale regularization is not required here because the human body already provides the metric constraint
    - **Design Motivation**: Optimizing cameras solely on human keypoints would overfit to the keypoints while ignoring scene structure consistency. Incorporating scene alignment anchors the camera poses, forming a complementary optimization feedback loop

### Loss & Training
The total loss is $\min L_{\text{Humans}} + \lambda L_{\text{Places}}$, solved using a two-stage optimization strategy: the first stage optimizes $\{\alpha, \gamma, \beta\}$ with $\lambda=0$ to stabilize scale and human positions; the second phase sets $\lambda$ and optimizes all parameters $\{\gamma, \beta, \phi, \theta, R, t, K, D\}$ to achieve joint human-scene-camera fine-tuning.

## Key Experimental Results

### Main Results

| Dataset | Method | W-MPJPE↓ | GA-MPJPE↓ | RRA@15↑ | s-CCA@15↑ |
|--------|------|----------|-----------|---------|-----------|
| EgoHumans | UnCaliPose | 3.51m | 0.67m | 0.39 | 0.44 |
| EgoHumans | MASt3R | - | - | 0.74 | 0.86 |
| EgoHumans | **HSfM** | **1.04m** | **0.21m** | **0.89** | **0.91** |
| EgoExo4D | UnCaliPose | 3.59m | - | 0.31 | 0.37 |
| EgoExo4D | MASt3R | - | - | 0.90 | 0.81 |
| EgoExo4D | **HSfM** | **0.50m** | - | **0.89** | **0.84** |

### Ablation Study

| Configuration | W-MPJPE↓ | GA-MPJPE↓ | RRA@15↑ | CCA@15↑ |
|------|----------|-----------|---------|---------|
| HSfM (init.) | 4.28m | 0.51m | 0.79 | 0.38 |
| M1: No human gradient to camera | 3.94m | 0.57m | 0.79 | 0.40 |
| M2: No scene loss | 1.29m | 0.24m | 0.73 | 0.24 |
| M3: HSfM (Full) | **1.04m** | **0.21m** | **0.89** | **0.46** |

### Key Findings
- Joint optimization is key: after removing the scene loss (M2), human localization remains acceptable but camera accuracy drops significantly (CCA@15 drops from 0.46 to 0.24), indicating that human and scene constraints provide complementary contributions to camera estimation.
- Multi-person effect: increasing the number of people used in the optimization from 1 to 4 reduces W-MPJPE from 1.69m to 1.28m, and increases RRA@15 from 0.82 to 0.90, verifying that more human correspondences effectively strengthen BA.
- The improvement on EgoExo4D is smaller than on EgoHumans, as EgoExo4D often contains only one person, leading to weaker scale constraints.

## Highlights & Insights
- **Humans as Scale Anchors**: Restoring metric scale using the prior height information from human statistical models is a highly natural and elegant approach, avoiding the need for external information such as known object dimensions or GPS in traditional methods.
- **Contact-Constraint-Free Grounding**: After optimization, human bodies stand naturally on the ground without explicit "foot-to-ground contact" constraints, demonstrating that joint human-scene optimization implicitly resolves the grounding issue.
- **Human Keypoints for Cross-View BA**: Treating human keypoints as feature correspondences in SfM provides a robust alternative in wide-baseline scenarios where traditional feature matching fails.

## Limitations & Future Work
- Relies on known cross-view identity correspondences (re-identification); in practical scenarios, this requires an additional ReID module.
- When severe indoor occlusions occur, the quality of 2D keypoint detection degrades, affecting the optimization performance.
- Scene reconstruction quality remains limited (e.g., uneven ground issues); introducing stronger geometric priors or integrating with NeRF/3DGS for fine-grained reconstruction could be considered.
- Currently, this only handles single-frame static scenes; extending to video sequences can utilize temporal consistency to further improve accuracy.

## Related Work & Insights
- **vs UnCaliPose**: Also utilizes human keypoints for SfM, but UnCaliPose only optimizes humans and cameras without scene reconstruction, and requires ground-truth bone lengths. HSfM achieves superior camera estimation by incorporating scene optimization.
- **vs DUSt3R/MASt3R**: These methods perform dense scene reconstruction and camera estimation but do not handle humans. HSfM utilizes humans to provide scale constraints and additional correspondences, surpassing them even on camera metrics.
- This "human-as-anchor" concept can be generalized to other scene reconstruction tasks requiring metric scale, such as recovering scale using the statistical dimensions of vehicles or pedestrians in autonomous driving.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of introducing human priors into SfM is natural, yet the integration approach is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two large-scale datasets, comprehensive metrics, detailed ablations, and analysis on the number of people.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and standardized formulation expressions.
- Value: ⭐⭐⭐⭐ Unifies human and scene reconstruction, holding significant importance for understanding real-world human-environment interactions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)
- [\[CVPR 2025\] Reconstructing Humans with a Biomechanically Accurate Skeleton](reconstructing_humans_with_a_biomechanically_accurate_skeleton.md)
- [\[CVPR 2025\] Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [\[CVPR 2026\] TROPHIES: Temporal Reconstruction of Places, Humans, and Cameras from Multi-view Videos](../../CVPR2026/3d_vision/trophies_temporal_reconstruction_of_places_humans_and_cameras_from_multi-view_vi.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)

</div>

<!-- RELATED:END -->
