---
title: >-
  [Paper Note] ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate
description: >-
  [CVPR 2025][Autonomous Driving][Human Motion Recovery] This work constructs the first large-scale multi-modal rock climbing motion dataset, AscendMotion (412K frames, RGB+LiDAR+IMU), and proposes ClimbingCap, a method that accurately recovers the 3D motions of climbers in the world coordinate system through separate coordinate decoding, post-processing optimization, and semi-supervised training.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Human Motion Recovery"
  - "Rock Climbing"
  - "Multi-Modal"
  - "LiDAR"
  - "RGB"
  - "SMPL"
date: 2026-05-08
content_hash: 454824f212a3428f
---

# ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate

**Conference**: CVPR 2025  
**arXiv**: [2503.21268](https://arxiv.org/abs/2503.21268)  
**Code**: [http://www.lidarhumanmotion.net/climbingcap/](http://www.lidarhumanmotion.net/climbingcap/)  
**Area**: Autonomous Driving (Human Motion)  
**Keywords**: Human Motion Recovery, Rock Climbing, Multi-Modal, LiDAR, RGB, SMPL

## TL;DR

This work constructs the first large-scale multi-modal rock climbing motion dataset, AscendMotion (412K frames, RGB+LiDAR+IMU), and proposes ClimbingCap, a method that accurately recovers the 3D motions of climbers in the world coordinate system through separate coordinate decoding, post-processing optimization, and semi-supervised training.

## Background & Motivation

1. **Background**: Human Motion Recovery (HMR) research primarily focuses on **ground motions** such as running and walking. Existing methods like WHAM and GVHMR assume that the human body mainly moves along the horizontal direction when estimating global trajectories. While this assumption holds for ground activities, it cannot generalize to other scenarios.

2. **Limitations of Prior Work**: Rock climbing is an **off-ground motion** where climbers move upward by clinging to the wall with hands and feet, involving extreme limb extension and full-body exertion. Existing global HMR methods face two core issues in rock climbing scenarios: (1) inability to accurately recover the climber's position in the world coordinate system (especially in the vertical direction), and (2) inherent ambiguity in the conversion between camera and global coordinates.

3. **Key Challenge**: Rock climbing datasets are severely insufficient. Previously, there were only two public datasets, SPEED21 (2D, 46K frames) and CIMI4D (3D, 180K frames, amateur climbers), which are small-scale and lack challenging motions. The scarcity of data hampers the community's profound understanding of rock climbing.

4. **Goal**: (1) To construct a large-scale, high-quality rock climbing dataset; (2) to design a method capable of accurately recovering rock climbing motions in the global coordinate system; and (3) to effectively utilize easily obtainable unlabeled rock climbing data.

5. **Key Insight**: RGB and LiDAR modalities each have unique strengths—RGB is suitable for estimating pose in camera coordinates, while LiDAR's 3D point clouds naturally provide world coordinate information. Decoding them separately in different coordinate systems and then jointly optimizing them can effectively resolve the coordinate system ambiguity.

6. **Core Idea**: Utilizing RGB+LiDAR multi-modalities to decode poses separately in camera and global coordinates, followed by correcting the global trajectory of rock climbing motions via scene-aware post-processing and semi-supervised training.

## Method

### Overall Architecture

ClimbingCap consists of three stages: (1) Separate Coordinate Decoding (SCD)—RGB estimates the SMPL parameters in camera coordinates, while LiDAR estimates translation parameters in global coordinates; (2) Post-Processing Optimization—jointly optimizes the outcomes of both coordinate systems using three specialized loss functions; and (3) Semi-Supervised Training—leverages a teacher-student framework to utilize a large volume of unlabeled climbing data.

### Key Designs

1. **Separate Coordinate Decoding**:
    - **Function**: Predicts human motion parameters separately in camera and global coordinate systems.
    - **Mechanism**: Point clouds are transformed from world coordinates to camera coordinates via the extrinsic matrix $\mathcal{P}_c = \Omega_{w2c} \cdot \mathcal{P}_w$. RGB images extract visual features through ViT, and point clouds extract geometric features through PointNet++. The Camera Coordinate Decoder iteratively updates the SMPL pose $	heta$, shape $eta$, and camera translation $\Delta c$, while the Global Coordinate Decoder iteratively updates the global translation $\Gamma^{trans}$.
    - **Design Motivation**: The information sources of camera coordinates and global coordinates differ (RGB is proficient at pose estimation, whereas LiDAR excels at positioning). Separate decoding leverages the strengths of both.

2. **Post-Processing Optimization (Three Loss Functions)**:
    - **Function**: Jointly optimizes the pose and position output from the SCD stage in the world coordinate system.
    - **Global Refit Loss ($\mathcal{L}_{GR}$)**: Calculates the weighted Chamfer distance between SMPL vertices and human point clouds, imposing different distance thresholds on the torso and limbs ({limb} \leq d_{torso}$ because limbs are closer to the climbing wall during climbing).
    - **Scene Touch Loss ($\mathcal{L}_{ST}$)**: Prevents the SMPL model from penetrating the climbing wall scene mesh by calculating the penetration depth $\eta(v_i) = (v_i - q_j) \cdot n_j$, penalizing penetration.
    - **Velocity Smoothing Loss ($\mathcal{L}_{VLR}$)**: Constraints the directional smoothness of limb velocity to correct abnormal limb predictions.
    - **Design Motivation**: Climbers move closely along the climbing wall, making human-scene interaction constraints an indispensable prior. LiDAR provides 3D information in world coordinates, which enables such post-processing.

3. **Semi-Supervised Training**:
    - **Function**: Leverages unlabeled climbing data to improve model performance.
    - **Mechanism**: The model trained with SCD and post-processing serves as a teacher model, cloning its parameters to a student model. The teacher model generates pseudo-labels for unlabeled data, on which the student model is further trained.
    - **Design Motivation**: Annotating climbing data is labor-intensive (requiring IMU MoCap + manual correction), whereas unlabeled climbing videos/point clouds are easy to acquire. AscendMotion contains 441 minutes of unlabeled data (vs. 344 minutes of labeled data).

### Loss & Training

- Total loss in the SCD stage: $\mathcal{L} = \mathcal{L}_{kp3d} + \mathcal{L}_{kp2d} + \mathcal{L}_	heta^{smpl} + \mathcal{L}_eta^{smpl} + \mathcal{L}_{traj}$
- The post-processing stage optimizes global poses using the Adam optimizer.
- The semi-supervised stage adopts a teacher-student framework.

## Key Experimental Results

### Main Results (AscendMotion Horizontal/Vertical Scenarios)

| Method | Modality | MPJPE↓ | PA-MPJPE↓ | PCK@0.3↑ | WA-MPJPE↓ | W-MPJPE↓ | RTE↓ |
|------|------|--------|-----------|----------|-----------|----------|------|
| TRACE | RGB | 875.56/577.60 | 69.21/85.81 | 0.06/0.09 | 144.33/385.71 | 254.38/703.35 | 14.73/26.17 |
| GVHMR | RGB | 107.09/124.60 | 60.06/80.30 | 0.77/0.71 | 105.15/1002.11 | 202.45/1442.50 | 4.09/7.91 |
| WHAM | RGB | 110.92/143.17 | 76.09/73.36 | 0.76/0.62 | 229.42/1125.77 | 647.70/1499.85 | 5.16/9.04 |
| LiDARCapV2 | LiDAR | 244.60/234.52 | 192.17/156.39 | 0.53/0.50 | 282.12/1396.42 | 442.12/1518.29 | 16.42/10.85 |
| LEIR | L+R | 297.95/299.62 | 187.26/150.56 | 0.41/0.37 | 266.82/1313.09 | 282.31/1435.92 | 9.78/9.97 |
| **ClimbingCap** | **L+R** | **75.45/88.92** | **61.73/74.50** | **0.91/0.78** | **62.95/85.26** | **78.99/106.95** | **1.57/3.12** |

ClimbingCap significantly outperforms all baselines, with an especially pronounced advantage in vertical scenarios (GVHMR's W-MPJPE reaches up to 1442mm, whereas ClimbingCap achieves only 107mm). Zero-shot transfer to the CIMI4D dataset also achieves state-of-the-art results (MPJPE of 84.03mm).

### Ablation Study

| Configuration | MPJPE↓ | PA-MPJPE↓ | PCK@0.3↑ | WA-MPJPE↓ | W-MPJPE↓ | RTE↓ |
|------|--------|-----------|----------|-----------|----------|------|
| (1) RGB input only | 105.67 | 63.05 | 0.78 | 117.17 | 174.53 | 7.64 |
| (2) w/o $\mathcal{L}_{LWD}$ | 80.46 | 52.15 | 0.89 | 70.04 | 91.23 | 2.02 |
| (3) w/o $\mathcal{L}_{SDS}$ | 99.13 | 60.66 | 0.81 | 109.35 | 164.11 | 7.03 |
| (4) w/o $\mathcal{L}_{VLR}$ | 91.10 | 61.85 | 0.83 | 88.59 | 120.11 | 3.34 |
| (5) w/o semi-supervised | 77.43 | 52.57 | 0.90 | 65.30 | 82.09 | 1.83 |
| **Full Model** | **75.45** | **50.51** | **0.91** | **62.95** | **78.99** | **1.57** |

### Key Findings

- **LiDAR is Indispensable**: Removing LiDAR increases the MPJPE from 75.45 to 105.67 (+40%), and global trajectory metrics deteriorate significantly (with W-MPJPE more than doubling).
- **Velocity Smoothing Loss $\mathcal{L}_{SDS}$ is Most Crucial**: Removing it increases the MPJPE from 75.45 to 99.13, which is the most significant impact, demonstrating that motion consistency constraints are vital for rock climbing.
- **Scene Touch Loss Contribution is Significant**: Removing $\mathcal{L}_{VLR}$ raises the MPJPE to 91.10, severely degrading the global trajectory.
- **Semi-Supervised Learning Helps but with Limited Contribution**: Removing it only increases the MPJPE by 2mm (75.45→77.43), indicating that the primary improvement comes from the methodological design rather than the data volume.

## Highlights & Insights

- **Ingenuity in Dataset Construction**: The annotation pipeline of AscendMotion (PTP clock synchronization + multi-stage global optimization + manual correction) is elaborately designed. Data from 22 professional climbing instructors is more challenging than CIMI4D's amateur climbers—dataset quality triumphs over mere scale.
- **Separate Coordinate Decoding**: Separating the decoding of camera coordinates and global coordinates is a clever design. RGB is excellent for pose estimation but lacks depth information, while LiDAR naturally provides 3D global positions—allowing different modalities to focus on what they do best, and unifying them during the post-processing stage.
- **Scene-Aware Loss Design**: Scene Touch Loss utilizes the physical prior of climbers tightly clinging to the wall to encode human-scene interaction as optimization constraints, representing an effective application of domain knowledge.

## Limitations & Future Work

- The high hardware requirements for data acquisition (LiDAR + RGB + IMU + high-precision scanner) make it difficult to scale and popularize.
- The method assumes precise extrinsic matrices for coordinate transformation, which may be unreliable for calibration in wild outdoor scenarios.
- The contribution of semi-supervised training is limited (~2mm), potentially requiring more advanced pseudo-labeling strategies.
- Fine-grained interaction modeling of the climber's hand-foot grasping on holds is not considered.

## Related Work & Insights

- **vs GVHMR**: GVHMR estimates the global motion direction by predicting horizontal velocity, which completely fails in vertical climbing scenarios (W-MPJPE reaches up to 1442mm), indicating that the ground motion assumption is a major drawback.
- **vs WHAM**: WHAM achieves decent metrics in camera coordinates but extremely poor metrics in global coordinates, further confirming that RGB-only input cannot reliably estimate the global position of off-ground motion.
- **vs CIMI4D Dataset**: AscendMotion comprehensively outperforms CIMI4D across three dimensions: scale (412K vs. 180K frames), diversity (22 subjects on 12 walls vs. 12 subjects), and level of challenge (professional instructors vs. amateurs).

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematical solution to the rock climbing motion recovery task, with both the dataset and formulation representing substantial contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dataset cross-validation, zero-shot generalization testing, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems and detailed descriptions of the dataset construction.
- Value: ⭐⭐⭐⭐ The AscendMotion dataset holds long-term value for the community, and the design of the method is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LiSu: A Dataset and Method for LiDAR Surface Normal Estimation](lisu_a_dataset_and_method_for_lidar_surface_normal_estimation.md)
- [\[ICCV 2025\] UAVScenes: A Multi-Modal Dataset for UAVs](../../ICCV2025/autonomous_driving/uavscenes_a_multi-modal_dataset_for_uavs.md)
- [\[CVPR 2025\] Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method](towards_satellite_image_road_graph_extraction_a_global-scale_dataset_and_a_novel.md)
- [\[CVPR 2025\] Distilling Multi-modal Large Language Models for Autonomous Driving](distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)

</div>

<!-- RELATED:END -->
