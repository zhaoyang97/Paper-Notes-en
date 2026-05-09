---
title: >-
  [Paper Note] InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset
description: >-
  [CVPR 2026][Autonomous Driving][Camera Pose Estimation] This paper presents InCaRPose, an in-cabin relative camera pose estimation model built upon a frozen ViT backbone and a Transformer decoder. Trained exclusively on synthetic data, it generalizes to real in-cabin environments, achieving metric-scale translation prediction and real-time inference (>45 FPS). The authors also release an accompanying real-world, high-distortion in-cabin test dataset, In-Cabin-Pose.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Camera Pose Estimation
  - In-Cabin Perception
  - Fisheye Camera
  - Sim-to-Real Transfer
  - Transformer
date: 2026-05-08
content_hash: 7f9d8cb237ef6f5f
---

# InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset

**Conference**: CVPR 2026
**arXiv**: [2604.03814](https://arxiv.org/abs/2604.03814)
**Code**: [https://github.com/felixstillger/InCaRPose](https://github.com/felixstillger/InCaRPose)
**Area**: Autonomous Driving / 3D Vision
**Keywords**: Camera Pose Estimation, In-Cabin Perception, Fisheye Camera, Sim-to-Real Transfer, Transformer

## TL;DR

This paper presents InCaRPose, an in-cabin relative camera pose estimation model built upon a frozen ViT backbone and a Transformer decoder. Trained exclusively on synthetic data, it generalizes to real in-cabin environments, achieving metric-scale translation prediction and real-time inference (>45 FPS). The authors also release an accompanying real-world, high-distortion in-cabin test dataset, In-Cabin-Pose.

## Background & Motivation

**Background**: Camera extrinsic calibration is a fundamental task in computer vision. In in-cabin monitoring (ICAM) scenarios, cameras are used for driver monitoring, occupant pose estimation, and airbag control. Existing methods rely on geometric feature matching and epipolar geometry, or on deep learning models trained on large-scale data.

**Limitations of Prior Work**: In-cabin environments present several unique challenges: (1) rearview-mirror-mounted cameras frequently change their extrinsics as drivers adjust them; (2) in-cabin cameras typically use wide-angle/fisheye lenses, introducing severe distortion; (3) cameras operate in the near-infrared (NIR) spectrum, differing from standard RGB imagery; (4) airbag control requires occupant location within 15–50 ms post-collision, demanding metric-scale absolute translation estimation and real-time inference. Existing methods such as Reloc3r can only predict translation direction rather than absolute distance, and require large-scale training data.

**Key Challenge**: Existing general-purpose pose estimation models either require large amounts of training data and specific camera intrinsics, or provide only scale-ambiguous translation directions, failing to jointly satisfy the in-cabin safety application requirements of small-data training, metric-scale translation, fisheye distortion handling, and real-time inference.

**Goal**: (1) Accurately estimate relative pose in real in-cabin environments using only synthetic training data; (2) directly process highly distorted fisheye images without undistortion; (3) predict metric-scale absolute translation; (4) perform real-time inference to support time-critical safety applications.

**Key Insight**: The problem is reformulated as reference-relative pose estimation, avoiding dependence on a vehicle-specific coordinate frame. A frozen self-supervised ViT backbone (DINOv3) is used to extract domain-invariant features, enabling cross-domain transfer with a small amount of synthetic data.

**Core Idea**: Combine a frozen DINOv3 backbone, a Transformer cross-attention decoder, and lightweight prediction heads, trained on synthetic in-cabin data, to achieve metric-scale relative pose estimation in real in-cabin environments.

## Method

### Overall Architecture

The input consists of a reference view and a target view. Patch-level features are extracted via a frozen ViT backbone and fed into a Transformer cross-attention decoder that fuses information from both views. An MLP prediction head then regresses the relative rotation and translation. The backbone is fully frozen; only the decoder and prediction heads are trained. Bidirectional pose prediction can optionally be enabled to strengthen geometric consistency supervision.

### Key Designs

1. **Frozen ViT Backbone and Reference-Relative Formulation**:

    - Function: Extract domain-invariant features and eliminate dependence on vehicle-specific coordinate frames.
    - Mechanism: DINOv3 (or DINOv2, DUNE) is used as a frozen feature extractor, producing patch-level latent features for each image independently. Pose estimation is reformulated as: given a calibrated reference pose $T_{v1}$, estimate the relative transform $T_{rel}$ such that $T_{v2} = T_{v1} \cdot T_{rel}$. During training, $T_{rel} = T_{v1}^{-1} T_{v2}$. This reference-relative formulation is vehicle-agnostic and requires no retraining per vehicle model.
    - Design Motivation: Freezing the backbone provides two benefits: (1) it preserves robust cross-domain feature representations acquired during pretraining; (2) it prevents noisy gradients from randomly initialized components from corrupting fine-grained features in early training. The reference-relative formulation eliminates the need for a global vehicle coordinate system.

2. **Transformer Cross-Attention Decoder**:

    - Function: Fuse spatial features from both views to capture geometric relationships.
    - Mechanism: Backbone tokens are linearly projected and passed through multiple decoder blocks. Each block consists of self-attention (for feature refinement) and cross-attention (for cross-view interaction), with LayerNorm and residual connections. 2D RoPE (Rotary Position Embedding) is introduced to inject spatial positional information into queries and keys, avoiding learned positional tokens, which is more reliable under limited training data. The decoder uses 12 attention heads and an MLP expansion ratio of 4.
    - Design Motivation: Cross-attention is critical for capturing spatial relationships between the two views. 2D RoPE generalizes better than learned positional encodings in low-data regimes. Residual convolutional bottlenecks are used for dimensionality reduction and feature fusion.

3. **Multiple Output Representations and Bidirectional Prediction**:

    - Function: Support flexible pose parameterizations and provide stronger supervision signals.
    - Mechanism: Five pose parameterizations are supported: rotation vector, intrinsic/extrinsic Euler angles (6D), quaternion (7D), and rotation matrix (12D). Post-processing enforces valid rotations: quaternions are normalized, and rotation matrices are orthogonalized via SVD with $\det = +1$ enforced. During training, bidirectional prediction (simultaneously predicting forward and inverse relative poses) is enabled without requiring image-order augmentation. At inference, bidirectional prediction can be disabled to maximize speed.
    - Design Motivation: Bidirectional prediction forces the network to learn bidirectional camera transformation relationships, providing a consistency constraint as an additional supervision signal. Empirically, quaternion representation yields the best results.

### Loss & Training

AdamW optimizer with learning rate $1 \times 10^{-6}$, weight decay $1 \times 10^{-5}$, and batch size 8. Synthetic training data is rendered in Blender using 11 different vehicles (8 for training, 3 for validation), with randomly placed mannequins and objects. Rotations (±80° x/y, ±50° z) and translations (±20 cm per axis) are uniformly sampled, yielding approximately 5,000 pure-rotation pairs and 1,500 rotation-plus-translation pairs. ColorJitter augmentation is applied to prevent overfitting. Fisheye images are processed directly without undistortion (scaled with zero-padding to preserve the full field of view).

## Key Experimental Results

### Main Results

**In-Cabin-Pose Real-World Dataset**:

| Model | Rotation Error (°) Median | Translation Error (m) Median | Direction Error (°) Median |
|-------|---------------------------|------------------------------|----------------------------|
| InCaRPose-Small224 | 4.43 | 0.08 | 37.74 |
| InCaRPose-Base224 | 3.55 | 0.09 | 42.45 |
| **InCaRPose-Large224** | **2.75** | **0.07** | **23.46** |
| Reloc3r224 (no undistortion) | 12.73 | – | 76.79 |
| Reloc3r512 (undistorted) | 3.23 | – | 13.05 |
| SIFT Matching (undistorted) | 4.83 | – | 28.30 |

**7-Scenes Indoor Dataset**:

| Model | Rotation Error (°) Mean | Translation Error (m) Median |
|-------|-------------------------|------------------------------|
| RelPoseNet | 9.30 | 0.21 |
| Relformer | 6.27 | 0.18 |
| RelPoseGNN | 5.20 | 0.17 |
| Reloc3r224 | 7.96 | – |
| **InCaRPose-Large224** | **2.55** | **0.13** |

### Ablation Study

| Configuration | Rotation Mean (°) | Translation Mean (m) | Note |
|---------------|-------------------|----------------------|------|
| InCaRPose-Small | 6.11 | 0.11 | Fastest, lower accuracy |
| InCaRPose-Base | 4.91 | 0.12 | Intermediate |
| InCaRPose-Large | **4.15** | **0.10** | Best accuracy |
| DINOv3-Base | 4.91 | 0.12 | Standard backbone |
| DUNE-Base504 | **3.87** | 0.12 | Slightly better with DUNE backbone |

**Inference Speed (RTX 4090, single GPU)**:

| Configuration | FPS |
|---------------|-----|
| InCaRPose-Small224 | ~70 |
| InCaRPose-Base224 | ~67 |
| InCaRPose-Large224 | >45 |

### Key Findings

- InCaRPose-Large achieves a median rotation error of 2.75° and translation error of 0.07 m on real in-cabin data, demonstrating strong sim-to-real transfer despite being trained solely on synthetic data.
- On 7-Scenes, the mean rotation error of 2.55° is 65% lower than Reloc3r's 7.96°, while also providing metric translation.
- A frozen backbone is essential: DINOv3's pretrained features exhibit strong cross-domain generalization.
- Larger backbones yield greater improvements on high-distortion fisheye images (Small→Large: 6.11°→4.15° on in-cabin data), while the gap is smaller on the standard images in 7-Scenes.
- All configurations maintain real-time performance (>45 FPS), with Small and Base approaching 70 FPS.

## Highlights & Insights

- **Synthetic Training with Real-World Generalization**: Using only approximately 6,500 synthetic image pairs, the model achieves strong performance in real in-cabin environments, attributable to the domain-invariant features of the frozen DINOv3 backbone. The paradigm of "frozen foundation model + lightweight task head" is highly valuable in data-scarce scenarios.
- **End-to-End Fisheye Processing**: Directly processing fisheye images without undistortion is a pragmatic design choice—geometric cues near image boundaries are important for pose estimation, and undistortion discards information. This also eliminates the computational overhead of undistortion in deployment.
- **Generalizability of the Reference-Relative Formulation**: By avoiding vehicle-specific coordinate systems, the same model can be deployed across different vehicle models, which is critical for production-scale applications.

## Limitations & Future Work

- Translation direction error remains relatively large (median 23.46°), and translation estimation under extreme z-axis motion remains challenging.
- The real-world test dataset is collected from a single vehicle interior; cross-vehicle generalization requires further validation.
- The supported translation range is limited to in-cabin camera adjustment ranges (±20 cm); large-displacement scenarios have not been evaluated.
- Future work could explore leveraging multi-frame temporal information or integrating IMU measurements.

## Related Work & Insights

- **vs. Reloc3r (large-scale trained pose estimation)**: Reloc3r is trained on large datasets but predicts only translation direction, without metric distance. InCaRPose uses far less training data yet provides metric translation, making it more suitable for safety-critical applications.
- **vs. SIFT + RANSAC (classical methods)**: Classical methods degrade on distorted images and suffer from scale ambiguity. InCaRPose processes distorted images end-to-end directly.
- **vs. PoseNet / SCR (absolute pose methods)**: Absolute methods require scene-specific training or dense 3D reconstruction. InCaRPose's reference-relative formulation is more flexible.

## Rating

- Novelty: ⭐⭐⭐ — Methodological innovation is limited (frozen backbone + Transformer + MLP); contributions lie primarily in problem formulation and dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated on three datasets (real in-cabin, 7-Scenes, Cambridge Landmarks) with detailed speed analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated with thorough engineering details.
- Value: ⭐⭐⭐⭐ — The dataset and problem formulation offer direct value to the in-cabin perception community and address practical safety requirements.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multimodal_learning_in_3d_human_p.md)
- [\[CVPR 2026\] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [\[CVPR 2026\] EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)
- [\[CVPR 2026\] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)
- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](../../ICCV2025/autonomous_driving/6dope-gs_online_6d_object_pose_estimation_using_gaussian_splatting.md)

<!-- RELATED:END -->
