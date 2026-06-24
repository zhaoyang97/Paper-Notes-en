---
title: >-
  [Paper Note] Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection
description: >-
  [ECCV 2024][Autonomous Driving][LiDAR 3D Object Detection] E-SSL3D proposes a joint spatio-temporal equivariant self-supervised pre-training framework. By jointly training the 3D feature encoder with spatial equivariance (using a classification objective for rotation, and contrastive objectives for translation/scaling/flipping) and temporal equivariance (using 3D scene flow to constrain the consistency of feature transformations between adjacent frames)…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "LiDAR 3D Object Detection"
  - "Self-Supervised Learning"
  - "Equivariance"
  - "Scene Flow"
  - "Pre-training"
date: 2026-05-08
content_hash: d3a4aec3cd09633b
---

# Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection

**Conference**: ECCV 2024  
**arXiv**: [2404.11737](https://arxiv.org/abs/2404.11737)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: LiDAR 3D Object Detection, Self-Supervised Learning, Equivariance, Scene Flow, Pre-training

## TL;DR

E-SSL3D proposes a joint spatio-temporal equivariant self-supervised pre-training framework. By jointly training the 3D feature encoder with spatial equivariance (using a classification objective for rotation, and contrastive objectives for translation/scaling/flipping) and temporal equivariance (using 3D scene flow to constrain the consistency of feature transformations between adjacent frames), the detector achieves 3D object detection performance close to training from scratch with 100% data while using only 20% labeled data in low-data scenarios.

## Background & Motivation

**Background**: LiDAR 3D object detection is crucial for autonomous driving, but point cloud labeling is expensive and difficult. Self-supervised learning (SSL) can leverage massive unlabeled LiDAR data to learn general representations.

**Core Problem — Invariance vs. Equivariance**:
   - Mainstream SSL methods (such as BYOL, MoCo) encourage features to remain **invariant** to transformations.
   - However, the regression output of 3D object detection is naturally **equivariant**—if the input is rotated, the bounding boxes should also rotate by the same angle.
   - Training for invariance contradicts detection equivariance: encouraging rotation invariance leads to the loss of orientation information.

**Limitations of Prior Work**:
   - PointContrast only performs contrastive learning on rigid transformations.
   - STRL uses the BYOL framework to encourage invariance across time, failing to preserve transformation information.
   - ALSO uses occupancy prediction for pre-training, which is a generative method and specific to network architectures.
   - They lack modeling of real-world object motion and deformation.

**Key Insight**: Equivariant representations should be learned by simultaneously utilizing spatial geometric transformations and real object motion between sequential temporal frames. Different transformations should employ different equivariant learning objectives.

## Method

### Overall Architecture

E-SSL3D consists of three components trained jointly:
1. **Spatial Equivariance Branch**: Learns equivariant features for rigid geometric transformations (rotation, translation, scaling, flipping).
2. **Temporal Equivariance Branch**: Learns temporal equivariant features using adjacent LiDAR frames and 3D scene flow.
3. **Joint Optimization**: Weighted sum of three loss functions.

The network architecture includes: a 3D feature encoder $f$, a projector $m$, a predictor $q$, a transformation classifier $s$, and a target network $(f', m')$ updated via EMA.

### Key Designs

**1. Spatial Equivariance — Contrastive Learning + Transformation Classification**

Two rigid transformations are randomly sampled for each input point cloud to create two augmented views.

**(a) Point-level Contrastive Loss PointInfoNCE**:
$$\mathcal{L}_{pnce} = -\sum_{(i,j) \in \mathcal{P}_+} \log \frac{\exp(\mathbf{x}_i \cdot \mathbf{x}_j / \tau)}{\sum_{(\cdot,k) \in \mathcal{P}_+} \exp(\mathbf{x}_i \cdot \mathbf{x}_k / \tau)}$$
- Encourages features of matched points to be similar and unmatched points to be different, sampling 2048 points.

**(b) Equivariance-by-Classification**:
- Discretizes transformations (e.g., dividing rotation into 10 angles) and uses classifier $s$ to predict which transformation was applied.
- Trains the network to preserve transformation information via cross-entropy loss $\mathcal{L}_{ce}$.
- **Key Findings**: Rotation is suitable for the classification objective (large transformation magnitude, distinguishable via 10-fold classification), while translation and scaling are suitable for contrastive objectives (small transformation range, hard to distinguish via classification).

**2. Temporal Equivariance — 3D Scene Flow**

- Uses real object motion between consecutive LiDAR frames as natural temporal augmentation.
- Estimates 3D scene flow $d_{t-1 \to t} \in \mathbb{R}^3$ using a pre-trained PV-RAFT network.

**(a) Online-Target Network in BYOL Framework**:
- The online network processes the current frame $p_t$, and the target network processes the previous frame $p_{t-1}$.
- The target network is updated via EMA to avoid representation collapse: $\xi \leftarrow \gamma\xi + (1-\gamma)\theta$.

**(b) Scene Flow Warp in Feature Space**:
- Scene flow is a point-level transformation, but features are in sparse voxel representations.
- Warps previous-frame points to current-frame positions using scene flow, re-voxelizes to obtain warped voxel coordinates, and samples features from $h_{t-1}$ to get $h_{t-1}^{warp}$.

**(c) Flow Equivariance Loss**:
$$\mathcal{L}_{flow} = \frac{1}{HW}\|\hat{z}_{t-1} - \hat{y}_t\|_2^2$$
- Minimizes the L2 distance between the warped previous-frame projected features and the current-frame predicted features.

**3. Joint Loss Function**
$$\mathcal{L} = \lambda_{pnce}\mathcal{L}_{pnce} + \lambda_{ce}\mathcal{L}_{ce} + \lambda_{flow}\mathcal{L}_{flow}$$
where $\lambda_{pnce}=0.01$, $\lambda_{ce}=1$, and $\lambda_{flow}=300$.

### Loss & Training

- Pre-training data: KITTI-360 + SemanticKITTI (with validation sequences removed).
- Uses front field-of-view (FFOV) scenes to reduce the pre-training-to-fine-tuning distribution gap.
- Spatial augmentation: rotation $(-\pi/2, \pi/2)$, translation $(0m, 0.2m)$, scaling $(0.95, 1.05)$, 50% probability flip.
- AdamW optimizer, cyclic learning rate of $10^{-4}$, 80 epochs, batch size 56, 8x A6000.
- Initial EMA decay $\gamma_{base}=0.999$.
- Downstream fine-tuning: KITTI dataset, AdamW, learning rate $3 \times 10^{-3}$, 80 epochs.

## Key Experimental Results

### Main Results

**3D mAP of SECOND Detector on KITTI (Pre-trained on KITTI-360)**:

| Data Ratio | No Pre-training | PointContrast | STRL | ALSO | **E-SSL3D** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5% | 54.99 | 55.40 | 52.94 | **59.69** | 58.60 |
| 20% | 61.86 | 61.72 | 61.74 | **66.39** | 65.28 |
| 100% | 66.69 | 65.74 | 67.78 | 69.18 | **69.81** |

**3D mAP of VoxelRCNN Detector on KITTI**:

| Data Ratio | No Pre-training | PointContrast | STRL | ALSO | **E-SSL3D** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5% | 65.54 | 64.96 | 63.87 | 67.00 | 66.52 |
| 20% | 68.78 | 68.97 | 69.59 | 70.95 | **71.63** |
| 100% | 71.77 | 70.72 | 70.71 | 72.06 | **72.41** |

Fine-tuning after pre-training with 20% data can yield performance close to training from scratch with 100% data.

### Ablation Study

**Effect of Joint Spatio-Temporal Equivariance (VoxelRCNN, 5% KITTI)**:

| Spatial Equivariance | Temporal Equivariance | Car(M) | Ped(M) | Cyc(M) | mAP |
|:---:|:---:|:---:|:---:|:---:|:---:|
| No | No | 78.85 | 49.13 | 58.62 | 64.50 |
| No | Yes | 77.80 | 49.73 | 61.74 | 65.82 |
| Yes | No | 77.34 | 50.34 | 61.71 | 66.01 |
| Yes | Yes | **78.93** | 48.55 | **64.40** | **66.52** |

**Comparison of Equivariant Learning Methods**: Rotation using classification is better than contrastive (large transformation magnitude); translation/scaling using contrastive is better than classification (small transformation range).

### Key Findings

- **Equivariance is superior to invariance**: E-SSL3D systematically outperforms the invariance-based method STRL.
- **Different transformations require different equivariant objectives**: The strategy of using classification for rotation and contrastive for translation outperforms unified approaches.
- **Temporal equivariance is an effective signal**: 3D scene flow provides real object motion information unmatched by spatial augmentation.
- **Most beneficial in low-data scenarios**: The improvement is most significant in the 20% data scenario, with diminishing returns of pre-training at 100% data.
- **Limited improvement for Car class**: This class has sufficient samples in KITTI and is relatively "simple," leaving little room for pre-training benefits.
- **Faster convergence than ALSO**: E-SSL3D converges in 10-20 epochs, while ALSO requires 75 epochs.

## Highlights & Insights

- **First to introduce 3D scene flow to equivariant self-supervised learning**: Utilizing real-world object motion as temporal augmentation.
- **Systematic study of optimal equivariant objectives for different transformations**: Revealing the relationship between transformation magnitude and equivariant learning strategies.
- **Universal pre-training framework**: The same pre-trained backbone can be applied to various detectors like SECOND and VoxelRCNN.
- **Key challenge of equivariance vs. invariance**: Standard SSL encourages invariance, but downstream regression tasks require equivariance.

## Limitations & Future Work

- Limited pre-training improvement for the Car class (simple/dense samples).
- Not always superior to ALSO (generative method); combining the two might unlock better results.
- Decreased impact of pre-training at 100% data, with main value lies in low-data scenarios.
- Scene flow estimation depends on a frozen pre-trained model; end-to-end learning might be superior.
- Only validated on sparse convolutional backbones, not yet extended to Transformer-based detectors.

## Related Work & Insights

- **PointContrast**: Pioneer of point-level contrastive learning; E-SSL3D extends its equivariance.
- **BYOL**: The foundation of the online-target network framework, adapted for the temporal equivariance branch.
- **ALSO**: Representative of generative approaches based on occupancy prediction pre-training.
- **FlowE**: Flow equivariance method in the image domain; E-SSL3D generalizes it to 3D LiDAR.
- **Insight**: SSL objectives should align with the geometric properties of downstream tasks.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Engineering Practicality | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

**Classification Objective**: For rotation transformations, continuous rotations are discretized into 10 categories, and the applied rotation angle is predicted through an extra classification head trained with cross-entropy loss $\mathcal{L}_{ce}$. Key finding: classification is more effective for rotation, while contrastive is more effective for translation/scaling. This is because rotation has a large range ($-\pi/2$ to $\pi/2$) and obvious differences between different angles, whereas translation and scaling have small ranges and are hard to classify finely.

### Temporal Equivariance: 3D Scene Flow Equivariance

Utilizes sequential LiDAR frame pairs $(p_{t-1}, p_t)$ to model natural motion transformations. 3D scene flow describes the displacement vector $d_{t-1\to t} \in \mathbb{R}^3$ of each point in the point cloud from time $t-1$ to $t$.

An online-target dual-branch structure similar to BYOL is adopted:
1. Uses a pre-trained PV-RAFT network to estimate the 3D scene flow between consecutive frames.
2. Warps the previous-frame features $h_{t-1}$ into the current-frame coordinate system via the scene flow to get $h_{t-1}^{warp}$.
3. Minimizes the L2 distance between the warped features and the current-frame ground-truth features.

$$\mathcal{L}_{flow} = \frac{1}{HW} \|\hat{z}_{t-1} - \hat{y}_t\|_2^2$$

Target network parameters are updated via EMA of the online network ($\gamma_{base}=0.999$) to avoid representation collapse.

### Overall Loss Function

$$\mathcal{L} = \lambda_{pnce}\mathcal{L}_{pnce} + \lambda_{ce}\mathcal{L}_{ce} + \lambda_{flow}\mathcal{L}_{flow}$$

where $\lambda_{pnce}=0.01$, $\lambda_{ce}=1$, and $\lambda_{flow}=300$.

### Feature Warp Details

3D feature maps are sparse tensors (voxel features + 3D coordinates). Applying flow to the previous-frame point cloud yields the warped point cloud $p_t^{warp}$, which is then standardly voxelized to obtain new coordinates. Voxel features are sampled from $h_{t-1}$ at corresponding positions to formulate $h_{t-1}^{warp}$.

## Key Experimental Results

### Pre-training Settings
- Dataset: KITTI-360 (100k scenes) + SemanticKITTI (48k scans)
- Backbone: SparseVoxel backbone (shared with SECOND/VoxelRCNN)
- Training: 8× A6000 GPUs, batch size 56, 80 epochs, but converges in about 10-20 epochs.

### SECOND Detector (KITTI fine-tune)

| Data Ratio | Method | Car moderate | Cyclist moderate | mAP |
|:---:|:---:|:---:|:---:|:---:|
| 5% | No Pre-training | 73.01 | 45.44 | 54.99 |
| 5% | E-SSL3D | 74.96 | 54.13 | **58.60** |
| 20% | No Pre-training | 77.12 | 54.99 | 61.86 |
| 20% | E-SSL3D | 78.70 | 62.92 | **65.28** |
| 100% | No Pre-training | 81.03 | 63.54 | 66.69 |
| 100% | E-SSL3D | 81.64 | 69.54 | **69.81** |

### VoxelRCNN Detector (KITTI fine-tune)

| Data Ratio | Method | mAP |
|:---:|:---:|:---:|
| 5% | No Pre-training | 65.54 |
| 5% | E-SSL3D | 66.52 |
| 20% | No Pre-training | 68.78 |
| 20% | E-SSL3D | **71.63** |
| 100% | E-SSL3D | **72.41** |

**Key Conclusion**: 20% labeled data + E-SSL3D pre-training $\approx$ 100% data training from scratch.

### Ablation Study (VoxelRCNN, 5% Data)

| Spatial Equivariance | Temporal Equivariance | mAP |
|:---:|:---:|:---:|
| ✗ | ✗ | 64.50 |
| ✗ | ✓ | 65.82 |
| ✓ | ✗ | 66.01 |
| ✓ | ✓ | **66.52** |

Both spatial and temporal equivariance make independent contributions, and using them jointly yields the best results.

## Highlights & Insights

1. **Deep insight into equivariance vs. invariance**: Explicitly points out that the 3D detection task itself is equivariant, so pre-training should also encourage equivariance rather than invariance. This perspective has a stronger theoretical foundation than blindly applying BYOL/MoCo.
2. **First to introduce 3D scene flow to equivariant self-supervised learning**: Utilizing real motion between sequential frames as a self-supervised signal, which is more natural and information-rich than artificial augmentations.
3. **Experimental finding on transformation-loss matching**: Rotation is suitable for classification loss, while translation/scaling is suitable for contrastive loss. This experimental conclusion provides practical reference value for subsequent work.
4. **Excellent data efficiency**: 20% of data is sufficient to reach full-data training performance, which is of great significance for autonomous driving scenarios with high annotation costs.

## Limitations & Future Work

1. **Limited improvement for Car class**: Car has sufficient samples in KITTI and is relatively easy to detect, resulting in unremarkable pre-training gains.
2. **Not consistently surpassing ALSO**: It trades wins back and forth with ALSO (a generative occupancy-prediction-based method) without showing an overwhelming advantage.
3. **Scene flow estimation relies on external models**: Using a frozen PV-RAFT to estimate flow, where the accuracy ceiling of the flow model limits the performance of temporal equivariance.
4. **Diminishing returns under full data**: The improvement of pre-training under 100% data is limited; the main value lies in low-data scenarios.
5. **Limited to sparse convolutional backbones**: Not yet extended to Transformer-based detectors (such as SST), which limits its generalizability.

## Related Work & Insights

| Method | Type | Core Strategy | Pros & Cons |
|:---|:---|:---|:---|
| PointContrast | Discriminative | Point-level contrastive (equivariant rigid transformation) | Concise but lacks temporal information |
| STRL | Discriminative | Temporal invariance (BYOL-style) | Encourages invariance, which contradicts the detection task |
| ALSO | Generative | Occupancy prediction | Strong performance but trains more layers and converges slowly |
| **E-SSL3D** | **Discriminative** | **Spatial equivariance + Temporal flow equivariance** | **Theory-driven, converges fast, only trains 3D backbone** |

E-SSL3D only pre-trains the 3D backbone, whereas ALSO also trains 2D convolutional layers. Thus, E-SSL3D is slightly disadvantaged in fairness but remains highly competitive.

## Related Work & Insights

- **Generalization of Equivariance Ideas**: This approach can be extended to other equivariant tasks (semantic segmentation, instance segmentation) and other modalities (radar, multi-modal fusion).
- **Complementarity with MAE-like Methods**: E-SSL3D is a discriminative method that can be combined with generative methods like GD-MAE to construct stronger pre-training strategies.
- **Potential of Scene Flow**: Flow equivariance essentially leverages temporal consistency, which can be further extended to multi-frame aggregation or flow-based data augmentation.

## Rating
- Novelty: ⭐⭐⭐⭐ — 3D scene flow equivariance is a novel contribution, and the transformation-loss matching analysis is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across multiple detectors and data ratios with thorough ablations, but only evaluated on KITTI.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous mathematical definitions, and well-motivated arguments.
- Value: ⭐⭐⭐⭐ — Practically significant for low-data autonomous driving scenarios, and the equivariance idea is insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)
- [\[ECCV 2024\] CSOT: Cross-Scan Object Transfer for Semi-Supervised LiDAR Object Detection](csot_cross-scan_object_transfer_for_semi-supervised_lidar_object_detection.md)
- [\[ECCV 2024\] FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection](fsd-bev_foreground_self-distillation_for_multi-view_3d_object_detection.md)
- [\[ECCV 2024\] Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)
- [\[CVPR 2026\] STUR3D: Spatio-Temporal Unified Representation Learning for 3D Object Detection](../../CVPR2026/autonomous_driving/stur3d_spatio-temporal_unified_representation_learning_for_3d_object_detection.md)

</div>

<!-- RELATED:END -->
