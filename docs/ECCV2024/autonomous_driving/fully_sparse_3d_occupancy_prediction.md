---
title: >-
  [Paper Note] Fully Sparse 3D Occupancy Prediction
description: >-
  [ECCV 2024][Autonomous Driving][3D Occupancy] This work proposes SparseOcc, the first fully sparse 3D occupancy prediction network, which achieves efficient occupancy prediction via a sparse voxel decoder and a mask-guided Mask Transformer, and designs a RayIoU evaluation metric to address the depth-direction inconsistent penalty of traditional mIoU.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "3D Occupancy"
  - "Sparse Network"
  - "Mask Transformer"
  - "Evaluation Metric"
  - "BEV Perception"
date: 2026-05-08
content_hash: 05f60689c74b187a
---

# Fully Sparse 3D Occupancy Prediction

**Conference**: ECCV 2024  
**arXiv**: [2312.17118](https://arxiv.org/abs/2312.17118)  
**Code**: [GitHub](https://github.com/MCG-NJU/SparseOcc)  
**Area**: Autonomous Driving  
**Keywords**: 3D Occupancy, Sparse Network, Mask Transformer, Evaluation Metric, BEV Perception

## TL;DR

This work proposes SparseOcc, the first fully sparse 3D occupancy prediction network, which achieves efficient occupancy prediction via a sparse voxel decoder and a mask-guided Mask Transformer, and designs a RayIoU evaluation metric to address the depth-direction inconsistent penalty of traditional mIoU.

## Background & Motivation

3D occupancy prediction is a critical task in autonomous driving, providing finer geometric descriptions of scenarios than 3D object detection. However, existing methods suffer from severe efficiency bottlenecks:

**Redundancy in Dense Representations**: Existing methods (e.g., BEVFormer-Occ, SurroundOcc) construct dense 3D volumes (such as 200x200x16), but statistics show that over 90% of the voxels are empty.

**Huge Computational Overhead**: Dense methods only achieve 2-3 FPS (on an A100), which falls far short of real-time requirements.

**Pseudo-Sparse Methods**: Although methods like VoxFormer introduce sparse queries, they still contain sparse-to-dense modules (such as MAE) and are not fully sparse in essence.

**Flawed Evaluation Metrics**: Traditional voxel-level mIoU suffers from depth-direction inconsistent penalties, allowing models to cheat by predicting thicker surfaces.

This paper simultaneously advances the field from two perspectives: (1) proposing a fully sparse architecture design; (2) designing a more reasonable RayIoU metric at the evaluation level.

## Method

### Overall Architecture

SparseOcc consists of three modules:
1. **Image Encoder**: Backbone + FPN to extract multi-view 2D features.
2. **Sparse Voxel Decoder**: Reconstructs the sparse 3D geometric structure of the scene from coarse to fine.
3. **Mask Transformer**: Predicts the mask and label for each semantic/instance segment utilizing sparse queries.

The entire pipeline does not incorporate any dense designs—it does not rely on dense 3D features, lacks sparse-to-dense modules, and avoids global attention.

### Key Designs

#### Sparse Voxel Decoder

This module adopts a coarse-to-fine pipeline structure consisting of 3 levels:

- **Initialization**: Uniformly distributes a set of coarse voxel queries (e.g., 25x25) in 3D space.
- **Operations per Level**: (1) $2 \times$ upsampling (each voxel splits into 8) (2) estimating the occupancy score of each voxel (3) removing empty voxels via top-k pruning.
- **Feature Acquisition**: For each voxel query, self-attention aggregates local/global features. A linear layer then generates 3D sampling offsets, which project into multi-view image space to fetch features.

Key design parameter: $k=32000$ (which is only 5% of the total 640,000 voxels), demonstrating that 5% sparsity is sufficient to cover the scene.

**Temporal Modeling**: Capitalizing on the flexibility of global sampling reference points, it transforms them to historical timestamps to sample historical multi-view features, avoiding warping operations of dense features.

**Supervision**: A weighted BCE Loss is applied at each level, where class weights are assigned inversely proportional to their ratios: $w_c = \sum(M_i) / M_c$.

#### Mask Transformer

Inspired by Mask2Former, this module uses $N$ sparse semantic/instance queries:

- **Mask-guided Sparse Sampling**: Randomly selects 3D points within the mask predicted from the previous level and projects them to multi-view images to extract features. This improves inference speed by 50% with superior performance compared to dense cross-attention.
- **Prediction**: Query embeddings pass through a linear classifier to predict class labels, and through an MLP to generate mask embeddings, which perform a dot product with sparse voxel embeddings for mask prediction.
- **Prediction Space Constraints**: Mask prediction is restricted to the sparse space output by the sparse voxel decoder, rather than the entire scene.

**Loss Function**: Following Hungarian matching, the network uses Focal Loss (classification) + DICE Loss + BCE Mask Loss + Occ Loss (voxel decoder).

#### RayIoU Evaluation Metric

Designed to address three major problems of traditional voxel-level mIoU:

**Problem Analysis**:
1. Dense methods can cheat by predicting thicker surfaces—training BEVFormer with a visible mask artificially inflates mIoU by 5–15.
2. Thin surface prediction is overly penalized—a deviation of just one voxel results in zero IoU.
3. The visible mask only considers visible regions at the current frame, neglecting the scene completion capability.

**RayIoU Solution**: Simulates LiDAR behavior by casting query rays into the predicted 3D occupancy volume:
- Computes the distance and category of the ray to the first occupied voxel.
- True positive criteria: identical class + depth $L_1$ error $<$ threshold (1m/2m/4m).
- The final metric is the average of the three thresholds.
- Rays originate from 8 LiDAR positions (supporting temporal projection to evaluate scene completion capability).
- Rays are resampled to balance the weights at different distances.

### Loss & Training

- **Optimizer**: AdamW, global batch size 8, initial learning rate 2e-4, cosine annealing.
- **Training**: 24 epochs, ResNet-50 backbone.
- **Mask Transformer**: 3-level weight sharing.
- **Loss**: $L = L_{\text{focal}} + L_{\text{mask}} + L_{\text{dice}} + L_{\text{occ}}$

## Key Experimental Results

### Main Results (Occ3D-nuScenes val)

| Method | Backbone | Frames | RayIoU | FPS |
|---|---|---|---|---|
| BEVFormer (4f) | R101 | 4 | 32.4 | 3.0 |
| BEVDet-Occ (2f) | R50 | 2 | 29.6 | 2.6 |
| BEVDet-Occ-Long (8f) | R50 | 8 | 32.6 | 0.8 |
| FB-Occ (16f) | R50 | 16 | 33.5 | 10.3 |
| **SparseOcc (8f)** | R50 | 8 | **34.0** | **17.3** |
| SparseOcc (16f) | R50 | 16 | 35.1 | 12.5 |
| SparseOcc (16f, 48ep) | R50 | 48ep | 36.1 | 12.5 |

Under a weak setting (R50, 8 frames, 704x256), SparseOcc outperforms FB-Occ (the challenge winner) by +1.6 RayIoU while operating 1.7x faster.

### Ablation Study

**Sparse vs. Dense Voxel Decoder**:

| Decoder Type | RayIoU | FPS |
|---|---|---|
| Dense coarse-to-fine | 29.9 | 6.3 |
| Dense patch-based | 25.8 | 7.8 |
| **Sparse coarse-to-fine** | **29.9** | **24.0**|

The sparse decoder runs 4x faster with no performance degradation.

**Mask Transformer + Sparse Sampling**:

| Mask Transformer | Cross-Attention | RayIoU | FPS |
|---|---|---|---|
| None | - | 27.0 | 29.0 |
| Yes | Dense | 28.7 | 16.2 |
| Yes | Sparse + Mask-guided | **29.2** | **24.0** |

**Voxel Sparsity**: $k=32000$ (5%) is optimal; continuing to increase density introduces noise and reduces precision.

**Temporal Frames**: Performance consistently improves as the number of frames increases up to 12 frames where it saturates, but inference speed decreases linearly.

### Key Findings

1. Only 5% of the voxels are sufficient to cover scene geometry—dense representations show significant redundancy.
2. Mask-guided sparse sampling is stronger and faster than dense cross-attention.
3. RayIoU successfully avoids the thick-surface cheating problem, offering a more equitable evaluation.
4. Threshold pruning is comparable to top-k pruning in performance and offers better scene generalization.

## Highlights & Insights

- **First Fully Sparse Occupancy Network**: Thoroughly eliminates dense designs from the architecture to realize real-time inference at 17.3 FPS.
- **Model-Independent RayIoU Metric**: Resolves evaluation issues shared by the community and has been widely adopted by subsequent works.
- **Elegant Integration with Mask2Former**: Successfully transfers the mask transformer paradigm in 2D segmentation to 3D sparse occupancy.
- **Empirical Finding of 5% Sparsity**: Provides crucial design guidance for future sparse methods.

## Limitations & Future Work

1. top-k pruning is a dataset-dependent hyperparameter; though threshold-based pruning generalizes better, it requires tuning.
2. Early pruning in the sparse voxel decoder might lose tiny objects (e.g., pedestrians, traffic cones).
3. The efficiency of temporal modeling degrades linearly with the number of frames, necessitating more efficient temporal fusion schemes.
4. Currently, only semantic queries are employed, and validation on instance-level occupancy (panoptic occupancy) remains insufficient.

## Related Work & Insights

- **SparseBEV**: A fully sparse 3D detection method; SparseOcc extends it from pillar queries to 3D voxel queries.
- **Mask2Former / Mask3D**: Mask transformer approaches for 2D/3D segmentation that provide the mask prediction paradigm.
- **FB-Occ**: Winner of the CVPR 2023 Occupancy Challenge; its complex and dense design is surpassed by the clean architecture of SparseOcc.
- The sparse philosophy can be further extended to 4D occupancy prediction and end-to-end driving.

## Rating

- **Novelty**: 5/5 - Two major contributions: the first fully sparse occupancy network + RayIoU metric.
- **Experimental Thoroughness**: 4/5 - Sufficient ablation studies, though validation on more backbones and datasets is needed.
- **Writing Quality**: 5/5 - Thorough problem analysis and compelling motivation for RayIoU.
- **Value**: 5/5 - Highly practical with real-time inference and open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [\[ECCV 2024\] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multi-modal_3d_occupancy_prediction_for_autonomous_driving.md)
- [\[CVPR 2025\] SparseAlign: A Fully Sparse Framework for Cooperative Object Detection](../../CVPR2025/autonomous_driving/sparsealign_a_fully_sparse_framework_for_cooperative_object_detection.md)
- [\[ECCV 2024\] Monocular Occupancy Prediction for Scalable Indoor Scenes](monocular_occupancy_prediction_for_scalable_indoor_scenes.md)

</div>

<!-- RELATED:END -->
