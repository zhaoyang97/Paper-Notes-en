---
title: >-
  [Paper Note] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving
description: >-
  [NeurIPS 2025 Spotlight][Autonomous Driving][sparse perception model] SQS presents the first query-based 3D Gaussian splatting pre-training framework for sparse perception models (SPMs). By self-supervisedly reconstructing RGB images and depth maps, the method learns fine-grained 3D representations, and introduces a query interaction module to fuse pre-trained Gaussian queries with task-specific queries. SQS achieves significant improvements over existing pre-training methods…
tags:
  - "NeurIPS 2025 Spotlight"
  - "Autonomous Driving"
  - "sparse perception model"
  - "3D Gaussian splatting"
  - "pre-training"
  - "query interaction"
date: 2026-05-08
content_hash: 6164c420e553b0df
---

# SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2509.16588](https://arxiv.org/abs/2509.16588)  
**Code**: N/A  
**Area**: Autonomous Driving
**Keywords**: sparse perception model, 3D Gaussian splatting, pre-training, query interaction, autonomous driving

## TL;DR

SQS presents the first query-based 3D Gaussian splatting pre-training framework for sparse perception models (SPMs). By self-supervisedly reconstructing RGB images and depth maps, the method learns fine-grained 3D representations, and introduces a query interaction module to fuse pre-trained Gaussian queries with task-specific queries. SQS achieves significant improvements over existing pre-training methods on occupancy prediction (+1.3 mIoU) and 3D object detection (+1.0 NDS).

## Background & Motivation

Visual perception models for autonomous driving fall into two major paradigms: dense BEV-centric methods (e.g., BEVFormer) and sparse query-centric methods (e.g., DETR3D, SparseBEV). Sparse methods are increasingly favored in industrial deployment due to their faster inference speed, as they bypass explicit dense representation construction.

However, supervised approaches rely heavily on costly and time-consuming precise annotations, leaving large amounts of unlabeled data underutilized. Existing pre-training methods (e.g., UniPAD, GaussianPretrain, VisionPAD) all depend on dense BEV or voxel representations and cannot be directly applied to SPMs. The root cause lies in the fact that implicit queries in sparse query models lack explicit spatial positions and semantic meanings, making rendering-based pre-training methods inapplicable without modification.

**Key Insight**: SQS introduces a set of learnable Gaussian queries. During pre-training, a 3D Gaussian splatting mechanism dynamically predicts Gaussian attributes and reconstructs multi-view images and depth maps, enabling sparse queries to learn fine-grained 3D geometric representations. After pre-training, a query interaction module fuses the learned Gaussian queries with downstream task queries.

## Method

### Overall Architecture

SQS adopts a two-stage design:
- **Pre-training stage**: Image encoder + Gaussian Transformer decoder → predict 3D Gaussian attributes → render RGB and depth maps for self-supervised training.
- **Fine-tuning stage**: Load the pre-trained image backbone; fuse pre-trained Gaussian queries with task-specific queries via the query interaction module.

### Key Designs

1. **Gaussian Transformer Decoder and Gaussian Queries**: Each Gaussian query is initialized as a learnable anchor $g_k \in \mathbb{R}^{K \times C}$, paired with a zero-initialized high-dimensional query vector $q_k \in \mathbb{R}^{K \times D}$, where $K$ is set to 25,600. Queries interact with multi-scale image features through self-attention and deformable cross-attention, iteratively refining Gaussian attributes (position, covariance, opacity, color). 3D sparse convolution is employed to model spatial relationships among Gaussian queries to reduce memory cost. Positions $\mu$ are predicted as incremental offsets, while other attributes are directly replaced at each layer.

2. **Query Interaction Module (for fine-tuning)**: Addresses the challenge that different tasks in sparse methods employ different queries and decoders. The pre-trained model parameters are frozen, and Gaussian anchors and query features are obtained by inference on each test sample. Low-quality anchors are filtered via an opacity threshold $\alpha_\text{thresh}$; then, for each task query, its $k$ nearest Gaussian queries are found via a $k$-nearest-neighbor algorithm, and local attention fusion is performed:
$$q_t = \text{LocalAttn}(q_t + \text{MLP}(\mu_t),\ q_k + \text{MLP}(g_k))$$
This spatially-aware local attention mechanism is both efficient and fully exploits pre-trained queries.

3. **Reconstruction Loss Design**: L1 loss is applied to supervise both RGB and depth reconstruction. LiDAR points serve as depth ground truth; the depth loss is computed only at valid LiDAR pixels. The total loss is:
$$\mathcal{L} = \omega_1 \mathcal{L}_{rgb} + \omega_2 \mathcal{L}_{depth}$$
where $\omega_1 = 1.0$ and $\omega_2 = 0.05$.

### Loss & Training

Pre-training uses the AdamW optimizer with weight decay 0.01, a linear warm-up of 500 steps to $2\times10^{-4}$ followed by cosine decay, batch size 8, and 20 training epochs. Only random horizontal flipping is used as data augmentation. The fine-tuning stage directly adopts the official configurations of downstream models without modification. The image backbone is ResNet101-DCN (for occupancy prediction) or ResNet50/101 (for detection), combined with FPN to generate 4-scale feature maps.

## Key Experimental Results

### Main Results — 3D Semantic Occupancy Prediction (SurroundOcc val)

| Method | SC IoU | SSC mIoU | Note |
|---|---|---|---|
| MonoScene | 23.96 | 7.31 | Monocular baseline |
| BEVFormer | 30.50 | 16.75 | Dense BEV method |
| SurroundOcc | 31.49 | 20.30 | Dense method SOTA |
| GaussianFormer | 29.83 | 19.10 | Sparse query baseline |
| **GaussianFormer + SQS** | **31.52** | **20.40** | **+1.69 IoU, +1.30 mIoU** |

### Main Results — 3D Object Detection (nuScenes val)

| Method | Backbone | Input Size | NDS | mAP |
|---|---|---|---|---|
| SparseBEV (R50) | ResNet50 | 704×256 | 55.8 | 44.8 |
| **SparseBEV + SQS (R50)** | ResNet50 | 704×256 | **56.6** | **45.2** |
| SparseBEV (R101) | ResNet101 | 1408×512 | 59.2 | 50.1 |
| **SparseBEV + SQS (R101)** | ResNet101 | 1408×512 | **60.2** | **50.9** |

### Ablation Study (SurroundOcc val, 1/4 training data)

| Config | Render RGB | Render Depth | Load Backbone | Query Interaction | IoU | mIoU |
|---|---|---|---|---|---|---|
| Baseline | - | - | - | - | 25.8 | 15.2 |
| Model A | ✓ | - | ✓ | - | 23.8 | 12.2 |
| Model B | - | ✓ | ✓ | - | 27.9 | 17.3 |
| Model C | ✓ | ✓ | ✓ | - | 28.2 | 17.5 |
| Model D | ✓ | ✓ | - | ✓ | 26.3 | 15.9 |
| Model E | - | - | - | ✓ | 25.7 | 15.3 |
| **SQS** | ✓ | ✓ | ✓ | ✓ | **28.5** | **18.0** |

### Key Findings

- Depth rendering contributes substantially (+2.1 IoU/mIoU), while RGB-only rendering actually degrades performance (−2.0 IoU, −3.0 mIoU), underscoring the critical role of depth supervision in learning geometric representations.
- The query interaction module alone, without pre-training, provides negligible benefit (Model E vs. Baseline: Δ = 0.1), confirming that the quality of pre-trained queries is the key factor.
- Data efficiency analysis shows that with only 10% of labeled data, SQS yields +3.7 mIoU improvement — substantially larger than the +1.3 gain under full data.
- SQS is a plug-and-play design compatible with arbitrary sparse query perception models.

## Highlights & Insights

- SQS is the first pre-training framework specifically designed for SPMs, filling a gap in the SPM pre-training landscape.
- The Gaussian query concept is elegant — it introduces the geometric representational power of 3DGS into sparse query learning, driving queries to acquire rich 3D spatial knowledge via rendering reconstruction.
- The query interaction module is architecturally clean — spatially-aware local attention bridges task queries across different architectures, enabling truly plug-and-play integration.
- The advantage is more pronounced in data-scarce settings (10% annotations), demonstrating strong practical value.
- The finding that depth rendering substantially outweighs RGB rendering provides clear design guidance for future pre-training work.

## Limitations & Future Work

- The plug-in pre-trained model introduces additional computational and memory overhead.
- Utilization of pre-trained queries across different downstream tasks is insufficiently differentiated at the semantic level.
- Application to end-to-end autonomous driving frameworks (e.g., SparseAD, GaussianAD) remains unexplored.
- Pre-training relies solely on LiDAR as depth ground truth, creating a dependency on sensor configuration.
- Sensitivity analysis of the $k$-nearest-neighbor parameter $k$ and the opacity threshold in query interaction is insufficient.

## Related Work & Insights

- Compared to dense pre-training methods such as GaussianPretrain and VisionPAD, SQS is the first to extend 3DGS-based pre-training to the sparse query paradigm.
- The query interaction idea is transferable to other scenarios requiring cross-architecture knowledge transfer.
- The effectiveness of 3DGS as a self-supervised pre-training objective is further validated.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](../../ICCV2025/autonomous_driving/gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[ICLR 2026\] Beyond Visual Reconstruction Quality: Object Perception-aware 3D Gaussian Splatting for Autonomous Driving](../../ICLR2026/autonomous_driving/beyond_visual_reconstruction_quality_object_perception-aware_3d_gaussian_splatti.md)
- [\[ICCV 2025\] INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](../../ICCV2025/autonomous_driving/instinct_instance-level_interaction_architecture_for_query-based_collaborative_p.md)
- [\[AAAI 2026\] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](../../AAAI2026/autonomous_driving/expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)

</div>

<!-- RELATED:END -->
