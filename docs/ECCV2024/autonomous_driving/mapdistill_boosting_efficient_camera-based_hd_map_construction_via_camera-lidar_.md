---
title: >-
  [Paper Note] MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation
description: >-
  [ECCV 2024][Autonomous Driving][HD Map] This paper introduces knowledge distillation into the HD map construction task for the first time, proposing the MapDistill framework. By leveraging a dual BEV transform module, cross-modal relation distillation, dual-level feature distillation, and Map Head distillation, it transfers knowledge from a camera-LiDAR fusion teacher model to a lightweight, camera-only student model. This achieves a **+7.7 mAP** improvement or a **4.5x speed…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "HD Map"
  - "Knowledge Distillation"
  - "BEV Perception"
  - "Cross-modal Distillation"
  - "Lightweight Deployment"
date: 2026-05-08
content_hash: 30979bdbd45e0265
---

# MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation

**Conference**: ECCV 2024  
**arXiv**: [2407.11682](https://arxiv.org/abs/2407.11682)  
**Code**: None (unreleased)  
**Area**: Autonomous Driving / HD Map Construction / Knowledge Distillation  
**Keywords**: HD Map, Knowledge Distillation, BEV Perception, Cross-modal Distillation, Lightweight Deployment

## TL;DR

This paper introduces knowledge distillation into the HD map construction task for the first time, proposing the MapDistill framework. By leveraging a dual BEV transform module, cross-modal relation distillation, dual-level feature distillation, and Map Head distillation, it transfers knowledge from a camera-LiDAR fusion teacher model to a lightweight, camera-only student model. This achieves a **+7.7 mAP** improvement or a **4.5x speedup** on nuScenes.

## Background & Motivation

Online High-Definition (HD) map construction is a crucial task in autonomous driving systems, providing precise static environmental information for planning and navigation. Recently, multi-view camera-based methods have gathered attention due to their low cost of deployment. However, they face a **Key Challenge**:

**Lack of depth info**: Camera images naturally lack explicit 3D geometric information, making current methods heavily reliant on large backbone networks (e.g., ResNet50 + Swin Transformer) to compensate for this deficiency.

**Performance-efficiency trade-off**: Experiments show that the representation capacity of the backbone is directly correlated with model performance—larger models yield better results but lead to slower inference, sacrificing the cost advantages of camera-only methods.

**Void in KD methods**: While knowledge distillation in BEV space has succeeded in 3D object detection (e.g., BEVDistill, UniDistill), **KD methods for HD map construction remain unexplored**.

**Why can't KD methods for 3D detection be directly applied?** Two key differences exist:
- The head of 3D detection outputs target classification + localization, whereas the Map head outputs map element classification + point regression, which are structurally different.
- 3D detection KD usually aligns foreground target features and suppresses background, which is clearly inapplicable to HD map construction (as map elements themselves are distributed across the entire BEV space).

Experimental verification: Direct application of BEVDistill to HD mapping yields only a +1.2 mAP improvement, and UniDistill yields +2.3 mAP, both significantly lower than the +7.7 mAP achieved by MapDistill.

## Method

### Overall Architecture

MapDistill adopts a teacher-student framework:
- **Teacher model**: A camera-LiDAR fusion model based on MapTR (ResNet50 + SECOND), frozen after training.
- **Student model**: A lightweight model based on the camera branch of MapTR (ResNet18), equipped with a **dual BEV transform module**.
- **Distillation objectives**: A three-way distillation loss (cross-modal relation + dual-level feature + Map Head), utilized only during the training phase.

During inference, only the student model (camera-only) is deployed, enjoying the advantages of lightweight and highly efficient deployment.

### Key Designs

#### 1. Dual BEV Transform Module (Dual BEV Transform)

**Function**: Transforms multi-view camera features into two different BEV subspaces, simulating the dual-modal (camera + LiDAR) BEV features of the teacher model.

**Mechanism**:
- **Subspace 1**: Uses GKT (attention-based 2D-to-BEV transform) to generate $\mathbf{F}_{C_{sub1}}^S \in \mathbb{R}^{H \times W \times C}$
- **Subspace 2**: Uses LSS (depth estimation-based projection transform) to generate $\mathbf{F}_{C_{sub2}}^S \in \mathbb{R}^{H \times W \times C}$
- The features of both subspaces are concatenated and fused into $\mathbf{F}_{fused}^S$ via a fully convolutional network.

**Design Motivation**: The teacher model has two independent branches (camera BEV and LiDAR BEV), so the student model requires a corresponding 'dual-path' structure to effectively mimic the teacher's cross-modal interactions. Employing two different 2D-to-BEV transformations (GKT focusing on semantics and LSS on geometry) allows the two subspaces to capture complementary info, acting as an analogy to the teacher's camera-LiDAR dual-stream design.

#### 2. Cross-modal Relation Distillation ($\mathcal{L}_{relation}$)

**Function**: Allows the student model to learn the cross-modal attention relationships between the camera and LiDAR from the teacher model.

**Mechanism**:
- Teacher side: Computes camera-to-LiDAR attention $A_{c2l}^T$ and LiDAR-to-camera attention $A_{l2c}^T$

$$A_{c2l}^T = \text{softmax}\left(\frac{\mathbf{Fp}_{C_{bev}}^T \cdot \text{Transpose}(\mathbf{Fp}_{L_{bev}}^T)}{\sqrt{D_k}}\right)$$

- Student side: Computes corresponding attentions $A_{c2l}^S$ and $A_{l2c}^S$ using the patch features of the dual BEV subspaces.
- Aligns the attention of both sides using KL divergence:

$$\mathcal{L}_{relation} = D_{KL}(A_{c2l}^T || A_{c2l}^S) + D_{KL}(A_{l2c}^T || A_{l2c}^S)$$

**Design Motivation**: Cross-modal interaction is the core source of the teacher model's advantage. By mimicking this interaction pattern, the student model can indirectly acquire LiDAR-like 3D perception capabilities. Ablation studies confirm that cross-modal relations are more effective than unimodal relations (mAP: 53.6 vs. 52.0).

#### 3. Dual-level Feature Distillation ($\mathcal{L}_{feature}$)

**Function**: Aligns both low-level and high-level feature representations simultaneously in a unified BEV space.

**Mechanism**:
- Low-level distillation: Aligns the fused BEV features $\mathcal{L}_{low} = \text{MSE}(\mathbf{F}_{fused}^T, \mathbf{F}_{fused}^S)$
- High-level distillation: Aligns the Map Encoder outputs $\mathcal{L}_{high} = \text{MSE}(\mathbf{F}_{high}^T, \mathbf{F}_{high}^S)$

$$\mathcal{L}_{feature} = \mathcal{L}_{low} + \mathcal{L}_{high}$$

**Design Motivation**: Low-level features contain rich spatial and geometric information, while high-level features contain semantic information. Dual-level alignment is more comprehensive than single-level alignment. Ablations show that dual-level (53.6 mAP) > high-level only (52.9) > low-level only (52.7).

#### 4. Map Head Distillation ($\mathcal{L}_{head}$)

**Function**: Constrains the student model's final predictions (map element classification + point locations) to approximate the outputs of the teacher model.

**Mechanism**: Uses teacher predictions as pseudo-labels to supervise the student:

$$\mathcal{L}_{head} = \mathcal{L}_{Focal}(\mathbf{F}_{cls}^T, \mathbf{F}_{cls}^S) + \mathcal{L}_{p2p}(\mathbf{F}_{point}^T, \mathbf{F}_{point}^S)$$

- Focal Loss for classification
- Manhattan Distance for point regression

**Design Motivation**: Feature-level distillation is implicit, whereas Head-level distillation provides a direct output alignment signal; the two are complementary. Ablations show that using both classification and point distillation simultaneously (53.6 mAP) outperforms using either alone (51.8/51.9).

### Loss & Training

Total training loss:

$$\mathcal{L} = \mathcal{L}_{map} + \lambda_1 \mathcal{L}_{relation} + \lambda_2 \mathcal{L}_{feature} + \lambda_3 \mathcal{L}_{head}$$

- $\mathcal{L}_{map}$: Original MapTR map loss (classification + point-to-point + edge direction)
- Training configuration: 8 $\times$ A6000 GPUs, batch size of 64, AdamW optimizer, initial learning rate of $4 \times 10^{-3}$
- Teacher model: ResNet50 + SECOND, pretrained for 24 epochs
- Student model: ResNet18, distilled for 110 epochs

## Key Experimental Results

### Main Results

**nuScenes val set (mAP)**:

| Method | Student Modality | Teacher Modality | Backbone | AP_ped | AP_div | AP_bou | mAP |
|------|---------|---------|----------|--------|--------|--------|-----|
| MapTR | C | — | R50 | 45.3 | 51.5 | 53.1 | 50.3 |
| MapTR | C&L | — | R50&Sec | 55.9 | 62.3 | 69.3 | 62.5 |
| MapTR (baseline) | C | — | R18 | 39.6 | 49.9 | 48.2 | **45.9** |
| BEVDistill | C | L | R18 | 42.4 | 48.5 | 50.2 | 47.1 (+1.2) |
| UniDistill | C | C&L | R18 | 43.9 | 48.6 | 52.1 | 48.2 (+2.3) |
| **MapDistill (C teacher)** | C | C | R18 | 43.3 | 48.8 | 51.9 | 48.0 (+2.1) |
| **MapDistill (L teacher)** | C | L | R18 | 45.9 | 50.7 | 53.6 | 50.1 (+4.2) |
| **MapDistill (C&L teacher)** | C | C&L | R18 | **49.2** | **54.5** | **57.1** | **53.6 (+7.7)** |

### Ablation Study

**Contribution of each distillation loss component**:

| Setting | $\mathcal{L}_{relation}$ | $\mathcal{L}_{feature}$ | $\mathcal{L}_{head}$ | mAP |
|------|:-:|:-:|:-:|------|
| Baseline (no KD) | ✗ | ✗ | ✗ | 45.9 |
| (a) Relation KD only | ✔ | ✗ | ✗ | 48.8 (+2.9) |
| (b) Feature KD only | ✗ | ✔ | ✗ | 48.4 (+2.5) |
| (c) Head KD only | ✗ | ✗ | ✔ | 49.0 (+3.1) |
| (d) Relation + Feature | ✔ | ✔ | ✗ | 50.3 |
| (e) Feature + Head | ✗ | ✔ | ✔ | 50.8 |
| (f) Relation + Head | ✔ | ✗ | ✔ | 51.1 |
| **(g) All** | **✔** | **✔** | **✔** | **53.6** |

**Ablation of cross-modal relation distillation**:

| Relation Type | mAP |
|---------|-----|
| No relation distillation | 50.8 |
| Unimodal relation | 52.0 |
| Hybrid relation | 52.4 |
| **Cross-modal relation (Ours)** | **53.6** |

### Key Findings

1. **Fusion Teacher >> Unimodal Teacher**: The Camera-LiDAR fusion teacher yields a **+7.7 mAP** gain, significantly outperforming the LiDAR-only teacher (+4.2) and camera-only teacher (+2.1).
2. **Complementarity of the three-way distillation losses**: Each loss alone contributes approximately +2.5 to 3.1 mAP; when combined, they reach +7.7, demonstrating a strong synergistic effect.
3. **Cross-modal relations are key**: Cross-modal attention distillation outperforms unimodal relation distillation (53.6 vs. 52.0) and even hybrid relations (52.4).
4. **Effectiveness of dual BEV transform**: The dual-path structure using different transformation mechanisms (GKT + LSS) achieves optimal results, whereas single transformations or using identical transformations perform worse.

## Highlights & Insights

- **Filling a literature gap**: Systematically applies knowledge distillation to HD map construction for the first time, noting that task differences prevent direct migration of 3D detection KD methods.
- **Ingenious design of dual BEV transform**: Employs two different 2D-to-BEV transformations (attention-based vs. depth-estimation-based) to simulate the teacher's camera-LiDAR dual-stream architecture, achieving "pseudo-multi-modality" with a camera-only setup.
- **Deployment-friendly**: All distillation losses exist only during the training phase, incurring absolutely no additional computational overhead during inference.
- **Comprehensive ablations**: Conducts thorough ablation studies on each loss component, relationship type, feature level, and BEV transform combination.

## Limitations & Future Work

- Training requires running both teacher and student models concurrently, resulting in **high training costs**.
- The student backbone is fixed to ResNet18; more lightweight architectures (e.g., MobileNet) or Neural Architecture Search (NAS) remain unexplored.
- Evaluation is limited to nuScenes; other data sets (e.g., Argoverse2) are not yet integrated.
- The quality upper bound of the teacher model dictates the student's performance ceiling; stronger fusion teachers can be explored.
- The dual BEV transform module introduces additional parameters; parameter sharing schemes could be investigated.

## Related Work & Insights

- **MapTR**: Provides the baseline architecture for both teacher and student models. Its unified map-element representation facilitates convenient distillation.
- **BEVDistill / UniDistill**: Pioneers of cross-modal distillation in BEV space, though not optimized specifically for HD map construction tasks.
- **GKT / LSS**: Two complementary 2D-to-BEV transformation methods, which conveniently fit the construction of the dual-path structure to simulate dual-modal features.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introduces KD into HD map construction for the first time; the dual BEV transform and cross-modal relation distillation are ingeniously designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely detailed ablations covering the three-way losses, relation types, feature levels, BEV transform combinations, and hyperparameter analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured; provides rigorous problem analysis and accurately articulates the differences from 3D detection KD.
- **Value**: ⭐⭐⭐⭐ — The +7.7 mAP gain or 4.5x speedup is highly practical; it serves as a strong baseline for KD-based HD map construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Stream Query Denoising for Vectorized HD-Map Construction](stream_query_denoising_for_vectorized_hd-map_construction.md)
- [\[ECCV 2024\] Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)
- [\[ICML 2025\] SafeMap: Robust HD Map Construction from Incomplete Observations](../../ICML2025/autonomous_driving/safemap_robust_hd_map_construction_from_incomplete_observations.md)
- [\[ECCV 2024\] MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping](maptracker_tracking_with_strided_memory_fusion_for_consistent_vector_hd_mapping.md)
- [\[CVPR 2025\] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](../../CVPR2025/autonomous_driving/tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)

</div>

<!-- RELATED:END -->
