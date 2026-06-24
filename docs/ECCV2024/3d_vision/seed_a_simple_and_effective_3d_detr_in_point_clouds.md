---
title: >-
  [Paper Note] SEED: A Simple and Effective 3D DETR in Point Clouds
description: >-
  [ECCV 2024][3D Vision][3D Object Detection] SEED proposes a simple and effective 3D DETR detector. It obtains high-quality queries in a coarse-to-fine manner through a Dual Query Selection (DQS) module, and achieves flexible query interaction by utilizing geometric structural information of 3D objects with a Deformable Grid Attention (DGA) module, reaching new SOTA on Waymo and nuScenes.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Object Detection"
  - "DETR"
  - "Point Clouds"
  - "Query Selection"
  - "Deformable Attention"
date: 2026-05-08
content_hash: e1fa17f0602709e0
---

# SEED: A Simple and Effective 3D DETR in Point Clouds

**Conference**: ECCV 2024  
**arXiv**: [2407.10749](https://arxiv.org/abs/2407.10749)  
**Code**: [https://github.com/happinesslz/SEED](https://github.com/happinesslz/SEED)  
**Area**: 3D Vision  
**Keywords**: 3D Object Detection, DETR, Point Clouds, Query Selection, Deformable Attention

## TL;DR
SEED proposes a simple and effective 3D DETR detector. It obtains high-quality queries in a coarse-to-fine manner through a Dual Query Selection (DQS) module, and achieves flexible query interaction by utilizing geometric structural information of 3D objects with a Deformable Grid Attention (DGA) module, reaching new SOTA on Waymo and nuScenes.

## Background & Motivation
The DETR paradigm has become a mainstream method in the 2D detection field. By modeling object detection as a set prediction problem, it elegantly eliminates hand-crafted anchors and Non-Maximum Suppression (NMS) post-processing. However, in 3D point cloud object detection, DETR-based methods have not yet demonstrated the same outstanding performance as in the 2D domain, still lagging behind state-of-the-art non-DETR 3D detectors.

**Two Core Challenges**:

**Difficulty in Query Selection** — Point clouds are highly sparse and unevenly distributed, making it challenging to obtain appropriate object queries. Existing methods (e.g., TransFusion, ConQueR) only perform single-step Top-N selection, without considering the quality of selected queries for bounding box localization.

**Insufficient Query Interaction** — How to utilize the rich geometric structural information of point clouds for effective query interaction has not been fully explored. While objects in 2D images may occupy the entire image (requiring a global receptive field), 3D objects typically occupy only small local regions, where local attention is sufficient.

**Core Idea**: Design a coarse-to-fine dual query selection mechanism to ensure high-quality queries, and use gridded deformable attention to fully exploit the geometric information of 3D objects for query interaction.

## Method

### Overall Architecture
1. The point cloud is input into a classic voxel-based 3D backbone to extract voxel features, which are then converted into BEV (Bird's Eye View) features.
2. BEV features are flattened after adding positional encodings and sent to the DQS module to select high-quality queries.
3. The selected queries are fed into a 6-layer SEED Decoder Layer for self-attention (inter-query interaction) + DGA (query-BEV interaction) to output the final detection results.

### Key Designs
1. **Dual Query Selection (DQS)**:

    - **Function**: Selects high-quality queries from BEV features in a coarse-to-fine manner.
    - **Foreground Query Selection (Coarse)**: Uses a binary classifier to distinguish foreground/background, and retains $N_c = H \times W \times r$ coarse queries from the BEV features with the highest confidence based on a ratio $r = 0.3$. The goal is to ensure high recall.
    - **Quality Query Selection (Fine)**: After the coarse queries are enhanced by a SEED Decoder Layer, three FFN branches predict classification scores $S_c$, localization scores $S_l$ (predicting 3D IoU), and regression boxes $B_c$. The quality score is computed as:

    $$S_q^i = \begin{cases} (S_c^i)^{1-\beta} \cdot (S_l^i)^{\beta}, & \text{if } S_c^i > \tau \\ S_c^i, & \text{otherwise} \end{cases}$$

      From these, $N_f = 1000$ queries with the highest quality scores are selected. Their bounding box information is concatenated and mapped through an MLP to generate geometry-aware high-quality queries.
    - **Design Motivation**: A single-step selection cannot guarantee both recall and quality; the two-step strategy first ensures coverage and then precision. Furthermore, introducing localization scores helps filter out queries with high confidence but poor localization.

2. **Deformable Grid Attention (DGA)**:

    - **Function**: Replaces standard cross-attention in the SEED Decoder Layer, utilizing geometric information of 3D objects for effective query-BEV interaction.
    - **Mechanism**:
        - The estimated proposal box is uniformly divided into $k \times k$ (default 5×5) grid points as reference points.
        - Offset values $\Delta g$ are predicted via queries and added to the grid points to obtain the final sampling positions.
        - BEV features are sampled (using bilinear interpolation) and multiplied by the predicted attention weights.
    - **DGA Formula**: $$\text{DGA}(g, F_{bev}) = \sum_{j=1}^{K} A_j \cdot \phi(F_{bev}(g_j + \Delta g_j))$$
    - **Design Motivation**: Combines the advantages of both box attention (utilizing geometric information) and deformable attention (flexible receptive fields). Pure box attention relies heavily on box accuracy, while pure deformable attention does not exploit geometric structures. DGA uses the grid as a baseline and adds offsets, achieving the benefits of both.

3. **Quality-aware Hungarian Matching (QHM)**:

    - **Function**: Modifies the Hungarian matching strategy of DETR.
    - **Mechanism**: When calculating the classification cost, the quality score $S_f$ (fusing classification and localization scores) replaces the conventional classification score, biasing the matching towards proposals with high localization quality.
    - **Matching Cost**: $$\mathcal{C}_{match} = \lambda_{cls} \mathcal{C}_{cls} + \lambda_{reg} \mathcal{C}_{reg} + \lambda_{giou} \mathcal{C}_{giou}$$

### Loss & Training
- Final loss = DETR head loss + DQS loss
- DQS loss: BCE loss for classification scores, IoU loss for localization scores, and Smooth-L1 loss for regression.
- Uses AdamW optimizer with an initial learning rate of 0.001.
- Trained on WOD for 24 epochs with 20% data, and 12 epochs with 100% data.
- Evaluated/trained with 8 V100 GPUs, with a batch size of 24.
- Uses a fade strategy (disabling data augmentation in the last epoch) and query contrast strategy.

## Key Experimental Results

### Main Results

**Waymo Open Dataset (val, 100% data, single-frame)**:

| Method | Type | Vehicle APH(L2) | Ped APH(L2) | Cyclist APH(L2) | mAPH(L2) |
|------|------|-----------------|-------------|-----------------|----------|
| ConQueR | DETR | 68.2 | 64.7 | 70.1 | 67.7 |
| FocalFormer3D | DETR | 67.6 | 66.8 | 72.6 | 69.0 |
| DSVT-Voxel | Non-DETR | 71.0 | 71.5 | 73.7 | 72.1 |
| **SEED-S** | DETR | 69.7 | 68.1 | 74.5 | **70.8** |
| **SEED-B** | DETR | 71.4 | 70.8 | 76.1 | **72.8** |
| **SEED-L** | DETR | **71.5** | **71.8** | **77.3** | **73.5** |

**Waymo (val, multi-frame)**:

| Method | Frames | mAPH(L2) |
|------|------|----------|
| DSVT-Voxel | 3 | 75.0 |
| SEED-B | 3 | 75.8 |
| SEED-L | 3 | **76.1** |

**nuScenes (val)**:

| Method | NDS | mAP |
|------|-----|-----|
| TransFusion-L | 70.1 | 65.1 |
| Uni3DETR | 68.5 | 61.7 |
| **SEED** | **71.2** | **66.6** |

### Ablation Study

**Contributions of Individual Components (Waymo val, 20% data)**:

| Config | DQS | DGA | mAPH(L2) | Gain |
|------|-----|-----|----------|------|
| Baseline | ✗ | ✗ | 64.6 | - |
| +DQS | ✓ | ✗ | 67.4 | +2.8 |
| +DGA | ✗ | ✓ | 66.4 | +1.8 |
| SEED | ✓ | ✓ | **68.2** | **+3.6** |

**Comparison of Query Selection Strategies**:

| Strategy | mAPH(L2) | Description |
|------|----------|------|
| Learnable (CMT) | 66.6 | Learnable queries |
| Heatmap (TransFusion) | 65.0 | Worst; queries originated directly from BEV itself |
| Top-N (ConQueR) | 66.8 | Single-step selection |
| **DQS (Ours)** | **68.2** | Coarse-to-fine two-step selection |

**Comparison of Attention Mechanisms**:

| Attention Type | mAPH(L2) | Description |
|-----------|----------|------|
| Global Attention | OOM | Out of Memory (OOM) |
| Deformable Attn | 67.5 | Flexible but does not utilize geometric information |
| Box Attention | 67.5 | Utilizes geometry but is not flexible enough |
| **DGA (Ours)** | **68.2** | Achieves both flexibility and geometric information |

### Key Findings
- DQS contributes the most (+2.8 mAPH), validating the effectiveness of the coarse-to-fine query selection strategy.
- DGA outperforms deformable attention and box attention by 0.7 and 0.7 mAPH/L2, respectively.
- Heatmap-based selection performs the worst, as directly retrieving queries from BEV features is not conducive to decoder stacking.
- QHM brings a larger improvement to Vehicles compared to Pedestrians and Cyclists, as the localization scores of large rigid objects are easier to estimate.
- SEED-S also outperforms existing DETR methods in terms of speed (13.5 FPS on RTX 3090).

## Highlights & Insights
1. **Coarse-to-Fine Dual Query Selection** — The strategy design of first ensuring recall and then accuracy is highly reasonable. Moreover, introducing localization scores to evaluate query quality is a novel and effective practice.
2. **Deformable Grid Attention** — Uniformly partitioning grids inside the proposal box and predicting offsets elegantly integrates geometric priors and receptive field flexibility.
3. **First 3D DETR Method to Outperform Non-DETR SOTA** — SEED-L outperforms DSVT-Voxel (L2 mAPH 73.5 vs 72.1), demonstrating the potential of the DETR paradigm in 3D detection.
4. **Three Model Scales (S/B/L) Offered** — Convenient for balancing speed and accuracy.

## Limitations & Future Work
- Insufficient performance in detecting far-away small objects, which are easier to recognize in 2D camera images — future work could incorporate multi-modal fusion.
- 3D backbone enhancements (e.g., DSVT) are orthogonal to SEED; their combination may yield further improvements.
- The prediction accuracy of localization scores is inherently limited by the quality of the initial proposals.
- The two-step DQS increases inference latency, though the overall speed remains acceptable.

## Related Work & Insights
- The analysis of query selection strategies in 2D DETR methods (such as DINO, DN-DETR, etc.) is highly valuable.
- The design philosophy of DGA can be generalized to other attention scenarios that require geometric priors.
- The concept of quality-aware Hungarian matching is inspired by IoU rectification techniques in non-DETR methods such as AFDetv2 and PillarNet.
- Provides a strong baseline for the development of the DETR paradigm in 3D detection.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3D Single-Object Tracking in Point Clouds with High Temporal Variation](3d_single-object_tracking_in_point_clouds_with_high_temporal_variation.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](../../ICCV2025/3d_vision/easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[ECCV 2024\] FLAT: Flux-Aware Imperceptible Adversarial Attacks on 3D Point Clouds](flat_flux-aware_imperceptible_adversarial_attacks_on_3d_point_clouds.md)
- [\[ECCV 2024\] Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds](heterogeneous_graph_learning_for_scene_graph_prediction_in_3d_point_clouds.md)
- [\[ECCV 2024\] Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)

</div>

<!-- RELATED:END -->
