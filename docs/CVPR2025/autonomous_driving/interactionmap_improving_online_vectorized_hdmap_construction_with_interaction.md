---
title: >-
  [Paper Note] InteractionMap: Improving Online Vectorized HDMap Construction with Interaction
description: >-
  [CVPR 2025][Autonomous Driving][HD Map Construction] This paper proposes InteractionMap, which comprehensively enhances information interaction in online vectorized HD map construction through three modules: point-level and instance-level relation embedding, keyframe-based hierarchical temporal fusion, and geometry-aware classification-localization alignment. It achieves state-of-the-art (SOTA) performance on both nuScenes (71.8 mAP) and Argoverse2 (74.7 mAP).
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "HD Map Construction"
  - "Online Vectorization"
  - "Temporal Fusion"
  - "Relation Embedding"
  - "Geometry-aware Alignment"
  - "DETR"
date: 2026-05-08
content_hash: 11857d76abf7eb16
---

# InteractionMap: Improving Online Vectorized HDMap Construction with Interaction

**Conference**: CVPR 2025  
**arXiv**: [2503.21659](https://arxiv.org/abs/2503.21659)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: HD Map Construction, Online Vectorization, Temporal Fusion, Relation Embedding, Geometry-aware Alignment, DETR

## TL;DR

This paper proposes InteractionMap, which comprehensively enhances information interaction in online vectorized HD map construction through three modules: point-level and instance-level relation embedding, keyframe-based hierarchical temporal fusion, and geometry-aware classification-localization alignment. It achieves state-of-the-art (SOTA) performance on both nuScenes (71.8 mAP) and Argoverse2 (74.7 mAP).

## Background & Motivation

High-definition (HD) vectorized maps are core components of autonomous driving systems, containing instance-level information such as lane dividers, road boundaries, and pedestrian crossings. Traditional HD maps rely on offline LiDAR SLAM-based construction, which suffers from high maintenance costs and difficult updates. Recently, DETR-based end-to-end vectorization methods (e.g., MapTR, MapTRv2) have become mainstream, yet three key issues remain:

1. **Limitations of Point-Set Representation**: Point-set representations in DETR-like detectors have limited capability in modeling instance-level information and lack explicit utilization of geometric priors.
2. **Insufficiency in Temporal Consistency**: Single-frame prediction is unstable in complex scenarios (such as vehicle occlusions), and existing streaming strategies are limited by GRU capacity, leading to the forgetting of long-term memory.
3. **Classification-Localization Misalignment**: The classification and regression branches are optimized independently, resulting in high-confidence predictions potentially corresponding to low-quality localization outputs.

## Method

### Overall Architecture

InteractionMap introduces three core modules to the DETR-like map vectorization framework:
- **Relation Embedding Module (REM)**: Injects explicit point-level and instance-level geometric relation priors into the self-attention of the decoder.
- **Temporal Fusion Module (TFM)**: Integrates temporal information from local to global based on keyframe hierarchical temporal fusion.
- **Geometry-aware Alignment Module (GAM)**: Resolves the classification-localization misalignment issue through geometry-aware classification loss and matching cost.

### Key Designs

**1. Keyframe Hierarchical Temporal Fusion (KFS)**

Drawing inspiration from keyframe strategies in robotic navigation, temporal fusion is divided into two levels:
- **Local BEV Fusion**: Recursively fuses BEV features of adjacent frames using a GRU to maintain local temporal consistency. The BEV features of the previous frame are aligned via ego-pose transformation: $\mathcal{F}_{submap}^t = ResBlock(LN(GRU(\tilde{\mathcal{F}}_{submap}^{t-1}, \mathcal{F}_{local}^t)))$
- **Global BEV Fusion**: Keyframes are selected based on a distance stride $d_{stride}$ instead of fixed time intervals, offering two strategies: KFS-streaming (recursive GRU fusion) and KFS-stacking (concatenation + ResBlock fusion).

**2. Relation Embedding Module (REM)**

Incorporates explicit geometric relation priors into the decoupled self-attention of the decoder:
- **Point-level Relation Embedding (PRE)**: Encodes spatial and directional relations between points based on normalized coordinate differences and edge direction cosine similarity differences.
- **Instance-level Relation Embedding (IRE)**: Establishes a joint semantic-geometric relationship between instances based on class score ranking relations and inter-instance Chamfer distances.

Both exhibit unbiasedness (relation value is 0 when $i=j$), and are projected into a high-dimensional space via sinusoidal positional encoding, linear transformation, and ReLU.

**3. Geometry-aware Alignment Module (GAM)**

Inspired by IoU-aware Focal Loss in 2D object detection, three geometry-aware classification scores (GCS) are proposed:
- $s_{p2p}$: Normalized point-to-point L1 distance score
- $s_{dir}$: Edge direction cosine similarity score  
- $s_{giou}$: Normalized GIoU score

In Focal Loss, the foreground objects use GCS as a soft target representation, replacing the hard label 1:

$$\mathcal{L}_{GFL} = \sum_{i=1}^{N_{pos}} s_{geo_i} BCE(s_{geo_i}, p_i) + \sum_{j=1}^{N_{neg}} \alpha p_j^{\gamma} BCE(p_j, 0)$$

Simultaneously, Geometry-aware Focal Cost (GFC) is introduced into the matching cost to suppress candidates with inaccurate localization.

### Loss & Training

The total loss consists of detection, segmentation, and auxiliary losses:
- **Detection Loss**: $\mathcal{L}_{det} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{p2p}\mathcal{L}_{p2p} + \lambda_{dir}\mathcal{L}_{dir}$
- **Segmentation Loss**: Query-based instance segmentation (MGFL + Dice Loss), providing BEV-level supervision.
- **Auxiliary Loss**: Depth prediction + perspective view semantic segmentation.

## Key Experimental Results

### nuScenes Validation Set (24 epochs, R50)

| Method | Temporal | AP_ped | AP_div | AP_bou | mAP |
|------|------|--------|--------|--------|-----|
| MapTRv2 | ✗ | 59.8 | 62.4 | 62.4 | 61.5 |
| HIMap | ✗ | 62.6 | 68.4 | 69.1 | 66.7 |
| HRMapNet | ✓ | 65.8 | 67.4 | 68.5 | 67.2 |
| **InteractionMap-R** | **✓** | **71.3** | **70.8** | **72.8** | **71.6** |
| **InteractionMap-C** | **✓** | **69.7** | **72.7** | **73.0** | **71.8** |

Achieves a **+10.3 mAP** improvement compared to MapTRv2 (24ep), and a **+4.6 mAP** improvement compared to HRMapNet.

### Argoverse2 Validation Set (110 epochs, R50)

| Method | AP_ped | AP_div | AP_bou | mAP |
|------|--------|--------|--------|-----|
| MapTRv2 | 68.1 | 68.3 | 69.7 | 68.7 |
| HIMap | 71.3 | 75.0 | 74.7 | 73.7 |
| **InteractionMap-C** | **73.8** | **75.5** | **74.9** | **74.7** |

## Highlights & Insights

1. **Systematic Information Interaction Design**: Progressive relation modeling from point-level to instance-level, along with hierarchical temporal fusion from local to global, reflects a deep understanding of HD map task characteristics.
2. **Geometry-Aware Alignment**: Adapts mature classification-localization alignment methods from 2D object detection to the map vectorization task, addressing the long-ignored misalignment prior issue in this field for the first time.
3. **Introduction of Keyframe Strategy**: Selecting keyframes based on distance strides rather than fixed time intervals is highly aligned with the practical needs of autonomous driving scenarios (frequent updates are unnecessary at low speeds, whereas more frames are needed at high speeds).
4. **Significant Performance Improvement**: Improvement of over 10 mAP points compared to MapTRv2 under the nuScenes 24 epochs condition, demonstrating the importance of information interaction.

## Limitations & Future Work

1. Employs only camera-only input (6 cameras) without exploring LiDAR fusion scenarios.
2. The KFS strategy introduces additional storage overhead for BEV features, requiring a trade-off with memory constraints in practical deployment.
3. KFS-streaming and KFS-stacking show trade-offs on different datasets in ablation studies, and a self-adaptive selection mechanism is lacking.
4. The Relation Embedding Module increases the computational complexity of the decoder, potentially affecting real-time performance.

## Related Work & Insights

- **Online HD Map Construction**: MapTR → MapTRv2 → BeMapNet → PivotNet → MapQR → MGMap → HIMap
- **Temporal Fusion Strategies**: StreamMapNet (streaming), SQD-MapNet (query denoising), MapTracker (tracking strategy)
- **Relationship Interaction of Map Elements**: ADMap (cascaded interaction), InsightMapper (intra-instance aggregation), GeMap (decoupled self-attention), HoMap (higher-order modeling)
- **Classification-Localization Alignment**: VFL (IoU-aware focal loss), LD (localization distillation)

## Rating

- **Novelty**: 3/5 — Although individual modules build upon existing foundations, their combination and adaptation to the HD map construction scenario are highly systematic.
- **Effectiveness**: 5/5 — Achieves significant SOTA improvements on two mainstream datasets.
- **Clarity**: 4/5 — The methodology description is clear, and the mathematical derivations are thorough.
- **Significance**: 4/5 — Substantially advances the field of HD map construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[CVPR 2025\] Online Video Understanding: OVBench and VideoChat-Online](online_video_understanding_ovbench_and_videochat-online.md)
- [\[ECCV 2024\] Stream Query Denoising for Vectorized HD-Map Construction](../../ECCV2024/autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)
- [\[CVPR 2025\] Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [\[CVPR 2025\] DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction](decoupledgaussian_object-scene_decoupling_for_physics-based_interaction.md)

</div>

<!-- RELATED:END -->
