---
title: >-
  [Paper Note] MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception
description: >-
  [ICCV 2025][Autonomous Driving][Multi-task Learning] This paper proposes the MAESTRO framework, which generates task-specific features and suppresses inter-task interference in multi-task 3D perception through three modu…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Multi-task Learning"
  - "3D Perception"
  - "BEV Segmentation"
  - "3D Object Detection"
  - "Occupancy Prediction"
date: 2026-05-08
content_hash: 65a9712cac5a32b3
---

# MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception

**Conference**: ICCV 2025
**arXiv**: [2509.17462](https://arxiv.org/abs/2509.17462)
**Code**: To be released
**Area**: Autonomous Driving
**Keywords**: Multi-task Learning, 3D Perception, BEV Segmentation, 3D Object Detection, Occupancy Prediction

## TL;DR

This paper proposes the MAESTRO framework, which generates task-specific features and suppresses inter-task interference in multi-task 3D perception through three modules: Class-wise Prototype Generator (CPG), Task-Specific Feature Generator (TSFG), and Scene Prototype Aggregator (SPA). MAESTRO simultaneously surpasses single-task models on 3D object detection, BEV map segmentation, and 3D occupancy prediction.

## Background & Motivation

Autonomous driving perception systems must simultaneously execute multiple tasks: 3D object detection focuses on movable foreground objects, BEV map segmentation focuses on static background structures, and 3D occupancy prediction requires attention to both foreground and background. Multi-task learning (MTL) can improve computational efficiency via a shared backbone, but inconsistent gradient directions across tasks lead to **task conflicts** that degrade per-task performance.

Existing MTL methods partially alleviate these conflicts but still fail to generate truly task-specific feature representations, leaving performance below independently trained single-task models. The authors identify fundamental differences in the semantic cues and spatial regions attended to by different tasks, motivating a mechanism that **enhances task-relevant information and suppresses irrelevant information** from shared features.

## Method

### Overall Architecture

MAESTRO first extracts 2D features from multi-view images via a shared backbone, then lifts them to a 3D voxel representation $F_s \in \mathbb{R}^{C \times X \times Y \times Z}$ using the Lift-Splat-Shoot (LSS) method. The features are subsequently processed by CPG, TSFG, and SPA in sequence to generate task-specific features, which are then forwarded to individual task heads for prediction.

### Key Designs

1. **Class-wise Prototype Generator (CPG)**: Semantic categories are partitioned into a foreground group (vehicles, pedestrians, etc.) and a background group (roads, sidewalks, etc.). A lightweight mask classifier computes per-voxel semantic confidence $S_v$ from the shared voxel features, generates binary masks $B_k$ based on the highest-confidence class, and applies average pooling over masked regions to obtain class prototypes $P_k$. Foreground prototypes are assigned to the detection task, background prototypes to BEV segmentation, and both are jointly assigned to occupancy prediction. The core formula is $P_k = \text{AvgPool}(F_s \otimes B_k)$. **Design Motivation**: Semantic grouping provides targeted prior information for each task.

2. **Task-Specific Feature Generator (TSFG)**: Comprises three sub-modules. (a) **Task-dependent feature transformation**: Transforms shared voxel features into BEV-domain features (detection/segmentation) or voxel-domain features (occupancy prediction). (b) **Adaptive feature enhancement**: Computes dot products between the prototype group and the transformed features to generate prototype-level features (activating spatial regions semantically aligned with prototypes), while channel attention generates prototype-aware features; these are concatenated and convolved to yield the enhanced feature $\tilde{F}_t$. (c) **Feature suppression**: A lightweight CNN derives a suppression score map $S^{supp}_t$ from the prototype-aware features, which is applied element-wise to the enhanced features via gating: $F^{TS}_t = \tilde{F}_t \otimes S^{supp}_t$. **Design Motivation**: The enhance-then-suppress paradigm thoroughly eliminates task-irrelevant information.

3. **Scene Prototype Aggregator (SPA)**: Detection prototypes $P_{Det}$ are generated from predicted bounding boxes via RoIAlign from the detection head, and map prototypes $P_{Map}$ are generated from predicted masks via mask average pooling from the segmentation head. These task-guided prototypes are aggregated into the occupancy prediction prototype group through semantic alignment rules, forming scene prototypes as initial queries for the occupancy decoder. **Design Motivation**: Complementary semantic information from detection and segmentation is exploited to enhance occupancy prediction without degrading other tasks.

### Loss & Training

The total loss is the sum of six terms:
$$L_{total} = L_{depth} + L_{CPG} + L_{Sup} + L_{det} + L_{map} + L_{occ}$$

- $L_{CPG}$ supervises class mask classification using Dice Loss + Lovász Loss
- $L_{Sup}$ supervises task-wise suppression score maps using Focal Loss
- Each task head employs the respective losses from CenterPoint (detection), BEVFusion (segmentation), and OccFormer (occupancy prediction)
- ResNet-50 backbone, AdamW optimizer (lr=1e-4), 24 epochs, without CBGS

## Key Experimental Results

### Main Results

| Method | mAP | NDS | mIoU (Map) | mIoU (Occ) | Latency (ms) |
|--------|-----|-----|-----------|-----------|--------------|
| Baseline-STL | 33.8 | 41.7 | 47.5 | 36.5 | 405.9 |
| Baseline-MTL | 32.7 | 38.2 | 43.5 | 36.0 | 219.6 |
| BEVFusion MTL | 33.6 | 39.2 | 44.0 | - | - |
| DualBEV (STL) | 35.2 | 42.5 | - | - | 65.1 |
| DifFUSER (STL) | - | - | 48.3 | - | 92.2 |
| FB-Occ (STL) | - | - | - | 37.4 | 129.7 |
| **MAESTRO-MTL** | **36.4** | **43.2** | **51.3** | **38.6** | 250.3 |

Compared to Baseline-STL, MAESTRO achieves gains of +2.6% mAP, +3.8% mIoU (Map), and +2.1% mIoU (Occ) across the three tasks while reducing latency by 155.6 ms; gains over Baseline-MTL are even more pronounced.

### Ablation Study

| CPG | TSFG (Det) | TSFG (Map) | TSFG (Occ) | SPA | mAP | NDS | mIoU (Map) | mIoU (Occ) |
|-----|-----------|-----------|-----------|-----|-----|-----|-----------|-----------|
| ✗ | ✗ | ✗ | ✗ | ✗ | 31.3 | 32.3 | 40.4 | 33.6 |
| ✓ | ✗ | ✗ | ✗ | ✗ | 31.3 | 32.4 | 40.3 | 34.6 |
| ✓ | ✓ | ✓ | ✓ | ✗ | 32.6 | 34.3 | 44.2 | 36.9 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **32.6** | **34.3** | **44.2** | **36.9** |

TSFG sub-module ablation: prototype-level features contribute +1.8% mIoU (Map), prototype-aware features +1.3%, and feature suppression +0.7%. In SPA, removing map prototypes reduces occupancy mIoU by 0.6%, and further removing detection prototypes reduces it by an additional 0.4%.

### Key Findings

- The MTL framework is the first to comprehensively surpass independently trained STL models across all three 3D perception tasks.
- Foreground/background semantic grouping prototypes serve as effective priors for alleviating task conflicts.
- The feature suppression module contributes most significantly to BEV map segmentation, which requires suppressing foreground interference.
- Output information from detection and segmentation provides significant complementary benefits to occupancy prediction.

## Highlights & Insights

- The core insight is distinctive: different 3D perception tasks inherently attend to foreground and background differently, and leveraging this semantic divergence to design prototype groupings is a natural and effective approach.
- The two-stage "enhance-then-suppress" feature refinement mechanism is elegantly designed—first amplifying relevant information via prototype groups, then filtering irrelevant components via learnable suppression score maps.
- SPA's use of outputs from already-trained tasks as auxiliary information is a clever approach to modeling inter-task dependencies.
- The overall framework exhibits high modularity, with each component's contribution independently verifiable.

## Limitations & Future Work

- Ablation experiments are conducted on 1/4 of the training set, and the generalizability of the ablation conclusions remains to be validated.
- The manual foreground/background grouping rules may not generalize to a broader range of tasks (e.g., lane detection).
- Evaluation is conducted only on the nuScenes validation set; performance on other large-scale datasets (e.g., Waymo) has not been verified.
- The suppression score maps require additional GT supervision (RoI masks), increasing annotation dependency.
- The effects of temporal information fusion and larger backbone networks remain unexplored.

## Related Work & Insights

- Compared to existing MTL methods such as HENet and SOGDet, MAESTRO not only alleviates task conflicts but also achieves genuinely task-specific feature generation through the prototype mechanism.
- The MoE approach in TaskExpert shares a conceptual similarity with the prototype grouping idea in this work, as both explore how to assign distinct feature processing pathways to different tasks.
- The prototype learning concept originates from Prototypical Networks and is innovatively applied here to multi-task feature disentanglement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The foreground/background prototype grouping combined with enhancement-suppression design is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive main experiment comparisons with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-executed illustrations.
- **Value**: ⭐⭐⭐⭐ Achieving MTL superiority over STL in multi-task 3D perception represents meaningful progress.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](../../AAAI2026/autonomous_driving/task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)
- [\[ICCV 2025\] DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic](duet_dual_incremental_object_detection_via_exemplar-free_task_arithmetic.md)
- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[ICCV 2025\] Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception](unleashing_the_temporal_potential_of_stereo_event_cameras_for_continuous-time_3d.md)
- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)

</div>

<!-- RELATED:END -->
