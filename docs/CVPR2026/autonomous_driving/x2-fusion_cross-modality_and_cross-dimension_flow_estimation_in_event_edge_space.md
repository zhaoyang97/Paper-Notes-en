---
title: >-
  [Paper Note] x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space
description: >-
  [CVPR 2026][Autonomous Driving][optical flow] This paper proposes x2-Fusion, which constructs a unified Event Edge Space anchored on spatiotemporal edge signals from event cameras. Image, LiDAR…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "optical flow"
  - "scene flow"
  - "event camera"
  - "multimodal fusion"
  - "edge space"
date: 2026-05-08
content_hash: fbe0ea62e461cd04
---

# x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space

**Conference**: CVPR 2026
**arXiv**: [2603.16671](https://arxiv.org/abs/2603.16671)
**Code**: N/A
**Area**: Autonomous Driving
**Keywords**: optical flow, scene flow, event camera, multimodal fusion, edge space

## TL;DR

This paper proposes x2-Fusion, which constructs a unified Event Edge Space anchored on spatiotemporal edge signals from event cameras. Image, LiDAR, and event features are aligned into this homogeneous edge space, followed by reliability-aware adaptive fusion and cross-dimension contrastive learning to jointly estimate 2D optical flow and 3D scene flow, achieving state-of-the-art performance on both synthetic and real-world datasets.

## Background & Motivation

Optical flow and scene flow are fundamental tools for dynamic scene understanding. Existing multimodal fusion methods fuse image, LiDAR, and event features while keeping them in their respective heterogeneous spaces, which introduces three problems:

**High complexity**: Without a shared channel basis, pairwise modality alignment is required, leading to an excessive number of modules.

**Information erosion**: Heterogeneous spaces delay fusion to late stages, making early-stage distortion difficult to correct.

**High fragility**: Without a common representational basis, alignment itself collapses under degraded conditions.

Core insight: Event cameras naturally provide spatiotemporal edge signals — pixel-level brightness changes that precisely mark motion edges — and can serve as a universal "edge anchor" for unifying all modalities.

## Method

### Overall Architecture

Event Edge Encoder pre-training → frozen as edge prototype → image/LiDAR encoders aligned to Event Edge Space → reliability-aware adaptive fusion → cross-dimension contrastive learning → 2D/3D flow output.

### Key Designs

#### 1. Event Edge Space

**Why edges?** Edges are modality-agnostic structural information that remains consistent across different sensors.

**Why events?** Event cameras trigger precisely at motion edges, share 2D coordinates with images, and share sparse asynchronous sampling characteristics with LiDAR — making them a natural bridge between the two.

**Event Edge Encoder pre-training**: Voxelized event streams → sparse 3D CNN → multi-scale feature pyramid. Self-supervised pre-training: predict future edge intensity from past events.

Edge intensity is defined as: $e^E(x,y) = \tilde{A}^E(x,y)(1 - \tilde{\sigma}_t(x,y))$, combining normalized event activity and temporal variance.

#### 2. Image–LiDAR Alignment

The event encoder is frozen, and its features serve as fixed edge prototypes. Image and LiDAR encoders are mapped to the same dimensional space via projection heads.

Edge-anchored symmetric regularization: L1 distances among the three modalities are computed separately in 2D (pixel-level) and 3D (point-level), weighted by the event edge map $e^E$:
$$\mathcal{L}_{align} = \lambda_{2D} \cdot \mathcal{L}_{align}^{2D} + \lambda_{3D} \cdot \mathcal{L}_{align}^{3D}$$

#### 3. Reliability-Aware Adaptive Fusion

Two-level reliability estimation:
- **Global reliability** $\omega_m$: measures each modality's consistency with the event motion signal via spatiotemporal decomposition (temporal difference + spatial gradient)
- **Local reliability** $\mathcal{A}_m(x)$: high-pass filtering + average pooling + grouped convolution followed by softmax

Fusion: $F_{fused}(x) = \sum_m \frac{\omega_m \mathcal{A}_m(x)}{\sum_n \omega_n \mathcal{A}_n(x)} Z_m(x)$

A cross-attention Transformer further enhances the fused features.

#### 4. Cross-Dimension Contrastive Learning

Explicit constraints are imposed on inter-frame motion consistency and 2D–3D geometric consistency, enabling optical flow and scene flow to mutually reinforce each other.

## Key Experimental Results

### EKubric Synthetic Dataset

| Method | EPE_2D ↓ | ACC_1px ↑ | EPE_3D ↓ | ACC_.05 ↑ |
|--------|----------|-----------|----------|-----------|
| RPEFlow | 0.439 | 95.99% | 0.027 | 95.33% |
| **x2-Fusion** | **0.430** | **96.86%** | **0.024** | **96.78%** |

### DSEC Real-World Dataset

| Method | EPE_2D ↓ | ACC_1px ↑ | EPE_3D ↓ |
|--------|----------|-----------|----------|
| RPEFlow | 0.326 | 95.28% | 0.103 |
| **x2-Fusion** | **0.305** | **95.60%** | **0.092** |

### Degraded Scenarios

| Condition | Improvement |
|-----------|-------------|
| Extreme lighting | Significant |
| Sparse LiDAR | Significant |

### Ablation Study

| Configuration | EPE_2D | EPE_3D | Note |
|---------------|--------|--------|------|
| w/o Event Edge Space | +0.05 | +0.003 | Homogeneous space is critical for fusion |
| w/o reliability fusion | +0.03 | +0.002 | Adaptive weighting is especially important under degraded conditions |
| w/o cross-dimension contrast | +0.02 | +0.003 | 2D–3D mutual enhancement is effective |

### Key Findings

- Event Edge Space is the first design to unify three modalities into a homogeneous edge space.
- Reliability-aware fusion yields the greatest advantage in degraded scenarios.
- Cross-dimension contrastive learning enables the 2D and 3D tasks to mutually promote each other.

## Highlights & Insights

1. The design philosophy of Event Edge Space is elegant — using the natural edge signals of event cameras as a "universal anchor."
2. Fusion is simplified from "pairwise alignment in heterogeneous spaces" to "weight assignment within a homogeneous space."
3. Edge intensity serves as alignment weights — enforcing precise alignment at edges while relaxing constraints elsewhere.

## Limitations & Future Work

1. Event encoder pre-training increases the complexity of the training pipeline.
2. Freezing the event encoder may limit adaptive capacity.
3. Dynamic object occlusion is not currently addressed.
4. Dependence on event camera hardware limits applicability in image-only and LiDAR-only scenarios.

## Related Work & Insights

- Compared to RPEFlow (staged fusion): the unified space design is more concise.
- Compared to VisMoFlow (hand-crafted physical space): Event Edge Space is data-driven and more generalizable.
- The perspective of event cameras as "edge sensors" is worth exploring in a broader range of tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The Event Edge Space concept is highly original; the homogeneous fusion paradigm is unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic and real-world data, with degraded scenario evaluation.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear; paradigm comparison figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides a fundamentally new perspective for multimodal fusion-based flow estimation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](../../ICLR2026/autonomous_driving/x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[CVPR 2026\] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)
- [\[CVPR 2026\] FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision](flashcap_millisecond-accurate_human_motion_capture_via_flashing_leds_and_event-b.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [\[CVPR 2026\] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)

</div>

<!-- RELATED:END -->
