---
title: >-
  [Paper Note] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction
description: >-
  [CVPR 2026][Autonomous Driving][Geospatial contrastive learning] This paper proposes MapGCLR, a geospatial contrastive learning strategy that enforces consistent BEV feature representations for overlapping regions across different traversals. Operating within a semi-supervised framework, the method achieves 13%–42% relative performance gains on online vectorized HD map construction using only 5%–20% of labeled data.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Geospatial contrastive learning
  - online HD map
  - semi-supervised learning
  - BEV features
  - multi-traversal
date: 2026-05-08
content_hash: 2f5f29334352dbbb
---

# MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction

**Conference**: CVPR 2026
**arXiv**: [2603.10688](https://arxiv.org/abs/2603.10688)
**Code**: N/A
**Area**: Autonomous Driving / HD Map Construction
**Keywords**: Geospatial contrastive learning, online HD map, semi-supervised learning, BEV features, multi-traversal

## TL;DR

This paper proposes MapGCLR, a geospatial contrastive learning strategy that enforces consistent BEV feature representations for overlapping regions across different traversals. Operating within a semi-supervised framework, the method achieves 13%–42% relative performance gains on online vectorized HD map construction using only 5%–20% of labeled data.

## Background & Motivation

**Background**: Online HD map construction has emerged as a scalable alternative to offline HD maps for autonomous driving. Methods such as MapTR, MapTRv2, and MapTracker predict vectorized map elements (lane lines, road boundaries, etc.) in real time from 360° visual inputs, yet these supervised approaches still rely on large volumes of annotated data.

**Limitations of Prior Work**: (1) HD map annotation is extremely costly, requiring specialized sensor platforms and partial human labeling; (2) existing semi-supervised methods (PseudoMapTrainer, Lilja et al.) depend on pseudo-label generation and are primarily designed for semantic segmentation rather than vectorized prediction; (3) the geospatial consistency inherent in multi-traversal data has not been adequately exploited.

**Key Challenge**: Annotation cost is the primary bottleneck for online HD map construction. Autonomous vehicles naturally accumulate large volumes of unlabeled multi-traversal data during routine operation—the central question is how to leverage this freely available resource.

**Goal**: To improve BEV feature representation quality by exploiting geospatial consistency in unlabeled multi-traversal data, thereby enhancing online vectorized HD map construction under limited annotation budgets.

**Key Insight**: BEV grid cells corresponding to the same geographic location across different traversals serve as a natural self-supervised signal, replacing manually designed image augmentations.

**Core Idea**: The same geographic location observed in different traversals should yield consistent BEV features—this constraint is used as the objective for contrastive learning.

## Method

### Overall Architecture

The semi-supervised training pipeline consists of two branches: (1) **Supervised branch**: a small set of labeled samples passes through the full MapTRv2 encoder–decoder to compute the supervised loss $\mathcal{L}_{sup}$; (2) **Self-supervised branch**: large-scale unlabeled multi-traversal data is encoded into BEV feature grids, and geospatial contrastive loss $\mathcal{L}_{GCLR}$ is applied. Each batch contains $n$ labeled samples and $2m$ unlabeled samples ($m$ reference–adjacent pairs).

### Key Designs

1. **Geospatial Multi-Traversal Analysis and Data Partitioning**: A systematic dataset analysis method is proposed. All poses are transformed into a global reference frame and partitioned by city region. For each traversal, a perception-range bounding box (lateral $\pm x$ m, longitudinal $\pm y$ m) is computed per pose according to vehicle heading, and all bounding boxes within a traversal are merged into a polygon. If this polygon intersects with that of another traversal, the pair is classified as "multi-traversal"; otherwise it is "single-traversal" (cases where only two trajectories mutually cross are also classified as single-traversal due to insufficient diversity). A spatial graph $G=(V,E)$ is then constructed, where nodes represent vehicle poses and edges connect pose pairs whose perception-grid IoU falls within $[\text{IoU}_{min}, \text{IoU}_{max}]$, ensuring that overlapping regions are sufficiently related yet not identical.

2. **Geospatial Contrastive Learning**: Building on the SimCLR framework, image augmentations are replaced by geospatial correspondences. Given the BEV grids $B_{SSL,R}$ and $B_{SSL,A}$ of a reference pose $R$ and an adjacent pose $A$, both are transformed into the global coordinate frame. **Positive samples**: BEV cells $c_a$ in the overlapping region of the reference grid are randomly sampled as anchors; the corresponding cell $c_p$ in the adjacent grid is identified via nearest-neighbor search. **Negative samples**: BEV cells are randomly sampled from both grids, excluding anchors and positives. A projection head $h$ maps BEV cell features $\mathbf{f}$ into a contrastive space $\mathbf{z} \in \mathcal{Z}$, decoupling the learning domain from the application domain.

3. **InfoNCE Contrastive Loss**: $\mathcal{L}_{GCLR} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau)}{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau) + \sum_{k=1}^K \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k^-) / \tau)}$, where $\text{sim}(\cdot,\cdot)$ denotes cosine similarity and $\tau$ is the temperature parameter. The loss encourages BEV cell embeddings at the same geographic location to be similar across traversals while pushing apart embeddings from different locations.

### Loss & Training

The total loss is a weighted combination of the supervised and contrastive losses: $\mathcal{L}_{semi} = \lambda_{sup} \mathcal{L}_{sup} + \lambda_{GCLR} \mathcal{L}_{GCLR}$. The weighting factors serve both for magnitude normalization and relative influence control. The architecture is based on MapTRv2 with a ResNet-50 backbone; labeled and unlabeled data are processed jointly within each training batch in a single-stage training procedure.

## Key Experimental Results

### Main Results

| Labeled Data | SSL | AP_dsh | AP_sol | AP_bou | AP_cen | AP_ped | mAP | Abs. Gain | Rel. Gain |
|---|---|---|---|---|---|---|---|---|---|
| 2.5% | ✗ | 4.3 | 5.0 | 9.6 | 11.9 | 1.5 | 6.5 | — | — |
| 2.5% | ✓ | 5.2 | 6.7 | 12.2 | 17.0 | 1.6 | **8.5** | +2.0 | **+31%** |
| 5% | ✗ | 10.3 | 9.5 | 20.5 | 19.1 | 7.3 | 13.3 | — | — |
| 5% | ✓ | 15.4 | 18.7 | 24.8 | 25.4 | 9.9 | **18.9** | +5.6 | **+42%** |
| 10% | ✗ | 17.6 | 20.9 | 31.9 | 27.1 | 12.4 | 22.0 | — | — |
| 10% | ✓ | 20.8 | 30.5 | 34.5 | 32.4 | 18.2 | **27.3** | +5.3 | **+24%** |
| 20% | ✗ | 27.2 | 32.1 | 38.9 | 34.7 | 22.3 | 31.0 | — | — |
| 20% | ✓ | 31.2 | 38.8 | 39.9 | 37.5 | 26.9 | **34.9** | +3.9 | **+13%** |

> Evaluated on the Argoverse 2 dataset. Consistent gains are observed across all labeling ratios; the fewer the labeled samples, the larger the relative gain—a 42% relative improvement at 5% annotation is roughly equivalent to doubling the labeled set.

### Ablation Study

| Supervised-Only Data Ratio | mAP |
|---|---|
| 2.5% | 6.5 |
| 5% | 13.3 |
| 5% + SSL | **18.9** |
| 10% | 22.0 |
| 10% + SSL | **27.3** |
| 20% | 31.0 |
| 30% | 36.6 |
| 40% | 39.8 |

> 5% + SSL (18.9) approaches 10% supervised-only (22.0); 10% + SSL (27.3) approaches 20% supervised-only (31.0). The semi-supervised approach is approximately equivalent to doubling the annotation budget.

### Key Findings

- Qualitative PCA visualizations demonstrate that the semi-supervised BEV feature space exhibits clearer semantic separation, particularly at road boundaries and ego-lane regions.
- The supervised-only baseline produces anomalous feature clusters at fixed BEV grid positions unrelated to geographic content; geospatial contrastive learning completely eliminates this artifact.
- The Argoverse 2 dataset exhibits substantial multi-traversal overlap, making it naturally well-suited for the proposed method.
- Relative gains decrease monotonically as the labeling ratio increases (42% at 5% → 13% at 20%), confirming the method's particular value under data-scarce conditions.

## Highlights & Insights

- **Natural augmentation**: The core insight is treating geospatial overlap across traversals as a natural data augmentation—no handcrafted augmentation policy is needed, as repeated real-world driving already provides the best possible augmentations.
- **Simplicity and effectiveness**: The method is a clean extension of SimCLR-style contrastive learning without additional complex modules, yet yields substantial gains.
- **Dataset analysis as a contribution**: The multi-traversal partitioning and spatial graph construction methodology constitutes a valuable standalone tool for any multi-traversal-based autonomous driving research.
- **Compatibility with vectorized methods**: Unlike prior semi-supervised approaches restricted to semantic segmentation, MapGCLR is the first to achieve semi-supervised learning for vectorized HD map construction.

## Limitations & Future Work

- Validation is limited to the single-frame MapTRv2 architecture; integration with temporal-memory-based methods such as MapTracker or StreamMapNet remains unexplored.
- A two-stage training paradigm (self-supervised pre-training followed by fine-tuning) is not compared against the current single-stage approach.
- The projection head design is relatively simple (single layer); more expressive projection architectures may yield further improvements.
- The method requires multi-traversal coverage and may be less effective in newly developed areas or infrequently traveled road segments.
- The impact of dynamic scene changes at the same location (e.g., construction, seasonal variation) on feature consistency is not addressed.

## Related Work & Insights

- **SimCLR**: The foundational contrastive learning framework; MapGCLR extends the notion of "augmentation" from image transformations to geospatial overlap.
- **MapTRv2**: The standard baseline for vectorized HD map construction, upon which the SSL branch is built.
- **HRMapNet / RTMap**: These methods exploit multi-traversal data to construct global map priors but introduce additional complexity at inference time; MapGCLR uses multi-traversal data only during training.
- **Broader inspiration**: The geospatial contrastive learning paradigm is generalizable to other BEV perception tasks such as 3D object detection and occupancy prediction, where repeated observations of the same location should yield consistent representations.

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐** |

---

## Related Papers

- [\[CVPR 2026\] EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)
- [\[CVPR 2026\] Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them](failure_modes_for_deep_learning-based_online_mapping_how_to_measure_and_address_.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)
- [\[CVPR 2026\] ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)
- [\[CVPR 2026\] Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them](failure_modes_for_deep_learning-based_online_mapping_how_to_measure_and_address_.md)
- [\[AAAI 2026\] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors](../../AAAI2026/autonomous_driving/priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)

<!-- RELATED:END -->
