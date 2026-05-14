---
title: >-
  [Paper Note] DAMap: Distance-aware MapNet for High Quality HD Map Construction
description: >-
  [ICCV 2025][Autonomous Driving][HD Map] This paper identifies two inherent deficiencies in current HD map construction methods regarding high-quality prediction — inappropriate classification labels and suboptimal task-s…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "HD Map"
  - "High-Quality Prediction"
  - "Task Alignment"
  - "Deformable Attention"
  - "Focal Loss"
date: 2026-05-08
content_hash: 85b16e9d7d22b861
---

# DAMap: Distance-aware MapNet for High Quality HD Map Construction

**Conference**: ICCV 2025
**arXiv**: [2510.22675](https://arxiv.org/abs/2510.22675)
**Code**: [github.com/jpdong-xjtu/DAMap](https://github.com/jpdong-xjtu/DAMap)
**Area**: Autonomous Driving / Online HD Map Construction
**Keywords**: HD Map, High-Quality Prediction, Task Alignment, Deformable Attention, Focal Loss

## TL;DR

This paper identifies two inherent deficiencies in current HD map construction methods regarding high-quality prediction — inappropriate classification labels and suboptimal task-specific features — and proposes DAMap (comprising three components: DAFL, HLS, and TMDA) to systematically address task misalignment, achieving consistent gains of 2–3 mAP across multiple baselines on NuScenes and Argoverse2.

## Background & Motivation

Online vectorized HD map construction is critical for autonomous driving safety. While DETR-style methods represented by MapTRv2 have achieved notable progress, they perform poorly on **high-quality prediction** (simultaneously achieving high classification scores and high localization accuracy):
- At a 0.5m threshold, recall is only 35% even when the classification threshold is lowered to 0.6.
- This leads to dangerous scenarios such as missed detections of pedestrian crossings.

The authors identify two fundamental causes:

**Inappropriate classification labels**: In one-to-many matching, each GT instance corresponds to multiple candidate predictions (7 in MapTRv2), all sharing the same classification label of "1." Candidates with poor localization quality also receive label "1," causing the model to learn to output predictions with high classification scores but low localization accuracy.

**Suboptimal task features**: Classification and localization share instance features extracted via cross-attention. Classification requires semantic information from salient regions within the instance, while localization requires precise positional information at instance boundaries. Sharing features prevents either task from reaching its optimum.

## Method

### Overall Architecture

A standard BEV perception pipeline: multi-view images → shared backbone → BEV features → Transformer decoder (self-attention + cross-attention + FFN) → classification and localization heads. DAMap introduces three plug-and-play components on top of this framework.

### Key Designs

1. **Distance-aware Focal Loss (DAFL)**:

    - Core Idea: Use localization quality as the classification label rather than a binary label.
    - The localization loss $\mathcal{L}_{dist}$ is converted to a probability via maximum likelihood estimation: $P_{dist}^i = e^{-\lambda \mathcal{L}_{dist}^i}$
        - When $\mathcal{L}_{dist} = 0$, $P_{dist} = 1$ (perfect localization → high label).
        - When $\mathcal{L}_{dist} \to \infty$, $P_{dist} \to 0$ (poor localization → low label).
    - Replacing binary labels with continuous labels yields DAFL:
    $\text{DAFL}(p, y) = -(y-p)^\gamma (y\log(p) + (1-y)\log(1-p))$
    - $y \in [0,1]$ serves as the localization confidence, enabling classification scores to reflect localization quality.
    - For negative samples $(y=0)$, DAFL reduces exactly to the standard Focal Loss.

2. **Hybrid Loss Scheme (HLS)**:

    - Problem: Randomly initialized queries in early decoder layers produce poor localization quality, degrading the training signal for DAFL.
    - Solution: Apply standard Focal Loss in the first $L_1$ decoder layers and DAFL in the final $L_2$ layers.
    - This exploits the cascaded nature of the decoder — predictions from later layers exhibit higher localization accuracy and are thus better suited for DAFL.
    - Total classification loss: $\mathcal{L}_{cls} = \sum_{l=1}^{L_1} \text{FL}(p,y) + \sum_{l=1}^{L_2} \text{DAFL}(p,y)$
    - HLS+DAFL is a pure loss-function improvement with **zero additional parameters or computational cost at inference**.

3. **Task Modulated Deformable Attention (TMDA)**:

    - Query channels are doubled, with half dedicated to classification and half to localization.
    - After self-attention, the query is split into $\mathbf{Q}_{cls}$ and $\mathbf{Q}_{loc}$.
    - Key design choices:
        - Task-specific attention weights: $\mathbf{A}_{cls} = W_a \mathbf{Q}_{cls}$, $\mathbf{A}_{loc} = W_a' \mathbf{Q}_{loc}$
        - Task-shared sampling offsets: $\Delta r = W_p \text{Cat}(\mathbf{Q}_{cls}, \mathbf{Q}_{loc})$
    - Design rationale: (1) jointly learning both task-specific weights and offsets introduces too many variables and is difficult to optimize; (2) offset optimization has an unbounded objective and is inherently harder to train.
    - Outputs: $\hat{\mathbf{Q}}_{cls} = \text{Softmax}(\mathbf{A}_{cls})\mathbf{V}$, $\hat{\mathbf{Q}}_{loc} = \text{Softmax}(\mathbf{A}_{loc})\mathbf{V}$
    - Each output is further enhanced by independent FFNs before being fed into the corresponding task head.

### Loss & Training

- Localization loss: L1 distance measuring deviation between predicted and GT points.
- Classification loss: HLS combining Focal Loss and DAFL.
- Matching: Follows MapTRv2's one-to-many instance matching with point-level matching.
- The hyperparameter $\lambda$ controls the sensitivity of converting localization loss to probability.

## Key Experimental Results

### Main Results

| Method | Epoch | mAP(hard) | mAP(easy) | Gain(hard) | Gain(easy) |
|--------|-------|-----------|-----------|------------|------------|
| **NuScenes ResNet-50** | | | | | |
| MapTRv2† | 24 | 36.6 | 60.4 | - | - |
| MapTRv2+**Ours** | 24 | 39.0 | 62.8 | +2.4 | +2.4 |
| MapQR† | 24 | 43.3 | 66.4 | - | - |
| MapQR+**Ours** | 24 | **46.0** | **68.8** | +2.6 | +2.4 |
| Mask2Map | 24 | - | 71.6 | - | - |
| Mask2Map+**Ours** | 24 | - | **72.6** | - | +1.0 |
| MapTRv2† | 110 | 44.9 | 68.3 | - | - |
| MapTRv2+**Ours** | 110 | **47.4** | **70.4** | +2.5 | +2.1 |
| **Argoverse2 ResNet-50** | | | | | |
| MapTRv2† | 6 | 38.1 | 63.6 | - | - |
| MapTRv2+**Ours** | 6 | 40.9 | 66.2 | +2.8 | +2.6 |
| MapQR† | 6 | 41.1 | 65.4 | - | - |
| MapQR+**Ours** | 6 | **43.6** | **67.4** | +2.5 | +2.0 |

### Ablation Study

| Component | AP_ped | AP_div | AP_bou | mAP(easy) | Gain |
|-----------|--------|--------|--------|-----------|------|
| Baseline | 58.1 | 60.8 | 62.3 | 60.4 | - |
| +DAFL | 58.6 | 61.1 | 63.1 | 60.9 | +0.5 |
| +DAFL+HLS | 58.5 | 62.9 | 63.4 | 61.6 | +1.2 |
| +TMDA (alone) | 58.5 | 63.1 | 63.2 | 61.6 | +1.2 |
| +DAFL+HLS+TMDA | **58.5** | **64.7** | **65.1** | **62.8** | **+2.4** |

| TMDA Design Variant | mAP | Params |
|---------------------|-----|--------|
| Baseline | 60.4 | 40M |
| Setting 1: Fully task-specific (weights + offsets) | 60.7 | 52M |
| Setting 2: Task-specific offsets + shared weights | 60.9 | 52M |
| **Setting 4 (Ours): Shared offsets + task-specific weights** | **61.6** | 52M |

### Key Findings

- The three components are complementary, addressing distinct issues: DAFL resolves label misalignment, TMDA resolves feature conflict, and HLS unlocks the full potential of DAFL.
- HLS+DAFL alone (with zero inference overhead) yields 1.1–2.1 mAP improvement.
- TMDA's "shared offsets + task-specific weights" design outperforms the intuitively appealing "fully task-specific" design, demonstrating that reducing optimization variables is more effective.
- Consistent improvements are observed across different datasets (NuScenes/Argoverse2), baselines (MapTRv2/MapQR/Mask2Map), backbones (ResNet-50/Swin-B), and training schedules (6/24/110 epochs).
- Gains are larger when the centerline category is included (4–5 mAP improvement), indicating greater value in multi-category scenarios.

## Highlights & Insights

- **Precise problem analysis**: The recall@0.5m analysis clearly exposes the high-quality prediction bottleneck in existing methods.
- **Elegant design of DAFL**: Converting the unbounded localization loss to a $[0,1]$ probability via likelihood estimation as a soft label is related to QFL but is more principled and elegant.
- **Cascade insight of HLS**: Leveraging the cascaded nature of the decoder — using FL for convergence in early layers and DAFL for refinement in later layers — reflects a deep understanding of training dynamics.
- All components are plug-and-play and can be combined with any DETR-style HD map method.

## Limitations & Future Work

- The hyperparameter $\lambda$ in DAFL requires tuning to control the sensitivity of the localization-to-probability conversion.
- TMDA introduces additional parameters (40M → 52M), which may require trade-off consideration in resource-constrained scenarios.
- Compatibility with long-sequence temporal modeling in online/streaming settings has not been explored.
- Although recall for high-quality prediction improves, the absolute values still leave considerable room for further gains.
- The paper focuses exclusively on classification–localization misalignment and does not address relational modeling among different map element categories.

## Related Work & Insights

- **GFL/VFNet**: Quality–classification alignment methods in 2D detection; this paper migrates the concept to HD map construction.
- **MapTRv2/MapQR**: DETR-style HD map baselines; DAMap is fully complementary to both.
- **Double-Head R-CNN/TSD**: Pioneers of task decoupling; TMDA realizes decoupling at the Deformable Attention level.
- The alignment idea for high-quality prediction is generalizable to other BEV perception tasks such as 3D detection and motion planning.

## Rating

- Novelty: ⭐⭐⭐⭐ Each component has its own design highlights, though the core ideas are adapted from 2D detection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across multiple datasets, baselines, backbones, and training schedules with highly detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Excellent motivation analysis and clear method description.
- Value: ⭐⭐⭐⭐ Strong practical utility due to the plug-and-play design, especially the zero inference overhead of HLS+DAFL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[AAAI 2026\] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors](../../AAAI2026/autonomous_driving/priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[CVPR 2026\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2026/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)

</div>

<!-- RELATED:END -->
