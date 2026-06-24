---
title: >-
  [Paper Note] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning
description: >-
  [CVPR2025][Remote Sensing][UAV] This paper proposes a new task named UAV Scene Change Captioning (UAV-SCC) and a novel HDC-CL framework. It models the overlapping and non-overlapping regions of image pairs under moving viewpoints using a Dynamic Adaptive Layout Transformer, enhances viewpoint shift direction awareness via hierarchical cross-modal directional consistency calibration, and constructs a dedicated benchmark dataset.
tags:
  - "CVPR2025"
  - "Remote Sensing"
  - "UAV"
  - "Change Captioning"
  - "View Change"
  - "Transformer"
  - "Cross-Modal Alignment"
date: 2026-05-08
content_hash: 223c5c03b2de9103
---

# Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning

**Conference**: CVPR2025  
**arXiv**: [2603.12832](https://arxiv.org/abs/2603.12832)  
**Code**: None  
**Area**: Remote Sensing  
**Keywords**: UAV, Change Captioning, View Change, Transformer, Cross-Modal Alignment

## TL;DR

This paper proposes a new task named UAV Scene Change Captioning (UAV-SCC) and a novel HDC-CL framework. It models the overlapping and non-overlapping regions of image pairs under moving viewpoints using a Dynamic Adaptive Layout Transformer, enhances viewpoint shift direction awareness via hierarchical cross-modal directional consistency calibration, and constructs a dedicated benchmark dataset.

## Background & Motivation

### Background

**Background**: Traditional change captioning assumes a fixed viewpoint with pixel-level aligned image pairs, focusing solely on describing semantic changes in the temporal dimension.

### Limitations of Prior Work

**Limitations of Prior Work**: In UAV scenarios, the camera is in motion, introducing spatial layout inconsistency in image pairs due to **viewpoint shift**, with only partial scene overlap.

### Key Challenge

**Key Challenge**: Two major challenges: (1) effectively modeling the relationship between overlapping and non-overlapping regions to handle parallax effects; (2) capturing direction clues brought by viewpoint motion to correctly interpret scene changes.

### Core Idea

**Core Idea**: Existing methods primarily address changes in aligned scenes and cannot cope with the partial overlap and spatial layout inconsistencies caused by dynamic UAV viewpoints.

## Method

### Three Stages of the HDC-CL Framework

**1. Image Alignment**

- **Shift Voting Mechanism**: Estimates the overlap mask of the image pair.
    - Computes the pairwise feature similarity between patches of two images to find the best match and relative displacement $\Delta$ for each patch.
    - Votes to count the frequency of each $\Delta$ and selects the $\Delta^*$ with the highest cumulative similarity as the dominant displacement.
    - Generates a binary common mask based on this to distinguish overlapping and non-overlapping regions.

- **Dynamic Adaptive Layout Transformer (DALT)**:
    - Decomposes the features of each image into three types of regions: global (glo), common (com), and difference (diff).
    - Assigns a learnable [CLS] token to each region category.
    - Jointly models different regions in a unified multi-head self-attention encoder to obtain region-aware features.

**2. Scene Change Distillation**

- **Context Feature Decoupling**: Uses independent encoders (GE, CE, DE) to extract [CLS]-level semantics for global, common, and difference regions.
- **Hierarchical Consistency Constraints**:
    - Global consistency (InfoNCE): Aligns the background semantics of the image pair.
    - Region consistency (InfoNCE): Aligns the invariant semantics in overlapping regions.
    - Independence regularization (HSIC): Minimizes statistical dependence between pre- and post-difference features to encourage capturing diverse change information.
- **Scene Change Distillation**: Cross-attention models cross-image correspondence in common regions, and a residual mechanism extracts local differences, which are fused with global differences to obtain a unified change representation D.

**3. Caption Generation**

- A Transformer decoder generates directional descriptions based on the change representation D.
- **HCM-OCC (Hierarchical Cross-modal Directional Consistency Calibration)**:
    - Computes the visual direction vector $\Delta d = D_{forward} - D_{reverse}$
    - Computes the textual direction vector $\Delta t = T_{forward} - T_{reverse}$
    - Bidirectional margin ranking loss aligns visual and textual directional semantics.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{cap} + \lambda(\mathcal{L}_{con} + \mathcal{L}_{align})$

## Key Experimental Results

### UAV-SCC Dataset
- UAV-SCCSimple: 9,017 image pairs, average caption length ~27 words, 3 captions/pair
- UAV-SCCRich: 7,054 image pairs, average caption length ~14 words, 5 captions/pair

### Main Results (Comparison with 6 Baselines)

| Method | UAV-SCCSimple (B/M/R/C/S) | UAV-SCCRich (B/M/R/C/S) |
|------|---------------------------|-------------------------|
| CARD | 27.49/26.23/42.98/48.66/30.76 | 18.66/16.46/45.03/15.75/11.87 |
| **HDC-CL** | **31.13/27.34/44.58/54.68/33.09** | **19.26/18.45/44.32/19.16/13.00** |

- CIDEr score gains 6.02 on Simple and 3.41 on Rich (compared to the strongest baseline CARD).
- BLEU-4 gains 3.64 on Simple.

### Ablation Study
- The joint utilization of the three losses (global, region, and HSIC) achieves the best performance.
- The effectiveness of the shift voting mechanism in DALT is validated through ablation.
- The HCM-OCC directional consistency calibration brings stable improvements.
- Using HSIC regularization alone gains 4.38 CIDEr (13.56 $\rightarrow$ 17.94) on Rich.
- Combining the three losses achieves a peak CIDEr of 19.16 on Rich, validating the complementarity of hierarchical constraints.

## Highlights & Insights

1. **Practical value of the new task definition**: UAV-SCC fills the gap in change captioning under moving viewpoints, closer to real UAV applications than fixed-viewpoint change captioning.
2. **Elegant Shift Voting mechanism**: Adaptively estimates overlapping regions without extra annotations to handle parallax.
3. **Unique direction-aware design**: HCM-OCC aligns forward/reverse directional gaps with textual direction, enabling the model to perceive viewpoint shift directions.
4. **Contribution of comprehensive new benchmarks**: Two dataset versions (Simple/Rich) are built to support evaluation at different granularities.

## Limitations & Future Work

1. The dataset scale is relatively limited (~9K pairs), which may be insufficient to train large-scale models.
2. Features are extracted using only ResNet-101, with no experiments conducted using stronger visual backbones or pre-trained VLMs.
3. Shift Voting assumes a single global shift, offering limited adaptability to complex rotation or scale transformations.
4. Captioning evaluation metrics (BLEU, METEOR, CIDEr) may not fully reflect the accuracy of directional descriptions.
5. Lack of comparison with large multimodal models (e.g., GPT-4V).
6. The dataset is constructed based on images from existing public datasets, which may limit scene diversity.
7. The forward and reverse directions of descriptions only consider two reciprocal directions, without modeling more complex rotational directional relationships.

## Rating

- Novelty: ⭐⭐⭐⭐ (new task + full methodology + new dataset)
- Experimental Thoroughness: ⭐⭐⭐⭐ (comprehensive comparisons, detailed ablations)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, intuitive figures)
- Value: ⭐⭐⭐⭐ (new direction for UAV scene understanding, significant contribution to benchmark)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[ICCV 2025\] Information-Bottleneck Driven Binary Neural Network for Change Detection](../../ICCV2025/remote_sensing/information-bottleneck_driven_binary_neural_network_for_change_detection.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[CVPR 2026\] UniChange: Unifying Change Detection with Multimodal Large Language Model](../../CVPR2026/remote_sensing/unichange_unifying_change_detection_with_multimodal_large_language_model.md)
- [\[CVPR 2026\] Sparsely Timing the Change: A Spiking Temporal Framework for Remote Sensing Interpretation](../../CVPR2026/remote_sensing/sparsely_timing_the_change_a_spiking_temporal_framework_for_remote_sensing_inter.md)

</div>

<!-- RELATED:END -->
