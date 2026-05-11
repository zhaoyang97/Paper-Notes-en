---
title: >-
  [Paper Note] AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios
description: >-
  [AAAI 2026][Object Detection][RMOT] This paper introduces AerialMind, the first large-scale Referring Multi-Object Tracking (RMOT) benchmark dataset for UAV scenarios, and proposes HawkEyeTrack (HETrack)…
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "RMOT"
  - "UAV"
  - "vision-language fusion"
  - "multi-object tracking"
  - "benchmark dataset"
date: 2026-05-08
content_hash: a4e9f403cc425f24
---

# AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios

**Conference**: AAAI 2026
**arXiv**: [2511.21053v2](https://arxiv.org/abs/2511.21053v2)
**Code**: [Available (dataset)](https://github.com/shawnliang420/AerialMind)
**Area**: Object Detection / Multi-Object Tracking / Vision-Language
**Keywords**: RMOT, UAV, vision-language fusion, multi-object tracking, benchmark dataset

## TL;DR
This paper introduces AerialMind, the first large-scale Referring Multi-Object Tracking (RMOT) benchmark dataset for UAV scenarios, and proposes HawkEyeTrack (HETrack), a method that achieves language-guided multi-object tracking in aerial UAV scenes via a co-evolutionary fusion encoder and a scale-adaptive contextual refinement module.

## Background & Motivation
The RMOT task enables detection and tracking of specific targets in video through natural language instructions, and constitutes a fundamental capability for intelligent robotic systems. However, existing RMOT research is almost entirely confined to ground-level perspectives (e.g., Refer-KITTI, Refer-BDD), failing to address the wide-area surveillance demands of UAV aerial imagery. UAVs are increasingly important in large-scale surveillance and embodied intelligence due to their bird's-eye view and high maneuverability, yet aerial scenes introduce unique challenges including drastic appearance variation, complex spatial relationships, dynamic scene changes, and diverse semantic expressions. Existing RMOT datasets and methods cannot be directly adapted to these challenges.

## Core Problem
1. **Dataset absence**: The lack of a large-scale RMOT benchmark for UAV scenarios impedes research on aerial vision-language perception.
2. **High annotation cost**: RMOT requires simultaneous annotation of temporal trajectories and natural language descriptions, making conventional manual annotation time-consuming.
3. **Inefficient vision-language fusion**: Existing early-fusion/late-fusion paradigms suffer from modality gaps or "language signal dilution."
4. **Difficulty in small-object perception**: In aerial scenes, the effective receptive field on high-resolution feature maps is limited, and small-scale targets are easily overwhelmed by background noise.

## Method

### Overall Architecture
HETrack is built on the Deformable DETR architecture, using ResNet50 as the visual backbone and RoBERTa as the language encoder. The key innovations are two modules inserted between the encoder and decoder:
1. **Co-evolutionary Fusion Encoder (CFE)**: Enables bidirectional co-evolutionary interaction between vision and language during encoding.
2. **Scale Adaptive Contextual Refinement (SACR)**: Enhances small-object perception between the encoder output and the decoder.
3. The decoder employs a Semantic Guidance Module for semantically guided query augmentation.

### Key Designs
**CFE (Co-evolutionary Fusion Encoder)**:
- Core Idea: The structured processing of visual features and the guidance process of linguistic information should not operate independently but should be deeply interleaved and mutually reinforcing.
- Stacks $N_e$ blocks, each containing:
    - **Bidirectional Fusion Layer (BFL)**: Implements bidirectional information flow (visual→language and language→visual) via multi-head attention. Visual features provide concrete anchors for linguistic concepts, while linguistic concepts guide the filtering and enhancement of visual features.
    - **Deformable Encoding Layer (DEL)**: Performs efficient spatial relationship modeling on the fused features.
- The final encoder output is modulated by the sentence-level global feature $\mathbf{T}_s$, granting the model holistic control over the overall referring intent.

**SACR (Scale Adaptive Contextual Refinement)**:
- Captures multi-scale context on the highest-resolution feature map using parallel dilated convolutions (dilation rates = {6, 12, 18}) without sacrificing spatial resolution.
- Adaptive channel recalibration: GAP → 1D convolution (kernel size adaptively determined by channel dimension: $k = |\log_2(C) + b / \gamma|_{\text{odd}}$) → Sigmoid → channel weighting, suppressing background noise and emphasizing key channels for small targets.

**Semantic Guidance Module**: Detection queries perform cross-attention with word-level features and are concatenated with tracking queries before being fed into the decoder.

**COALA Annotation Framework** (innovation in dataset construction):
- Four-stage agent collaboration: Scene Understanding Prompt generation (SUP-Agent) → Semi-automatic Object Labeling (SOL-Agent, requiring only two clicks from annotators to define temporal boundaries) → Consistency Checking (CC-Agent, cross-modal spatiotemporal logical reasoning verification) → Expression Expansion (EE-Agent, semantically equivalent diversified paraphrasing).

### Loss & Training
- Total loss = $\lambda_{cls}\mathcal{L}_{cls} + \lambda_{L1}\mathcal{L}_{L1} + \lambda_{giou}\mathcal{L}_{giou} + \lambda_{ref}\mathcal{L}_{ref}$
- $\mathcal{L}_{cls}$: focal loss; $\mathcal{L}_{L1}$: L1 regression loss; $\mathcal{L}_{giou}$: GIoU loss.
- Loss weights: $\lambda_{cls}=2, \lambda_{L1}=5, \lambda_{giou}=2, \lambda_{ref}=2$
- AdamW optimizer, initial learning rate $1\times10^{-4}$, decayed by $10\times$ at epoch 40, trained for 100 epochs.
- 8× A100 GPUs, batch size 1, 300 object queries.
- Inference score threshold 0.5, referring matching threshold $\beta_{ref}=0.4$.
- 51.4M trainable parameters; 15.6 FPS on a single RTX 4080.

## Key Experimental Results

**AerialMind dataset scale**: 93 video sequences, 24.6K expressions, 293.1K instances, 46.14M bounding box annotations — far exceeding Refer-KITTI-V2 (9.8K expressions).

**In-domain (VisDrone test set)**:

| Method | HOTA | DetA | AssA | HOTA_S | HOTA_M |
|--------|------|------|------|--------|--------|
| TransRMOT | 23.54 | 13.18 | 42.24 | 27.21 | 24.05 |
| TempRMOT | 26.24 | 13.06 | 53.22 | 28.14 | 23.77 |
| MGLT | 26.16 | 14.83 | 46.47 | 26.39 | 26.10 |
| **HETrack** | **31.46** | **21.57** | **46.23** | **34.37** | **31.12** |

**Cross-domain (UAVDT test set)**: HETrack achieves HOTA 31.60, DetA 21.35, LocA 83.98 — best across all metrics.

**Refer-KITTI-V2 (ground-level scene)**: HOTA 35.40, comparable to HFF-Track (36.18), validating the generalizability of the method.

### Ablation Study
- Removing both CFE and SACR: HOTA drops from 31.46 to 26.41 (−5.05), demonstrating that both modules contribute substantially.
- Removing CFE only: HOTA drops to 28.27 (−3.19); CFE contributes more, confirming that vision-language co-evolutionary fusion is central.
- Removing SACR only: HOTA drops to 29.89 (−1.57); SACR is effective for small-object detection but contributes relatively less.
- Fusion strategy comparison: CFE's bidirectional fusion outperforms Concat (28.88), Add (30.39), and Cross-Attn (30.52).
- SACR internal ablation: dilated convolution only → 29.70; channel recalibration only → 29.13; their combination achieves 31.46.
- Referring threshold $\beta_{ref}=0.4$ is optimal; both higher and lower values degrade performance.
- Attribute-level analysis: HETrack shows clear advantages under Low Resolution (38.49%), Fast Motion (35.41%), and Night (35.4%) conditions.

## Highlights & Insights
1. **First UAV RMOT benchmark**: Fills the data gap for language-guided tracking in aerial scenes; dataset scale far exceeds all existing RMOT datasets.
2. **COALA annotation framework innovation**: Four-stage agent collaboration reduces manual annotation to a "two-click + review" workflow, significantly lowering annotation cost.
3. **First attribute-level evaluation**: Frame-level annotation of 8 challenge attributes (night, occlusion, low resolution, viewpoint change, scale variation, fast motion, rotation, low resolution), with the introduction of HOTA_S and HOTA_M metrics.
4. **"Co-evolutionary" idea in CFE**: Rather than simple early or late fusion, visual structuring and language guidance are made to iteratively co-evolve.
5. **Interesting cross-domain generalization finding**: Cross-domain HOTA is unexpectedly higher; the authors attribute this to UAVDT containing only vehicle categories, making the semantic space simpler.

## Limitations & Future Work
1. **LLM reasoning not exploited**: The current architecture is based on conventional VL fusion paradigms and does not incorporate advanced reasoning from large language models.
2. **Insufficient deployment efficiency**: With 51.4M parameters and 15.6 FPS, real-time deployment on resource-constrained UAV platforms remains challenging.
3. **Dataset dependency on existing annotations**: Built upon VisDrone/UAVDT extensions, inheriting a small number of annotation errors from the original datasets.
4. **Trade-off between detection and localization accuracy**: HETrack improves DetA while LocA slightly decreases (82.77 vs. 83+ for other methods).
5. **Limited object categories**: The training set covers 10 object categories, and the cross-domain test set contains only vehicles, leaving validation on richer categories lacking.

## Related Work & Insights
- **vs. TransRMOT/TempRMOT**: These are pioneering RMOT works but are limited to ground-level scenes; HETrack achieves approximately 5–8 HOTA points improvement on AerialMind.
- **vs. iKUN**: iKUN performs RMOT without retraining but achieves very low performance on Refer-KITTI-V2 (10.32 HOTA).
- **vs. HFF-Track**: An AAAI 2025 work; HFF-Track (36.18) slightly outperforms HETrack (35.40) on Refer-KITTI-V2, but HETrack shows stronger detection recall (41.16 vs. 36.86).
- **Dataset comparison**: AerialMind's expression count (24.6K), instance count (293.1K), and bounding box annotations (46.14M) far exceed all existing RMOT datasets.

The agent collaboration paradigm of the COALA annotation framework is transferable to annotation efficiency improvements in other video understanding tasks. The co-evolutionary fusion paradigm of CFE also serves as a reference for any task requiring cross-modal alignment, such as referring segmentation and VQA.

## Rating
- Novelty: ⭐⭐⭐⭐ First UAV RMOT benchmark; method design is sound but not paradigm-breaking.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three-dimensional evaluation (in-domain / cross-domain / ground-level) + attribute-level analysis + comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; motivation for both the dataset and method is well articulated.
- Value: ⭐⭐⭐ The dataset annotation framework and cross-modal fusion approach offer meaningful reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking](when_trackers_date_fish_a_benchmark_and_framework_for_underwater_multiple_fish_t.md)
- [\[CVPR 2026\] UAVGen: Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](../../CVPR2026/object_detection/uavgen_visual_prototype_conditioned_focal_region_generation_for_uav_based_object_detection.md)
- [\[AAAI 2026\] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)
- [\[CVPR 2026\] HeROD: Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection](../../CVPR2026/object_detection/herod_heuristic_inspired_reasoning_data_efficient_rod.md)
- [\[AAAI 2026\] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)

</div>

<!-- RELATED:END -->
