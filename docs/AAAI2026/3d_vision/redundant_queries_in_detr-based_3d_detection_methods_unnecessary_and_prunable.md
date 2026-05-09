---
title: >-
  [Paper Note] Redundant Queries in DETR-Based 3D Detection: Unnecessary and Prunable
description: >-
  [AAAI 2026][3D Vision][3D Object Detection] This paper proposes GPQ (Gradually Pruning Queries), a method that progressively prunes redundant object queries in DETR-based 3D detectors using classification scores. Without introducing any additional learnable parameters, GPQ can be applied as a fine-tuning step directly on pretrained checkpoints, achieving up to 67.86% FLOPs reduction and 65.16% inference time reduction on edge devices.
tags:
  - AAAI 2026
  - 3D Vision
  - 3D Object Detection
  - DETR
  - Query Pruning
  - Model Compression
  - Autonomous Driving
date: 2026-05-08
content_hash: 6487c121b7750821
---

# Redundant Queries in DETR-Based 3D Detection: Unnecessary and Prunable

**Conference**: AAAI 2026
**arXiv**: [2412.02054](https://arxiv.org/abs/2412.02054)
**Code**: To be confirmed
**Area**: 3D Vision / 3D Object Detection
**Keywords**: 3D Object Detection, DETR, Query Pruning, Model Compression, Autonomous Driving

## TL;DR

This paper proposes GPQ (Gradually Pruning Queries), a method that progressively prunes redundant object queries in DETR-based 3D detectors using classification scores. Without introducing any additional learnable parameters, GPQ can be applied as a fine-tuning step directly on pretrained checkpoints, achieving up to 67.86% FLOPs reduction and 65.16% inference time reduction on edge devices.

## Background & Motivation

### State of the Field

DETR-based methods are widely adopted in 3D object detection, relying on predefined object queries that interact with image features through transformer layers to produce detection results. However, these methods typically require far more queries than the actual number of targets (e.g., 900 queries), whereas scenes such as nuScenes rarely contain more than 100 objects. This results in a positive-to-negative sample ratio as high as 8:1, where a large number of queries are repeatedly assigned as negatives during Hungarian matching, causing their classification scores to be continuously suppressed.

### Core Observation

The authors analyze the selection frequency of each query at inference time across methods including PETR, PETRv2, FocalPETR, and StreamPETR. The distribution is found to be highly skewed: a small number of queries account for the vast majority of detections, while many queries are almost never selected as final predictions — with some queries in PETR never being selected at all.

### Limitations of Prior Work

Conventional transformer pruning methods (e.g., attention head pruning, token pruning) cannot be directly applied to 3D object detection:

- **No prunable targets**: Attention heads in 3D detection are implemented via reshaping, so changing their count does not affect computation.
- **Structural mismatch**: Query and key dimensions are unequal in 3D detection ($N_q \neq N_k$), making the attention matrix non-square.
- **Scale discrepancy**: The number of tokens in 3D detection (at least 4,000) is far larger than in ViT (fewer than 200), making token pruning prohibitively expensive.

## Method

### Mechanism

Each query is treated as the minimal pruning unit, with its classification score used as the pruning criterion. Queries with the lowest classification scores contribute the least and are removed first.

### GPQ Algorithm

1. **Load pretrained checkpoint**: Start from a trained model with a large number of queries.
2. **Standard forward pass**: Obtain per-query classification scores after each iteration.
3. **Periodic pruning**: Every $n$ iterations, identify queries with the lowest classification scores and permanently remove them.
4. **Repeat until target count**: Gradually reduce the query count from $N_q$ to $N_q'$.

The entire process introduces no additional learnable parameters and requires no learnable binary masks, completing within a few epochs.

### Theoretical Analysis: Why Pruning Works

The key property is the independence among queries. In MLP and cross-attention layers, the query matrix $Q$ appears only once; by the row-independence property of matrix multiplication ($AB \equiv \text{Concat}_{i}(A_i B)$), removing a row does not affect the results of other rows. The only coupling arises in self-attention, where $Q$ serves simultaneously as query, key, and value. The authors argue that self-attention's indirect sampling of image features has far less impact than the direct interaction in cross-attention, so removing low-contribution queries introduces minimal interference.

### Why Not Train with Fewer Queries Directly

The authors visualize reference point distributions: queries pruned from 900 to 300 retain a compact, well-organized distribution (inheriting knowledge from large-scale training), whereas training from scratch with 300 queries yields a scattered distribution with weaker representational capacity. GPQ also enables flexible generation of multiple lightweight model variants from a single checkpoint.

## Key Experimental Results

### Experimental Setup

- **Dataset**: nuScenes (23,000+ samples, 6 surround-view cameras, 10 categories)
- **Detectors**: DETR3D, PETR, PETRv2, FocalPETR, StreamPETR, RayDN
- **Metrics**: mAP, NDS, error metrics (mATE/mASE/mAOE/mAVE/mAAE), FPS, GFLOPs

### Main Results (Table 2)

| Model | Backbone | Queries | mAP | NDS | FPS |
|-------|----------|---------|-----|-----|-----|
| PETR | ResNet50 | 900/- | 31.74% | 0.3668 | 6.9 |
| PETR | ResNet50 | 300/- (from scratch) | 31.19% | 0.3536 | 8.9 |
| PETR | ResNet50 | 900→300 (GPQ) | **32.85%** | **0.3884** | 8.9 |
| PETR | ResNet50 | 900→150 (GPQ) | 30.52% | 0.3671 | 9.3 |
| StreamPETR | ResNet50 | 900/- | 37.83% | 0.4734 | 16.1 |
| StreamPETR | ResNet50 | 300/- (from scratch) | 33.62% | 0.4429 | 18.5 |
| StreamPETR | ResNet50 | 900→300 (GPQ) | **39.42%** | **0.4941** | 18.7 |
| FocalPETR | ResNet50 | 900/- | 32.44% | 0.3752 | 16.4 |
| FocalPETR | ResNet50 | 900→300 (GPQ) | **33.17%** | **0.3925** | 19.6 |

Notably, applying GPQ to prune from 900 to 300 queries in PETR, FocalPETR, and StreamPETR yields mAP that **surpasses** the baseline trained from scratch with 900 queries. PETR achieves a 1.35× speedup.

### Edge Deployment Results (Table 3 — Jetson Nano B01)

| Model | Backbone | Queries | GFLOPs | Latency (ms) | FLOPs Reduction | Latency Reduction |
|-------|----------|---------|--------|--------------|-----------------|-------------------|
| StreamPETR | ResNet18 | 900 | 172.08 | 1520 | — | — |
| StreamPETR | ResNet18 | 900→300 | 123.90 | 916 | 28.00% | 39.74% |
| StreamPETR | ResNet18 | 900→150 | 112.51 | 791 | 34.62% | 47.96% |
| StreamPETR | w/o backbone | 900 | 87.78 | 1030 | — | — |
| StreamPETR | w/o backbone | 900→150 | 28.21 | 359 | **67.86%** | **65.16%** |

Acceleration is most pronounced in the pure transformer component (without backbone), confirming that GPQ precisely targets the computational bottleneck.

## Ablation Study

- **Pruning criterion** (Table 5): Pruning by highest classification score (GPQ-H) causes significant performance degradation (mAP 34.34%); pruning by matching cost (GPQ-C) achieves 38.78%; GPQ's lowest-score criterion performs best (39.42%).
- **Gradual vs. one-shot pruning**: Removing 600 queries at once (GPQ-1) yields only 35.71% mAP, far below the gradual strategy (39.42%), validating the necessity of progressive pruning.
- **Comparison with other methods** (Table 4): ToMe (token merging) actually slows down 3D detection due to the large overhead of computing the similarity matrix; GBC provides speedup but at the cost of detection accuracy; GPQ achieves favorable trade-offs on both dimensions.
- **Fully converged models** (Table 6): Applying GPQ to a StreamPETR model trained for 90 epochs still outperforms a model trained from scratch with 300 queries for 90 epochs.
- **Pruning during training** (Table 7): GPQ can be applied concurrently with training, without requiring a fully trained model as a prerequisite.

## Highlights & Insights

- **Minimalist yet effective**: Without introducing any learnable parameters, GPQ achieves lossless or even improved query pruning through simple score-based ranking and progressive removal.
- **Plug-and-play**: Applicable as a fine-tuning step to pretrained checkpoints of any DETR-based detector; a single checkpoint can export multiple lightweight model variants.
- **First systematic study of query redundancy**: The paper characterizes the highly imbalanced query selection frequency in 3D detection and addresses a previously overlooked problem.
- **Edge-deployment friendly**: Significant practical speedups are demonstrated on the Jetson Nano platform.

## Limitations & Future Work

- Validation is limited to the nuScenes dataset; other 3D detection benchmarks such as Waymo and KITTI are not evaluated.
- The method relies on classification scores as the pruning criterion, which may be less effective when score distributions are more uniform.
- Spatial coverage of queries is not considered; score-only pruning may leave certain spatial regions underrepresented.
- Edge device experiments use random dummy inputs rather than real data; actual inference speedups may be affected by I/O and other factors.
- Generalizability to 2D detection (e.g., ConditionalDETR) is only preliminarily verified and warrants broader validation.

## Related Work & Insights

- **DETR-based 3D detectors**: PETR, PETRv2, StreamPETR, FocalPETR, Far3D, DETR3D, etc., all use predefined queries to interact with image features.
- **Transformer pruning**: Attention head pruning (Michel et al.), stochastic layer dropping (Fan et al.), sparsity in ViT (Chen et al.), joint width-depth pruning (ZipLM), token pruning (EViT), etc.
- **Token Merging/Pruning**: ToMe (ICLR 2023) merges similar tokens but incurs excessive overhead in 3D detection due to the large token count.
- **GBC** (ICCV 2025): Provides speedup but leads to accuracy degradation.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

Overall: ⭐⭐⭐⭐ — The method is minimalist yet addresses a practical pain point. Experiments cover multiple detectors and deployment scenarios, offering direct reference value for industrial deployment of DETR-based detectors. The primary novelty lies in the observation of query redundancy and its systematic empirical validation, while the technical contribution itself is relatively straightforward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection](distilling_future_temporal_knowledge_with_masked_feature_reconstruction_for_3d_o.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[AAAI 2026\] Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark](griffin_aerial-ground_cooperative_detection_and_tracking_dataset_and_benchmark.md)
- [\[CVPR 2026\] r4det 4d radar camera fusion 3d detection](../../CVPR2026/3d_vision/r4det_4d_radar_camera_fusion_3d_detection.md)
- [\[CVPR 2026\] SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection](../../CVPR2026/3d_vision/span_spatial_projection_alignment_mono3d.md)

</div>

<!-- RELATED:END -->
