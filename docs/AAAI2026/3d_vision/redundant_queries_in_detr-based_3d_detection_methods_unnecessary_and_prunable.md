---
title: >-
  [Paper Note] Redundant Queries in DETR-Based 3D Detection: Unnecessary and Prunable
description: >-
  [AAAI 2026][3D Vision][3D Object Detection] Proposes GPQ (Gradually Pruning Queries) to progressively prune a large number of redundant object queries in DETR-based 3D detectors based on classification scores. Removing queries requires no extra learnable parameters and can be directly accomplished by fine-tuning on pre-trained checkpoints, achieving up to a 67.86% FLOPs reduction and a 65.16% inference time reduction on edge devices.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Object Detection"
  - "DETR"
  - "Query Pruning"
  - "Model Compression"
  - "Autonomous Driving"
date: 2026-05-08
content_hash: c6ee84466c2c3b7a
---

# Redundant Queries in DETR-Based 3D Detection: Unnecessary and Prunable

**Conference**: AAAI 2026  
**arXiv**: [2412.02054](https://arxiv.org/abs/2412.02054)  
**Code**: To be confirmed  
**Area**: 3D Vision / 3D Object Detection  
**Keywords**: 3D Object Detection, DETR, Query Pruning, Model Compression, Autonomous Driving

## TL;DR

Proposes GPQ (Gradually Pruning Queries) to progressively prune a large number of redundant object queries in DETR-based 3D detectors based on classification scores. Removing queries requires no extra learnable parameters and can be directly accomplished by fine-tuning on pre-trained checkpoints, achieving up to a 67.86% FLOPs reduction and a 65.16% inference time reduction on edge devices.

## Background & Motivation

### Background

DETR-based methods are widely used in 3D object detection. Their core mechanism generates detection results by interacting predefined object queries with image features in transformer layers. However, these methods typically require setting a number of queries (e.g., 900) that far exceeds the actual number of target objects, such as in nuScenes where target objects usually do not exceed 100. This results in a negative-to-positive sample ratio of up to 8:1, meaning a massive amount of queries are repeatedly matched as negative samples during Hungarian matching, consistently suppressing their classification scores.

### Key Observation

The authors analyzed the frequency at which each query is selected as the final prediction during inference in methods like PETR, PETRv2, FocalPETR, and StreamPETR, revealing an extremely imbalanced distribution: a small subset of queries handles the vast majority of detection tasks, whereas many queries are almost never selected, and some queries in PETR are never selected at all.

### Limitations of Prior Work

Traditional transformer pruning methods (such as attention head pruning and token pruning) are difficult to apply directly to 3D detection:

- **Non-existent pruning targets**: Attention heads in 3D detection are implemented via reshaping, and modifying their count does not reduce the computational cost.
- **Structural inconsistency**: In 3D detection, the query and key dimensions are unequal ($N_q \neq N_k$), making the attention matrix non-square.
- **Token difference**: 3D detection generates significantly more tokens than ViTs (at least 4000 vs. less than 200), making the overhead of token pruning prohibitively high.

## Method

### Mechanism

Each query is treated as the minimal pruning unit, with the classification score as the pruning metric. Queries with the lowest classification scores contribute the least and are prioritized for removal.

### GPQ Algorithm Pipeline

1. **Load pretrained checkpoint**: Start from a trained model containing a large number of queries.
2. **Normal forward propagation**: Obtain classification scores for each query after each iteration.
3. **Periodic pruning**: Trigger pruning every $n$ iterations to select and permanently remove queries with the lowest classification scores.
4. **Repeat until target number**: Gradually reduce the query count from the initial $N_q$ to $N_q'$.

The entire process introduces no extra learnable parameters nor requires learnable binary masks, and can be completed within a few epochs.

### Theoretical Analysis: Why Pruning Works

The independence between queries is crucial. In MLPs and cross-attention, the query matrix $Q$ appears only once. Due to the row independence of matrix multiplication ($AB \equiv \text{Concat}_{i}(A_i B)$), deleting a row does not affect the results of other rows. The only impact comes from self-attention, where $Q$ acts as query, key, and value simultaneously. However, the authors demonstrate that the indirect sampling effect of self-attention on image features is much smaller than the direct interaction of cross-attention, and thus removing low-contribution queries introduces minimal disturbance.

### Why Not Train Directly with Fewer Queries

The authors visualized the distribution of reference points: queries pruned from 900 to 300 still maintain a clustered and organized distribution (inheriting the knowledge of large-scale training), whereas directly training with 300 queries results in a scattered distribution and weaker representation ability. GPQ also allows for flexibly generating model variants with different query counts from a single checkpoint.

## Key Experimental Results

### Experimental Setup

- **Dataset**: nuScenes (23,000+ samples, 6 multi-view cameras, 10 categories)
- **Detectors**: DETR3D, PETR, PETRv2, FocalPETR, StreamPETR, RayDN
- **Evaluation Metrics**: mAP, NDS, various error metrics (mATE/mASE/mAOE/mAVE/mAAE), FPS, GFLOPs

### Main Results (Table 2)

| Model | Backbone | Queries | mAP | NDS | FPS |
|------|----------|---------|-----|-----|-----|
| PETR | ResNet50 | 900/- | 31.74% | 0.3668 | 6.9 |
| PETR | ResNet50 | 300/- (Scratch) | 31.19% | 0.3536 | 8.9 |
| PETR | ResNet50 | 900→300 (GPQ) | **32.85%** | **0.3884** | 8.9 |
| PETR | ResNet50 | 900→150 (GPQ) | 30.52% | 0.3671 | 9.3 |
| StreamPETR | ResNet50 | 900/- | 37.83% | 0.4734 | 16.1 |
| StreamPETR | ResNet50 | 300/- (Scratch) | 33.62% | 0.4429 | 18.5 |
| StreamPETR | ResNet50 | 900→300 (GPQ) | **39.42%** | **0.4941** | 18.7 |
| FocalPETR | ResNet50 | 900/- | 32.44% | 0.3752 | 16.4 |
| FocalPETR | ResNet50 | 900→300 (GPQ) | **33.17%** | **0.3925** | 19.6 |

Key findings: After pruning 900 queries to 300 via GPQ, model performances in PETR, FocalPETR, and StreamPETR even **surpass** the baseline trained from scratch with 900 queries. PETR achieves an acceleration of up to 1.35x.

### Edge Device Deployment Results (Table 3 - Jetson Nano B01)

| Model | Backbone | Queries | GFLOPs | Time (ms) | FLOPs Reduction | Time Reduction |
|------|----------|---------|--------|----------|-----------|---------|
| StreamPETR | ResNet18 | 900 | 172.08 | 1520 | - | - |
| StreamPETR | ResNet18 | 900→300 | 123.90 | 916 | 28.00% | 39.74% |
| StreamPETR | ResNet18 | 900→150 | 112.51 | 791 | 34.62% | 47.96% |
| StreamPETR | w/o backbone | 900 | 87.78 | 1030 | - | - |
| StreamPETR | w/o backbone | 900→150 | 28.21 | 359 | **67.86%** | **65.16%** |

After removing the backbone, the speedup of the pure transformer part is even more significant, demonstrating that GPQ operates precisely on the computational bottleneck.

## Ablation Study

- **Pruning Metrics** (Table 5): Pruning by highest classification score (GPQ-H) leads to a significant performance drop (34.34% mAP); pruning by matching cost (GPQ-C) achieves 38.78%, while the original GPQ pruning by lowest classification score is optimal (39.42%).
- **Progressive vs. One-step Pruning**: Pruning 600 queries at once (GPQ-1) yields an mAP of only 35.71%, far below the progressive strategy's 39.42%, verifying the necessity of gradual pruning.
- **Comparison with Other Methods** (Table 4): ToMe (token merging) actually slows down in 3D detection (due to the excessive overhead of similarity matrix computation), and GBC achieves speedup but drops in detection accuracy; GPQ balances both speed and accuracy.
- **Fully Converged Model** (Table 6): Applying GPQ to StreamPETR trained for 90 epochs shows that the 300-query variant still outperforms the 300-query model trained from scratch for 90 epochs.
- **Training-Synchronized Pruning** (Table 7): GPQ can be executed synchronously during the training process, bypassing the need for a full training phase before pruning.

## Highlights & Insights

- **Simple yet Effective**: It introduces zero learnable parameters, relying solely on classification score ranking and progressive removal to achieve query pruning with no performance loss, or even minor gains.
- **Plug-and-Play**: Acting as a fine-tuning step, it can be directly applied to pre-trained checkpoints of any DETR-based detector. Multiple lightweight variants can be flexibly exported from a single checkpoint.
- **First to Focus on Query Redundancy**: This work systematically analyzes the imbalanced query selection frequency in 3D detection, filling a gap in this research direction.
- **Edge Deployment Friendly**: Significant practical latency reduction is validated on Jetson Nano.

## Limitations & Future Work

- The method is only validated on the nuScenes dataset, without covering other 3D detection benchmarks like Waymo or KITTI.
- The approach relies on classification scores as the pruning metric, which might drop in performance for scenarios where classification scores are uniformly distributed.
- It does not consider the spatial distribution of queries—only pruning based on scores might lead to insufficient coverage in certain spatial regions.
- Edge device experiments use random dummy inputs instead of real data; actual inference acceleration may be affected by I/O and other factors.
- Only preliminary validation is performed on 2D detection (ConditionalDETR), and its generalizability awaits more comprehensive verification.

## Related Work & Insights

- **DETR-based 3D Detectors**: PETR, PETRv2, StreamPETR, FocalPETR, Far3D, DETR3D, etc., all of which interact predefined queries with image features.
- **Transformer Pruning Methods**: Attention head pruning (Michel et al.), layer random dropping (Fan et al.), ViT sparsity exploration (Chen et al.), joint width and depth pruning (ZipLM), token pruning (EViT), etc.
- **Token Merging/Pruning**: ToMe (ICLR 2023) merges similar tokens but incurs excessive overhead in 3D detection due to the massive number of tokens.
- **GBC** (ICCV 2025): Provides speedup but leads to a drop in detection accuracy.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

Overall Rating: ⭐⭐⭐⭐ — The method is extremely simple yet directly targets realistic bottlenecks. The experiments cover a variety of detectors and deployment scenarios, providing direct reference value for industrial deployment of DETR-based detectors. The novelty mainly lies in the observation of "identifying and systematically verifying query redundancy", while the technique itself is relatively straightforward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SEED: A Simple and Effective 3D DETR in Point Clouds](../../ECCV2024/3d_vision/seed_a_simple_and_effective_3d_detr_in_point_clouds.md)
- [\[AAAI 2026\] MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection](monoclue_object-aware_clustering_enhances_monocular_3d_object_detection.md)
- [\[AAAI 2026\] Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection](distilling_future_temporal_knowledge_with_masked_feature_reconstruction_for_3d_o.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[AAAI 2026\] Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark](griffin_aerial-ground_cooperative_detection_and_tracking_dataset_and_benchmark.md)

</div>

<!-- RELATED:END -->
