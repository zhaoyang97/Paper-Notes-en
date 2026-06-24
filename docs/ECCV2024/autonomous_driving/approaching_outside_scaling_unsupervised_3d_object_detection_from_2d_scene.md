---
title: >-
  [Paper Note] Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene
description: >-
  [ECCV2024][Autonomous Driving][Unsupervised 3D Object Detection] This paper proposes the LiSe method, which incorporates 2D image information into unsupervised 3D object detection. Through adaptive sampling and weak model aggregation strategies in self-paced learning, it significantly improves the detection capability for long-range and small targets.
tags:
  - "ECCV2024"
  - "Autonomous Driving"
  - "Unsupervised 3D Object Detection"
  - "LiDAR-Camera Fusion"
  - "Self-paced Learning"
  - "Pseudo Labels"
date: 2026-05-08
content_hash: 5859c0c5d0833f23
---

# Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene

**Conference**: ECCV2024  
**arXiv**: [2407.08569](https://arxiv.org/abs/2407.08569)  
**Code**: [GitHub](https://github.com/Ruiyang-061X/LiSe)  
**Area**: 3D Vision  
**Keywords**: Unsupervised 3D Object Detection, LiDAR-Camera Fusion, Self-paced Learning, Pseudo Labels

## TL;DR

This paper proposes the LiSe method, which incorporates 2D image information into unsupervised 3D object detection. Through adaptive sampling and weak model aggregation strategies in self-paced learning, it significantly improves the detection capability for long-range and small targets.

## Background & Motivation

Unsupervised 3D object detection aims to discover and localize 3D objects in a scene without using any annotated data, which is crucial for autonomous driving safety. Existing methods (e.g., MODEST, OYSTER) mainly rely on LiDAR point clouds:

- **LiDAR Sparsity**: Point clouds of long-range or small-sized targets are extremely sparse, with very few returned points, making it difficult to distinguish foreground from background.
- **Bias in Self-Training**: During iterative training, the model tends to overfit easy-to-detect samples (e.g., near-range, large-sized), gradually losing the ability to detect difficult samples.
- **Complementary Value of 2D Images**: RGB images feature rich textures, and open-vocabulary 2D detectors exhibit strong recognition capabilities for distant and small objects, which can compensate for the limitations of LiDAR.

The authors observe that LiDAR and 2D images are complementary across different distances and object resolutions, presenting the first attempt to fuse both modalities for unsupervised 3D detection.

## Core Problem

1. How to effectively fuse LiDAR and 2D image information under an unsupervised setting to generate high-quality pseudo labels?
2. The degradation of detection performance on long-tail samples (long-range, small-volume objects) due to model overfitting to common/easy samples during self-training.
3. How to aggregate multiple "weak models" with different preferences trained across different rounds into a comprehensive "strong model"?

## Method

### Overall Architecture

LiSe consists of three core components: LiDAR-camera fusion for pseudo-label generation, adaptive sampling, and weak model aggregation, unified in a self-paced learning pipeline.

### 1. LiDAR and 2D Scene Fusion

**LiDAR Pseudo-Box Generation**:

- Adopts a multi-traversal method: when passing the same location multiple times, points with unchanged positions are considered static background, while displaced points are treated as foreground objects.
- Computes a point persistence score (ppScore) for each point, where a low score indicates dynamic points.
- Builds a graph structure and uses an modified DBSCAN clustering to filter out static clusters, then fits 3D bounding boxes to foreground clusters.

**Image Pseudo-Box Translation**:

- Uses the GroundingDINO open-vocabulary detector to obtain 2D bounding boxes.
- Takes 2D boxes as prompts for SAM (Segment Anything Model) to obtain fine-grained 2D masks.
- Projects LiDAR points onto the 2D plane using camera intrinsic and extrinsic matrices, keeping 3D points that fall within the masks.
- Applies a region-growing algorithm to cluster the retained 3D points and fits 3D bounding boxes.

**Distance-Aware Fusion Strategy**:

$$\mathcal{B}_{final} = \mathcal{B}_{LiDAR} \cup \{b_i \mid d(b_i) \geq d_{min},\, b_i \in \mathcal{B}_{img}\}$$

Merges only the image pseudo-boxes with distances exceeding $d_{min}$ (set to 10m in experiments), since near-range LiDAR is already sufficiently accurate, and image boxes would otherwise introduce modality conflicts.

### 2. Adaptive Sampling

Addressing the issue of model overfitting to easy samples during self-training:

- **Distance-Volume Metric**: Divides objects into four groups based on distance (near 0-30m / far >30m) and volume (small <5m³ / large >5m³).
- Computes the initial distribution $Q_{init}$ before training and the distribution $Q$ after inference.
- Groups with an increased proportion after inference (indicating the model is already proficient) $\rightarrow$ downsampled in the next round.
- Groups with a decreased proportion after inference (indicating performance degradation) $\rightarrow$ upsampled in the next round.

Sampling score formula:

$$R(g_i) = \begin{cases} 1 - (Q(g_i) - Q_{init}(g_i)) & \text{if } Q(g_i) > Q_{init}(g_i) \\ 1 + (Q_{init}(g_i) - Q(g_i)) & \text{if } Q(g_i) \leq Q_{init}(g_i) \end{cases}$$

### 3. Weak Model Aggregation

Models of different rounds excel at different target types due to varying SQL-level sampling rates (e.g., round $t$ excels at large objects, round $t+1$ excels at small objects), and are thus referred to as "weak models". The aggregation method is:

$$\Theta_t = \lambda \cdot \Theta_{t-1} + (1 - \lambda) \cdot \theta_t \quad (T_s \leq t \leq T)$$

Starting from round $T_s$, the weights of the current weak model are weighted-averaged with the historical aggregated model using the aggregation coefficient $\lambda$, progressively building a model with stronger comprehensive capabilities. In the experiments, $T_s=8$ and $\lambda=0.999$.

### Self-Paced Learning Pipeline

1. **Seed Training**: Trains the initial detector using fused pseudo labels.
2. **Self-Training Iterations** (total $T=10$ rounds): Inference using the model from the previous round $\rightarrow$ adjusting pseudo-label distribution via adaptive sampling $\rightarrow$ training a new model $\rightarrow$ weak model aggregation.

## Key Experimental Results

The backbone network is PointRCNN, and the evaluation metrics are AP_BEV and AP_3D (IoU=0.25).

### Main Results on nuScenes

| Method | 0-30m | 30-50m | 50-80m | 0-80m |
|------|-------|--------|--------|-------|
| Supervised Upper Bound | 39.8/34.5 | 12.9/10.0 | 4.4/2.9 | 22.2/18.2 |
| MODEST (T=10) | 24.8/17.1 | 5.5/1.4 | 1.5/0.3 | 11.8/6.6 |
| OYSTER (T=2) | 26.6/19.3 | 4.4/1.8 | 1.7/0.4 | 12.7/8.0 |
| **LiSe (T=10)** | **35.0/24.0** | **11.4/4.4** | **4.8/1.3** | **19.8/11.4** |

- Full range: achieves an improvement of **+7.1% AP_BEV** and **+3.4% AP_3D** compared to OYSTER.
- Long-range (50-80m): AP_BEV of 4.8% **outperforms the supervised model's** 4.4%.

### Main Results on Lyft

| Method | 0-30m | 50-80m | 0-80m |
|------|-------|--------|-------|
| MODEST (T=10) | 73.8/71.3 | 27.0/24.8 | 57.3/55.1 |
| **LiSe (T=10)** | **76.7/74.0** | **46.6/43.7** | **65.6/62.5** |

- Long-range (50-80m) gains **+19.4% AP_BEV** and **+18.9% AP_3D**, showing extremely outstanding performance.

### Ablation Study Key Findings

- Using 2D pseudo-boxes alone is inferior to LiDAR at close range, but fusion yields comprehensive improvements.
- The >10m threshold in distance-aware fusion is optimal, avoiding modality conflicts at close range.
- The distance and volume factors inside adaptive sampling are complementary, yielding the best performance when used jointly.
- Weak model aggregation performs best when starting from a later round ($T_s=8$) with a large $\lambda=0.999$ (slow update schedule).

## Highlights & Insights

1. **Pioneering LiDAR+2D Fusion for Unsupervised 3D Detection**: Utilizes GroundingDINO + SAM to generate image-side pseudo-boxes, effectively supplementing long-range and small targets.
2. The design of **distance-aware fusion** is simple yet effective, resolving modality conflicts by introducing a single threshold.
3. **Adaptive sampling** logo-alleviates long-tail overfitting from the perspective of dynamic data distribution adjustment, presenting a versatile and generalizable strategy.
4. **Weak model aggregation** leverages the complementarity of models in different rounds, yielding a stronger model without extra training overhead.
5. Long-range detection outperforms supervised methods, demonstrating the unique value of 2D information in distant perception.

## Limitations & Future Work

1. **Dependency on Pre-trained 2D Models**: The performance of GroundingDINO and SAM directly impacts the quality of image-side pseudo-labels, potentially requiring domain adaptation in new environments.
2. **High Computational Overhead**: Requires executing the GroundingDINO + SAM + LiDAR pipeline alongside 10 rounds of self-training, leading to high training costs.
3. **Simplistic Fusion Strategy**: Only conducts late fusion at the pseudo-label level, without exploring the potential of feature-level fusion.
4. **Class-Agnostic Detection**: Currently detects only the position and size of objects, without distinguishing specific categories.
5. The distance threshold $d_{min}$ and the volume threshold of 5m³ are manually set, which may not scale or apply to all scenarios.

## Related Work & Insights

| Comparison Dimension | MODEST/OYSTER | Motion-Flow Methods | LiSe |
|---------|-------------|---------|------|
| Input Modalities | LiDAR-only | LiDAR-only | LiDAR + Camera |
| Pseudo-Label Source | Multi-traversal / Clustering | Scene Flow | Multi-traversal + Open-vocabulary detection + SAM |
| Long-range Capability | Weak | Weak | Strong |
| Training Strategy | Naive Self-training | Single-stage Pseudo-labels | Self-paced Learning (Adaptive sampling + Model aggregation) |
| Long-tail Mitigation | None | None | Distance-volume adaptive sampling |

## Insights & Connections

- **2D Foundation Models Empowering 3D Perception**: The combination of open-vocabulary detectors and SAM provides a reliable image-side prior for unlabeled 3D detection, a paradigm that can be extended to other 3D tasks.
- **3D Adaptation of Self-Paced Learning**: Incorporating 3D-specific attributes (such as distance and volume) into the sampling strategy represents an effective customization of general self-paced learning for the 3D domain.
- **Weak Model Aggregation vs. Model Soup**: Similar to methods like Model Soup, but performing online aggregation during self-training without requiring an additional model selection process.
- Future research could consider extending this method to open-vocabulary 3D detection, leveraging GroundingDINO's text-alignment capability to assign class semantics to pseudo-labels.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to fuse 2D scene information in unsupervised 3D detection, with well-designed adaptive sampling and weak model aggregation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated on nuScenes and Lyft, with ablation studies covering all components and hyperparameters.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured, with intuitive diagrams and a well-articulated motivation.
- Value: ⭐⭐⭐⭐ — Outperforms supervised methods in long-range detection, demonstrating the potential of unsupervised LiDAR-camera fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SimPB: A Single Model for 2D and 3D Object Detection from Multiple Cameras](simpb_a_single_model_for_2d_and_3d_object_detection_from_multiple_cameras.md)
- [\[CVPR 2025\] Cubify Anything: Scaling Indoor 3D Object Detection](../../CVPR2025/autonomous_driving/cubify_anything_scaling_indoor_3d_object_detection.md)
- [\[ECCV 2024\] Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)
- [\[ECCV 2024\] OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)
- [\[ECCV 2024\] Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance](weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)

</div>

<!-- RELATED:END -->
