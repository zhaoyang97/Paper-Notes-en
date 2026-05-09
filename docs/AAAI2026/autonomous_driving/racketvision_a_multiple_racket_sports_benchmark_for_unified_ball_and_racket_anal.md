---
title: >-
  [Paper Note] RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis
description: >-
  [AAAI 2026][Autonomous Driving][Ball Sports Analysis] This paper presents RacketVision—the first large-scale benchmark covering three racket sports (table tennis, tennis, and badminton)—which introduces racket pose annotations for the first time and defines three interconnected tasks: ball tracking, racket pose estimation, and ball trajectory prediction. The work reveals the critical role of cross-attention fusion in multimodal trajectory prediction.
tags:
  - AAAI 2026
  - Autonomous Driving
  - Ball Sports Analysis
  - Racket Pose Estimation
  - Trajectory Prediction
  - Multimodal Fusion
  - Cross-Sport Learning
date: 2026-05-08
content_hash: d755f05dc9a0e3fd
---

# RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis

**Conference**: AAAI 2026
**arXiv**: [2511.17045](https://arxiv.org/abs/2511.17045)
**Code**: None
**Area**: Sports Analysis / Computer Vision
**Keywords**: Ball Sports Analysis, Racket Pose Estimation, Trajectory Prediction, Multimodal Fusion, Cross-Sport Learning

## TL;DR

This paper presents RacketVision—the first large-scale benchmark covering three racket sports (table tennis, tennis, and badminton)—which introduces racket pose annotations for the first time and defines three interconnected tasks: ball tracking, racket pose estimation, and ball trajectory prediction. The work reveals the critical role of cross-attention fusion in multimodal trajectory prediction.

## Background & Motivation

### Limitations of Existing Datasets

Racket sports (table tennis, tennis, badminton) are among the most widely participated sports globally and have attracted substantial attention in computer vision research. Existing datasets such as TrackNet, TrackNetv2, and WASB suffer from two critical shortcomings:

**Single-sport limitation**: Each dataset covers only one sport, precluding the exploitation of shared prior knowledge across different racket sports (e.g., cross-sport regularities in ball motion patterns).

**Absence of racket annotations**: Despite rackets being the central component of racket sports, no existing dataset provides racket-related annotations or analysis. This not only limits sports analytics but also constrains complex human–object interaction understanding and neural avatar modeling.

### Paper Goals

To address this gap, this paper proposes the RacketVision benchmark, aiming to:
- Broaden the range of sports covered to enable unified model training and exploit cross-sport shared priors.
- Go beyond conventional ball tracking by defining racket keypoints and supporting a novel racket pose estimation task.
- Introduce ball trajectory prediction as an integrative task to enable downstream applications such as tactical analysis and robotics.

## Method

### Overall Architecture

RacketVision constructs a complete pipeline consisting of three interconnected tasks:

1. **Ball Tracking**: Given RGB frames (single or multiple), predict ball position and a visibility flag.
2. **Racket Pose Estimation**: Given an RGB frame, predict the bounding box and 5 keypoints for each racket.
3. **Ball Trajectory Prediction**: Given historical ball positions and (optionally) racket poses, predict future ball trajectories.

The relationship among the three tasks is as follows: ball trackers and racket pose estimators are first trained on sparsely annotated ground-truth frames, then deployed over full video clips to generate dense per-frame predictions ("soft labels"). These continuous sequences are segmented into historical and future data segments to serve as training data for the trajectory predictor.

### Key Designs

#### 1. Dataset Construction Pipeline

- **Video source**: 942 top-level professional match broadcast videos collected from YouTube, covering three racket sports.
- **Two-stage annotation pipeline**:
  - **Stage 1**: Crowdsourced annotators segment raw videos into valid clips (5–10 seconds, with the ball in motion).
  - **Stage 2**: A sparse annotation strategy is applied—20% of frames are uniformly sampled from each clip for manual annotation.
- **Annotation content**: Ball position (red dot) + visibility flag; racket bounding box (orange rectangle) + 5 sport-specific keypoints.
- **Dataset scale**: 1,672 video clips, 435,179 frames, approximately 12,755 seconds, 64,042 ball annotations, and 24,621 racket annotations.

#### 2. Task Formulation

**Ball Tracking** adopts two settings:
- Single-frame setting: uses only the target frame $I_t$ to assess static detection capability.
- Multi-frame setting: uses the target frame together with the preceding 5 frames $\{I_{t-5}, \ldots, I_t\}$ to leverage temporal context.

**Racket Pose Estimation** uses a single-frame setting, predicting a bounding box and 5 keypoints for each racket per frame:

$$\{(x_i, y_i)\}_{i=1}^{5} \in \mathbb{R}^{10}$$

**Ball Trajectory Prediction** is evaluated under two modality settings and two duration settings:
- Modality: ball position only vs. ball position + racket pose.
- Duration: short trajectory (history 20 frames, prediction 5 frames) vs. long trajectory (history 80 frames, prediction 20 frames).

#### 3. Multimodal Fusion Strategies (Core Contribution)

Three fusion approaches are explored for the trajectory prediction task:

- **Ball-Only**: A unimodal baseline using only historical ball coordinates as input.
- **Concat fusion**: Ball coordinate embeddings and racket pose embeddings are concatenated along the feature dimension before being fed into the backbone, treating all features equally at each timestep.
- **Cross-Attention fusion**: The ball trajectory sequence serves as Query, while the racket pose sequence serves as Key and Value. The model dynamically weights and extracts the most relevant racket pose information at each timestep of the ball trajectory, effectively filtering noise and focusing on key events such as ball–racket contact.

### Loss & Training

- Ball tracking: standard object detection/segmentation losses.
- Racket pose estimation: the RTMPose framework is adopted, with RTMDet as the detector for bounding box generation.
- Trajectory prediction: trained using dense "soft labels" generated by the perception models; evaluation metrics are ADE and FDE (pixel-level Euclidean distance).

## Key Experimental Results

### Main Results

#### Ball Tracking Results (MS-TrackNetV3, BM=✓, #F=4, best multi-sport training model)

| Sport | Metric | Multi-Sport Best | Single-Sport Best | Gain |
|-------|--------|-----------------|-------------------|------|
| Table Tennis | mAP | **71.1** | 68.3 | +2.8 |
| Tennis | mAP | **81.9** | 68.7 | **+19.2%** |
| Badminton | mAP | **83.1** | 72.5 | **+14.6%** |
| Table Tennis | MDE | **3.41** | 6.63 | −48.6% |

#### Racket Pose Estimation Results

| Training Mode | Sport | PCK@0.2 | MPJPE | mOKS | Det. mAP |
|---------------|-------|---------|-------|------|----------|
| Single-Sport | Table Tennis | 75.6 | 10.6 | 0.453 | 72.4 |
| **Multi-Sport** | **Table Tennis** | **81.8** | **9.71** | **0.498** | **78.4** |
| Single-Sport | Tennis | 83.7 | 5.87 | 0.574 | 73.4 |
| **Multi-Sport** | **Tennis** | **89.6** | **5.34** | **0.630** | **79.4** |
| Single-Sport | Badminton | 82.1 | 5.45 | 0.601 | 69.8 |
| **Multi-Sport** | **Badminton** | **88.5** | **5.00** | **0.668** | **75.5** |

#### Trajectory Prediction Results (Long Trajectory, History=80, Future=20)

| Model | Input | Fusion | Table Tennis ADE | Tennis ADE | Badminton ADE |
|-------|-------|--------|-----------------|------------|---------------|
| LSTM | Ball | — | 113.9 | 62.5 | 118.7 |
| LSTM | Ball+Racket | Concat | 139.9 (+22.8%) | 76.8 (+22.9%) | 134.5 (+13.3%) |
| **LSTM** | **Ball+Racket** | **CrossAttn** | **101.3 (−11.1%)** | **55.5 (−11.2%)** | **114.6 (−3.5%)** |
| Transformer | Ball | — | 145.3 | 89.9 | 142.7 |
| Transformer | Ball+Racket | CrossAttn | 127.3 | 74.8 | 122.5 |

### Ablation Study

#### Keypoint Detection Difficulty Analysis (Multi-Sport RTMPose, PCK@0.2)

| Sport | Top | Bottom | Handle | Left | Right |
|-------|-----|--------|--------|------|-------|
| Table Tennis | 97.6 | 97.3 | 97.9 | **64.8** | **64.8** |
| Tennis | 98.6 | 98.9 | 92.6 | 79.7 | 80.1 |
| Badminton | 99.4 | 99.7 | 97.3 | 74.6 | 75.5 |

**Key Findings**: Structural keypoints (top, bottom, handle) achieve accuracy above 92%, whereas side keypoints yield only 64.8%–80.1% due to hand occlusion, motion blur, and viewpoint sensitivity.

### Key Findings

1. **Multi-sport training universally improves generalization**: mAP improves by 19.2% on tennis and 14.6% on badminton, indicating that training on diverse data compels the model to learn more robust features.
2. **Background modeling substantially reduces localization error**: Median-frame background subtraction reduces MDE by 54%–61%.
3. **Naive concatenation fusion is detrimental**: The Concat approach underperforms the ball-only baseline across all settings, as racket information is irrelevant or absent in the majority of trajectory samples.
4. **Cross-attention successfully unlocks the value of racket data**: LSTM+CrossAttn achieves the best overall performance, accurately predicting directional turning points following ball–racket contact.

## Highlights & Insights

- **Elegant dataset design**: The sparse annotation strategy (20% frame sampling) achieves a favorable balance between annotation cost and data quality.
- **Effective cross-sport knowledge transfer**: Unified training enhances generalization, confirming the existence of shareable visual priors across different racket sports.
- **Counter-intuitive finding—fusion can hurt**: Naively concatenating multimodal features degrades performance, underscoring the importance of fusion architecture design.
- **Selective attention mechanism of Cross-Attention**: The model automatically disregards racket information in non-contact frames while attending to it during contact frames, demonstrating the advantage of intelligent fusion.

## Limitations & Future Work

- Side keypoint detection accuracy remains low (64.8%–80.1%), requiring better handling of hand occlusion and motion blur.
- The improvement from cross-attention is limited in the short-trajectory setting, where most samples involve a ball in flight with no contact event.
- Annotations are currently 2D only; extension to 3D trajectory reconstruction and rotation estimation is a natural future direction.
- Trajectory prediction employs only simple LSTM and Transformer backbones; more advanced temporal models warrant exploration.

## Related Work & Insights

- **TrackNet series**: Pioneering work in single-sport ball tracking; RacketVision significantly surpasses these in scale and diversity.
- **RTMPose/YOLO**: General-purpose pose estimation and detection models are successfully transferred to sports analysis.
- **Multimodal fusion insights**: Attention-based fusion mechanisms widely used in autonomous driving are equally critical in sports analytics.

## Rating

- Novelty: ⭐⭐⭐⭐ (First multi-sport dataset with racket pose annotations, filling an important gap)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive evaluation across three tasks with multi-dimensional analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, in-depth analysis)
- Value: ⭐⭐⭐⭐ (Dataset and fusion insights make important contributions to the community)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](../../ICCV2025/autonomous_driving/uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)
- [\[AAAI 2026\] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors](priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[AAAI 2026\] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework](when_person_re-identification_meets_event_camera_a_benchmark_dataset_and_an_attr.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](../../CVPR2026/autonomous_driving/an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)

<!-- RELATED:END -->
