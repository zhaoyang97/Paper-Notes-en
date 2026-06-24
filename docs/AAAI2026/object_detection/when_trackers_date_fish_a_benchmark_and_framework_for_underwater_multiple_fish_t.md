---
title: >-
  [Paper Note] When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking
description: >-
  [AAAI 2026 Oral][Object Detection][Underwater Multiple Fish Tracking] This paper proposes MFT25, a large-scale underwater multiple fish tracking dataset (15 sequences, 408K annotations), and SU-T, a tracking framework combining UKF with FishIoU, achieving state-of-the-art performance of 34.1 HOTA and 44.6 IDF1. Statistical analyses further reveal fundamental differences between fish tracking and terrestrial object tracking.
tags:
  - "AAAI 2026 Oral"
  - "Object Detection"
  - "Underwater Multiple Fish Tracking"
  - "MOT Benchmark"
  - "Unscented Kalman Filter"
  - "FishIoU"
  - "Nonlinear Motion Model"
date: 2026-05-08
content_hash: 36e5b4ce2fea5c09
---

# When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking

**Conference**: AAAI 2026 Oral  
**arXiv**: [2507.06400](https://arxiv.org/abs/2507.06400)  
**Code**: [https://vranlee.github.io/SU-T/](https://vranlee.github.io/SU-T/)  
**Area**: Object Detection  
**Keywords**: Underwater Multiple Fish Tracking, MOT Benchmark, Unscented Kalman Filter, FishIoU, Nonlinear Motion Model

## TL;DR

This paper proposes MFT25, a large-scale underwater multiple fish tracking dataset (15 sequences, 408K annotations), and SU-T, a tracking framework combining UKF with FishIoU, achieving state-of-the-art performance of 34.1 HOTA and 44.6 IDF1. Statistical analyses further reveal fundamental differences between fish tracking and terrestrial object tracking.

## Background & Motivation

**Multiple Fish Tracking (MFT)** is a core technology for marine ecological research, aquaculture optimization, and fishery resource management. By continuously tracking and associating individual targets across video sequences, it enables quantitative analysis of fish movement patterns, collective interactions, and environmental adaptation mechanisms.

**Unique Challenges of Underwater Tracking**:

**High Morphological Similarity**: Individuals of the same species are highly similar in body shape, color, and texture (t-SNE analysis reveals that intra-class overlap and inter-class separation in fish features are far worse than in pedestrian datasets), leading to frequent identity switches and trajectory fragmentation.

**Nonlinear Motion Patterns**: Fish swimming direction changes are abrupt and unpredictable (angular velocity distribution analysis shows far greater directional instability than terrestrial targets such as pedestrians), rendering the linear motion assumption of standard Kalman filtering inapplicable.

**Distinctive Morphology**: Fish bodies are fusiform/streamlined rather than rectangular like pedestrians or vehicles; the anterior region (head) contains more discriminative features.

**Limitations of Existing Datasets**:
- BrackishMOT (49K annotations), 3D-ZeF (86K annotations), and MFT22 (155K annotations) are limited in scale.
- Insufficient environmental diversity, with most sequences recorded under simplified, controlled conditions.
- Lack of large-scale, high-quality, standardized benchmarks.

## Method

### Overall Architecture

SU-T (Scale-aware and Unscented Tracker) follows the SDE (Separate Detection and Embedding) paradigm and consists of three core components:

1. **Detector**: YOLOX-based pyramid design with decoupled heads.
2. **Association Module**: UKF motion prediction + FishIoU matching cost + Hungarian algorithm.
3. **Optional Re-ID Module**: GeM Pooling for appearance feature extraction.

### Key Designs

#### 1. **MFT25 Dataset Construction**

- **Capture Devices**: Canon EOS R6 and Sony α7M3.
- **Scene Coverage**: Industrial recirculating aquaculture tanks and laboratory aquaria, encompassing varying lighting conditions (daylight to nighttime) and both top-down and side-view perspectives.
- **Fish Species**: Grouper and koi carp at different developmental stages, providing substantial appearance variation.
- **Scale**: 15 video sequences, 223 trajectories, 48,066 frames, **408,578 precise annotations** (2.6–8.3× more than prior datasets).
- **All real video footage**, with no synthetic augmentation.

| Dataset | Sequences | Trajectories | FPS | Frames | Annotations |
|--------|--------|--------|-----|------|---------|
| BrackishMOT | 98 | 638 | 25 | 14,017 | 49,364 |
| 3D-ZeF | 8 | 32 | 60 | 14,398 | 86,452 |
| MFT22 | 10 | 234 | 25 | 9,100 | 155,437 |
| **MFT25** | **15** | **223** | **25** | **48,066** | **408,578** |

#### 2. **Unscented Kalman Filter (UKF)**

Standard KF assumes a linear motion model, which is unsuitable for the nonlinear swimming patterns of fish. UKF addresses nonlinearity through deterministic sampling:

**Sigma Point Generation**: For state vector $\mathbf{x} \in \mathbb{R}^n$, $2n+1$ sigma points are generated:
- $\mathcal{X}_0 = \mathbf{x}$
- $\mathcal{X}_i = \mathbf{x} + (\sqrt{(n+\lambda)\mathbf{P}})_i$, $i=1,...,n$
- $\mathcal{X}_{i+n} = \mathbf{x} - (\sqrt{(n+\lambda)\mathbf{P}})_i$

**Prediction Step**: Sigma points are propagated through the nonlinear state transition function $\mathbf{f}$ and combined via weighted summation to obtain the predicted state and covariance.

**Update Step**: Sigma points are transformed through the nonlinear measurement function $\mathbf{h}$; the Kalman gain $\mathbf{K}_k = \mathbf{P}_{xz} \mathbf{P}_{zz}^{-1}$ is computed to update the state estimate.

#### 3. **FishIoU: Scale-Aware Association**

A association metric tailored to the distinctive morphology and motion patterns of fish, composed of five weighted components:

$$\text{FishIoU} = \omega_1 \cdot \text{IoU} + \omega_2 \cdot \text{cIoU} + \omega_3 \cdot \alpha_r + \omega_4 \cdot \alpha_a - \omega_5 \cdot s \cdot d_c$$

Component definitions:
- **Standard IoU** ($\omega_1=1.0$): Basic overlap measure.
- **Central Region IoU cIoU** ($\omega_2=0.3$): Defines a central region via asymmetric inward cropping $B^c = [x_1+\alpha w, y_1+\beta h, x_2-\gamma w, y_2-\beta h]$ ($\alpha=0.15, \beta=0.3, \gamma=0.25$), emphasizing anterior fish features.
- **Aspect Ratio Consistency** $\alpha_r$ ($\omega_3=0.1$): $\min(r_1,r_2)/\max(r_1,r_2)$.
- **Area Ratio Consistency** $\alpha_a$ ($\omega_4=0.2$): Captures the constraint that fish body size should not change abruptly between frames.
- **Center Distance Penalty** $d_c$ ($\omega_5=0.4$): Incorporates a small-target scaling factor $s=1-e^{-\min(a_1,a_2)/1000}$.

### Association Strategy

A three-stage cascaded association with progressive confidence handling is adopted (inspired by ByteTrack/OC-SORT/HybridSORT):

1. **Stage 1**: High-confidence detections are matched to existing tracklets using FishIoU.
2. **Stage 2**: Remaining tracklets are matched against low-confidence detections (to recover occluded targets).
3. **Stage 3**: Unmatched detections are reconnected to historical observations of tracklets (to handle sudden direction changes).

### Loss & Training

- Detector: YOLOX, trained on the MFT25 training split.
- Re-ID: SBS-S101 model (selected via ablation).
- Hardware: NVIDIA A100 GPU.
- Evaluation Metrics: HOTA + CLEAR MOT metrics.

## Key Experimental Results

### Main Results

| Method | Type | HOTA↑ | IDF1↑ | MOTA↑ | AssA↑ | IDs↓ |
|------|------|-------|-------|-------|-------|------|
| FairMOT | JDE | 22.2 | 26.9 | 47.5 | 13.9 | 939 |
| OC-SORT | SDE | 25.0 | 34.6 | 46.7 | 17.8 | 550 |
| BoT-SORT | SDE | 26.8 | 36.8 | 49.1 | 19.4 | 500 |
| ByteTrack | SDE | 31.8 | 40.4 | 69.6 | 20.4 | 489 |
| TrackFormer | Transformer | 30.4 | 35.3 | **74.6** | 17.7 | 718 |
| HybridSORT† | SDE | 32.7 | 41.7 | 69.2 | 21.7 | 562 |
| **SU-T†** | **SDE** | **34.1** | **44.6** | 69.0 | **23.6** | **544** |

SU-T† surpasses the previous best by 1.4 HOTA points (34.1). The advantage is especially pronounced on identity-preservation metrics (IDF1 44.6, AssA 23.6).

### Ablation Study

**Re-ID Model Selection**:

| Model | HOTA↑ | IDF1↑ | IDs↓ |
|------|-------|-------|------|
| SBS-R50 | 31.0 | 39.9 | 713 |
| SBS-R101 | 30.3 | 38.1 | 678 |
| SBS-S50 | 32.7 | 41.7 | 562 |
| **SBS-S101** | **33.8** | **43.7** | **550** |

IBN variants do not consistently improve performance, indicating that domain adaptation techniques designed for terrestrial tracking do not necessarily transfer to the underwater setting.

**Association Metric Ablation**:

| Method | HOTA↑ | IDF1↑ | IDs↓ |
|------|-------|-------|------|
| Center Distance | 28.9 | 37.3 | 1273 |
| Standard IoU | 32.8 | 40.1 | 579 |
| CIoU | 30.7 | 39.6 | 727 |
| DIoU | 30.8 | 39.6 | 728 |
| HMIoU | 32.3 | 38.4 | 613 |
| GIoU | 32.9 | 40.0 | 573 |
| **FishIoU** | **33.4** | **41.7** | 607 |
| **FishIoU†** | **33.6** | **43.3** | **547** |

**Motion Model Ablation**:

| Model | IoU Type | HOTA↑ | IDF1↑ | IDs↓ |
|------|---------|-------|-------|------|
| KF | FishIoU | 33.1 | 41.0 | 612 |
| AKF | FishIoU | 22.6 | 24.0 | 2368 |
| STF | FishIoU | 31.6 | 38.1 | 663 |
| **UKF** | FishIoU | **33.2** | **41.6** | **609** |
| **UKF†** | FishIoU | **34.1** | **44.6** | **544** |

UKF consistently outperforms all other motion models, validating that nonlinear motion modeling is better suited to the complex swimming patterns of fish. AKF performs worst (HOTA 22.6), likely because its adaptive mechanism becomes unstable under the high-frequency directional changes exhibited by fish.

### Key Findings

1. **Fundamental Differences Between Fish and Terrestrial Tracking**:
    - Directional instability: Fish angular velocity variation is substantially higher than that of pedestrians (Figure 9).
    - Velocity fluctuation: Average speed variation in MFT25 is far greater than in other datasets (Figure 10).
    - Feature separability: Track-level embeddings of fish overlap severely, with inter-class margins far smaller than those of pedestrians (cosine similarity heatmaps and t-SNE visualizations).
2. **Limitations of Transformer-Based Trackers**: TrackFormer achieves the highest MOTA (74.6) but shows weaker identity-preservation metrics and high computational cost.
3. **Domain Adaptation Techniques Do Not Transfer Directly**: IBN normalization optimized for terrestrial tracking does not yield consistent improvements in the underwater domain.
4. **High Inter-Sequence Heterogeneity**: HOTA ranges from 20.3 to 65.2 across sequences, underscoring the diversity of challenges in underwater tracking.
5. SU-T also achieves competitive performance on MOT17/MOT20 (60.4/56.5 HOTA), demonstrating generalization capability.

## Highlights & Insights

1. **MFT25 Dataset**: Currently the largest underwater multiple fish tracking benchmark, with 2.6–8.3× more annotations than predecessors, filling a critical gap in the field.
2. **Biologically Inspired FishIoU**: Asymmetric inward cropping emphasizes anterior fish features, while area consistency constraints reflect the temporal stability of fish body size—elegantly encoding domain knowledge into the metric function.
3. **Theoretical Soundness of UKF**: The nonlinear nature of fish swimming (sudden turns, rapid acceleration) makes UKF inherently more appropriate than standard KF.
4. **Systematic Statistical Analysis**: Multiple quantitative dimensions—angular velocity distributions, velocity evolution, and feature separability—are used to rigorously characterize the uniqueness of fish tracking.

## Limitations & Future Work

1. Appearance discrimination among visually similar fish remains the primary bottleneck, with limited Re-ID effectiveness.
2. Performance degrades noticeably in extreme density scenarios with heavy fish occlusion (SN-011 sequence HOTA is only 20.3).
3. The weight parameters $\omega_1$–$\omega_5$ and morphological parameters $\alpha, \beta, \gamma$ in FishIoU are manually set and could be learned adaptively.
4. The dataset covers only two fish species (grouper and koi carp); species diversity could be expanded.
5. Fish motion in 3D space is not considered (only 2D projection).

## Related Work & Insights

- ByteTrack's dual-confidence matching and OC-SORT's historical observation recovery are successfully integrated into SU-T.
- HybridSORT's HMIoU inspired the design of FishIoU, with fish morphological priors additionally incorporated in this work.
- Non-human MOT datasets such as BEE24 (bee tracking) highlight the importance of domain-specific benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ — The UKF + FishIoU combination has domain-specific character, though the overall framework is relatively conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive comparisons, extensive ablations, cross-domain generalization validation, and detailed statistical analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, excellent visualizations, and persuasive statistical analysis.
- Value: ⭐⭐⭐⭐ — The MFT25 dataset contribution offers long-term value to the underwater vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multiple Object Tracking as ID Prediction](../../CVPR2025/object_detection/multiple_object_tracking_as_id_prediction.md)
- [\[CVPR 2026\] GMT: Effective Global Framework for Multi-Camera Multi-Target Tracking](../../CVPR2026/object_detection/gmt_effective_global_framework_for_multi-camera_multi-target_tracking.md)
- [\[ECCV 2024\] WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](../../ECCV2024/object_detection/walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)
- [\[AAAI 2026\] Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection](towards_multiple_missing_values-resistant_unsupervised_graph_anomaly_detection.md)
- [\[AAAI 2026\] AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios](aerialmind_towards_referring_multi-object_tracking_in_uav_sc.md)

</div>

<!-- RELATED:END -->
