---
title: >-
  [Paper Note] BlinkTrack: Feature Tracking over 80 FPS via Events and Images
description: >-
  [ICCV 2025][Video Understanding][Event Camera] BlinkTrack introduces a differentiable Kalman filter into a learning framework to address the challenges of asynchronous data association and uncertainty-aware fusion betwee…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Event Camera"
  - "Feature Tracking"
  - "Kalman Filter"
  - "Multimodal Fusion"
  - "High Frame Rate Tracking"
date: 2026-05-08
content_hash: 9a6dfe90672b4190
---

# BlinkTrack: Feature Tracking over 80 FPS via Events and Images

**Conference**: ICCV 2025
**arXiv**: [2409.17981](https://arxiv.org/abs/2409.17981)  
**Code**: [GitHub](https://github.com/ColieShen/BlinkTrack)  
**Area**: Video Understanding
**Keywords**: Event Camera, Feature Tracking, Kalman Filter, Multimodal Fusion, High Frame Rate Tracking

## TL;DR

BlinkTrack introduces a differentiable Kalman filter into a learning framework to address the challenges of asynchronous data association and uncertainty-aware fusion between event cameras and conventional cameras, achieving feature tracking at over 80 FPS with significantly superior performance in occlusion scenarios compared to existing methods.

## Background & Motivation

Feature tracking is a foundational task in computer vision, underpinning applications such as SfM, SLAM, and object tracking. Event cameras asynchronously detect scene changes with extremely high temporal resolution, making them well-suited for high-frequency tracking and extreme lighting conditions. However, they suffer from a critical limitation: **they cannot capture fine-grained texture information**, and in slow-motion scenes the spatial signal becomes sparse, leading to error accumulation and tracking loss.

Fusing event data with conventional image frames is a promising direction, but it poses three key challenges:

**Data Association**: Event cameras detect changes asynchronously (e.g., at 100 Hz), while conventional cameras operate at a fixed frame rate (e.g., 30 Hz), making the two streams inherently unsynchronized.

**Uncertainty-Aware Fusion**: The two modalities exhibit different uncertainty distributions — event cameras are unreliable in low-texture regions, while conventional cameras degrade under motion blur or extreme lighting — requiring adaptive weighting.

**Runtime Efficiency**: High frame rate tracking demands minimal computational overhead; approaches such as attention mechanisms, though effective, are too heavyweight.

**Core Insight**: The Kalman filter naturally supports asynchronous data fusion and uncertainty modeling, but conventional implementations require manual parameter tuning. BlinkTrack extends this into a learnable framework — the network learns to predict uncertainty, and the Kalman filter automatically weights and fuses inputs based on that uncertainty — enabling end-to-end training.

## Method

### Overall Architecture

BlinkTrack consists of three core components:
- **Event Module**: Generates initial tracking predictions and uncertainty estimates from event data.
- **Image Module**: Generates relocalization predictions and uncertainty estimates from image frames.
- **Kalman Filter**: Receives asynchronous predictions from both modules and performs optimal fusion based on timestamps and uncertainty.

### Key Designs

1. **Event Module**: Adopts a paradigm of matching a reference image patch against an event patch:

    - **Pyramid Feature Encoder**: Two shallow U-Nets encode the reference patch $P_{evt_{ref}}$ (cropped from a grayscale frame) and the event patch $P_{evt_j}$ (preprocessed via SBT-Max) respectively, constructing a 2-level feature pyramid for multi-scale perception and computing the correlation map $C_{evt_j}$.
    - **Dual-LSTM Displacement Predictor**: A feature LSTM (ConvLSTM) propagates historical feature information, while a displacement LSTM aggregates displacement features and produces the final displacement $\Delta\hat{p}$ via a gating mechanism.
    - **Uncertainty Predictor**: A 5-layer CNN maps features to 2-class scores (certain/uncertain), normalized to $[0, 1]$ via Softmax, then mapped to $[0, 10]$ through a parabolic function, and placed into the covariance matrix $\hat{\Sigma}_{evt_j} \in \mathbb{R}^{2 \times 2}$.

2. **Image Module**: Lightweight design (>50 FPS), inspired by PIPs:

    - **Pyramid Encoder**: A RAFT-like architecture encodes the full frame into a feature map $F_{img}$ at 1/8 resolution.
    - **Multi-Scale Correlation Pyramid**: 4-level average pooling → correlation with reference features → sampling and concatenating $7 \times 7$ patches.
    - **MLP-Mixer Prediction Head**: Aggregates reference features, target features, correlation maps, and positional encodings to predict displacement and uncertainty.
    - Supports **iterative refinement**: features and correlation maps are re-sampled at the predicted position to progressively converge to the true location.

3. **Differentiable Kalman Filter**: Adopts a constant-velocity motion model with state $x = (x, y, v_x, v_y)^T$:

    - **Prediction Step**: Propagates state and uncertainty according to the motion model and time interval.
    - **Update Step**: Receives observations $(\Delta\hat{p}, \hat{\Sigma})$ from either module and fuses them via the Kalman gain $K_k = P_{k|k-1} H^T S_k^{-1}$.
    - Under high uncertainty, the filter relies more on the internal constant-velocity model; under low uncertainty, it defers more to the current observation.
    - The entire process is differentiable, with uncertainty supervised indirectly through ground truth.

### Loss & Training

Event and image modules are trained in stages:
- **Event Module Training**: The encoder and displacement predictor are first trained without the uncertainty predictor or Kalman filter ($\mathcal{L}_{\hat{disp}}$); the encoder and displacement predictor are then frozen, and the uncertainty predictor is trained ($\mathcal{L}_{\tilde{disp}} + w_1 \mathcal{L}_{vis}$).
- **Image Module Training**: The Kalman filter and uncertainty training are enabled from the start ($\mathcal{L}_{image} = \mathcal{L}_{\tilde{disp}} + w_2 \mathcal{L}_{\hat{disp}} + w_3 \mathcal{L}_{uncert}$).
- Visibility loss $\mathcal{L}_{vis} = \text{CrossEntropy}(1-\hat{\sigma}, g)$ serves as a proxy supervision signal for uncertainty.
- Training data comes from the proposed MultiTrack dataset (high frame rate synthesis + occlusion).

Dataset contributions (MultiTrack, EC-occ, EDS-occ): The synthetic pipeline generates color images, event data, occluded trajectories, and visibility labels, using FILM for frame interpolation and DVS-Voltmeter for event synthesis.

## Key Experimental Results

### Main Results

**EC and EDS Datasets (Feature Age / Expected FA)**:

| Method | Data | EC FA↑ | EC Exp FA↑ | EDS FA↑ | EDS Exp FA↑ |
|--------|------|--------|-----------|---------|------------|
| EKLT | E | 0.811 | 0.775 | 0.325 | 0.205 |
| Deep-EV-Tracker | E | 0.795 | 0.787 | 0.549 | 0.451 |
| Ours (E) | E | 0.833 | 0.819 | 0.568 | 0.474 |
| D-Tracker + KLT | E+I | 0.735 | 0.730 | 0.594 | 0.503 |
| Ours (E+I w. K) | E+I | **0.851** | **0.845** | **0.653** | **0.550** |

### Ablation Study

| Component | EC Exp FA↑ | EC-occ↑ | EDS Exp FA↑ | EDS-occ↑ | #Params |
|-----------|-----------|---------|------------|---------|---------|
| Full (Pyramid + Dual LSTM) | **0.819** | **0.522** | **0.474** | **0.343** | 33.1M |
| w/o Pyramid | 0.789 | 0.439 | 0.395 | 0.283 | 29.3M |
| Feature LSTM only | 0.794 | 0.473 | 0.428 | 0.337 | 32.7M |
| Displacement LSTM only | 0.617 | 0.332 | 0.282 | 0.186 | 31.9M |
| w/o LSTM | 0.568 | 0.338 | 0.294 | 0.225 | 31.6M |

Occlusion experiments: The Kalman filter yields the most substantial gains in occlusion scenarios — E+I+K improves $\delta_{occ}$ on EC-occ occluded points from 11.4 to 28.5 (+149%).

### Key Findings

- The event module alone already outperforms Deep-EV-Tracker, with the pyramid feature encoder contributing substantially.
- Naive fusion (replacing the initial point) can even degrade performance, whereas Kalman-based fusion yields consistent and significant improvements.
- Runtime efficiency: event module <9 ms/frame, Kalman filter <1 ms, overall >80 FPS (multimodal) and >100 FPS (event preprocessing only).
- Uncertainty visualizations confirm that the model correctly outputs high uncertainty during occlusion.
- Under extreme lighting (DSEC dataset), CoTracker fails completely while BlinkTrack remains operational.
- The MultiTrack dataset provides larger displacement ranges and more occlusion than MultiFlow, better unleashing the learning potential of the model.

## Highlights & Insights

- Elegantly integrates classical Kalman filtering with deep learning: the network predicts uncertainty while the filter performs optimal fusion.
- Architecture design explicitly accounts for efficiency: the event module uses patch-level features to avoid full-frame encoding; the image module employs a lightweight encoder to achieve >50 FPS relocalization.
- The two modules are loosely coupled through the Kalman filter, enabling modular replacement and asynchronous inference.
- The proposed datasets fill a gap in occlusion-annotated benchmarks for event-camera feature tracking.

## Limitations & Future Work

- Network capacity and receptive field may impose an upper bound on current performance.
- The MultiTrack training data does not model motion blur, leading to a slight performance degradation for the image module + Kalman on EDS.
- The constant-velocity motion assumption is relatively simplistic; an Extended Kalman Filter (EKF) could handle more complex motion patterns.
- The two modules are trained separately; joint training may yield further gains at the cost of additional computational resources.

## Related Work & Insights

- Compared to the naive fusion strategy of Deep-EV-Tracker (replacing the initial point), the Kalman-based fusion approach demonstrates clear advantages.
- Unlike FF-KDT, which aligns events to the image frame rate before fusion, BlinkTrack preserves the native high temporal resolution of event data.
- The differentiable Kalman filter paradigm is generalizable to other asynchronous multimodal fusion settings (e.g., LiDAR + camera, IMU + vision).

## Rating

- Novelty: ⭐⭐⭐⭐ (The use of a differentiable Kalman filter for asynchronous multimodal feature tracking is a natural and elegant design.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets + occlusion benchmarks + ablations + runtime analysis + long-term stability.)
- Writing Quality: ⭐⭐⭐⭐ (Method description is thorough, with complete supplementary material.)
- Value: ⭐⭐⭐⭐ (A practical advance in event-camera tracking; >80 FPS meets real-world deployment requirements.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Towards Efficient General Feature Prediction in Masked Skeleton Modeling](towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [\[ACL 2026\] ArrowGEV: Grounding Events in Video via Learning the Arrow of Time](../../ACL2026/video_understanding/arrowgev_grounding_events_in_video_via_learning_the_arrow_of_time.md)
- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](online_dense_point_tracking_with_streaming_memory.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)

</div>

<!-- RELATED:END -->
