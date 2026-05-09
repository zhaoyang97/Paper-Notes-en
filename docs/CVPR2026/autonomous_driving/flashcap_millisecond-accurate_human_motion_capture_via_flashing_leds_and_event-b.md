---
title: >-
  [Paper Note] FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision
description: >-
  [CVPR 2026][Autonomous Driving][Event camera] FlashCap is proposed as the first motion capture system combining flashing LEDs with event cameras, where each LED is assigned a unique flashing frequency for identity recognition. The system enables the construction of FlashMotion, the first human motion dataset with 1000Hz annotation precision (7.15 million frames), and introduces the ResPose baseline, reducing motion timing error from ~50ms to ~5ms and lowering pose estimation MPJPE by approximately 40%.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Event camera
  - human motion capture
  - LED markers
  - high temporal resolution
  - spiking neural network
date: 2026-05-08
content_hash: cade70ae78989dc5
---

# FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision

**Conference**: CVPR 2026
**arXiv**: [2603.19770](https://arxiv.org/abs/2603.19770)
**Code**: Coming soon
**Area**: Autonomous Driving / Human Pose Estimation
**Keywords**: Event camera, human motion capture, LED markers, high temporal resolution, spiking neural network

## TL;DR

FlashCap is proposed as the first motion capture system combining flashing LEDs with event cameras, where each LED is assigned a unique flashing frequency for identity recognition. The system enables the construction of FlashMotion, the first human motion dataset with 1000Hz annotation precision (7.15 million frames), and introduces the ResPose baseline, reducing motion timing error from ~50ms to ~5ms and lowering pose estimation MPJPE by approximately 40%.

## Background & Motivation

1. **State of the Field**: Precise motion timing (PMT) is critical in competitive sports and similar domains — a 2ms difference in a luge race can determine medal outcomes. Current human pose estimation (HPE) research predominantly targets spatial accuracy, with insufficient attention to temporal precision. Existing motion capture systems, including Vicon (optical markers, ~330Hz), Xsens (IMU, 60–240Hz), and standard RGB cameras (30–60Hz), cannot meet millisecond-level temporal requirements.
2. **Limitations of Prior Work**: (a) High-speed RGB cameras (≥1000Hz) offer high frame rates but are prohibitively expensive (NAC HX-7s exceeds $45,000, roughly 9× the cost of event cameras), require intense illumination, and impose bandwidth and storage demands two orders of magnitude higher than event cameras; (b) The highest annotation frame rate among publicly available human motion datasets is only 120Hz (BEAHM), an order of magnitude below millisecond precision; (c) Existing temporal annotation approaches are limited by the sampling ceiling of auxiliary modalities or interpolation errors, preventing them from exceeding 120Hz.
3. **Root Cause**: How can high temporal resolution (1000Hz) human motion capture and annotation be achieved at low cost and low bandwidth?
4. **Paper Goals**: (a) Build a novel low-cost motion capture system that circumvents the bottleneck of high-speed cameras; (b) Collect the first multi-modal human motion dataset with 1000Hz annotation precision; (c) Propose and evaluate HPE baseline methods operating at high temporal resolution.
5. **Starting Point**: Event cameras offer microsecond-level temporal resolution and extremely low bandwidth, but deriving high-frequency ground-truth annotations from event streams remains a key challenge. The authors creatively employ LEDs with distinct flashing frequencies as body markers — event cameras can precisely capture LED flashing patterns, and frequency analysis automatically resolves LED identity and position, directly generating 1000Hz 2D joint location annotations from the event stream.
6. **Core Idea**: Joint identity is encoded via unique LED flashing frequencies; the event camera natively captures flashing patterns at high temporal resolution; a frequency-matching algorithm automatically generates 1000Hz pose annotations — all at low cost, low bandwidth, and without high-speed cameras.

## Method

### Overall Architecture

The FlashCap system comprises three components: (1) **MoCap Outfit**: a wearable garment embedding 17 LEDs and 17 IMUs; (2) **Multi-modal acquisition rig**: an event camera (Prophesee 1280×720) + RGB camera (Hikrobot 1920×1200, 20fps) + LiDAR (Ouster OS-1 128-line, 20fps), with a beam splitter enabling pixel-aligned and temporally synchronized event-RGB capture; (3) **Annotation pipeline**: automatically identifies LED flashing patterns from the event stream, resolves their identities, and generates 1000Hz 2D joint location annotations. This system underpins the FlashMotion dataset, and the ResPose baseline is proposed for high-temporal-resolution HPE.

### Key Designs

1. **LED Flashing Encoding and Identity Recognition**

    - **Function**: Assigns a unique identity to each joint LED via a distinctive flashing frequency, enabling the event camera to automatically distinguish different joints.
    - **Mechanism**: Each LED $i$ flashes at a configurable frequency (~4000Hz) with unique on-time $t_i^p$ and off-time $t_i^n$ (in the 100–300μs range), forming a distinctive flashing signature. The event camera asynchronously triggers events $e=(h,w,t,p)$, where high-density event regions correspond to LED locations. The annotation pipeline proceeds in four steps: (a) **Event clustering**: segment the event stream into 1ms event frames and apply DBSCAN to identify high-density regions; (b) **Frequency identification**: analyze positive and negative polarity event sequences per cluster to compute average on-time $\bar{t_j^p}$, off-time $\bar{t_j^n}$, and flashing period $\bar{T_j}$; (c) **Noise filtering**: temporal smoothing combined with outlier rejection; (d) **LED-cluster matching**: compute distance $d_{ji} = \alpha \cdot d_{ji}^t + \beta \cdot d_{ji}^p$ (on/off-time distance + period distance) and apply bipartite graph matching to find the optimal correspondence.
    - **Design Motivation**: Unlike traditional optical markers (requiring high camera frame rates) or RFID (insufficient precision), LED flashing frequency encoding is naturally compatible with the asynchronous operation of event cameras — any luminance change exceeding the threshold triggers an event, with timestamp precision at the microsecond level.

2. **FlashMotion Dataset**

    - **Function**: Provides the first multi-modal human motion dataset with 1000Hz annotation precision.
    - **Mechanism**: 20 volunteers (10 male, 10 female), 4 scenes (indoor and outdoor), 11 major action categories comprising 19 sub-categories, 240 sequences. The dataset includes 144,350 RGB frames, 144,350 LiDAR point cloud frames, and 2 hours of event and IMU data. **2D annotations at 1000Hz** (automatically generated by the annotation pipeline and manually corrected); **3D annotations at 60Hz** (SMPL parameters derived from IMU and LiDAR fusion). Total annotated frames: 7.15 million, representing an order-of-magnitude increase over existing datasets.
    - **Design Motivation**: Annotation frame rates in existing HPE datasets are bounded by traditional optical systems (maximum 120Hz). The FlashCap LED scheme directly generates native 1000Hz annotations from the event stream, bypassing the frequency bottleneck of conventional optical MoCap.

3. **ResPose: Residual Pose Estimation Baseline**

    - **Function**: Achieves 1000Hz pose estimation by leveraging structural priors from low-frame-rate RGB and micro-motion captured by high-frequency events.
    - **Mechanism**: The high-resolution pose is computed as $P_i = P_{rgb} + P_i^{\Delta}$, where $P_{rgb}$ is the anchor pose provided by a low-frame-rate RGB branch (e.g., ViTPose), and $P_i^{\Delta}$ is the residual pose estimated by the event branch. The event branch employs a **SNN-CNN hybrid encoder**: local event patches of size $32 \times 32$ are dynamically cropped centered on RGB anchors, integrated temporally via Leaky Integrate-and-Fire (LIF) spiking neurons, and processed with $1 \times 1$ convolutions to suppress background noise. A **multi-modal residual Transformer** concatenates RGB anchor features and event features before feeding them into a Transformer encoder, which models kinematic constraints across all 17 joints via global self-attention. The model is trained end-to-end with an L2 loss.
    - **Design Motivation**: High-frequency event streams encode micro-motion variations (residual signals) rather than complete spatial structure. Treating RGB frames as structural anchors and events as residual corrections is a natural and efficient decomposition. SNNs are inherently well-suited to processing asynchronous event data.

### Loss & Training

ResPose is trained end-to-end using an L2 distance loss that minimizes the error between predicted poses and the 1000Hz ground truth. The RGB branch is initialized from a pretrained ViTPose; the event branch is trained from scratch.

## Key Experimental Results

### Main Results

Precise Motion Timing (PMT) — estimated timing error for joint crossing a line (ms):

| Method | Kicking | Punching | Jumping |
|--------|---------|----------|---------|
| ViTPose (RGB) | 48.5 | 62.3 | 31.4 |
| Hybrid ANN-SNN (Event) | 85.2 | 54.1 | 66.7 |
| LEIR (RGB+Event) | 112.4 | 135.8 | 78.2 |
| **ResPose (Ours)** | **7.2** | **4.8** | **6.5** |

High temporal resolution HPE (1000Hz):

| Method | MPJPE↓ | PCK0.3↑ | PCK0.5↑ |
|--------|--------|---------|---------|
| ViTPose (linear interp.) | 10.06 | 0.96 | 0.98 |
| Hybrid ANN-SNN | 22.48 | 0.82 | 0.91 |
| EventPointPose | 51.61 | 0.48 | 0.74 |
| EvSharp2Blur | 8.78 | 0.95 | 0.96 |
| ResPose (ANN Variant) | 8.12 | 0.95 | 0.96 |
| **ResPose (Ours, SNN)** | **5.66** | **0.97** | **0.99** |

### Ablation Study

Annotation pipeline ablation (precision / recall):

| Configuration | Kicking Precision | Kicking Recall | Note |
|---------------|-------------------|----------------|------|
| w/o $d_{ji}^t$ | 43.34% | 97.80% | Removing on/off-time distance → extensive mismatches |
| w/o $d_{ji}^p$ | 69.70% | 97.56% | Removing period distance → degraded matching quality |
| w/o outlier filtering | 96.52% | 95.69% | Noise interference causes missed detections |
| w/o tracking | 98.38% | 98.16% | Cannot recover under occlusion |
| **Full pipeline** | **99.99%** | **98.99%** | Near-perfect precision |

### Key Findings

- **ResPose achieves an order-of-magnitude improvement on the PMT task**: timing error is reduced from ~50ms (RGB-only) and ~55–86ms (event-only) to ~5–7ms, demonstrating the effectiveness of combining RGB structural anchors with event-based residual correction.
- **Existing pure-event methods fail on PMT** (LEIR error: 78–136ms), indicating that high temporal resolution input does not automatically yield high temporal resolution output — training with 1000Hz ground truth is essential.
- **The SNN encoder outperforms its ANN variant**: MPJPE decreases from 8.12 to 5.66, confirming the inherent advantage of spiking neural networks for asynchronous event data.
- The annotation pipeline achieves 99.99% precision and 98.82% recall, closely matching manual annotation and validating the robustness of the LED frequency encoding scheme.
- Spline interpolation from 100Hz high-speed cameras still introduces substantial error on fast motions (28.5px on jumping), confirming the necessity of native 1000Hz annotations.

## Highlights & Insights

- The combination of **LED frequency encoding and event cameras** is remarkably elegant: hardware design circumvents software algorithmic limitations. Unlike coloring each LED differently (an RGB-camera approach), encoding identity via distinct flashing frequencies is naturally compatible with the asynchronous working principle of event cameras, at minimal cost. This "hardware-in-the-loop" annotation paradigm is transferable to any domain requiring high-frequency annotation.
- **Residual decomposition (RGB anchor + event residual)** is an elegant framework for cross-temporal-resolution fusion: RGB provides low-frequency structural priors, while events supply high-frequency motion increments. This decomposition generalizes beyond HPE to high-speed object tracking, high-frequency surface deformation estimation, and related tasks.
- The system's end-to-end completeness is impressive — spanning hardware (LED garment and multi-modal rig), software (annotation pipeline and baseline method), and dataset, forming a closed and coherent loop.

## Limitations & Future Work

- LED markers still require specially designed garments, limiting applicability in naturalistic settings. Future work may explore marker-free approaches that directly estimate high-frequency poses from the high dynamic range of event cameras.
- The 17 LEDs cover only coarse-grained joints and cannot capture fine-grained motions such as finger articulation. Increasing the number of LEDs may introduce frequency conflicts, as the space of unique flashing signatures is finite.
- Current 3D annotations are limited to 60Hz (constrained by IMU and LiDAR), while 1000Hz annotations are restricted to 2D. Future work combining multi-view event cameras could enable 1000Hz 3D annotation.
- The FlashMotion dataset remains limited in scale and scene diversity (20 subjects, 4 scenes); extension to larger populations and more action types (e.g., gymnastics, combat sports) is warranted.
- The SNN-CNN hybrid encoder is relatively simple; more sophisticated event representation learning methods, such as Transformers with fine temporal resolution, may yield further improvements.

## Related Work & Insights

- **vs. BEAHM**: BEAHM is the previously highest frame-rate event-based HPE dataset (120Hz, reconstructed via multi-view calibrated RGB cameras). FlashMotion raises the annotation frame rate 8-fold to 1000Hz, with a more native annotation mechanism that does not depend on the RGB frame-rate ceiling.
- **vs. DHP19**: DHP19 uses a 100Hz Vicon system for ground truth, constrained by Vicon's sampling rate. FlashCap's LED scheme is independent of external MoCap systems and achieves a 10× improvement in temporal resolution.
- **vs. EventCap**: EventCap uses event cameras for HPE but relies on 100Hz marker-free MoCap for ground truth. FlashCap's key innovation is that its annotations are natively event-derived, so temporal resolution is not bounded by any auxiliary system.
- High-speed RGB cameras (e.g., Basler, used for validation) are costly and bandwidth-intensive; FlashCap achieves comparable or superior temporal precision at approximately 1/9th the cost.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The creative combination of LED frequency encoding and event cameras establishes a groundbreaking paradigm for high-frequency motion capture.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ System validation, dataset quality verification, two novel tasks, and comprehensive ablations — exceptionally thorough.
- Writing Quality: ⭐⭐⭐⭐ The narrative progresses logically from system design to dataset to methodology, with clear structure.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for millisecond-level motion capture; the dataset and system offer significant value to the broader HPE community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting](sharp_short-window_streaming_for_accurate_and_robust_prediction_in_motion_foreca.md)
- [\[AAAI 2026\] MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation](../../AAAI2026/autonomous_driving/mambaseg_harnessing_mamba_for_accurate_and_efficient_image-e.md)
- [\[CVPR 2026\] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)
- [\[CVPR 2026\] x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[CVPR 2026\] ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)

<!-- RELATED:END -->
