---
title: >-
  [Paper Note] PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking
description: >-
  [AAAI 2026][Video Understanding][Multi-Object Tracking] This paper proposes PlugTrack, a framework that achieves, for the first time, adaptive fusion of Kalman filters and data-driven motion predictors via a Context Moti…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Multi-Object Tracking"
  - "Kalman Filter"
  - "Adaptive Fusion"
  - "Motion Prediction"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: 3aa19425f7ff5aa6
---

# PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking

**Conference**: AAAI 2026
**arXiv**: [2511.13105](https://arxiv.org/abs/2511.13105)
**Code**: [https://github.com/VisualScienceLab-KHU/PlugTrack](https://github.com/VisualScienceLab-KHU/PlugTrack)
**Area**: Video Understanding
**Keywords**: Multi-Object Tracking, Kalman Filter, Adaptive Fusion, Motion Prediction, Plug-and-Play

## TL;DR

This paper proposes PlugTrack, a framework that achieves, for the first time, adaptive fusion of Kalman filters and data-driven motion predictors via a Context Motion Encoder (CME) and an Adaptive Blending factor Generator (ABG), yielding significant improvements under both linear and nonlinear motion scenarios.

## Background & Motivation

### State of the Field
The dominant paradigm in Multi-Object Tracking (MOT) is tracking-by-detection, whose core pipeline consists of: detection → motion prediction → association matching. The motion predictor is the critical component for maintaining target identities.

### Limitations of Prior Work

**Linear assumption of the Kalman filter**: As the standard motion predictor, the Kalman filter is computationally efficient but assumes linear motion, leading to poor performance on datasets with nonlinear motion such as DanceTrack.

**Limitations of data-driven predictors**: Methods such as DiffMOT (diffusion-model-based) and TrackSSM (state-space-model-based) can capture nonlinear dynamics but suffer from domain overfitting and high computational overhead.

**False binary opposition**: The community treats Kalman filters and data-driven methods as mutually exclusive choices, overlooking their complementarity.

### Key Findings (Motivating Experiments)

The authors conduct a per-tracklet performance analysis of predictors on MOT17 and DanceTrack:

- **MOT17** (predominantly linear motion): the Kalman filter outperforms data-driven predictors on **60.3%** of tracklets.
- **DanceTrack** (predominantly nonlinear motion): the Kalman filter still wins on **34%** of tracklets.

This striking finding reveals that **linear motion patterns appear frequently even in datasets specifically designed for complex nonlinear motion**. Real-world tracking scenarios inherently contain a mixture of linear and nonlinear motion patterns, motivating an adaptive unified framework.

### Root Cause
The field lacks a plug-and-play adaptive fusion framework that dynamically decides—based on motion context—whether to trust the Kalman filter or the data-driven predictor, rather than treating them as mutually exclusive alternatives.

## Method

### Overall Architecture

PlugTrack consists of two core components:
1. **Context Motion Encoder (CME)**: analyzes motion patterns from multiple perceptual perspectives to produce multi-perceptive motion features.
2. **Adaptive Blending factor Generator (ABG)**: transforms multi-perceptive features into adaptive blending factors that perform coordinate-wise weighted fusion of the two predictors' outputs.

Final prediction: $\hat{B}_{ABG} = \tilde{\alpha} \odot \hat{B}_{KF} + (1 - \tilde{\alpha}) \odot \hat{B}_{DP}$

### Key Designs

#### 1. **Context Motion Encoder (CME) — Multi-Perceptive Analysis**

CME comprises three specialized modules that analyze motion characteristics from different perspectives:

**(a) Motion Pattern Module (MPM)**: encodes the temporal motion information of a tracklet using an LSTM, capturing complex motion patterns such as acceleration, deceleration, and directional changes:

$$\mathbf{f}_{MPM} = \text{LSTM}(\tilde{\mathbf{T}}_{1:t}) = \mathbf{h}_t \in \mathbb{R}^{128}$$

**(b) Prediction Discrepancy Module (PDM)**: quantifies the prediction discrepancy between the Kalman filter and the data-driven predictor. Large discrepancies typically indicate transitions from linear to nonlinear motion. The discrepancy vector is processed by an MLP:

$$\mathbf{f}_{PDM} = \text{MLP}(\hat{B}_{t+1}^{KF} - \hat{B}_{t+1}^{DP}) \in \mathbb{R}^{32}$$

**(c) Uncertainty Quantification Module (UQM)**: exploits the Normalized Innovation Squared (NIS) of the Kalman filter to quantify its prediction confidence. A high NIS value indicates low confidence in the Kalman filter's prediction, implying nonlinear motion:

$$\text{NIS}_{t,i} = \frac{(B_{t,i} - \hat{B}_{t,i})^2}{S_{t,ii}}$$

The mean and standard deviation of NIS are aggregated via a sliding window to obtain a 4-dimensional uncertainty vector $\sigma_{KF} \in \mathbb{R}^4$, from which features $\mathbf{f}_{UQM} \in \mathbb{R}^{32}$ are extracted by an MLP.

The outputs of the three modules are concatenated and encoded into multi-perceptive motion features: $\mathbf{f}_{mult} = \text{Encoder}(\text{Concat}(\mathbf{f}_{MPM}, \mathbf{f}_{PDM}, \mathbf{f}_{UQM}))$

**Design Motivation**: A single module cannot comprehensively understand motion context. MPM provides temporal patterns, PDM reveals the degree of agreement or disagreement between the two predictors, and UQM supplies a self-reliability assessment of the Kalman filter. The three modules work in concert to achieve "multi-perceptive" understanding.

#### 2. **Adaptive Blending factor Generator (ABG) — Coordinate-Level Fusion**

ABG maps $\mathbf{f}_{mult}$ to a 4-dimensional blending factor $\tilde{\alpha} = (\alpha_x, \alpha_y, \alpha_w, \alpha_h) \in [0,1]$:

$$\hat{B}_{ABG} = \tilde{\alpha} \odot \hat{B}_{KF} + (1 - \tilde{\alpha}) \odot \hat{B}_{DP}$$

**Coordinate-level adaptation example**: When horizontal linear motion occurs (MPM detects a stable horizontal pattern, UQM reports low uncertainty), ABG assigns a high weight to the Kalman filter for the x-coordinate ($\alpha_x > 0.5$); when vertical nonlinear motion occurs (PDM reports large prediction discrepancy), ABG relies on the data-driven predictor for the y-coordinate ($\alpha_y < 0.5$).

#### 3. **Monte Carlo Alpha Search (MCAS) — Training Supervision Signal Generation**

**Problem addressed**: Directly training ABG tends to converge to dataset-specific biases (e.g., always assigning high weight to the Kalman filter on MOT17) rather than learning an adaptive strategy.

**Core method**: A discrete search space $\mathcal{A} = \{0.3, 0.4, 0.5, 0.6, 0.7\}^4$ (625 candidate combinations in total) is defined, and Gaussian noise is added to each training batch for exploration:

$$\tilde{\mathcal{A}}_b = \text{clamp}(\mathcal{A} + \epsilon_b, 0, 1), \quad \epsilon_b \sim \mathcal{N}(0, 0.1^2)$$

Each candidate combination is evaluated by prediction accuracy (SmoothL1 + GIoU), and the optimal combination $\alpha^*$ is selected as the pseudo ground truth for ABG:

$$\mathcal{L}_{MCAS} = \text{MSE}(\tilde{\alpha}, \alpha^*)$$

**MCAS is not used at inference time**; ABG directly predicts the optimal blending factor, preserving real-time efficiency.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{SmoothL1} + \mathcal{L}_{GIoU} + \mathcal{L}_{MCAS}$$

Training uses the Adam optimizer with a learning rate of 0.001, batch size of 2048, and fixed-length tracklets of 5 frames. The model is trained for 220 epochs on DanceTrack and 270 epochs on MIX (MOT17 & MOT20).

## Key Experimental Results

### Main Results (DanceTrack Test Set — Nonlinear Motion)

| Method | Type | HOTA | IDF1 | AssA | DetA | MOTA |
|--------|------|------|------|------|------|------|
| OC-SORT | KF-based | 55.1 | 54.2 | 38.0 | 80.3 | 89.4 |
| C-BIoU | KF-based | 60.6 | 61.6 | 45.4 | 81.3 | 91.6 |
| DiffMOT | Data-driven | 62.3 | 63.0 | 47.2 | 82.5 | 92.8 |
| TrackSSM | Data-driven | 57.7 | 57.5 | 41.0 | 81.5 | 92.2 |
| **Ours (TrackSSM)** | Fusion | 59.2 (+1.5) | 59.0 (+1.5) | 42.9 (+1.9) | 81.9 | 92.2 |
| **Ours (DiffMOT)** | Fusion | **63.3 (+1.0)** | **64.1 (+1.1)** | **48.4 (+1.2)** | 82.5 | 92.4 |

### Ablation Study (DanceTrack Validation Set)

| MPM | PDM | UQM | HOTA | AssA | IDF1 |
|:---:|:---:|:---:|------|------|------|
| ✗ | ✗ | ✗ | 59.2 | 44.5 | 59.7 |
| ✓ | ✗ | ✗ | 60.2 | 45.8 | 61.2 |
| ✓ | ✓ | ✗ | 60.4 | 46.0 | 61.8 |
| ✓ | ✗ | ✓ | 60.3 | 46.1 | 61.4 |
| ✓ | ✓ | ✓ | **60.8** | **46.6** | **61.7** |

**Alpha range analysis**: $[0.3, 0.7]$ achieves the best performance across both base predictors. An overly wide range ($[0.1, 0.9]$) allows extreme values that completely ignore one predictor, while an overly narrow range ($[0.4, 0.6]$) restricts adaptive capacity.

### Key Findings

1. **Strong cross-domain generalization**: HOTA improves by +6.5 when transferring from DanceTrack to MOT20, and still improves by +1.8 in the reverse direction.
2. **Minimal parameter overhead**: only 0.54M additional parameters (a 22% increase over TrackSSM and 4.7% over DiffMOT), while FPS remains above the real-time threshold of 20 (34.2 and 24.7 FPS, respectively).
3. **Qualitative example of coordinate-level adaptation**: at frame 485 of DanceTrack, $\alpha_x=0.874$ (trusting the Kalman filter) and $\alpha_y=0.413$ (trusting DiffMOT), because horizontal motion is linear while vertical motion is nonlinear.

## Highlights & Insights

1. **Compelling core insight**: the paper uses empirical evidence to demonstrate that even on DanceTrack, 34% of tracklets favor the Kalman filter, thereby refuting the assumption that "nonlinear datasets do not need Kalman filters."
2. **Plug-and-play design**: existing motion predictors are not modified; the framework can directly augment any data-driven predictor.
3. **MCAS training strategy**: elegantly resolves the bias collapse problem caused by directly optimizing blending factors.
4. **Coordinate-independent blending factors**: different spatial dimensions can adopt different optimal fusion strategies—this is the first validation of such an approach in MOT.

## Limitations & Future Work

1. Currently fuses only two predictors (Kalman filter + one data-driven predictor); the framework could be extended to multi-way fusion of multiple motion paradigms (e.g., OC-SORT, Hybrid-SORT).
2. The MCAS search space is discrete (625 candidates); continuous optimization may be more efficient.
3. The LSTM used in CME has limited capacity; stronger temporal modeling may yield better motion understanding.
4. Training requires inference results from both predictors to be available in the training data.

## Related Work & Insights

- **SORT / DeepSORT / ByteTrack**: established the foundational tracking-by-detection + Kalman filter paradigm.
- **DiffMOT**: diffusion-model-based motion prediction; one of the base predictors in this work.
- **TrackSSM**: state-space-model-based motion prediction; the other base predictor.
- **Monte Carlo methods**: MCAS is inspired by the successful application of Monte Carlo search in 3D scene understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ — the "bridging classical and modern" approach is both novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — three datasets + cross-domain experiments + efficiency analysis + detailed ablation + qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ — motivation is well-argued and supported by convincing data.
- Value: ⭐⭐⭐⭐ — the plug-and-play framework has direct engineering deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](../../CVPR2026/video_understanding/occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[AAAI 2026\] Learning Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Networks](learning_topology-driven_multi-subspace_fusion_for_grassmannian_deep_network.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](../../CVPR2026/video_understanding/tcei_test_time_calibration_experience_intuition_mot.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](../../CVPR2026/video_understanding/storm_referring_multi_object_tracking.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](../../CVPR2026/video_understanding/fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)

</div>

<!-- RELATED:END -->
