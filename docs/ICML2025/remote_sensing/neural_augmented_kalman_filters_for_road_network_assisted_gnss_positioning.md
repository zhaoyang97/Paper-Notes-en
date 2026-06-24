---
title: >-
  [Paper Note] Neural Augmented Kalman Filters for Road Network Assisted GNSS Positioning
description: >-
  [ICML 2025][Remote Sensing][GNSS positioning] A Temporal Graph Neural Network (TGNN) is proposed to integrate open-source road network information into GNSS Kalman filtering. The TGNN predicts the most likely road segments on the graph structure and dynamically estimates their uncertainties, reducing the P95 localization error from 77.23m to 55.02m (a 29% reduction) in real-world urban data.
tags:
  - "ICML 2025"
  - "Remote Sensing"
  - "GNSS positioning"
  - "Kalman filter"
  - "road network"
  - "temporal graph neural network"
  - "urban localization"
date: 2026-05-08
content_hash: 3550c1054945c93d
---

# Neural Augmented Kalman Filters for Road Network Assisted GNSS Positioning

**Conference**: ICML 2025  
**arXiv**: [2507.00654](https://arxiv.org/abs/2507.00654)  
**Code**: To be confirmed  
**Area**: Remote Sensing  
**Keywords**: GNSS positioning, Kalman filter, road network, temporal graph neural network, urban localization

## TL;DR

A Temporal Graph Neural Network (TGNN) is proposed to integrate open-source road network information into GNSS Kalman filtering. The TGNN predicts the most likely road segments on the graph structure and dynamically estimates their uncertainties, reducing the P95 localization error from 77.23m to 55.02m (a 29% reduction) in real-world urban data.

## Background & Motivation

**Background**: GNSS provides global positioning capability. Out of approximately 5.6 billion receivers in 2023, 10% are used for road transport. However, in dense urban areas, reflections from tall landmarks (multipath effects) and blockage (non-line-of-sight errors) result in positioning errors of up to 20 meters, which struggle to fall below 10 meters even when using 3D models. Lane-level applications (navigation, delivery) require 2-10 meter precision.

**Limitations of Prior Work**: Additional sensors (IMU/LiDAR) are expensive or unavailable. Open-source road networks like OpenStreetMap offer a free global alternative, but their observations are inherently multimodal Gaussians (multiple candidate roads) and cannot be directly integrated into a single-hypothesis KF. Existing solutions typically use heuristics or HMM+Viterbi to select a single road segment before constructing Gaussian observations; however, (1) Viterbi only uses historical information without looking ahead, causing incorrect road selection in noisy scenarios; (2) the standard deviation of road observations is fixed heuristically, failing to adapt to varying scenarios.

**Key Challenge**: Selecting the correct road segment and dynamically adjusting weights based on confidence is required, but existing methods lack flexibility in both selection and weight allocation.

**Goal**: (1) Replace heuristics/Viterbi with learnable road segment selection; (2) dynamically predict road observation uncertainty to achieve adaptive KF updates.

**Key Insight**: Bidirectional Viterbi (Oracle) can leverage future information to make better choices, but is unavailable online. A TGNN can be trained to approximate the Oracle quality using only historical information.

**Core Idea**: Use a TGNN to learn joint spatio-temporal reasoning over the road graph, predicting road segment probabilities and uncertainties to enhance KF positioning in an end-to-end manner.

## Method

### Overall Architecture

At each time step: (1) GNSS pseudoranges $\to$ KF prediction and GNSS update are performed to obtain $\mathbf{x}_{\text{GNSS}}^+$; (2) a road subgraph centered on the current position with a 50m radius is extracted $\to$ TGNN predicts the probability and standard deviation of each segment; (3) the most likely road segment is selected, and its predicted standard deviation is used to construct a Gaussian observation $\to$ KF road measurement update yields the final $\mathbf{x}_{\text{RN}}^+$.

The KF state has 8 dimensions: 3D position + 3D velocity + clock bias/drift. Road measurement update: $\mathbf{x}_{\text{RN}}^+ = \mathbf{x}_{\text{GNSS}}^+ + \mathbf{K}(\mathbf{z} - \mathbf{H}\mathbf{x}_{\text{GNSS}}^+)$, where $\mathbf{z}$ is the position and heading of the selected road segment, and $\mathbf{V}$ represents the covariance predicted by the TGNN.

### Key Designs

1. **TGNN Architecture**:

    - **Function**: Simultaneously process road graph spatial topology and user trajectory temporal information.
    - **Mechanism**: Process user-level and road-level features via dual paths through $L$ repeating blocks: (a) Feature transformation – projected separately by two MLPs; (b) Local message passing – LSTM captures trajectory temporal dynamics ($x \to x$), while GCN propagates neighborhood info on the road graph ($r \to r$); (c) Cross message passing – user features interact with mean-pooled road features ($x, r \to x$), and user features are concatenated and interact with each segment ($r, x \to r$). Finally, a linear layer + softmax is used to obtain road probabilities.
    - **Design Motivation**: GCN considers graph connectivity (neighboring road contexts), LSTM leverages historical trajectories (velocity/heading changes), and cross-message passing enables multi-scale fusion of the two types of information.

2. **Standard Deviation Prediction Head**:

    - **Function**: Dynamically estimate road observation uncertainty.
    - **Mechanism**: Add a linear projection head to the user features of the final TGNN layer to output parallel $\sigma_\parallel^2$ and perpendicular $\sigma_\perp^2$ road variances, with exp applied to guarantee positive values. When uncertain, it outputs a large variance $\to$ KF relies more on GNSS; when confident, it outputs a small variance $\to$ road weight increases.
    - **Design Motivation**: Incorporating road information with a fixed standard deviation across all scenarios is overly rigid. End-to-end optimization of the MSE loss enables the network to learn scenario-adaptive behaviors.

3. **Oracle Supervision and Training Objectives**:

    - **Function**: Provide high-quality training labels for the TGNN.
    - **Mechanism**: Bidirectional Viterbi (utilizing past + future positions) serves as the Oracle to obtain the optimal road sequence. The training loss consists of two parts: (a) Cross-entropy $L_{\text{CE}} = \text{CE}(r_{\text{oracle}}^*, P_\phi(\mathbf{r}))$ — learning to mimic the Oracle; (b) MSE $L_{\text{MSE}} = \text{MSE}(\mathbf{x}_{\text{gt}}, \mathbf{x}_{\text{RN}}^+)$ — end-to-end optimization of the localization error from the KF output. The total loss is $L = L_{\text{CE}} + \lambda L_{\text{MSE}}$.
    - **Design Motivation**: Since the KF is fully differentiable, the standard deviation prediction head can be optimized directly via backpropagation of the KF output error.

### Input Features

Road segment: Endpoint coordinates, segment length, number of lanes, speed limit, road type, one-way markers. User: KF state mean/uncertainty, Viterbi probability prior. Missing information is filled with default values.

## Key Experimental Results

### Localization Error Comparison (Real-world data from 4 cities, 3-fold CV, Table 1)

| Method | Road Selection | Standard Deviation | HE@50th (m)↓ | HE@95th (m)↓ |
|------|---------|-------|-------------|-------------|
| Least Squares (LS) | — | — | 20.43 | 115.97 |
| GNSS-only KF | — | — | 10.75 | 77.23 |
| KF + Instant | Nearest Distance | Grid Search | 7.96 | 68.86 |
| KF + Viterbi | HMM | Grid Search | 8.02 | 68.27 |
| **KF + TGNN** | **TGNN** | **TGNN** | **8.74±0.15** | **55.02±2.21** |
| KF + Oracle | Bidirectional Viterbi | 0 | 3.72 | 11.40 |

### Ablation Study (Table 2-3)

| Ablation Case | HE@50th (m) | HE@95th (m) |
|--------|-------------|-------------|
| TGNN + TGNN Std. Dev. | 8.74±0.15 | **55.02±2.21** |
| Viterbi + TGNN Std. Dev. | 8.69±0.18 | 67.08±3.49 |
| TGNN + Grid Search Std. Dev. | 8.36±0.38 | 63.72±9.00 |
| MLP (No GCN, No LSTM) | 9.13±0.31 | 59.68±3.94 |
| GNN (No LSTM) | 8.82±0.17 | 56.90±2.83 |

### Key Findings

1. TGNN reduces the P95 error from 77.23m to 55.02m (a **29% reduction**), an additional 13m reduction compared to Viterbi.
2. **The standard deviation prediction head makes a critical contribution**: replacing it with grid search degrades the P95 from 55.02 to 63.72 (+8.7m).
3. LSTM contributes approximately 2m (GNN $\to$ TGNN), and GCN contributes about 3m (MLP $\to$ GNN).
4. Sensitivity to receptive field: too small (<30m) misses the correct road, while too large (>80m) leads to excess candidates; the optimal size is around 50m.
5. The standard dev of 10 random initializations is only $\pm$0.15/$\pm$2.21m, indicating method robustness.
6. P50 of TGNN (8.74) is slightly inferior to Viterbi (8.02), but P95 is significantly ahead—bringing the greatest benefit in challenging scenarios.

## Highlights & Insights

- **End-to-End Differentiable KF**: The standard deviation prediction head directly backpropagates errors via the KF output positioning error, allowing the network to learn "when to trust road information".
- **Clever Oracle Supervision**: Employs bidirectional Viterbi (computable at the data level) as training labels, avoiding the need for additional annotations.
- **Global Applicability**: Only relies on OpenStreetMap without needing proprietary commercial maps or extra sensors.
- **Engineering Characteristics Maintained**: The KF framework preserves real-time operational capacity, interpretability, and extensible fusion with other sensors.

## Limitations & Future Work

- Limited data scale — evaluated on only 4 cities; cross-city/cross-country generalization is not fully verified.
- Employs only road centerlines; the lack of lane-level information leads to systematic offsets in parallel multi-lane scenarios.
- No integrated IMU – incorrect road selection under extremely poor GNSS signals can lead to continuous KF drift.
- Inference latency is not quantified.
- Does not handle multi-layer road structures (e.g., overlapping elevated roads and ground lanes).

## Related Work & Insights

- Revach et al. (2022) KalmanNet: Learns optimal Kalman gains but does not handle road networks.
- Jalalirad et al. (2023): Uses GNNs to estimate pseudorange errors; this paper utilizes their dataset.
- Map matching (Hu et al., 2023): Post-processing to determine road sequences, without providing real-time location improvements.
- Inspiration: The TGNN+KF paradigm can be generalized to other geographic-prior-assisted positioning tasks (such as building height maps, terrain models, etc.).

## Rating

⭐⭐⭐⭐ — The first end-to-end scheme combining deep learning, road networks, and GNSS KF. The standard deviation prediction head is simple yet effective. The 29% improvement in P95 difficult scenarios is of engineering significance. The main drawbacks are the small data scale and the lack of IMU integration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Information-Bottleneck Driven Binary Neural Network for Change Detection](../../ICCV2025/remote_sensing/information-bottleneck_driven_binary_neural_network_for_change_detection.md)
- [\[CVPR 2026\] Beyond Endpoints: Path-Centric Reasoning for Vectorized Off-Road Network Extraction](../../CVPR2026/remote_sensing/beyond_endpoints_path-centric_reasoning_for_vectorized_off-road_network_extracti.md)
- [\[CVPR 2026\] LNEM: Lunar Neural Elevation Model](../../CVPR2026/remote_sensing/lnem_lunar_neural_elevation_model.md)
- [\[CVPR 2026\] Spectrally Distilled Representations Aligned with Instruction-Augmented LLMs for Satellite Imagery](../../CVPR2026/remote_sensing/spectrally_distilled_representations_aligned_with_instruction-augmented_llms_for.md)
- [\[CVPR 2026\] RoadGIE: Towards A Global-Scale Aerial Benchmark for Generalizable Interactive Road Extraction](../../CVPR2026/remote_sensing/roadgie_towards_a_global-scale_aerial_benchmark_for_generalizable_interactive_ro.md)

</div>

<!-- RELATED:END -->
