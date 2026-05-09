---
title: >-
  [Paper Note] NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping
description: >-
  [ICLR 2026][Autonomous Driving][dynamic map] NeMo-map is proposed as a continuous spatio-temporal dynamic map based on neural implicit functions, directly mapping spatial-temporal coordinates to Semi-Wrapped Gaussian Mixture Model (SWGMM) parameters. It eliminates the spatial discretization and temporal segmentation constraints of conventional methods, achieving lower NLL and smoother velocity distributions on real pedestrian tracking data.
tags:
  - ICLR 2026
  - Autonomous Driving
  - dynamic map
  - neural implicit representation
  - semi-wrapped Gaussian mixture
  - human motion patterns
  - spatio-temporal continuity
date: 2026-05-08
content_hash: a992570993541419
---

# NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping

**Conference**: ICLR 2026
**arXiv**: [2510.14827](https://arxiv.org/abs/2510.14827)
**Code**: Unavailable
**Area**: Autonomous Driving
**Keywords**: dynamic map, neural implicit representation, semi-wrapped Gaussian mixture, human motion patterns, spatio-temporal continuity

## TL;DR
NeMo-map is proposed as a continuous spatio-temporal dynamic map based on neural implicit functions, directly mapping spatial-temporal coordinates to Semi-Wrapped Gaussian Mixture Model (SWGMM) parameters. It eliminates the spatial discretization and temporal segmentation constraints of conventional methods, achieving lower NLL and smoother velocity distributions on real pedestrian tracking data.

## Background & Motivation

**Background**: Motion-on-demand (MoD) maps encode statistical motion patterns in the environment to assist robots navigating crowded scenes. Existing methods such as CLiFF-map and STeF-map fit local motion distributions on discrete grids.

**Limitations of Prior Work**: Grid discretization leads to information loss and boundary discontinuities; time is typically segmented by hour, precluding smooth modeling of cross-period transitions; manual selection of grid resolution is environment-dependent.

**Key Challenge**: Discrete representations cannot query motion distributions at arbitrary spatio-temporal coordinates, and sparse regions require interpolation or imputation.

**Goal**: (a) Eliminate spatial discretization; (b) enable continuous and smooth querying over both space and time; (c) preserve the multimodal nature of motion directions.

**Key Insight**: Model the mapping $(x, y, t) \to$ SWGMM parameters as a continuous function via neural implicit representations.

**Core Idea**: Use a learnable spatial feature grid + SIREN temporal encoding + MLP to directly output continuous spatio-temporal motion distribution parameters.

## Method

### Overall Architecture
The input is a spatio-temporal coordinate $(\mathbf{x}, t)$, and the output is $J$ sets of SWGMM parameters $\{w_j, \bm{\mu}_j, \bm{\Sigma}_j\}$. Spatial coordinates are bilinearly interpolated over a learnable grid $\mathbf{G}_s$ to obtain spatial features $\mathbf{f}_s(\mathbf{x})$; time is encoded into $\mathbf{f}_t(t)$ via a SIREN network. The concatenated features are fed into an MLP to produce mixture model parameters.

### Key Designs

1. **Learnable Spatial Feature Grid**:

    - Function: Provides local feature encoding for each spatial location.
    - Mechanism: Maintains a feature grid $\mathbf{G}_s \in \mathbb{R}^{H \times W \times C_s}$ and performs bilinear interpolation at query position $\mathbf{x}$ to yield continuous spatial features.
    - Design Motivation: Captures local motion pattern variations more effectively than a pure coordinate-input MLP while preserving spatial continuity.

2. **SIREN Temporal Encoding**:

    - Function: Encodes continuous time $t$ into a feature vector.
    - Mechanism: Employs a network with periodic sinusoidal activations, inherently suited to encoding the periodic variation of motion patterns throughout the day.
    - Design Motivation: Human motion patterns exhibit daily periodicity; the periodic activations of SIREN naturally capture such regularities.

3. **SWGMM Parametric Output**:

    - Function: The MLP outputs mixture weights, means, and covariance matrices.
    - Mechanism: Each component models the joint distribution of speed $\rho$ and direction $\theta$; the directional dimension is wrapped with period $2\pi$ (winding number $k \in \{-1,0,1\}$), permitting speed–direction correlations.
    - Design Motivation: More accurate than the discretized directional histograms of STeF-map and more flexible than the independence assumption of VMGMM.

### Loss & Training
Negative log-likelihood loss: $\mathcal{L}(\theta) = -\frac{1}{N}\sum_i \log p(\mathbf{v}_i | \Phi_\theta(\mathbf{x}_i, t_i))$

## Key Experimental Results

### Main Results (ATC Shopping Mall Dataset, NLL↓)

| Method | NLL↓ | NLL Gap vs. NeMo |
|------|------|-----------------|
| **NeMo-map** | **0.775** | — |
| Online CLiFF-map | 1.527 | +0.752 |
| CLiFF-map | 1.964 | +1.189 |
| STeF-map | 5.576 | +4.801 |

### ETH/UCY Dataset Comparison

| Scene | NeMo NLL | CLiFF NLL | Gain |
|------|----------|-----------|------|
| ETH | -0.384 | 0.112 | +0.496 |
| HOTEL | -0.838 | 0.701 | +1.539 |
| UNIV | 0.404 | 0.518 | +0.114 |
| ZARA | -0.342 | 0.068 | +0.410 |

Training efficiency: NeMo-map trains on a full day of data in under 20 minutes.

### Key Findings
- NeMo significantly outperforms all baselines across all datasets and scenes (p<0.001).
- In sparse regions, NeMo produces smoother velocity distributions, avoiding the discontinuities of discrete methods.
- The model also performs better on downstream trajectory prediction tasks.

## Highlights & Insights
- **Continuous spatio-temporal querying** eliminates the core limitations of MoD: no predefined grid resolution is required, and temporal segmentation discontinuities are removed.
- The cylindrical visualization of SWGMM (direction wrapped around the circle, speed along the cylinder axis) is highly intuitive and facilitates understanding of multimodal motion patterns.

## Limitations & Future Work
- Validation is limited to pedestrian scenarios; other dynamic agents such as vehicles or cyclists have not been tested.
- The resolution of the learnable spatial grid still requires manual specification.
- No comprehensive comparison with deep learning–based trajectory prediction models is provided.

## Related Work & Insights
- **vs. CLiFF-map**: CLiFF discretizes space and relies on offline batch processing; NeMo operates in continuous space with end-to-end training.
- **vs. STeF-map**: STeF discretizes direction (8-bin histogram) and does not model speed; NeMo jointly models continuous direction and speed.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing neural implicit representations into dynamic maps is a natural yet effective innovation.
- Experimental Thoroughness: ⭐⭐⭐ Two datasets with statistical significance testing, but scene diversity is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear and concise, with rigorous mathematical description of SWGMM.
- Value: ⭐⭐⭐⭐ Offers practical contributions to motion modeling for robot navigation.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [\[AAAI 2026\] I-INR: Iterative Implicit Neural Representations](../../AAAI2026/autonomous_driving/i-inr_iterative_implicit_neural_representations.md)
- [\[AAAI 2026\] Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception](../../AAAI2026/autonomous_driving/rethinking_the_spatio-temporal_alignment_of_end-to-end_3d_perception.md)
- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../CVPR2026/autonomous_driving/sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[AAAI 2026\] RAST: A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction](../../AAAI2026/autonomous_driving/rast_a_retrieval_augmented_spatio-temporal_framework_for_traffic_prediction.md)

<!-- RELATED:END -->
