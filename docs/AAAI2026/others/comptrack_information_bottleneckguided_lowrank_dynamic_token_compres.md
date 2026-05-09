---
title: >-
  [Paper Note] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking
description: >-
  [AAAI 2026 (Oral)][3D single object tracking] This paper proposes CompTrack—the first 3D single object tracking framework that simultaneously addresses both spatial redundancy and information redundancy in LiDAR point clouds. A Spatial Foreground Predictor (SFP) filters background noise via information entropy, while an Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module estimates effective rank via online SVD and compresses foreground tokens into compact proxy tokens. CompTrack achieves state-of-the-art performance on nuScenes and Waymo while running in real time at 90 FPS.
tags:
  - AAAI 2026 (Oral)
  - 3D single object tracking
  - point cloud
  - token compression
  - information bottleneck
  - low-rank approximation
date: 2026-05-08
content_hash: a555b1ba9f014469
---

# CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking

**Conference**: AAAI 2026 (Oral)
**arXiv**: [2511.15580](https://arxiv.org/abs/2511.15580)
**Code**: Unavailable
**Area**: Other
**Keywords**: 3D single object tracking, point cloud, token compression, information bottleneck, low-rank approximation

## TL;DR

This paper proposes CompTrack—the first 3D single object tracking framework that simultaneously addresses both spatial redundancy and information redundancy in LiDAR point clouds. A Spatial Foreground Predictor (SFP) filters background noise via information entropy, while an Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module estimates effective rank via online SVD and compresses foreground tokens into compact proxy tokens. CompTrack achieves state-of-the-art performance on nuScenes and Waymo while running in real time at 90 FPS.

## Background & Motivation

LiDAR-based 3D single object tracking (SOT) is a critical task in autonomous driving and robotics. Due to the inherent sparsity of point clouds, existing methods face a dual redundancy problem, yet address only half of it: (1) **Spatial redundancy**—a large number of irrelevant background points overwhelm the sparse target features, causing severe signal-to-noise ratio issues and computational waste; (2) **Information redundancy**—not all foreground points are equally informative. Points on large flat surfaces such as vehicle hoods provide ambiguous localization cues (analogous to the aperture problem in optical flow), whereas points at corners and edges carry unique structural information. Existing methods (e.g., P2P, MBPTrack) address only spatial redundancy, while information redundancy within the foreground leads to low-rank feature matrices and limited localization accuracy.

## Method

### Overall Architecture

CompTrack consists of two stages: (1) a pillar encoder converts raw point clouds into BEV feature maps; (2) SFP filters background noise (addressing spatial redundancy); (3) IB-DTC compresses foreground tokens into compact proxy tokens (addressing information redundancy); (4) a prediction head directly regresses target parameters $(x, y, z, \theta)$.

### Key Designs

1. **Spatial Foreground Predictor (SFP)**: From an information-theoretic perspective, the paper proves that when the occupancy probability $p \ll 1$ in BEV, filtering empty/background pillars is theoretically lossless. The SFP is implemented as a lightweight CNN (using grouped convolutions) that predicts a spatial importance heatmap from the concatenated BEV features of the template and search region, supervised with Gaussian circle ground truth (peaked at the ground-truth box center) via MSE loss.
2. **Information Bottleneck-Guided Dynamic Token Compression (IB-DTC)**: The core idea is to formalize foreground compression as an information bottleneck optimization problem, with low-rank approximation as a practical surrogate. Online SVD is applied to analyze the singular value distribution of the foreground feature matrix; the effective rank $K$ is dynamically determined by an energy retention threshold $\tau$. The top-$K$ queries are then selected from a learnable query pool and fused with the SVD prior as $Q_\text{act} = S_K \cdot Q_\text{learn} + Q_\text{SVD}$, and $K$ proxy tokens are generated via cross-attention. Since SVD is used only to determine an integer index rather than to back-propagate gradients, the entire module remains end-to-end trainable.
3. **Adaptive Mask Training Strategy**: Since $K$ varies dynamically per sample, the tensor dimension is fixed to a maximum length $L$ during training. Binary masks force the attention weights of inactive queries to zero after softmax, so gradients flow only through the adaptively selected $K$ active queries.

### Loss & Training

- Total loss $= \theta_1 \cdot L_\text{pred}$ (MSE on SFP heatmap) $+ \theta_2 \cdot L_\text{track}$ (tracking regression loss)
- Tracking loss $= \lambda_1 \cdot L_{(x,y)} + \lambda_2 \cdot L_z + \lambda_3 \cdot L_\text{rot}$
- The SVD compression module requires no additional sparsity regularization; the compression ratio is determined entirely by the intrinsic rank of the data.
- SVD computation takes less than 1 ms on an RTX 3090, introducing negligible overhead.

## Key Experimental Results

### Main Results

**KITTI dataset (Success/Precision):**

| Method | Car | Ped | Van | Cyclist | Mean | FPS |
|--------|-----|-----|-----|---------|------|-----|
| P2P (IJCV'25) | 73.6/85.7 | **69.6/94.0** | **70.3/83.9** | 75.5/94.6 | **71.7/89.4** | 65 |
| CompTrack | 73.4/85.2 | 69.5/**94.7** | 68.5/82.5 | **76.0/94.8** | 71.4/89.3 | **90** |
| MBPTrack | 73.4/84.8 | 68.6/93.9 | 61.3/72.7 | 76.7/94.3 | 70.3/87.9 | 50 |
| CXTrack | 69.1/81.6 | 67.0/91.5 | 60.0/71.8 | 74.2/94.3 | 67.5/85.3 | 34 |

### Ablation Study

- Removing SFP: background noise contaminates foreground regions, significantly degrading tracking accuracy.
- Removing IB-DTC: redundant foreground tokens are retained, reducing efficiency with a slight accuracy drop.
- Removing SVD prior (learnable queries only): compression ratio becomes fixed and dynamic adaptability is lost.
- Removing learnable queries (SVD basis only): task-specific adaptation is absent, leading to accuracy degradation.
- Energy retention threshold $\tau$: optimal performance is achieved in the range of 0.9–0.95.

### Key Findings

- CompTrack matches P2P on KITTI (71.4 vs. 71.7) while running 1.4× faster (90 vs. 65 FPS).
- CompTrack achieves new state-of-the-art results on the large-scale nuScenes and Waymo benchmarks.
- FLOPs are only 0.94G, equivalent to 76% of P2P.
- The divide-and-conquer strategy for dual redundancy elimination is effective: SFP handles coarse spatial filtering while IB-DTC refines information content.

## Highlights & Insights

- The combination of information bottleneck theory and low-rank approximation provides a rigorous theoretical foundation for token compression, as opposed to a heuristic design.
- Dynamically determining the compression ratio via online SVD is a novel approach—different targets (e.g., compact vehicles vs. complex pedestrians) automatically receive different compression rates.
- The hybrid design of SVD prior and learnable queries elegantly circumvents the non-differentiability of SVD.
- Interpreting point cloud information redundancy through the lens of the aperture problem establishes a theoretical bridge between 2D vision and 3D tracking.

## Limitations & Future Work

- CompTrack does not surpass P2P's mean metric on KITTI; its advantages are primarily reflected in efficiency and large-scale datasets.
- The code is not open-sourced, and reproducibility remains to be verified.
- BEV representation may lose information along the vertical axis, limiting applicability to scenes with tall structures.
- Multi-object tracking and occlusion scenarios remain unexplored.

## Related Work & Insights

- Information bottleneck-guided token compression is generalizable to 2D vision Transformers (e.g., token pruning in ViT).
- The online SVD method for estimating effective rank is applicable to any scenario requiring dynamic computation allocation.
- The divide-and-conquer approach to spatial-information dual redundancy elimination is transferable to other sparse data tasks (e.g., radar, event cameras).
- The aperture problem analogy for point cloud sparsity offers a valuable theoretical perspective.
- PillarHist's BEV encoding balances fine-grained geometric preservation with computational efficiency.

## Core Equations

- **Information bottleneck objective**: $\min\ I(X_{fg}; X_{proxy})\ \text{s.t.}\ I(X_{proxy}; y) \geq I_0$
- **Low-rank approximation error**: $\|X_{fg} - X_{proxy}\|_F^2 = \sum_{i=K+1}^{N} \sigma_i^2$ (rapid singular value decay renders the error negligible)
- **Energy retention**: $K = \min\!\left\{k : \sum_{i=1}^{k} \sigma_i^2 \geq \tau \cdot \sum_{j=1}^{N} \sigma_j^2\right\}$
- **Hybrid query**: $Q_\text{act} = S_K \cdot Q_\text{learn} + Q_\text{SVD}$ (SVD prior + learnable adaptation)

## Efficiency Analysis

| Method | FLOPs | FPS | Device | Mean (KITTI) |
|--------|-------|-----|--------|--------------|
| CompTrack | 0.94G | 90 | 3090 | 71.4/89.3 |
| P2P | 1.23G | 65 | 3090 | 71.7/89.4 |
| MBPTrack | 2.88G | 50 | 3090 | 70.3/87.9 |
| CXTrack | 4.63G | 34 | 3090 | 67.5/85.3 |

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | Information bottleneck + SVD-guided dynamic compression with solid theoretical grounding |
| Technical Depth | 5 | Complete derivation from information theory to low-rank approximation to end-to-end differentiability |
| Experimental Thoroughness | 4 | Three benchmarks (KITTI/nuScenes/Waymo) with comprehensive ablation studies |
| Writing Quality | 4 | Theoretical motivation and methodological derivation are clearly presented |
| Value | 4 | 90 FPS real-time performance, directly applicable to autonomous driving |

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](../../NeurIPS2025/others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[AAAI 2026\] ShortageSim: Simulating Drug Shortages under Information Asymmetry](shortagesim_simulating_drug_shortages_under_information_asymmetry.md)
- [\[AAAI 2026\] Scalable Vision-Guided Crop Yield Estimation](scalable_vision-guided_crop_yield_estimation.md)

<!-- RELATED:END -->
