---
title: >-
  [Paper Note] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking
description: >-
  [AAAI 2026 (Oral)][Autonomous Driving][3D Single Object Tracking] Proposes CompTrack—the first 3D single object tracking framework to address the dual challenges of spatial and informational redundancy in LiDAR point clouds simultaneously: the Spatial Foreground Predictor (SFP) filters background noise based on information entropy, and the Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module estimates the effective rank via online SVD to compress the foregr…
tags:
  - "AAAI 2026 (Oral)"
  - "Autonomous Driving"
  - "3D Single Object Tracking"
  - "Point Cloud"
  - "Token Compression"
  - "Information Bottleneck"
  - "Low-Rank Approximation"
date: 2026-05-08
content_hash: f3ec26e792acba52
---

# CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking

**Conference**: AAAI 2026 (Oral)  
**arXiv**: [2511.15580](https://arxiv.org/abs/2511.15580)  
**Code**: None  
**Area**: Other  
**Keywords**: 3D Single Object Tracking, Point Cloud, Token Compression, Information Bottleneck, Low-Rank Approximation

## TL;DR

Proposes CompTrack—the first 3D single object tracking framework to address the dual challenges of spatial and informational redundancy in LiDAR point clouds simultaneously: the Spatial Foreground Predictor (SFP) filters background noise based on information entropy, and the Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module estimates the effective rank via online SVD to compress the foreground into compact proxy tokens; achieves state-of-the-art performance on nuScenes and Waymo while running in real-time at 90 FPS.

## Background & Motivation

LiDAR-based 3D single object tracking (SOT) is a critical task for autonomous driving and robotics. Due to the inherent sparsity of point clouds, existing methods face a dual redundancy problem but only address half of it: (1) **Spatial Redundancy**—a massive amount of irrelevant background points drowns out the few target features, causing severe signal-to-noise ratio issues and computational waste; (2) **Informational Redundancy**—not all points within the foreground are equally important. Points on large flat surfaces (such as a vehicle hood) provide ambiguous localization cues (analogous to the aperture problem in optical flow), whereas points at corners/edges possess unique structural information. Existing methods (such as P2P and MBPTrack) only handle spatial redundancy, and leaving informational redundancy in the foreground results in low-rank feature matrices and limited localization accuracy.

## Method

### Overall Architecture

CompTrack consists of two stages: (1) a Pillar Encoder converts raw point clouds into BEV feature maps; (2) SFP filters background noise (addressing spatial redundancy); (3) IB-DTC compresses the foreground into compact proxy tokens (addressing informational redundancy); (4) a prediction head directly regresses target parameters $(x, y, z, \theta)$.

### Key Designs

1. **Spatial Foreground Predictor (SFP)**: Proves from an information-theoretic perspective that when the BEV occupancy probability $p \ll 1$, filtering empty/background pillars is theoretically lossless. Specifically, it is implemented as a lightweight CNN (using grouped convolutions) that predicts spatial importance heatmaps from concatenated BEV features of the template and search areas, trained with MSE supervision using a Gaussian circle ground truth (with the peak at the center of the bounding box).
2. **Information Bottleneck-Guided Dynamic Token Compression (IB-DTC)**: The core mechanism reformulates foreground compression as an information bottleneck optimization, with low-rank approximation as a practical surrogate. It utilizes online SVD to analyze the singular value distribution of the foreground feature matrix, dynamically determining the effective rank $K$ based on an energy preservation threshold $\tau$. It then selects the top $K$ queries from a learnable query pool and fuses them with the SVD prior ($Q_{act} = S_K \cdot Q_{learn} + Q_{SVD}$), and finally generates $K$ proxy tokens through cross-attention. Since SVD is only used to decide integer indices rather than backpropagating gradients, the entire module is end-to-end trainable.
3. **Adaptive Mask Training Strategy**: Since $K$ varies dynamically for each sample, the tensor dimensions are kept fixed to a maximum length $L$ during training. For each sample, binary masks are applied to zero out the attention weights of inactive queries after softmax, ensuring that gradients only flow through the adaptively selected $K$ active queries.

### Loss & Training

- Total Loss = $\theta_1 \cdot \mathcal{L}_{pred}$ (MSE of SFP heatmap) + $\theta_2 \cdot \mathcal{L}_{track}$ (tracking regression loss)
- Tracking Loss = $\lambda_1 \cdot \mathcal{L}_{x,y} + \lambda_2 \cdot \mathcal{L}_z + \lambda_3 \cdot \mathcal{L}_{rot}$
- The SVD compression module does not require additional sparsity-inducing regularization, as the compression ratio is directly determined by the intrinsic rank of the data.
- SVD computation takes less than 1 ms (on RTX 3090), representing negligible overhead.

## Key Experimental Results

### Main Results

**KITTI Dataset (Success/Precision):**

| Method | Car | Ped | Van | Cyclist | Mean | FPS |
|------|-----|-----|-----|---------|------|-----|
| P2P (IJCV'25) | 73.6/85.7 | **69.6/94.0** | **70.3/83.9** | 75.5/94.6 | **71.7/89.4** | 65 |
| CompTrack | 73.4/85.2 | 69.5/**94.7** | 68.5/82.5 | **76.0/94.8** | 71.4/89.3 | **90** |
| MBPTrack | 73.4/84.8 | 68.6/93.9 | 61.3/72.7 | 76.7/94.3 | 70.3/87.9 | 50 |
| CXTrack | 69.1/81.6 | 67.0/91.5 | 60.0/71.8 | 74.2/94.3 | 67.5/85.3 | 34 |

### Ablation Study

- Removal of SFP: Large amounts of background noise mix into the foreground region, resulting in a significant decrease in tracking precision.
- Removal of IB-DTC: Redundant foreground tokens are retained, reducing efficiency and slightly degrading accuracy.
- Removal of SVD Prior (using learnable queries only): The compression ratio becomes fixed, and dynamic adaptability is lost.
- Removal of Learnable Queries (using SVD basis only): Lacks task-specific adaptation, leading to decreased precision.
- Energy preservation threshold $\tau$: Performance is optimal within the range of 0.9-0.95.

### Key Findings

- CompTrack performs on par with P2P on KITTI (71.4 vs. 71.7) but runs 1.4 times faster (90 vs. 65 FPS).
- Achieves new SOTA on the large-scale nuScenes and Waymo datasets.
- FLOPs are only 0.94G, which is 76% of P2P.
- The divide-and-conquer strategy for dual redundancy elimination is highly effective: SFP is responsible for rough spatial filtering, while IB-DTC handles fine-grained information refinement.

## Highlights & Insights

- The combination of information bottleneck theory and low-rank approximation provides a rigorous theoretical foundation for token compression, moving beyond heuristic designs.
- The concept of dynamically determining the compression ratio via online SVD is novel—different targets (e.g., compact vehicles vs. complex pedestrians) automatically receive different compression ratios.
- The hybrid approach of SVD prior + learnable queries cleverly bypasses the non-differentiable nature of SVD.
- Understanding point cloud informational redundancy from the perspective of the aperture problem establishes a theoretical bridge between 2D vision and 3D tracking.

## Limitations & Future Work

- It does not outperform P2P on the average metrics of KITTI; its advantages are primarily reflected in efficiency and performance on large-scale datasets.
- The code is not open-source, and reproducibility remains to be verified.
- BEV representation may lose vertical information, presenting limitations in scenarios like high-rise buildings.
- Multi-object tracking and occluded scenarios have not yet been explored.

## Related Work & Insights

- Information bottleneck-guided token compression can be generalized to 2D Vision Transformers (such as token pruning in ViTs).
- The method of estimating the effective rank via online SVD can be applied to any scenario requiring dynamic computation allocation.
- The divide-and-conquer strategy of spatial-informational dual redundancy elimination can be transferred to other sparse data tasks (such as radar and event cameras).
- The analogy to the "aperture problem" in point cloud sparsity provides a valuable theoretical perspective.
- The BEV encoding method of PillarHist balances fine-grained geometric preservation with computational efficiency.

## Core Formulas

- **Information Bottleneck Objective**: $\min I(X_{fg}; X_{proxy}) \quad \text{s.t.} \quad I(X_{proxy}; y) \geq I_0$
- **Low-Rank Approximation Error**: $\|X_{fg} - X_{proxy}\|_F^2 = \sum_{i=k+1}^N \sigma_i^2$ (rapid decay of singular values makes the error negligible)
- **Energy Preservation**: $K = \min\left\{k : \sum_{i=1}^k \sigma_i^2 \geq \tau \cdot \sum_{j=1}^N \sigma_j^2\right\}$
- **Hybrid Query**: $Q_{act} = S_K \cdot Q_{learn} + Q_{SVD}$ (SVD prior + learnable adaptation)

## Efficiency Analysis

| Method | FLOPs | FPS | Device | Mean(KITTI) |
|------|-------|-----|------|-------------|
| CompTrack | 0.94G | 90 | 3090 | 71.4/89.3 |
| P2P | 1.23G | 65 | 3090 | 71.7/89.4 |
| MBPTrack | 2.88G | 50 | 3090 | 70.3/87.9 |
| CXTrack | 4.63G | 34 | 3090 | 67.5/85.3 |

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Information bottleneck + SVD-guided dynamic compression; solid theoretical foundation. |
| Technical Depth | 5 | Complete derivation from information theory to low-rank approximation to end-to-end differentiability. |
| Experimental Thoroughness | 4 | Evaluated on three benchmarks (KITTI/nuScenes/Waymo) with thorough ablation studies. |
| Writing Quality | 4 | Clear theoretical motivation and methodological derivations. |
| Value | 4 | Real-time performance at 90 FPS, directly applicable to autonomous driving. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking](../../ICCV2025/autonomous_driving/trackany3d_transferring_pretrained_3d_models_for_category-unified_3d_point_cloud.md)
- [\[ICLR 2026\] Multi-Head Low-Rank Attention (MLRA)](../../ICLR2026/autonomous_driving/multi-head_low-rank_attention.md)
- [\[AAAI 2026\] Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction](global-lens_transformers_adaptive_token_mixing_for_dynamic_link_prediction.md)
- [\[ICLR 2026\] Low-Latency Neural LiDAR Compression with 2D Context Models](../../ICLR2026/autonomous_driving/low-latency_neural_lidar_compression_with_2d_context_models.md)
- [\[AAAI 2026\] Understanding Dynamic Scenes in Egocentric 4D Point Clouds](understanding_dynamic_scenes_in_ego_centric_4d_point_clouds.md)

</div>

<!-- RELATED:END -->
