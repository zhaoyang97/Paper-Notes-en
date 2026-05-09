---
title: >-
  [Paper Note] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking
description: >-
  [AAAI 2026][Autonomous Driving][Point cloud tracking] CompTrack is proposed as the first framework to simultaneously address dual redundancy in LiDAR point clouds: SFP filters background noise via information entropy analysis to resolve spatial redundancy; IB-DTC estimates effective rank via online SVD and adaptively determines compression ratio to compress foreground into low-rank proxy tokens, resolving information redundancy. Achieves state-of-the-art on nuScenes (61.04% Success) at 90 FPS.
tags:
  - AAAI 2026
  - Autonomous Driving
  - Point cloud tracking
  - spatial redundancy
  - information redundancy
  - information bottleneck
  - SVD
  - low-rank approximation
  - dynamic token compression
date: 2026-05-08
content_hash: d38c33eb5cb942b3
---

# CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking

**Conference**: AAAI 2026
**arXiv**: [2511.15580](https://arxiv.org/abs/2511.15580)
**Area**: 3D Single Object Tracking / Autonomous Driving
**Keywords**: Point cloud tracking, spatial redundancy, information redundancy, information bottleneck, SVD, low-rank approximation, dynamic token compression

## TL;DR

CompTrack is proposed as the first framework to simultaneously address dual redundancy in LiDAR point clouds: SFP filters background noise via information entropy analysis to resolve spatial redundancy; IB-DTC estimates effective rank via online SVD and adaptively determines compression ratio to compress foreground into low-rank proxy tokens, resolving information redundancy. Achieves state-of-the-art on nuScenes (61.04% Success) at 90 FPS.

## Background & Motivation

**State of the Field**: LiDAR-based 3D single object tracking is a fundamental task in autonomous driving, with methods categorized into appearance matching and motion-centric paradigms.

**Limitations of Prior Work**: The inherent sparsity of LiDAR point clouds introduces dual redundancy — (1) **Spatial redundancy**: abundant background points overwhelm sparse target features; (2) **Information redundancy**: points on large flat surfaces in the foreground provide ambiguous localization cues (analogous to the aperture problem in optical flow), while corner points carry structural information.

**Root Cause**: Existing methods primarily address spatial redundancy while entirely ignoring the information redundancy and low-rank structure of foreground feature matrices.

**Starting Point**: Foreground feature matrices are intrinsically low-rank and can be compressed via optimal low-rank approximation (truncated SVD), which naturally corresponds to the information bottleneck principle.

**Core Idea**: Spatial redundancy is removed by an information entropy-guided foreground predictor; information redundancy is resolved by estimating the effective rank via online SVD and compressing via cross-attention with learned queries.

## Method

### Overall Architecture

BEV representation with a two-stage pipeline: Stage 1 SFP filters background → Stage 2 IB-DTC compresses foreground → prediction head outputs $(x,y,z,\theta)$.

### Key Designs

1. **Spatial Foreground Predictor (SFP)**
    - **Function**: Filters spatial redundancy from an information-theoretic perspective
    - **Design Motivation**: When BEV occupancy probability $p \ll 1$, empty pillars carry negligible information, making their removal theoretically lossless
    - **Implementation**: A lightweight CNN produces a spatial importance heatmap, applied element-wise to enhance foreground and suppress background
    - **Supervision**: CenterPoint-style 2D Gaussian heatmap + MSE loss

2. **IB-DTC Module**
    - **Function**: Compresses redundant foreground $\mathbf{X}_{fg} \in \mathbb{R}^{N \times C}$ into proxy tokens $\mathbf{X}_{proxy} \in \mathbb{R}^{K \times C}$ ($K \ll N$)
    - **Mechanism**: Tractable surrogate for the IB objective — optimal low-rank approximation via the Eckart–Young theorem
    - **Three-step implementation**:
        - **Online rank estimation**: Fast non-backpropagated SVD; effective rank $K$ determined by cumulative energy threshold $\tau=0.99$ (average $\approx 78$)
        - **SVD-guided dynamic queries**: $\mathbf{Q}_{act} = \mathbf{S}_K \mathbf{Q}_{learn} + \mathbf{Q}_{SVD}$ (residual learning)
        - **Guided cross-attention**: $\mathbf{X}_p = \text{Softmax}(\frac{\mathbf{Q}_{act} W_q (X'_{fg} W_k)^T}{\sqrt{C}}) X'_{fg} W_v$
    - **Training**: Adaptive masking — tensors padded to fixed maximum length $L$, with only the first $K$ positions contributing to the loss

3. **End-to-End Optimization**
    - $\mathbf{L}_{total} = \theta_1 \mathbf{L}_{pred} + \theta_2 \mathbf{L}_{track}$
    - SVD is used solely to determine integer indices; gradients propagate through learned queries and cross-attention

## Key Experimental Results

### KITTI Comparison

| Method | Mean Success/Precision | FLOPs | FPS |
|--------|----------------------|-------|-----|
| P2P (IJCV'25) | 71.7 / 89.4 | 1.23G | 65 |
| MBPTrack (ICCV'23) | 70.3 / 87.9 | 2.88G | 50 |
| **CompTrack** | **71.4 / 89.3** | **0.94G** | **90** |

### nuScenes SOTA

| Method | Mean Success/Precision |
|--------|----------------------|
| P2P | 59.22 / 71.19 |
| MBPTrack | 57.48 / 69.88 |
| **CompTrack** | **61.04 / 73.68** |

### Waymo Cross-Dataset Generalization

| Method | Mean | Pedestrian |
|--------|------|------------|
| P2P | 47.2 / 62.9 | 37.4 / 58.1 |
| **CompTrack** | **48.6 / 65.7** | **39.0 / 62.7** |

### Ablation Study (nuScenes)

| Config | SFP | IB-DTC | Mean Success | FPS |
|--------|-----|--------|-------------|-----|
| Baseline | ✗ | ✗ | 59.38 | 48 |
| +SFP | ✓ | ✗ | 60.01 | 55 |
| +IB-DTC | ✗ | ✓ | 59.95 | 75 |
| **Full** | **✓** | **✓** | **61.04** | **90** |

### SVD-Guided Query Fusion

| Strategy | Success | Precision |
|----------|---------|-----------|
| Learned query only | 60.70 | 73.25 |
| SVD only | 60.15 | 72.50 |
| **Additive fusion** | **61.04** | **73.68** |

### Key Findings

- SFP and IB-DTC are fully complementary; their combination improves FPS from 48 to 90
- Online SVD introduces less than 1 ms latency
- Performance is stable across energy thresholds in the range 0.99–0.999
- Average effective rank $K \approx 78$, confirming the low-rank nature of foreground features
- FLOPs are 24% lower than P2P with a 38% speed improvement

## Highlights & Insights

1. **Clear dual redundancy decomposition**: The spatial + information redundancy framework is novel and self-consistent; the aperture problem analogy is intuitive
2. **Theoretical connection from IB to low-rank approximation**: Compression is not arbitrary — the IB framework motivates truncated SVD as the theoretically optimal solution
3. **Residual fusion of SVD prior and learned queries**: Simple yet effective, outperforming more complex concatenation schemes
4. **Accuracy improves alongside efficiency**: Redundancy removal not only accelerates inference but also reduces interference

## Limitations & Future Work

1. Performance remains limited in extremely sparse scenarios with partially visible targets
2. Temporal information is not exploited
3. Fusion with RGB data has not been explored
4. Variable $K$ across samples in a batch increases implementation complexity
5. The impact of pillar encoder choice is not thoroughly investigated

## Related Work & Insights

- The "online SVD rank estimation → dynamic compression" paradigm in IB-DTC is generalizable to other feature redundancy scenarios
- The design pattern of low-rank prior + learnable residual queries is broadly applicable
- Information-theoretic analysis of point cloud sparsity provides theoretical grounding for efficiency optimization in 3D perception

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐⭐: The dual redundancy framework and IB-DTC design are highly innovative
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Three benchmarks, 21 SOTA comparisons, multi-dimensional ablations
- **Writing Quality** ⭐⭐⭐⭐: Clear motivation and coherent theoretical derivation
- **Value** ⭐⭐⭐⭐: Win-win on efficiency and accuracy; 90 FPS meets autonomous driving requirements

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Multi-Head Low-Rank Attention (MLRA)](../../ICLR2026/autonomous_driving/multi-head_low-rank_attention.md)
- [\[AAAI 2026\] Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction](global-lens_transformers_adaptive_token_mixing_for_dynamic_link_prediction.md)
- [\[AAAI 2026\] Understanding Dynamic Scenes in Egocentric 4D Point Clouds](understanding_dynamic_scenes_in_ego_centric_4d_point_clouds.md)
- [\[ICCV 2025\] TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking](../../ICCV2025/autonomous_driving/trackany3d_transferring_pretrained_3d_models_for_category-unified_3d_point_cloud.md)
- [\[ICLR 2026\] MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](../../ICLR2026/autonomous_driving/marc_memory-augmented_rl_token_compression_for_efficient_video_un.md)

<!-- RELATED:END -->
