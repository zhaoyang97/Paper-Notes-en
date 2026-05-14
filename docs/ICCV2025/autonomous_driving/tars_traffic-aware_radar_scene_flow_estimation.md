---
title: >-
  [Paper Note] TARS: Traffic-Aware Radar Scene Flow Estimation
description: >-
  [ICCV2025][Autonomous Driving][Radar scene flow] This paper proposes TARS, a traffic-aware radar scene flow estimation method that constructs a Traffic Vector Field (TVF) via joint object detection…
tags:
  - "ICCV2025"
  - "Autonomous Driving"
  - "Radar scene flow"
  - "traffic vector field"
  - "point cloud motion estimation"
  - "multi-task learning"
  - "autonomous driving perception"
date: 2026-05-08
content_hash: f8f59bd037023bf1
---

# TARS: Traffic-Aware Radar Scene Flow Estimation

**Conference**: ICCV2025
**arXiv**: [2503.10210](https://arxiv.org/abs/2503.10210)
**Code**: To be confirmed
**Area**: Autonomous Driving
**Keywords**: Radar scene flow, traffic vector field, point cloud motion estimation, multi-task learning, autonomous driving perception

## TL;DR

This paper proposes TARS, a traffic-aware radar scene flow estimation method that constructs a Traffic Vector Field (TVF) via joint object detection, capturing rigid-body motion at the traffic level rather than the instance level. TARS surpasses the state of the art by 15% and 23% on the VOD and a proprietary dataset, respectively.

## Background & Motivation

- Scene flow provides critical motion information for autonomous driving, describing per-point displacement vectors between two consecutive point cloud frames.
- Existing LiDAR scene flow methods exploit an **instance-level rigid-body motion assumption**: the scene is decomposed into multiple rigidly moving objects and a static background.
- However, instance-level approaches are **ill-suited for radar point clouds** for three reasons:
  1. **Extreme sparsity**: Radar point clouds are an order of magnitude sparser than LiDAR (VOD dataset: ~256 points per frame).
  2. **Lack of shape information**: Reliable instance-level matching is infeasible.
  3. **Inter-frame "deformation"**: Sparsity causes significant variation in the point distribution of the same object across consecutive frames.
- Radar advantages: greater robustness to adverse weather and an order-of-magnitude lower cost.
- Core problem: How to preserve the rigid-body motion assumption while accommodating radar sparsity?
- Proposed solution: **Elevate the rigid-body motion assumption from the instance level to the traffic level**.

## Method

### Overall Architecture

TARS adopts a hierarchical architecture with $L$ layers, combining two branches:
1. **Scene Flow Branch**: Hierarchical coarse-to-fine scene flow estimation.
2. **Object Detection (OD) Branch**: Provides feature maps encoding traffic-level contextual information.

Both branches are **jointly trained**; feature maps from the OD branch supply traffic-level priors to the scene flow branch.

### Input and Encoding

- Two input point clouds $P \in \mathbb{R}^{N \times 5}$ and $Q \in \mathbb{R}^{M \times 5}$.
- Each point has 5-dimensional features: x, y, z coordinates + Relative Radial Velocity (RRV) + Radar Cross Section (RCS).
- A multi-scale point encoder (PointNet) extracts features.
- Farthest point sampling downsamples the point sets at each layer, yielding multi-scale point set pairs.

### Two-Level Motion Understanding

#### 1. Point-Level Motion Understanding

- A **dual-attention mechanism** extracts motion cues from neighboring points (replacing unstable MLPs).
- **Cross-attention**: Computed between point $p_i$ and its $K$ nearest neighbors in $Q$, yielding matching embeddings.
- **Self-attention**: Combines matching embeddings with neighborhood information in $P$, producing point-level flow embeddings.
- A coarse flow from the previous layer is used to warp and align points, reducing the search range.
- Unlike HALFlow, direction vectors are removed to alleviate inter-point distance issues, and heterogeneous key/value pairs are adopted.

#### 2. Traffic-Level Scene Understanding

The core innovation: modeling traffic-level motion consistency via a TVF (Traffic Vector Field).

**TVF Definition**: A discrete grid map encoding traffic information of road participants and the environment, where each cell contains a motion vector. A **coarse grid** (e.g., 2 m × 2 m) is used to capture high-level understanding rather than per-point details.

### TVF Encoder

TVF construction proceeds in two stages:

**Scene Update**:
- A GRU updates the TVF across layers, taking OD feature maps (adapted to the TVF shape via CNN and pooling) as input.
- The TVF serves as the GRU hidden state, with OD features as the input.
- The scene representation is progressively refined across hierarchy levels.

**Flow Painting**:
- Flow embeddings and point features from the previous layer are projected onto the coarse grid.
- Since each grid cell may contain multiple points with different motion patterns, **point-to-grid self-attention** is used to adaptively extract motion features.
- Traffic features and motion features are fused via spatial attention.
- **Axial attention** ($\omega$ blocks) provides a global receptive field, modeling rigid-body motion dependencies in traffic (e.g., motion patterns of vehicles in the same lane).

### TVF Decoder

- Perceives latent rigid-body motion within spatial context.
- For each point $p_i$, **grid-to-point cross-attention** is applied by querying the surrounding $\mathcal{N}_{TVF}$ TVF cells.
- The attention receptive field is restricted to the local region of each point, focusing on relevant local rigid-body motion.
- Query: previous-layer flow embedding + point features; Key/Value: TVF grid cells.

### Scene Flow Prediction

- Point-level and traffic-level flow embeddings are concatenated:
$$\textbf{e}^l = \text{Concat}(\textbf{e}_\text{point}, \textbf{e}_\text{traffic}, \text{Interp}(\textbf{e}^{l-1}))$$
- A self-attention layer is applied before predicting the final scene flow $F^l$.

### Temporal Update Module

- A PointGRU layer exploits multi-frame temporal information (distinct from the cross-layer GRU in the TVF encoder).
- Point features at time $t-2$ initialize the hidden state.
- During training, $T$-frame mini-clips are sampled as sequences.

### Loss & Training

Weakly supervised training without scene flow ground truth annotations; a composite loss is used:
1. **Soft Chamfer Loss** $\mathcal{L}_{sc}$: Aligns the warped $P$ with $Q$.
2. **Spatial Smoothness Loss** $\mathcal{L}_{ss}$: Encourages neighboring points to have similar flow vectors.
3. **Radial Displacement Loss** $\mathcal{L}_{rd}$: Constrains the radial flow component using radar RRV measurements.
4. **Foreground Loss** $\mathcal{L}_{fg}$: Uses pseudo ground truth from a LiDAR multi-object tracking model.
5. **Background Loss** $\mathcal{L}_{bg}$: Uses ego-motion transformation as pseudo ground truth for static points ($\lambda = 0.5$).

### Ego-Motion Handling

- **TARS-ego**: An additional ego-motion head is trained (for fair comparison with CMFlow).
- **TARS-superego**: Ego-motion is provided as a known input for compensation (simulating real autonomous driving conditions).

## Key Experimental Results

### VOD Dataset

| Method | EPE↓(m) | AccS↑(%) | AccR↑(%) | RNE↓ | MRNE↓ | SRNE↓ |
|--------|---------|----------|----------|------|-------|-------|
| RaFlow | 0.226 | 19.0 | 39.0 | 0.090 | 0.114 | 0.087 |
| CMFlow (SOTA) | 0.130 | 22.8 | 53.9 | 0.052 | 0.072 | 0.049 |
| **TARS-ego** | **0.092** | **39.0** | **69.1** | **0.037** | **0.061** | **0.034** |
| TARS-superego | 0.048 | 76.6 | 86.4 | 0.019 | 0.057 | 0.014 |

TARS-ego reduces EPE from 0.130 m to 0.092 m (the first method to breach the AccR threshold of 0.1 m), improving AccS and AccR by 16.2% and 15.2%, respectively.

### Proprietary Dataset (High-Resolution Radar)

| Method | MEPE↓(m) | MagE↓ | DirE↓(rad) | AccS↑(%) | AccR↑(%) |
|--------|----------|-------|------------|----------|----------|
| PointPWC-Net+GRU | 0.213 | 0.178 | 0.762 | 49.0 | 60.5 |
| HALFlow+GRU | 0.170 | 0.135 | 0.721 | 50.9 | 63.8 |
| **TARS** | **0.069** | **0.059** | **0.599** | **69.8** | **86.8** |

MEPE is reduced from 0.170 m to 0.069 m (−59%), with AccS and AccR improving by 18.9% and 23.0%, respectively.

### Ablation Study (Proprietary Dataset)

| Configuration | MEPE↓ | AccS↑ | AccR↑ |
|---------------|-------|-------|-------|
| Point-level only | 0.178 | 47.9 | 61.6 |
| + Traffic-level (w/o OD feature map) | 0.144 | 45.0 | 63.3 |
| + OD feature map (fine grid) | 0.104 | 51.4 | 69.9 |
| + Coarse grid (w/o global attention) | 0.074 | 65.6 | 84.2 |
| + Global attention (full TARS) | **0.069** | **69.8** | **86.8** |

Key Findings:
- The coarse grid (2 m × 2 m vs. 1 m × 1 m) is critical for traffic-level understanding.
- Global attention (vs. local convolution) improves AccS by +4.2%.
- $\mathcal{N}_{TVF} = 9$ (surrounding neighborhood) in the TVF decoder achieves the best performance.

### Loss Function Ablation (VOD Dataset)

- The background loss $\mathcal{L}_{bg}$ yields a significant overall improvement: AccR increases from 62.4% to 69.1%.
- Weight $\lambda = 0.5$ achieves a balance between accuracy on moving points and overall accuracy.

## Highlights & Insights

**Strengths**:
- The first work to elevate the rigid-body motion assumption from the instance level to the traffic level, effectively accommodating radar sparsity.
- The coarse-grid design of TVF avoids over-fitting to point-level details.
- Traffic context is obtained via OD branch feature maps (rather than detection outputs), reducing dependence on detection accuracy.
- Substantially outperforms the state of the art on two datasets (15% and 23%).
- Effectively mitigates the inherent limitation of radar in measuring tangential velocity.

**Limitations**:
- Still relies on a LiDAR multi-object tracking model to generate foreground pseudo ground truth (not fully LiDAR-free).
- On the VOD dataset, point clouds are extremely sparse (256 points/frame), making the effect of $\mathcal{N}_{TVF}$ in the TVF decoder less pronounced.
- End-to-end joint optimization of the OD and scene flow branches remains unexplored.

## Personal Reflections

- The framing of "instance-level vs. traffic-level" is precise and directly addresses the core challenge in radar scene flow.
- The TVF design philosophy is inspiring: when data is too sparse for fine-grained matching, raising the level of abstraction is an effective strategy.
- The combination of coarse grid and global attention is well-motivated: the coarse grid preserves high-level semantics, while global attention models lane-level motion correlations.
- Leveraging joint detection feature maps rather than detection results is more robust, as feature maps carry richer information than bounding box outputs.
- The PointGRU temporal module and the cross-layer GRU in the TVF encoder serve distinct roles: the former accumulates temporal information, while the latter updates the scene representation across scales.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[AAAI 2026\] Meta Dynamic Graph for Traffic Flow Prediction](../../AAAI2026/autonomous_driving/meta_dynamic_graph_for_traffic_flow_prediction.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[NeurIPS 2025\] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](../../NeurIPS2025/autonomous_driving/learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)
- [\[ICLR 2026\] x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](../../ICLR2026/autonomous_driving/x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)

</div>

<!-- RELATED:END -->
