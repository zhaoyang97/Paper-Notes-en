---
title: >-
  [Paper Note] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection
description: >-
  [ICCV 2025][Autonomous Driving][3D lane detection] This paper proposes SparseLaneSTP, which integrates lane geometry priors (parallelism, continuity) and temporal information into a sparse Transformer architecture. Through Catmull-Rom spline representation, spatio-temporal attention mechanisms, and temporal regularization, the method achieves state-of-the-art performance on multiple 3D lane detection benchmarks.
tags:
  - ICCV 2025
  - Autonomous Driving
  - 3D lane detection
  - sparse Transformer
  - spatio-temporal priors
  - Catmull-Rom spline
  - temporal fusion
date: 2026-05-08
content_hash: 77f6d75333440390
---

# SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection

**Conference**: ICCV 2025
**arXiv**: [2601.04968](https://arxiv.org/abs/2601.04968)
**Code**: N/A
**Area**: Autonomous Driving
**Keywords**: 3D lane detection, sparse Transformer, spatio-temporal priors, Catmull-Rom spline, temporal fusion

## TL;DR

This paper proposes SparseLaneSTP, which integrates lane geometry priors (parallelism, continuity) and temporal information into a sparse Transformer architecture. Through Catmull-Rom spline representation, spatio-temporal attention mechanisms, and temporal regularization, the method achieves state-of-the-art performance on multiple 3D lane detection benchmarks.

## Background & Motivation

3D lane detection is a core task in autonomous driving, requiring the direct output of 3D lane markings in the vehicle coordinate system. Existing methods suffer from the following limitations:

**Inherent drawbacks of BEV-based methods**: Conventional approaches first transform front-view (FV) features into a bird's-eye view (BEV) representation before performing lane estimation. Whether using IPM (which assumes a flat ground plane) or learned mappings, the BEV transformation introduces errors that cause feature-to-surface misalignment, which is difficult to compensate for in subsequent stages.

**Sparse methods neglect priors**: Recent DETR-based sparse detection methods (e.g., LATR) avoid BEV representation errors but entirely ignore geometric priors of lane lines (e.g., parallelism, continuity). These priors have been shown effective in BEV methods such as LaneCPP, yet have not been successfully adapted to sparse architectures.

**Temporal information remains underutilized**: Lane markings are static, and observations from historical frames hold significant potential for disambiguation in scenarios involving occlusion or poor visibility. Nevertheless, existing methods have largely failed to exploit temporal fusion effectively.

The authors' core insight is that the query design in sparse Transformers is naturally suited to an object-centric temporal propagation paradigm (e.g., StreamPETR), and the static nature of lane markings makes temporal consistency regularization feasible.

## Method

### Overall Architecture

SparseLaneSTP takes a single-frame RGB image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ as input and outputs $N$ 3D lane lines. The pipeline consists of: CNN backbone for feature extraction → lane instance segmentation for query initialization → Transformer decoder (with spatio-temporal attention STA + deformable cross-attention DCA) → prediction head outputting spline control points and classification probabilities.

### Key Designs

1. **Catmull-Rom (CR) Spline Continuous Representation**:

    - **Mechanism**: In sparse query design, the 3D positions of control points serve as internal model states and should directly correspond to precise locations on the lane curve. B-Spline control points do not lie on the curve (see Fig. 2 in the paper), making them unsuitable for sparse design. CR splines guarantee that the curve passes through all control points, naturally matching this requirement.
    - **Formulation**: The $i$-th lane line is represented as $\mathbf{f}_i(s) = [s^3, s^2, s, 1] \cdot \mathbf{M}_{CR} \cdot \mathbf{P}_i$, where $s \in [0,1]$ and $\mathbf{P}_i \in \mathbb{R}^{M \times 4}$ contains 3D spatial coordinates and a visibility component.
    - **Design Motivation**: The continuous representation requires minimal post-processing and enables training with dense ground truth. The longitudinal $y$ component is predefined as uniformly distributed to avoid over-parameterization.

2. **Spatio-Temporal Attention Mechanism (STA)**:

    - **Mechanism**: STA replaces the redundant pairwise interactions in global self-attention with three structure-prior-driven attention types:
        - **Same-Lane Attention (SLA)**: Restricts interactions to adjacent control points within the same lane, capturing intrinsic continuity.
        - **Parallel Neighbor Attention (PNA)**: Encourages interactions between adjacent parallel lanes to enforce geometric constraints.
        - **Temporal Cross-Attention (TCA)**: Leverages queries propagated from historical frames stored in a memory queue to interact with current-frame queries.
    - **Memory Queue**: A FIFO strategy stores query embeddings and control points from the past $T$ frames; historical control points are propagated to the current frame coordinate system via ego-motion transformation.
    - **Design Motivation**: Reducing redundant interactions and focusing on lane-specific spatial relationships; TCA enables disambiguation using historical observations in occlusion and low-visibility scenarios.

3. **Spatio-Temporal Regularization**:

    - **Spatial Regularization** $\mathcal{L}_{spatial}$: Inherited from LaneCPP; encourages lane parallelism, road surface smoothness, and suppression of excessive curvature.
    - **Temporal Consistency Regularization** $\mathcal{L}_{temp}$: Based on an exponential moving average of temporal predictions, this term constrains the current-frame prediction to be consistent with the historical average. A visibility-weighted L1 loss is used: $\mathcal{L}_{temp} = \frac{1}{N} \sum_i \int_s \bar{\mathbf{f}}_{v,i}^{(t)}(s) \cdot \| \mathbf{f}_{3D,i}(s) - \bar{\mathbf{f}}_{3D,i}^{(t)}(s) \|_1$.
    - **Design Motivation**: Lane markings are static; inter-frame predictions should remain consistent. Temporal regularization prevents prediction drift and gradual disappearance.

### Loss & Training

- Regression loss: L1 loss for $x$ and $z$ components
- Visibility loss: Binary cross-entropy
- Classification loss: Focal Loss
- Additional regularization: $\mathcal{L}_{spatial} + \mathcal{L}_{temp}$
- The prediction head uses weight-shared MLPs across layers, with sigmoid normalization to $[0,1]$ followed by scaling to the target range
- Auxiliary task: Lane instance segmentation for query initialization

## Key Experimental Results

### Main Results

**OpenLane Dataset Comparison (Table 4):**

| Method | Backbone | F1(%)↑ | X-err near(m)↓ | X-err far(m)↓ | Z-err near(m)↓ | Z-err far(m)↓ |
|--------|----------|--------|----------------|---------------|----------------|---------------|
| PersFormer | EfficientNet-B7 | 50.5 | 0.485 | 0.553 | 0.364 | 0.431 |
| LATR | ResNet-50 | 61.9 | 0.219 | 0.259 | 0.075 | 0.104 |
| PVALane | Swin-B | 63.4 | 0.226 | 0.257 | 0.093 | 0.119 |
| GroupLane | ConvNext-B | 64.1 | 0.320 | 0.441 | 0.233 | 0.402 |
| **SparseLaneSTP** | **ResNet-50** | **66.1** | **0.203** | **0.240** | **0.066** | **0.092** |

**ONCE-3DLanes Dataset (Table 6):**

| Method | F1(%)↑ | Precision(%)↑ | Recall(%)↑ | CD(m)↓ |
|--------|--------|--------------|-----------|--------|
| LATR | 80.59 | 86.12 | 75.73 | 0.052 |
| GroupLane | 80.73 | 82.56 | 78.90 | 0.053 |
| **SparseLaneSTP** | **82.75** | **86.47** | **79.33** | **0.048** |

### Ablation Study

**Incremental Contribution Analysis (Table 1, OpenLane, 2-layer decoder):**

| Configuration | F1(%)↑ | Gain |
|---------------|--------|------|
| Baseline (discrete + global attention) | 61.8 | — |
| + CR spline continuous representation | 62.9 | +1.1 |
| + Spatio-temporal attention (STA) | 65.0 | +2.1 |
| + Regularization | 65.3 | +0.3 |

**Attention Combination Ablation (Table 2):**

| Attention Type | F1(%)↑ |
|----------------|--------|
| Global self-attention | 62.9 |
| SLA + PNA | 63.8 |
| SLA + PNA + TCA | **65.0** |

### Key Findings

- Temporal Cross-Attention (TCA) yields the largest gain (+1.2%), validating the importance of temporal information for lane detection
- The optimal temporal window is $T=3$ frames: fewer frames provide insufficient temporal context, while more frames introduce redundancy
- A lightweight 2-layer decoder with STA achieves 65.3% F1 at 16.5 FPS, surpassing LATR's 6-layer model (61.9% / 12.1 FPS); temporal integration adds only 9% overhead

## Highlights & Insights

- **The insight of replacing B-Spline with CR splines is particularly precise**: the property that control points lie on the curve perfectly matches the 3D positional semantics of sparse queries
- **Structured attention design** outperforms global attention, demonstrating that incorporating inductive biases in tasks with well-defined geometric structure is superior to general-purpose solutions
- Temporal information provides notable recovery in occlusion and low-visibility scenarios (qualitative results in Fig. 6 show that the non-temporal model loses detections while the temporal model maintains stability)
- An additional contribution is a high-quality 3D lane dataset (250m range, with occlusion annotations) based on a self-labeling pipeline

## Limitations & Future Work

- Only monocular front-view cameras are used; multi-camera surround-view settings are not explored
- The self-labeled dataset relies on the quality of 2D detectors and visual odometry
- Semantic lane information (solid/dashed/stop lines, etc.) is not leveraged to assist temporal matching
- Extension to 3D lane tracking to more fully exploit temporal information is identified by the authors as a future direction

## Related Work & Insights

- The method inherits spatial prior ideas from LaneCPP and successfully adapts them to a sparse architecture, demonstrating the feasibility of transferring domain knowledge across architectural paradigms
- Temporal fusion draws on the object-centric query propagation paradigm from StreamPETR / Sparse4Dv2, offering greater efficiency than BEV-based temporal fusion
- Insight: The combination of structured attention and domain-prior regularization may generalize to other detection tasks with geometric constraints (e.g., road boundary detection, guardrail detection)

## Rating

- **Novelty**: ⭐⭐⭐⭐ The insight of adapting CR splines to sparse queries is novel; the spatio-temporal attention design is well-motivated but not entirely unprecedented
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks plus a self-constructed dataset, thorough ablations, and complete efficiency analysis
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-formatted equations, and rich figures
- **Value**: ⭐⭐⭐⭐ Practically valuable for 3D lane detection engineering; demonstrates the importance of temporal fusion for lane detection

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[NeurIPS 2025\] Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection](../../NeurIPS2025/autonomous_driving/spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection.md)
- [\[AAAI 2026\] Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception](../../AAAI2026/autonomous_driving/rethinking_the_spatio-temporal_alignment_of_end-to-end_3d_perception.md)
- [\[NeurIPS 2025\] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving](../../NeurIPS2025/autonomous_driving/futuresightdrive_thinking_visually_with_spatiotemporal_cot_f.md)

<!-- RELATED:END -->
