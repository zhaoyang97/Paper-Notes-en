---
title: >-
  [Paper Note] FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection
description: >-
  [CVPR 2025][3D Vision][3D object detection] This paper proposes FASTer, which adaptively selects focal tokens and compresses sequences via an Adaptive Scaling mechanism, and progressively aggregates long-term temporal point cloud information using a grouped hierarchical fusion strategy. It achieves new state-of-the-art (SOTA) performance on the Waymo Open Dataset with the lowest latency (75ms) and memory footprint (2856M).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D object detection"
  - "temporal fusion"
  - "focal tokens"
  - "adaptive scaling"
  - "grouped hierarchical fusion"
  - "point cloud sequences"
date: 2026-05-08
content_hash: 488cb2efc8194720
---

# FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.01899](https://arxiv.org/abs/2503.01899)  
**Code**: [MSunDYY/FASTer](https://github.com/MSunDYY/FASTer)  
**Area**: 3D Vision  
**Keywords**: 3D object detection, temporal fusion, focal tokens, adaptive scaling, grouped hierarchical fusion, point cloud sequences

## TL;DR

This paper proposes FASTer, which adaptively selects focal tokens and compresses sequences via an Adaptive Scaling mechanism, and progressively aggregates long-term temporal point cloud information using a grouped hierarchical fusion strategy. It achieves new state-of-the-art (SOTA) performance on the Waymo Open Dataset with the lowest latency (75ms) and memory footprint (2856M).

## Background & Motivation

### Background
LiDAR-based 3D object detection is a core perception task in autonomous driving. Due to the inherent sparsity of single-frame point clouds, researchers have begun leveraging multi-frame temporal information to improve detection performance. Current mainstream temporal detectors adopt a region-based paradigm: first generating coarse proposals, and then sampling points within the proposal regions to encode and fuse features.

### Limitations of Prior Work
1. **Inefficient indiscriminate sampling**: Existing methods treat all points equally and sample a fixed, large number of points (e.g., 192). However, the number of points reflected by different instances varies drastically (from a few to thousands), leading to wasted computation and storage on background and padding points.
2. **Complexity growth with frame counts**: Storing full historical point clouds and executing complete spatial-temporal fusion on each frame results in a surge of overhead as the frame count increases (e.g., MSF requires 400ms latency and 6083M GPU memory for 8 frames).
3. **Simple concatenation limiting global information interaction**: Prior methods concatenate the outputs of each frame along the channel dimension after spatial and temporal fusion, failing to effectively extract and exchange global contextual info.

### Key Challenge
How to process longer point cloud sequences while simultaneously ensuring detection performance and computational efficiency?

### Key Insight
Observing that the number of effective points varies significantly across different instances, this work reformulates region detection as a variable-length sequence modeling problem—dynamically compressing sequence length and preserving only the most valuable "focal tokens" to represent objects.

### Core Idea
This work introduces the concept of focal tokens to point cloud detection: adaptively evaluating the contribution of each point using attention maps, selecting the most contributing points as focal tokens for storage and subsequent fusion, and designing a grouped hierarchical fusion strategy to progressively compress long-term sequences into a single, information-dense sequence.

## Method

### Overall Architecture
FASTer consists of four core modules: (1) a Region Proposal Network to generate coarse proposals; (2) Single-frame Sequence Processing (SSP) to encode geometric features of the current frame and acquire focal tokens; (3) Multi-frame Sequence Processing (MSP) to perform lightweight temporal fusion using the stored focal points; and (4) a dual-layer decoder to aggregate outputs from SSP and MSP. The overall workflow is: over-sampling the current frame $\to$ adaptively scaling and selecting focal tokens $\to$ storing them in a memory bank $\to$ lightweight sampling of historical frames $\to$ grouped hierarchical fusion $\to$ decoding outputs.

### Key Designs

#### 1. Adaptive Scaling Mechanism (Adaptive Scaling / Ad-MHSA)

- **Function**: Adaptively evaluates the contribution of each token during the multi-head self-attention process, selects the most valuable focal tokens, and compresses the sequence length.
- **Mechanism**: Utilizes attention maps to calculate the contribution score of each token to the global representation. Specifically, for each token $i$, the maximum value across all attention heads is taken, summed, and normalized via sigmoid to obtain the contribution score $S_i$. Then, the $N_s$ tokens with the highest scores are selected as focal tokens.
- **Design Motivation**: Unlike images where each pixel has explicit semantics (allowing Class Token Attention or score prediction), an individual point in a point cloud cannot easily establish direct relations with global queries and lacks explicit semantic supervision. Therefore, the attention-map-based solution is better suited for point cloud characteristics—scores and token selection are determined by the network globally, without manual hyperparameters or extra supervision.
- **Actual Effect**: Progressively compresses sampling points from $4K=192$ down to $K=48$ focal tokens. Stored points are reduced from 180k in the full scene to only around 1.8k–2.6k, which is virtually negligible.

#### 2. Focal Token Collection and Storage

- **Function**: Efficiently stores focal points of historical frames, avoiding duplicate storage and substantially reducing memory overhead.
- **Mechanism**: Due to overlaps among proposals, a single point may be sampled by multiple proposals. Repetition is avoided by constructing a unique index matrix: the point index matrix $I \in \mathbb{R}^{M \times 4K}$ of each proposal is recorded during initial sampling. After adaptive scaling, the unique indices of final focal points across all proposals are extracted, and only these deduplicated focal points are stored.
- **Design Motivation**: Existing methods (e.g., MPPNet, MSF) require storing full historical point clouds, leading to storage demands that scale linearly with the frame count. Storing only focal points reduces the complexity from $O(FN)$ to nearly $O(N)$.
- **Storage Comparison**: Full point cloud ~180k points $\to$ Proposal region sampling ~13.8k $\to$ FASTer focal storage ~1.8k-2.6k.

#### 3. Grouped Hierarchical Fusion

- **Function**: Progressively aggregates long-term temporal sequence information, compressing $T$ historical sequences into a single, information-dense sequence.
- **Mechanism**:
    - First, Ad-MHSA is executed on each historical frame sequence to compress the sequence length.
    - The $T$ temporal sequences are divided into groups at equal intervals (e.g., 16 frames into 4 groups with equally spaced time indices), ensuring each group has roughly equal global importance.
    - Intra-Group Fusion (IGF) is performed within each group: each sequence is first processed with max pooling to obtain global representations, which are concatenated, compressed via Conv, and then fused with original sequence features.
    - Ad-MHSA and IGF are executed alternately, progressively compressing long-term sequences into a single sequence.
- **Design Motivation**: The "encode-fuse-concatenate" paradigm of traditional methods limits the interaction and aggregation of global information. Equally spaced grouping ensures each group contains information from different time periods, and sharing the processing network across groups enhances generalization.
- **Mechanism**: Intra-group fusion extracts global features via MaxPool $\to$ Channel concatenation $\to$ Conv compression $\to$ Residual connection $\to$ Flattened into a single sequence.

### Loss & Training
- Total Loss = Confidence Loss (Binary Cross Entropy) + $\alpha \times$ Regression Loss.
- Outputs of each decoder layer are supervised during the training phase, while only the MSP decoder output is used during inference.

## Key Experimental Results

### Main Results

**Waymo Val Set (16 frames)**:

| Method | Frames | ALL mAPH(L2) | Latency | GPU Memory |
|------|------|-------------|------|------|
| MPPNet | 4 | 74.22 | 332ms | 4153M |
| MSF | 8 | 75.46 | 400ms | 6083M |
| PTT | 32 | 75.48 | 99ms | 6938M |
| **FASTer** | **16** | **75.92** | **75ms** | **2856M** |

**Waymo Test Set**:

| Method | Frames | mAP/mAPH(L2) |
|------|------|-------------|
| MSF | 8 | 78.30/76.96 |
| **FASTer** | 16 | 78.53/77.21 |
| **FASTer** | 32 | 78.82/77.54 |

### Key Findings

1. FASTer-16 frames achieves the best efficiency-performance trade-off with a latency of 75ms and 2856M GPU memory. The latency is $5.3\times$ lower and the GPU memory is $2.1\times$ lower than MSF-8 frames.
2. FASTer-32 frames achieves 76.06 ALL mAPH(L2) on the validation set, outperforming PTT-64 frames (75.71) while using only half the frames.
3. Optimal or near-optimal results are achieved across all three categories: Vehicle, Pedestrian, and Cyclist.
4. Stored focal points are only around 1.8k–2.6k (1.4% of the 180k full point cloud), making the storage overhead virtually negligible.

### Ablation Study
- Adaptive Scaling vs. Random Sampling/FPS Sampling: Adaptive Scaling leads significantly across all frame count configurations.
- Grouped Hierarchical Fusion vs. Direct Concatenation/Alternative Fusion: Grouped Hierarchical Fusion performs better on both 16 and 32 frames.
- Extra Point Augmentation (EPA) training strategy: Reduces dependence on RPN and improves generalization.

## Highlights & Insights

1. **Novel and intuitive concept of focal tokens**: Reformulates region point cloud detection as a variable-length sequence compression problem. Token selection driven by attention achieves adaptive sampling, preserving critical information while massively reducing redundancy.
2. **Substantial efficiency gains**: While outperforming the previous SOTA, its latency and memory footprint are significantly better than competing methods. This is crucial for actual deployment.
3. **Unified spatial-temporal fusion**: Eliminates the explicit separation of space and time fusion, progressively unifying them via grouped hierarchical fusion, which is highly beneficial for global context extraction.
4. **High scalability**: Enabled by the focal storage strategy, FASTer scales easily to longer sequences (32 or 64 frames) without causing resource bottlenecks.

## Limitations & Future Work

1. The quality of focal token selection depends on the accuracy of the attention maps; a staged training strategy might be required in the early training phases to guide learning.
2. It still relies on the RPN to generate initial proposals, so detection performance is bounded to some extent by proposal quality.
3. Verified only on the Waymo Open Dataset, lacking generalization evaluation on other large-scale datasets like nuScenes.
4. The equally spaced grouping strategy is heuristic and may not be the optimal grouping method for all scenarios.

## Related Work & Insights

- **Comparison with MPPNet/MSF**: As classic region-based temporal detectors, MPPNet and MSF adopt the "encode-fuse-concatenate" paradigm. FASTer fundamentally alters this paradigm through focal tokens and hierarchical fusion.
- **Comparison with PTT**: PTT lowers overhead by discarding historical points and modeling only box trajectories, which sacrifices historical semantic details. FASTer retains the semantic information of focal points while achieving even lower overhead.
- **Application of token compression in point clouds**: Transfers the concept of token compression from the ViT field (e.g., EViT, ToMe) to point cloud detection, with appropriate adaptations to address the lack of explicit semantics in point clouds.

## Rating

⭐⭐⭐⭐ (4/5)

The work is highly complete, offering a clear problem definition and effective solutions. The dual improvement in efficiency and performance holds strong practical significance for autonomous driving deployment scenarios. However, validation on only a single dataset is a limitation, and the visualization as well as theoretical analysis of focal tokens could be further strengthened.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FSHNet: Fully Sparse Hybrid Network for 3D Object Detection](fshnet_fully_sparse_hybrid_network_for_3d_object_detection.md)
- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[CVPR 2025\] SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts](sp3d_boosting_sparsely-supervised_3d_object_detection_via_accurate_cross-modal_s.md)
- [\[CVPR 2025\] MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection](monoplace3d_learning_3d-aware_object_placement_for_3d_monocular_detection.md)
- [\[CVPR 2026\] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](../../CVPR2026/3d_vision/magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)

</div>

<!-- RELATED:END -->
