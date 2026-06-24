---
title: >-
  [Paper Note] Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion
description: >-
  [ECCV 2024][Autonomous Driving][Semantic Scene Completion] Addressing the coarse temporal information utilization in camera-based semantic scene completion (SSC), this paper proposes a Hierarchical Temporal Context Learning (HTCL) paradigm: it first measures fine-grained correspondence between present and historical frames using Cross-frame Pattern Affinity (CPA), and then adaptively samples to compensate for incomplete observations through Affinity-guided Dynamic Refinement…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Semantic Scene Completion"
  - "Temporal Context Learning"
  - "Cross-frame Affinity"
  - "Deformable Convolution"
  - "Occupancy Prediction"
date: 2026-05-08
content_hash: 0e6a525d59e5a8fa
---

# Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion

**Conference**: ECCV 2024  
**arXiv**: [2407.02077](https://arxiv.org/abs/2407.02077)  
**Code**: [https://github.com/Arlo0o/HTCL](https://github.com/Arlo0o/HTCL)  
**Area**: Autonomous Driving  
**Keywords**: Semantic Scene Completion, Temporal Context Learning, Cross-frame Affinity, Deformable Convolution, Occupancy Prediction

## TL;DR
Addressing the coarse temporal information utilization in camera-based semantic scene completion (SSC), this paper proposes a Hierarchical Temporal Context Learning (HTCL) paradigm: it first measures fine-grained correspondence between present and historical frames using Cross-frame Pattern Affinity (CPA), and then adaptively samples to compensate for incomplete observations through Affinity-guided Dynamic Refinement (ADR). HTCL ranks 1st on SemanticKITTI, and even surpasses LiDAR-based methods in mIoU on OpenOccupancy.

## Background & Motivation

**Background**: 3D Semantic Scene Completion (SSC) is a crucial dense perception task in autonomous driving, requiring joint inference of scene geometry and semantic information from limited sensor observations. Due to cost advantages, camera-based SSC approaches have received widespread attention, with representative methods including MonoScene, TPVFormer, OccFormer, and VoxFormer.

**Limitations of Prior Work**: Early methods rely solely on single-frame information, offering extremely limited observations. VoxFormer-T first introduces temporal modeling, but its method is highly simplistic—directly stacking and aggregating multi-frame images. This naive temporal modeling presents two core issues:

**Semantic Position Drift**: The same semantic content shifts unpredictably across different viewpoints; direct aggregation leads to blurred prediction signals.

**Redundant Information Interference**: Stacking assumes a natural pixel-level correspondence across different frames, failing to differentiate between valid information and redundant noise.

**Key Challenge**: SSC is a fine-grained, dense perception task requiring semantic predictions for every single voxel. Coarse-grained temporal aggregation not only fails to provide effective complementary information but also hampers learning. Strategies from temporal 3D detection (region-level coarse-grained) and video depth estimation (matching tasks) are not applicable to SSC scenarios.

**Key Insight**: Decompose temporal context modeling into two hierarchical steps—first measuring "which temporal details are relevant" (affinity measurement), and then "how to effectively utilize the relevant information" (dynamic refinement sampling).

**Core Idea**: Construct hierarchical temporal context learning to model fine-grained cross-frame correspondence via scale-aware isolation and multiple independent learner groups, and then utilize affinity weights to guide deformable 3D convolutions for adaptive sampling position refinement.

## Method

### Overall Architecture
The HTCL framework consists of three main components:
- **Aligned Temporal Volume Construction (Top Branch)**: Relative poses are estimated via PoseNet, and homography warping is utilized to align historical frame features.
- **Voxel Feature Volume Construction (Bottom Branch)**: EfficientNetB7 UNet extracts present frame features, extending the LSS paradigm to build the voxel feature volume.
- **Reliable Temporal Aggregation**: Reliable temporal information is extracted via CPA and ADR, and aggregated into voxel features using Weighted Voxel Cross-Attention (WVA).

### Key Designs

#### 1. Temporal Content Alignment (TCA): Feature Volume vs. Cost Volume

**Function**: Constructs temporally aligned feature representations to provide a foundation for subsequent affinity computations.

**Mechanism**: Unlike cost volumes in video depth estimation, SSC is a dense perception reconstruction problem rather than a matching task. Thus, fine-grained feature contexts are preserved instead of computing matching costs.

**Implementation details**:
1. Present and historical frames estimate relative poses via a lightweight PoseNet.
2. Historical features are aligned to the present coordinates using Homography Warping:
$$\text{Warp}(\mathbf{p}) = \mathbf{K}_i \cdot (\mathbf{R}_{0,i} \cdot (\mathbf{K}_0^{-1} \cdot \mathbf{p} \cdot d_j) + \mathbf{t}_{0,i})$$
3. Present features are lifted along the depth dimension and concatenated with warped historical features to construct the temporal feature volume $V_{tem}$.

**Design Motivation**: Preserving feature contexts allows subsequent modules to fully exploit fine-grained semantic information. Ablation experiments prove that the feature volume improves mIoU by 1.11 over the cost volume.

#### 2. Cross-frame Pattern Affinity (CPA): Fine-grained Correspondence Modeling

**Function**: Measures region-level semantic correspondences between historical and present frames, filtering the most relevant temporal contexts.

**Mechanism**: Enhances cosine similarity by introducing Scale-aware Isolation and Multi-group Context to overcome the limitations of traditional cosine similarity in dense distributions.

**Implementation details**:
1. **Multi-group Context Generation**: Multi-group context features are generated for historical and present feature volumes using 3D dilated convolutions with different dilation rates (1, 2, 4):
$$\mathbf{H}_i = \text{GN}(\delta(\text{Atrous}_i(V_{tem}^{his})))$$
2. **Similarity Calculation with Scale-aware Isolation**: Subtract the respective means within each group scale before computing cosine similarity:
$$\mathbf{A}_i = \frac{\sum_{j=0}^{C}(\mathbf{C}_i^j - \bar{\mathbf{C}}_i)(\mathbf{H}_i^j - \bar{\mathbf{H}}_i)}{\sqrt{\sum_{j=0}^{C}(\mathbf{C}_i^j - \bar{\mathbf{C}}_i)^2} \sqrt{\sum_{j=0}^{C}(\mathbf{H}_i^j - \bar{\mathbf{H}}_i)^2}}$$
3. **Aggregation**: Concatenate multi-group affinity matrices along the channel dimension to obtain the cross-frame pattern affinity $\hat{\mathbf{A}}$.

**Design Motivation**:
- Scale-aware Isolation: Eliminates scale bias and prevents original cosine similarity from assigning high scores to dissimilar vectors.
- Multi-group Learners: Inspired by ensemble learning, different dilation rates focus on patterns at varying spatial scales, enhancing discriminative capability in dense distributions.

#### 3. Affinity-guided Dynamic Refinement (ADR)

**Function**: Uses affinity weights to adaptively refine sampling positions, dynamically compensating for incomplete observations.

**Mechanism**: Performs deformable sampling at high-affinity positions and their neighborhoods, combining affinity weights to weigh the features.

**Key Formulation**:
$$V_{def} = \sum_{k=1}^{K_w} w_k \cdot V_{tem}(\mathbf{p} + \mathbf{p}_k + \Delta \mathbf{p}_k) \cdot a_k$$
where $\Delta \mathbf{p}_k$ is the deformable offset, $w_k$ is the spatial weight, and $a_k$ is the affinity weight.

A multi-level refinement block is constructed using a three-layer cascaded 3D deformable convolution, aggregating outputs into a reliable temporal volume:
$$\widetilde{V}_{tem} = W(\text{Concat}\{V_{def}^1, V_{def}^2, V_{def}^3\})$$

#### 4. Weighted Voxel Attention (WVA)

**Function**: Aggregates reliable temporal information into voxel features.

Integration formula: $V_{ret} = \alpha \cdot \text{CrossAtt}(Q, K, V) + V_{vox}$

where the learnable coefficient $\alpha$ is initialized to 0 and gradually increases during training, preventing immature temporal information from interfering with voxel feature learning in the early stages.

### Loss & Training

Standard SSC loss (cross-entropy + Scene-Class Affinity loss) is used, with the AdamW optimizer, a learning rate of $1 \times 10^{-4}$, weight decay of 0.01, trained for 24 epochs with a batch size of 4. By default, the present frame plus the preceding 3 historical frames are used.

Two configurations are provided:
- **HTCL-S**: Stereo pipeline based on stereo depth estimation.
- **HTCL-M**: Mono pipeline based on monocular depth estimation.

## Key Experimental Results

### Main Results

**SemanticKITTI Test Set**:

| Method | Input | IoU↑ | mIoU↑ | car | road | building | vegetation |
|------|------|--------|--------|------|-------|----------|-----------|
| MonoScene | M | 34.16 | 11.08 | 18.8 | 54.7 | 14.4 | 14.9 |
| TPVFormer | M | 34.25 | 11.26 | 19.2 | 55.1 | 14.8 | 13.9 |
| SurroundOcc | M | 34.72 | 11.86 | 20.6 | 56.9 | 15.2 | 14.9 |
| OccFormer | M | 34.53 | 12.32 | 21.6 | 55.9 | 15.7 | 16.8 |
| VoxFormer-S | S | 42.95 | 12.20 | 20.8 | 53.9 | 19.8 | 22.4 |
| VoxFormer-T | S-T | 43.21 | 13.41 | 21.7 | 54.1 | 23.5 | 24.4 |
| **HTCL-S** | **S-T** | **44.23** | **17.09** | **27.3** | **64.4** | **25.9** | **25.3** |

Improvement over VoxFormer-T: IoU +1.02, **mIoU +3.68** (relative improvement of 27.4%)

**OpenOccupancy Validation Set** (surpasses LiDAR methods):

| Method | Input | IoU↑ | mIoU↑ |
|------|------|--------|--------|
| LMSCNet (LiDAR) | L | 27.3 | 11.5 |
| JS3C-Net (LiDAR) | L | 30.2 | 12.5 |
| MonoScene (Camera) | M | 18.4 | 6.9 |
| TPVFormer (Camera) | M | 15.3 | 7.8 |
| **HTCL-M (Camera)** | **M-T** | **21.4** | **14.1** |

The mIoU of HTCL-M (14.1) exceeds the LiDAR-based methods JS3C-Net (12.5) and LMSCNet (11.5)!

### Ablation Study

**Ablation of Architecture Components (SemanticKITTI Validation Set)**:

| Component Variation | IoU(%) | mIoU(%) | Description |
|---------|--------|---------|------|
| Full HTCL-S | **45.51** | **17.13** | All components |
| Replace Feature Volume with Cost Volume | 43.07 | 15.18 | TCA: Feature volume outperforms cost volume (+1.95) |
| Without Scale-aware Isolation | 42.79 | 14.65 | CPA: Scale-aware isolation contributes +1.95 mIoU |
| Without Multi-group Context | 42.96 | 15.14 | CPA: Multi-group context contributes +1.87 mIoU |
| Without Affinity Weights | 43.97 | 15.85 | ADR: Affinity weights contribute +1.28 mIoU |
| Replace Deformable with standard convolution | 44.13 | 15.98 | ADR: Deformable convolution contributes +1.15 mIoU |

**Ablation of Historical Frames**:

| Historical Frames | mIoU(%) | Inference Time (s) |
|---------|---------|-----------|
| 1 frame | 15.08 | 0.268 |
| 2 frames | 16.43 | 0.283 |
| **3 frames (default)** | **17.13** | 0.297 |
| 4 frames | 17.31 | 0.311 |
| 5 frames | 17.42 | 0.324 |

### Key Findings

1. **Feature volumes are significantly superior to cost volumes**: SSC requires fine-grained semantics. Retaining feature context is more critical than computing matching costs.
2. **Both components of CPA contribute approximately 2 mIoU respectively**: Scale-aware isolation and multi-group independent learning are both indispensable.
3. **3 frames offer the optimal trade-off between efficiency and effectiveness**: Beyond 3 frames, performance gains diminish while latency increases linearly.
4. **Camera-based methods can outperform LiDAR in semantics**: Though LiDAR provides accurate geometric measurements (IoU advantage), rich visual semantics allow camera-based methods to surpass LiDAR in mIoU.
5. **The progressive learning strategy of WVA** (initializing $\alpha$ to 0) ensures model training stability.

## Highlights & Insights

1. **Hierarchical decomposition philosophy**: Deconstructing the master problem of "temporal information utilization" into two clean logical steps: "correlation measurement $\rightarrow$ correlation utilization".
2. **Significance of camera surpassing LiDAR**: Surpassing LiDAR methods in mIoU on OpenOccupancy proves that temporal modeling can compensate for the lack of geometric precision.
3. **Deep understanding of SSC task properties**: Clarifies that SSC is not a matching task (unlike video depth estimation), thus advocating feature volumes over cost volumes.
4. **Innovation in similarity metrics**: Improving the adaptability of cosine similarity to dense scenarios, similar to Pearson correlation coefficient principles.
5. **Progressive learning design of $\alpha$** prevents early-stage temporal noise from disrupting core representation learning, reflecting curriculum learning principles.

## Limitations & Future Work

1. Inference speed (~0.3s/frame) still falls short of real-time deployment constraints. More lightweight temporal aggregation schemes can be explored.
2. PoseNet relies heavily on the accuracy of ego-motion estimations; errors in ego-motion estimation impact temporal alignment quality.
3. Offset learning in deformable convolutions might exhibit limited efficacy under extreme occlusion scenarios.
4. Future work could extend HTCL to multi-camera (surround-view) SSC environments (e.g., the 6-camera setup in nuScenes).
5. Information refinement and utilization strategies for long-term temporal modeling (>5 frames) warrant further investigation.

## Related Work & Insights

- **VoxFormer/VoxFormer-T**: A crucial baseline for SSC, which HTCL directly improves upon in terms of temporal modeling.
- **MonoScene / TPVFormer / OccFormer**: Representative methods in camera-based SSC.
- **Video Depth Estimation Methods**: HTCL draws inspiration from video depth estimation techniques in utilizing homography warping to construct temporal volumes.
- **Deformable Convolutions / Deformable Attention**: The core machinery undergirding the ADR module.
- Insight: In dense perception tasks, the utilization of temporal information should be selective rather than all-encompassing.

## Rating
- Novelty: ⭐⭐⭐⭐ The hierarchical temporal modeling paradigm is novel, featuring innovative combined designs with CPA and ADR.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Exceptionally comprehensive, including testing across two benchmarks, thorough ablation studies, two pipeline variants (Stereo/Mono), and temporal frame count analyses.
- Writing Quality: ⭐⭐⭐⭐ The methodology motivation and design logic are highly coherent, backed by high-quality illustrations and charts.
- Value: ⭐⭐⭐⭐⭐ Overpassing LiDAR in mIoU using a camera setup is a significant milestone, providing broad inspiration for temporal modeling in SSC and occupancy prediction domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](../../AAAI2026/autonomous_driving/towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)
- [\[NeurIPS 2025\] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](../../NeurIPS2025/autonomous_driving/learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)
- [\[ECCV 2024\] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [\[CVPR 2026\] OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective](../../CVPR2026/autonomous_driving/occufly_a_3d_vision_benchmark_for_semantic_scene_completion_from_the_aerial_pers.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](../../CVPR2026/autonomous_driving/sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)

</div>

<!-- RELATED:END -->
