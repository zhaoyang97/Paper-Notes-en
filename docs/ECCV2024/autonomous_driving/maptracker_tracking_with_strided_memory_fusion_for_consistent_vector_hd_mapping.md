---
title: >-
  [Paper Note] MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping
description: >-
  [ECCV 2024][Autonomous Driving][Vector HD Mapping] Redefines online vector HD mapping as a tracking task. It achieves temporally consistent HD map reconstruction through a strided memory buffer fusion mechanism with dual representations (BEV grid + road element vector), significantly outperforming existing methods on nuScenes and Argoverse2 with 76.1 and 76.9 mAP, respectively.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Vector HD Mapping"
  - "Tracking Paradigm"
  - "Memory Mechanism"
  - "BEV Perception"
  - "Temporal Consistency"
date: 2026-05-08
content_hash: bce48ee40609e035
---

# MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping

**Conference**: ECCV 2024  
**arXiv**: [2403.15951](https://arxiv.org/abs/2403.15951)  
**Code**: Available ([https://map-tracker.github.io](https://map-tracker.github.io))  
**Area**: Autonomous Driving / HD Mapping  
**Keywords**: Vector HD Mapping, Tracking Paradigm, Memory Mechanism, BEV Perception, Temporal Consistency

## TL;DR

Redefines online vector HD mapping as a tracking task. It achieves temporally consistent HD map reconstruction through a strided memory buffer fusion mechanism with dual representations (BEV grid + road element vector), significantly outperforming existing methods on nuScenes and Argoverse2 with 76.1 and 76.9 mAP, respectively.

## Background & Motivation

Online Vector HD Mapping reconstructs road geometry (crosswalks, lane dividers, road boundaries) in real-time from on-board sensor data, which is crucial for autonomous driving. Reconstructing a consistent urban HD map from a single drive would drastically reduce mapping costs for tens of thousands of cities globally.

Existing methods suffer from the following key issues:

**Detection instead of Tracking**: Mainstream methods like MapTRv2 and StreamMapNet adopt a **frame-by-frame detection** paradigm, detecting road elements from scratch in each frame without enforcing temporal consistency. Although they may utilize reconstruction results from the previous frame as a guide (e.g., conditional detection in StreamMapNet), they cannot guarantee that the same road element is identified as the same instance across different frames.

**Limitations of Memory Mechanisms**: Methods like StreamMapNet employ standard RNN-style recurrent latent embeddings as memory, compressing all historical information into a single latent memory. In complex environments with severe vehicle occlusions, a single memory is prone to "memory loss"—occluded road structures are difficult to recover from a single memory.

**Flaws in Ground Truth and Evaluation Metrics**: The ground truth (GT) of existing benchmark datasets (MapTR and StreamMapNet versions) suffers from inter-frame inconsistencies (such as improper crosswalk merging strategies and incorrectly segmented lane dividers). Moreover, the standard mAP metric does not penalize temporally inconsistent reconstructions, failing to measure the temporal consistency of the methods.

**Key Insight**: HD map reconstruction is inherently a **tracking** problem—road elements persist in the physical world and should be associated across frames rather than re-detected in each frame. Combining a redundant but robust multi-frame memory buffer (instead of a single memory) can significantly improve temporal consistency.

## Method

### Overall Architecture

The core of MapTracker is the **dual-representation memory mechanism**, which contains two types of memory buffers:
1. **BEV Memory**: A 2D latent image in the Birds-Eye-View coordinate system that accumulates spatial context.
2. **VEC Memory**: A latent vector for each road element that maintains element-level tracking information.

The sensor stream is processed sequentially: the image backbone extracts features $\rightarrow$ the BEV module constructs and fuses the BEV memory $\rightarrow$ the VEC module utilizes BEV information and vector memory for tracking and geometric reconstruction.

### Key Designs

1. **BEV Memory Buffer and Strided Fusion**: Resolving the information loss issue of a single memory.

   The BEV memory $\mathbf{M}_{\text{BEV}}(t) \in \mathbb{R}^{50 \times 100 \times 256}$ is an ego-centric 2D latent image covering a region of 15m left/right and 30m front/back. The buffer retains the memory of the most recent 20 frames, making the memory mechanism redundant but robust.

   **BEV Query Propagation**: Align the previous frame's BEV memory to the current frame via an affine transformation $P_{t-1}^t$ based on ego-vehicle motion and bilinear interpolation. Regions newly entering the field of view are initialized with a learnable embedding $\mathbf{M}_{\text{BEV}}^{\text{init}}$ (MaskBlend operation).

   **Strided Memory Selection**: Instead of using all 20 frames (which is computationally heavy and redundant), four memories with spatial strides based on the vehicle's travel distance are selected (the nearest frames to 1m/5m/10m/15m from the current position). They are fused using a lightweight residual convolutional block after coordinate alignment.

   **Design Motivation**: Spatial striding rather than temporal striding ensures uniform spatial coverage and efficiency. It simultaneously retains memory information of both close proximity (fine details) and long distance (global context).

2. **VEC Memory and Tracking-based Query Propagation**: Transforming HD mapping from a detection paradigm to a tracking paradigm.

   The VEC memory $\mathbf{M}_{\text{VEC}}(t) \in \{\mathbb{R}^{512}\}$ is a set of vector latent representations, each corresponding to an active road element. Upon initialization, it is divided into two parts:

    $\mathbf{M}_{\text{VEC}}(t) = [\mathbf{M}_{\text{VEC}}^{\text{prop}}(t), \mathbf{M}_{\text{VEC}}^{\text{new}}(t)]$

    - $\mathbf{M}_{\text{VEC}}^{\text{prop}}(t)$: **Propagation queries**, which are tracked road elements inherited from $\mathbf{M}_{\text{VEC}}(t-1)$, aligned to the coordinate system via PropMLP. PropMLP takes the positional encodings of the rotation quaternions and translation vectors from the motion transformation $P_{t-1}^t$, concatenates them with the vector latents, and maps them.
    - $\mathbf{M}_{\text{VEC}}^{\text{new}}(t)$: **New detection queries**, consisting of 100 learnable embeddings $\mathbf{M}_{\text{VEC}}^{\text{init}}$, used to discover road elements newly entering the field of view.

   **Vector Memory Fusion**: For each road element, the historical latent vectors of the same element are selected from the buffer (using the same spatial striding strategy) and fused via cross-attention. The query is the propagation latent of the current frame, and the keys/values are the aligned latents from historical frames.

   **Design Motivation**: It borrows the query propagation paradigm from visual object tracking models such as TrackFormer and MOTR. Propagation queries provide cross-frame association, and new detection queries handle the discovery of new elements, forming a complete tracking-detection closed loop.

3. **Consistency-Aware Benchmark and Evaluation Metrics**: Not only improving the method, but also fixing the evaluation framework.

   **GT Refinement**: Resolves two categories of issues in existing benchmarks—large crosswalks in nuScenes occasionally splitting into small loops, and lane dividers in Argoverse2 being incorrectly segmented into short segments. Frame-to-frame optimal bipartite matching is used to establish the temporal correspondences of road elements (ground truth tracks).

   **Consistency mAP (C-mAP)**: Introduces a temporal consistency check on top of the standard mAP pipeline. If the "ancestor" of a reconstructed element (the element in the previous frame within the same track) is not matched as a correct detection, the match in the current frame is also deemed inconsistent and removed. This effectively penalizes temporally inconsistent reconstructions.

   **Design Motivation**: Standard mAP evaluates each frame independently; thus, even if a method produces "correct" outputs for individual frames but fails in temporal consistency, the issue goes undetected. C-mAP fills this evaluation gap.

### Loss & Training

**Total Loss**:

$$\mathcal{L} = \mathcal{L}_{\text{BEV}} + \mathcal{L}_{\text{track}} + \lambda_5 \mathcal{L}_{\text{trans}}$$

- **BEV Loss**: $\mathcal{L}_{\text{BEV}} = \lambda_1 \mathcal{L}_{\text{focal}} + \lambda_2 \mathcal{L}_{\text{dice}}$, for segmentation supervision.
- **Tracking Loss**: $\mathcal{L}_{\text{track}} = \lambda_3 \mathcal{L}_{\text{focal}} + \lambda_4 \mathcal{L}_{\text{line}}$, extending the matching loss of MOTR. Propagation queries inherit labels using matchmaking results from previous frames, while new queries undergo Hungarian matching.
- **Transformation Loss**: $\mathcal{L}_{\text{trans}}$ trains PropMLP to maintain vector geometry and category invariance.
- Weights: $\lambda_1=10.0, \lambda_2=1.0, \lambda_3=5.0, \lambda_4=50.0, \lambda_5=0.1$

**Three-stage Training**:
1. Pre-train the image backbone and BEV encoder (using only $\mathcal{L}_{\text{BEV}}$).
2. Warm-start the vector decoder with other parameters frozen (large batch size to accelerate convergence, vector memory enabled after 500 warmup steps).
3. Jointly train all parameters.

- Optimizer: AdamW, initial lr=5e-4, cosine decay to 1.5e-6.
- 8$\times$ RTX A5000, training on nuScenes for 72 epochs takes ~3 days, inference runs at ~10 FPS.

## Key Experimental Results

### Main Results

**nuScenes Dataset** (using consistency GT):

| Method | Backbone | APp | APd | APb | mAP | C-mAP |
|------|----------|-----|-----|-----|-----|-------|
| MapTRv2 | R50 | 69.6 | 68.5 | 70.3 | 69.5 | 50.5 |
| StreamMapNet | R50 | 70.0 | 72.9 | 68.3 | 70.4 | 56.4 |
| **MapTracker** | **R50** | **80.0** | **74.1** | **74.1** | **76.1** | **69.1** |

**Argoverse2 Dataset** (using consistency GT):

| Method | Backbone | APp | APd | APb | mAP | C-mAP |
|------|----------|-----|-----|-----|-----|-------|
| MapTRv2 | R50 | 68.3 | 75.6 | 68.9 | 70.9 | 56.1 |
| StreamMapNet | R50 | 70.5 | 74.2 | 66.1 | 70.3 | 57.5 |
| **MapTracker** | **R50** | **77.0** | **80.0** | **73.7** | **76.9** | **68.3** |

MapTracker achieves a significant lead on both datasets: mAP increases by **8%+**, and C-mAP increases by **19%+**.

### Ablation Study

**Core Component Ablation** (nuScenes consistency GT):

| Method | Task | Memory | mAP | C-mAP | Description |
|------|------|------|-----|-------|------|
| Baseline (w/o temporal) | Detection | None | 69.9 | 56.1 | StreamMapNet with temporal modules removed |
| StreamMapNet | Conditional detection | Single memory | 70.4 | 56.4 | RNN-style memory |
| MapTracker (tracking-only) | Tracking | Single memory | 70.8 | 62.4 | Tracking paradigm significantly improves C-mAP |
| + Memory Fusion | Tracking | Buffer (recent 4 frames) | 74.9 | 68.1 | Memory buffer drastically improves performance |
| **+ Strided Selection** | **Tracking** | **Strided buffer** | **76.1** | **69.1** | Spatial striding provides further optimization |

**Strided Distance Selection Ablation**:

| Buffer Size | Stride Distance (m) | mAP | C-mAP |
|-----------|------------|-----|-------|
| 4 | None (recent 4 frames) | 74.9 | 68.1 |
| 20 | {0,0,0,0} | 75.0 | 68.2 |
| 20 | {1,5,10,15} | **76.1** | **69.1** |

### Key Findings

- **Tracking vs. Detection**: Simply switching to the tracking paradigm (without adding memory fusion) boosts C-mAP from 56.4 to 62.4 (+6%), demonstrating the core contribution of tracking to temporal consistency.
- **Huge Gains from Memory Fusion**: Adding a memory buffer and fusion increases mAP from 70.8 to 74.9 (+4.1%) and C-mAP from 62.4 to 68.1 (+5.7%), representing the largest source of performance improvement.
- **Spatial Striding Outperforms Temporal Striding**: {0,0,0,0} (repeating the most recent frames) yields mAP=75.0 vs. {1,5,10,15} (spatial striding) yields mAP=76.1, as spatial striding provides better spatial coverage.
- **Geographically Non-Overlapping Evaluation**: On a stricter non-overlapping test set, MapTracker exhibits an even larger advantage (100$\times$50m range: 36.2 vs. 23.0 mAP), demonstrating superior generalization capabilities.
- **Impact of GT Quality**: All methods show performance improvements on the refined consistency GT, indicating that GT quality is vital for training.

## Highlights & Insights

1. **Paradigm Shift**: Redefining HD mapping from "frame-by-frame detection" to "continuous tracking", which perfectly aligns with the physical characteristics of road elements (persisting in the real world).
2. **Dual-Representation Memory Design**: BEV grid memory captures spatial context, while VEC vector memory maintains element-level tracking. The two are complementary, each equipped with its own buffer and fusion mechanisms.
3. **Benchmark Contribution**: Beyond methodological improvements, the work refines the GT data and evaluation metrics, facilitating a fairer and more meaningful assessment.
4. **Simple and Effective Memory Selection**: The spatial striding strategy is training-free yet yields significant gains simply by ensuring uniform spatial coverage.

## Limitations & Future Work

- **Inference Speed**: ~10 FPS, which is slower compared to single-frame methods like MapTRv2 due to the overhead of maintaining and fusing memory buffers.
- **Reliance on Ego-Motion Estimation**: Memory alignment requires accurate frame-to-frame motion transformation and is sensitive to localization errors (alleviated by introducing random perturbations during training).
- **Performance Drop on nuScenes Non-Overlapping Set**: Large errors in coordinate regression suggest that there is still room for improvement in geometric generalization.
- Future Exploration: Extending the tracking + memory fusion design to other core autonomous driving perception tasks, such as 3D object detection or occupancy networks.
- Future Exploration: Incorporating multi-modal sensors like LiDAR to further enhance robustness in occluded scenarios.

## Related Work & Insights

- **TrackFormer/MOTR**: Provided tracking paradigms via query propagation, which MapTracker introduces to the HD mapping domain.
- **StreamMapNet**: The most direct baseline, adopting conditional detection + RNN-style memory. MapTracker upgrades this baseline to tracking + buffer-based memory.
- **BEVFormer**: Offered perspective-to-BEV cross-attention mechanism, which MapTracker directly inherits.
- **Sparse4Dv2/v3**: Utilizes RNN-style memory and temporal denoising in 3D object detection, showing similar conceptual counterparts in Vector HD mapping.
- Insight: The combination of a **tracking paradigm + redundant memory** could be a general solution for all online perception tasks that require temporal consistency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The structural combination of a tracking paradigm and strided memory fusion, while not a single localized breakthrough, yields outstanding joint effects.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Very comprehensive, with evaluations across two datasets, multiple GT versions, core component ablations, stride distance ablations, and geographically non-overlapping evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ — A very clear problem definition; the global map comparison in Fig. 1 is exceptionally convincing.
- **Value**: ⭐⭐⭐⭐⭐ — Significantly improves the temporal consistency of HD mapping, refines benchmark data and metrics, and makes prominent contributions to the autonomous driving community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation](mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)
- [\[AAAI 2026\] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors](../../AAAI2026/autonomous_driving/priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)
- [\[ECCV 2024\] Risk-Aware Self-Consistent Imitation Learning for Trajectory Planning in Autonomous Driving](risk-aware_self-consistent_imitation_learning_for_trajectory_planning_in_autonom.md)
- [\[ECCV 2024\] Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention](accelerating_online_mapping_and_behavior_prediction_via_dire.md)
- [\[ECCV 2024\] Stream Query Denoising for Vectorized HD-Map Construction](stream_query_denoising_for_vectorized_hd-map_construction.md)

</div>

<!-- RELATED:END -->
