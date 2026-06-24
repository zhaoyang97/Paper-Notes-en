---
title: >-
  [Paper Note] SimPB: A Single Model for 2D and 3D Object Detection from Multiple Cameras
description: >-
  [ECCV 2024][Autonomous Driving][Multi-view Detection] The authors propose SimPB, a unified model that concurrently performs multi-camera 2D detection and BEV-space 3D detection using a hybrid decoder (multi-view 2D decoder + 3D decoder) in a cyclic 3D→2D→3D manner, achieving excellent results on both tasks on the nuScenes dataset.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Multi-view Detection"
  - "Joint 2D-3D Detection"
  - "Hybrid Decoder"
  - "Query Allocation and Aggregation"
  - "Transformer"
date: 2026-05-08
content_hash: 1a9ae8438b89ceaf
---

# SimPB: A Single Model for 2D and 3D Object Detection from Multiple Cameras

**Conference**: ECCV 2024  
**arXiv**: [2403.10353](https://arxiv.org/abs/2403.10353)  
**Code**: [https://github.com/nullmax-vision/SimPB](https://github.com/nullmax-vision/SimPB)  
**Area**: Autonomous Driving  
**Keywords**: Multi-view Detection, Joint 2D-3D Detection, Hybrid Decoder, Query Allocation and Aggregation, Transformer

## TL;DR

The authors propose SimPB, a unified model that concurrently performs multi-camera 2D detection and BEV-space 3D detection using a hybrid decoder (multi-view 2D decoder + 3D decoder) in a cyclic 3D→2D→3D manner, achieving excellent results on both tasks on the nuScenes dataset.

## Background & Motivation

Multi-view 3D object detection is a core task in autonomous driving perception. Existing methods can be categorized into two paradigms:

**Pure 3D detection** (Fig. 1a): Direct calculation of BEV 3D predictions from multi-view images, which fails to fully exploit the mature benefits of 2D detectors.

**Two-stage methods** (Fig. 1b): An independent 2D detector is first used to obtain predictions, which are then utilized for 3D query initialization or token selection (e.g., MV2D, Far3D, Focal-PETR).

Two-stage approaches suffer from three intrinsic issues:

- **Locality**: 2D detectors treat the same object observed by different cameras as independent instances, causing the 3D detector to easily focus on local rather than global information.
- **Unidirectional Interaction**: 2D information is used only once during query initialization, preventing subsequent 3D decoder layers from iteratively updating the 2D semantics.
- **Inconsistent Optimization**: 2D and 3D detectors have different architectures (e.g., CNN vs. Transformer), making joint optimization challenging.

The core idea of SimPB is to design a unified end-to-end model where 2D and 3D detection interact cyclically (3D→2D→3D) within a hybrid decoder, continuously updating and refining their correlation.

## Method

### Overall Architecture

SimPB follows a DETR-like framework:

1. **Backbone + Encoder**: A shared backbone (ResNet/V2-99) extracts multi-view multi-scale features, which are then enhanced by a deformable transformer encoder.
2. **Hybrid Decoder**: The core innovation. Each hybrid layer consists of one multi-view 2D decoder layer and one 3D decoder layer, with a total of 6 layers stacked alternately.
3. **Temporal Fusion**: Adopts the top-K historical query propagation from StreamPETR.

The inputs are $N$ 3D queries (containing 3D anchors), which iteratively undergo Dynamic Query Allocation $\rightarrow$ 2D decoder $\rightarrow$ Adaptive Query Aggregation $\rightarrow$ 3D decoder.

### Key Designs

1. **Dynamic Query Allocation**: Dynamically allocates 3D queries to each camera to form 2D queries. Using the camera's intrinsic and extrinsic parameters, the $K$ keypoints (center + 8 corner points) of the 3D anchors are projected onto each image plane. Validity is determined by whether these projected points fall within the image boundary. A 3D→2D mapping matrix $T \in \mathbb{R}^{N \times M}$ is constructed:
    $Q_{2d} = T^T \cdot Q_{3d}$
   where $M$ is the total number of valid 2D queries across all cameras (which varies dynamically). Compared to uniform allocation, dynamic allocation leverages camera geometric information, making it more precise.

2. **Query-Group Attention**: The 2D queries are grouped by camera. Within-group self-attention and cross-attention are restricted using an attention mask—queries within the same group can attend to each other, while cross-group queries cannot. This prevents 2D queries from different cameras from interfering with each other:
    $\mathbf{X} = \text{softmax}(\mathcal{M} + \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{C}})\mathbf{V}$
   where $\mathcal{M}(i,j) = 0$ (same group) or $-\infty$ (different groups).

3. **Adaptive Query Aggregation**: After 2D detection, the 2D queries are aggregated back into 3D queries. First, the 2D queries are enhanced using a truncation indicator: $\tilde{Q}_{2d} = Q_{2d} \cdot \text{MLP}(\text{Concat}(Q_{2d}, \mathbb{1}_{center}))$. Then, they are weighted-averaged using the mapping matrix:
    $Q_{2d}^{fused} = \frac{T \cdot \tilde{Q}_{2d}}{\sum_j^M T_j}$
   They are then merged into 3D queries using residual connections and self-attention: $Q_{3d}^{agg} = \text{Self-Attn}(Q_{3d} + Q_{2d}^{fused})$. Additional 3D auxiliary supervision is applied to further guide the aggregated queries.

### Loss & Training

Total Loss = 2D detection loss + 3D detection loss:

- $\mathcal{L}_{3d}$: Standard 3D detection loss + depth map auxiliary supervision.
- $\mathcal{L}_{2d} = \mathcal{L}_{detr2d} + \lambda_{alpha} \mathcal{L}_{alpha}$, where $\mathcal{L}_{alpha}$ is the 3D bounding box observation angle loss (sin/cos encoded), with $\lambda_{alpha}=0.5$.

Training details:
- Each hybrid layer contains one layer for each task, totaling 6 layers, with loss weights set to $L\_2d = 1, L\_3d = 1$.
- AdamW, lr = $4 \times 10^{-4}$, batch size 48, 8$\times$ A800 GPUs.
- Trained for 100 epochs (main experiments) / 24 epochs (ablations).
- No TTA, CBGS, or future frame enhancement.

## Key Experimental Results

### Main Results (nuScenes val, 3D Detection)

| Method | Backbone | Resolution | mAP | NDS | mAOE |
|------|----------|--------|-----|-----|------|
| StreamPETR | R50 | 704×256 | 43.2 | 53.7 | 0.445 |
| SparseBEV | R50 | 704×256 | 43.2 | 54.5 | 0.396 |
| Sparse4Dv3 | R50 | 704×256 | 46.9 | 56.1 | 0.476 |
| **SimPB** | **R50** | **704×256** | **47.5** | **58.1** | **0.355** |
| Sparse4Dv3† | R101 | 1408×512 | 53.7 | 62.3 | 0.306 |
| **SimPB†** | **R101** | **1408×512** | **53.9** | **62.9** | **0.280** |

### 2D Detection Results (nuScenes val)

| Method | Backbone | AP | AP50 | AP75 |
|------|----------|-----|------|------|
| DeformableDETR | R50 | 23.0 | 46.5 | 20.1 |
| MV2D† | R50 | 22.6 | 45.6 | 19.8 |
| **SimPB†** | **R50** | **25.6** | **49.5** | **23.7** |
| MV2D† | R101 | 27.1 | 52.3 | 25.0 |
| **SimPB†** | **R101** | **28.8** | **54.1** | **27.6** |

### Ablation Study

| Configuration (L2d, L3d, Lhybrid) | mAP | NDS | Description |
|---------------------------|-----|-----|------|
| (0, 1, 6) Pure 3D decoder | 39.7 | 50.4 | Baseline |
| (1, 0, 6) Pure 2D decoder | 39.7 | 50.3 | 2D + 3D auxiliary supervision is surprisingly close to pure 3D |
| (3, 3, 1) Two-stage cascade | 41.9 | 52.3 | 2D→3D unidirectional |
| **(1, 1, 3) Cyclic interaction** | **42.1** | **52.7** | **Optimal Configuration of SimPB** |

| Query Allocation Strategy | mAP | NDS |
|----------------|-----|-----|
| Uniform allocation | 36.5 | 47.4 |
| Center point only | 41.0 | 51.5 |
| Center + front/back center points | 41.4 | 52.0 |
| **Center + 8 corner points** | **42.1** | **52.7** |

### Key Findings

- SimPB uses only 3 layers of 3D decoders (with the other 3 layers being 2D decoders) yet outperforms other methods that use 6 layers of 3D decoders, demonstrating that information interaction in the 2D decoder is highly effective.
- The performance of the pure 2D decoder configuration with 3D auxiliary supervision (Experiment B) is remarkably close to that of the pure 3D decoder (Experiment A), which validates the effectiveness of Dynamic Query Allocation + Adaptive Query Aggregation.
- Cyclic 3D→2D→3D interaction (F) outperforms the two-stage cascaded 2D→3D (E), confirming the value of continuous interactive learning.
- The observation angle loss $\mathcal{L}_{alpha}$ significantly reduces the orientation error mAOE (from 0.611 to 0.492), with $\lambda_{alpha}=0.5$ being optimal.
- SimPB demonstrates robust generalization capabilities: integrating it into DETR3D brings an improvement of +2.4% mAP / +5.2% NDS, and into Sparse4Dv2 yields +3.1% mAP / +3.0% NDS.

## Highlights & Insights

1. **Unified Paradigm (Single 2D+3D Model)**: Introduces the first end-to-end single model that simultaneously outputs multi-view 2D and BEV 3D detection results, avoiding the information loss inherent in two-stage approaches.
2. **Exquisite Cyclic Interaction Design**: The cyclic 3D→2D→3D iteration allows the two tasks to mutually benefit each other, rather than passing information unidirectionally.
3. **Dynamic Query Allocation Utilizing Camera Geometry**: Determines which cameras each query belongs to based on the 3D anchor projection, leading to a 5.6% mAP improvement over uniform allocation.
4. **Truncation-aware Aggregation**: Introduces a truncation indicator during 2D→3D aggregation, helping the model to better handle objects truncated at camera boundaries.
5. **Plug-and-Play Compatibility**: SimPB's 2D decoder can replace some decoder layers of existing 3D detectors, exhibiting high generalizability.

## Limitations & Future Work

1. **Increased Inference Latency**: The query allocation and aggregation processes can become bottlenecks, which is also pointed out by the authors themselves.
2. Evaluation is limited to the nuScenes dataset; verification on other datasets like Waymo or Argoverse was not conducted.
3. The ground truth (GT) for 2D detection is projected from 3D annotations, which may introduce inconsistencies caused by occlusions in real-world scenarios.
4. Deeper 2D-3D coupling remains to be explored (e.g., initiating the interaction during the encoder stage).

## Related Work & Insights

- The two-stage ideas of MV2D/Far3D inspired SimPB, but SimPB upgrades the unidirectional 2D→3D pipeline to a cyclic 3D→2D→3D architecture.
- The set prediction paradigm of DETR/Deformable-DETR provides the foundation for unifying 2D and 3D detection.
- The temporal query propagation mechanism of StreamPETR is directly adapted.
- The grouping concept of query-group attention can be transferred to multi-sensor fusion scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified 2D+3D end-to-end detection paired with the cyclic interactive design is novel; Dynamic Query Allocation is elegant and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive ablation studies (decoder configuration, allocation strategy, aggregation strategy, observation angle loss, and generalization capability) and evaluations on both 2D and 3D tasks are provided.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly structured, with Fig. 1 offering an intuitive comparison of the three paradigms.
- **Value**: ⭐⭐⭐⭐ — Offers a fresh perspective on unified 2D+3D modeling for multi-view perception, with robust generalization capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene](approaching_outside_scaling_unsupervised_3d_object_detection_from_2d_scene.md)
- [\[ECCV 2024\] MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection](monowad_weather-adaptive_diffusion_model_for_robust_monocular_3d_object_detectio.md)
- [\[ECCV 2024\] Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains](rethinking_lidar_domain_generalization_single_source_as_multiple_density_domains.md)
- [\[ECCV 2024\] OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)
- [\[ECCV 2024\] Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance](weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)

</div>

<!-- RELATED:END -->
