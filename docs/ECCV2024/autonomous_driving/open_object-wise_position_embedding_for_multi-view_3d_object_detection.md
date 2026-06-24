---
title: >-
  [Paper Note] OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection
description: >-
  [ECCV 2024][Autonomous Driving][Multi-view 3D Detection] OPEN is proposed to predict object center depth from pixel-specific depth priors using an Object-wise Depth Encoder (ODE), and design an Object-wise Position Embedding (OPE) to inject this information into the Transformer decoder to generate 3D object-aware features, achieving state-of-the-art performance of 64.4% NDS on nuScenes.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Multi-view 3D Detection"
  - "Depth Estimation"
  - "Position Embedding"
  - "DETR"
  - "nuScenes"
date: 2026-05-08
content_hash: 4194efd366dd1df1
---

# OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection

**Conference**: ECCV 2024  
**arXiv**: [2407.10753](https://arxiv.org/abs/2407.10753)  
**Code**: [https://github.com/AlmoonYsl/OPEN](https://github.com/AlmoonYsl/OPEN)  
**Area**: Autonomous Driving  
**Keywords**: Multi-view 3D Detection, Depth Estimation, Position Embedding, DETR, nuScenes

## TL;DR

OPEN is proposed to predict object center depth from pixel-specific depth priors using an Object-wise Depth Encoder (ODE), and design an Object-wise Position Embedding (OPE) to inject this information into the Transformer decoder to generate 3D object-aware features, achieving state-of-the-art performance of 64.4% NDS on nuScenes.

## Background & Motivation

Accurate depth information is crucial for multi-view 3D detection. Existing methods utilize LiDAR projection points for pixel-wise depth supervision, but suffer from two overlooked problems:

**Mismatch in Depth Distribution**: Depth obtained from LiDAR projections is distributed on the object surface, whereas object queries in DETR-based detectors are defined at the object center. Surface depth $\neq$ object center depth, leading to misalignment between the supervision signals and detection targets.

**Difficulty for Distant Objects**: It is extremely challenging to perform overall fine-grained depth estimation for distant objects, whereas predicting only the object center depth is relatively easier.

Limitations of existing position embedding schemes:
- **Ray-aware PE** (StreamPETR): Generates a 3D mesh grid in the camera frustum, leaving depth candidates uncertain.
- **Point-aware PE** (3DPPE): Encodes pixel-wise depth predictions, ignoring the importance of object-wise depth.

## Method

### Overall Architecture

OPEN is built upon the StreamPETR baseline and consists of three core components: Pixel-wise Depth Encoder (PDE) $\rightarrow$ Object-wise Depth Encoder (ODE) $\rightarrow$ Object-wise Position Embedding (OPE), progressively estimating and injecting depth information from coarse to fine.

### Key Designs

1. **Pixel-wise Depth Encoder (PDE)**:

    - Takes multi-view features $\mathbf{F}_i \in \mathbb{R}^{C \times H \times W}$ as input, using an MLP to encode camera intrinsics $\mathbf{K}$ to modulate the features.
    - Predicts a pixel-wise depth map $\mathbf{D}_i \in \mathbb{R}^{H \times W \times 1}$ via DepthNet (residual blocks + deformable convolution).
    - Fuses regressed depth and probabilistic depth to generate the final pixel-wise depth.
    - Uses the $8\times$ downsampled depth map of LiDAR projection points as supervision.
    - **Design Motivation**: Pixel-wise depth serves as a prior for subsequent object-wise depth prediction, providing full-scene depth awareness.

2. **Object-wise Depth Encoder (ODE)**:

    - Projects pixel coordinates $(u,v)$ combined with pixel depth to camera coordinates: $\mathbf{p}_{(m,n)} = \mathbf{K}^{-1}(u \times D, v \times D, D, 1)^T$.
    - Predicts $k=13$ 3D offsets from image features, which are added to the reference points to obtain 3D sampling points.
    - Projects 3D sampling points back to pixel coordinates of the current and prior frames, sampling and weight-aggregating features: $\mathbf{E}_{(m,n)} = \phi(\sum_{j=1}^k \mathbf{A}_j \cdot \text{Concat}(\mathbf{F}_i(\mathbf{p}^*), \mathbf{F}'_i(\mathbf{p}^*)))$.
    - Feeds depth embeddings and image features into an FFN to predict object-wise depth $\mathbf{d} \in \mathbb{R}^{(H \times W) \times 1}$ and object centers $\mathbf{c} \in \mathbb{R}^{(H \times W) \times 2}$.
    - **Mechanism**: Aggregates temporal and spatial neighborhood information via attention to reason about object center depth from object surface depth.
    - **Supervision**: Supervised using the center depth annotations projected from 3D ground truth bounding boxes.

3. **Object-wise Position Embedding (OPE)**:

    - Concatenates object centers and object-wise depths to get $\mathbf{o}_j = (x, y, d_j)$.
    - Transforms coordinates to the LiDAR coordinate system: $\mathbf{O}_j = \mathbf{R}^{-1} \mathbf{K}^{-1} \mathbf{o}'_j$.
    - Generates position embeddings using 3D cosine position encoding + MLP after normalization: $\mathbf{OPE}_j = \text{MLP}(\text{PE}_{3D}(\text{Norm}(\mathbf{O}_j)))$.
    - Added to the corresponding image features to interact with object queries in the Transformer decoder.
    - **Key Advantage**: Compared to the uncertain depth of ray-aware PE and surface depth of point-aware PE, OPE directly encodes the 3D position of the object center, aligning perfectly with the definition of DETR queries.

4. **Depth-aware Focal Loss (DFL)**:

    - Introduces a depth score $\mathbf{s} = e^{-\text{L2}(\hat{\mathbf{C}} - \mathbf{C})}$ to measure the distance between the predicted center and the GT center.
    - Modulates the classification label of focal loss using $\mathbf{s}$ as soft labels, coupling classification confidence with localization accuracy.
    - Encourages the network to focus more on 3D object center information.

### Loss & Training

$$\mathcal{L} = \lambda_1 \mathcal{L}_{PDE} + \lambda_2 \mathcal{L}_{ODE} + \lambda_3 \mathcal{L}_{DFL} + \lambda_4 \mathcal{L}_{reg}$$

where $\lambda_1=1.0, \lambda_2=5.0, \lambda_3=2.0, \lambda_4=0.25$. Hungarian Matching is used for label assignment.

Training Details: AdamW, batch size = 16, $8\times$ V100, streaming video training for 90 epochs (val) / 60 epochs (test), initial lr = 4e-4 + cosine annealing, no CBGS.

## Key Experimental Results

### Main Results

**nuScenes Val Set**:

| Method | Backbone | NDS↑ | mAP↑ | mATE↓ | mAVE↓ |
|------|----------|------|------|-------|-------|
| StreamPETR† | R50 | 55.0 | 45.0 | 0.613 | 0.265 |
| SparseBEV† | R50 | 55.8 | 44.8 | 0.581 | 0.247 |
| **OPEN†** | **R50** | **56.4** | **46.5** | **0.573** | **0.235** |
| Far3D† | R101 | 59.4 | 51.0 | 0.551 | 0.238 |
| **OPEN†** | **R101** | **60.6** | **51.6** | **0.528** | **0.222** |

**nuScenes Test Set**:

| Method | Backbone | NDS↑ | mAP↑ |
|------|----------|------|------|
| Sparse4Dv2 | V2-99 | 63.8 | 55.6 |
| StreamPETR | V2-99 | 63.6 | 55.0 |
| **OPEN** | **V2-99** | **64.4** | **56.7** |

### Ablation Study

**Contribution of each component (V2-99, $320\times800$, 24ep)**:

| Configuration | NDS↑ | mAP↑ | Compared to Baseline |
|------|------|------|---------|
| Baseline (StreamPETR) | 59.4 | 50.3 | - |
| +PDE | 59.4 | 50.5 | +0.2 mAP |
| +PDE+ODE | 59.7 | 50.6 | +0.3 NDS |
| +PDE+ODE+OPE | 60.8 | 52.4 | **+1.1 NDS, +1.8 mAP** |
| +PDE+ODE+OPE+DFL | **61.3** | **52.1** | +1.9 NDS, +1.8 mAP |

**Comparison of Position Embeddings**:

| Method | NDS↑ | mAP↑ | NDS > 40m↑ |
|------|------|------|----------|
| Ray-aware PE | 59.4 | 50.3 | 36.8 |
| Point-aware PE | 60.0 | 51.6 | 37.9 |
| **OPE** | **60.8** | **52.4** | **39.1** |

### Key Findings

- **OPE is the core contribution**: In component ablation studies, OPE contributes +1.1 NDS and +1.8 mAP, far exceeding PDE (+0.2 mAP) and ODE (+0.1 mAP).
- **More prominent advantage for long-range objects**: On distant objects ($>40$m), OPE outperforms Ray-aware PE by 2.3 NDS and Point-aware PE by 1.2 NDS, demonstrating that object-wise depth is more significant for distant objects.
- **PDE is an indispensable prior**: Removing PDE degrades NDS by 1.4%, showing that pixel-wise depth is critical as a basis for object-wise depth reasoning.
- **Temporal information assists ODE**: Disabling temporal cues decreases NDS by 0.3%, indicating that temporal cues contribute to more accurate object-wise depth predictions.
- Attention map visualizations reveal that OPE exhibits more focused attention weights on challenging objects (occluded or distant).

## Highlights & Insights

- **Simple yet insight-driven design**: Starting from the straightforward observation that "LiDAR depth is on the surface rather than at the center," the necessity of object-wise depth is derived, leading to a clean and elegant design.
- **Plug-and-play**: OPE can easily replace existing position embedding modules in DETR-based detectors.
- Significantly improves long-range detection performance through better depth representation without adding complex inference pipelines.

## Limitations & Future Work

- The object-wise depth supervision of ODE relies on the projection of 3D ground truth bounding boxes, still requiring accurate 3D annotations.
- Currently, ODE predicts object depth pixel-by-pixel, which is redundant for non-object regions, suggesting sparsification as a viable direction.
- In the DFL experiment, mAP slightly decreases from 52.4 to 52.1, indicating that the soft label strategy might introduce minor noise.
- The integration with LiDAR fusion or BEV-space detectors remains unexplored.

## Related Work & Insights

- Inherits the position embedding concepts from PETR/StreamPETR, but advances from pixel-wise to object-wise.
- The point-aware PE of 3DPPE is a direct preceding work; OPEN addresses its limitation of neglecting object-wise center depth.
- The concepts from Depth-aware Focal Loss (modulating classification loss with geometric information) can be generalized to other 3D tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The approach of object-wise depth + position embedding is original, with accurate observation and logical design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive, involving multiple backbones/resolutions/datasets, component ablation, distance-segmented analysis, PE comparison, and attention visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear illustrations (especially the PE comparisons in Fig. 1 and Fig. 4) along with persuasive arguments on motivation.
- **Value**: ⭐⭐⭐⭐ — Achieves state-of-the-art results on nuScenes; the underlying design concepts hold guiding significance for future multi-view detection research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection](fsd-bev_foreground_self-distillation_for_multi-view_3d_object_detection.md)
- [\[AAAI 2026\] FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection](../../AAAI2026/autonomous_driving/fq-petr_fully_quantized_position_embedding_transformation_fo.md)
- [\[ECCV 2024\] Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance](weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [\[CVPR 2026\] SToRe3D: Sparse Token Relevance in ViTs for Efficient Multi-View 3D Object Detection](../../CVPR2026/autonomous_driving/store3d_sparse_token_relevance_in_vits_for_efficient_multi-view_3d_object_detect.md)

</div>

<!-- RELATED:END -->
