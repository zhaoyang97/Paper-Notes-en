---
title: >-
  [Paper Note] ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation
description: >-
  [CVPR 2025][Autonomous Driving][Occupancy Prediction] This paper proposes ProtoOcc, which enhances the contextual information of low-resolution voxels by mapping 2D image clustering prototypes into the 3D voxel query space via **prototype-aware view transformation**. Together with a **multi-perspective occupancy decoding** strategy, it reconstructs high-resolution 3D occupancy scenes from the enhanced voxels. It achieves competitive performance compared to high-resolution met…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Occupancy Prediction"
  - "Prototype"
  - "View Transformation"
  - "Low-Resolution"
  - "Multi-Perspective Decoding"
date: 2026-05-08
content_hash: 0e4e32464321ad3e
---

# ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation

**Conference**: CVPR 2025  
**arXiv**: [2503.15185](https://arxiv.org/abs/2503.15185)  
**Code**: [https://kuai-lab.github.io/cvpr2025protoocc](https://kuai-lab.github.io/cvpr2025protoocc)  
**Area**: Autonomous Driving / 3D Occupancy Prediction / View Transformation  
**Keywords**: Occupancy Prediction, Prototype, View Transformation, Low-Resolution, Multi-Perspective Decoding  

## TL;DR
This paper proposes ProtoOcc, which enhances the contextual information of low-resolution voxels by mapping 2D image clustering prototypes into the 3D voxel query space via **prototype-aware view transformation**. Together with a **multi-perspective occupancy decoding** strategy, it reconstructs high-resolution 3D occupancy scenes from the enhanced voxels. It achieves competitive performance compared to high-resolution methods (Occ3D mIoU 37.80 vs. PanoOcc 38.11) while using a 75% smaller voxel resolution.

## Background & Motivation
In camera-based 3D occupancy prediction (3DOP), the resolution of voxel queries is crucial for the quality of view transformation. High-resolution voxel queries yield solid performance but represent a heavy computational burden (hampering real-time deployment), whereas low-resolution queries save computation but suffer from severe information loss. Existing methods rely solely on low-level image features for 2D-3D cross-attention during view transformation. When the query resolution decreases, these low-level features are insufficient to support accurate 3D scene reconstruction. Therefore, a method that can still encode rich visual context under low-resolution queries is highly desirable.

## Core Problem
How to maintain high-quality 2D-to-3D view transformation under the constraint of low-resolution voxel queries? Specifically: how to encode and preserve accurate 3D semantic occupancy information using fewer parameters and a smaller spatial volume?

## Method

### Overall Architecture
Multi-view images $\rightarrow$ Image backbone (ResNet50+FPN) extracts multi-scale features $\rightarrow$ **Prototype-aware View Transformation** (clustering images into prototypes $\rightarrow$ mapping to 3D voxel space $\rightarrow$ optimizing prototype quality via contrastive learning) $\rightarrow$ Prototype-enhanced low-resolution voxel queries $\rightarrow$ **Multi-Perspective Occupancy Decoding** (voxel augmentation $\rightarrow$ upsampling $\rightarrow$ consistency regularization) $\rightarrow$ High-resolution 3D occupancy prediction

### Key Designs
1. **Prototype Mapping**: Iterative superpixel clustering is used to group image features into $M$ prototypes (high-level visual structural representations, e.g., layouts and boundaries). Through Feature Aggregation & Dispatch, 2D prototypes are projected into the 3D voxel space by computing a prototype-voxel affinity matrix $\mathbf{A}$, normalizing it with sigmoid, and performing aggregation (prototype $\rightarrow$ voxel) and dispatching (voxel $\rightarrow$ prototype-aware query). This enables low-resolution voxel queries to carry high-level 2D structural information.

2. **Prototype Optimization**: Standard 3DOP losses do not directly optimize clustering quality. A pseudo-mask-based contrastive learning scheme is designed. Specifically, pseudo-semantic masks are generated using SEEDS superpixels or SAM, and a contrastive loss $\mathcal{L}_{cls}$ is computed between prototype-aware pixel features and mask centroids to enhance the discriminative capability among prototypes.

3. **Multi-Perspective Occupancy Decoding**: Transitioning from low-resolution to high-resolution is an ill-posed problem. Voxel augmentation (feature-level: Random Dropout/Gaussian noise; spatial-level: transposition/flipping) is applied to generate multiple "perspectives". After upsampling via weight-shared transposed 3D convolutions, a consistency regularization loss $\mathcal{L}_{cons}$ is enforced among the predictions from different augmented versions (aligning distributions toward a sharpened mean).

### Loss & Training
$$\mathcal{L}_{total} = \sum_{p=0}^{P}(\lambda_1\mathcal{L}_{occ}^{(p)} + \lambda_2\mathcal{L}_{Lov}^{(p)}) + \lambda_3\mathcal{L}_{cls} + \lambda_4\mathcal{L}_{cons}$$
- ResNet50 backbone, 12 epochs, input 432×800
- Base: 100×100×16, Small: 50×50×16, Tiny: 50×50×4

## Key Experimental Results

### Occ3D-nuScenes Validation Set

| Method | Query Size | mIoU |
|------|---------|------|
| PanoOcc (Base) | 100×100×16 | 38.11 |
| **ProtoOcc (Base)** | 100×100×16 | **39.01** |
| PanoOcc (Small) | 50×50×16 | 35.78 |
| **ProtoOcc (Small)** | 50×50×16 | **37.80** |
| PanoOcc (Tiny) | 50×50×4 | 33.99 |
| **ProtoOcc (Tiny)** | 50×50×4 | **35.68** |

**Key Findings**: ProtoOcc-Small (37.80) is almost on par with PanoOcc-Base (38.11) while using a 75% smaller voxel resolution!

### Inference Efficiency (Small Query)

| Method | Inference Time | FLOPs(G) | Parameters | mIoU |
|------|---------|----------|--------|------|
| PanoOcc-Base | 266ms | 1310 | 46.24M | 38.11 |
| ProtoOcc-Small | **105ms** | 378 | 16.11M | 37.80 |

Saves 60% of inference time, 71% of FLOPs, and 65% of parameters.

### SemanticKITTI Validation Set

| Baseline | +ProtoOcc | IoU Gain | mIoU Gain |
|------|------------|--------|---------|
| VoxFormer-S | +ProtoOcc | +0.35 | +0.88 |
| VoxFormer-B | +ProtoOcc | +0.85 | +1.22 |
| Symphonies-S | +ProtoOcc | +1.35 | +0.86 |

ProtoOcc consistently improves all baselines as a plug-and-play module.

### Ablation Study
- **Voxel Mapping Only**: mIoU +0.02 (mapping alone provides minor improvement)
- **+ Prototype Optimization**: mIoU +0.77 (contrastive learning is crucial)
- **Multi-Perspective Decoding Only**: mIoU +1.47 (significant contribution from voxel augmentation and consistency regularization)
- **Full ProtoOcc**: mIoU +2.02
- **Number of Prototypes $M$**: 350 is optimal (either too coarse or too fine degrades performance)
- **Augmentation Combinations**: Random Dropout + consistency regularization yields the best performance (37.25)

## Highlights & Insights
- **Cross-space mapping from prototype to voxel**: First to introduce 2D image prototype representations into the view transformation of 3D occupancy prediction, encoding high-level geometric structures under low-resolution constraints.
- **Significant computational efficiency**: 75% fewer voxels $\rightarrow$ ~60% faster inference with almost no performance degradation. This is highly valuable for real-time deployment in autonomous driving.
- **Plug-and-play**: Consistently improves two baselines (VoxFormer and Symphonies) on SemanticKITTI.
- **Multi-perspective decoding strategy**: Voxel augmentation + consistency regularization is an elegant solution to the ill-posed low-to-high resolution problem.
- **Attention map visualization**: ProtoOcc focuses on small, critical objects (e.g., pedestrians, motorcycles), whereas the baseline is distracted by visually dominant areas.

## Limitations & Future Work
- Prototype clustering uses a simple superpixel method; more advanced clustering (such as learning-based) may yield further improvements.
- The combination of augmentations in multi-perspective decoding requires manual experimental selection.
- Only validated on nuScenes and SemanticKITTI, with other datasets (e.g., Waymo) left untested.
- Limited utilization of temporal information.

## Related Work & Insights
- **PanoOcc**: Standard voxel queries + deformable attention. ProtoOcc achieves ~2% higher mIoU under the same query size and comparable performance with a 75% smaller query.
- **COTR**: Uses large queries first and then downsamples. ProtoOcc directly uses small queries, making it more efficient.
- **DFA3D**: Another method for enhancing view transformation. ProtoOcc-Small outperforms DFA3D (37.80 vs. 36.27) with shorter inference time (105ms vs. 153ms).

## Inspirations & Connections
- The idea of "prototype mapping" can be extended to other tasks requiring 2D-to-3D transition (e.g., 3D detection, 3D segmentation).
- The augmentation + consistency pipeline in multi-perspective decoding is similar to multi-crop + consistency in self-supervised learning, which can be extended to other 3D prediction tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Prototype-to-voxel mapping and multi-perspective voxel decoding are novel designs, though individual components build upon existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive, covering two benchmarks, multi-resolution setups, efficiency analysis, extensive ablation studies, and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, systematic methodology descriptions, and high-quality illustrations.
- **Value**: ⭐⭐⭐⭐⭐ Addresses the core efficiency challenge in the real-time deployment of 3DOP, significantly reducing computation by 75% with almost zero performance loss.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](rethinking_temporal_fusion_with_a_unified_gradient_descent_view_for_3d_semantic_.md)
- [\[CVPR 2025\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](../../ICCV2025/autonomous_driving/evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](../../CVPR2026/autonomous_driving/proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
