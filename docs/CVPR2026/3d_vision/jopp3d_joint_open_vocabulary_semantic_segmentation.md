---
title: >-
  [Paper Note] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas
description: >-
  [CVPR 2026][3D Vision][Open-vocabulary 3D segmentation] This paper proposes JOPP-3D — the first open-vocabulary semantic segmentation framework that jointly processes 3D point clouds and panoramic images. It decomposes p…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Open-vocabulary 3D segmentation"
  - "joint point cloud–panorama segmentation"
  - "icosahedral tangential decomposition"
  - "SAM+CLIP semantic alignment"
  - "3D-to-panorama back-projection"
date: 2026-05-08
content_hash: 3d295fe56ce61e10
---

# JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas

**Conference**: CVPR 2026
**arXiv**: [2603.06168](https://arxiv.org/abs/2603.06168)
**Code**: None
**Area**: 3D Vision
**Keywords**: Open-vocabulary 3D segmentation, joint point cloud–panorama segmentation, icosahedral tangential decomposition, SAM+CLIP semantic alignment, 3D-to-panorama back-projection

## TL;DR

This paper proposes JOPP-3D — the first open-vocabulary semantic segmentation framework that jointly processes 3D point clouds and panoramic images. It decomposes panoramas into 20 perspective views via icosahedral tangential projection to accommodate SAM/CLIP, extracts mask-isolated instance-level CLIP embeddings for 3D semantic segmentation, and back-projects results to the panoramic domain via depth correspondence. Without any training, the method achieves 80.9% mIoU on S3DIS, surpassing all supervised approaches.

## Background & Motivation

**Background**: 3D semantic segmentation relies on large-scale annotations and fixed category sets. Vision-language models such as CLIP excel at open-vocabulary 2D segmentation, but perform poorly when applied directly to panoramas (spherical distortion) and 3D point clouds (lack of pre-training).

**Limitations of Prior Work**:

1. Spherical distortion in panoramas prevents foundation models pre-trained on perspective images (e.g., CLIP, SAM) from being directly applicable.
2. Cubemap-based approaches (6 faces × 90°) introduce boundary discontinuity artifacts; DAN-based adapters require supervised training.
3. Cross-modal alignment of 2D vision-language features to 3D is difficult — direct per-point CLIP encoding introduces substantial semantic noise.
4. Joint open-vocabulary semantic segmentation of panoramas and point clouds has not been explored.

**Key Challenge**: Extending the capabilities of CLIP/SAM to both panoramas and 3D point clouds in a training-free manner is non-trivial, as each modality presents distinct geometric challenges.

**Goal**: To establish a unified framework for open-vocabulary semantic segmentation of both point clouds and panoramas simultaneously.

**Key Insight**: Panoramas are projected onto the 20 tangent faces of a regular icosahedron to generate perspective views compatible with CLIP/SAM. 3D point clouds are reconstructed from these views, semantic alignment is performed at the 3D instance level, and results are back-projected to the panoramic domain.

**Core Idea**: Tangential decomposition → 3D instance extraction → mask-isolated CLIP semantic alignment → depth-correspondence panoramic back-projection.

## Method

### Overall Architecture

A three-stage training-free pipeline: (1) **Tangential Decomposition** — each panoramic RGB-D image is projected onto the 20 faces of a regular icosahedron, yielding 20 tangential perspective views (640×480, FOV=100°) with corresponding depth maps; 3D points from all views are aggregated and voxelized into a global point cloud. (2) **3D Instance Extraction + Semantic Alignment** — Mask3D (weakly supervised) or SAM3D (unsupervised) generates 3D instance proposals; each instance is projected onto its $K$ best tangential views, SAM generates 2D mask crops, CLIP encodes the masked crops, and multi-view features are averaged to obtain the instance semantic embedding. (3) **Language Query + 3D-to-Panorama Back-Projection** — natural language queries produce 3D semantic segmentation results, which are back-projected to the panoramic domain via depth correspondence.

### Key Designs

1. **Icosahedral Tangential Decomposition**

    - Panoramas are projected onto the 20 faces of a regular icosahedron, with each face having a FOV of 100° — exceeding the 73.1° of Eder et al. and the 90° of Cubemap approaches.
    - Overlapping fields of view between adjacent faces eliminate the boundary discontinuity artifacts of Cubemap.
    - Ray directions for each pixel are computed via face rotation matrices, mapped to equirectangular coordinates, and sampled with bilinear interpolation for RGB and nearest-neighbor interpolation for depth.
    - Focal length is determined by the horizontal field of view, maximizing contextual coverage within a geometrically stable range.
    - Local 3D point clouds are reconstructed from all 20 tangent faces and aggregated across all panoramas, then voxelized into a global reconstruction.

2. **Mask-Isolated Instance-Level CLIP Encoding**

    - For each 3D instance, projections onto all tangential views are computed and the $K$ views with the most projected points are selected.
    - SAM is prompted with projected points to generate 2D instance masks and crops.
    - **Masking precedes CLIP encoding** — masks are applied to crops before CLIP ingestion; normalized feature vectors from $K$ views are averaged to obtain the instance semantic embedding.
    - Masking is confirmed as a critical design choice through ablation: without it, semantics from large-area classes (floor/ceiling) severely contaminate other instances, causing Open mIoU to collapse from 74.6% to 33.6%.

3. **Depth-Correspondence 3D-to-Panorama Semantic Back-Projection**

    - Each panoramic depth pixel is back-projected into 3D, and its semantic label is retrieved from the semantic point cloud via nearest-neighbor search.
    - **Cross-scene depth correspondence propagation**: when adjacent panoramas share depth overlap in corridor or doorway regions, semantic labels from labeled neighbor panoramas are propagated to unlabeled regions in the current panorama.
    - This addresses semantic incompleteness in regions of large depth discontinuity (doorways/corridors) that arise from direct nearest-neighbor lookup.

### Loss & Training

JOPP-3D is a **fully training-free** inference pipeline: frozen Mask3D/SAM3D provides 3D instance proposals, frozen SAM performs 2D segmentation, frozen CLIP performs semantic encoding, and natural language queries enable open-vocabulary classification. The weakly supervised variant uses Mask3D pre-trained on S3DIS Areas 1, 2, 3, 4, and 6; the unsupervised variant uses SAM3D. Inference time: 4.8 min/panorama (single RTX A6000); 1.7 seconds per language query.

## Key Experimental Results

### Main Results

**3D Point Cloud Semantic Segmentation**

| Dataset | Method | Supervision | mIoU (%) | mAcc (%) |
|---------|--------|-------------|----------|----------|
| S3DIS | PointTransformerV3 | Fully supervised | 73.4 | 78.9 |
| | Concerto | Fully supervised | 77.4 | 85.0 |
| | OpenMask3D | Weakly supervised | 36.7 | 43.6 |
| | JOPP-3D(u) | Unsupervised | 59.4 | 70.1 |
| | **JOPP-3D** | **Weakly supervised** | **80.9** | **87.0** |
| ToF-360 | SFSS-MMSI | Unsupervised | 23.2 | 46.3 |
| | **JOPP-3D(u)** | **Unsupervised** | **30.9** | **47.5** |

**Panoramic Image Semantic Segmentation**

| Dataset | Method | mIoU (%) | Open mIoU (%) |
|---------|--------|----------|--------------|
| Stanford-2D-3D-s | PanoSAMic (fully supervised) | 61.7 | -- |
| | OPS (weakly supervised) | 41.1 | 42.6 |
| | SAM3 (unsupervised) | 54.2 | 62.8 |
| | **JOPP-3D** | **70.1** | **74.6** |
| ToF-360 | HoHoNet | 27.5 | -- |
| | **JOPP-3D(u)** | **30.7** | **47.4** |

### Ablation Study

| Configuration | Open mIoU (%) | Impact |
|---------------|--------------|--------|
| Full JOPP-3D | **74.6** | -- |
| w/o SAM Mask (direct CLIP without masking) | 33.6 | −41.0 |
| w/o Tangential Decomp. (direct panorama) | 41.4 | −33.2 |
| w/o Depth Correspondence | 67.0 | −7.6 |

### Key Findings

- Masked CLIP encoding contributes remarkably: 33.6% → 74.6% (+41.0%); unmasked CLIP features are severely contaminated by large-area classes.
- Tangential decomposition is indispensable: 41.4% → 74.6% (+33.2%); CLIP/SAM nearly fails on spherically distorted images.
- Depth correspondence provides a +7.6% gain, with the most significant improvements in doorway and corridor regions.
- Mask3D vs. SAM3D: weakly supervised 74.6% vs. unsupervised 59.9%; high-quality 3D instance proposals are the primary performance bottleneck.
- The open-vocabulary approach can retrieve fine-grained objects labeled as "clutter" in ground truth (clocks, posters, etc.), demonstrating practical value.

## Highlights & Insights

- The first open-vocabulary segmentation framework to jointly handle 3D point clouds and panoramic images, surpassing all supervised methods without any training.
- The icosahedral tangential decomposition is an elegant design: 100° FOV provides better contextual coverage and fewer boundary artifacts than Cubemap.
- The +41.0% ablation result from masked CLIP encoding is striking — a simple modification with a substantial effect.
- The concept of using 3D as a consistency "anchor" for 2D is generalizable to video understanding, multi-view consistent segmentation, and related tasks.

## Limitations & Future Work

- Requires RGB-D panoramic input; scenes with only RGB panoramas cannot be processed.
- The weakly supervised Mask3D variant requires pre-training data; cross-domain generalization (e.g., outdoor scenes) remains unverified.
- Inference speed is slow (4.8 min/image), making real-time application impractical.
- Coarse labels such as "clutter" penalize open-vocabulary methods for their fine-grained recognition ability in quantitative evaluation.
- Validation is limited to indoor scenes; applicability to large-scale outdoor environments has not been explored.

## Related Work & Insights

- **vs. OpenMask3D**: Both target open-vocabulary 3D segmentation, but OpenMask3D operates on perspective RGB-D sequences for instance segmentation, whereas this work addresses scene-level semantic segmentation from panoramas and point clouds — 80.9% vs. 36.7% mIoU.
- **vs. OPS**: OPS requires training a DAN adapter to handle panoramic distortion; this work's training-free tangential decomposition outperforms it (70.1% vs. 41.1%), and OPS does not perform 3D segmentation.
- **vs. SAM3**: An RGB-only method achieving 54.2% mIoU on panoramas; this work reaches 70.1% by incorporating depth information and 3D alignment.
- Insights: Tangential decomposition combined with foundation models constitutes a general paradigm for panoramic image processing; mask-cropped CLIP instance-level semantic alignment is applicable to any task requiring open-vocabulary instance-level features.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to propose joint open-vocabulary segmentation of point clouds and panoramas; tangential decomposition and depth correspondence are novel designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets, dual-task (2D+3D) evaluation, four ablation conditions, and extensive qualitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework presentation, high-quality figures and tables, systematic method description.
- **Value**: ⭐⭐⭐⭐⭐ Training-free approach surpasses supervised methods; tangential decomposition and masked CLIP paradigm are broadly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)
- [\[CVPR 2026\] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2026\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weaklysupervised_s.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)

</div>

<!-- RELATED:END -->
