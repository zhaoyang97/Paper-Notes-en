---
title: >-
  [Paper Note] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas
description: >-
  [CVPR 2026][3D Vision][Open-vocabulary segmentation] This paper proposes JOPP-3D, the first framework for joint open-vocabulary semantic segmentation on 3D point clouds and panoramic images. It maps panoramas onto icosahedron faces via tangential decomposition, extracts semantically aligned 3D instance embeddings using SAM and CLIP, and achieves 80.9% mIoU on S3DIS under weak supervision, surpassing all closed-vocabulary methods.
tags:
  - CVPR 2026
  - 3D Vision
  - Open-vocabulary segmentation
  - point cloud semantic segmentation
  - panoramic images
  - vision-language models
  - cross-modal alignment
date: 2026-05-08
content_hash: 9e6003f06df2b10e
---

# JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas

**Conference**: CVPR 2026
**arXiv**: [2603.06168](https://arxiv.org/abs/2603.06168)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: Open-vocabulary segmentation, point cloud semantic segmentation, panoramic images, vision-language models, cross-modal alignment

## TL;DR

This paper proposes JOPP-3D, the first framework for joint open-vocabulary semantic segmentation on 3D point clouds and panoramic images. It maps panoramas onto icosahedron faces via tangential decomposition, extracts semantically aligned 3D instance embeddings using SAM and CLIP, and achieves 80.9% mIoU on S3DIS under weak supervision, surpassing all closed-vocabulary methods.

## Background & Motivation

Semantic understanding of complex real-world environments is a fundamental requirement for autonomous systems. However, existing approaches face three key challenges:

- **Annotation bottleneck**: Large-scale annotation of 3D point clouds and panoramic images is extremely costly, especially in dynamic and unstructured environments.
- **Modality isolation**: Existing segmentation methods typically operate on either 2D images or 3D point clouds, lacking a unified cross-modal understanding framework.
- **Closed-vocabulary limitation**: Traditional methods are constrained by predefined category sets and cannot generalize to novel object categories.

Although vision-language models such as CLIP support open-vocabulary inference, they are designed for perspective images and cannot be directly applied to panoramic images (which suffer from severe geometric distortion) or 3D point clouds. No prior work has simultaneously achieved open-vocabulary semantic segmentation on both 3D point clouds and panoramic images.

## Method

### Overall Architecture

The pipeline consists of three stages: (1) **Tangential Decomposition** — projecting panoramas onto the 20 faces of a regular icosahedron to generate tangential perspective views and 3D point clouds; (2) **3D Instance Extraction and Semantic Alignment** — obtaining class-agnostic 3D instances via Mask3D/SAM3D and extracting semantic embeddings using SAM and CLIP; (3) **3D-to-Panorama Semantic Back-Projection** — mapping 3D semantics back to the panorama domain via depth correspondence.

### Key Designs

1. **Tangential Decomposition**: The panorama $\mathbf{I}^p$ and depth map $\mathbf{D}^p$ are projected onto the 20 faces of a regular icosahedron, each with a field of view of $\text{FOV}=100°$ (larger than the $73.1°$ of prior methods), with inter-view overlap introduced to mitigate boundary discontinuities. For each pixel, the face-rotated camera-space direction is computed, converted to spherical coordinates, and mapped to equirectangular coordinates:
$$u_r = \frac{\theta_r + \pi}{2\pi}W_e, \quad v_r = \frac{\phi_r + \pi/2}{\pi}H_e$$
Depth values are corrected to Z-depth and transformed to world coordinates via camera pose. All 20 faces are aggregated and voxelized to produce the global colored 3D reconstruction $\mathcal{P}^{3D}$. **Design Motivation**: A larger FOV provides richer contextual coverage while remaining compatible with VLMs.

2. **3D Instance Semantic Alignment**: For each 3D instance mask $\mathcal{M}_j$, its points are projected onto each tangential perspective view, and the Top-K views with the most matching pixels are selected. SAM is invoked with the matched coordinates as reference points to generate instance crops, which are then masked and fed into the CLIP image encoder. The semantic embedding of the 3D instance is obtained by aggregating CLIP features across K views:
$$\mathbf{e}_j^{3D} = \frac{1}{K}\sum_{k=1}^{K} \frac{\text{CLIP}(\mathbf{S}_{j,k} \odot \mathbf{C}_{j,k})}{\|\text{CLIP}(\mathbf{S}_{j,k} \odot \mathbf{C}_{j,k})\|_2}$$
**Design Motivation**: Instance mask isolation prevents semantic contamination from surrounding objects; multi-view aggregation improves embedding robustness.

3. **Depth-Correspondence-Based 3D-to-Panorama Semantic Back-Projection**: Panorama depth pixels are transformed to global coordinates, and semantic labels are assigned via nearest-neighbor matching: $q^p(u,v) = q_{i^*}$, where $i^* = \mathcal{N}(\mathbf{X}^p(u,v))$. Depth-correspondence consistency propagation is applied across overlapping depth regions between adjacent panoramas, completing missing pixel semantics from neighboring scenes and ensuring semantic continuity in depth-discontinuous regions such as doorways and corridors.

### Loss & Training

- **Weakly supervised variant JOPP-3D**: Uses a frozen Mask3D pretrained on S3DIS Areas 1–4 and 6 to provide 3D instance proposals.
- **Unsupervised variant JOPP-3D(u)**: Replaces Mask3D with SAM3D, requiring no annotated data.
- Neither variant requires semantic label supervision — semantic understanding relies entirely on zero-shot inference from CLIP.
- Natural language queries support arbitrary categories; ambiguous categories (e.g., "board") are replaced with more specific alternatives (e.g., "white board").

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|---------|----------|------|
| S3DIS 3D (weak supervision) | mIoU | **80.9%** | 77.4% (Concerto, fully supervised closed-vocab) | +3.5% |
| S3DIS 3D (weak supervision) | mAcc | **87.0%** | 85.0% (Concerto, fully supervised closed-vocab) | +2.0% |
| S3DIS 3D (unsupervised) | mIoU | 59.4% | 36.7% (OpenMask3D, weakly supervised) | +22.7% |
| Stanford-2D-3D-s Panorama (weak supervision) | mIoU | **70.1%** | 61.7% (PanoSAMic, fully supervised closed-vocab) | +8.4% |
| Stanford-2D-3D-s Panorama (weak supervision) | Open mIoU | **74.6%** | 62.8% (SAM3, unsupervised) | +11.8% |
| Stanford-2D-3D-s Panorama (unsupervised) | mIoU | 52.8% | 41.1% (OPS, weakly supervised) | +11.7% |
| ToF-360 3D (unsupervised) | mIoU | **30.9%** | 23.2% (SFSS-MMSI, unsupervised closed-vocab) | +7.7% |
| ToF-360 Panorama (unsupervised) | mIoU | **30.7%** | 27.5% (HoHoNet, unsupervised closed-vocab) | +3.2% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| JOPP-3D (Mask3D instances) | mIoU 80.9% (3D), 70.1% (Panorama) | Full weakly supervised model |
| JOPP-3D(u) (SAM3D instances) | mIoU 59.4% (3D), 52.8% (Panorama) | Unsupervised variant; still outperforms prior open-vocabulary methods |
| Masked CLIP crop | Effective | Masking background prevents semantic contamination and improves embedding quality |
| Depth correspondence propagation | Effective | Resolves semantic incompleteness in doorway/corridor regions |
| FOV=100° vs. 73.1° | 100° superior | Larger field of view provides richer contextual coverage |

### Key Findings

- The weakly supervised open-vocabulary method (80.9%) outperforms the fully supervised closed-vocabulary state of the art (77.4%), demonstrating the remarkable zero-shot reasoning potential of VLMs.
- The quality of 3D instance proposals is the primary performance bottleneck: the gap between Mask3D (weak supervision) and SAM3D (unsupervised) is 21.5% mIoU.
- Mask-based CLIP crop isolation is critical for embedding quality — it prevents semantic confusion among multiple objects within wide-FOV views.
- Depth correspondence propagation effectively resolves spatial discontinuities in panoramic semantics.

## Highlights & Insights

- **First joint solution**: JOPP-3D is the first framework to simultaneously achieve open-vocabulary semantic segmentation on both 3D point clouds and panoramic images, with the two modalities mutually reinforcing each other.
- **Training-free tangential decomposition**: Geometric transformation of panoramas into perspective views eliminates the training overhead required for learning deformations, directly enabling compatibility with SAM and CLIP.
- **Weak supervision surpasses full supervision**: The open-vocabulary weakly supervised approach outperforms all closed-vocabulary fully supervised methods on S3DIS, highlighting the impressive zero-shot capability of VLMs.
- **Modular design**: The 3D instance extractor can be flexibly replaced (Mask3D → SAM3D) to accommodate different supervision conditions.

## Limitations & Future Work

- The framework relies heavily on the quality of 3D instance proposals — Mask3D still requires pretraining data, and SAM3D incurs a notable performance drop in the unsupervised setting.
- The nearest-neighbor-based 3D-to-panorama back-projection may introduce noisy labels when depth estimates are inaccurate.
- Evaluation is limited to indoor scenes (S3DIS, ToF-360); applicability to large-scale outdoor environments has not been validated.
- Inference speed is not reported; per-instance encoding with SAM and CLIP may be slow for large-scale scenes.

## Related Work & Insights

- OpenMask3D addresses only 3D instance segmentation; this work extends it to full semantic segmentation with panoramic semantic back-projection, forming a complete cross-modal understanding pipeline.
- The tangential decomposition approach (icosahedron projection) offers a general, training-free adaptation strategy for applying foundation models to panoramic images.
- The use of 3D instance masks as semantic aggregation units (rather than per-point embeddings) effectively reduces noise and merits adoption in other 3D-VLM tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First joint open-vocabulary segmentation framework for 3D point clouds and panoramas; tangential decomposition combined with depth back-projection is an effective design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two datasets (S3DIS and ToF-360) with dual 3D and 2D assessment under both weakly supervised and unsupervised paradigms.
- Writing Quality: ⭐⭐⭐ Overall structure is clear, but some descriptions are verbose; mathematical derivations are detailed yet lack conciseness.
- Value: ⭐⭐⭐⭐ The result of weak supervision surpassing full supervision is highly convincing, and the framework exhibits strong generalizability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)
- [\[CVPR 2026\] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2026\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)

<!-- RELATED:END -->
