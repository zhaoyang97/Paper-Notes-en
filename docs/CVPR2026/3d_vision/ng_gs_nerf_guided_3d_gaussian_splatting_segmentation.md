---
title: >-
  [Paper Note] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes the NG-GS framework, which leverages the continuous modeling capability of NeRF to address the boundary discretization problem in 3DGS segmentation. It co…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "segmentation"
  - "NeRF"
  - "boundary refinement"
  - "hash encoding"
date: 2026-05-08
content_hash: 65352c49b6c8f82f
---

# NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation

**Conference**: CVPR 2026 Highlight  
**arXiv**: [2604.14706](https://arxiv.org/abs/2604.14706)  
**Code**: [github.com/BJTU-KD3D/NG-GS](https://github.com/BJTU-KD3D/NG-GS)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, segmentation, NeRF, boundary refinement, hash encoding

## TL;DR

This paper proposes the NG-GS framework, which leverages the continuous modeling capability of NeRF to address the boundary discretization problem in 3DGS segmentation. It constructs a continuous feature field via RBF interpolation, combined with multi-resolution hash encoding and joint NeRF-GS optimization, to achieve high-quality object segmentation.

## Background & Motivation

Although 3DGS has enabled efficient and photorealistic novel view synthesis, its discrete Gaussian representation leads to jagged artifacts at object boundaries during segmentation. Existing 3DGS segmentation methods—including feature distillation, feed-forward inference, and mask lifting—largely overlook the discreteness of Gaussian primitives at boundaries. While directly removing boundary Gaussians with abrupt transitions can improve segmentation, it degrades visual quality. The core insight is to exploit the continuous representation of NeRF to adjust the coordinates and attributes of 3DGS primitives at boundaries.

## Method

### Overall Architecture

The pipeline consists of two stages: (1) **Edge Gaussian Continualization**—ambiguous boundary Gaussians are identified via mask variance analysis, and a continuous feature field is constructed through RBF interpolation and multi-resolution hash encoding; (2) **Joint NeRF-GS Optimization**—alignment loss and spatial continuity loss coordinate the outputs of the two models, ensuring smooth boundary transitions and cross-view consistency.

### Key Designs

1. **Mask Variance Boundary Gaussian Detection**: For each Gaussian point, a set of mask signals is generated from multi-view SAM, and the variance $\sigma_i^2$ of the mask values is computed. Points with variance exceeding threshold $\tau$ are designated as boundary Gaussians, forming the boundary set $\mathcal{B}$. Query points are sampled by expanding the bounding box in the image plane.

2. **RBF Interpolation + Multi-Resolution Hash Encoding**: For each query point, K-NN is used to locate neighboring Gaussians, and RBF kernel-weighted interpolation generates continuous features $\mathbf{f}^{inter}$. Simultaneously, multi-resolution hash encoding extracts coarse-to-fine spatial features $\mathbf{f}^{hash}$. Both are fed into a lightweight NeRF module, where the interpolated features serve as conditioning vectors that modulate the NeRF hidden layers via FiLM.

3. **Joint NeRF-GS Optimization**: An alignment loss enforces RGB color and opacity consistency between 3DGS and NeRF in boundary regions; a continuity loss constrains color consistency among neighboring boundary Gaussian points; a gradient smoothness loss penalizes abrupt changes; and a mask loss applies NeRF density-weighted supervision.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{align} + \lambda_m \mathcal{L}_{mask} + \lambda_c \mathcal{L}_{cont} + \lambda_s \mathcal{L}_{smth}$. Both NeRF and 3DGS are jointly trained using the Adam optimizer. A local 7×7 variance term in the boundary region is used to promote spatial smoothness within the alignment loss.

## Key Experimental Results

### Main Results

| Dataset | Metric | COB-GS | NG-GS | Gain |
|---------|--------|--------|-------|------|
| NVOS | B-mIoU | 79.1% | **84.7%** | +5.6pp |
| NVOS | mIoU | 92.1% | **92.6%** | +0.5pp |
| LERF-OVS | B-mIoU | Prev. SOTA | **+4.4pp** | Significant |
| ScanNet | B-mIoU | Prev. SOTA | **+6.8pp** | Significant |

The proposed method consistently outperforms all baselines on all metrics across three benchmarks, with the most notable gains in boundary mIoU.

### Ablation Study

- RBF interpolation and hash encoding contribute complementary benefits: the former provides continuity while the latter offers multi-scale representation.
- Joint NeRF-GS optimization outperforms optimizing either model independently.
- A mask variance threshold of $\tau=0.6$ achieves the best balance between precision and recall.

### Key Findings

- The substantial gains in boundary mIoU (5–7pp) confirm the effectiveness of the continualization approach.
- Positioning NeRF as a continuous refinement network rather than a replacement for 3DGS proves to be the correct design choice.
- The method generalizes directly to multi-object segmentation scenarios.

## Highlights & Insights

- The idea of complementing NeRF's continuity with 3DGS's efficiency is conceptually novel.
- Mask variance-based automatic boundary Gaussian detection requires no manual annotation.
- The design of using FiLM modulation to incorporate RBF interpolated features into NeRF is concise and effective.

## Limitations & Future Work

- The NeRF module introduces additional training and inference overhead.
- Boundary Gaussian detection depends on the quality of the 2D segmentation model (SAM).
- The computational efficiency of K-NN queries and RBF interpolation in large-scale scenes requires optimization.

## Related Work & Insights

- The complementary NeRF-GS paradigm can be extended to other 3DGS tasks such as editing and generation.
- The application of multi-resolution hash encoding to boundary refinement can be adapted to other fine-grained refinement tasks.
- Mask variance analysis provides a simple yet effective baseline for automatic boundary detection.

## Rating

7/10 — The method is elegantly designed with significant boundary mIoU improvements, though the additional computational overhead warrants consideration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2026\] Action-guided Generation of 3D Functionality Segmentation Data](action-guided_generation_of_3d_functionality_segmentation_data.md)
- [\[CVPR 2026\] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](directfisheye-gs_enabling_native_fisheye_input_in_gaussian_splatting_with_cross-.md)
- [\[CVPR 2026\] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)
- [\[NeurIPS 2025\] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](../../NeurIPS2025/3d_vision/haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)

</div>

<!-- RELATED:END -->
