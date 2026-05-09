---
title: >-
  [Paper Note] SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting
description: >-
  [ICCV 2025][LLM Evaluation][3D edge reconstruction] This paper proposes SketchSplat, which represents 3D edges as parametric sketches (line segments + Bézier curves) and directly optimizes edge parameters via differentiable rendering by sampling Gaussian points from sketches. Combined with adaptive topology control and an improved 2D edge detector, the method achieves state-of-the-art accuracy, completeness, and compactness on CAD datasets.
tags:
  - ICCV 2025
  - LLM Evaluation
  - 3D edge reconstruction
  - differentiable rendering
  - Gaussian splatting
  - parametric curves
  - CAD modeling
date: 2026-05-08
content_hash: c074ba4d2dc30202
---

# SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting

**Conference**: ICCV 2025  
**arXiv**: [2503.14786](https://arxiv.org/abs/2503.14786)  
**Code**: [https://oceanying.github.io/SketchSplat](https://oceanying.github.io/SketchSplat)  
**Area**: LLM Evaluation  
**Keywords**: 3D edge reconstruction, differentiable rendering, Gaussian splatting, parametric curves, CAD modeling

## TL;DR

This paper proposes SketchSplat, which represents 3D edges as parametric sketches (line segments + Bézier curves) and directly optimizes edge parameters via differentiable rendering by sampling Gaussian points from sketches. Combined with adaptive topology control and an improved 2D edge detector, the method achieves state-of-the-art accuracy, completeness, and compactness on CAD datasets.

## Background & Motivation

3D edge reconstruction is a fundamental task in computer vision and graphics, underpinning applications such as CAD modeling, SLAM, and autonomous driving. Existing methods fall into two categories:

**Traditional methods** (line matching + triangulation): accurate but unable to handle lines and curves simultaneously, and multi-view matching is not robust.

**Differentiable rendering methods** (NEF, EMAP, EdgeGS): first optimize a set of 3D edge points, then fit parametric edges. The fitting step relies solely on reconstructed 3D points without referencing 2D edge images, leading to:
   - Noise in the point set causing broken fitted edges
   - Misalignment between fitted 3D edges and 2D images

Core motivation: Can **parametric 3D edges be optimized directly**, aligning the optimization process with 2D edge images through differentiable rendering?

## Method

### Overall Architecture

Multi-view RGB images → 2D edge detection (proposed 2DGS-SN method) → 3D edge point initialization via EdgeGS → fitting into parametric sketches → sampling Gaussian points from sketches → differentiable rasterization → L1 loss computation → backpropagation to optimize sketch parameters + topology operations → output optimized parametric 3D edges

### Key Designs

1. **Sketch Representation and Differentiable Optimization**:

    - 3D edges are represented as two types of sketches: line segments $l \in \mathbb{R}^{2 \times 3}$ (2 control points) and cubic Bézier curves $c \in \mathbb{R}^{4 \times 3}$ (4 control points)
    - Each sketch carries optimizable opacity $o$ and local scale $s \in \mathbb{R}^3$
    - **Core innovation**: the sampling step — points are sampled from each sketch at a fixed step size (5 mm), each inheriting the opacity/scale of its parent sketch, with the principal direction set to the local tangent of the sketch
    - The sampled Gaussian points are rendered to 2D via the 3DGS rasterization pipeline and compared with GT edge maps using an L1 loss
    - Gradients are backpropagated to sketch parameters $S_i = \{l_i \text{ or } c_i, o_i, s_i\}$, enabling end-to-end optimization
    - Design motivation: 3D Gaussians serve as an intermediate representation that naturally supports differentiable rendering and is easily extensible to different curve types

2. **Adaptive Topology Control**: Four topology operations are interleaved during optimization to improve compactness and connectivity:

    - **End-point Merging**: two sketch endpoints are merged into one when their distance is < 10 mm (maintained via an optimizable point set $P$, with control points referenced by index)
    - **Overlapping Merging**: if one sketch is covered by another by more than 80% (proportion of sampled points in proximity), the smaller sketch is merged into the larger
    - **Co-linear Merging**: two line segments with similar directions (< 5°), small offset (< 10 mm), and small gap are merged into a longer segment
    - **Sketch Filtering**: after training, sketches invisible in more than 90% of views are removed, addressing erroneous reconstruction caused by occlusion
    - AABB-based fast filtering is used to skip sketch pairs that do not satisfy conditions, accelerating the process

3. **2DGS-SN Edge Detector**:

    - Existing neural edge detectors (PiDiNet, DexiNed) are inaccurate on CAD objects, exhibiting systematic offsets and texture-induced false detections
    - A geometry-cue-based detection method is proposed, combining foreground mask $A$, depth map gradient $g(D)$ (from 2DGS-reconstructed depth), and normal map gradient $g(N)$ (from a normal estimator):
    $E = G_f * (A | (g(D) > t_d) | (g(N) > t_n))$
    - Gradients are computed with the Sobel operator, thresholded and OR-merged, then smoothed with a Gaussian filter

### Loss & Training

- L1 loss between rendered edge maps and GT edge maps, with equal-ratio sampling of foreground and background
- Adam optimizer, 1000 epochs per scene; losses from all training views are accumulated before each parameter update
- Scenes are normalized to a 1 m³ bounding box; topology operation thresholds are set to 1% of scene size
- Initialization and sketch optimization each take approximately 5 minutes per scene, totaling ~10 minutes

## Key Experimental Results

### Main Results (ABC-NEF dataset, 82 CAD models)

| Method | Detector | A↓(mm) | C↓(mm) | R5↑ | P5↑ | F5↑ | R10↑ | F10↑ |
|------|--------|--------|--------|-----|-----|-----|------|------|
| LIMAP | LSD | 9.9 | 18.7 | 36.2 | 43.0 | 39.0 | 84.3 | 90.4 |
| NEF | PiDiNeT | 11.9 | 16.9 | 11.4 | 15.7 | 13.0 | 64.0 | 93.3 |
| EMAP | 2DGS-SN | 8.8 | 7.9 | 63.5 | 70.4 | 66.3 | 90.4 | 95.1 |
| EdgeGS | 2DGS-SN | 7.4 | 7.2 | 75.7 | 86.9 | 80.3 | 92.9 | 95.8 |
| **SketchSplat** | **2DGS-SN** | **6.8** | **5.8** | **90.8** | **92.9** | **91.3** | **95.4** | **96.5** |

This is the first method to achieve both A, C < 7 mm and R5/P5/F5 > 90%. With the 2DGS-SN detector, F5 improves by 11 percentage points over EdgeGS.

### Ablation Study

| Configuration | A↓ | C↓ | R5↑ | P5↑ | F5↑ | # Edges |
|------|-----|-----|-----|-----|-----|--------|
| Full SketchSplat | 6.8 | 5.8 | 90.8 | 92.9 | 91.3 | 44.28 |
| w/o 2DGS-SN (DexiNed) | 8.5 | 8.6 | 47.5 | 54.1 | 50.3 | 33.52 |
| w/o Merge (no topology merging) | 6.6 | 5.7 | 91.8 | 92.7 | 91.7 | **140.72** |
| w/o Filter | 6.8 | 5.8 | 90.9 | 92.9 | 91.3 | 44.37 |

Ablation on initialization methods:

| Initialization | A↓ | C↓ | F5↑ | # Edges |
|--------|-----|-----|-----|--------|
| EdgeGS → SketchSplat | 6.8 | 5.8 | 91.3 | 44.28 |
| EMAP → SketchSplat | 6.9 | 6.2 | 89.3 | 31.70 |

### Key Findings

- The 2DGS-SN detector yields significant improvements for all methods (EMAP, EdgeGS, SketchSplat), indicating that 2D edge quality is a critical bottleneck
- Topology merging reduces the number of edges from 140 to 44, greatly improving compactness with negligible loss in other metrics
- Starting from different initializations, SketchSplat consistently and substantially outperforms the initialization method, demonstrating the general effectiveness of differentiable optimization
- Compared to EdgeGS (140.8 edges), SketchSplat achieves higher accuracy with only 44.3 edges
- Training efficiency: ~10 minutes per scene, comparable to EdgeGS and far faster than NEF (1.5 h) and EMAP (2.5 h)

## Highlights & Insights

- **Architectural innovation**: 3D Gaussians serve as an intermediate representation bridging parametric edges and 2D images, cleverly exploiting the differentiable rendering capability of 3DGS
- **Precise problem formulation**: the paper identifies the fundamental flaw of the two-stage "reconstruct point set then fit" paradigm — the decoupling of fitting from image supervision
- **End-to-end optimization**: sketch parameters are directly constrained by 2D image loss, ensuring 3D–2D consistency of reconstructed edges
- **Elegant engineering**: topology operations are interleaved within optimization rather than applied as post-processing, ensuring results remain consistent with images

## Limitations & Future Work

- The 2DGS-SN detector depends on the quality of depth and normal maps, requiring a prior 2DGS scene reconstruction
- Only line segments and cubic Bézier curves are handled; more complex curve types are not explored
- CAD models containing cylinders and spheres are excluded (silhouette edges are inconsistent across views)
- Initialization still relies on existing methods (EdgeGS/EMAP); fully end-to-end initialization from scratch has not been achieved

## Related Work & Insights

- Conceptually similar to DiffVG (2D differentiable vector graphics optimization), but extended to 3D; Gaussian-based sampling avoids the need to customize differentiation for each curve type
- The geometric-cue-based approach of the 2DGS-SN detector (leveraging reconstructed depth/normals) is generalizable to other tasks requiring precise edges
- The sketch representation can be integrated into CAD reconstruction pipelines, with potential applications in CAD reverse engineering

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First method to directly optimize parametric 3D edges via differentiable sketch splatting
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Impressive results on ABC-NEF, with qualitative evaluation on DTU/Replica as well
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, detailed method description, and high-quality figures
- **Value**: ⭐⭐⭐⭐ Opens a new paradigm for 3D edge reconstruction with F5 improvements exceeding 10%

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues](neural_multi-view_self-calibrated_photometric_stereo_without_photometric_stereo_.md)
- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)
- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](../../NeurIPS2025/llm_evaluation/mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](../../NeurIPS2025/llm_evaluation/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)
- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](../../AAAI2026/llm_evaluation/deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)

<!-- RELATED:END -->
