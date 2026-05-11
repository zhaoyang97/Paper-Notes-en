---
title: >-
  [Paper Note] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation
description: >-
  [3D Vision] CutS3D is the first method to introduce 3D information (monocular depth estimation) into unsupervised instance segmentation. By cutting semantic regions in 3D point clouds…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 69bb1969815f3237
---

# CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation

## Basic Information
- **Conference**: ICCV 2025
- **arXiv**: 2411.16319
- **Code**: [leonsick.github.io/cuts3d](https://leonsick.github.io/cuts3d)
- **Area**: 3D Vision / Unsupervised Instance Segmentation
- **Keywords**: Unsupervised Instance Segmentation, 3D Semantic Segmentation, Normalized Cut, Depth Estimation, Pseudo Labels

## TL;DR

CutS3D is the first method to introduce 3D information (monocular depth estimation) into unsupervised instance segmentation. By cutting semantic regions in 3D point clouds, it separates overlapping instances in 2D, and introduces a spatial confidence mechanism to improve pseudo-label quality, surpassing CutLER and other SoTA methods on multiple benchmarks.

## Background & Motivation

- **Problem Definition**: Unsupervised instance segmentation aims to segment each object instance in an image without relying on manual annotations.
- **Limitations of Prior Work**:
    - CutLER (SoTA): Constructs a semantic affinity graph using DINO self-supervised features and extracts pseudo labels via MaskCut. However, it only considers 2D semantic relationships and cannot separate co-category instances that overlap or connect in 2D image space (e.g., two tennis players in front of and behind each other).
    - FreeSOLO, CuVLER, and similar methods are also constrained by 2D semantic information.
    - Humans naturally perceive the world in 3D, using spatial boundaries to distinguish instances.
- **Key Insight**: Modern zero-shot monocular depth estimators can obtain accurate 3D information without manual annotation, so incorporating 3D does not violate the unsupervised setting. Cutting semantic masks in 3D can correctly separate instances that overlap in 2D.

## Method

### Overall Architecture

CutS3D augments the CutLER pipeline with three core components: (1) **LocalCut**: instance cutting in 3D; (2) **Spatial Importance Sharpening**: enhancing the semantic affinity graph with depth information; and (3) **Spatial Confidence**: evaluating pseudo-label quality to improve detector training.

### LocalCut: 3D Instance Cutting

1. Obtain a depth map $D$ using ZoeDepth zero-shot monocular depth estimation.
2. Back-project orthogonally to a point cloud $P = \{p_1, ..., p_m\}$.
3. Using the initial semantic bisection $B$ from NCut as a basis, set points outside the semantic region to background depth.
4. Construct a k-NN graph $G^{3D}$ on the point cloud with edge weights defined by Euclidean distance.
5. After truncating the graph with threshold $\tau_\text{knn}$, apply **MinCut** (Dinic's algorithm) to cut instances in 3D space.
6. Source node $s$ and sink node $t$ are selected to bridge semantic and 3D spatial information:
    - $s = p_{\lambda_\max}$ (point corresponding to the maximum NCut eigenvalue, semantic foreground)
    - $t = p_{\lambda_\min}$ (point corresponding to the minimum NCut eigenvalue, semantic background)
7. Mask boundaries are refined with CRF.

### Spatial Importance Sharpening

**Objective**: Make the semantic affinity graph aware of 3D boundaries, so that initial semantic masks more completely cover object instances.

A spatial importance map is obtained by applying Gaussian blur to the depth map $D$ and computing the difference (highlighting high-frequency depth change regions):

$$\Delta D = |G_\sigma * D - D|$$

This is normalized to $[\beta, 1.0]$ ($\beta=0.45$), and then used to sharpen the semantic affinity matrix via element-wise exponentiation:

$$\mathbf{W}_{i,j} = W_{i,j}^{1 - \Delta D_{n_{i,j}}}$$

Semantic similarity is suppressed at regions of large depth variation (object boundaries), encouraging NCut to cut along 3D boundaries.

### Spatial Confidence

**Design Motivation**: Pseudo labels contain ambiguity; evaluating their quality provides a cleaner learning signal.

**Computation**: $T$ values are uniformly sampled between $\tau_\text{knn}^{min}$ and $\tau_\text{knn}$, LocalCut is executed for each, and the results are averaged to obtain a confidence map:

$$\text{SC}_{i,j} = \frac{1}{T}\sum_{t=1}^{T} \text{BC}_{i,j}(t)$$

Intuition: Objects with clear 3D boundaries yield consistent cuts across different thresholds (high confidence), while those with ambiguous boundaries produce unstable results (low confidence).

**Three Modes of Use**:
1. **Confidence Copy-Paste Selection**: Only high-confidence masks are selected for copy-paste data augmentation.
2. **Confidence Alpha-Blending**: Alpha blending is performed proportional to confidence (low-confidence regions are semi-transparent).
3. **Spatial Confidence Soft Target Loss**: Mask loss is reweighted per patch:

$$L_\text{mask} = \sum_{(i,j)} \text{SC}_{i,j} \cdot \text{BCE}(\hat{M}_{i,j}, M_{i,j})$$

## Key Experimental Results

### Main Results: Zero-Shot Unsupervised Instance Segmentation

| Method | COCO val2017 AP^mask | COCO val2017 AP^mask_50 | COCO20K AP^mask | COCO20K AP^mask_50 |
|------|---------------------|------------------------|-----------------|-------------------|
| FreeSOLO | 4.3 | 9.4 | 4.3 | 9.7 |
| CutLER | 9.7 | 18.9 | 10.0 | 19.6 |
| CuVLER | 9.8 | 19.3 | 10.0 | 20.0 |
| ProMerge+ | 8.9 | - | 9.0 | - |
| **CutS3D** | **10.7** | **20.8** | **10.9** | **21.3** |

CutS3D outperforms the best competing method on COCO val2017 by +0.9 AP^mask and +1.5 AP^mask_50.

### Ablation Study

| Method | AP^box_50 | AP^box | AP^mask_50 | AP^mask |
|------|----------|--------|-----------|---------|
| CutLER (DiffNCuts) | 22.1 | 12.3 | 18.7 | 9.4 |
| + LocalCut | 22.9 | 12.5 | 18.9 | 9.5 |
| + Spatial Importance | 23.3 | 12.6 | 19.2 | 9.8 |
| + Spatial Confidence | 23.9 | 13.0 | 20.1 | 10.2 |
| + 3 rounds self-training | 24.3 | 13.3 | 20.8 | 10.7 |

Each component contributes independently; the synergy between LocalCut and Spatial Importance Sharpening is the most pronounced.

### Spatial Confidence Component Analysis

| Confidence Copy-Paste | Alpha Blend | SC Loss | AP^mask |
|-----------------|-------------|---------|---------|
| ✗ | ✗ | ✗ | 8.5 |
| ✓ | ✗ | ✗ | 8.8 |
| ✓ | ✓ | ✗ | 9.0 |
| ✓ | ✓ | ✓ | 9.1 |

### Zero-Shot Object Detection (Average over 6 Datasets)

| Method | Average AP^box_50 | Average AP^box |
|------|------------------|----------------|
| CuVLER | 21.3 | 11.3 |
| CutLER | 21.6 | 11.6 |
| **CutS3D** | **23.9** | **12.5** |

CutS3D with a single feature extractor outperforms CuVLER, which uses an ensemble of 6 DINO models, demonstrating that 3D information is more effective than additional feature extractors.

### Depth Source Comparison

| Depth Estimator | AP^mask_50 | AP^mask |
|-----------|-----------|---------|
| ZoeDepth | 18.0 | 9.1 |
| Kick Back & Relax | 17.8 | 9.1 |
| Marigold | 17.7 | 9.0 |
| MiDaS (Small) | 17.6 | 8.9 |

Performance is comparable across different depth estimators, indicating that the method is robust to the choice of depth source.

## Highlights & Insights

1. **First use of 3D information in unsupervised instance segmentation**: Leveraging zero-shot depth estimation does not violate the unsupervised setting, yet significantly improves instance separation.
2. **Semantics-space dual-driven MinCut**: NCut defines semantic foreground/background, while MinCut performs the actual cut in 3D space — an elegant combination of the two.
3. **Complementary effect of Spatial Importance Sharpening**: Improving the initial semantic mask enables LocalCut to more accurately locate 3D boundaries.
4. **Patch-level quality assessment via Spatial Confidence**: More fine-grained than the scalar reweighting of CuVLER, reflecting pseudo-label reliability at the patch level.
5. **Single model vs. ensemble**: CutS3D with only one feature extractor and a depth estimator surpasses CuVLER's 6-model ensemble.

## Limitations & Future Work

- The method depends on the quality of monocular depth estimation; although experiments show robustness to depth source choice, extreme scenarios may still be limiting.
- Validation is limited to natural image datasets; performance in specialized domains such as medical imaging or remote sensing remains untested.
- k-NN graph construction for LocalCut may introduce computational overhead on large-scale images.
- Orthographic projection approximation may introduce errors in scenes with large depth ranges.
- Spatial Confidence requires multiple runs of LocalCut, increasing pseudo-label generation time.

## Related Work & Insights

- **CutLER**: The direct predecessor of the proposed pipeline; this work augments its MaskCut and self-training strategy.
- **CuVLER**: Uses a 6-model ensemble and scalar soft target loss; this work achieves greater efficiency through 3D information and patch-level confidence.
- **DiffNCuts**: A fine-tuned DINO feature extractor used as the backbone in this work.
- **Insight**: Introducing 3D information (e.g., zero-shot depth) into other tasks that rely on 2D semantics represents a generalizable enhancement strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First application of 3D information to unsupervised segmentation; spatial confidence design is elegant)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Comprehensive validation on 6 datasets, detailed ablations, and informative depth source comparison)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear figures, rigorous mathematical derivation, and persuasive qualitative comparisons)
- **Value**: ⭐⭐⭐⭐ (Introduces a new dimension to unsupervised segmentation with a concise and generalizable approach)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
