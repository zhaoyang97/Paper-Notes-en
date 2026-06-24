---
title: >-
  [Paper Note] LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting
description: >-
  [ICML 2025][3D Vision][3D Gaussian Splatting] Proposed LaGa, which establishes cross-view semantic connections via 3D scene decomposition and constructs view-aggregated semantic representations using adaptive clustering with dual-factor re-weighting. This addresses the overlooked view-dependent semantics issue in 3D Language Gaussian Splatting, achieving a 3D mIoU of 64.0% (+18.7%) on LERF-OVS.
tags:
  - "ICML 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "open-vocabulary segmentation"
  - "view-dependent semantics"
  - "scene decomposition"
  - "CLIP"
date: 2026-05-08
content_hash: 42cb24d6b70708e1
---

# LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting

**Conference**: ICML 2025  
**arXiv**: [2505.24746](https://arxiv.org/abs/2505.24746)  
**Code**: [GitHub](https://github.com/SJTU-DeepVisionLab/LaGa)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, open-vocabulary segmentation, view-dependent semantics, scene decomposition, CLIP

## TL;DR

Proposed LaGa, which establishes cross-view semantic connections via 3D scene decomposition and constructs view-aggregated semantic representations using adaptive clustering with dual-factor re-weighting. This addresses the overlooked view-dependent semantics issue in 3D Language Gaussian Splatting, achieving a 3D mIoU of 64.0% (+18.7%) on LERF-OVS.

## Background & Motivation

**Background**: The mainstream approach for open-vocabulary scene understanding in 3D-GS is to project 2D semantic features from VLMs (like CLIP) onto 3D Gaussians via differentiable rasterization. During training, the 3D features are optimized to align the rendered feature maps with 2D semantics; during inference, the rendered feature maps are used for pixel-wise understanding.

**Limitations of Prior Work**: These methods perform well on 2D rendered feature maps, but their performance drops sharply when directly using 3D features for 3D perception (e.g., retrieving 3D Gaussians). This degradation occurs because they ignore a fundamental problem: the same 3D object exhibits different semantics when observed from different viewpoints—referred to by the authors as "view-dependent semantics." For instance, a passport's title is recognizable from the front, but completely unidentifiable from the back or sides.

**Key Challenge**: Simply projecting 2D semantics onto 3D Gaussians causes each Gaussian to only inherit viewpoint-specific semantics, leading to false positives (noisy Gaussians falsely retrieved) and false negatives (target Gaussians missed), causing a fundamental gap between 2D and 3D understanding.

**Goal**: How to construct a 3D representation within the 3D-GS framework that retains multi-view semantic information, enabling direct 3D scene understanding without relying on 2D rendering.

**Key Insight**: The authors first conduct two quantitative analyses: (1) semantic similarity distribution analysis, showing that the intra-object similarity of semantic features across different views is often even lower than the inter-object similarity between different objects; (2) semantic retrieval completeness analysis, showing that about 50% of 2D semantic features fail to retrieve the corresponding 3D objects completely. These analyses provide strong evidence for the existence of view-dependent semantics.

**Core Idea**: Establish cross-view semantic connections by aggregating multi-view 2D masks into 3D objects via 3D scene decomposition, and then extract representative semantic descriptors through adaptive clustering and weighted aggregation to preserve critical view-dependent information.

## Method

### Overall Architecture

The pipeline of LaGa consists of three steps: (1) Data preparation: extract 2D masks using SAM and corresponding semantic features using CLIP; (2) 3D scene decomposition: train Gaussian affinity features via contrastive learning, and cluster the multi-view 2D masks into coherent 3D objects using HDBSCAN to establish cross-view semantic connections; (3) View-aggregated semantic representation: perform K-means clustering on the multi-view semantics of each 3D object to extract representative descriptors, and apply a dual-factor weighting strategy to suppress noise. During inference, objects are retrieved directly in 3D space based on text queries, bypassing the need for 2D feature map rendering.

### Key Designs

1. **Contrastive Learning-Driven 3D Scene Decomposition**:

    - **Function**: Aggregate multi-view 2D masks into class-agnostic but structurally coherent 3D objects, establishing cross-view connections.
    - **Mechanism**: Train an affinity feature $\mathbf{f}_\mathbf{g} \in \mathbb{R}^{C'}$ for each 3D Gaussian, and render them to 2D to obtain each mask prototype $\hat{\mathbf{f}}_\mathbf{M}$ via masked average pooling. The training objective pulls features within the same mask together while pushing features outside the mask apart (contrastive loss). After training, HDBSCAN is used to cluster all mask prototypes, grouping multi-view masks of the same 3D object into a set $\mathcal{S}_i$, and then Gaussians are assigned to objects based on the similarity between prototypes and Gaussian affinity features.
    - **Design Motivation**: 2D segmentation primarily captures object boundaries, which are stable across views and unaffected by high-level semantic variations. This allows scene decomposition to bypass interference from view-dependent semantics, establishing reliable cross-view connections—utilizing the stability of low-level segmentation versus the instability of high-level semantics.

2. **Adaptive Cross-View Descriptor Extraction**:

    - **Function**: Extract a set of representative descriptors for each 3D object that effectively summarizes multi-view semantic variations.
    - **Mechanism**: Perform K-means clustering on the multi-view semantic feature set $\mathcal{V}^{\mathcal{S}_i}$ of each object $\mathcal{G}^{\mathcal{S}_i}$, where the cluster centers serve as semantic descriptors. The number of descriptors $N^{\mathcal{G}^{\mathcal{S}_i}}$ is adaptively determined using the silhouette score—assigning more descriptors to objects with higher semantic complexity.
    - **Design Motivation**: The semantic complexity of different objects varies significantly (e.g., simple objects like walls vs. complex objects like a passport with text), making a fixed number of descriptors unsuitable. Adaptive clustering ensures that each object retains its multi-view semantic variations in the most compact way.

3. **Dual-Factor Weighted Descriptor Aggregation**:

    - **Function**: Compute the matching score of an object to a text query during inference while suppressing noisy descriptors.
    - **Mechanism**: Given a text query $\mathbf{q}$, the object-level score is computed as $\text{REL}(\mathcal{G}^{\mathcal{S}_i}, \mathbf{q}) = \max_\mathbf{d} \omega^\mathbf{d} \cdot \text{Rel}(\mathbf{d}, \mathbf{q})$. The weight $\omega^\mathbf{d}$ is determined by the product of two factors: (i) direction consistency—the cosine similarity between the descriptor and the global average feature, suppressing anomalous descriptors that deviate from mainstream semantics (e.g., a book spine looking like a "knife"); (ii) internal compactness—the L2 norm of the cluster center, where semantically consistent cluster centers have larger norms (since aligned member directions do not cancel out during averaging) and inconsistent ones have smaller norms.
    - **Design Motivation**: Not all viewpoint semantics are equally reliable. Direction consistency ensures mainstream semantics are prioritized, while internal compactness ensures only highly confident descriptors receive high weights, complementary to suppressing noise.

### Loss & Training

The scene decomposition phase employs a contrastive loss: $\mathcal{L} = \sum_\mathbf{I} \sum_\mathbf{M} \sum_\mathbf{p} (1-2\mathbf{M}(\mathbf{p})) \max(\langle \hat{\mathbf{f}}_\mathbf{M}, \mathbf{F}_\mathbf{I}(\mathbf{p}) \rangle, 0)$. The semantic representation construction phase does not require additional training and is purely post-processing.

## Key Experimental Results

### Main Results: LERF-OVS Dataset (3D mIoU %)

| Method | Type | Figurines | Teatime | Ramen | Waldo Kitchen | Mean |
|------|------|-----------|---------|-------|---------------|------|
| SAGA | 3D | 36.2 | 19.3 | 53.1 | 14.4 | 30.7 |
| LangSplat | 3D | 25.9 | 35.6 | 29.3 | 33.5 | 31.1 |
| OpenGaussian | 3D | 61.1 | 59.1 | 29.2 | 31.9 | 45.3 |
| SuperGSeg* | 3D | 43.7 | 55.3 | 18.1 | 26.7 | 35.9 |
| **LaGa** | **3D** | **64.1** | **70.9** | **55.6** | **65.6** | **64.0** |
| N2F2 | 2D | 47.0 | 69.2 | 56.6 | 47.9 | 54.4 |
| OccamLGS* | 2D | 58.6 | 70.2 | 51.0 | 65.3 | 61.3 |

### Cross-Dataset Validation

| Dataset | Metric | OpenGaussian | LaGa |
|--------|------|-------------|------|
| 3D-OVS | mIoU | — | 95.3 |
| ScanNet (19 Classes) | mIoU / mAcc | 24.7 / 41.5 | **32.5 / 49.1** |
| ScanNet (15 Classes) | mIoU / mAcc | 30.1 / 48.3 | **35.5 / 53.5** |
| ScanNet (10 Classes) | mIoU / mAcc | 38.3 / 55.2 | **42.6 / 63.2** |

### Key Findings

- LaGa outperforms all 3D methods in 3D understanding by +18.7% mIoU, and even surpasses the best 2D method (OccamLGS 61.3 $\rightarrow$ LaGa 64.0), marking the first time a 3D method beats 2D methods on this benchmark.
- The Waldo Kitchen scene shows the largest improvement (31.9 $\rightarrow$ 65.6) due to the abundance of objects and massive viewpoint variations, where OpenGaussian's rigid feature assignment fails severely.
- The improvement on 3D-OVS is limited (95.3%) because forward-facing scenes and fewer objects make the view-dependent semantics problem less prominent.
- It also outperforms OpenGaussian on ScanNet, validating its generalizable capacity to large-scale scenes.

## Highlights & Insights

- The discovery and quantitative validation of the "view-dependent semantics" problem is a core contribution—providing strong evidence through two analytical experiments, which makes the paper highly convincing at the problem formulation level.
- The method itself is a post-processing workflow (scene decomposition + clustering + weighting) that does not modify the 3D-GS training or inference pipelines, making it both simple and practical.
- The insight that contrastive learning-based scene decomposition is unaffected by view-dependent semantics—recognizing that low-level segmentation is stable across views while high-level semantics is not—presents a hierarchical thinking that can be transferred to other 3D understanding tasks.

## Limitations & Future Work

- Scene decomposition relies on the quality of 2D masks from SAM; extreme occlusion or transparent objects may cause failure.
- The clustering step is an offline post-processing procedure, which does not support online updates for new views or new objects.
- Different objects with highly similar appearances (e.g., a row of identical cups) might be incorrectly grouped together.
- Performance on large-scale outdoor scenes and dynamic scenes has not been verified.

## Related Work & Insights

- **vs OpenGaussian**: The key difference is that OpenGaussian uses rules to select representative views to assign a single CLIP feature to each Gaussian, ignoring multi-view information. LaGa utilizes clustering + weighting to preserve multi-view semantics.
- **vs LangSplat/N2F2**: These methods rely on rendering 2D feature maps, which yield good 2D results but poor 3D results. LaGa operates directly in 3D space, outperforming 2D methods in 3D understanding for the first time.
- **Insights**: The view-dependent semantics problem also exists in other 3D representations like NeRFs and point clouds; the clustering + re-weighting strategy is generic and broadly applicable.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The discovery and analysis of the view-dependent semantics problem is an original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation across three datasets plus quantitative analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain of problem motivation $\rightarrow$ methodology $\rightarrow$ experiments.
- Value: ⭐⭐⭐⭐⭐ A +18.7% mIoU improvement proves that addressing view-dependent semantics is a critical bottleneck in 3D understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](../../CVPR2025/3d_vision/envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Online Language Splatting](../../ICCV2025/3d_vision/online_language_splatting.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](../../CVPR2026/3d_vision/extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](../../CVPR2025/3d_vision/s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
