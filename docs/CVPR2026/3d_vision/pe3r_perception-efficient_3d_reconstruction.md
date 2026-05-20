---
title: >-
  [Paper Note] PE3R: Perception-Efficient 3D Reconstruction
description: >-
  [CVPR 2026][3D Vision][3D semantic reconstruction] PE3R proposes a tuning-free, feed-forward 3D semantic reconstruction framework that directly generates semantic 3D point clouds from pose-free 2D images via three module…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D semantic reconstruction"
  - "open-vocabulary segmentation"
  - "tuning-free"
  - "feed-forward inference"
  - "semantic point cloud"
date: 2026-05-08
content_hash: 63b66c5fe9685d7a
---

# PE3R: Perception-Efficient 3D Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2503.07507](https://arxiv.org/abs/2503.07507)  
**Code**: [https://github.com/hujiecpp/PE3R](https://github.com/hujiecpp/PE3R)  
**Area**: 3D Vision
**Keywords**: 3D semantic reconstruction, open-vocabulary segmentation, tuning-free, feed-forward inference, semantic point cloud

## TL;DR
PE3R proposes a tuning-free, feed-forward 3D semantic reconstruction framework that directly generates semantic 3D point clouds from pose-free 2D images via three modules — pixel embedding disambiguation, semantic point cloud reconstruction, and global view perception — achieving a 9× speedup while establishing new state-of-the-art performance on open-vocabulary segmentation and depth estimation.

## Background & Motivation

**Background**: 2D-to-3D perception has made significant progress, with methods such as NeRF and 3DGS enabling multi-view 3D scene reconstruction and semantic extraction. The emergence of 2D foundation models including CLIP and SAM has further advanced open-vocabulary 3D segmentation.

**Limitations of Prior Work**: Existing methods suffer from three compounding challenges — poor scene generalization (requiring per-scene training), cross-view semantic inconsistency (mismatched semantic labels across viewpoints), and high computational cost (typically ranging from tens of minutes to several hours). For instance, LangSplat requires 149 minutes and Feature-3DGS requires 648 minutes.

**Key Challenge**: There is a fundamental tension between semantic consistency and inference efficiency — ensuring cross-view semantic coherence demands complex optimization, whereas efficient feed-forward methods struggle to guarantee semantic coherence. Moreover, most methods rely on additional inputs such as known camera parameters and depth maps.

**Goal**: (1) How to achieve efficient 3D semantic reconstruction without pose or depth supervision? (2) How to maintain semantic consistency across views and across object hierarchies? (3) How to support open-vocabulary natural language interaction?

**Key Insight**: The authors observe that SAM/SAM2 can provide hierarchical object mask decompositions, CLIP can encode semantics, and feed-forward geometry estimators such as DUSt3R can directly predict 3D point clouds from pose-free images. Integrating these three components into a cohesive pipeline simultaneously addresses both semantic consistency and efficiency.

**Core Idea**: Cross-view semantic ambiguity is resolved via area-weighted spherical interpolation, combined with feed-forward geometry prediction and global similarity normalization, enabling zero-shot generalization for 3D semantic reconstruction.

## Method

### Overall Architecture
PE3R operates in three stages: the input is a set of pose-free RGB images and the output is a semantically annotated 3D point cloud supporting natural language queries.
- **Stage 1: Pixel Embedding Disambiguation** — Each image is decomposed into hierarchical masks using SAM/SAM2; the masks are CLIP-encoded and aggregated via area-weighted spherical interpolation to produce cross-view consistent dense pixel embeddings.
- **Stage 2: Semantic Point Cloud Reconstruction** — DUSt3R directly predicts 3D point clouds from multi-view images; semantic-embedding-guided outlier detection and refinement are then applied for denoising.
- **Stage 3: Global View Perception** — User text queries are encoded and cosine similarity is computed against the semantic features of 3D points; global min-max normalization enables open-vocabulary localization.

### Key Designs

1. **Area-Weighted Spherical Interpolation**:

    - Function: Resolves semantic ambiguity across views and across hierarchical levels.
    - Mechanism: Given two unit embeddings $\mathbf{F}_A$ and $\mathbf{F}_B$, the area ratio is defined as $t = \frac{\text{area}_B}{\text{area}_A + \text{area}_B}$, and the aggregated embedding is computed as $\hat{\mathbf{F}}_B = a\mathbf{F}_A + b\mathbf{F}_B$, where $a = \frac{\sin((1-t)\theta)}{\sin\theta}$ and $b = \frac{\sin(t\theta)}{\sin\theta}$. This preserves unit norm after interpolation and guides the semantic direction toward the more reliable large-area features.
    - Design Motivation: Small masks (e.g., chair legs) yield unstable semantic features, whereas large masks (e.g., an entire chair) are more reliable. Spherical interpolation preserves the L2 norm, preventing drift from CLIP's embedding space. Two key properties — norm preservation and semantic guidance — ensure both geometric soundness and semantic informativeness during disambiguation.

2. **Two-Level Embedding Integration (Within-view + Cross-view Aggregation)**:

    - Function: Performs semantic aggregation both within individual views and across views.
    - Mechanism: Within a single view, masks are processed in descending order of area, with smaller masks aligned toward larger ones (part-to-whole consistency). Across views, a SAM2 tracker identifies correspondences for the same object across different viewpoints; masks with IoU < 0.1 are treated as new tracking targets and inserted accordingly.
    - Design Motivation: The two-step strategy addresses hierarchical ambiguity before viewpoint ambiguity. Cross-view fusion is skipped when tracking is unreliable, ensuring robustness.

3. **Semantic-Guided Outlier Detection and Refinement**:

    - Function: Removes spatial noise from the 3D point cloud predicted by DUSt3R.
    - Mechanism: For each pixel $P_{i,j}$, the average 3D Euclidean distance to pixels sharing the same semantic label within a $k \times k$ window is computed as $L_{i,j}$. Points exceeding a threshold are flagged as outliers. During refinement, rather than directly modifying 3D coordinates, the RGB values of outlier pixels are blended with the mean of surrounding semantic regions in image space: $\hat{y} = \alpha x + (1-\alpha)y$; the result is then re-fed into the point cloud predictor.
    - Design Motivation: Smoothing in image space is more efficient than geometric regularization in 3D space and exploits the input-output characteristics of the feed-forward model to indirectly correct 3D predictions.

### Loss & Training
PE3R is a tuning-free framework that requires no training. All modules (SAM/SAM2, CLIP, DUSt3R) are used directly with pretrained weights, and the entire pipeline completes within 5 minutes — compared to 43 minutes for the fastest prior method, LERF.

## Key Experimental Results

### Main Results

| Dataset | Metric | PE3R (Ours) | GOI (Prev. SOTA) | Gain |
|---------|--------|-------------|------------------|------|
| Mip-NeRF360 | mIoU | 0.8951 | 0.8646 | +3.5% |
| Mip-NeRF360 | mPA | 0.9617 | 0.9569 | +0.5% |
| Replica | mIoU | 0.6531 | 0.6169 | +5.9% |
| Replica | mP | 0.8444 | 0.8088 | +4.4% |
| ScanNet++ | mIoU | 0.2248 | 0.2101 (GOI emb) | +7.0% |

Runtime comparison (Mip-NeRF360 dataset):

| Method | Preprocessing | Training | Total |
|--------|---------------|----------|-------|
| Feature-3DGS | 25 min | 623 min | 648 min |
| LangSplat | 50 min | 99 min | 149 min |
| GOI | 8 min | 37 min | 45 min |
| **PE3R** | **5 min** | **—** | **5 min** |

### Ablation Study

| Configuration | mIoU (Mip360) | mIoU (Replica) | Note |
|---------------|--------------|----------------|------|
| Full PE3R | 0.8951 | 0.6531 | Complete model |
| w/o area-weighted interpolation | ~0.82 | ~0.59 | Semantic consistency degrades |
| w/o cross-view aggregation | ~0.85 | ~0.61 | Within-view disambiguation alone is insufficient |
| w/o semantic refinement | ~0.87 | ~0.63 | Point cloud noise degrades segmentation accuracy |

Multi-view depth estimation (average over 5 datasets):

| Method | Abs Rel↓ | delta<1.25↑ |
|--------|----------|-------------|
| COLMAP | 9.3 | 67.8 |
| DUSt3R | 4.7 | 64.5 |
| MASt3R | 3.3 | 74.9 |
| **PE3R** | **2.5** | **79.1** |

### Key Findings
- Area-weighted spherical interpolation is the most critical component, simultaneously resolving both hierarchical and viewpoint ambiguity.
- On the large-scale ScanNet++ dataset, PE3R's embedding quality substantially outperforms all baselines, indicating that the disambiguation strategy is particularly advantageous in complex scenes.
- The semantic refinement module indirectly improves 3D geometric quality through image-space smoothing, yielding notable gains on depth estimation as well.

## Highlights & Insights
- **Tuning-free + 9× speedup**: The entire pipeline runs in 5 minutes by relying exclusively on pretrained models. This compositional innovation (SAM + CLIP + DUSt3R) demonstrates that effective orchestration of existing components can be more practically impactful than designing novel individual modules.
- **Mathematical elegance of spherical interpolation**: Area-weighted spherical interpolation simultaneously satisfies norm preservation and semantic guidance, constituting a genuinely elegant design whose underlying idea is transferable to any setting requiring feature fusion on the hypersphere.
- **Image-space refinement instead of 3D regularization**: Correcting 3D outputs by modifying input images cleverly exploits the input-output characteristics of feed-forward predictors, avoiding costly 3D post-processing.

## Limitations & Future Work
- The SAM2 tracker may fail under fast motion or large-baseline scenarios, limiting the robustness of cross-view aggregation.
- The current representation is restricted to point clouds; surface mesh or implicit surface support is absent, making the method unsuitable for applications requiring mesh outputs.
- The blending factor $\alpha$ in semantic refinement is set manually; an adaptive strategy could be explored.
- The mIoU on ScanNet++ remains at only 22.48%, indicating substantial room for improvement on large-scale, complex indoor scenes.

## Related Work & Insights
- **vs. LangSplat**: LangSplat aligns CLIP embeddings with 3DGS but requires per-scene training (99 min); PE3R achieves zero-shot generalization via feed-forward inference and is 9× faster.
- **vs. GOI**: GOI enforces multi-view consistency through text-image alignment but still requires 37 minutes of training; PE3R surpasses it in both accuracy and speed.
- **vs. LSM (Large Spatial Model)**: LSM also adopts a feed-forward approach for open-vocabulary 3D segmentation, but PE3R's disambiguation strategy yields superior performance across all benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐ — The core contribution lies in the disambiguation strategy and pipeline design; individual modules are not novel, but their combination achieves outstanding results.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 7 datasets and 2 tasks with comprehensive baselines.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with rigorous mathematical derivations and informative figures.
- Value: ⭐⭐⭐⭐⭐ — Tuning-free, near-real-time 3D semantic reconstruction has substantial practical value; code is publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception](long_scope_fully_sparse_long_range_cooperative_3d_perception.md)
- [\[CVPR 2026\] SoPE: Spherical Coordinate-Based Positional Embedding for Enhancing Spatial Perception of 3D LVLMs](sope_spherical_coordinate-based_positional_embedding_for_enhancing_spatial_perce.md)
- [\[CVPR 2026\] SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)
- [\[ICLR 2026\] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](../../ICLR2026/3d_vision/urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [\[CVPR 2026\] APC: Transferable and Efficient Adversarial Point Counterattack for Robust 3D Point Cloud Recognition](apc_adversarial_point_counterattack.md)

</div>

<!-- RELATED:END -->
