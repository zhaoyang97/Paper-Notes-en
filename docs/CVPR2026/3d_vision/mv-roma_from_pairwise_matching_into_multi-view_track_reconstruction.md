---
title: >-
  [Paper Note] MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction
description: >-
  [CVPR 2026][3D Vision][Multi-view matching] This paper proposes MV-RoMa, the first multi-view dense matching model that simultaneously estimates dense correspondences from a single source image to multiple target images via a Track-Guided multi-view encoder and a pixel-aligned multi-view refiner, producing geometrically consistent tracks for SfM and achieving state-of-the-art performance on HPatches, ETH3D, IMC, and related benchmarks.
tags:
  - CVPR 2026
  - 3D Vision
  - Multi-view matching
  - dense correspondence
  - track reconstruction
  - SfM
  - feature fusion
date: 2026-05-08
content_hash: 9ef27ef5a3b957db
---

# MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.27542](https://arxiv.org/abs/2603.27542)
**Code**: [Project Page](https://icetea-cv.github.io/mv-roma/)
**Area**: 3D Vision / Feature Matching
**Keywords**: Multi-view matching, dense correspondence, track reconstruction, SfM, feature fusion

## TL;DR

This paper proposes MV-RoMa, the first multi-view dense matching model that simultaneously estimates dense correspondences from a single source image to multiple target images via a Track-Guided multi-view encoder and a pixel-aligned multi-view refiner, producing geometrically consistent tracks for SfM and achieving state-of-the-art performance on HPatches, ETH3D, IMC, and related benchmarks.

## Background & Motivation

1. **Background**: Feature matching is a foundational task for 3D reconstruction and visual localization. Dense matching methods such as RoMa and DKM already produce high-quality pairwise correspondences.
2. **Limitations of Prior Work**: Existing methods are inherently pairwise; in multi-view tasks such as SfM, pairwise results must be chained into multi-view tracks, a process prone to fragmentation and geometric inconsistency.
3. **Key Challenge**: Post-processing optimization methods (e.g., PixSfM, DFSfM) can only refine on top of initial pairwise matches, require per-track optimization, and are bottlenecked by the quality of those initial matches.
4. **Goal**: Achieve multi-view consistent dense correspondences directly at the model level, eliminating accumulated chaining errors.
5. **Key Insight**: Sparse geometric priors derived from initial pairwise matches are embedded as "track tokens" into the feature encoder to guide multi-view feature interaction.
6. **Core Idea**: A track-token-guided multi-view encoder combined with a pixel-aligned attentional refiner enables the first end-to-end multi-view dense matching framework.

## Method

### Overall Architecture

**Input**: one source image $I_0$ and multiple co-visible target images $\{I_v\}$. An off-the-shelf matcher (UFM by default) first produces initial pairwise matches, from which sparse multi-view track tokens are constructed via visibility-grouped k-means clustering. The pipeline then proceeds in three stages: (1) a Track-Guided multi-view encoder injects multi-view context into a DINOv2 backbone to produce geometrically consistent dense features; (2) a global matcher generates coarse correspondences; (3) a multi-view refiner applies pixel-aligned attention with progressive upsampling to yield full-resolution dense correspondences. **Output**: dense warp fields $W^{0\to v}$ and confidence maps from the source image to each target image.

### Key Designs

1. **Track Token Construction and Clustering Sampling**

   - **Function**: Extract compact multi-view geometric priors from pairwise matching results.
   - **Mechanism**: Each track token contains a 2D coordinate vector $\mathbf{u}_i \in \mathbb{R}^{2V}$ (positions across all views) and a visibility mask $\mathbf{m}_i \in \{0,1\}^V$. Tracks are first grouped by visibility pattern; within each group, k-means clustering selects representative tracks (the track closest to each centroid), yielding $T=512$ tokens in total. This avoids the spatial redundancy and noise sensitivity of random sampling.
   - **Design Motivation**: More compact and spatially uniform than random sampling; visibility-based grouping ensures proper handling of partially visible tracks.

2. **Track-Guided Multi-View Encoder**

   - **Function**: Inject multi-view geometric context into the DINOv2 feature extraction process.
   - **Mechanism**: After each Transformer block in the latter half of DINOv2, three operations are inserted: (i) **Attentional Sampling** — cross-attention using track coordinates as queries and image features as keys/values, with a spatial distance bias $B_v$, sampling image information into track tokens; (ii) **Track Transformer** — self-attention along the view axis for each track (without view-index embeddings to ensure view-agnostic processing), with visibility masks applied to occluded views; (iii) **Attentional Splatting** — the reverse operation, writing updated multi-view-aware track features back to the image grid.
   - **Design Motivation**: Avoids the $O(V \cdot H \cdot W)^2$ complexity of global multi-view cross-attention by using sparse tracks as information conduits, reducing complexity to $O(T \cdot HW \cdot V)$.

3. **Multi-View Matching Refiner**

   - **Function**: Inject multi-view context at fine resolution to produce final full-resolution dense correspondences.
   - **Mechanism**: Built on RoMa's coarse-to-fine framework, multi-view attention is inserted at stride 4 and stride 1 levels. Target-view features are first warped to the source-view coordinate system using the previous-level warp, followed by **pixel-aligned attention** — each pixel attends only to its corresponding cross-view locations. Cross-view attention and ConvNeXt spatial propagation are applied alternately for $N$ iterations, outputting warp residuals.
   - **Design Motivation**: Global cross-attention is prohibitively expensive; the pixel-aligned design reduces complexity to $O(HW \cdot V)$ and performs fine-grained adjustments only where needed.

### Loss & Training

- RoMa's robust loss is used; training is conducted on MegaDepth + ScanNet for 200K steps.
- Learning rate $3\times10^{-5}$, decayed by $10\times$ after 20K steps, batch size 4.
- Default configuration: 1 source + 4 target images, 512 track tokens.

## Key Experimental Results

### Main Results

**HPatches Homography Estimation (AUC %)**:

| Method | DLT @1/3/5px | RANSAC @1/3/5px |
|--------|-------------|-----------------|
| RoMa | 41.0/67.9/76.9 | 44.7/72.6/81.4 |
| **MV-RoMa** | **46.1/71.9/80.1** | **47.2/73.2/81.8** |

**ETH3D 3D Triangulation**:

| Method | Accuracy 1/2/5cm | Completeness 1/2/5cm |
|--------|-------------------|----------------------|
| RoMa | 75.58/86.25/94.95 | 5.64/15.73/38.60 |
| **MV-RoMa** | **85.88/92.99/98.05** | 3.95/9.94/23.81 |
| RoMa + Dense-SfM | 84.79/92.62/97.77 | **7.38/17.06/36.35** |

**Multi-View Pose Estimation (AUC %)**:

| Method | Texture-Poor @3/5/10° | IMC @3/5/10° |
|--------|-----------------------|--------------|
| RoMa + Dense-SfM | 49.94/66.23/81.41 | 48.48/60.79/73.90 |
| **MV-RoMa** | **51.79/66.77/81.74** | **51.31/62.92/75.92** |

### Ablation Study

| Configuration | ETH3D Acc@2cm | ETH3D Comp@2cm | HPatches AUC@3px |
|---------------|---------------|----------------|------------------|
| RoMa baseline | 86.25 | 15.73 | 67.9 |
| + MV-Encoder | Improved | Improved | Improved |
| + MV-Encoder + MV-Refiner | **92.99** | Best | **71.9** |

### Key Findings

- The largest gain appears at the strict HPatches @1px threshold (41.0→46.1), indicating that multi-view consistency primarily improves fine-grained matching precision.
- The small gap between DLT and RANSAC results suggests extremely high inlier ratios, reflecting strong intrinsic match quality.
- MV-RoMa shows pronounced advantages on the Texture-Poor dataset, confirming that multi-view feature interaction is especially effective in low-texture scenarios.
- The MV-Encoder and MV-Refiner provide complementary contributions; removing either degrades performance.

## Highlights & Insights

- **Using sparse track tokens as multi-view information conduits** is the central innovation. It avoids the quadratic complexity of global multi-view attention while retaining sufficient geometric priors. This "sparse proxy" strategy is broadly applicable to other multi-view tasks.
- **Pixel-aligned attention** is an elegant design — warping features before attention transforms a spatial search problem into a local alignment problem, substantially reducing computation.
- **Paradigm shift from post-hoc refinement to end-to-end prediction**: the method no longer inherits the quality ceiling of pairwise matching.

## Limitations & Future Work

- The method still relies on an off-the-shelf matcher (UFM) to construct initial track tokens; unreliable initial matches yield untrustworthy priors.
- The default setting handles only $1+4=5$ images; scaling to more views requires reconsideration of track token count and attention complexity.
- Completeness on ETH3D falls short of some post-processing methods (Dense-SfM), partly due to the conservative NMS sampling strategy.
- Future work could explore eliminating the dependency on an initial matcher and learning multi-view priors directly within the network.

## Related Work & Insights

- **vs. RoMa**: The direct predecessor of MV-RoMa. RoMa achieves state-of-the-art pairwise dense matching; MV-RoMa augments it with multi-view consistency, yielding further improvements across all metrics.
- **vs. PixSfM / DFSfM**: These are post-processing optimization methods that refine existing tracks. MV-RoMa directly produces multi-view consistent matches without per-track optimization.
- **vs. Tracktention**: The sample–transform–splat design is inspired by Tracktention, which targets video tracking; MV-RoMa adapts this paradigm to sparse multi-view matching.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First multi-view dense matching model with an elegant track token design, though the overall framework still builds upon RoMa.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated on four benchmarks (HPatches, ETH3D, IMC, Texture-Poor) with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear figures, fluent pipeline description, and consistent notation.
- **Value**: ⭐⭐⭐⭐ — Directly advances SfM and 3D reconstruction research by filling the gap in multi-view dense matching.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](coherent_human-scene_reconstruction_from_multi-person_multi-view_video_in_a_sing.md)
- [\[CVPR 2026\] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [\[CVPR 2026\] Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)
- [\[CVPR 2026\] Learning Multi-View Spatial Reasoning from Cross-View Relations](learning_multi-view_spatial_reasoning_from_cross-view_relations.md)
- [\[ICCV 2025\] MV-Adapter: Multi-view Consistent Image Generation Made Easy](../../ICCV2025/3d_vision/mv-adapter_multi-view_consistent_image_generation_made_easy.md)

</div>

<!-- RELATED:END -->
