---
title: >-
  [Paper Note] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes
description: >-
  [3D Vision] InstaScene proposes a unified framework for instance decomposition and complete reconstruction from cluttered scenes. It constructs a spatial contrastive learning scheme via tracked Gaussian rasterization for accurate instance segmentation, and designs an in-situ generation pipeline that leverages available observations and geometric cues to guide a 3D generative model toward complete object reconstruction.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: 64999fc8f587fb4b
---

# InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2507.08416](https://arxiv.org/abs/2507.08416)
- **Code**: [Project Page](https://zju3dv.github.io/instascene/)
- **Area**: 3D Vision
- **Keywords**: 3D instance segmentation, scene decomposition, complete reconstruction, Gaussian Splatting, contrastive learning, 3D generative priors

## TL;DR
InstaScene proposes a unified framework for instance decomposition and complete reconstruction from cluttered scenes. It constructs a spatial contrastive learning scheme via tracked Gaussian rasterization for accurate instance segmentation, and designs an in-situ generation pipeline that leverages available observations and geometric cues to guide a 3D generative model toward complete object reconstruction.

## Background & Motivation

Humans naturally recognize and mentally complete occluded objects in cluttered environments, yet endowing robots with equivalent perceptual capabilities remains highly challenging:

**Holistic modeling vs. instance understanding**: General-purpose 3D reconstruction methods (NeRF/3DGS) treat scenes as a whole and cannot support instance-level interaction.

**Segmentation without completeness**: Open-vocabulary scene understanding methods (OpenScene/LangSplat) can query and segment objects but cannot recover complete geometry.

**Generation without alignment**: Category-specific generative methods can predict complete shapes but are inconsistent with real scenes in terms of scale and appearance.

**Noisy 2D priors**: Existing methods rely on lifting 2D segmentation masks to 3D, but masks in cluttered scenes are noisy and inconsistent across viewpoints.

**Practical demands**: Downstream tasks such as robotic manipulation, scene editing, and simulation require complete instance-level 3D models.

## Method

### Overall Architecture

InstaScene takes a pre-reconstructed 2D Gaussian Splatting scene as input and proceeds in three stages:

1. **Mask clustering and filtering**: Tracked Gaussian rasterization for cross-view 2D mask matching.
2. **Spatial contrastive learning**: Joint 2D/3D mask supervision for training an instance feature field.
3. **In-situ generation**: Incomplete objects extracted via decomposition are completed using a generative model.

### Spatial Gaussian Tracker

Core problem: How to associate multi-view 2D segmentation masks with the same 3D instance?

Approach: The method exploits the traceability of Gaussian rasterization. For each mask $m_{i,j}$ in view $I_i$, the Gaussian points that contribute significantly (transmittance > 0.5) to the masked region during rasterization are collected to form a spatial tracker $P_{i,j}$.

Cross-view matching is determined by a view consensus ratio:

$$\mathcal{C}(P_{i,j}, P_{k,l}) = \frac{N_{contain}(P_{i,j}, P_{k,l})}{N_{vis}(P_{i,j}, P_{k,l})}$$

When $\mathcal{C}$ exceeds 0.9, the two masks are considered to belong to the same instance. Under-segmented masks — where a single tracker intersects multiple trackers from the same frame — are detected and filtered.

### Spatial Contrastive Learning

Design motivation: 3D masks are sufficiently robust but sparse (after DBSCAN filtering), while 2D masks are denser but noisy. The two sources are complementary and are used jointly as supervision.

A 16-dimensional feature $f_i^{3d}$ is attached to each Gaussian and rendered to pixel-level features $\mathbf{F}$ via rasterization. The contrastive learning loss is:

$$\mathcal{L}_{CF}(\mathcal{F}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{|\{f_i\}|}\log\frac{\exp(f_i^j \cdot \bar{f}_i / \phi_i)}{\sum_{k=1}^{N}\exp(f_i^j \cdot \bar{f}_k / \phi_k)}$$

The total training loss integrates supervision from three levels:

$$\mathcal{L}_{\mathcal{F}} = \lambda_1 \mathcal{L}_{CF}(\mathbf{F}_i) + \lambda_2 \mathcal{L}_{CF}(\bar{\mathbf{F}}_i) + \lambda_3 \mathcal{L}_{CF}(\mathbf{f}_i^{3d})$$

- $\mathbf{F}_i$: intra-view 2D mask feature contrast
- $\bar{\mathbf{F}}_i$: cross-view mask feature contrast between adjacent views
- $\mathbf{f}_i^{3d}$: 3D Gaussian point feature contrast supervised by clustered 3D masks

At segmentation time, cosine similarity between each Gaussian's feature and the mean feature of a coarse 3D instance is computed, with threshold $\tau_{seg} = 0.9$.

### In-Situ Generation

Objective: The reconstructed complete object should align with the real scene in both appearance and scale, not merely be geometrically complete.

**Fully-conditioned diffusion**: All available observations are used to control a 3D diffusion model (MVDFusion), alternating among multiple optimal views as conditions:

$$\bar{\epsilon}_\theta^n = \frac{1}{N_k}\sum_{k=1}^{N_k}\epsilon_\theta^n(x_t^n, y^k, \hat{\pi}_n^k)$$

**Geometry-aware feature warping**: At each diffusion step, noisy latent features of known views are projected onto visible pixels of target views via rendered depth. Surface normals from the 2DGS-fused mesh are used to filter back-projected pixels, enforcing consistency over known regions.

**Occlusion-aware viewpoint selection**:
1. Sixteen viewpoints are placed around the segmented object.
2. The viewpoints with the least scene occlusion are selected as generation conditions.
3. Occluded viewpoints are completed by the generative model.
4. Original observations and generated views are jointly used to fine-tune the object's 2DGS.

## Key Experimental Results

### Main Results: 3D Instance Segmentation (LERF-Mask Dataset)

| Method | Figurines | Teatime | Kitchen | Avg. mIoU (%) |
|--------|-----------|---------|---------|----------------|
| LangSplat | 58.1 | 73.0 | 50.7 | 60.6 |
| GSGrouping | 59.0 | 72.3 | 43.1 | 58.1 |
| **InstaScene** | **85.7** | **93.7** | **77.3** | **85.6** |

InstaScene outperforms the best baseline by **25+ percentage points** in average mIoU, with a particularly pronounced advantage on the cluttered Kitchen scene.

### Ablation Study: Spatial Contrastive Learning Components

| Configuration | Figurines | Teatime | Kitchen | Avg. |
|---------------|-----------|---------|---------|------|
| Noisy 2D masks only | 80.3 | 90.1 | 71.2 | 80.5 |
| 3D masks only | 81.5 | 88.5 | 67.0 | 79.0 |
| + Filtered 2D masks | 83.9 | 91.4 | 75.4 | 83.6 |
| + Cross-view 2D masks (Full) | **85.7** | **93.7** | **77.3** | **85.6** |

### In-Situ Generation Quantitative Comparison (Replica-CAD Dataset)

| Method | PSNR(known)↑ | PSNR(unknown)↑ | CD↓ | F1↑ | Vol-IoU↑ |
|--------|--------------|----------------|-----|-----|----------|
| MVDFusion (single-view) | 17.19 | 17.46 | 0.081 | 0.150 | 0.531 |
| InstantMesh (single-view) | 23.05 | 22.83 | 0.045 | 0.382 | 0.570 |
| SpaRP (multi-view) | 25.09 | 23.03 | 0.037 | 0.406 | 0.590 |
| **InstaScene** | **32.57** | **29.02** | **0.016** | **0.767** | **0.716** |

InstaScene achieves rendering quality on known regions approaching the original 2DGS (31.67 dB), while substantially outperforming all baselines on unknown regions.

### Key Findings

1. **Complementary supervision is effective**: 2D masks provide density and 3D masks provide robustness; their joint use significantly outperforms either alone.
2. **Tracked rasterization is critical**: Leveraging the traceability of Gaussian rasterization for cross-view matching is more reliable than video tracking or CLIP-based features.
3. **In-situ vs. generic generation**: Generic Image-to-3D methods suffer from severe artifacts in cluttered scenes, including broken handles and scale mismatches.
4. **Geometric warping is necessary**: Alternating view conditioning alone still produces floaters and inconsistencies; incorporating geometry-aware feature warping yields significant improvement.

## Highlights & Insights

1. **Novel problem formulation**: InstaScene is the first to systematically unify scene decomposition and complete reconstruction in a single framework, bridging the gap between perception and complete modeling.
2. **Elegant exploitation of tracked rasterization**: The Gaussian Splatting rasterization process inherently encodes rich spatial association information, which is naturally repurposed for cross-view mask matching.
3. **Progressive information aggregation**: Spatial priors produced during segmentation (geometry, viewpoints, masks) directly guide the subsequent generation module.
4. **Robustness to real cluttered scenes**: The method demonstrates strong performance on complex real-world scenes from the ZipNeRF dataset.
5. **Application potential**: Decomposed complete objects can be directly used for scene manipulation (e.g., relocating a stroller), providing a foundation for robotic manipulation.

## Limitations & Future Work

1. **No support for dynamic objects**: The current framework assumes a static scene.
2. **Transparent/highly reflective objects**: Decomposition and reconstruction of transparent or highly reflective surfaces are not handled.
3. **Generative model domain gap**: The quality of in-situ generation is limited by the training data distribution of the underlying 3D generative model.
4. **Computational cost**: The full pipeline — scene reconstruction → feature field training → per-object generation — is relatively long.

## Related Work & Insights

- **LangSplat/GSGrouping**: Distill semantic features into Gaussian points, but exhibit insufficient discriminability in cluttered scenes.
- **DP-Recon** (concurrent work): Also leverages generative priors to improve sparse/occluded regions, but separates geometry completion and texture completion into two stages.
- **MaskClustering/SAI3D**: Exploit reprojection spatial consistency for 3D segmentation; InstaScene extends this with rasterization-based tracking.
- **Insight**: Segmentation and generation should not be treated in isolation — spatial priors from segmentation are key to controlling generation quality.

## Rating

⭐⭐⭐⭐ (4/5)

The problem formulation is forward-looking, and both the spatial contrastive learning design and the in-situ generation pipeline are well-motivated. Experimental comparisons are thorough and improvements are substantial. Limitations include high computational cost and the inability to handle dynamic or transparent scenes. This work represents a significant advance in cluttered scene understanding.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[ICCV 2025\] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)

<!-- RELATED:END -->
