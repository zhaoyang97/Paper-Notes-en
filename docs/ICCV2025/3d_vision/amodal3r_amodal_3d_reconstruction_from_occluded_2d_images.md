---
title: >-
  [Paper Note] Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images
description: >-
  [ICCV 2025][3D Vision][Amodal 3D Reconstruction] This paper proposes Amodal3R, an end-to-end occlusion-aware 3D reconstruction model that introduces mask-weighted cross-attention and an occlusion-aware attention layer on…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Amodal 3D Reconstruction"
  - "Occlusion-aware"
  - "diffusion model"
  - "Cross-Attention"
  - "TRELLIS"
date: 2026-05-08
content_hash: d314aae4f41c0323
---

# Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images

**Conference**: ICCV 2025
**arXiv**: [2503.13439](https://arxiv.org/abs/2503.13439)
**Code**: [Project Page](https://sm0kywu.github.io/Amodal3R/)
**Area**: 3D Vision / 3D Reconstruction / Occlusion Completion
**Keywords**: Amodal 3D Reconstruction, Occlusion-aware, diffusion model, Cross-Attention, TRELLIS

## TL;DR
This paper proposes Amodal3R, an end-to-end occlusion-aware 3D reconstruction model that introduces mask-weighted cross-attention and an occlusion-aware attention layer on top of TRELLIS, enabling direct reconstruction of complete 3D object geometry and appearance from partially occluded 2D images in the 3D latent space, substantially outperforming prior two-stage "2D completion → 3D reconstruction" pipelines.

## Background & Motivation
Existing image-to-3D reconstruction models (e.g., TRELLIS, Real3D, GaussianAnything) assume fully visible objects in the input images, whereas inter-object occlusion is ubiquitous in real-world scenes. Prior approaches to occluded 3D reconstruction adopt a two-stage strategy: first completing the occluded object into a full 2D image using an amodal completion model (e.g., pix2gestalt), then feeding the result into a 3D reconstruction model. This paradigm suffers from two critical drawbacks:
1. **Lack of 3D geometric understanding in 2D completion**: 2D models rely on appearance priors rather than 3D structural cues, potentially producing geometrically implausible completions.
2. **Multi-view inconsistency**: Independently applying 2D completion to each view in a multi-view setting yields inconsistent results across views, which confuses downstream 3D reconstruction (experiments show that using 4 inconsistently completed views performs worse than using a single view).

## Core Problem
How to directly reconstruct complete 3D objects (geometry + appearance) from partially occluded 2D images without relying on an intermediate 2D completion step? The core challenges are: (1) generating plausible complete 3D geometry and appearance from partial observations; and (2) ensuring geometric and photometric consistency between visible and completed regions.

## Method

### Overall Architecture
Amodal3R is built upon TRELLIS, a conditional 3D diffusion model that performs denoising in a sparse 3D latent space. The inputs consist of a partially occluded object image $x$, a visible-region mask $M_{vis}$, and an occlusion mask $M_{occ}$ (obtainable via SAM). The model extracts image features $\mathbf{c}_{dino}$ via DINOv2, and incorporates two new modules into each transformer block to inject occlusion priors, performing reconstruction and completion directly in the 3D latent space. TRELLIS itself operates in two stages: stage 1 predicts active voxel center positions (compressed occupancy), and stage 2 recovers the corresponding latent variables (SLAT representation).

### Key Designs
1. **Mask-weighted Cross-Attention**: In the existing image-conditioned cross-attention layer, the visibility mask $M_{vis}$ is converted into patch-level weights $\mathbf{c}_{vis}$ aligned with DINOv2 features, which modulate the attention matrix as: $A_{ij} = \frac{c_{vis,j} \exp(S_{ij})}{\sum_k c_{vis,k} \exp(S_{ik})}$. Tokens corresponding to visible regions receive higher attention weights, while tokens in fully occluded regions ($c_{vis,j}=0$) are skipped. A key advantage is that this modulates only the attention distribution without altering the pretrained model's parameter structure.

2. **Occlusion-aware Attention Layer**: An additional cross-attention layer is inserted after the mask-weighted cross-attention layer to process the occlusion mask $M_{occ}$ explicitly. This layer enables the model to distinguish among three region types: visible regions, occluded regions (blocked by a foreground object, requiring completion), and background regions (not belonging to the object), thereby explicitly guiding completion reasoning.

3. **Synthetic Occlusion Data Generation**: Training data is generated with occlusions via two strategies: (a) **Random 2D occlusion** — random strokes, ellipses, and rectangles are overlaid on rendered 2D images to simulate occlusion, applicable to single-view training; (b) **3D-consistent occlusion** — a random walk starting from a random triangle on the 3D mesh selects adjacent faces to form a contiguous occluded region, ensuring multi-view consistency for evaluating contact occlusion scenarios.

4. **Multi-view Input Support**: During the multi-step denoising process, different views are used as conditioning at different steps. Views are sorted by visibility $|M_{vis}|$; the most visible image conditions the early denoising steps (determining coarse geometry), while less visible views are used in later steps (detail refinement).

### Loss & Training
- Training follows the flow matching framework with objective $\min_\theta \mathbb{E} \|\upsilon_\theta(\ell^{(t)}, x, t) - (\epsilon - \ell^{(0)})\|^2$
- Fine-tuned from TRELLIS pretrained weights; only the sparse structure flow transformer and SLAT flow transformer are trained (DINOv2 encoder and VAE decoder are frozen)
- CFG (classifier-free guidance) with drop rate = 0.1
- AdamW optimizer, learning rate 1e-4
- 4 × A100 (40G), batch size 16, 20K steps, approximately 1 day of training

## Key Experimental Results

### GSO Dataset (Single-view)

| Method | FID↓ | KID↓ | COV↑ | P-FID↓ | CLIP↑ | MMD↓ |
|--------|------|------|------|--------|-------|------|
| GaussianAnything + pix2gestalt | 92.26 | 1.30 | 0.74 | 34.69 | 35.92 | 5.03 |
| Real3D + pix2gestalt | 91.21 | 2.02 | 0.75 | 23.92 | 19.61 | 9.21 |
| TRELLIS + pix2gestalt | 58.82 | 5.87 | 0.76 | 26.43 | 31.65 | 4.17 |
| **Amodal3R (Ours)** | **30.64** | **0.35** | **0.81** | **7.69** | **39.61** | **3.62** |

### Toys4K Dataset (Single-view)

| Method | FID↓ | KID↓ | COV↑ | P-FID↓ | CLIP↑ | MMD↓ |
|--------|------|------|------|--------|-------|------|
| TRELLIS + pix2gestalt | 43.05 | 6.83 | 0.80 | 26.04 | 26.28 | 6.87 |
| **Amodal3R (Ours)** | **23.45** | **0.42** | **0.83** | **5.00** | **37.09** | **5.89** |

### GSO Dataset (4-view)

| Method | FID↓ | KID↓ | COV↑ | P-FID↓ | CLIP↑ | MMD↓ |
|--------|------|------|------|--------|-------|------|
| TRELLIS + pix2gestalt+MV | 60.37 | 1.85 | 0.83 | 19.68 | 31.75 | 4.21 |
| **Amodal3R (Ours)** | **26.27** | **0.22** | **0.84** | **5.03** | **38.74** | **3.61** |

### Ablation Study
- **Naive concatenation (directly appending mask tokens to DINOv2 tokens)**: The model can perform basic completion, but occluded region textures are inconsistent with visible regions and geometry is inaccurate (FID=31.96).
- **Mask-weighted attention only**: Rendering quality and texture consistency improve, but geometric artifacts remain (e.g., holes in shoes, broken back of a monster figure) (FID=30.53).
- **Occlusion-aware layer only**: Geometry improves, but appearance quality is insufficient (FID=31.77, CLIP=40.19 is highest but KID=0.57).
- **Full model (both components)**: Achieves optimal geometry and consistent texture simultaneously (FID=30.64, KID=0.35).
- Key finding: **Multi-view inconsistent 2D completions perform worse than a single view** — independently applying pix2gestalt to 4 views and feeding into TRELLIS/LaRa degrades performance compared to using a single view.

## Highlights & Insights
- **End-to-end single-stage design**: Eliminates error accumulation from 2D completion and multi-view inconsistency by performing direct inference in the 3D latent space.
- **Elegant adaptation of pretrained models**: Mask-weighted attention modulates only the attention distribution without altering model structure; the occlusion-aware layer is inserted as a standalone module with minimal perturbation to pretrained weights.
- **Trained on synthetic data only, generalizes to real scenes**: The model produces plausible 3D assets on Replica indoor scenes and in-the-wild images.
- **Fast inference**: Generation and rendering per object takes less than 10 seconds, on par with baselines.
- **Supports diverse generation**: As a generative model, multiple distinct yet plausible 3D completions can be sampled from the same occluded input.

## Limitations & Future Work
- **Limited training data**: Only approximately 20K synthetic 3D objects, predominantly furniture categories, limiting completion capability for complex or out-of-distribution objects. Scaling to large-scale datasets such as Objaverse-XL would improve performance.
- **Synthetic-only training**: The model cannot leverage contextual cues from the environment (e.g., occluder type implying the shape of the occluded object); constructing real-world amodal 3D datasets is an important future direction.
- **Uncontrollable completion**: Completion results are entirely determined by the model; future work could introduce text conditioning to allow user control over completion style.
- **Dependency on external segmentation**: SAM is required to provide the visibility mask and occlusion mask; segmentation quality affects downstream reconstruction.

## Related Work & Insights
- **vs. pix2gestalt + TRELLIS (two-stage)**: Amodal3R substantially outperforms across all metrics (GSO single-view FID: 30.64 vs. 58.82); the two-stage approach degrades in multi-view settings due to completion inconsistency, whereas Amodal3R improves consistently with more views.
- **vs. 3D Shape Completion methods** (e.g., DiffComplete, SDFusion): These methods complete shape from partial 3D input (point cloud/voxel) but do not recover texture/appearance, and additionally require recovering partial 3D geometry from occluded images first. Amodal3R reconstructs complete 3D geometry and appearance end-to-end from 2D images.
- **vs. LaRa** (ECCV 2024 multi-view reconstruction): LaRa performs poorly in multi-view occlusion scenarios (GSO 4V FID=97.53/172.84) due to its sensitivity to input completeness.

The occlusion-aware mechanism of Amodal3R could potentially be extended toward scene-level 3D understanding, enabling compositional 3D reconstruction of entire occluded scenes (preliminary scene decomposition results are demonstrated in the paper).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Proposes the first end-to-end amodal 3D reconstruction paradigm; mask-weighted attention and occlusion-aware layer designs are elegant and effective, though the overall framework is a fine-tuned adaptation of TRELLIS with limited architectural innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation on two synthetic datasets plus Replica and in-the-wild images, with comprehensive multi-metric comparisons and complete ablation studies; however, direct quantitative comparison with 3D shape completion methods is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Clear paper structure with well-motivated and logically coherent presentation of the method; design motivations for both key modules are well articulated; derivations are concise.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for occlusion-aware 3D reconstruction with practical relevance to 3D asset reconstruction and scene understanding in real-world settings, though current utility is constrained by the scale of training data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[ICLR 2026\] NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction](../../ICLR2026/3d_vision/nova3r_non-pixel-aligned_visual_transformer_for_amodal_3d_reconstruction.md)
- [\[ICCV 2025\] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [\[ICCV 2025\] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [\[ICCV 2025\] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)

</div>

<!-- RELATED:END -->
