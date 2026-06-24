---
title: >-
  [Paper Note] 3D Congealing: 3D-Aware Image Alignment in the Wild
description: >-
  [ECCV 2024][3D Vision][Image Alignment] 3D Congealing aligns a set of unannotated, semantically similar internet images into a shared 3D canonical space. By combining SDS guidance from a pre-trained diffusion model to obtain the 3D shape and DINO semantic feature matching to estimate poses and coordinate mappings, it requires no templates, pose annotations, or camera parameters.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Image Alignment"
  - "3D Canonical Space"
  - "NeRF"
  - "Score Distillation"
  - "Semantic Correspondence"
  - "DINO Features"
date: 2026-05-08
content_hash: 75dc6972162be2e0
---

# 3D Congealing: 3D-Aware Image Alignment in the Wild

**Conference**: ECCV 2024  
**arXiv**: [2404.02125](https://arxiv.org/abs/2404.02125)  
**Code**: [https://ai.stanford.edu/~yzzhang/projects/3d-congealing/](https://ai.stanford.edu/~yzzhang/projects/3d-congealing/) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Image Alignment, 3D Canonical Space, NeRF, Score Distillation, Semantic Correspondence, DINO Features  

## TL;DR
3D Congealing aligns a set of unannotated, semantically similar internet images into a shared 3D canonical space. By combining SDS guidance from a pre-trained diffusion model to obtain the 3D shape and DINO semantic feature matching to estimate poses and coordinate mappings, it requires no templates, pose annotations, or camera parameters.

## Background & Motivation

### Background

**Background**: Traditional "image congealing" (alignment) operates in 2D planes and can only handle scenarios with small viewpoint variations. However, internet images of the same class exhibit significant differences in shooting angles, lighting, and appearance, which 2D warping cannot handle due to large viewpoint rotations. Existing 3D reconstruction methods either require known poses (NeRF), rough pose initialization (SAMURAI), or multi-view images of the exact same object. The practical demand is: given a collection of internet images of semantically similar objects (with different shapes and textures), align them into a unified 3D space.

### Goal

**Goal**: How to align a collection of semantically similar images (even if they are not the same object instance) into a shared 3D canonical coordinate system without pose annotations, template priors, or camera parameters?

## Method

### Overall Architecture
A three-stage optimization pipeline (sharing a single canonical NeRF representation across all stages):
1. **Textual Inversion** $\rightarrow$ Find a textual embedding $y^*$ that describes the input images.
2. **SDS Optimization** $\rightarrow$ Generate the canonical 3D shape $\theta$ via Score Distillation Sampling (SDS) using MVDream and $y^*$.
3. **Pose Estimation + Coordinate Mapping** $\rightarrow$ Register each input image to the 3D shape using DINO features.

### Key Designs
1. **Diffusion Model Prior + Textual Inversion**: Instead of fine-tuning the diffusion model (which is memory-intensive), this work learns a textual embedding $y^*$ from the input images via Textual Inversion. Then, $y^*$ is frozen, and the NeRF shape is optimized using the SDS loss. This is significantly more memory-efficient than DreamBooth3D-style approaches and generalizes well across various object categories.

2. **Semantic Feature Distance Function**: Instead of using pixel-level photometric loss (which is non-robust to lighting and appearance changes), semantic features extracted from DINO-v2 ViT-G/14 are used to calculate image similarity. Since semantic features are highly tolerant of identity variations, correspondences can be established even across different instances. An auxiliary IoU mask loss is also incorporated to ensure boundary alignment.

3. **Forward/Reverse Canonical Coordinate Mapping**: A complete bidirectional 2D $\leftrightarrow$ 3D mapping is established:

    - Forward: 2D pixels $\rightarrow$ (warp to rendered image via DINO feature matching) $\rightarrow$ (query 3D coordinates via NOCS rendering) $\rightarrow$ 3D canonical coordinates
    - Reverse: 3D coordinates $\rightarrow$ (find 2D projection via nearest neighbors) $\rightarrow$ (inverse warp to the real image) $\rightarrow$ 2D pixels
    - 2D correspondence between two images = Forward mapping of image 1 + Reverse mapping of image 2

4. **Pose Initialization Strategy**: Instead of gradient-descent-based initialization (which easily falls into local optima), this work employs exhaustive search: 3 FoVs $\times$ 16 azimuths $\times$ 16 elevations = 768 candidate poses, choosing the one with the minimum semantic distance.

### Loss & Training
- Stage 1: Textual Inversion using the diffusion model training loss, 1000 steps
- Stage 2: SDS loss (MVDream backbone) to optimize NeRF, 10000 steps
- Stage 3: IoU mask loss to optimize poses for 1000 steps + semantic distance + smoothness + $\ell_2$ regularization to optimize coordinate mapping for 4000 steps
- The entire pipeline is run on a single NVIDIA A5000 24GB GPU.

## Key Experimental Results
**Pose Estimation (NAVI Dataset, 35 Scenes)**

| Method | Extra Input | Rotation Error (°) $\downarrow$ | Translation Error $\downarrow$ |
|------|----------|-----------|----------|
| GNeRF | None | High | High |
| PoseDiffusion | None (Pre-trained) | Medium | Medium |
| **3D Congealing (Ours)** | **None** | **Best** | **Best** |
| SAMURAI | Pose direction annotation | Comparable | Comparable |

**Semantic Correspondence Matching (SPair-71k)**

| Method | Mean PCK@0.1 |
|------|-------------|
| ASIC | 32.1 |
| DINOv2-ViT-G/14 | 55.0 |
| **Ours** | **57.2** |

### Ablation Study
- Removing pose initialization (No Pose Init): Performance drops significantly — pose optimization is highly prone to local optima.
- Removing IoU loss (No IoU Loss): Performance drops — initializing poses alone is not accurate enough.
- Consistent results across 3 random seeds demonstrate that the method is robust to canonical shape initialization.

## Highlights & Insights
- **Decoupling and combining the 3D prior capability of SDS with the semantic matching capability of DINO** is an elegant design: SDS is responsible for "what is a reasonable 3D shape", and DINO is responsible for "how the input image corresponds to the 3D shape".
- **Using Textual Inversion instead of DreamBooth** for image conditioning is a smart memory-saving choice.
- Analysis-by-synthesis style pose initialization (exhaustive search) may be brute-force, but is crucial for avoiding local optima.
- Establishing cross-image correspondences through a 3D intermediate representation handles large viewpoint changes much better than pure 2D feature matching.
- **Cross-category alignment** (e.g., cat + dog $\rightarrow$ shared 3D space) demonstrates the strong generalization capability of the proposed method.

## Limitations & Future Work
- Dependence on the quality of shapes generated by diffusion models: if SDS produces an incorrect shape (e.g., incorrect position of a water gun handle), all downstream steps fail.
- DINO features of symmetric objects cannot distinguish between front and back, leading to pose ambiguities.
- The optimization pipeline is relatively slow: approximately 2 hours per scene (1h for NeRF optimization + 15min for pose + 45min for mapping).
- Requires foreground masks (though they can be obtained automatically using SAM).
- Cannot handle severe occlusions or extreme deformations.

## Related Work & Insights
- **Neural Congealing/GANgealing**: 2D warping methods that fail to handle large viewpoint rotations; 3D Congealing outperforms them by leveraging 3D reasoning.
- **GNeRF**: Uses GANs for poseless NeRF, but designed for single-lighting scenes and fails under multi-lighting conditions; this work uses semantic features instead of photometric loss.
- **SAMURAI**: Requires pose orientation initialization labels; the proposed method is completely unsupervised while achieving comparable accuracy.
- **DreamBooth3D**: Fine-tunes the diffusion model to reconstruct 3D shapes but does not perform image registration; this work achieves both 3D reconstruction and registration in a much more efficient manner.

## Connection to My Research Direction
- The combined framework of SDS + DINO can inspire other tasks requiring both "3D understanding + semantic understanding".
- The concept of a 3D canonical space serves as a valuable reference for cross-instance/cross-category contrastive learning research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes a new problem (3D Congealing), features an elegant framework design, and integrates diffusion model priors with semantic matching.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rich quantitative and qualitative experiments covering pose estimation and correspondences; ablation studies validate the necessity of each component.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, rigorous methodological derivation, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ The combined utilization pattern of SDS + semantic features and the concept of a 3D canonical space are highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ICLR 2026\] TIGaussian: Disentangle Gaussians for Spatial-Aware Text-Image-3D Alignment](../../ICLR2026/3d_vision/tigaussian_disentangle_gaussians_for_spatial-awared_text-image-3d_alignment.md)
- [\[ECCV 2024\] Learning 3D-Aware GANs from Unposed Images with Template Feature Field](learning_3d-aware_gans_from_unposed_images_with_template_feature_field.md)

</div>

<!-- RELATED:END -->
