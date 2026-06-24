---
title: >-
  [Paper Note] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models
description: >-
  [CVPR 2025][Image Generation][3D Hair Reconstruction] By automatically creating the largest 3D synthetic hair dataset to date (40K styles), this paper trains a diffusion-Transformer-based scalp texture generation model. It is the first to directly predict the latent texture map of individual hair strands (rather than guide strands) conditioned on an image, achieving diverse 3D hair reconstruction including afros and bald patterns from a single image.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "3D Hair Reconstruction"
  - "Diffusion Models"
  - "Synthetic Data"
  - "Scalp Texture"
  - "Strand-Level Reconstruction"
date: 2026-05-08
content_hash: 86937ce4ecbc92cd
---

# DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2505.06166](https://arxiv.org/abs/2505.06166)  
**Code**: [Project Page](https://radualexandru.github.io/difflocks/)  
**Area**: Image Generation  
**Keywords**: 3D Hair Reconstruction, Diffusion Models, Synthetic Data, Scalp Texture, Strand-Level Reconstruction

## TL;DR
By automatically creating the largest 3D synthetic hair dataset to date (40K styles), this paper trains a diffusion-Transformer-based scalp texture generation model. It is the first to directly predict the latent texture map of individual hair strands (rather than guide strands) conditioned on an image, achieving diverse 3D hair reconstruction including afros and bald patterns from a single image.

## Background & Motivation

**Background**: 3D hair reconstruction is crucial for digital humans, but existing public datasets are extremely small (USC-HairSalon has only 343 samples, CT2Hair has only 10), leading existing methods to rely on low-dimensional intermediate representations (such as guide strands + upsampling + post-processing) to cope with the lack of data.

**Limitations of Prior Work**: (1) The guide strands + upsampling pipeline cannot model the spatial relationship between hair strands, which limits the reconstruction accuracy of complex hairstyles (especially curly hair); (2) Low-dimensional scalp embeddings (e.g., 32×32 in HAAR) lose structural details; (3) Existing methods cannot handle Afro hairstyles and baldness patterns.

**Key Challenge**: Insufficient data $\rightarrow$ must use low-dimensional representations $\rightarrow$ unable to model high-dimensional spatial structures $\rightarrow$ unable to express complex hairstyles. If data is sufficient, many simplifying assumptions are no longer needed.

**Goal**: Build a sufficiently large training set to enable the model to directly learn in high-dimensional space (256×256 scalp texture, with one hair-strand latent code per texel), bypassing guide strands and post-processing.

**Key Insight**: Automating large-scale 3D hair data generation using Blender geometry nodes to drive high-dimensional modeling through data scale.

**Core Idea**: 40K synthetic data + DINOv2 conditioning + Hourglass Diffusion Transformer to directly generate individual hair strand latents on a 256×256 scalp texture.

## Method

### Overall Architecture
Given a single input RGB image, DINOv2 extracts local and global features as conditioning signals. The diffusion model denoises a 256×256 scalp texture + density map, where each texel contains a 64-dimensional strand latent code. After sampling, texels are sampled based on the probability in the density map, and a pre-trained strand decoder is used to decode approximately 100K strands in parallel.

### Key Designs

1. **Large-Scale Automated 3D Hair Dataset**:

    - Function: Provides 40K diverse 3D hairstyle training data.
    - Mechanism: Starting from 75 manually created basic guide strands (about 50 strands each, taking a few minutes to make), the pipeline automatically interpolates, clumps, curls, adds noise, and performs physical simulation through 58 Blender geometry nodes (with 349 auxiliary nodes). 110 random parameters control the geometry and materials, and 255 HDRIs provide lighting diversity. Finally, each sample contains approximately 100K strands and a 768×768 RGB render.
    - Design Motivation: Key insight—with sufficient data, guide strands, low-dimensional embeddings, upsampling, and post-processing all become unnecessary.

2. **Strand-Level Scalp Texture Representation**:

    - Function: Encodes the full 3D hairstyle in a 2D texture space.
    - Mechanism: A Strand VAE compresses each hair strand into a 64-dimensional latent code $z$ (Encoder: 1D convolution, Decoder: modulated SIREN), which is assigned to the corresponding texel of the 256×256 UV scalp texture based on the strand's root position. The push-pull algorithm is used to fill sparse regions. An additional density map $D \in [0,1]^{256\times256}$ is introduced to represent the probability of generating a strand at each position.
    - Design Motivation: Compared to the 32×32 guide-strand texture in HAAR, the 256×256 strand-level texture can directly model the spatial relationship between individual strands. The self-attention in the Transformer is naturally suited to capture this relationship.

3. **Conditional Scalp Diffusion Model**:

    - Function: Generates scalp texture and density maps from a single image.
    - Mechanism: An Hourglass Diffusion Transformer (HDiT) is used as the backbone, with DINOv2 image features as the conditioning signal. Diffusion is performed at the pixel level (EDM framework) on the concatenated texture + density map $[T, D]$. The local patch features and global CLS features of DINOv2 provide spatial and semantic conditioning respectively.
    - Design Motivation: DINOv2 features are much richer than traditional orientation maps, encoding both appearance and semantic information. The pre-trained visual backbone enables the model trained on synthetic data to generalize well to real-world images.

### Loss & Training
Strand VAE: position L1 + direction L1 + curvature L1 + KL regularization. Diffusion model: EDM standard denoising loss. Curvature loss is particularly important for curly hair, as local shape affects visual perception more than global position.

## Key Experimental Results

### Main Results

| Method | Recall↑ | L2 Distance↓ | Supported Hair Types |
|------|---------|--------|------------|
| Neural Strands | Medium | Medium | Primarily straight |
| HAAR | Good | Good | Multiple, excluding Afros |
| **Ours** | **Best** | **Best** | **All types including Afros and baldness** |

### Ablation Study

| Configuration | Effect Description |
|------|---------|
| Guide strands (32×32) vs. Individual strands (256×256) | 256×256 resolution significantly improves reconstruction quality for curly hair |
| Without curvature loss | Degradation of shape in highly curly strands |
| DINOv2 vs. orientation map conditioning | DINOv2 generalizes better to real-world images |
| Without density map | Unable to model uneven density patterns like baldness and hairline contours |

### Key Findings
- This work successfully reconstructs Afro-style hairstyles and baldness patterns from a single image for the first time.
- Training solely on synthetic data generalizes well to real-world photos, thanks to the pre-trained features of DINOv2.
- The generated hair strands can be directly integrated into real-time game engines like Unreal Engine without requiring additional densification.
- The 40K dataset is the core driving force behind the performance improvement.

## Highlights & Insights
- The strategy of "data scale unlocking modeling dimensionality"—by expanding the data, all previously required simplifying assumptions (guide strands, low-dimensional embedding, post-processing) are removed.
- The dual-channel design of density map + latent texture elegantly handles uneven density issues (baldness, hairline).
- The introduction of curvature loss reflects a deep understanding of curly hair visual perception.

## Limitations & Future Work
- A domain gap still exists between synthetic and real data; extreme real-world scenarios may cause failure.
- Hair color/materials are not handled; only geometry is reconstructed.
- Relies on the fixed scalp topology of SMPL-X.
- Text conditioning can be introduced in the future to enable text-driven 3D hair generation.

## Related Work & Insights
- **vs HAAR**: HAAR uses a VAE to generate in a 32×32 scalp space of guide strands, whereas DiffLocks uses diffusion in a 256×256 individual strand-level space, drastically improving accuracy and diversity.
- **vs Perm**: Perm decouples global shape and local details, but is limited by small datasets. DiffLocks uses large-scale data for direct end-to-end learning.
- **vs Multi-view methods**: DiffLocks only requires a single image. While its accuracy is slightly lower, its practicality is significantly higher.

## Rating
- Novelty: ⭐⭐⭐⭐ The strategy of large-scale data and high-dimensional modeling is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative comparisons + qualitative evaluations on real images; the dataset will be publicly released.
- Writing Quality: ⭐⭐⭐⭐ Clear description of methods and detailed data generation pipeline.
- Value: ⭐⭐⭐⭐ Possesses direct application value for the digital human industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [\[CVPR 2025\] CustAny: Customizing Anything from A Single Example](custany_customizing_anything_from_a_single_example.md)
- [\[ICCV 2025\] FaceCraft4D: Animated 3D Facial Avatar Generation from a Single Image](../../ICCV2025/image_generation/facecraft4d_animated_3d_facial_avatar_generation_from_a_single_image.md)
- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)

</div>

<!-- RELATED:END -->
