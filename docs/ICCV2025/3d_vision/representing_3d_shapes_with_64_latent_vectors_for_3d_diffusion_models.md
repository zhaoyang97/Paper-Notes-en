---
title: >-
  [Paper Note] Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models
description: >-
  [ICCV 2025][3D Vision][3D VAE] This paper proposes COD-VAE, a two-stage autoencoder framework—comprising a progressive encoder, a triplane decoder, and uncertainty-guided token pruning—that encodes 3D shapes into only 64 one-dimensional latent vectors, achieving a 16× compression ratio and 20.8× generation speedup while maintaining reconstruction quality.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D VAE
  - Compact Latent Space
  - 3D Diffusion Models
  - Triplane Decoding
  - Token Pruning
date: 2026-05-08
content_hash: 4f9d45cd672e4a00
---

# Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models

**Conference**: ICCV 2025
**arXiv**: [2503.08737](https://arxiv.org/abs/2503.08737)
**Code**: [GitHub](https://github.com/join16/COD-VAE)
**Area**: 3D Vision
**Keywords**: 3D VAE, Compact Latent Space, 3D Diffusion Models, Triplane Decoding, Token Pruning

## TL;DR

This paper proposes COD-VAE, a two-stage autoencoder framework—comprising a progressive encoder, a triplane decoder, and uncertainty-guided token pruning—that encodes 3D shapes into only 64 one-dimensional latent vectors, achieving a 16× compression ratio and 20.8× generation speedup while maintaining reconstruction quality.

## Background & Motivation

Within the latent diffusion model (LDM) framework, constructing a compressed latent space via a VAE is critical for efficient 3D diffusion models. The 3D domain presents unique challenges: 3D objects are inherently irregular, sparse, and continuous, making direct generation considerably more difficult.

**Limitations of Prior Work**:

**Explicit 3D representations (point clouds / voxels / octrees)**: Require larger latent spaces and specially designed networks, making training and scaling difficult.

**VecSet framework**: 3DShape2VecSet established the foundational framework for encoding 3D shapes as sets of 1D vectors, but its core cross-attention layers—serving as learnable downsampling/interpolation—offer limited compression ratios. VecSet requires 512 or even 1024 latent vectors to achieve high-quality reconstruction.

**Decoding bottleneck**: VecSet directly maps latent vectors to neural fields and must process over 2 million query points ($128^3$ grid), incurring enormous computational cost.

**Core Problem**: A large number of latent vectors leads to prohibitively high computational cost in subsequent diffusion models (quadratic complexity of self-attention), severely limiting 3D generation efficiency.

**Mechanism**: Intermediate representation spaces are introduced as bridges to replace direct mappings, enabling higher-compression encoding and more efficient decoding.

## Method

### Overall Architecture

COD-VAE adopts a two-stage autoencoder scheme:
- **Encoder**: Point cloud → intermediate point patches $\mathcal{H}$ (moderate compression) → compact latent vectors $\mathcal{Z}$ (high compression) → global information propagated back to high-resolution points.
- **Decoder**: Latent vectors → triplane feature reconstruction → shallow MLP decoding of neural fields.

The key innovation is the introduction of intermediate representation spaces (point patches and triplanes) as bridges between 3D points and latent vectors.

### Key Designs

1. **Progressive Encoder**:

   Three feature sets at different resolutions: high-resolution point features $\mathcal{G}^{(0)} \in \mathbb{R}^{N \times C}$ ($N$ tokens), intermediate point patches $\mathcal{H}^{(0)} \in \mathbb{R}^{L \times C}$ ($L$ tokens), and compact vectors $\mathcal{F} \in \mathbb{R}^{M \times C}$ ($M$ tokens), where $M \ll L \ll N$.

   Each encoder block performs progressive transformations:

   $\mathcal{H}^{(l)} = \text{SelfAttn}^3(\text{CrossAttn}(\mathcal{H}^{(l-1)}, \mathcal{G}^{(l-1)}))$
   $\mathcal{F}^{(l)} = \text{SelfAttn}(\text{CrossAttn}(\mathcal{F}^{(l-1)}, \mathcal{H}^{(l)}))$
   $\mathcal{G}^{(l)} = \text{CrossAttn}(\mathcal{G}^{(l-1)}, \mathcal{F}^{(l)})$

   Core Idea: Progressive compression from high resolution → medium resolution → compact vectors, with global information propagated back to high-resolution points for further refinement. This progressive strategy achieves a higher compression ratio than VecSet's direct cross-attention mapping.

2. **Triplane Decoder**:

   Rather than directly mapping latent vectors to query-point occupancy values as in VecSet, COD-VAE first reconstructs dense triplane features, then decodes neural fields via bilinear interpolation and a shallow MLP.

   - Learnable token sequences $\mathbf{e} \in \mathbb{R}^{(R/f \times R/f) \times C}$ encode triplane token positions.
   - Initialization: initial triplane tokens are queried from the decoded vectors $\mathcal{F}'$ via cross-attention.
   - Token processing: ViT-style transformer blocks (with uncertainty-guided pruning).
   - Final projection: a linear layer projects tokens to triplane features.

   For a query point $\mathbf{q}$, features are retrieved from the three planes and summed, then passed through a shallow MLP to obtain the occupancy value. This eliminates the cross-attention bottleneck over 2M+ query points present in VecSet.

3. **Uncertainty-Guided Token Pruning**:

   Although triplane decoding is efficient, the transformer processing dense triplanes remains expensive (quadratic growth with triplane resolution).

   An auxiliary uncertainty head at the beginning of the decoder predicts reconstruction error:

   $u(\mathbf{q}) = \psi_{xy}(U_{xy}, \mathbf{q}) \cdot \psi_{yz}(U_{yz}, \mathbf{q}) \cdot \psi_{xz}(U_{xz}, \mathbf{q})$

   Only the top 25% highest-uncertainty tokens are retained for subsequent processing, pruning 75% of computation in simple regions. The uncertainty head is trained with an MSE loss to predict reconstruction error $\mathcal{L}_{rec}(\mathbf{q})$.

### Loss & Training

**Two-stage training**:

- **Stage 1**: Train the autoencoder (without KL block and latent decoder) using binary cross-entropy loss and uncertainty loss.
- **Stage 2**: Freeze Stage 1 components; train the KL block and latent decoder using MSE loss to align latent decoder output $\mathcal{F}'$ with encoder output $\mathcal{F}$, plus KL divergence regularization and reconstruction error.

This two-stage strategy allows the latent decoder to focus on channel decompression, improving VAE precision.

**Implementation details**: 4 encoder blocks + 12 decoder layers, $C = 512$, $D = 32$, patch size $f = 8$. Training samples 4096 uniformly sampled query points + 4096 near-surface points.

## Key Experimental Results

### Main Results (ShapeNet Reconstruction)

| Method | M | IoU↑ | CD↓ | F1↑ |
|--------|---|------|-----|-----|
| VecSet (AE) | 32 | 87.8 | 0.021 | 91.3 |
| VecSet (AE) | 64 | 91.2 | 0.017 | 94.7 |
| VecSet (AE) | 512 | 96.3 | 0.013 | 98.0 |
| **COD-VAE (AE)** | **32** | **96.1** | **0.012** | **98.0** |
| **COD-VAE (AE)** | **64** | **96.5** | **0.012** | **98.2** |
| VecSet (VAE) | 512 | 96.2 | 0.013 | 98.0 |
| **COD-VAE (VAE)** | **64** | **96.3** | **0.012** | **98.0** |

COD-VAE with only 64 vectors surpasses VecSet-512 in reconstruction quality. With 32 vectors, it matches VecSet-512 (IoU 96.1 vs. 96.2), achieving a **16× compression ratio**.

### Generation Results (ShapeNet Class-Conditional Generation)

| Method | M | Rendering-FID↓ | Surface-FPD↓ | Sampling Throughput↑ | End-to-End Throughput↑ |
|--------|---|----------------|-------------|----------------------|------------------------|
| VecSet | 32 | 65.56 | 0.800 | 39.77 | 2.81 |
| VecSet | 64 | 54.47 | 0.629 | 21.92 | 2.63 |
| VecSet | 512 | 44.18 | 0.521 | 2.59 | 1.16 |
| **COD-VAE** | **32** | **37.34** | **0.473** | **41.19** | **24.17** |
| **COD-VAE** | **64** | **37.05** | **0.460** | **22.49** | **16.09** |

COD-VAE with $M=64$ achieves Rendering-FID of 37.05 (vs. VecSet-512's 44.18), with end-to-end throughput **13.9×/20.8× faster**.

### Ablation Study

**Encoder design ablation (ShapeNet AE, M=32)**:

| Encoder | IoU↑ | CD↓ | F1↑ |
|---------|------|-----|-----|
| CrossAttn (VecSet) | 91.2 | 0.016 | 94.7 |
| CrossAttn ×12 | 93.4 | 0.014 | 96.5 |
| CrossAttn ×24 | 93.5 | 0.014 | 96.6 |
| **Progressive (Ours)** | **96.1** | **0.012** | **98.0** |

**Decoder design ablation**:

| Decoder | IoU↑ | CD↓ | F1↑ | Reconstruction Speed (sample/s)↑ |
|---------|------|-----|-----|----------------------------------|
| CrossAttn (VecSet) | 95.8 | 0.013 | 97.8 | 3.43 |
| CrossAttn ×3 | 96.3 | 0.013 | 98.0 | 1.47 |
| **Triplane (Ours)** | **96.1** | **0.012** | **98.0** | **42.68** |

The triplane decoder achieves 42.68 sample/s reconstruction speed, 12.4× faster than VecSet's cross-attention, with nearly identical IoU.

### Objaverse Complex Object Results

| Method | M | IoU↑ | CD↓ | F1↑ | Throughput↑ | Memory (GB) |
|--------|---|------|-----|-----|-------------|-------------|
| VecSet | 64 | 71.2 | 0.054 | 85.2 | 1.99 | 5.46 |
| VecSet | 1024 | 79.8 | 0.051 | 87.0 | 0.23 | 9.96 |
| **COD-VAE** | **64** | **79.9** | **0.046** | **88.0** | **4.97** | **3.08** |

On the more complex Objaverse dataset, COD-VAE ($M=64$) even surpasses VecSet ($M=1024$), with 21.6× higher throughput and 69% lower memory usage.

### Key Findings

1. **Intermediate representation spaces are essential**: The progressive encoder improves IoU by 4.9% over VecSet's direct mapping (91.2→96.1); simply adding more cross-attention layers cannot compensate.
2. **A large number of latent vectors is not a prerequisite for high-quality reconstruction**: 64 vectors suffice to surpass the quality of 512 vectors.
3. **Triplane decoding achieves both quality and efficiency**: 12× reconstruction speedup with comparable quality.
4. **Pruning 75% of tokens does not degrade reconstruction quality**: Uncertainty-guided pruning accurately identifies simple regions.

## Highlights & Insights

- **16× compression breakthrough**: Demonstrates that the information redundancy in 3D shapes is far greater than previously recognized; 64 vectors suffice to represent complex 3D shapes.
- **Elegant two-stage design**: Point patches serve as the bridge on the encoding side; triplanes serve as the bridge on the decoding side—each optimally suited to its role.
- **Generalizable uncertainty pruning**: The adaptive compute allocation strategy can be extended to other 3D tasks requiring processing of large token sets.
- **Effective training strategy**: The two-stage training allows the latent decoder to focus on channel decompression, yielding greater stability than end-to-end training.

## Limitations & Future Work

- Triplane representations may have inherent expressiveness limitations for non-convex objects (information gaps in the xy/yz/xz projections).
- The uncertainty head prediction is based on initial triplane tokens, and its accuracy is constrained by feature quality at that stage.
- The 25% retention ratio is fixed; adaptive adjustment could further improve efficiency.
- Texture/color reconstruction is not addressed; the current method handles geometry only (occupancy field).

## Related Work & Insights

- VecSet serves as the direct baseline; this work systematically addresses its limited compression ratio by introducing intermediate representation spaces.
- TiTok's approach to compressing latent tokens in the 2D domain inspired the exploration in the 3D direction.
- Triplane representations (e.g., EG3D) are widely used in 3D generation; this work innovatively adopts them as an intermediate representation for decoding.
- Uncertainty-based token pruning can be borrowed for efficiency optimization in visual transformers.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The two-stage autoencoder scheme is conceptually clear, and the design of intermediate representation spaces demonstrates considerable depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual datasets (ShapeNet + Objaverse), dual tasks (reconstruction + generation), and comprehensive ablation and efficiency analyses.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Excellent paper structure; the logical chain from motivation to method is complete, with clear figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ The 20.8× generation speedup has significant practical implications for 3D diffusion models; the compact latent space can serve as a foundation for a variety of downstream generation methods.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] χ: Symmetry Understanding of 3D Shapes via Chirality Disentanglement](kh_symmetry_understanding_of_3d_shapes_via_chirality_disentanglement.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)

<!-- RELATED:END -->
