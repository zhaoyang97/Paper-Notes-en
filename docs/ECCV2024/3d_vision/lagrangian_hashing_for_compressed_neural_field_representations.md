---
title: >-
  [Paper Note] Lagrangian Hashing for Compressed Neural Field Representations
description: >-
  [ECCV 2024][3D Vision][Neural Field Compression] Combines the Eulerian grid hash table of InstantNGP with a Lagrangian point cloud representation to store movable Gaussian feature points in hash buckets, achieving a compact neural field representation with a **1.8-2.8x reduction in parameters** without losing reconstruction quality.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Neural Field Compression"
  - "Hash Table"
  - "Lagrangian Representation"
  - "Gaussian Mixture"
  - "NeRF"
date: 2026-05-08
content_hash: be9b06555d58372d
---

# Lagrangian Hashing for Compressed Neural Field Representations

**Conference**: ECCV 2024  
**arXiv**: [2409.05334](https://arxiv.org/abs/2409.05334)  
**Code**: [https://theialab.github.io/laghashes/](https://theialab.github.io/laghashes/)  
**Area**: Model Compression  
**Keywords**: Neural Field Compression, Hash Table, Lagrangian Representation, Gaussian Mixture, NeRF

## TL;DR

Combines the Eulerian grid hash table of InstantNGP with a Lagrangian point cloud representation to store movable Gaussian feature points in hash buckets, achieving a compact neural field representation with a **1.8-2.8x reduction in parameters** without losing reconstruction quality.

## Background & Motivation

**Background**: Feature grid-based neural field methods (such as InstantNGP) have made significant progress in speed and quality, but usually require large storage overheads. Meanwhile, point-based methods like 3D Gaussian Splatting, while flexible, require millions of points.

**Limitations of Prior Work**: Existing feature grid methods adopt an Eulerian approach to distribute features over a uniform grid, failing to adaptively allocate representation budgets based on scene complexity—simple and complex regions use the same feature density.

**Key Challenge**: Eulerian grids offer high indexing efficiency but fixed spatial allocation, whereas Lagrangian point clouds provide high spatial adaptability but complex indexing that requires acceleration structures.

**Goal**: How to retain the fast indexing advantage of hash tables while allowing feature points to adaptively cluster towards regions requiring greater representation capability.

**Key Insight**: In the high-resolution hash layers of InstantNGP, expand each bucket to store multiple Gaussian feature points with positions, forming a hybrid Eulerian-Lagrangian representation.

**Core Idea**: Embed a Gaussian Mixture Model with learnable positions within each hash bucket, allowing feature points to automatically migrate to surface regions requiring higher representation capability during training.

## Method

### Overall Architecture

Based on the multi-scale hash architecture of InstantNGP, standard Eulerian feature grids are retained in the shallow (low-resolution) layers, while the hash buckets in the deep (high-resolution) layers are extended to Lagrangian Gaussian mixture representations. During query, for coordinate $\mathbf{x}$, features extracted from each layer are concatenated and fed into the MLP decoder:

$$\mathcal{F}(\mathbf{x}) = \text{MLP}(\mathbf{f}_1(\mathbf{x}) \oplus \mathbf{f}_2(\mathbf{x}) \oplus \ldots \oplus \mathbf{f}_L(\mathbf{x}); \boldsymbol{\theta})$$

### Key Designs

1. **Multi-scale Representation**: Following the $L=16$ layer hash structure of InstantNGP, resolution increases geometrically as $N_l = N_{\min} \cdot b^l$. The first $L - \tilde{L}$ layers are standard Eulerian features, and the last $\tilde{L}$ layers are Lagrangian representations. The design motivation is to introduce point clouds only in the high-resolution layers (where hash collisions are most severe) to alleviate collisions.

2. **Per-bucket Gaussian Mixture**: Each hash bucket stores $K$ isotropic Gaussians, with parameters including mean $\boldsymbol{\mu}_k$, standard deviation $\sigma_k$, and feature vector $\mathbf{f}_k$. To query position $\mathbf{x}$, the per-bucket features are aggregated via Gaussian weighting:

$$\mathbf{F}(\mathbf{x}) = \sum_k \mathcal{N}_k(\mathbf{x}) \cdot \mathbf{f}_k, \quad \mathcal{N}_k(\mathbf{x}) = \frac{1}{(2\pi)^{1/2}\sigma_k} \exp\left(-\frac{\|\mathbf{x} - \boldsymbol{\mu}_k\|_2^2}{2\sigma_k^2}\right)$$

The standard deviation is positively correlated with the grid resolution and decays from $50\times$ of the grid cell size to $5\times$ during training to ensure smooth early convergence. The core mechanism is to reuse the hash table as an indexing structure, avoiding extra nearest neighbor searches.

3. **Guidance Loss**: Inspired by the EM algorithm, a guidance loss based on KL divergence is designed to move Gaussian points toward the surface. For a sampled point $\mathbf{x}$ along a ray, the nearest Gaussian is found (E-step), and then the KL divergence between the PDF of this Gaussian and the NeRF integration weight $W(\mathbf{x}) = T(\mathbf{x}) \cdot \tau(\mathbf{x})$ is minimized (M-step):

$$\mathcal{L}_{\text{guide}}^l(\mathbf{x}) = W(\mathbf{x}) \cdot \min_{k,v}\left(-\log(\alpha_{v,l}) + \frac{\|\mathbf{x} - \boldsymbol{\mu}_{k,v,l}\|_2^2}{2\sigma_{k,v,l}^2}\right)$$

Intuitive explanation: If $W(\mathbf{x}) \approx 1$ (i.e., the point is on the surface), the mean of one Gaussian should be close to $\mathbf{x}$. Essentially, it acts as a one-way Chamfer distance between surface points and Gaussian points.

### Loss & Training

The total loss is a weighted sum of three terms:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{dist}} \mathcal{L}_{\text{dist}} + \lambda_{\text{guide}} \mathcal{L}_{\text{guide}}$$

- $\mathcal{L}_{\text{recon}}$: Pixel reconstruction loss (Huber Loss + Volume rendering)
- $\mathcal{L}_{\text{dist}}$: Distortion loss, promoting clear surfaces within the volume
- $\mathcal{L}_{\text{guide}}$: Guidance loss, with $\lambda_{\text{guide}} = 0.1$ and a warm-up schedule

Training uses Adam with lr=$10^{-2}$ for parameters, lr=$10^{-3}$ for Gaussian positions, for 20K iterations. The standard deviation of the Gaussians decays exponentially during training.

## Key Experimental Results

### Main Results - NeRF Synthetic Dataset (Novel View Synthesis)

| Method | Params | Lego | Mic | Materials | Chair | Hotdog | Ficus | Drums | Ship | Avg. PSNR↑ |
|------|--------|------|-----|-----------|-------|--------|-------|-------|------|----------|
| InstantNGP (B=2¹⁹) | 12.10M | 35.67 | 36.85 | 29.60 | 35.71 | 37.37 | 33.95 | 25.44 | 30.29 | 33.11 |
| **Ours (B=2¹⁷)** | **6.68M** | 35.60 | 36.45 | 29.63 | 35.61 | 37.23 | 33.89 | 25.67 | 30.84 | **33.12** |
| Ours (B=2¹⁷·⁹) | 12.13M | 35.74 | 36.78 | 29.66 | 35.76 | 37.30 | 34.02 | 25.75 | 31.01 | 33.25 |

### Ablation Study - Tanks & Temples

| Ablation Option | Avg. PSNR↑ |
|--------|----------|
| Full | **27.94** |
| w/o $\mathcal{L}_{\text{dist}}$ | 27.70 |
| w/o $\mathcal{L}_{\text{guide}}$ | 27.75 |

| Gaussians per Bucket K | Params | PSNR↑ |
|-------------|--------|-------|
| No Mixture | 0.50M | 27.49 |
| 2 | 0.67M | 27.82 |
| **4** | **0.92M** | **27.94** |
| 8 | 1.41M | 27.99 |

### Key Findings

- Parameter count is cut in half (6.68M vs 12.10M) while PSNR remains largely on par (33.12 vs 33.11).
- Similarly achieves 1.8x compression on the Tanks & Temples dataset without a drop in PSNR (28.55 vs 28.51).
- The guidance loss successfully migrates Gaussian points to surface regions, resolving artifacts (such as truck surface microstructures) caused by hash collisions.
- K=4 offers the best parameter-performance trade-off, and using 2 Lagrangian layers at the finest levels yields the best performance.

## Highlights & Insights

- The idea of a **hybrid Eulerian-Lagrangian** representation is ingenious: low-resolution layers with few collisions use grids, while high-resolution layers with heavy collisions use point clouds.
- Reusing the hash table as the indexing structure for point clouds avoids additional acceleration structures like KD-trees.
- The EM analogy of the guidance loss is clear and intuitive, elegantly formalizing the constraint that "points should be near the surface" into a KL divergence minimization.
- Achieves compression capabilities on par with CompactNGP (which is specifically designed for compression) but is more general-purpose.

## Limitations & Future Work

- Currently, the standard deviation decays along a fixed schedule rather than being optimized end-to-end as a learnable parameter.
- Lagrangian representations are only used in the finest two layers; more flexible layer selection strategies are worth exploring.
- While training time is comparable to InstantNGP, actual inference introduces some overhead due to the Gaussian weighting calculations.
- Direct compression comparisons with 3DGS were not conducted (only a degraded version with 12k Gaussians was compared).

## Related Work & Insights

- **InstantNGP**: The backbone architecture of this work, employing multi-resolution hash grids.
- **3D Gaussian Splatting**: The source of the Lagrangian point cloud concept, though it requires COLMAP initialization and millions of points.
- **CompactNGP**: Also pursues compact NeRF representations using a hash probing strategy.
- **Insights**: Introducing Lagrangian concepts to other grid-based methods (like TensoRF decomposition) could bring similar compression gains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The hybrid Eulerian-Lagrangian hash representation is a novel combinational innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes 2D image fitting, NeRF synthesis, real-world scenes, compression comparisons, and extensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear physical analogies, and complete mathematical derivations.
- **Value**: ⭐⭐⭐⭐ — Provides a new technical pathway for neural field compression with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)
- [\[CVPR 2025\] SiNR: Sparsity Driven Compressed Implicit Neural Representations](../../CVPR2025/3d_vision/sinr_sparsity_driven_compressed_implicit_neural_representations.md)
- [\[ECCV 2024\] Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [\[ECCV 2024\] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [\[ECCV 2024\] Invertible Neural Warp for NeRF](invertible_neural_warp_for_nerf.md)

</div>

<!-- RELATED:END -->
