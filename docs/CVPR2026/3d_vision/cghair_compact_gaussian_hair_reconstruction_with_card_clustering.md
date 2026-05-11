---
title: >-
  [Paper Note] CGHair: Compact Gaussian Hair Reconstruction with Card Clustering
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] CGHair is proposed, achieving over 200× compression of appearance parameters and 4× acceleration in strand reconstruction while maintaining comparable visual quality…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "hair reconstruction"
  - "hair card clustering"
  - "compact representation"
  - "appearance compression"
date: 2026-05-08
content_hash: 21e9aee1772af1ed
---

# CGHair: Compact Gaussian Hair Reconstruction with Card Clustering

**Conference**: CVPR 2026
**arXiv**: [2604.03716](https://arxiv.org/abs/2604.03716)
**Code**: [Project Page](https://humansensinglab.github.io/CGHair/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, hair reconstruction, hair card clustering, compact representation, appearance compression

## TL;DR

CGHair is proposed, achieving over 200× compression of appearance parameters and 4× acceleration in strand reconstruction while maintaining comparable visual quality, via hair-card-guided hierarchical clustering and a shared Gaussian appearance codebook.

## Background & Motivation

**Background**: 3DGS-based hair reconstruction methods (e.g., GaussianHair) model strands via cylindrical Gaussian chains, enabling high-fidelity real-time rendering. However, dense hair modeling requires millions of Gaussian primitives, resulting in prohibitive storage and rendering costs.

**Limitations of Prior Work**: Methods such as GaussianHair assign independent spherical harmonic coefficients to each strand, ignoring the strong structural and appearance redundancy inherent in a given hairstyle, leading to severe parameter redundancy.

**Key Challenge**: High fidelity demands a large number of independent parameters, whereas practical deployment requires compact representations.

**Goal**: To exploit the intrinsic structural similarity of hair and design a compact Gaussian hair representation.

**Key Insight**: Drawing inspiration from the "hair card" concept widely used in the game and film industries—clustering similar strands into cards and sharing textures among them.

**Core Idea**: Hierarchical clustering combined with a shared appearance codebook, simultaneously achieving strand-level fidelity and extreme compression.

## Method

### Overall Architecture

A four-stage pipeline: (1) efficient strand reconstruction with generative priors; (2) hair card generation (strand clustering + card geometry + texture mapping); (3) compact Gaussian representation via card clustering (shared appearance codebook); (4) multi-view appearance optimization.

### Key Designs

1. **Generative-Prior-Accelerated Strand Reconstruction**: The PERM parametric hair geometry model is adopted, representing hair in scalp UV space as a guide map $\mathbf{G} = \mathcal{D}_{guide}(\alpha)$ and a style map $\mathbf{S} = \mathcal{D}_{style}(\beta)$, decoded via pretrained StyleGAN2 and VAE decoders. The hairstyle encodings $\alpha, \beta$ and the decoders are jointly optimized, with photometric supervision applied via cylindrical Gaussians along strands.
   **Design Motivation**: Directly optimizing latent textures is inefficient and requires expensive diffusion regularization. Leveraging PERM's strong geometric priors enables a 4× speedup (vs. GaussianHaircut) and is fully automatic (vs. GaussianHair which requires manual intervention).

2. **Hierarchical Hair Card Generation**:

   - **Strand Clustering**: Each strand's 3D points are concatenated into a vector, and k-means clustering groups them into $N_c$ clusters (800 for real hair, 400 for synthetic).
   - **Card Geometry Construction**: Using the cluster-center strand as the axis, normal vectors are solved via constrained optimization: $\{n_k^*\} = \arg\min \sum_k \sum_{p_i} \|(p_i - \bar{p}_k) \cdot n_k\|$, ensuring all strands lie within the card plane.
   - **Texture Generation**: 3D strands are projected onto the card surface by optimizing UV coordinates and normal displacements $\delta_i$, producing anti-aliased texture maps.
   **Design Motivation**: Hair cards are the industry-standard simplification for hair, directly encoding geometric and appearance redundancy.

3. **Shared Gaussian Appearance Codebook**: The $N_c$ cards are further clustered by texture features into $N_T=64$ groups, each sharing an appearance codebook of $K=10$ entries ($D=64$ dimensions). Each strand selects codebook entries via Gumbel-Softmax soft indexing:
   $\mathbf{F}_{strand} = \sum_{k=1}^K \pi_k \cdot \mathbf{F}_k$
   A globally shared MLP decoder $\phi_{dec}$ then decodes the low-dimensional features into per-Gaussian spherical harmonic coefficients:
   $[\mathbf{SH}_1, ..., \mathbf{SH}_{|\mathcal{S}|}] = \phi_{dec}(\mathbf{F}_{strand})$
   **Design Motivation**: Directly mapping codebook entries to SH coefficients for all Gaussians requires 4k+ dimensions. Extreme compression is achieved through a low-dimensional latent space combined with a shared decoder.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_p + \mathcal{L}_a + \mathcal{L}_o$$
- $\mathcal{L}_p$: L1 + D-SSIM photometric loss
- $\mathcal{L}_a$: alpha mask supervision loss
- $\mathcal{L}_o$: opacity smoothness regularization
- Geometric parameters are frozen for the first 7k iterations; thereafter, joint optimization proceeds with a reduced learning rate.

## Key Experimental Results

### Main Results — Strand Reconstruction Quality

| Metric | CGHair | $\mathcal{D}_{PCA}$ only | Frozen $\alpha,\beta$ | $\mathcal{D}_{style}$ only |
|--------|--------|--------------------------|----------------------|---------------------------|
| Pos. Error ↓ | **0.148** | 0.162 | 0.159 | 0.182 |
| Cur. Error ↓ | **6.73** | 9.12 | 6.98 | 7.48 |

### Ablation Study — Compact Representation

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Size(MB)↓ | Compression↑ |
|--------|-------|-------|--------|-----------|--------------|
| Unique (per-strand SH) | 33.99 | 0.982 | 0.019 | 163.7 | 1.0 |
| w/o card clustering | 28.00 | 0.938 | 0.042 | 22.42 | 7.30 |
| Single-SH | 30.48 | 0.958 | 0.037 | 0.46 | 355.9 |
| **Latents (full)** | **32.40** | **0.970** | **0.026** | **0.71** | **230.6** |

### Key Findings

1. The full CGHair achieves PSNR 32.40 with only 0.71 MB of appearance parameters, yielding a 230× compression ratio compared to per-strand SH (163.7 MB) with only a 1.59 dB PSNR drop.
2. Card clustering is critical: without clustering, globally shared features yield only PSNR 28.00, demonstrating the necessity of card-level structured clustering.
3. A codebook size of $K=10$ is sufficient; increasing to 90 yields only marginal PSNR improvement at the cost of significantly increased storage.
4. Latent dimension $D=64$ represents the optimal trade-off.

## Highlights & Insights

- The well-established industrial concept of "hair cards" is elegantly introduced into academic hair reconstruction, unifying structural priors with data-driven reconstruction.
- A hierarchical compression strategy is employed: strands → cards → card groups → shared codebook, exploiting redundancy at different granularities.
- The design is plug-and-play: the CGHair module can be directly applied on top of reconstructed strands from GaussianHair as a post-processing compression stage.
- The Gumbel-Softmax soft indexing enables end-to-end training, avoiding the codebook collapse commonly observed in VQ-based methods.

## Limitations & Future Work

- The cluster counts $N_c$ and $N_T$ are set manually; adaptive determination could be considered.
- The method currently supports only static hair and does not handle dynamic or motion scenarios.
- The compressed PSNR remains ~1.6 dB below the uncompressed baseline, which may be insufficient for applications demanding extreme detail fidelity.
- The method relies on PERM pretrained priors; generalization to unconventional hairstyles has not been thoroughly validated.

## Related Work & Insights

- General-purpose 3DGS compression methods such as CompGS do not exploit the structural redundancy specific to hair; CGHair serves as a paradigm for domain-specialized compression.
- The quantized encoding concept from VQVAE is successfully transferred to 3D hair representation.
- The approach offers inspiration for other 3D reconstruction tasks with strong structural redundancy, such as cloth and vegetation.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of card clustering and shared codebook is novel, though individual technical components are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-granularity ablations are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The pipeline is clearly presented with rich illustrations.
- Value: ⭐⭐⭐⭐ A 200× compression ratio carries significant implications for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)
- [\[NeurIPS 2025\] DGH: Dynamic Gaussian Hair](../../NeurIPS2025/3d_vision/dgh_dynamic_gaussian_hair.md)
- [\[NeurIPS 2025\] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction](../../NeurIPS2025/3d_vision/motion_matters_compact_gaussian_streaming_for_free-viewpoint_video_reconstructio.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[CVPR 2026\] SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation](sgi_structured_2d_gaussians_for_efficient_and_compact_large_image_representation.md)

</div>

<!-- RELATED:END -->
