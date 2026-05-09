---
title: >-
  [Paper Note] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization
description: >-
  [ICCV 2025][Image Generation][3D shape generation] OAT proposes an adaptive octree tokenization scheme guided by quadric error metrics (QEM) that dynamically allocates token budgets according to local geometric complexity. It reduces token count by 50% while preserving reconstruction quality, and builds upon this foundation an autoregressive model, OctreeGPT, for high-quality text-to-3D generation.
tags:
  - ICCV 2025
  - Image Generation
  - 3D shape generation
  - octree
  - adaptive tokenization
  - autoregressive model
  - vector quantization
date: 2026-05-08
content_hash: 95116ffd77dd5fbe
---

# Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization

**Conference**: ICCV 2025

**arXiv**: [2504.02817](https://arxiv.org/abs/2504.02817)

**Area**: 3D Generation / Image Generation

**Keywords**: 3D shape generation, octree, adaptive tokenization, autoregressive model, vector quantization

## TL;DR

OAT proposes an adaptive octree tokenization scheme guided by quadric error metrics (QEM) that dynamically allocates token budgets according to local geometric complexity. It reduces token count by 50% while preserving reconstruction quality, and builds upon this foundation an autoregressive model, OctreeGPT, for high-quality text-to-3D generation.

## Background & Motivation

- **Inefficiency of fixed-length representations**: Existing 3D shape VAEs (e.g., 3DShape2VecSet, Craftsman) encode all shapes into fixed-size latent vectors regardless of geometric complexity. Simple cuboids and finely detailed sculptures consume identical representational capacity, leading to wasted tokens for simple shapes and insufficient representation for complex ones.
- **Limitations of conventional octrees**: Although methods such as OctFusion and XCube incorporate sparsity via sparse voxels or octrees, they still subdivide all cells containing surface geometry to the deepest level. A simple cube (8 vertices) requires a comparable number of nodes to a highly detailed sculpture.
- **Core observation**: 3D data are inherently sparse, with information concentrated on surfaces; different objects and different regions of the same object exhibit substantially varying geometric complexity, necessitating adaptive representation strategies.

## Method

### Overall Architecture

OAT consists of two core components: (1) quadric-error-guided adaptive octree tokenization, and (2) the OctreeGPT autoregressive generative model. The pipeline proceeds as follows: input mesh → point cloud sampling → adaptive octree construction → Perceiver encoding into variable-length latents → residual quantization → OctreeGPT generation.

### Key Designs

**1. Adaptive Octree Construction via Quadric Error**

Drawing on the quadric error metric (QEM) from mesh simplification, OAT measures local geometric complexity within each octree cell. For each cell, the sum of quadric matrices of all sampled points is computed and the position that minimizes the quadric error is solved. Subdivision criteria: (1) the maximum depth $L$ has not been reached; and (2) the mean quadric error exceeds a predefined threshold $T$. Intuitively, when local geometry approximates a plane the error approaches zero (no subdivision needed), whereas complex regions yield higher errors (requiring further subdivision).

**2. Perceiver-Based Adaptive Encoding**

- Each leaf node is represented using positional encoding (PE) and scale encoding (SE).
- Leaf nodes attend globally to all surface points via cross-attention (rather than only points within the local cell), capturing long-range dependencies.
- Subsequent self-attention enables information exchange among leaf nodes.
- Latents are aggregated bottom-up from leaf nodes to ancestor nodes by averaging child node representations.

**3. Multi-Scale Octree Residual Quantization**

Core idea: child nodes encode only the residual latent relative to the parent node, rather than a complete representation.

- Root nodes are quantized directly.
- Child nodes quantize the residual: $\text{Quantize}(\phi(v) - z_{\text{acc}}(v_{\text{parent}}))$
- Accumulated latents are maintained: $z_{\text{acc}}(v) = z_{\text{acc}}(v_{\text{parent}}) + z(v)$
- A shared codebook and quantization function are used throughout.

**4. OctreeGPT Autoregressive Generation**

- Octree nodes are traversed in breadth-first order; each token comprises a quantized index $q(v_i)$ and a structural code $\chi(v_i)$ (8-bit binary).
- Tree-structure positional encodings encode 3D coordinates and hierarchical depth information.
- A dual prediction head jointly predicts latent tokens and structural codes.
- The 8-bit structural code naturally encodes termination conditions for variable-length generation.
- Text conditioning: 77 CLIP embedding tokens are prepended.

### Loss & Training

- VAE training: occupancy field reconstruction (BCE loss) $+ \lambda_{\text{VQ}} \times$ VQ loss.
- GPT training: two cross-entropy losses (latent token classification + structural code classification), both 256-way.

## Key Experimental Results

### Shape Reconstruction (Discrete Latents)

| Method | Avg. Tokens | IoU | CD ($\times 10^{-3}$) |
|--------|-------------|------|-----------------------|
| Craftsman-VQ | 512 | 83.8 | 1.94 |
| OAT (w/o adaptive) | 607 | 88.3 | 1.85 |
| **OAT** | **439** | **88.6** | **1.78** |
| Craftsman-VQ | 1024 | 84.4 | 1.80 |
| **OAT** | **625** | **89.7** | **1.53** |

### Shape Generation

| Method | FID | KID ($\times 10^{-3}$) | CLIP | Time (s) |
|--------|-----|------------------------|-------|----------|
| Craftsman (image-to-3D) | 65.18 | 6.42 | 0.27 | 54.8 |
| XCube | 132.56 | 9.83 | 0.23 | 32.3 |
| Craftsman-VQ + GPT | 85.10 | 7.49 | 0.26 | 15.4 |
| **OctreeGPT** | **56.88** | **5.79** | **0.34** | **11.3** |

### Ablation Study

- Without adaptive subdivision, the uniform octree wastes tokens on simple large-volume objects (512 → 607 tokens), while occasionally saving tokens for small-volume objects.
- As token count increases, OAT consistently outperforms baselines at equivalent token budgets.
- The continuous latent variant (OAT-KL) achieves IoU 91.6 and CD 1.29 at 439 tokens, surpassing Craftsman at 512 tokens (IoU 91.0, CD 1.83).

## Highlights & Insights

1. **QEM-guided adaptive strategy**: The well-established quadric error metric from mesh simplification is innovatively applied to octree construction in an elegant and computationally efficient manner (requiring only the solution of a linear system).
2. **Seamless integration of residual quantization and tree structure**: The hierarchical relationship of the octree is exploited for residual encoding, whereby child nodes encode only incremental information, substantially compressing the latent space.
3. **Natural expression of variable-length generation**: The structural code allows the model to naturally determine when to cease subdivision, analogous to the EOS token in text generation.
4. **50% token reduction with improved quality and fastest inference**: The method achieves simultaneous gains in both efficiency and effectiveness, offering high practical deployment value.
5. **Compatible with continuous and discrete latents**: Flexible adaptation to different downstream tasks is achieved by switching between VQ and KL bottlenecks.
6. **Global rather than local attention**: Cross-attention in the encoder allows each leaf node to attend to surface points across the entire shape, effectively capturing global geometry.

## Limitations & Future Work

- The method addresses geometric shape reconstruction and generation only, **without handling texture information**.
- The adaptive subdivision threshold $T$ must be predefined and may require different settings across datasets.
- The Objaverse dataset underwent extensive filtering (800K → 207K); robustness to low-quality meshes has not been validated.
- Comparisons with XCube and OctFusion rely on pretrained models trained on different data, limiting the fairness of the evaluation to some extent.

## Related Work & Insights

- **Craftsman** [Li et al., 2024]: Fixed-length vector set representation; direct baseline for OAT.
- **OctFusion** [Xiong et al., 2024]: Octree combined with diffusion models, but employs uniform occupancy subdivision.
- **XCube** [Ren et al., 2024]: Sparse voxel features; also adopts a uniform subdivision strategy.
- **3DShape2VecSet** [Zhang et al., 2023]: Pioneered the paradigm of encoding 3D shapes as latent vector sets.
- **Perceiver** [Jaegle et al., 2021]: Query-based transformer enabling flexible-length encoding and decoding.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | 4/5 |
| Experimental Thoroughness | 4/5 |
| Value | 5/5 |
| Writing Quality | 4/5 |
| Overall | 4/5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation](../../CVPR2026/image_generation/evatok_adaptive_length_video_tokenization_for_eff.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] Generative Modeling of Shape-Dependent Self-Contact Human Poses](generative_modeling_of_shape-dependent_self-contact_human_poses.md)

</div>

<!-- RELATED:END -->
