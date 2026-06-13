---
title: >-
  [Paper Note] Fused-Planes: Why Train a Thousand Tri-Planes When You Can Share?
description: >-
  [ICLR 2026][3D Vision][tri-plane] This paper proposes Fused-Planes, which decomposes the Tri-Plane representation into shared class-level basis planes (macro) and object-specific detail planes (micro) via a macro-micro d…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "tri-plane"
  - "NeRF"
  - "shared representation"
  - "large-scale 3D"
  - "latent space"
date: 2026-05-08
content_hash: 15ff98e494aa3b74
---

# Fused-Planes: Why Train a Thousand Tri-Planes When You Can Share?

**Conference**: ICLR 2026
**arXiv**: [2410.23742](https://arxiv.org/abs/2410.23742)  
**Code**: [https://fused-planes.github.io](https://fused-planes.github.io)  
**Area**: 3D Vision / Large-Scale 3D Reconstruction
**Keywords**: tri-plane, NeRF, shared representation, large-scale 3D, latent space

## TL;DR
This paper proposes Fused-Planes, which decomposes the Tri-Plane representation into shared class-level basis planes (macro) and object-specific detail planes (micro) via a macro-micro decomposition. Combined with latent-space rendering, the method achieves 7× training speedup and 3× memory reduction while maintaining or surpassing the reconstruction quality of independently trained Tri-Planes.

## Background & Motivation

**Background**: Tri-Planar NeRF is a powerful 3D representation compatible with 2D vision models, but large-scale scene reconstruction requires independent training for each object — thousands of objects entail thousands of training runs, resulting in prohibitive computational cost.

**Limitations of Prior Work**: (a) Independent training ignores the structural similarities shared across objects of the same category; (b) existing shared-representation methods (e.g., CodeNeRF) either scale poorly (C3-NeRF handles only 20 scenes) or lack the advantages of a planar structure.

**Key Insight**: 3D objects within the same category (e.g., cars) share substantial geometric and texture patterns. Decomposing each object's Tri-Plane into a weighted combination of shared bases plus an object-specific residual can significantly reduce redundant computation.

**Core Idea**: $T_i = T_i^{mic} \oplus (W_i \cdot \mathcal{B})$ — each object's Tri-Plane is composed of a weighted sum of shared basis planes (macro) and object-specific micro features (micro).

## Method

### Overall Architecture
The method trains $M=50$ shared basis planes $\mathcal{B} = \{B_1, ..., B_{50}\}$ alongside per-object micro planes $T_i^{mic}$ and weight vectors $W_i$. These are concatenated to form the Fused-Plane, which is rendered in latent space and decoded to RGB via a learned decoder.

### Key Designs

1. **Macro-Micro Decomposition**: The macro plane $T_i^{mac} = \sum_k w_i^k B_k$ encodes class-level shared features (22 dimensions), while the micro plane $T_i^{mic}$ encodes object-specific details (10 dimensions), concatenated into a 32-dimensional feature. Each object only requires storage of the micro plane (480 KB) and a weight vector (811 B), rather than a full 1.5 MB Tri-Plane.

2. **Latent-Space Rendering**: An image autoencoder (based on SD VAE) is jointly trained to perform rendering in a low-dimensional latent space rather than RGB space, reducing rendering resolution and accelerating training. Critically, the autoencoder is trained jointly with Fused-Planes (not pre-trained), ensuring reconstruction quality.

3. **Two-Stage Training Strategy**: In Regime 1, the first 500 objects are used to jointly optimize all components (basis planes, encoder, decoder). In Regime 2, the encoder is frozen and the remaining objects are trained, since the encoder has already converged during R1.

### Loss & Training
$$\mathcal{L} = \mathcal{L}^{latent} + \mathcal{L}^{RGB} + 0.1 \cdot \mathcal{L}^{ae}$$

The three loss terms supervise latent-space rendering, RGB decoding, and autoencoder reconstruction, respectively.

## Key Experimental Results

### Main Results

| Method | Training (min/obj) | Storage (MB/obj) | ShapeNet PSNR | FPS |
|---|---|---|---|---|
| Tri-Planes | 64.32 | 1.50 | 28.15 | 42.9 |
| K-Planes | 75.35 | 410.17 | 30.88 | 14.3 |
| **Fused-Planes** | **8.96** | **0.48** | **30.47** | **91.3** |
| **Fused-Planes-ULW** | **7.16** | **0.0008** | **29.02** | - |

Compared to Tri-Planes, Fused-Planes achieves 7.2× faster training, 3.2× lower storage, +2.32 dB PSNR, and 2.1× faster rendering.

### Ablation Study

| Configuration | PSNR | Training (min) | Storage (MB) |
|---|---|---|---|
| RGB space (no latent) | 27.71 | 63.52 | 0.48 |
| Micro only (no sharing) | 27.64 | 12.84 | 1.50 |
| $M=1$ basis plane | 27.69 | 8.48 | 0.48 |
| $M=50$ basis planes | **28.64** | 8.92 | 0.48 |
| $M=75$ basis planes | 29.62 | 8.99 | 1348 (total) |

### Key Findings
- **Latent-space rendering is the key to speedup**: Training in latent space reduces time from 63.52 to 8.92 minutes (7.1× speedup) without quality degradation.
- **Shared basis planes are effective**: $M=50$ is the optimal configuration; more basis planes yield diminishing returns and increased memory.
- **ULW variant achieves extreme compression**: Without micro planes, each object requires only 811 B (weight vector), yet achieves 29.02 dB PSNR.
- **Multi-category training is feasible**: Training across 4 ShapeNet categories incurs only marginal quality degradation.
- **Scaling benefits**: At 10,000 objects, total memory is only 5 GB, compared to 14.6 GB for Tri-Planes and 4 TB for K-Planes.

## Highlights & Insights
- The **macro-micro decomposition** paradigm is transferable to other 3D representations — any per-object optimization framework can potentially benefit from extracting shared bases.
- **Joint training of latent-space rendering and representation learning** is critical — a pre-trained VAE cannot adapt to the atypical distribution of NeRF features.
- The method achieves training speeds approaching Instant-NGP while preserving the planar structure (2D compatibility), which is highly valuable for downstream generative tasks such as diffusion models over feature planes.

## Limitations & Future Work
- The quality ceiling is bounded by the Tri-Plane representation itself (30.47 vs. TensoRF's 36.74) — sharing accelerates training but does not improve representational capacity.
- The number of basis planes $M$ must be defined a priori, and the optimal $M$ may vary across categories.
- Validation is limited to synthetic datasets (ShapeNet and Basel Faces); generalization to real-world scenes remains unexplored.
- The encoder-freezing strategy may fail when the category distribution exhibits large variation.

## Related Work & Insights
- **vs. Tri-Planes**: Fused-Planes serves as a direct replacement — faster, more compact, and higher quality while maintaining plane compatibility.
- **vs. CodeNeRF**: CodeNeRF employs latent codes for sharing but lacks a planar structure; Fused-Planes preserves 2D compatibility.
- **vs. Instant-NGP**: NGP achieves comparable training speed but requires 189 MB/object versus 0.48 MB/object.

## Rating
- Novelty: ⭐⭐⭐⭐ The macro-micro decomposition is elegant and effective; joint latent-space training reflects genuine insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dataset, multi-baseline evaluation with comprehensive ablations, scalability analysis, and rendering speed benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, experiments are thorough, and tables are informative.
- Value: ⭐⭐⭐⭐ A practical acceleration framework for large-scale 3D reconstruction, compatible with downstream generative applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](../../ICCV2025/3d_vision/compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [\[NeurIPS 2025\] You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering](../../NeurIPS2025/3d_vision/you_can_trust_your_clustering_model_a_parameter-free_self-boosting_plug-in_for_d.md)
- [\[CVPR 2026\] Where, What, Why: Toward Explainable 3D-GS Watermarking](../../CVPR2026/3d_vision/where_what_why_toward_explainable_3d-gs_watermarking.md)
- [\[AAAI 2026\] Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?](../../AAAI2026/3d_vision/can_protective_watermarking_safeguard_the_copyright_of_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] HyPlaneHead: Rethinking Tri-plane-like Representations in Full-Head Image Synthesis](../../NeurIPS2025/3d_vision/hyplanehead_rethinking_tri-plane-like_representations_in_full-head_image_synthes.md)

</div>

<!-- RELATED:END -->
