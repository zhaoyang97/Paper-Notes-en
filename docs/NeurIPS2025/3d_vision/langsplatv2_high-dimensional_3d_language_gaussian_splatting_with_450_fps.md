---
title: >-
  [Paper Note] LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS
description: >-
  [NeurIPS 2025][3D Vision][3D language field] By treating each 3D Gaussian as a sparse code over a global dictionary, LangSplatV2 replaces the heavyweight decoder with a sparse coefficient field, achieving 476.2 FPS high-dimensional feature splatting and 384.6 FPS 3D open-vocabulary querying — a 47× speedup over LangSplat.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D language field
  - Gaussian splatting
  - sparse coding
  - real-time inference
  - open-vocabulary query
date: 2026-05-08
content_hash: 63dbff030349fa57
---

# LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS

**Conference**: NeurIPS 2025
**arXiv**: [2507.07136](https://arxiv.org/abs/2507.07136)
**Code**: [Project Page](https://langsplat-v2.github.io)
**Area**: 3D Vision
**Keywords**: 3D language field, Gaussian splatting, sparse coding, real-time inference, open-vocabulary query

## TL;DR

By treating each 3D Gaussian as a sparse code over a global dictionary, LangSplatV2 replaces the heavyweight decoder with a sparse coefficient field, achieving 476.2 FPS high-dimensional feature splatting and 384.6 FPS 3D open-vocabulary querying — a 47× speedup over LangSplat.

## Background & Motivation

**Background**: 3D language fields sit at the intersection of vision-language models and 3D environment modeling. LangSplat made significant progress by embedding CLIP features into 3D Gaussian Splatting, achieving a 199× speedup over LERF.

**Limitations of Prior Work**: LangSplat still falls short of real-time inference (8.2 FPS). Its heavyweight MLP decoder accounts for 97.1% of total inference time, severely limiting applications in AR, robotics, and related domains.

**Key Challenge**: CLIP features are 512-dimensional high-dimensional vectors; direct splatting is prohibitively expensive (1536-dim is 15× slower than 3-dim). However, compressing features to low dimensions via an encoder-decoder introduces a heavy MLP bottleneck.

**Goal**: Eliminate the decoder bottleneck and achieve real-time splatting of high-dimensional features.

**Key Insight**: The observation that millions of Gaussians in a scene encode only a limited number of distinct semantic concepts, making sparse coding a natural and efficient representation.

**Core Idea**: The language feature of each Gaussian is a sparse linear combination of K basis vectors from a global codebook; sparse coefficients are rendered in place of high-dimensional features.

## Method

### Overall Architecture

A $L$-dimensional sparse coefficient vector (with only $K$ non-zero entries) is learned per 3D Gaussian, alongside a shared global codebook of $L$ basis vectors each of dimension $D$. At inference: (1) splat the $K$-dimensional sparse coefficients → (2) recover the $D$-dimensional CLIP feature via matrix multiplication → (3) compute relevancy scores. The MLP decoder is entirely removed.

### Key Designs

1. **3D Sparse Coefficient Field**:

    - **Function**: Replace per-Gaussian high-dimensional language features with sparse coefficients and a global codebook.
    - **Design Motivation**: Millions of Gaussians correspond to only a limited set of semantic concepts, making them naturally amenable to sparse representation.
    - **Mechanism**: Each Gaussian's feature is $\mathbf{f}_i = \mathbf{w}_i \mathcal{S} = \sum_{l=1}^{L} w_{i,l} \mathbf{s}_l$, where $\mathbf{w}_i \in \mathbb{R}^{L}$ has only $K$ non-zero entries.
    - **Key Derivation**: $\mathbf{S} = \sum_{i \in \mathcal{N}} \mathbf{w}_i \mathcal{S} e_i = (\sum_{i \in \mathcal{N}} e_i \mathbf{w}_i) \mathcal{S}$
    - **Novelty**: Rendering $D$-dimensional features is equivalent to first rendering $L$-dimensional coefficients and then multiplying by the codebook, fully decoupling rendering dimensionality from feature dimensionality.

2. **Efficient Sparse Coefficient Splatting**:

    - **Function**: Exploit sparsity to accelerate CUDA alpha-blending.
    - **Design Motivation**: Standard splatting complexity is $O(|\mathcal{N}| \cdot L)$, which becomes a bottleneck for large $L$.
    - **Mechanism**: Each Gaussian stores only top-$K$ indices and coefficient values; alpha-blending operates exclusively on the $K$ non-zero elements.
    - Complexity is reduced from $O(|\mathcal{N}| \cdot L)$ to $O(|\mathcal{N}| \cdot K)$.
    - In practice, $K=4$; rendering three semantic scales in parallel yields an effective dimensionality of only 12.
    - **Novelty**: Rendering speed is completely decoupled from feature dimensionality.

3. **Global Codebook Learning**:

    - **Function**: Learn $L$ basis vectors of dimension $D$ for the entire scene.
    - **Design Motivation**: Capture a compact representation of all distinct semantic concepts present in the scene.
    - **Mechanism**: The $L$-dimensional sparse coefficients are softmax-normalized, top-$K$ entries are retained and re-normalized, and the codebook is learned end-to-end jointly with the coefficients.
    - **Parameters**: $L=64$, $K=4$, $D=512$ (CLIP feature dimension).
    - **Novelty**: No loss from dimensionality compression — modeling is performed directly in CLIP space.

### Loss & Training

- 3D Gaussians are first trained with RGB supervision for 30,000 iterations.
- Gaussian parameters are then frozen and the sparse coefficient field is trained for 10,000 iterations.
- OpenCLIP ViT-B/16 is used for CLIP feature extraction; SAM ViT-H is used for semantic segmentation.
- Three SAM-level semantic granularities are modeled simultaneously.

## Key Experimental Results

### Main Results

**LERF Dataset — 3D Open-Vocabulary Localization and Segmentation**:

| Method | Localization Acc (%) | Segmentation IoU (%) | Speed (FPS) |
|--------|----------------------|----------------------|-------------|
| LERF | — | — | ~0.04 |
| LangSplat | 84.3 | 51.4 | 8.2 |
| GAGS | 81.7 | 54.1 | — |
| **LangSplatV2** | **84.1** | **59.9** | **384.6** |

**Inference Time Breakdown (ms, A100 GPU)**:

| Method | Rendering | Decoding | Post-processing | Total | FPS |
|--------|-----------|----------|-----------------|-------|-----|
| LangSplat | 6.0 | 83.1 | 33.0 | 122.1 | 8.2 |
| LangSplat* | 2.0 | 83.1 | 0.5 | 85.6 | 11.7 |
| **LangSplatV2** | **2.0** | **0.1** | **0.5** | **2.6** | **384.6** |

**Feature Rendering Time on Different GPUs (ms)**:
- RTX 3090 and RTX 4090 run out of memory at feature dimensionality ≥ 1024.
- LangSplatV2 rendering time does not grow with feature dimensionality (consistently ≈ 2 ms).

### Ablation Study

**Effect of Codebook Size $L$ and Sparsity $K$** (LERF dataset, Overall IoU %):

| L / K | K=2 | K=4 | K=8 |
|-------|-----|-----|-----|
| L=32 | 56.8 | 58.1 | 58.5 |
| L=64 | 57.5 | **59.9** | 59.7 |
| L=128 | 57.2 | 59.5 | 59.8 |

($L=64$, $K=4$ achieves the best efficiency–accuracy trade-off.)

### Key Findings

- Removing the decoder yields a 47× speedup (8.2 → 384.6 FPS) while simultaneously improving segmentation accuracy by 8.5% IoU.
- The accuracy gain stems from eliminating information loss caused by dimensionality compression, as modeling is performed directly in CLIP space.
- $K=4$ is sufficient for high-quality semantic representation; further increasing $K$ yields marginal returns.
- LangSplatV2 outperforms LangSplat on the 3D-OVS and Mip-NeRF360 datasets as well.
- Complete decoupling of feature rendering speed from dimensionality is the core technical contribution.

## Highlights & Insights

- **Precise diagnosis of the decoding bottleneck**: Quantitative analysis pinpoints that 97.1% of inference time is spent in the decoding stage, motivating a targeted solution.
- **Elegant application of sparse coding**: The observation that millions of Gaussians map to a finite set of semantic concepts is both intuitive and powerful.
- **Concise and compelling mathematical derivation**: The linearity of rendering enables a clean decoupling of feature dimensionality from rendering dimensionality.
- **Engineering and theory in concert**: The CUDA sparse optimization delivers the theoretically predicted speedup in practice.

## Limitations & Future Work

- Codebook size $L$ and sparsity $K$ require manual configuration.
- The sparsity assumption may not hold in large-scale scenes with extremely rich semantics.
- Training proceeds in two stages (RGB → language); end-to-end joint training may yield further improvements.
- Adaptive $K$ values could be explored, as different regions vary in semantic complexity.

## Related Work & Insights

- LangSplat's autoencoder compression paradigm stands in sharp contrast to the sparse coding approach proposed here.
- 3D Gaussian compression methods (CompGS, LightGaussian) focus on RGB compression; this work addresses high-dimensional semantic features.
- Sparse coding has a well-established theoretical foundation in classical signal processing; its application to 3D scene understanding represents a meaningful innovation.
- The proposed approach has general utility for any task requiring high-dimensional feature splatting (CLIP / DINOv2 / SAM features).

## Rating

- Novelty: ⭐⭐⭐⭐ — The sparse coefficient field concept is clear, elegant, and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset evaluation, detailed timing analysis, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical flow from problem analysis to method derivation to experimental validation is exceptionally coherent.
- Value: ⭐⭐⭐⭐⭐ — A 47× speedup with improved accuracy directly advances the practical deployment of 3D language fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](planargs_high-fidelity_indoor_3d_gaussian_splatting_guided_by_vision-language_pl.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](../../CVPR2026/3d_vision/hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[NeurIPS 2025\] CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting](clipgaussian_universal_and_multimodal_style_transfer_based_on_gaussian_splatting.md)
- [\[NeurIPS 2025\] MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference](materialrefgs_reflective_gaussian_splatting_with_multi-view_consistent_material_.md)
- [\[NeurIPS 2025\] LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)

</div>

<!-- RELATED:END -->
