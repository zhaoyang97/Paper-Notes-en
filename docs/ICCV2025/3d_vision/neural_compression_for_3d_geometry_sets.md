---
title: >-
  [Paper Note] Neural Compression for 3D Geometry Sets
description: >-
  [ICCV 2025][3D Vision][3D geometry compression] This paper proposes NeCGS, the first neural compression paradigm capable of compressing geometry sets containing thousands of diverse 3D mesh models at ratios up to 900×…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D geometry compression"
  - "neural compression"
  - "TSDF"
  - "auto-decoder"
  - "geometry sets"
date: 2026-05-08
content_hash: f3634867a0f916a1
---

# Neural Compression for 3D Geometry Sets

**Conference**: ICCV 2025
**arXiv**: [2405.15034](https://arxiv.org/abs/2405.15034)
**Code**: [GitHub](https://github.com/rsy6318/NeCGS)
**Area**: 3D Vision
**Keywords**: 3D geometry compression, neural compression, TSDF, auto-decoder, geometry sets

## TL;DR

This paper proposes NeCGS, the first neural compression paradigm capable of compressing geometry sets containing thousands of diverse 3D mesh models at ratios up to 900×, achieving high-fidelity reconstruction via a TSDF-Def implicit representation and a quantization-aware auto-decoder.

## Background & Motivation

3D mesh models are widely used in computer graphics, VR, robotics, and related fields. As geometric data grows increasingly complex, efficient compression techniques become essential.

Limitations of prior work:

- **Voxel/point-cloud-based methods** (GPCC/VPCC) require high resolutions (≥$2^{10}$) for accurate representation, introducing redundancy
- **SDF/TSDF representations** require tensors of varying sizes, and complex models demand extremely large tensors
- **Neural implicit methods** (DeepSDF) exhibit limited capacity when handling large collections of models from diverse categories
- Most methods target single models or temporally correlated sequences, and cannot handle diverse, unrelated geometry sets

## Method

### Two-Stage Pipeline

**Stage 1: Regular Geometry Representation (RGR)** — Converts irregular 3D mesh models into unified, fixed-size regular 4D tensors.
**Stage 2: Compact Neural Representation (CNR)** — Exploits intra- and inter-model geometric similarity via a quantization-aware auto-decoder.

### TSDF-Def Representation

Extends the conventional TSDF by introducing an additional deformation at each grid point:

$$\mathbf{V}(u,v,w) := [\texttt{TSDF}(u,v,w), \Delta u, \Delta v, \Delta w]$$

where $\mathbf{V} \in \mathbb{R}^{K \times K \times K \times 4}$, and the deformation offsets are consumed during surface extraction via **Differentiable Marching Cubes (DMC)**.

Optimization objective:
$$\min_{\mathbf{V}} \mathcal{E}_{Rec}(\texttt{DMC}(\mathbf{V}), \mathbf{S}) + \lambda_{Reg}\|\mathbf{V}[...,1:3]\|_1$$

The $\ell_1$ regularization suppresses unnecessary deformations, since most regions can already be accurately represented by TSDF alone.

### Quantization-Aware Auto-Decoder

Each model $\mathbf{V}_i$ is associated with a latent feature $\mathbf{F}_i \in \mathbb{R}^{K' \times K' \times K' \times C}$, where $K' \ll K$:

$$\widehat{\mathbf{V}}_i = \mathcal{D}_{\mathcal{Q}(\boldsymbol{\Theta})}(\mathcal{Q}(\mathbf{F}_i))$$

Differentiable quantization $\mathcal{Q}(\cdot)$ is integrated into training to reduce quantization error.

### Loss & Training

$$\mathcal{L}(\widehat{\mathbf{V}}_i, \mathbf{V}_i) = \|\widehat{\mathbf{V}}_i - \mathbf{V}_i\|_1 + \lambda_1\|\mathbf{M}_i \odot (\widehat{\mathbf{V}}_i - \mathbf{V}_i)\|_1 + \lambda_2(1 - \texttt{SSIM}(\widehat{\mathbf{V}}_i, \mathbf{V}_i))$$

where $\mathbf{M}_i$ is a mask assigning higher weight to grid points near the surface.

### Entropy Coding

Quantized latent features and network parameters are compressed into a bitstream via Huffman coding.

## Key Experimental Results

### Compression Efficiency Across Datasets

| Method | Compression Time (h) | Decompression Time (ms) |
|--------|---------------------|------------------------|
| GPCC | 0.62 | 562.56 |
| VPCC | 39.34 | 762.87 |
| PCGCv2 | 1.76 | 100.32 |
| Draco | 0.06 | 365.18 |
| **NeCGS** | 10.01 | **98.95** |

### Ablation Study: TSDF vs. TSDF-Def

| Representation | CD↓ | NC↑ | F1-0.005↑ |
|---------------|-----|-----|-----------|
| TSDF K=64 | High | Low | Low |
| TSDF K=128 | Medium | Medium | Medium |
| **TSDF-Def K=64** | Low | High | High |
| **TSDF-Def K=128** | **Lowest** | **Highest** | **Highest** |

### Key Findings

1. NeCGS achieves compression ratios approaching **900×** on the DT4D dataset while preserving geometric details.
2. TSDF-Def preserves fine structures such as thin surfaces at low resolution (K=64), whereas conventional TSDF fails even at K=128.
3. NeCGS achieves the fastest decompression speed (98.95 ms), which is critical for downstream applications.
4. The framework supports dynamic scenes: new models can be incrementally added to an already-compressed set.

## Highlights & Insights

1. **Elegant design of TSDF-Def** — Introducing per-grid-point deformation offsets enables low-resolution tensors to represent fine geometric structures, unifying the representation size across models of varying complexity.
2. **Set-level compression** — Exploiting cross-model geometric similarity yields compression efficiency far beyond what single-model methods can achieve.
3. **Quantization-aware training** — Integrating quantization into the training loop effectively reduces quantization error.
4. **Incremental capability** — Supporting dynamic addition of new models substantially enhances practical utility.

## Limitations & Future Work

- Compression is time-intensive (~10 hours), making it an offline process.
- Performance degrades on mixed datasets with high inter-category diversity.
- The decoder architecture is fixed; different compression ratios are achieved by adjusting decoder size.

## Related Work & Insights

- **Single-model compression**: GPCC, VPCC, Draco, PCGCv2
- **Sequence compression**: SLRMA, SMPL/SMAL-driven methods
- **Neural implicit representations**: DeepSDF, various SDF/UDF methods

## Rating

- Novelty: ⭐⭐⭐⭐ (TSDF-Def + set-level neural compression)
- Technical Depth: ⭐⭐⭐⭐ (complete two-stage design, quantization-aware training)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (four datasets, comprehensive ablations)
- Value: ⭐⭐⭐⭐ (900× compression ratio, supports dynamic model addition)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression](linr-pcgc_lossless_implicit_neural_representations_for_point_cloud_geometry_comp.md)
- [\[ICCV 2025\] Geometry Distributions](geometry_distributions.md)
- [\[ICCV 2025\] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [\[ICCV 2025\] Learning 3D Scene Analogies with Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)
- [\[ICCV 2025\] BokehDiff: Neural Lens Blur with One-Step Diffusion](bokehdiff_neural_lens_blur_with_one-step_diffusion.md)

</div>

<!-- RELATED:END -->
