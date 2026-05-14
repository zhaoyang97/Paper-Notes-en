---
title: >-
  [Paper Note] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction
description: >-
  [CVPR 2026][3D Vision][HDR novel view synthesis] This paper proposes InstantHDR, the first feed-forward HDR novel view synthesis method. It introduces a geometry-guided appearance modeling module to resolve appearance in…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "HDR novel view synthesis"
  - "feed-forward 3D reconstruction"
  - "3D Gaussian splatting"
  - "multi-exposure fusion"
  - "tone mapping meta-network"
date: 2026-05-08
content_hash: 81c63af490ab3404
---

# InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.11298](https://arxiv.org/abs/2603.11298)
**Code**: Unavailable (to be released after review)
**Area**: 3D Reconstruction / High Dynamic Range Imaging
**Keywords**: HDR novel view synthesis, feed-forward 3D reconstruction, 3D Gaussian splatting, multi-exposure fusion, tone mapping meta-network

## TL;DR

This paper proposes InstantHDR, the first feed-forward HDR novel view synthesis method. It introduces a geometry-guided appearance modeling module to resolve appearance inconsistencies in multi-exposure fusion, and employs a MetaNet to predict scene-specific tone mapping parameters for generalization. The method reconstructs HDR 3D Gaussian scenes in seconds from uncalibrated multi-exposure LDR images, achieving +2.90 dB PSNR over GaussianHDR under sparse 4-view settings at approximately 700× faster speed.

## Background & Motivation

**Background**: HDR novel view synthesis (HDR-NVS) aims to reconstruct HDR scenes from multi-exposure LDR images and render novel views at arbitrary exposures. Existing optimization-based methods (HDR-GS, GaussianHDR) already produce high-quality results.

**Limitations of Prior Work**:

1. Optimization-based methods rely heavily on known camera poses, SfM-initialized dense point clouds, and per-scene optimization (GaussianHDR requires ~30 minutes per scene), limiting practical deployment.
2. Multi-exposure inputs cause appearance inconsistencies, leading to SfM point cloud collapse and complete failure of optimization-based methods under sparse-view settings.
3. Feed-forward 3D models (e.g., AnySplat) assume appearance consistency; directly applying them to multi-exposure inputs produces severe ghosting artifacts (e.g., the same white wall exhibits drastically different brightness across exposures).
4. Different cameras apply different tone curves (AgX/Filmic/Standard), making it difficult to learn a unified tone mapping.
5. Publicly available HDR datasets are extremely scarce (HDR-NeRF contains only 12 scenes), insufficient to support feed-forward model pretraining.

**Key Challenge**: The speed advantage of the feed-forward paradigm vs. the exposure inconsistency, CRF diversity, and data scarcity inherent to HDR scenes.

**Goal**: How to rapidly reconstruct high-quality HDR 3D scenes from uncalibrated, exposure-inconsistent multi-view LDR images without per-scene optimization?

**Key Insight**: Decouple geometry and appearance via a frozen geometry backbone and a trainable appearance branch; reuse intermediate-layer attention maps from the geometry encoder to guide cross-view fusion; employ a meta-network to predict CRF parameters for single-forward adaptation to diverse cameras.

**Core Idea**: Geometry-guided appearance modeling addresses exposure-inconsistent fusion; meta-network-predicted tone mapping enables generalization → single-forward HDR reconstruction.

## Method

### Overall Architecture

Given $V$ uncalibrated multi-exposure LDR images $\{I_v, \ell_v\}$, the dual-branch architecture proceeds as follows: ① The geometry branch (frozen pretrained alternating-attention Transformer from VGGT/AnySplat) estimates depth $D_v$ and pose $p_v$; ② The appearance branch applies exposure normalization $F_E$ → geometry-guided cross-view attention fusion $F_A$ → DoG high-resolution upsampling $F_U$ → Gaussian head $F_G$ merging both branches to produce HDR 3D Gaussians; ③ MetaNet $F_M$ predicts tone mapping parameters $\theta$ → LDR rendering at arbitrary exposure $\ell$.

### Key Designs

1. **Geometry-guided Appearance Modeling**

    A three-stage pipeline:

    - **(a) Exposure Normalization $F_E$**: Computes relative log-exposure $\tilde{\ell}_v = \ell_v - \bar{\ell}$, encodes it as a $d$-dimensional embedding $e_v$ via sinusoidal positional encoding, and generates per-view affine parameters $(\gamma_v, \beta_v) = \text{FiLM}(\mathbf{e}_v, \bar{a}_v, \bar{a})$ via FiLM layers to modulate appearance tokens → aligning all views to a common irradiance level.
    - **(b) Geometry-guided Cross-view Attention $F_A$**: A key observation is that the $Q$ and $K$ matrices at layer 14 of the frozen geometry encoder already encode reliable cross-view geometric correspondences, even under exposure differences ranging from 0.5s to 32s. These $Q$, $K$ matrices are directly reused to guide cross-view fusion of appearance features: $\tilde{t}_v^A = \text{softmax}(QK^\top/\sqrt{d}) \hat{t}_v^A$ → zero additional computational overhead.
    - **(c) DoG High-resolution Upsampling $F_U$**: Patch-level features lose high-frequency texture details. A shallow CNN extracts full-resolution features $g_v$; high-frequency residuals $(g_v - g_{v\downarrow\uparrow})$ are added to the upsampled irradiance features → recovering pixel-level texture.

2. **Tone Mapping MetaNet**

    - The tone mapper $g_\theta$ is a two-layer MLP ($3 \to h \to 3$, ReLU + sigmoid) mapping log-irradiance to $[0,1]$ LDR values.
    - Unlike optimization-based methods that overfit a separate MLP per scene, MetaNet predicts all weights and biases of $g_\theta$ from scene context (LDR features $g_v$ + exposure embeddings $e_v$ + predicted HDR Gaussians $G$).
    - The concatenated inputs are encoded via strided convolutions followed by global pooling → scene-level descriptor $\theta \in \mathbb{R}^{d_\theta}$.
    - Supports single-forward adaptation to diverse camera tone curves (AgX/Filmic/Standard) without per-scene optimization.

3. **HDR-Pretrain Dataset**

    - 168 Blender-rendered indoor scenes based on HSSD open-source indoor assets.
    - Per scene: $5 \times 7$ view grids (2.5°/5° steps), 5-level exposure bracketing, 32-bit HDR ground truth, depth and normal maps.
    - Three tone mapping operators (AgX/Filmic/Standard) applied randomly to increase CRF diversity.
    - Resolution: $448 \times 448$; rendered with Cycles path tracing.
    - Fills a community gap in feed-forward HDR pretraining data (the largest prior HDR dataset contained only 16 scenes).

### Loss & Training

- $\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_g \mathcal{L}_g$, where $\mathcal{L}_{\text{RGB}} = \text{MSE}(I_v, L_v(\ell_v)) + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}}$
- $\mathcal{L}_g$ is a depth consistency loss supervised only on the top 30% confidence pixels — avoiding unreliable regions such as reflective surfaces and sky.
- The geometry encoder and decoder head are fully frozen; only the appearance branch, Gaussian head, and MetaNet are trained.
- AdamW + cosine LR, peak $2 \times 10^{-4}$, 1K warmup steps, 30K iterations, bf16 precision; trained on $8 \times$ A6000 GPUs for approximately 2 days.
- $\lambda_{\text{perc}} = 0.05$, $\lambda_g = 0.1$; 2–10 context views sampled per iteration.
- Post-optimization: low-opacity Gaussians ($\sigma < 0.01$) are pruned, followed by 1K iterations of joint MSE + SSIM optimization.

## Key Experimental Results

### Main Results (HDR-NeRF Real Dataset)

| Method | 4-view PSNR↑ | 4-view SSIM↑ | 8-view PSNR↑ | 18-view PSNR↑ | Time↓ |
|--------|-------------|-------------|-------------|--------------|-------|
| AnySplat | 12.10 | 0.517 | 13.30 | 13.91 | ~1–2s |
| GaussianHDR | ~19.26 | ~0.691 | ~24.96 | ~29.36 | ~1833s |
| HDR-GS | ~15.40 | — | — | ~28.90 | ~910s |
| InstantHDR (zero-shot) | 18.44 | 0.721 | 18.95 | 19.48 | ~1–2s |
| **InstantHDR_1K** | **22.16** | **0.762** | **25.32** | 29.19 | ~30–40s |

### Ablation Study (HDR-NeRF Real, 8-view)

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------------|-------|-------|--------|
| Full InstantHDR | **18.95** | **0.724** | **0.269** |
| w/o exposure normalization | 13.72 | — | — |
| w/o MetaNet | 16.32 | — | — |
| w/o cross-view attention | 17.63 | — | — |
| w/o high-resolution upsampling | — | — | 0.386 |

### Key Findings

- Zero-shot mode outperforms AnySplat by +5.65–+8.07 dB PSNR — exposure normalization and cross-view fusion are the core differentiators.
- Under sparse 4-view settings, InstantHDR_1K surpasses GaussianHDR by +2.90 dB (22.16 vs. 19.26) — feed-forward geometric priors effectively compensate for sparse input.
- Speed: InstantHDR_1K ~30–40s/scene vs. GaussianHDR ~1833s → ~50× speedup.
- Ablation shows exposure normalization has the largest impact (−5.23 dB) — brightness inconsistency fundamentally disrupts cross-view fusion.
- Removing MetaNet causes training instability (16.32 dB) — the model cannot adapt to different CRFs.
- Under dense 18-view settings, performance is comparable to GaussianHDR (29.19 vs. 29.36) at ~50× faster speed.

## Highlights & Insights

- **First feed-forward paradigm for HDR-NVS**: reconstruction time drops from ~30 minutes to seconds — a qualitative leap in speed.
- **Reusing intermediate-layer attention maps from a frozen geometry encoder** is an elegant design — providing reliable cross-view geometric correspondence guidance at zero additional computational cost.
- **MetaNet predicting all CRF parameters** enables "one network, multiple cameras" — a paradigm shift from per-scene MLP overfitting to meta-learned parameter prediction.
- **HDR-Pretrain dataset** fills a community gap — 168 scenes, more than 10× larger than the previously largest HDR dataset.

## Limitations & Future Work

- Zero-shot HDR output tends to be over-bright — extreme radiance values are difficult to accurately predict in a single forward pass.
- A PSNR gap remains between InstantHDR and GaussianHDR under dense-view synthetic settings (~2–6 dB) — the latter employs a dedicated 3D-2D dual-branch tone mapping.
- Only a simple single-branch tone mapper (two-layer MLP) is used; more sophisticated CRF modeling is a promising direction for improvement.
- Fine-tuning on HDR-Plenoxels real scenes is required for generalization to HDR-NeRF real scenes — a domain gap persists.
- Dynamic scene HDR reconstruction remains unexplored.

## Related Work & Insights

- **vs. GaussianHDR**: An optimization-based method requiring ~30 min/scene with SfM point cloud initialization; point cloud collapse causes artifacts under sparse views. InstantHDR requires neither poses nor point clouds, surpassing it by +2.90 dB PSNR under sparse 4-view settings.
- **vs. AnySplat**: A feed-forward 3D reconstruction method that assumes appearance consistency; multi-exposure inputs produce severe ghosting. InstantHDR outperforms it by +5.65 dB in zero-shot mode, with the core difference being exposure normalization and cross-view attention fusion.
- **vs. HDR-GS**: A strong optimization-based method but slow. InstantHDR_1K significantly outperforms it under sparse settings (22.16 vs. 15.40).
- **Insights**: The approach of reusing intermediate-layer attention maps from a frozen backbone for cross-view guidance is generalizable to multi-view segmentation, cross-view editing, and related tasks. The MetaNet paradigm of predicting module parameters is applicable to adaptive dehazing, white balance, and similar problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First feed-forward HDR-NVS; geometry-guided appearance modeling and MetaNet are both novel designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-view settings, dual LDR/HDR evaluation, comprehensive ablation, and rich qualitative results.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, method diagrams are intuitive, and experiments are well-organized.
- **Value**: ⭐⭐⭐⭐ Pioneers the feed-forward HDR-NVS direction with practically significant speed improvements, though a gap with optimization-based methods remains under dense-view settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](../../ICLR2026/3d_vision/dynamic_novel_view_synthesis_in_high_dynamic_range.md)

</div>

<!-- RELATED:END -->
