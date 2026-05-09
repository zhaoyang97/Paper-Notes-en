---
title: >-
  [Paper Note] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction
description: >-
  [CVPR 2026][3D Vision][HDR novel view synthesis] This paper proposes InstantHDR, the first feed-forward HDR novel view synthesis method. It achieves multi-exposure fusion via geometry-guided appearance modeling, and employs a meta-network to learn scene-adaptive tone mappers. The method reconstructs HDR 3D scenes from uncalibrated multi-exposure LDR images in a single forward pass, running ~700× faster than optimization-based methods (feed-forward) and ~20× faster (with post-optimization).
tags:
  - CVPR 2026
  - 3D Vision
  - HDR novel view synthesis
  - 3D Gaussian splatting
  - feed-forward reconstruction
  - tone mapping
  - multi-exposure fusion
date: 2026-05-08
content_hash: 450eb34637738233
---

# InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.11298](https://arxiv.org/abs/2603.11298)
**Code**: To be released (code, models, and datasets will be made public after review)
**Area**: 3D Vision
**Keywords**: HDR novel view synthesis, 3D Gaussian splatting, feed-forward reconstruction, tone mapping, multi-exposure fusion

## TL;DR

This paper proposes InstantHDR, the first feed-forward HDR novel view synthesis method. It achieves multi-exposure fusion via geometry-guided appearance modeling, and employs a meta-network to learn scene-adaptive tone mappers. The method reconstructs HDR 3D scenes from uncalibrated multi-exposure LDR images in a single forward pass, running ~700× faster than optimization-based methods (feed-forward) and ~20× faster (with post-optimization).

## Background & Motivation

HDR novel view synthesis aims to reconstruct HDR scenes from multi-view LDR images captured at varying exposures. Existing methods suffer from critical bottlenecks:

- **Optimization-based methods** (HDR-GS, GaussianHDR): Require accurate camera poses and dense initial point clouds; per-scene optimization takes 15–30 minutes and fails under sparse views due to SfM point cloud collapse.
- **Feed-forward methods** (AnySplat, VGGT): Fast but ignore the HDR problem; they assume exposure-invariant appearance, causing severe ghosting artifacts when directly fusing multi-exposure inputs.
- **Data bottleneck**: Publicly available HDR scene datasets are extremely scarce (at most a dozen scenes), far insufficient for large-scale pre-training of feed-forward models.

Introducing the feed-forward paradigm into HDR reconstruction faces four major challenges: exposure-induced appearance inconsistency, difficulty in pixel-level geometric alignment, varying camera response functions across devices, and scarcity of HDR data.

## Method

### Overall Architecture

A dual-branch architecture: (1) **Geometry branch** (frozen) — a pre-trained alternating-attention Transformer estimates depth maps and camera poses; (2) **Appearance branch** (trainable) — normalizes exposure, fuses cross-view irradiance, and recovers pixel-level details. Outputs from both branches are merged by a Gaussian Head to generate HDR 3D Gaussians, which are then rendered into controllable-exposure LDR images via tone-mapping parameters predicted by MetaNet.

### Key Designs

1. **Exposure Normalization $F_E$**: Defines relative log-exposure $\tilde{\ell}_v = \ell_v - \bar{\ell}$ and obtains exposure embeddings $\mathbf{e}_v$ via sinusoidal positional encoding. A FiLM layer predicts per-view affine parameters $(\gamma_v, \beta_v)$ from the exposure embeddings and appearance feature mean, modulating appearance tokens as $\hat{\boldsymbol{t}}_v^A = \boldsymbol{t}_v^A \odot (1+\gamma_v) + \beta_v$, aligning all views to a shared irradiance level. **Design motivation**: Eliminating brightness variation is a prerequisite for cross-view fusion.

2. **Geometry-Guided Cross-View Attention $F_A$**: The global attention maps from the frozen geometry encoder are found to encode reliable cross-view geometric correspondences — accurately matching elements such as leaves, cups, and door frames even under extreme exposure variation (0.5s–32s). The Q and K matrices from layer 14 are directly reused to guide appearance fusion:
$$\tilde{\boldsymbol{t}}_v^A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)\hat{\boldsymbol{t}}_v^A$$
**Design motivation**: Multi-exposure images capture complementary information (bright exposures reveal shadows; dark exposures preserve highlights), and fusion requires geometric correspondences.

3. **MetaNet Scene-Adaptive Tone Mapping**: The meta-network $F_M$ takes LDR features, exposure embeddings, and predicted HDR Gaussians as input, and uses a convolutional encoder with global pooling to predict all weights and biases $\boldsymbol{\theta}$ of a tone mapper $g_{\boldsymbol{\theta}}$ (a two-layer MLP). The tone-mapping formula is $\mathbf{L}_v(\ell) = g_{\boldsymbol{\theta}}(\log \mathbf{H}_v + (\ell - \bar{\ell})\cdot\log 2)$, supporting rendering at arbitrary target exposures. **Design motivation**: Different cameras use different color transforms (AgX, Filmic, etc.), and learning a unified mapper fails to generalize; scene-adaptive prediction is necessary.

### Loss & Training

- Total loss: $\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_g \mathcal{L}_g$
    - RGB loss: MSE + $\lambda_{\text{perc}} \cdot \mathcal{L}_{\text{perc}}$ (perceptual loss), $\lambda_{\text{perc}}=0.05$
    - Geometric consistency loss: aligns frozen DPT head depth with rendered depth, supervised only on the top-30% confidence pixels, $\lambda_g=0.1$
- No 3D or HDR supervision; end-to-end training uses only multi-view LDR images with known exposure times
- Training: 30K iterations, 8× A6000 GPUs, ~2 days, AdamW (lr=2e-4), bf16 precision
- **HDR-Pretrain dataset**: 168 Blender-rendered indoor scenes from HSSD, with diverse lighting and 3 tone-mapping operators (AgX / Filmic / Standard), 35 views × 5 exposures per scene

### Post-Optimization (Optional)

After feed-forward inference, lightweight post-optimization is available: pruning low-opacity Gaussians ($\sigma < 0.01$), followed by 1K iterations of fine-tuning with MSE + SSIM loss, taking ~30–40 seconds per scene.

## Key Experimental Results

### Main Results

| Dataset | Metric | InstantHDR | Prev. SOTA | Gain |
|--------|------|----------|----------|------|
| HDR-NeRF Real (4-view, post-opt.) | PSNR | **22.16 dB** | 19.26 (GaussianHDR) | +2.90 dB |
| HDR-NeRF Real (4-view, post-opt.) | SSIM | **0.762** | 0.691 (GaussianHDR) | +0.071 |
| HDR-NeRF Real (4-view, post-opt.) | Time | **32 s** | 1833 s (GaussianHDR) | **~57×** faster |
| HDR-NeRF Real (18-view, post-opt.) | PSNR | 29.19 | **29.36** (GaussianHDR) | −0.17 (comparable) |
| HDR-NeRF Syn (8-view, zero-shot) | PSNR | **22.58** | 14.51 (AnySplat) | +8.07 dB |
| HDR-NeRF Syn (8-view, post-opt.) | PSNR | 32.75 | **34.49** (GaussianHDR) | −1.74 |
| HDR Syn (8-view, post-opt.) | PSNR (μ-law) | **27.55** | 27.69 (HDR-GS) | comparable |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Note |
|------|-------|-------|--------|------|
| InstantHDR (full) | 18.95 | 0.724 | 0.269 | Full model |
| w/o MetaNet | 16.32 | 0.699 | 0.289 | Cannot adapt to different CRFs; unstable training |
| w/o Exposure Norm | 13.72 | 0.693 | 0.278 | **Largest drop**; brightness inconsistency disrupts fusion |
| w/o Cross-view Attn | 17.63 | 0.702 | 0.277 | Ghosting on smooth surfaces |
| w/o Upsampling | 19.20 | 0.718 | 0.386 | Coarse structure preserved but fine details lost (large LPIPS drop) |

### Key Findings

- Exposure normalization is the most critical module — removing it causes a 5.23 dB PSNR drop.
- The feed-forward model significantly outperforms optimization-based methods under sparse views (4-view: +2.90 dB vs. GaussianHDR), owing to the rich geometric priors provided by the feed-forward foundation model.
- Zero-shot HDR outputs tend to be over-bright (extreme radiance values are difficult to predict accurately in a single forward pass); 1K post-optimization largely alleviates this issue.
- The attention maps of the frozen geometry encoder reliably match features across extreme exposure variations — a finding with general implications for multi-exposure fusion.

## Highlights & Insights

- **Paradigm shift**: The first feed-forward HDR NVS method, reducing HDR reconstruction time from minutes to seconds.
- **Elegant geometry-guided appearance fusion**: Directly reusing attention maps from the frozen geometry encoder for cross-view correspondence introduces zero additional computational overhead.
- **MetaNet predicts tone-mapping parameters**: Transforms CRF learning — typically requiring per-scene optimization — into a feed-forward prediction, distinguishing it from fixed or globally learnable parameter approaches.
- **HDR-Pretrain dataset fills a critical gap**: 168 scenes far exceed existing HDR datasets (the largest of which contains only 14 scenes), advancing research on feed-forward HDR reconstruction.

## Limitations & Future Work

- Under dense views (18-view), the method remains slightly behind GaussianHDR (−0.17 dB), possibly limited by the single-branch tone-mapping design.
- Zero-shot HDR outputs tend to be over-bright; accurately predicting extreme radiance values in a feed-forward manner remains an open problem.
- The synthetic-to-real domain gap requires fine-tuning on HDR-Plenoxels real scenes for evaluation on real-world data.
- HDR-Pretrain covers only 168 indoor scenes; diversity could be extended to outdoor, dynamic, and other scenarios.

## Related Work & Insights

- Feed-forward geometry methods such as AnySplat and VGGT encode rich cross-view correspondence information in their attention maps — a finding generalizable to other tasks requiring multi-view fusion.
- FiLM conditioning for exposure normalization is a concise and effective approach, with potential for other conditioning scenarios (e.g., illumination or weather adaptation).
- The meta-learning idea of using a MetaNet to predict network weights can be extended to other rendering parameter prediction tasks requiring scene-adaptive behavior.

## Rating

- Novelty: ⭐⭐⭐⭐ — First feed-forward HDR NVS; geometry-guided appearance fusion and MetaNet tone mapping are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparisons across multiple settings (4/8/18-view, zero-shot/post-opt., LDR/HDR) with full ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Problem definition is clear, the pipeline figure is intuitive, and mathematical notation is consistent.
- Value: ⭐⭐⭐⭐⭐ — Reduces HDR 3D reconstruction from minutes to seconds with high practical value; the dataset has broad potential for reuse.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)

</div>

<!-- RELATED:END -->
