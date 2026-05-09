---
title: >-
  [Paper Note] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing
description: >-
  [ICCV 2025][3D Vision][HDR novel view synthesis] SeHDR is proposed as the first framework for synthesizing HDR novel views from single-exposure multi-view LDR images. It generates bracketed exposures in 3D Gaussian space (Bracketed 3D Gaussians) and merges them into an HDR scene representation via differentiable Neural Exposure Fusion (NeEF).
tags:
  - ICCV 2025
  - 3D Vision
  - HDR novel view synthesis
  - 3D Gaussian splatting
  - exposure bracketing
  - single exposure
  - differentiable rendering
date: 2026-05-08
content_hash: 14b368e08e5dc54b
---

# SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing

**Conference**: ICCV 2025
**arXiv**: [2509.20400](https://arxiv.org/abs/2509.20400)
**Code**: [https://github.com/yiyulics/SeHDR](https://github.com/yiyulics/SeHDR)
**Area**: 3D Vision / HDR Imaging
**Keywords**: HDR novel view synthesis, 3D Gaussian splatting, exposure bracketing, single exposure, differentiable rendering

## TL;DR
SeHDR is proposed as the first framework for synthesizing HDR novel views from single-exposure multi-view LDR images. It generates bracketed exposures in 3D Gaussian space (Bracketed 3D Gaussians) and merges them into an HDR scene representation via differentiable Neural Exposure Fusion (NeEF).

## Background & Motivation

**State of the Field**: HDR novel view synthesis (HDR-NVS) reconstructs HDR scenes from multi-view LDR images. Existing methods (HDR-NeRF, HDRGS) require multi-view and multi-exposure inputs — different views must provide images at different exposures to supply complementary information for over- and under-exposed regions.

**Limitations of Prior Work**: (1) Multi-exposure capture is impractical — acquiring images at different exposures from the same viewpoint risks motion blur and misalignment, while calibrating cameras across different views and exposures is even more challenging. (2) Directly applying existing multi-exposure HDR-NVS methods to single-exposure inputs fails due to the absence of complementary exposure information. (3) Applying single-image HDR reconstruction frame-by-frame before 3DGS introduces multi-view inconsistency, producing floater artifacts and blurring.

**Root Cause**: In single-exposure inputs, information in over- and under-exposed regions is lost due to quantization and saturation clipping — an inherently ill-posed problem — yet multi-exposure capture remains prohibitively difficult in practice.

**Paper Goals**: Learn an HDR scene representation from standard single-exposure multi-view LDR images for HDR novel view synthesis.

**Starting Point**: Drawing inspiration from exposure bracketing in computational photography — synthesizing Gaussians at different exposures in 3D Gaussian space and then fusing them into HDR, thereby avoiding the multi-view inconsistency introduced by 2D per-image HDR reconstruction.

**Core Idea**: Learn base 3D Gaussians in linear color space → manipulate exposures to generate multiple sets of bracketed 3D Gaussians sharing the same geometry but at different exposures → fuse them into HDR Gaussians in spherical harmonics (SH) space via differentiable Neural Exposure Fusion (NeEF).

## Method

### Overall Architecture
Single-exposure LDR multi-view input → learn base 3D Gaussians (SH coefficients parameterized in linear color space) → exposure manipulation to generate bracketed Gaussians → NeEF fuses them into HDR Gaussians in SH space → HDR novel view rendering.

### Key Designs

1. **Linear Color Space 3D Gaussians**:

    - **Function**: Transforms the color representation of 3DGS from sRGB space to linear radiance space.
    - **Mechanism**: Estimates the camera response function (CRF) and inversely maps LDR inputs to linear radiance. SH coefficients are parameterized in linear space, making exposure manipulation (scalar multiplication) physically meaningful.
    - **Design Motivation**: Exposure variation in sRGB space is nonlinear and cannot be handled by simple scaling; in linear space, exposure change is equivalent to scalar multiplication.

2. **Bracketed 3D Gaussians**:

    - **Function**: Generates multiple sets of 3D Gaussians at different exposures from single-exposure base Gaussians.
    - **Mechanism**: Keeps the geometric attributes (position, covariance) of the Gaussians fixed while adjusting only the linear color values to simulate different exposures. Higher and lower exposure values are sampled to generate multiple sets of bracketed Gaussians.
    - **Design Motivation**: A 3D extension of exposure bracketing from computational photography — operating in 3D space inherently guarantees multi-view consistency, avoiding the multi-view inconsistency of 2D single-image HDR methods.

3. **Neural Exposure Fusion (NeEF)**:

    - **Function**: Fuses multiple sets of bracketed exposure Gaussians into a single HDR Gaussian.
    - **Mechanism**: Operates directly in the SH coefficient space — learns a weighted fusion strategy over SH coefficients from different exposures. The fusion network runs in SH space and outputs HDR SH coefficients. The entire process is differentiable and trained end-to-end.
    - **Design Motivation**: Fusing in SH space rather than pixel space preserves the view-dependent illumination modeling capability of 3DGS while being more computationally efficient.

### Loss & Training
No HDR ground-truth supervision is required. End-to-end training is performed using a photometric loss between rendered LDR images and input LDR images.

## Key Experimental Results

### Main Results

| Method | Input Requirement | HDR Quality (PSNR) | Notes |
|--------|------------------|-------------------|-------|
| SeHDR | Single exposure | **+14.3 dB over baseline** | No HDR ground truth needed |
| HDRGS | Multi-exposure | Fails on single-exposure input | Relies on multi-exposure complementarity |
| Single-image HDR + 3DGS | Single exposure | Blurring / floater artifacts | Multi-view inconsistency |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Full SeHDR | Best | Linear space + bracketed exposure + NeEF |
| w/o linearization | Degraded | Exposure manipulation in sRGB space is physically unsound |
| Direct SH concatenation (w/o NeEF) | Degraded | Simple concatenation is inferior to learned fusion |
| Varying number of bracketed exposures | Moderate is optimal | Diminishing returns with too many exposures |

### Key Findings
- Performing exposure bracketing in 3D space inherently resolves multi-view consistency — this is the core advantage over 2D single-image HDR methods.
- Self-supervised training without HDR ground truth is a major practical advantage, achievable using only photometric loss on input LDR images.
- Linear color space parameterization is a critical prerequisite for physically correct exposure manipulation.

## Highlights & Insights
- **The 3D generalization of exposure bracketing** is a natural and elegant idea — lifting a classical computational photography concept into 3D Gaussian space.
- The method is directly applicable to any standard multi-view dataset without requiring specialized multi-exposure capture, substantially lowering the barrier to HDR-NVS.
- Performing fusion in SH space preserves view-dependent illumination modeling capability.

## Limitations & Future Work
- The accuracy of CRF estimation affects the quality of linearization.
- Recovery capability is limited for severely over- or under-exposed regions where information is completely lost.
- Standard 3DGS training conditions are required (multi-view images, SfM-estimated poses).
- Currently restricted to static scenes.

## Related Work & Insights
- **vs. HDR-NeRF**: Requires multi-exposure inputs and uses implicit representations; SeHDR uses single-exposure inputs with explicit 3DGS.
- **vs. HDRGS (Cai et al.)**: Also builds on 3DGS but requires multi-exposure inputs; SeHDR is the first to achieve single-exposure HDR-NVS.
- **vs. single-image HDR methods (e.g., LDR2HDR)**: 2D methods introduce multi-view inconsistency; SeHDR operates in 3D space to avoid this issue.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First single-exposure HDR-NVS framework; the 3D generalization of exposure bracketing is a conceptually novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compared against multiple methods and carefully designed baselines; the 14.3 dB improvement is substantial.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and the methodological pipeline flows naturally.
- Value: ⭐⭐⭐⭐⭐ Substantially lowers the data acquisition barrier for HDR-NVS, making it applicable to any standard multi-view dataset.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](../../CVPR2026/3d_vision/physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)
- [\[ICCV 2025\] HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis](humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)

<!-- RELATED:END -->
