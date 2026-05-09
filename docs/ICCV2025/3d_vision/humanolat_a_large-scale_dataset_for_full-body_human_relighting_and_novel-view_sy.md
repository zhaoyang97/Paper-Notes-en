---
title: >-
  [Paper Note] HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis
description: >-
  [ICCV 2025][3D Vision][Human Relighting] This paper introduces HumanOLAT — the first publicly available large-scale full-body multi-view OLAT (One-Light-at-a-Time) dataset, comprising 21 subjects × 3 poses × 40 viewpoints × 344 lighting conditions ≈ 850K frames, providing a high-quality benchmark for human relighting and novel-view synthesis.
tags:
  - ICCV 2025
  - 3D Vision
  - Human Relighting
  - OLAT Dataset
  - Light Stage
  - Novel-View Synthesis
  - Full-Body Capture
date: 2026-05-08
content_hash: 999d6839b705ee81
---

# HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis

**Conference**: ICCV 2025
**arXiv**: [2508.09137](https://arxiv.org/abs/2508.09137)
**Code**: [Project Page](https://vcai.mpi-inf.mpg.de/projects/HumanOLAT/)
**Area**: 3D Vision
**Keywords**: Human Relighting, OLAT Dataset, Light Stage, Novel-View Synthesis, Full-Body Capture

## TL;DR

This paper introduces HumanOLAT — the first publicly available large-scale full-body multi-view OLAT (One-Light-at-a-Time) dataset, comprising 21 subjects × 3 poses × 40 viewpoints × 344 lighting conditions ≈ 850K frames, providing a high-quality benchmark for human relighting and novel-view synthesis.

## Background & Motivation

### Limitations of Prior Work

Simultaneous relighting and novel-view rendering of full-body humans is a core challenge in computer vision, yet progress has been hampered by the absence of publicly available datasets:

**Light Stage equipment is scarce and expensive**: precise control of illumination and multi-view cameras is required.

**Full-body OLAT capture is inherently difficult**: larger capture volumes and longer session times are needed, during which even minor subject movement introduces significant artifacts.

**Existing datasets are limited**:

### State of the Field

ReNe / OpenIllumination: object-level only.

### Root Cause

Dynamic OLAT / Goliath-4: face or hand capture only.

### Mechanism

Ultrastage: provides only white-light and color-gradient illumination; no OLAT sequences.

## Method

### Data Capture Setup

A spherical dome equipped with:
- 40 RED Komodo 6K cameras
- 331 individually controllable LEDs (RGBAW)
- 360° coverage around the subject
- Synchronized capture at 30 FPS, 5K image resolution

### Dataset Contents

Approximately 40K frames per subject, ~850K frames in total:
- **1 white-light condition**: used for calibration, mesh reconstruction, and segmentation
- **2 color-gradient conditions**: for estimating per-pixel photometric normals
- **10 environment-light conditions**: loaded directly onto the Light Stage
- **331 OLAT conditions**: for image-based relighting

### Motion Compensation

Subjects cannot remain perfectly still during the ~11-second capture session. Following the method of Wenger et al.:
- One white-light tracking frame is inserted every 21 OLAT frames
- Co-Tracker3 is used to track ~12K sparse mesh points
- Linear interpolation is applied to obtain dense optical flow, which warps each OLAT frame to the target frame

### Photometric Normal Estimation

Normals are computed from color-gradient illuminations $g^+$, $g^-$ as:

$$\mathbf{n} = \frac{\mathbf{d}}{|\mathbf{d}|}, \quad \mathbf{d} = \frac{g^+ - g^-}{g^+ + g^-}$$

### Image-Based Relighting

Exploiting the linearity of light transport, the appearance under a target environment light $E_{\text{target}}$ is synthesized as:

$$I_{\text{target}} = \sum_{i=0}^{N_{\text{OLAT}}} \mathbf{c}_i I_i$$

## Experiments

### Inverse Rendering Method Comparison (OLAT Illumination)

### Main Results

| Method | PSNR↑ | LPIPS↓ | SSIM↑ |
|--------|-------|--------|-------|
| PRT-Gaussian | 24.06 | 0.212 | 0.810 |
| RNG | 27.38 | 0.139 | 0.905 |
| BiGS | 26.72 | 0.201 | **0.936** |
| **GS³** | **30.04** | **0.152** | 0.892 |

GS³ achieves the best overall performance, yet all methods still exhibit notable artifacts in hand and facial regions.

### Key Findings

- PRT-Gaussian struggles to accurately reconstruct geometry from OLAT frames, producing ghosting artifacts.
- GS³, RNG, and BiGS perform better but their renderings remain noticeably blurry.
- Even the best-performing methods fail to faithfully capture specular highlights and sharp cast shadows.
- IC-Light consistently fails to preserve facial details such as eye and mouth shape.

## Highlights & Insights

1. **Fills a critical data gap**: the first publicly available full-body OLAT dataset, providing an indispensable resource for the research community.
2. **Comprehensive evaluation framework**: covers multiple tasks including OLAT relighting, environment-light relighting, and illumination harmonization.
3. **Exposes limitations of existing methods**: even state-of-the-art 3DGS relighting approaches cannot adequately handle the complex light transport of the human body.
4. **High-quality data processing**: motion compensation and precise calibration (0.819-pixel reprojection error) ensure data integrity.

## Limitations & Future Work

- Only 21 subjects, limiting diversity.
- Static poses only; dynamic sequences are not supported.
- Approximately 20 LEDs are mounted on an opening hatch, introducing minor positional inaccuracies.
- Resolution of 5K, while high, is lower than that of Ultrastage (8K).

## Related Work & Insights

- Ultrastage: 100 full-body subjects but only white-light and color-gradient illumination.
- Dynamic OLAT, Goliath-4: face/hand OLAT capture only.
- IC-Light: an illumination harmonization method.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first publicly available full-body OLAT dataset)
- Technical Depth: ⭐⭐⭐⭐ (complete data processing pipeline)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multi-method, multi-task evaluation)
- Value: ⭐⭐⭐⭐⭐ (critically important for relighting research)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] IM360: Large-scale Indoor Mapping with 360 Cameras](im360_large-scale_indoor_mapping_with_360_cameras.md)

<!-- RELATED:END -->
