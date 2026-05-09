---
title: >-
  [Paper Note] Station2Radar: Query-Conditioned Gaussian Splatting for Precipitation Field
description: >-
  [ICLR 2026][3D Vision][Gaussian Splatting] This paper proposes Query-Conditioned Gaussian Splatting (QCGS), the first method to introduce 2D Gaussian Splatting into precipitation field generation. By fusing satellite imagery with sparse automatic weather station (AWS) observations, QCGS achieves flexible-resolution precipitation field reconstruction without radar input, reducing RMSE by over 50% compared to conventional gridded products.
tags:
  - ICLR 2026
  - 3D Vision
  - Gaussian Splatting
  - Precipitation Field Reconstruction
  - Implicit Neural Representation
  - Satellite-Station Fusion
  - Resolution-Agnostic Rendering
date: 2026-05-08
content_hash: 06116aa8a89f57de
---

# Station2Radar: Query-Conditioned Gaussian Splatting for Precipitation Field

**Conference**: ICLR 2026
**arXiv**: [2603.00418](https://arxiv.org/abs/2603.00418)
**Code**: N/A
**Area**: 3D Vision / Meteorological Remote Sensing
**Keywords**: Gaussian Splatting, Precipitation Field Reconstruction, Implicit Neural Representation, Satellite-Station Fusion, Resolution-Agnostic Rendering

## TL;DR
This paper proposes Query-Conditioned Gaussian Splatting (QCGS), the first method to introduce 2D Gaussian Splatting into precipitation field generation. By fusing satellite imagery with sparse automatic weather station (AWS) observations, QCGS achieves flexible-resolution precipitation field reconstruction without radar input, reducing RMSE by over 50% compared to conventional gridded products.

## Background & Motivation

**Background**: Precipitation forecasting relies on heterogeneous data sources — weather radars offer high accuracy but limited geographic coverage and high maintenance costs; weather stations provide precise point measurements but are extremely sparse; satellites offer high-resolution wide-area coverage but cannot directly retrieve rainfall estimates. Most existing deep learning precipitation methods (e.g., ConvLSTM, diffusion models) treat radar as the primary input.

**Limitations of Prior Work**: Radar networks are unavailable across much of the world, particularly in developing countries, limiting the applicability of radar-centric approaches. Conventional radar-free methods rely primarily on classical interpolation techniques (Barnes, Kriging), which spread station observations onto a grid using fixed Gaussian weights. These methods severely blur precipitation boundaries and are highly sensitive to station density. Satellite-direct estimation methods (e.g., Sat2Radar) suffer from systematic bias and produce fixed-resolution outputs.

**Key Challenge**: Accurate precipitation field reconstruction simultaneously requires: (1) ground-truth anchoring accuracy (available only from stations), (2) spatially continuous coverage (available only from satellites), and (3) resolution flexibility (absent from all existing methods). No prior framework unifies these three properties.

**Goal**: How can satellite imagery and sparse AWS observations be fused — without relying on radar — to generate high-resolution, structurally coherent, continuous precipitation fields?

**Key Insight**: The authors observe that classical Gaussian-weighted interpolation is mathematically equivalent to a special case of Gaussian Splatting — traditional interpolation uses fixed isotropic kernels, whereas GS allows learnable anisotropic kernels, adaptive amplitudes, and resolution-agnostic rendering. This observation bridges classical meteorological methods with emerging computer vision techniques.

**Core Idea**: Combine 2D Gaussian Splatting with implicit neural representations, conditioning on satellite features to predict adaptive Gaussian parameters, and selectively render only within precipitation support regions, enabling efficient, resolution-flexible precipitation field generation.

## Method

### Overall Architecture
QCGS is a three-stage pipeline. The inputs are satellite brightness temperature images (2 km resolution) and sparse AWS observations; the output is a continuous precipitation field at arbitrary resolution. (1) A Radar Point Proposal Network fuses satellite and AWS information to generate a coarse precipitation proxy field and identify rainfall support locations. (2) A Rainfall-Aware Point Sampling strategy selects query points from the proxy field. (3) An INR-based Gaussian parameter estimator predicts splatting parameters for each query point, which are composited via differentiable 2D Gaussian rendering to produce the final precipitation field. Training proceeds in two stages — the proposal network is trained first, followed by the Gaussian rendering module with the proposal network fixed.

### Key Designs

1. **Radar Point Proposal Network**:

    - Function: Fuses satellite imagery and AWS observations to generate a coarse precipitation proxy field and extract rainfall support locations.
    - Mechanism: A Graph Attention Network (GAT, 3 layers, 8 heads) extracts robust representations $z^t$ from irregular AWS observations, handling missing values and outliers. Satellite images are processed by a ConvNeXt U-Net encoder-decoder, with AWS representations injected into the decoder via cross-attention. The network outputs a coarse precipitation field $\hat{R}^t$ and candidate rainfall locations.
    - Design Motivation: AWS data, though sparse, provides precise ground-truth anchoring, while satellites offer spatially dense coverage with only indirect precipitation signals. The two are complementary — GAT handles the irregularity and noise of AWS data, while cross-attention enables cross-modal fusion. Ablation results show that adding AWS fusion improves CSI from 0.62 to 0.73.

2. **Rainfall-Aware Point Sampling**:

    - Function: Intelligently selects query points from the coarse precipitation field, concentrating computation on regions where precipitation is present.
    - Mechanism: Constructs a convex combination of three sampling probability terms — a gradient term $G$ emphasizing precipitation boundaries, a uniform term $U$ ensuring spatial coverage, and a heavy-rainfall term $H$ using softmax temperature to prioritize intense precipitation regions. The mixing weights are 0.3/0.4/0.3, with non-maximum suppression to avoid redundant points. Unlike standard GS, which renders the entire image plane, QCGS renders only query regions with precipitation.
    - Design Motivation: Light precipitation rarely causes high-impact events, whereas extreme precipitation events are of primary operational concern. Uniform sampling treats all regions equally, wasting computation on precipitation-free areas. Ablation experiments confirm that the three-term combination (CSI 0.76) outperforms any single term or two-term combination.

3. **INR-based Gaussian Parameter Estimator**:

    - Function: Predicts learnable Gaussian splatting parameters (covariance and amplitude) for each query point, enabling resolution-agnostic adaptive rendering.
    - Mechanism: Conditioned on intermediate satellite features, cross-attention predicts $\{\sigma_x, \sigma_y, \rho, \alpha\}$ for each query point — the first three define an anisotropic covariance matrix, and $\alpha$ controls amplitude. A key innovation: at AWS stations with nonzero observed precipitation, $\alpha$ is directly set to the observed value, serving as a ground-truth anchor. A 5-layer MLP (hidden size 128, sinusoidal positional encoding) is used as the INR network.
    - Design Motivation: Conventional GS methods optimize per image and cannot generalize. Conditioning on an INR enables parameter prediction to generalize across regions and seasons. AWS anchoring directly injects ground-truth precision constraints, avoiding the systematic bias of purely satellite-driven approaches.

### Loss & Training
The total loss combines reconstruction error and regularization: $\mathcal{L} = \text{MSE}(\tilde{R}^t, R^t) + \lambda_\sigma \sum_n (\sigma_x^{(n)} + \sigma_y^{(n)}) + \lambda_\alpha \sum_n \alpha^{(n)}$. Regularization terms on covariance and amplitude ($\lambda_\sigma = 10^{-3}$, $\lambda_\alpha = 10^{-4}$) prevent Gaussian kernels from over-expanding and causing excessive smoothing. Training uses the Adam optimizer (lr $10^{-4}$, cosine schedule), batch size 16, 100 epochs, on 8× H200 GPUs.

## Key Experimental Results

### Main Results

| Method | Type | Resolution | RMSE↓ | CSI↑ | FSS↑ | CC↑ |
|--------|------|------------|-------|------|------|-----|
| Pix2PixHD | Deep Learning | 0.5 km | 2.45 | 0.59 | 0.71 | 0.55 |
| NPM | Deep Learning | 0.5 km | 1.95 | 0.59 | 0.78 | 0.68 |
| BBDM | Deep Learning | 0.5 km | 1.68 | 0.64 | 0.84 | 0.75 |
| Kriging | Classical Interpolation | 2.0 km | 2.43 | 0.50 | 0.69 | 0.45 |
| **QCGS** | **Ours** | **0.5 km** | **1.23** | **0.74** | **0.91** | **0.90** |
| **QCGS** | **Ours** | **2.0 km** | **1.00** | **0.76** | **0.96** | **0.93** |

In comparison with global operational products on daily accumulated precipitation, QCGS also outperforms substantially: RMSE 6.68 vs. IMERG 14.08 / MSWEP 12.44; CC 0.95 vs. the best competing value of 0.78.

### Ablation Study

| Configuration | CSI↑ | Notes |
|---------------|------|-------|
| U-Net (ConvNeXt) only | 0.62 | Satellite-only baseline |
| + AWS fusion | 0.73 | Station fusion contributes +0.11 |
| + AWS fusion + GS (full) | 0.76 | GS rendering adds +0.03 |
| Uniform sampling only | 0.68 | Lacks boundary and heavy-rain focus |
| Three-term mixed sampling | 0.76 | Optimal combination |
| K = 1000 points | 0.69 | Insufficient point count |
| K = 6000 points | 0.76 | Best efficiency–accuracy trade-off |
| K = 9000 points | 0.77 | Diminishing returns |

### Key Findings
- AWS fusion is the largest contributing factor (CSI +0.11), demonstrating that sparse but precise ground observations are critical for precipitation field reconstruction.
- The resolution flexibility afforded by Gaussian Splatting allows a model trained at 2 km to outperform deep learning baselines trained at 0.5 km when evaluated at 0.5 km.
- Power spectral density analysis shows that QCGS most closely matches the radar spectrum across all spatial scales, while operational products lose variance at high wavenumbers.

## Highlights & Insights
- The observation of the equivalence between classical meteorological interpolation and Gaussian Splatting is elegant — traditional Gaussian-weighted interpolation is a fixed isotropic special case of GS, and this connection naturally transfers techniques from the 3DGS community to the meteorological domain.
- The selective rendering design is architecturally clean: Gaussian kernels are placed only within precipitation regions, avoiding futile computation over the predominantly precipitation-free domain, achieving both efficiency and accuracy.
- The AWS anchoring strategy is simple yet effective — directly setting the amplitude to the observed value at station locations acts as a hard constraint, ensuring the generated field is exactly accurate at all known measurement points.

## Limitations & Future Work
- The method depends on AWS station data — applicability is limited in regions with sparse station networks (e.g., Africa, open ocean); future work may explore degraded satellite-only operating modes.
- Experiments are limited to the Korean Peninsula domain (480×480 grid); scaling to global coverage remains an open challenge.
- The proposal network and Gaussian rendering module are trained in two separate stages; end-to-end joint training may yield further improvements.
- The work addresses only precipitation field *reconstruction* and does not tackle temporal forecasting; combining this framework with temporal extrapolation could form a complete radar-free precipitation nowcasting system.

## Related Work & Insights
- **vs. Sat2Radar (NPM)**: NPM is purely satellite-driven and produces fixed-resolution outputs; QCGS performs multi-source fusion with resolution flexibility, reducing RMSE by 37%.
- **vs. Classical Interpolation (Kriging)**: Kriging uses fixed isotropic kernels; QCGS learns adaptive anisotropic kernels, improving CSI from 0.50 to 0.76.
- **vs. 2D GS Image Methods (GaussianImage)**: Image-based GS optimizes per image and cannot generalize; QCGS achieves cross-scene generalization via a conditioned INR.
- The paradigm of transferring emerging CV techniques (GS/INR) to scientific domains warrants broader attention; analogous methods could be applied to other geophysical variables such as temperature fields and wind fields.

## Rating
- Novelty: ⭐⭐⭐⭐ — First application of 2D GS to precipitation field generation; the classical interpolation–GS equivalence observation is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Cross-scale comparisons are comprehensive (snapshot / hourly / daily); ablations are rigorous and include power spectral analysis.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly developed, mathematical notation is consistent, and figures are of high quality.
- Value: ⭐⭐⭐⭐ — Opens a new paradigm for radar-free precipitation monitoring, though the regional scope limits immediate broader impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting](augmented_radiance_field_a_general_framework_for_enhanced_gaussian_splatting.md)
- [\[ICLR 2026\] LiTo: Surface Light Field Tokenization](lito_surface_light_field_tokenization.md)
- [\[ICLR 2026\] Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting](stylos_multi-view_3d_stylization_with_single-forward_gaussian_splatting.md)
- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](../../CVPR2026/3d_vision/text-image_conditioned_3d_generation.md)
- [\[AAAI 2026\] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field](../../AAAI2026/3d_vision/physics-informed_deformable_gaussian_splatting_towards_unified_constitutive_laws.md)

</div>

<!-- RELATED:END -->
