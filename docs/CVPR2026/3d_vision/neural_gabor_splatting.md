---
title: >-
  [Paper Note] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction
description: >-
  [CVPR 2026][3D Vision][Gaussian Splatting] Neural Gabor Splatting embeds a lightweight MLP (SIREN architecture) into each Gaussian primitive…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Gaussian Splatting"
  - "High-frequency Surface Reconstruction"
  - "Neural Texture"
  - "MLP Primitive"
  - "Frequency-aware Densification"
date: 2026-05-08
content_hash: 76fcf8952bf3cc27
---

# Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2604.15941](https://arxiv.org/abs/2604.15941)  
**Code**: [https://github.com/haato-w/neural-gabor-splatting](https://github.com/haato-w/neural-gabor-splatting)  
**Area**: 3D Vision
**Keywords**: Gaussian Splatting, High-frequency Surface Reconstruction, Neural Texture, MLP Primitive, Frequency-aware Densification

## TL;DR

Neural Gabor Splatting embeds a lightweight MLP (SIREN architecture) into each Gaussian primitive, enabling a single primitive to represent complex spatially-varying color patterns. Combined with a frequency-aware densification strategy, this approach significantly improves high-frequency surface reconstruction quality under the same data budget.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the dominant method for novel view synthesis, owing to its explicit point-cloud representation that enables fast training, real-time rendering, and convenient editing. However, typical scenes require hundreds of thousands to millions of Gaussian primitives, incurring substantial memory overhead.

**Limitations of Prior Work**: Each Gaussian primitive can only represent a single color given a viewing direction. When scenes contain high-frequency details (e.g., checkerboard textures, hair, or regions with frequent color transitions), a large number of primitives is required to cover each color variation, causing the primitive count to grow dramatically.

**Key Challenge**: The limited expressive capacity of individual primitives is the fundamental source of storage overhead. Existing improvements each have their own constraints: 3D Gabor Splatting is restricted by the properties of the Gabor noise function, while textured Gaussians are limited by a preset texture resolution.

**Goal**: Enhance the expressive capacity of individual primitives so that fewer primitives suffice for high-quality high-frequency surface reconstruction.

**Key Insight**: Inspired by neural texture and deferred rendering, the paper parameterizes the intra-primitive color variation with a small MLP, enabling a single primitive to represent arbitrarily complex local patterns.

**Core Idea**: Each 2D Gaussian primitive is equipped with an independent lightweight SIREN MLP that takes local coordinates and viewing direction as input and outputs RGB color. The sinusoidal activations of SIREN naturally encode high-frequency signals without requiring additional positional encoding.

## Method

### Overall Architecture

Built upon 2D Gaussian Splatting (2DGS), each primitive maps 3D spatial points to local 2D coordinates $(u, v)$ via an affine transformation. Instead of spherical harmonic coefficients for color, $(u, v)$ and the viewing direction $\vec{d}$ are fed into a per-primitive MLP to obtain RGB values. The final pixel color is computed via alpha blending: $\mathbf{c} = \sum_k \hat{\boldsymbol{c}}_k(\Theta_k, u, v, \vec{d}) \alpha_k \hat{G}_k T_k$.

### Key Designs

1. **Neural Gabor Primitive (Per-Primitive MLP)**:

    - **Function**: Allows a single primitive to represent spatially-varying, view-dependent color.
    - **Mechanism**: Each primitive has an independent single-hidden-layer SIREN MLP with 6 hidden neurons. The input is a 5-dimensional vector $\mathbf{y} = (u, v, \vec{d})$, and the color prediction is $\hat{\boldsymbol{c}}_k = \text{Sigmoid}[\bar{\mathbf{W}}_k \sin\{\omega_0(\mathbf{W}_k \mathbf{y} + \boldsymbol{b}_k)\} + \bar{\boldsymbol{b}}_k]$, with frequency parameter $\omega_0 = 30$. The sinusoidal activations of SIREN implicitly perform positional encoding, enabling the network to represent high-frequency signals.
    - **Design Motivation**: Compared to discrete texels, MLPs provide a continuous, resolution-agnostic representation. Compared to fixed basis functions (spherical harmonics, Gabor), MLPs can learn arbitrarily complex color patterns. Independent per-primitive parameters enable fine-grained modeling.

2. **Frequency-aware Densification**:

    - **Function**: Controls primitive count growth and prioritizes primitive allocation in regions with missing high-frequency content.
    - **Mechanism**: Rather than gradient-based densification, rendering errors are computed in the frequency domain. FFT is applied to both the rendered image and the ground truth; components within specific frequency bands (0.01–0.10, 0.10–0.20, 0.20–0.40) are extracted, IFFT is applied, and the results are locally averaged to produce a frequency-domain error map. Per-pixel errors are back-projected to primitive space, and primitives with high error are selected for cloning or splitting.
    - **Design Motivation**: Gradient-based densification causes over-densification with neural Gabor primitives (because color variations learned by the MLP produce large gradients). Frequency-domain errors enable targeted primitive allocation in regions lacking high-frequency information.

3. **Progressive Opacity Reset**:

    - **Function**: Stabilizes primitive management during densification.
    - **Mechanism**: Replaces the hard opacity reset of vanilla 3DGS with a progressive reset; cloned/split primitives inherit the parent's MLP weights with an opacity correction applied.
    - **Design Motivation**: Hard resets can abruptly invalidate MLP parameters; progressive resets preserve training stability.

### Loss & Training

Standard $\lambda L_1 + (1-\lambda) L_{SSIM}$ loss. MLP weights are initialized following the SIREN initialization scheme. Every 100 iterations, 20 training views are randomly sampled as a GPU batch for accumulated error. Densification threshold: 0.01. Total training: 20k iterations.

## Key Experimental Results

### Main Results

| Method | High-Frequency PSNR/SSIM/LPIPS | Mip-NeRF360 PSNR/SSIM/LPIPS |
|--------|-------------------------------|------------------------------|
| 3DGS* | 23.97/0.8335/0.2769 | 27.23/0.8005/0.2931 |
| 2DGS* | 23.91/0.8279/0.2855 | 26.47/0.7804/0.3197 |
| NEST | 22.22/0.8588/0.2220 | - |
| NTS | 23.48/0.8139/0.3026 | 29.49/0.9028/0.2544 |
| **Ours** | **26.49/0.8808/0.2115** | 26.98/0.810/0.2521 |

### Ablation Study

| Densification Strategy | High-Frequency PSNR/SSIM/LPIPS |
|------------------------|-------------------------------|
| Frequency-aware (Ours) | 25.72/0.8619/0.2352 |
| Error-driven | 25.95/0.8619/0.2376 |
| Gradient-driven | 25.56/0.8534/0.2464 |

### Key Findings

- PSNR improves by 2.5+ dB on the High-Frequency dataset (vs. 2DGS), demonstrating the substantial advantage of neural Gabor primitives in high-frequency scenes.
- Under the same data budget, neural Gabor primitives yield significantly sharper visual quality; fine details such as hair and checkerboard patterns are far superior to standard methods.
- Frequency-aware densification achieves accuracy comparable to error-driven densification while providing band-level controllability.
- Advantages are more pronounced under low-budget settings (1%–5% data); NEST and NTS degrade rapidly under strict budgets.
- Training time is approximately 2× that of 2DGS, but comparable to other neural splatting methods such as NEST and NTS.

## Highlights & Insights

- **Minimalist MLP Design**: A single-hidden-layer SIREN with only 6 neurons has an extremely small parameter count yet achieves powerful high-frequency expressiveness through sinusoidal activations. This demonstrates the effectiveness of combining "micro-networks with the right activation function."
- **Controllability of Frequency-aware Densification**: The approach enables precise selection of which frequency band receives additional primitives, providing a fine-grained quality–capacity trade-off tool for storage-constrained scenarios.
- **Continuous vs. Discrete Representation**: Compared to texture map approaches, MLPs are inherently resolution-agnostic and free from texture aliasing artifacts.

## Limitations & Future Work

- The `atomicAdd` operations from per-primitive independent MLPs increase training time (approximately 2×).
- The method does not directly apply to volumetric phenomena (e.g., fog, smoke), and extension to dynamic scenes is non-trivial.
- For low-frequency scenes, the expressive capacity of the MLP may not be fully utilized, resulting in wasted parameters.
- Future directions include parameter sharing or codebook-based compression to further reduce storage.

## Related Work & Insights

- **vs. 3D Gabor Splatting**: 3D Gabor is constrained by the fixed form of the Gabor noise function; neural Gabor offers greater flexibility through MLP-based representation.
- **vs. NTS/NEST**: These methods rely on hash grids or tri-plane encodings and have limited expressive capacity under low budgets; neural Gabor is more robust in low-budget settings.
- **vs. Textured Gaussians**: Texture-based approaches are limited by preset resolution and exhibit view-direction dependency; MLPs are continuous and resolution-agnostic.

## Rating

- Novelty: ⭐⭐⭐⭐ The per-primitive MLP concept is intuitive and effective; the frequency-aware densification design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset comparisons are comprehensive; budget analysis and ablation studies are detailed.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; mathematical formulations are complete.
- Value: ⭐⭐⭐⭐ Provides a practical solution for high-frequency scene reconstruction under storage constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[CVPR 2026\] Neu-PiG: Neural Preconditioned Grids for Fast Dynamic Surface Reconstruction on Long Sequences](neu-pig_neural_preconditioned_grids_for_fast_dynamic_surface_reconstruction_on_l.md)
- [\[CVPR 2026\] Neural Field-Based 3D Surface Reconstruction of Microstructures from Multi-Detector Signals in Scanning Electron Microscopy](neural_field-based_3d_surface_reconstruction_of_microstructures_from_multi-detec.md)
- [\[CVPR 2026\] FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting](fact-gs_frequency-aligned_complexity-aware_texture_reparameterization_for_2d_gau.md)

</div>

<!-- RELATED:END -->
