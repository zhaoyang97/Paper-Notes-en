---
title: >-
  [Paper Note] A Physics-Informed Blur Learning Framework for Imaging Systems
description: >-
  [CVPR 2025][Image Restoration][PSF estimation] A physics-informed PSF learning framework is proposed, designing a new wavefront basis (where each basis only affects a single SFR direction) to eliminate gradient conflicts. Combined with curriculum learning (from center to periphery), it accurately estimates the spatially-varying PSF of imaging systems without requiring lens parameters.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "PSF estimation"
  - "wavefront aberration"
  - "curriculum learning"
  - "SFR"
  - "deblurring"
  - "imaging systems"
date: 2026-05-08
content_hash: 123700ec4d1283d9
---

# A Physics-Informed Blur Learning Framework for Imaging Systems

**Conference**: CVPR 2025  
**arXiv**: [2502.11382](https://arxiv.org/abs/2502.11382)  
**Code**: Publicly available  
**Area**: Image Restoration / Computational Optics  
**Keywords**: PSF estimation, wavefront aberration, curriculum learning, SFR, deblurring, imaging systems

## TL;DR
A physics-informed PSF learning framework is proposed, designing a new wavefront basis (where each basis only affects a single SFR direction) to eliminate gradient conflicts. Combined with curriculum learning (from center to periphery), it accurately estimates the spatially-varying PSF of imaging systems without requiring lens parameters.

## Background & Motivation

**Background**: Spatially-varying aberrations in imaging systems severely degrade image quality. Accurately characterizing the point spread function (PSF) is crucial for digital photography, industrial inspection, autonomous driving, astronomical observation, etc.

**Limitations of Prior Work**:
   - **Non-parametric models** (e.g., Degradation Transfer): Sparse independent sampling, unable to capture high-dimensional features of the PSF, and do not guarantee spatial smoothness.
   - **Simple parametric models** (e.g., heteroscedastic Gaussian, Fast Two-step): Overly simplified, and typical lens aberrations do not necessarily conform to Gaussian kernels.
   - **Optical simulation models**: Require detailed lens design parameters, which are restricted by intellectual property.
   - **Seidel PSF models**: Spherical aberration bases like $\rho^2$ simultaneously affect multiple SFR directions, leading to gradient conflicts during optimization.

**Key Challenge**: Existing Seidel bases couple spherical aberration and other terms with multi-directional SFRs. When the actual SFRs in the 0° and 90° directions differ, optimizing a single coefficient leads to gradient conflicts, similar to iterative interference in multi-task learning.

**Key Insight**: Decompose the Seidel bases into a new set of wavefront bases such that each basis affects only a single SFR direction, combined with curriculum learning to optimize step-by-step from the center to the periphery.

**Core Idea**: Decoupled bases + Curriculum learning + MLP proxy model = High-precision PSF estimation without lens parameters.

## Method

### Relationship between PSF and SFR
The PSF is defined by wavefront aberration:
$$\text{PSF}(H, \lambda) = |\mathcal{F}(A(\mathbf{p}) \exp(i \frac{2\pi W(H, \lambda, \mathbf{p})}{\lambda}))|^2$$
where $W$ is the wavefront aberration, and $\mathbf{p} = (\rho, \theta)$ is the pupil plane coordinate. SFR is a directional slice of the PSF.

### Key Designs

1. **New Wavefront Bases (Decoupled Bases)**

    - **Function**: Decompose the Seidel bases so that each basis function contains only a $\cos\theta$ or $\sin\theta$ component.
    - **Mechanism**: Define the index set $\mathcal{Q} = \{(2,2,0), (2,0,2), (3,1,0), (3,3,0), (4,2,0), (4,0,2), (5,1,0), (6,2,0), (6,0,2)\}$.
    - **Effect**: Each basis independently affects the horizontal or vertical SFR, eliminating gradient conflicts.
    - **Design Motivation**: Analogous to gradient conflict mitigation strategies in multi-task learning.

2. **Curriculum Learning Optimization Strategy**

    - **Function**: Gradually learn the PSF from the center to the edge of the image.
    - **Physical Basis**: According to aberration theory, the center is only affected by spherical aberration, while coma and field curvature progressively appear closer to the edges.
    - **Implementation**: Restrict the optimization of $H$ to a narrow field of view (FoV) at each step, with $H$ gradually increasing from 0 to 1.

3. **Two-Stage PSF Estimation**

    - **Stage 1: Monochromatic PSF Estimation**
        - MLP $\mathcal{G}_{\Theta_1}$ takes the normalized image height $H$ as input and outputs wavefront coefficients.
        - Generate $\text{SFR}^*$ through physical transformations and optimize by comparing with the measured SFR.
        - Loss: $\Theta_1^* = \arg\min_{\Theta_1} \sum_H \sum_\phi |\text{SFR}^*(H,\phi) - \text{SFR}(H,\phi)|$
    - **Stage 2: Cross-Channel PSF Shift Estimation**
        - MLP $\mathcal{G}_{\Theta_2}$ estimates the PSF translation of the red/blue channels relative to the green channel.
        - Use chromatic aberration area difference $\Delta\text{CA}$ as the optimization target.
        - Separating monochromatic aberration and chromatic aberration simplifies optimization.

4. **Calibration Process**

    - Capture checkerboard patterns under a controlled environment.
    - Multi-frame averaging for noise reduction in RAW format.
    - Convert to linear RGB format (to avoid the non-linear effects of the ISP).

### Loss & Training
- Use the estimated PSF to synthesize blurry image pairs to train deblurring networks (MPRNet/Restormer/FFTFormer).
- Training set: 500 images from Flickr2K, Test set: 100 images.
- Synthesize test blurry images using ground-truth PSFs.

## Key Experimental Results

### PSF Accuracy (PSNR/SSIM, Lens #63762)

| Method | H=0 Noise-free | H=0.7 Noise-free | H=1 Noise-free |
|------|-----------|-------------|-----------|
| Degradation Transfer | 41.98 / 0.937 | — | — |
| Fast Two-step | 42.24 / 0.943 | — | — |
| **Ours** | **42.08 / 0.945** | **49.19 / 0.968** | **50.16 / 0.983** |

### Deblurring Performance (PSNR/SSIM, Noise-free)

| Deblurring Method | Degradation Transfer | Fast Two-step | **Ours** |
|-----------|---------------------|---------------|----------|
| MPRNet | 30.55 / 0.873 | 30.22 / 0.870 | **31.24 / 0.894** |
| Restormer | 30.69 / 0.871 | 30.34 / 0.869 | **31.51 / 0.894** |
| FFTFormer | 30.53 / 0.872 | 30.34 / 0.868 | **31.36 / 0.891** |

### Ablation Study (PSNR/SSIM)

| Configuration | H=0 | H=0.7 | H=1 |
|------|------|-------|------|
| w/o Narrow FoV Optimization | 42.51/0.934 | 42.68/0.937 | 41.06/0.922 |
| w/o New Wavefront Bases | 42.18/0.931 | 47.48/0.950 | 44.58/0.954 |
| w/o Curriculum Learning | 41.56/0.937 | 48.58/0.957 | 46.02/0.955 |
| **Full Method** | **42.64/0.940** | **49.08/0.968** | **49.25/0.981** |

### Key Findings
- The three components yield the largest improvement for the marginal field of view (H=1): new wavefront bases (+4.67 dB), curriculum learning (+3.23 dB).
- Narrow FoV optimization is crucial for extreme locations.
- Real-world shoot validation: Both MUSIQ and MANIQA metrics outperform baseline methods.

## Highlights & Insights
- **No reliance on lens parameters**: Can be directly applied to mass-produced imaging devices.
- **Wavefront basis decoupling** is elegantly designed, analogizing to gradient conflict mitigation in multi-task learning.
- Two-stage separation of monochromatic and chromatic aberrations, reducing optimization difficulty.
- Validated effectiveness on an actual imaging system (Edmund lens + onsemi sensor).

## Limitations & Future Work
- Chromatic aberration correction in wide fields of view is still imperfect.
- Currently only handles blur caused by PSF, without considering motion blur.
- Requires controlled-environment calibration (using a checkerboard).

## Rating
- Novelty: ⭐⭐⭐⭐ Wavefront basis decoupling + curriculum learning is a unique combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Simulation + real-world shooting + ablation + multiple deblurring algorithms.
- Writing Quality: ⭐⭐⭐⭐⭐ Physical derivations are clear and rigorous.
- Value: ⭐⭐⭐⭐ Practical significance for improving the image quality of mass-produced devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PolarFree: Polarization-based Reflection-Free Imaging](polarfree_polarization-based_reflection-free_imaging.md)
- [\[ICML 2026\] Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](../../ICML2026/image_restoration/phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)
- [\[ICLR 2026\] Denoising Neural Reranker for Recommender Systems](../../ICLR2026/image_restoration/denoising_neural_reranker_for_recommender_systems.md)
- [\[ICCV 2025\] Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention](../../ICCV2025/image_restoration/consistent_time-of-flight_depth_denoising_via_graph-informed_geometric_attention.md)
- [\[CVPR 2025\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging](proximal_algorithm_unrolling_flexible_and_efficient_reconstruction_networks_for_.md)

</div>

<!-- RELATED:END -->
