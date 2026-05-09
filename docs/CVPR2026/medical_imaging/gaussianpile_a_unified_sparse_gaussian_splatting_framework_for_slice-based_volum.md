---
title: >-
  [Paper Note] GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction
description: >-
  [CVPR 2026][Medical Imaging][3D Gaussian Splatting] This paper proposes GaussianPile, which extends 3D Gaussian Splatting from surface appearance modeling to slice-based volumetric reconstruction by introducing a focus-aware physical imaging model (Focus Gaussian). On ultrasound and light-sheet microscopy data, the method achieves high-quality volumetric compression and reconstruction that is 11× faster than NeRF-based methods and reduces storage by 16× compared to voxel grids.
tags:
  - CVPR 2026
  - Medical Imaging
  - 3D Gaussian Splatting
  - volumetric data compression
  - slice-based imaging
  - focus-aware modeling
  - real-time rendering
date: 2026-05-08
content_hash: a301a1910f515a65
---

# GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.20611](https://arxiv.org/abs/2603.20611)
**Code**: N/A
**Area**: Medical Imaging / 3D Reconstruction
**Keywords**: 3D Gaussian Splatting, volumetric data compression, slice-based imaging, focus-aware modeling, real-time rendering

## TL;DR
This paper proposes GaussianPile, which extends 3D Gaussian Splatting from surface appearance modeling to slice-based volumetric reconstruction by introducing a focus-aware physical imaging model (Focus Gaussian). On ultrasound and light-sheet microscopy data, the method achieves high-quality volumetric compression and reconstruction that is 11× faster than NeRF-based methods and reduces storage by 16× compared to voxel grids.

## Background & Motivation
1. **State of the Field**: Modern biomedical imaging modalities (3D microscopy, volumetric ultrasound, etc.) generate data at an exponential rate, yet the cost of storing, transmitting, and analyzing such data has become a critical bottleneck. Implicit neural representations (INR) can achieve high compression ratios but require hours of training and inference, and tend to lose high-frequency details. 3DGS offers efficient fitting and real-time rendering but was designed to model surface appearance from multi-view images.
2. **Limitations of Prior Work**: Standard 3DGS discards internal volumetric information and, when applied directly to volumetric data, produces "ghost" artifacts—2D projections appear plausible while the internal 3D structure remains inconsistent. Prior work has adapted the rendering equation for integral-projection modalities (e.g., X-ray) and explored dense-slice modalities (e.g., MRI, theoretically zero thickness), but no suitable rendering model exists for **finite-thickness slice** modalities (e.g., ultrasound and light-sheet microscopy), where the imaging system has a limited axial depth of focus.
3. **Root Cause**: The standard 3DGS rendering pipeline assumes an "all-in-focus" model in which all Gaussians contribute to the image regardless of depth. This is physically inconsistent with slice-based imaging, where each slice should only capture signal from within a limited depth range around the focal plane. Without this physical constraint, Gaussian primitives grow unconstrained in the axial direction, introducing ghost artifacts across slices.
4. **Paper Goals**: How to design an appropriate Gaussian splatting rendering model for finite-thickness slice imaging systems, such that a single set of Gaussians can accurately reconstruct 2D slices while maintaining 3D volumetric consistency.
5. **Starting Point**: Explicitly incorporating the imaging system's point spread function (PSF) into the rendering process of Gaussian primitives, and modeling finite depth of focus via axial reparameterization of the covariance matrix.
6. **Core Idea**: Convolving the imaging system's axial sensitivity function into 3D Gaussian primitives to obtain "Focus Gaussians," whose covariances naturally encode depth-of-focus information, thereby jointly ensuring 2D slice fidelity and 3D volumetric consistency.

## Method

### Overall Architecture
The input is a set of 2D slice images (e.g., ultrasound or microscopy scan sequences), and the output is a collection of Focus Gaussian primitives that support both 2D slice rendering and 3D voxelization. The rendering pipeline consists of three stages: (1) **Scan**—project Gaussians onto slices at different depths; (2) **axial reparameterization and opacity modulation**—attenuate out-of-focus contributions according to the focus model; (3) **screen-space projection and additive rasterization**—compute 2D marginal distributions and accumulate them to form the final image.

### Key Designs

1. **Focus-aware Physical Modeling**

    - **Function**: Embed the finite depth-of-focus characteristics of the imaging system into the Gaussian rendering process.
    - **Mechanism**: The imaging system's PSF is modeled as an anisotropic Gaussian $\text{psf}(\mathbf{x}_c) \propto \exp(-\frac{1}{2}(\frac{x_c^2}{\sigma_x^2}+\frac{y_c^2}{\sigma_y^2}+\frac{z_c^2}{\sigma_z^2}))$, where $\sigma_z$ directly reflects the axial focusing capability. A sensitivity map $h(-z_c) = \exp(-z_c^2 / (2\sigma_z^2))$ is defined, and the rendered intensity of a slice is the convolution of the Gaussian primitives with this sensitivity map. This design unifies three imaging modalities: $\sigma_z \to \infty$ reduces to the all-in-focus model (X-ray / standard 3DGS), $\sigma_z \to 0$ reduces to the zero-thickness model (MRI), and finite $\sigma_z$ corresponds to ultrasound and light-sheet microscopy. In practice, $\sigma_z \approx \delta_z$ (the scan step size) is used as the default, based on the Nyquist sampling criterion.
    - **Design Motivation**: The all-in-focus assumption of standard 3DGS allows primitives to fit 2D slices well at their positions while maintaining an entirely different internal 3D structure—since no physical constraint is imposed—leading to volumetric inconsistency.

2. **Three-step Rendering of Focus Gaussians**

    - **Function**: Render slice images efficiently and in a physically correct manner.
    - **Mechanism**:
        - **Axial reparameterization**: Depth-of-focus information is injected into the covariance matrix as $\Sigma_e^{-1} = \Sigma_c^{-1} + \mathbf{e}_3 \mathbf{e}_3^\top / \sigma_z^2$, with the updated mean $\mu_e = \Sigma_e \Sigma_c^{-1} \mu_c$. This shrinks the axial support without disturbing the lateral structure.
        - **Opacity modulation**: $\text{opacity}_r = \exp(-\frac{1}{2}(\mu_c^\top \Sigma_c^{-1} \mu_c - \mu_e^\top \Sigma_e^{-1} \mu_e))$, which more strongly attenuates primitives that lie far from the focal plane. The physical intuition is that out-of-focus Gaussians become more transparent.
        - **Additive rasterization**: Unlike the alpha-blending used in surface rendering, pixel intensities in slice imaging are the linear superposition of Gaussian contributions $I(p) = \sum_{i} \tilde{\alpha}_i \exp(-\frac{1}{2} \mathbf{d}_i^\top \Sigma_{2d,i}^{-1} \mathbf{d}_i)$, reflecting the physical nature of volumetric projection.
    - **Design Motivation**: The covariance of a Focus Gaussian directly encodes the depth of focus, and gradient information depends on this representation—when the same Gaussian is observed by multiple slices, it shares identical covariance parameters, naturally enforcing 3D consistency. This is also more suitable for volumetric data than spherical harmonics (SH): view-dependent color is unnecessary, and removing SH parameters alone saves approximately 40% of storage.

3. **Quantization and Compression Scheme**

    - **Function**: Compress the Focus Gaussian representation to a 16× ratio or beyond.
    - **Mechanism**: All Gaussians are ordered by Morton Z-order to exploit spatial locality. Positions are normalized to the scene bounding box and quantized to 14 bits per axis; opacity is mapped to 12-bit integers; scales are log-transformed before 12-bit quantization to preserve multiplicative precision; quaternions are constrained to the positive hemisphere and quantized to 12 bits per component. Each attribute stream is then differentially encoded and compressed with LZMA entropy coding.
    - **Design Motivation**: The parameters of Gaussian primitives are highly structured and spatially correlated, making them well-suited to this pipeline of spatial ordering → differencing → entropy coding.

### Loss & Training
The loss function is $\mathcal{L} = \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$ with $\lambda = 0.2$. At each iteration, one slice (virtual camera) is selected at random. The Adam optimizer is used for 30K steps, starting from 1,000K Gaussians. Adaptive densification and pruning are applied between steps 500 and 25,000. Full backpropagation is implemented in CUDA. An optional differentiable voxelizer is used to evaluate 3D reconstruction quality.

## Key Experimental Results

### Main Results (2D Reconstruction Quality)

| Method | ABUS PSNR↑ | rDL-LSM PSNR↑ | TNNI1 PSNR↑ | Tribolium PSNR↑ | Typical Training Time |
|--------|-----------|--------------|------------|----------------|----------------------|
| HEVC | 29.67 | 29.34 | 35.76 | 35.51 | ~instant |
| INIF (INR) | 24.84 | 17.54 | 30.89 | 31.55 | 27m–1h24m |
| NeurComp (INR) | 19.85 | 30.32 | 39.25 | 32.59 | 11m–2h41m |
| 3DGS (vanilla) | 27.41 | 28.63 | 38.17 | 27.81 | 10m–1h |
| **Ours (10K iter)** | **32.25** | **33.78** | **40.87** | **35.32** | **2–4m** |
| **Ours (30K iter)** | **33.07** | **34.57** | **42.08** | **36.14** | **5–13m** |

### Ablation Study (Effect of $\sigma_z$ on Tribolium Data)

| $\sigma_z$ Setting | PSNR↑ | SSIM↑ | Training Time | # Gaussians |
|--------------------|--------|--------|---------------|-------------|
| $\delta_z/10$ | 34.25 | 0.927 | 8m9s | 215k |
| $\delta_z/2$ | 36.44 | 0.942 | 8m18s | 214k |
| **$\delta_z$ (default)** | **36.67** | **0.944** | **7m23s** | **211k** |
| $2\delta_z$ | 36.12 | 0.940 | 8m8s | 162k |
| $10\delta_z$ (near all-in-focus) | 23.05 | 0.858 | 9m17s | 66k |

### Key Findings
- **The "good 2D, bad 3D" phenomenon of standard 3DGS**: Standard 3DGS achieves a 2D PSNR of 27.41 on ABUS but only 28.49 in 3D; GaussianPile achieves 33.07 and 33.22, respectively—highly consistent 2D and 3D metrics, confirming that the Focus Gaussian physical model enforces volumetric consistency.
- **$\sigma_z = \delta_z$ is the optimal choice**: This is consistent with the theoretical expectation from the Nyquist sampling criterion. Setting $\sigma_z = 10\delta_z$ (approaching all-in-focus) causes a 13.6 dB drop in PSNR, directly validating the necessity of depth-of-focus modeling.
- **Substantial speed advantage**: The 30K-iteration run completes in an average of 8 minutes, approximately 5–11× faster than INR-based methods, while achieving higher quality.
- **Strong compression**: Post-quantization compression ratios of 16–26× are achieved, far surpassing standard 3DGS (0.1–3×, which can inflate beyond the original data size). The compression ratio is comparable to INR methods (15–16×), while quality and speed are both superior.
- Random initialization outperforms grid initialization—the densification strategy automatically guides Gaussians toward semantically meaningful regions.

## Highlights & Insights
- **Elegant integration of physical modeling**: The PSF convolution is decomposed into an additive correction to the covariance matrix, preserving the differentiability of the entire pipeline and the closed form of the Gaussian function. A single scalar parameter $\sigma_z$ unifies three imaging modalities (all-in-focus / zero-thickness / finite depth-of-focus), yielding a remarkably concise parameterization.
- **Insight from removing spherical harmonics**: The observation that view-dependent color is unnecessary in volumetric rendering may appear straightforward, yet it is consequential—it not only reduces parameter count by approximately 40% but also improves geometric fidelity by eliminating the risk of SH coefficients overfitting 2D projections at the expense of 3D consistency.
- **Additive rasterization vs. alpha-blending**: Choosing linear accumulation over an occlusion model, in accordance with the physical characteristics of volumetric imaging, is a simple but critical decision that ensures gradients correctly reflect volumetric contributions.

## Limitations & Future Work
- The current formulation assumes a spatially invariant PSF, whereas real optical systems often exhibit spatially varying aberrations. The authors identify learnable spatially varying PSFs as a promising direction.
- The method may over-smooth severely noisy or undersampled data; incorporating semantic or physical priors could help.
- Support for 4D spatiotemporal data (e.g., live-cell imaging) is absent.
- The method requires per-scene optimization and lacks a feed-forward model; large-scale pretraining with single-pass inference is an important future direction.
- Validation is limited to grayscale volumetric data; multi-channel scenarios (e.g., multi-label fluorescence imaging) are not addressed.

## Related Work & Insights
- **vs. standard 3DGS**: Standard 3DGS lacks physical modeling of slice-based imaging, resulting in the "good 2D, bad 3D" failure mode in volumetric reconstruction. GaussianPile addresses this fundamental issue via Focus Gaussians.
- **vs. INR methods (INIF / NeurComp / CoordNet)**: INR methods require hours of training and tend to lose high-frequency detail. GaussianPile completes training in minutes while preserving finer detail (PSNR improvement of 8–13 dB on ABUS).
- **vs. HEVC**: HEVC is fast but introduces significant quality degradation and does not support interactive 3D visualization. GaussianPile achieves higher quality while enabling real-time 2D rendering and 3D voxelization.
- **vs. Radiative Gaussian Splatting (X-ray)**: The latter handles integral projection ($\sigma_z \to \infty$), while GaussianPile handles finite depth of focus (finite $\sigma_z$)—the two approaches are complementary, together covering distinct imaging modalities.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The physical model design of Focus Gaussians is highly elegant, unifies three imaging modalities, and fills an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple datasets (ultrasound + microscopy), 2D/3D metrics, compression ratios, speed, and ablations are all covered, though comparisons with a broader set of dedicated compression methods are lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The physical model derivation is clear and rigorous; the figures are well-designed, particularly Figure 1's three-modality comparison.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a fundamental limitation of 3DGS in the volumetric domain, with direct practical relevance to medical image compression and visualization.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation](../../ICCV2025/medical_imaging/insideout_integrated_rgb-radiative_gaussian_splatting_for_comprehensive_3d_objec.md)
- [\[CVPR 2026\] MozzaVID: Mozzarella Volumetric Image Dataset](mozzavid_mozzarella_volumetric_image_dataset.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[AAAI 2026\] PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](../../AAAI2026/medical_imaging/pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)

<!-- RELATED:END -->
