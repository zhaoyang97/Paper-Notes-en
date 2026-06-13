---
title: >-
  [Paper Note] Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography
description: >-
  [ICCV 2025][Medical Imaging][Photoacoustic Computed Tomography] This paper proposes an efficient self-supervised joint reconstruction method that parameterizes the speed of sound (SOS) as either a pixel grid or a neural…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "Photoacoustic Computed Tomography"
  - "Speed of Sound Recovery"
  - "Neural Fields"
  - "Self-supervised Learning"
  - "Wavefront Aberration Correction"
date: 2026-05-08
content_hash: b636d46c44d6d9b9
---

# Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography

**Conference**: ICCV 2025
**arXiv**: [2409.10876](https://arxiv.org/abs/2409.10876)  
**Code**: [https://lukeli0425.github.io/Coord-SoS-PACT/](https://lukeli0425.github.io/Coord-SoS-PACT/)  
**Area**: Medical Imaging / Computational Imaging
**Keywords**: Photoacoustic Computed Tomography, Speed of Sound Recovery, Neural Fields, Self-supervised Learning, Wavefront Aberration Correction

## TL;DR

This paper proposes an efficient self-supervised joint reconstruction method that parameterizes the speed of sound (SOS) as either a pixel grid or a neural field, recovering SOS and high-quality photoacoustic images by backpropagating gradients through a differentiable imaging forward model. The method surpasses the current state of the art in accuracy while achieving a 35× speedup (40 seconds vs. 23 minutes).

## Background & Motivation

Photoacoustic computed tomography (PACT) is a non-invasive imaging modality analogous to ultrasound, with great promise for medical applications. Its principle is as follows: biological tissue absorbs laser energy and emits ultrasonic waves, which are captured by an external transducer array and used to reconstruct the initial pressure (IP) distribution.

However, conventional reconstruction algorithms (e.g., delay-and-sum DAS, filtered backprojection) assume a homogeneous SOS throughout the tissue, ignoring acoustic heterogeneity in practice:

- **Wavefront distortion**: Spatial variations in SOS cause wavefront distortion, introducing timing errors relative to the assumed constant SOS, leading to ringing, streaking, and ghosting artifacts.
- **High cost of direct SOS measurement**: Ultrasound CT (USCT) can directly measure SOS, but requires additional hardware, making it difficult and expensive to implement.
- **Computationally expensive joint reconstruction**: APACT (current SOTA) performs exhaustive search over wavefront coefficients, requiring 23 minutes per sample; moreover, its simplified wavefront model cannot handle global image scaling.
- **Supervised learning is infeasible**: PACT is a relatively new imaging modality, and large-scale real datasets with paired measurements, ground-truth aberrations, and IP/SOS labels are unavailable; domain shift is also a serious concern.

**Key Challenge**: Accurate SOS information is essential for high-quality imaging, yet acquiring SOS is both costly and difficult. This paper addresses the problem by parameterizing SOS using coordinate-based representations (pixel grid / neural field) and optimizing via a differentiable physical model in a self-supervised manner, requiring no external training data.

## Method

### Overall Architecture

1. Reconstruct a stack of DAS images using different delay parameters $d$ from the photoacoustic signals (analogous to a focal stack).
2. Parameterize SOS as a pixel grid (PG) or neural field (NF), and compute the point spread function (PSF) for each image patch via the physical model.
3. Recover sharp image patches via multi-channel deconvolution.
4. Self-supervised loss: enforce consistency between the predicted aberration and the observed aberration, and update SOS parameters via backpropagation.

### Key Designs

1. **Differentiable Imaging Forward Model**:

    - **Function**: Computes wavefront error and PSF based on the physical model, supporting end-to-end backpropagation.
    - **Mechanism**: Assumes a straight-ray acoustic model; the time of flight from sample point $\mathbf{r}'$ to transducer $\mathbf{r}$ is given by the path integral: $t(\mathbf{r}',\mathbf{r},\mathbf{v})=\int_{\mathbf{r}'}^{\mathbf{r}}\frac{1}{v(\mathbf{l})}dl$. The wavefront error is: $w(\theta;\mathbf{r}',\mathbf{v})=\int_{\mathbf{r}'}^{\mathbf{r}(\theta)}(1-\frac{v_0}{v(\mathbf{l})})dl$. The PSF is computed via a Fourier-domain transfer function: $H_i(\mathbf{k};d,\mathbf{v})=\frac{1}{2}(e^{-j|\mathbf{k}|(d-w(\angle\mathbf{k};\mathbf{r}'_i,\mathbf{v}))}+e^{j|\mathbf{k}|(d-w(\angle\mathbf{k}+\pi;\mathbf{r}'_i,\mathbf{v}))})$
    - **Design Motivation**: A fully differentiable physical forward model enables end-to-end gradient-based optimization of SOS, eliminating the need for exhaustive search.

2. **Multi-channel Deconvolution**:

    - **Function**: Exploits the redundancy provided by multiple delay values for robust image recovery.
    - **Mechanism**: For each image patch, different delay values yield different PSFs but correspond to the same sharp image. In the frequency domain: $\mathbf{Y}_i = \mathbf{H}_i(\mathbf{v})X_i$; reconstruction via pseudo-inverse: $\hat{X}_i = \frac{\mathbf{H}_i(\mathbf{v})^\top \mathbf{Y}_i}{\mathbf{H}_i(\mathbf{v})^\top \mathbf{H}_i(\mathbf{v})}$
    - **Design Motivation**: Single-channel deconvolution is severely affected by noise and PSF shape. Multiple channels (default: 16) provide an overdetermined system that substantially reduces ringing artifacts and recovers features lost at patch boundaries. The additional channels are generated digitally, introducing no new measurement noise.

3. **Coordinate-based SOS Representation (PG and NF)**:

    - **Function**: Represents the 2D SOS distribution in a compact parametric form.
    - **Mechanism**:
        - **Pixel Grid (PG)**: An interpolation grid at the same resolution as the IP image, regularized with total variation (TV) to enforce spatial smoothness.
        - **Neural Field (NF)**: A single fully-connected layer with 256 features and SIREN activations (sinusoidal), mapping pixel coordinates to SOS values. With only 1,027 parameters—far fewer than the ~200K pixels—the network provides implicit regularization.
    - **Design Motivation**: The low degree of freedom in NF (1,027 parameters vs. ~200K pixels) reduces the degrees of freedom of the inverse problem, acting as implicit regularization that makes the problem better-posed.

### Loss & Training

Self-supervised aberration matching loss:
$$L(\mathbf{v}_\phi) = \sum_{i=1}^{N} |\mathbf{k}| \|\mathbf{Y}_i - \mathbf{H}_i(\mathbf{v}_\phi)\hat{X}_i(\mathbf{Y}_i, \mathbf{H}_i(\mathbf{v}_\phi))\|_2^2 + \lambda\text{TV}(\mathbf{v}_\phi)$$

- NF: $\lambda=0$ (relies on implicit network regularization), 10 epochs, learning rate $5\times10^{-3}$
- PG: $\lambda=10^{-4}$ (TV regularization), 30 epochs, learning rate $10^{-1}$
- Optimizer: Adam; Hardware: NVIDIA RTX A6000 GPU
- Default number of delay channels: $M=16$

## Key Experimental Results

### Main Results

**Numerical simulation (average over 5 numerical phantoms):**

| Method | IP PSNR ↑ | IP SSIM ↑ | SOS PSNR ↑ | SOS SSIM ↑ | Time |
|--------|-----------|-----------|------------|------------|------|
| Conventional DAS | 21.49 | 0.372 | - | - | 0.13s |
| Dual-SOS DAS | 24.42 | 0.446 | - | - | 0.12s |
| APACT (SOTA) | 21.49 | 0.434 | 17.74 | 0.908 | **23 min** |
| Ours (PG) | 25.05 | 0.514 | 21.26 | 0.903 | 113.3s |
| **Ours (NF)** | **25.08** | **0.519** | **22.29** | **0.931** | **40.3s** |
| Ours (SOS oracle) | 25.61 | 0.537 | - | - | 2.6s |

The NF method outperforms APACT on all metrics while being **35× faster** (40s vs. 23min).

### Ablation Study

**Effect of number of delay channels (NF method):**

| Delay channels M | 4 | 8 | 16 | 24 | 32 |
|-----------------|---|---|----|----|-----|
| Computation time (s) | ~10 | ~20 | 40 | ~60 | ~80 |
| IP quality | Low | Moderate | Good | Slightly better | Marginally better |

Performance gains diminish beyond 16 channels; 16 is selected as the default for the best performance–time tradeoff.

**NF network size ablation**: Networks that are too small (excessively strong implicit regularization) or too large (insufficient regularization, excessive high-frequency capacity) both fail; a single layer with 256 features is optimal.

**TV regularization weight for PG**: $\lambda=10^{-4}$ is optimal; larger $\lambda$ encourages smoother SOS but may cause over-smoothing.

### Key Findings
- **Wavefront error analysis**: APACT models only the 0th- and 2nd-order Fourier components, omitting the 1st-order cosine component (which causes image scaling) and higher-order details. The proposed method accurately models both low- and high-order components.
- **Image scaling issue in APACT**: Due to the neglected 1st-order wavefront component, off-center PSFs cause image patches to shift toward the tissue center; APACT cannot resolve this, whereas the proposed method corrects it naturally through complete wavefront modeling.
- **Stripe artifacts in PG**: PG exhibits stripe-like artifacts in SOS reconstruction, arising from the path-integral nature of the straight-ray acoustic model, though IP image quality remains satisfactory.
- **Generalization to real data**: The method performs well on both a leaf phantom (with three SOS materials) and in vivo mouse liver data, without any modification for real data.

## Highlights & Insights
- **Self-supervision + physical model**: No external training data is required; end-to-end SOS recovery is achieved through physical consistency constraints, which is highly significant for data-scarce medical imaging.
- **Elegant integration of signal processing and neural networks**: The NF serves not as a black-box network but as an efficient, compact representation of SOS within a physical framework, enhancing interpretability and reliability.
- **Insightful focal stack analogy**: DAS image stacks with different delay parameters are analogized to focal stacks, where each delay corresponds to a different "focal depth"; multi-channel joint processing provides redundant information.
- **35× speedup**: This is clinically significant, making SOS recovery practically feasible.

## Limitations & Future Work
- The straight-ray acoustic model is a simplifying assumption; in the presence of strong acoustic heterogeneity, actual acoustic waves will bend.
- The stripe artifact problem in PG representation requires better regularization strategies.
- Validation is limited to 2D circular-array PACT systems; extension to 3D systems and other array configurations remains unexplored.
- The SIREN architecture and network size for NF require manual tuning; adaptive mechanisms may be preferable.
- Large-scale quantitative evaluation is lacking (only 5 numerical phantoms), and real data lacks ground-truth SOS for comparison.

## Related Work & Insights
- **APACT**: An adaptive PACT method that exhaustively searches wavefront coefficients patch by patch; the primary baseline for this work.
- **Dual-SOS DAS**: Assumes a uniform in-body SOS for first-order correction; fast but limited in accuracy.
- **SIREN (Neural Fields)**: Implicit neural representations with sinusoidal activations; used in this work as a compact parameterization of SOS.
- **UniDepth / DiffMorpher**: Similar coordinate-based representations applied to other computational imaging tasks.
- **Coordinate-based Neural Representations for AO**: Use of coordinate-based representations for wavefront estimation in adaptive optics.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing neural fields into SOS recovery for PACT is a novel cross-domain application; the differentiable forward model design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Full coverage of numerical simulations, physical phantoms, and in vivo experiments with thorough ablations, though the number of numerical phantoms is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Physical derivations are rigorous and clear; figures are well-crafted and intuitive; the comparative analysis with prior methods is thorough.
- **Value**: ⭐⭐⭐⭐ The 35× speedup makes SOS recovery clinically viable; the self-supervised framework offers important reference value for data-scarce domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](../../ICLR2026/medical_imaging/dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[ICCV 2025\] COIN: Confidence Score-Guided Distillation for Annotation-Free Cell Segmentation](coin_confidence_score-guided_distillation_for_annotation-free_cell_segmentation.md)
- [\[ICCV 2025\] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation](alleviating_textual_reliance_in_medical_language-guided_segmentation_via_prototy.md)

</div>

<!-- RELATED:END -->
