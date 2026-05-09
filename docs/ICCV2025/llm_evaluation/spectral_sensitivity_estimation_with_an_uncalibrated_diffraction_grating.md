---
title: >-
  [Paper Note] Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating
description: >-
  [ICCV 2025][LLM Evaluation][spectral sensitivity] A practical method is proposed for estimating camera spectral sensitivity using an uncalibrated diffraction grating film. By jointly estimating spectral sensitivity and grating efficiency, accurate closed-form solutions are obtained from a single capture of a light source with known spectrum. The method significantly outperforms traditional color chart approaches at an equipment cost of under $5 USD.
tags:
  - ICCV 2025
  - LLM Evaluation
  - spectral sensitivity
  - diffraction grating
  - camera calibration
  - closed-form solution
  - pixel-wavelength mapping
date: 2026-05-08
content_hash: fabef97b3cb86e85
---

# Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating

**Conference**: ICCV 2025
**arXiv**: [2508.00330](https://arxiv.org/abs/2508.00330)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: spectral sensitivity, diffraction grating, camera calibration, closed-form solution, pixel-wavelength mapping

## TL;DR
A practical method is proposed for estimating camera spectral sensitivity using an uncalibrated diffraction grating film. By jointly estimating spectral sensitivity and grating efficiency, accurate closed-form solutions are obtained from a single capture of a light source with known spectrum. The method significantly outperforms traditional color chart approaches at an equipment cost of under $5 USD.

## Background & Motivation

**Background**: Camera spectral sensitivity characterizes the camera's response to incident light at different wavelengths, and serves as a foundation for computer vision tasks such as color correction, illumination estimation, and material analysis. Accurate spectral sensitivity calibration is essential for color-faithful imaging.

**Limitations of Prior Work**:
   - **Traditional instrument-based methods**: Rely on precision equipment such as narrowband filters or monochromators, which are costly and time-consuming.
   - **Color chart methods**: Use reference targets with known spectral reflectance (e.g., ColorChecker), but natural object spectral reflectance constitutes a low-frequency signal; color patches are highly correlated, limiting wavelength resolution.
   - **Existing diffraction grating methods**: Require additional captures of calibrated reference targets to estimate grating efficiency, necessitating multiple scene changes and light sources, resulting in a complex workflow.
   - **Exif metadata methods**: Rely solely on camera metadata and cannot account for external factors such as lens filters; white balance ambiguity is also present.

**Key Challenge**: A diffraction grating separates light of different wavelengths to distinct spatial positions, theoretically enabling high-wavelength-resolution sensitivity estimation. However, the grating efficiency (non-uniform attenuation across wavelengths) is unknown, and prior methods require additional reference targets to calibrate it.

**Goal**: Can camera spectral sensitivity and grating efficiency be jointly estimated, without prior knowledge of grating efficiency, solely from captures of a light source with known spectrum passed through a diffraction grating?

**Key Insight**: By employing basis function representations, the originally bilinear problem (sensitivity × grating efficiency) is reformulated as a linear system admitting a closed-form solution.

## Method

### Overall Architecture
Images of a known-spectrum light source transmitted through an uncalibrated diffraction grating are captured, simultaneously yielding direct-light and diffracted-light observations. Using direct-light constraints (integral spectral equations) and diffracted-light constraints (wavelength separation equations), the camera spectral sensitivity $\mathbf{s}$ and the inverse grating efficiency $\boldsymbol{\eta}^{-1}$ are jointly solved.

### Key Designs

1. **Basis Function Representation and Linearization**:

    - Spectral sensitivity and the inverse grating efficiency are each expressed as linear combinations of basis functions:
    $\mathbf{s} = \mathbf{B}_s \mathbf{c}_s \in \mathbb{R}_+^f, \quad \boldsymbol{\eta}^{-1} = \mathbf{B}_\eta \mathbf{c}_\eta \in \mathbb{R}_+^f$
    - Sensitivity basis $\mathbf{B}_s$: obtained via SVD on sensitivity data from 44 cameras, with 7 bases per channel.
    - Grating efficiency basis $\mathbf{B}_\eta$: Fourier bases are used (grating efficiency is a low-frequency function), with 7 bases.
    - Wavelength sampling: $f=31$ (400 nm–700 nm, 10 nm interval).

2. **Direct-Light Constraint (Linear Constraint)**:
    $m_{\text{dir}} = \mathbf{e}^{\top}\mathbf{B}_s\mathbf{c}_s$
   The direct-light observation equals the inner product of the incident spectrum and the sensitivity, providing 3 linear equations (RGB channels).

3. **Diffracted-Light Constraint (Homogeneous Linear System)**:
   Mathematical derivation transforms the bilinear relationship into a homogeneous linear system:
    $\begin{bmatrix}\text{diag}(\mathbf{a})\mathbf{B}_\eta & -\mathbf{B}_s\end{bmatrix}\begin{bmatrix}\mathbf{c}_\eta \\ \mathbf{c}_s\end{bmatrix} = \mathbf{0}$
   where $\mathbf{a} = \text{diag}(\mathbf{e}^{-1})\mathbf{W}^{\dagger}\mathbf{m}_{\text{dif}}$ is a known quantity and $\mathbf{W}$ is the weight matrix (pixel-wavelength mapping).

4. **Closed-Form Solution**:
   Combining direct-light and diffracted-light constraints, the following constrained optimization problem is solved:
    $\mathbf{x}^* = \arg\min_{\mathbf{x}} \|\mathbf{A}_{\text{dif}}\mathbf{x}\|_2^2 \quad \text{s.t.} \quad [\mathbf{0} ~ \mathbf{A}_{\text{dir}}]\mathbf{x} = \mathbf{m}_{\text{dir}}$
   A closed-form solution is obtained via Lagrange multipliers.

5. **Pixel-Wavelength Mapping Estimation**:

    - **Fluorescent + LED scheme**: Spike spectra from fluorescent lamps are used to establish wavelength-pixel correspondences; LED captures then provide direct/diffracted observations.
    - **LED-only scheme**: A point-to-plane ICP algorithm minimizes the distance between diffracted observations and expected sensitivity curves to estimate the quadratic mapping function $\lambda = ap^2 + bp + c$.

## Key Experimental Results

### Synthetic Experiment Results (RE×$10^{-2}$, lower is better)

| Method | EOS 650D | Olympus EPL2 | Pentax K5 | Galaxy S20 | Mean |
|:---|:---|:---|:---|:---|:---|
| **Ours (LED+Flu)** | **2.84** | **7.25** | **2.17** | **4.16** | **4.11** |
| Ours (LED) | 11.2 | 8.81 | 8.81 | 6.47 | 8.82 |
| CC (Color Chart) | 3.75 | 8.04 | 3.94 | 4.25 | 5.00 |
| Exif+CC | 5.02 | 8.56 | 5.02 | 6.89 | 6.37 |

**Key Findings**: The LED+Flu scheme achieves the best results on 4 out of 5 cameras in synthetic data, with a mean error of only 4.11%.

### Real-World Experiment Results (RE×$10^{-2}$)

| Method | EOS RP | iPhone 15ProMax | Sony α1 | DJI Pocket3 | Mean |
|:---|:---|:---|:---|:---|:---|
| **Ours (LED+Flu)** | **3.53** | 5.36 | **4.17** | 5.77 | **4.71** |
| Ours (LED) | 11.9 | **5.12** | 5.45 | **5.76** | 7.06 |
| CC | 8.45 | 9.13 | 8.99 | 6.59 | 8.29 |
| Exif+CC | 8.18 | 15.0 | 9.45 | 7.68 | 10.08 |

**Key Findings**: In real-world scenarios, Ours (LED+Flu) achieves the best results on most cameras, reducing error by approximately 43% compared to the color chart method CC (4.71 vs. 8.29). The LED-only scheme even outperforms LED+Flu on certain cameras, demonstrating that ICP-based mapping estimation is also effective for non-spike spectra.

### Summary of Key Findings
1. The diffraction grating approach significantly outperforms color chart methods: higher wavelength resolution enables more accurate sensitivity estimation.
2. The LED+Flu scheme is overall optimal; however, the LED-only scheme requiring a single capture also yields reasonable results.
3. The Exif method produces the largest errors across all scenarios, as it cannot account for the effects of lenses and filters.
4. Color chart methods are sensitive to real-world noise, as the low-frequency nature of patch spectral reflectance limits wavelength resolution.

## Highlights & Insights

1. **Minimal equipment requirements**: Only a single diffraction grating film costing under $5 USD is needed (no calibration required), substantially lowering the cost barrier compared to traditional methods.
2. **Mathematical elegance of the closed-form solution**: Basis function representations cleverly linearize the bilinear problem, guaranteeing the existence and uniqueness of the solution.
3. **High practical utility**: In the simplest mode, only a single LED with known spectrum and one capture (two exposures) are required, with no reference targets needed.
4. **Integration of theory and practice**: The mathematical model is derived from the physical principles of diffractive optics and validated on consumer-grade cameras.

## Limitations & Future Work

1. The pixel-wavelength mapping fails when a spectrally uniform light source is used (though nearly perfectly uniform sources rarely exist in practice).
2. The LED-only scheme relies on ICP optimization and may converge to local optima under poor initialization.
3. The effects of optical aberrations such as chromatic aberration and vignetting on estimation accuracy are not discussed.
4. Experiments cover only 8 cameras; generalization to a broader range of devices requires further validation.

## Related Work & Insights

- **Narrowband filter methods**: High accuracy but expensive equipment, unsuitable for large-scale use.
- **Color chart methods**: Finlayson, Kawakami, and others use ColorChecker to estimate sensitivity, but are limited by low-frequency reflectance.
- **Diffraction grating methods**: Karge et al. (2014) use fluorescent + halogen lamps with reference targets; Toivonen et al. (2019) use multiple light sources and transmissive color charts — both involve complex workflows.
- **Exif metadata methods**: Solomatov & Akkaynak (2023) train a neural network using metadata, but white balance ambiguity remains an issue.

## Rating
- Novelty: ★★★★☆ (closed-form joint estimation of sensitivity and grating efficiency is proposed for the first time)
- Experimental Thoroughness: ★★★★☆ (synthetic and real-world experiments with multi-camera validation, though comparison methods are limited)
- Value: ★★★★★ (low equipment cost, simple operation, high accuracy)
- Writing Quality: ★★★★★ (rigorous mathematical derivation, transparent experimental setup, clear exposition of the physical model)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](../../ICLR2026/llm_evaluation/spectral_attention_steering_for_prompt_highlighting.md)
- [\[NeurIPS 2025\] RGB-to-Polarization Estimation: A New Task and Benchmark Study](../../NeurIPS2025/llm_evaluation/rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)
- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](../../NeurIPS2025/llm_evaluation/houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)
- [\[CVPR 2026\] ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](../../CVPR2026/llm_evaluation/ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)
- [\[CVPR 2026\] HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT](../../CVPR2026/llm_evaluation/hess_head_sensitivity_score_for_sparsity_redistribution_in_vggt.md)

</div>

<!-- RELATED:END -->
