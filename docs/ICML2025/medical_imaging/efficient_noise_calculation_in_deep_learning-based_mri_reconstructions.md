---
title: >-
  [Paper Note] Efficient Noise Calculation in Deep Learning-based MRI Reconstructions
description: >-
  [ICML2025][Medical Imaging][MRI reconstruction] An efficient method based on Jacobian Sketching is proposed. By probing the Jacobian diagonal elements of DL reconstruction networks via random phase vectors, it accelerates the computation of voxel-level noise variance in MRI reconstruction using an unbiased estimator. The computation and memory requirements are reduced by more than an order of magnitude while maintaining a 99.8% correlation coefficient with the Monte Carlo ref…
tags:
  - "ICML2025"
  - "Medical Imaging"
  - "MRI reconstruction"
  - "noise propagation"
  - "Jacobian Sketching"
  - "voxel variance"
  - "uncertainty quantification"
  - "g-factor"
date: 2026-05-08
content_hash: 21e15a95519b4590
---

# Efficient Noise Calculation in Deep Learning-based MRI Reconstructions

**Conference**: ICML2025  
**arXiv**: [2505.02007](https://arxiv.org/abs/2505.02007)  
**Code**: To be released  
**Area**: Medical Imaging / Uncertainty Quantification  
**Keywords**: MRI reconstruction, noise propagation, Jacobian Sketching, voxel variance, uncertainty quantification, g-factor

## TL;DR

An efficient method based on Jacobian Sketching is proposed. By probing the Jacobian diagonal elements of DL reconstruction networks via random phase vectors, it accelerates the computation of voxel-level noise variance in MRI reconstruction using an unbiased estimator. The computation and memory requirements are reduced by more than an order of magnitude while maintaining a 99.8% correlation coefficient with the Monte Carlo reference.

## Background & Motivation

### Background

**Background**: Noise analysis in classical pMRI: Classical parallel imaging methods like SENSE have explicit g-factor analyses to evaluate spatial noise amplification.

### Limitations of Prior Work

**Limitations of Prior Work**: Lack of noise analysis in DL reconstruction: Due to the non-linear complexity of neural networks, existing DL reconstruction methods typically ignore noise propagation and rely solely on global metrics such as PSNR/SSIM.

### Key Challenge

**Key Challenge**: Limitations of Monte Carlo: Traditional MC methods require thousands of repeated reconstructions, which is computationally prohibitive, especially for 3D/4D data.

### Core Idea

**Core Idea**: Importance of noise analysis: It directly impacts SNR evaluation, sampling strategy design, network architecture selection, and confidence in clinical deployment.

### Supplementary Notes

**Supplementary Notes**: Threefold contributions: (1) A theoretical framework connecting k-space noise with image variance; (2) An efficient implementation of Jacobian Sketching; (3) Extensive validation across different architectures and training paradigms.

## Method

### Core Theory

- **Noise Covariance**: For a reconstruction function $f$, after first-order Taylor expansion:
  $$\bm{\Sigma}_{\bm{x}} = \bm{J}_f \bm{A}^H \bm{\Sigma}_k \bm{A} \bm{J}_f^H = \bm{L}\bm{L}^H$$
  where $\bm{L} = \bm{J}_f \bm{A}^H \bm{\sigma}_k$, and $\bm{J}_f$ is the network Jacobian.
- **Unbiased Estimator (Theorem 3.1)**: For any random vector $\bm{v}$ satisfying $\mathbb{E}[\bm{v}\bm{v}^H]=\bm{I}$ :
  $$\mathbb{E}[(\bm{\Sigma}\bm{v}) \odot \bm{v}^*] = \text{diag}(\bm{\Sigma})$$
- **Cholesky Decomposition Simplification (Lemma 3.2)**: "Voxel variance = $\|\bm{l}_i\|_2^2$", eliminating the need to explicitly compute the full Jacobian.

### Jacobian Sketching Algorithm

1. Generate a random phase vector matrix $\bm{V}_S \in \mathbb{C}^{m \times S}$ ($v_i = e^{j\theta_i}$).
2. Transform via $\bm{\sigma}_k$ and $\bm{A}^H$: $\widetilde{\bm{W}}_S = \bm{A}^H \bm{\sigma}_k \bm{V}_S$.
3. Compute $\bm{U}_S = \bm{J}_f \widetilde{\bm{W}}_S$ via JVP.
4. Hadamard product + averaging: $\widehat{\text{diag}(\bm{\Sigma}_{\bm{x}})} = \frac{1}{S} \bm{U}_S \odot \bm{U}_S^H \cdot \mathbf{1}_S$.

### Random Vector Selection

- Complex Rademacher (random-phase) vectors have lower variance than standard complex Gaussians, making them more suitable for MRI scenarios.
- $S=1000$ probing vectors are sufficient to achieve high accuracy.

## Key Experimental Results

### Main Results: Generalization Across Architectures (R=8, α=1)

| Method | Knee PCC(%) | Knee NRMSE(%) | Brain PCC(%) | Brain NRMSE(%) |
|---|---|---|---|---|
| E2E-VarNet | 99.9±0.0 | 0.7±0.0 | 99.9±0.0 | 0.5±0.1 |
| MoDL | 99.9±0.0 | 0.5±0.0 | 99.7±0.0 | 1.1±0.1 |
| U-Net | 99.4±0.0 | 1.7±0.2 | 99.7±0.0 | 1.8±0.2 |
| SSDU | 99.9±0.0 | - | - | - |

The average correlation coefficient across all methods is 99.8%, with an average error of 0.8%.

### Ablation Study

- **Robustness to Noise Levels**: Maintains high accuracy within the range of $\alpha \in \{1, 5, 10, ..., 200\}$.
- **Robustness to Acceleration Factors**: Valid under R=4, 8, and 12.
- **Robustness to Sampling Patterns**: Stable under various undersampling patterns such as Poisson Disc and random sampling.
- **Computational Efficiency**: Computation is approximately 1/10 to 1/100 of MC with 3,000 iterations.

## Highlights & Insights

1. **Rigorous yet Practical Theory**: Complete logical chain from first-order approximation $\rightarrow$ covariance decomposition $\rightarrow$ unbiased estimator $\rightarrow$ efficient implementation.
2. **Architecture-Agnostic**: Only requires the network to be differentiable (existence of Jacobian), making it suitable for both data-driven and physics-driven architectures.
3. **Filling the Gap in DL-MRI**: Reintroduces noise analysis as a core component of reconstruction algorithms.
4. **Complex Rademacher Vectors**: A customized random probing scheme tailored for MRI scenarios, offering lower variance.
5. **Clinical Significance**: Variance maps can directly guide radiologists to identify regions of noise amplification, enhancing diagnostic confidence.
6. **Value in Unsupervised Scenarios**: In unsupervised training regimes like SSDU where fully-sampled reference data is unavailable, SNR maps can replace PSNR/SSIM for quality assessment.

### Relationship with Traditional g-factor

- Classical SENSE g-factor: $g_i = \sqrt{[(\bm{S}^H\bm{\Psi}^{-1}\bm{S})^{-1}]_{ii} \cdot [\bm{S}^H\bm{\Psi}^{-1}\bm{S}]_{ii}}$
- The proposed method can be viewed as a natural generalization of the g-factor in non-linear DL reconstruction, preserving voxel-level spatial resolution.
- When $f$ is linear, the proposed method degenerates exactly to the classical g-factor analysis.

## Limitations & Future Work

- Modeled based on first-order Taylor approximation, which may introduce errors in highly non-linear networks (such as pure U-Net).
- Propagation of noise in generative models (e.g., diffusion models) is not considered, where stochasticity introduces additional complexity.
- Only handles acquisition noise (aleatoric uncertainty), leaving epistemic uncertainty uncovered.
- Linear approximation may fail in extremely low SNR regions.
- Practical application on 3D volumetric data is not yet explored (only 2D slices have been validated).
- Integrating variance maps with clinical workflows to show noise amplification areas in real-time remains future work.

## Related Work & Insights

- **Classical g-factor (Pruessmann et al., 1999)**: The proposed method serves as its natural generalization in DL reconstruction.
- **Hutchinson Estimator (1990)**: The foundation of diagonal estimation for real-valued matrices, which this work extends to the complex domain and integrates with MRI operators.
- **Dawood et al. (2024)**: Analytical noise estimation designed specifically for k-space interpolation networks; the proposed method is more general.
- **SSDU (Yaman et al., 2020), N2R (Desai et al., 2022)**: Unsupervised/semi-supervised training paradigms, both of which are validated to be compatible with this work.
- **Insights**: Can be utilized to guide noise-aware sampling strategy design, neural architecture search, and real-time integration with clinical workflows.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](../../NeurIPS2025/medical_imaging/ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[NeurIPS 2025\] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](../../NeurIPS2025/medical_imaging/domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [\[CVPR 2025\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](../../CVPR2025/medical_imaging/decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[ICML 2025\] The Disparate Benefits of Deep Ensembles](the_disparate_benefits_of_deep_ensembles.md)
- [\[CVPR 2026\] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](../../CVPR2026/medical_imaging/solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)

</div>

<!-- RELATED:END -->
