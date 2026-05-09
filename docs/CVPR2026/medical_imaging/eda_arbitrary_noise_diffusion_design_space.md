---
title: >-
  [Paper Note] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models
description: >-
  [CVPR 2026][Medical Imaging][Arbitrary-noise diffusion] This paper proposes EDA, a framework that extends the EDM design space from isotropic Gaussian noise to arbitrary noise patterns. By driving SDEs with multivariate Gaussian distributions and multiple independent Wiener processes, EDA enables flexible noise diffusion while provably introducing no additional sampling overhead. With only 5 sampling steps, EDA achieves performance on par with or superior to 100-step Refusion and task-specific methods across three tasks: MRI bias field correction, CT metal artifact removal, and natural image shadow removal.
tags:
  - CVPR 2026
  - Medical Imaging
  - Arbitrary-noise diffusion
  - EDM unified framework
  - SDE design space
  - medical image denoising
  - shadow removal
date: 2026-05-08
content_hash: 9d7974971241c241
---

# Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2507.18534](https://arxiv.org/abs/2507.18534)
**Code**: [https://github.com/PerceptionComputingLab/EDA](https://github.com/PerceptionComputingLab/EDA)
**Area**: Diffusion Models / Image Restoration
**Keywords**: Arbitrary-noise diffusion, EDM unified framework, SDE design space, medical image denoising, shadow removal

## TL;DR
This paper proposes EDA, a framework that extends the EDM design space from isotropic Gaussian noise to arbitrary noise patterns. By driving SDEs with multivariate Gaussian distributions and multiple independent Wiener processes, EDA enables flexible noise diffusion while provably introducing no additional sampling overhead. With only 5 sampling steps, EDA achieves performance on par with or superior to 100-step Refusion and task-specific methods across three tasks: MRI bias field correction, CT metal artifact removal, and natural image shadow removal.

## Background & Motivation
EDM (Karras et al., NeurIPS 2022) unifies the design space of most diffusion models, offering flexible noise schedules and training objective choices. However, its closed-form perturbation kernel $p_t(x_t|x_0) = \mathcal{N}(x_t; s(t)x_0, s^2(t)\sigma^2(t)\mathbf{I})$ restricts diffusion to pixel-wise independent Gaussian noise. Recent methods such as Flow Matching, MeanFlow, and Cold Diffusion relax the Gaussian constraint and support more flexible noise distributions, but either degrade to ODE-based diffusion (sacrificing the robustness and diversity afforded by stochasticity) or lack rigorous theoretical foundations.

For image restoration tasks, the EDM framework exhibits two critical limitations: (1) enforced injection of Gaussian noise destroys task-relevant information in degraded images; and (2) it artificially increases the restoration distance and complexity, since the model must start from $P_{LQ} + N_{Gaus}$ rather than directly from the degraded image $P_{LQ}$. Allowing the model to diffuse task-specific noise patterns would shorten the restoration path and reduce task difficulty.

## Core Problem
How can the diffusion noise pattern be extended from pure Gaussian to arbitrary forms while preserving EDM's structural flexibility (selectable noise schedules and training objectives), thereby establishing a unified SDE-based diffusion design space? It is further required to prove that this generalization incurs no additional computational overhead.

## Method

### Overall Architecture
The core idea of EDA is to generalize the noise in the diffusion process from a scalar-variance isotropic Gaussian to a multivariate Gaussian distribution governed by an arbitrary set of basis functions $H_{x_0}$. The forward process progressively perturbs data into a degraded form, while the reverse process performs deterministic sampling via the PFODE starting directly from the degraded image.

**Input**: Degraded images (e.g., MRI with bias fields, CT with metal artifacts, natural images with shadows)
**Output**: Restored high-quality images
**Pipeline**: Define task noise → Multivariate Gaussian forward process → Train denoiser → 5-step deterministic sampling for restoration

### Key Designs

1. **Generalized Forward Process (Arbitrary Noise Modeling)**: The diffusion noise is defined as $N = \sum_{m=1}^{M} \frac{\eta + \epsilon_m}{\eta + 1} h_{m,x_0}$, where $H_{x_0} = [h_{1,x_0}, \ldots, h_{M,x_0}]$ is a set of basis functions (which may be pre-defined in any form), $\epsilon_m \sim \mathcal{N}(0,1)$ are independent Gaussian variables, and the parameter $\eta \geq 0$ controls stochasticity ($\eta=0$ for maximum randomness; $\eta \to \infty$ for near-deterministic behavior). The covariance of the resulting perturbation distribution is $\Sigma_{x_0} = H_{x_0} H_{x_0}^\top$, which replaces EDM's diagonal matrix $\sigma^2(t)\mathbf{I}$ and captures structured noise patterns through the basis functions.

2. **Multi-Wiener-Process-Driven SDE**: The forward process is driven by multiple independent Wiener processes: $dx = [f(t)x + \phi_{x_0}(t)]dt + g(t)\sum_{m=1}^M h_{m,x_0} d\omega_t^{(m)}$, where the drift coefficient, diffusion coefficient, and offset term are all analytically derived from the noise schedule $s(t), \sigma(t)$ and the basis functions $H_{x_0}$.

3. **PFODE Simplification and Zero-Overhead Proof (Proposition 2)**: The paper's central theoretical contribution — upon substituting the score function into the PFODE and approximating with the denoiser, all additional terms involving the basis functions $h_m$ and the covariance $\Sigma_{x_0}$ cancel analytically, yielding a deterministic sampling formula $\frac{dx}{dt} = (\frac{s'(t)}{s(t)} + \frac{\sigma'(t)}{\sigma(t)})x - \frac{\sigma'(t)s(t)}{\sigma(t)} D_\theta(x;\sigma)$ that is identical to that of EDM. This implies that regardless of noise complexity, the sampling procedure requires no modification.

4. **Three Noise Configurations (Proposition 1)**:

    - **Case 1 (Unified basis functions, optimal)**: Basis functions are data-independent, $H = H_j, \forall j$ (e.g., low-order Legendre polynomials and trigonometric functions for MRI bias fields).
    - **Case 2 (Sample-dependent basis functions, general)**: $H_{x_0}$ varies per sample, as in CT metal artifact removal and shadow removal where $H_A = [B - A]$.
    - **Case 3 (Non-Gaussian noise via discrete sampling)**: Supports non-Gaussian noise such as Poisson noise through discrete distribution matching.

5. **EDM as a Special Case of EDA (Proposition 3)**: When $\eta=0$ and the basis functions are pixel-level identity matrices $E_{i,j}$, EDA reduces to standard EDM.

### Loss & Training
- Unified training objective: $\mathcal{L} = \mathbb{E}_{x_0 \sim P_{data}} \mathbb{E}_{x \sim P(x_t|y)} \|D_\theta(x;\sigma) - x_0\|^2$
- MRI bias field correction: Operations are performed in the log domain (converting multiplicative noise to additive); basis functions use low-order Legendre polynomials and trigonometric functions $H_{3,5}$; $\eta = 0$.
- CT metal artifact removal: $\eta = 10$; noise is defined as the difference between metal-affected and metal-free CT images; weighted MSE balances metal and non-metal regions.
- Shadow removal: $\eta = 10$; noise is defined as the difference between shadowed and shadow-free images; based on the ShadowFormer architecture.
- Total diffusion steps $T = 100$ for all tasks; noise schedule follows the DDPM linear $\beta$ scheme.
- Training hardware: single NVIDIA RTX 3090.

## Key Experimental Results

### MRI Bias Field Correction (HCP Dataset)

| Method | SSIM ↑ | PSNR ↑ | COCO ↑ | CV(WM) ↓ |
|--------|--------|--------|--------|----------|
| N4 | 0.95 | 25.62 | 0.95 | 7.95 |
| ABCNet | - | - | - | - |
| Refusion (100 steps) | - | - | - | - |
| **EDA (5 steps)** | **Best** | **Best** | **Best** | **Best** |

Speed comparison: EDA 0.182 sec/slice vs. Refusion 9.665 sec/slice → **~53× speedup**

### CT Metal Artifact Removal (DeepLesion)

| Method | Domain | Mean PSNR/SSIM |
|--------|--------|----------------|
| InDuDoNet+ | Dual-domain | 41.50/0.9891 |
| DICDNet | Dual-domain | 41.83/0.9923 |
| Refusion | Image domain | 38.15/0.9793 |
| **EDA (5 steps)** | **Image domain** | **38.67/0.9823** |

Using only image-domain information, EDA surpasses Refusion and approaches dual-domain methods.

### Natural Image Shadow Removal (ISTD)

| Method | ALL PSNR ↑ | ALL SSIM ↑ | NS PSNR ↑ |
|--------|------------|------------|-----------|
| ShadowFormer | - | - | - |
| Refusion | - | - | - |
| **EDA** | **Best** | **Best** | **34.31** |

EDA achieves the best performance on both full-image and non-shadow regions; the non-shadow region PSNR of 34.31 dB demonstrates precise boundary-aware restoration.

### Ablation Study
- 5-step EDA matches or exceeds 100-step Refusion, validating the effectiveness of shortening the restoration distance.
- MeanFlow (an ODE-based method) is significantly outperformed by EDA on all three tasks, confirming the importance of SDE stochasticity for restoration quality — ODE methods tend to produce blurry average solutions.
- Case 1 basis functions (data-independent) are theoretically optimal but limited in applicability; Case 2 (data-dependent) is more general and also yields excellent empirical results.

## Highlights & Insights
- **Theoretical elegance**: Proposition 2 proves that arbitrary noise complexity does not increase sampling overhead, as all additional terms cancel analytically in the PFODE — a particularly clean theoretical result.
- **Unified perspective**: Flow Matching, Cold Diffusion, and EDM can all be understood within the EDA framework, providing a broader theoretical foundation for future diffusion model research.
- **Practical value**: High-quality restoration in 5 steps, with a 53× speedup that enables deployment in clinical high-throughput settings.
- **Transferable design principle**: The idea of encoding task-specific degradation patterns directly as diffusion noise is generalizable to other restoration tasks such as dehazing, super-resolution, and deblurring.

## Limitations & Future Work
- **Trade-off between stochasticity and applicability**: Case 1 offers maximum stochasticity but requires data-independent basis functions (limiting applicability); Cases 2–3 reduce stochasticity in exchange for generality.
- **Validation limited to restoration tasks**: The framework has not been evaluated on generative tasks (e.g., unconditional image generation), and its generative capacity remains unclear.
- **Image-domain only**: Sinogram-domain information is not exploited in CT metal artifact removal, leaving a gap relative to dual-domain state-of-the-art methods.
- **Basis function design requires domain knowledge**: MRI requires Legendre polynomials; CT requires image difference representations — basis function selection relies on prior understanding of task-specific degradation patterns.
- The paper does not discuss broader restoration tasks beyond medical and natural images (e.g., deraining, dehazing).

## Related Work & Insights
- **vs. EDM (Karras et al., 2022)**: EDA is a strict superset of EDM (Proposition 3); the core distinction lies in generalizing the noise covariance from $\sigma^2\mathbf{I}$ to $\Sigma_{x_0}$ while keeping the sampling formula unchanged.
- **vs. Flow Matching / MeanFlow**: Although Flow Matching supports arbitrary noise, it is restricted to ODE-based diffusion and lacks the stochasticity benefits of SDEs; experiments show that MeanFlow performs significantly worse than EDA on all three restoration tasks, as ODE methods produce blurry average solutions.
- **vs. Refusion**: Both are Gaussian-diffusion-based restoration methods, but Refusion requires 100 steps and starts from noise-corrupted degraded images; EDA requires only 5 steps and starts directly from the degraded image, achieving a 53× speedup with superior performance.

**Further connections**:
- The arbitrary-noise diffusion paradigm can be combined with multimodal generation, where different modalities exhibit distinct noise patterns — the EDA framework naturally accommodates such heterogeneity.
- The efficiency of 5-step high-quality sampling could be integrated into VLM-driven medical image analysis pipelines.
- Basis function design could potentially be automated via neural network learning rather than manual specification.

## Rating
- Novelty: ⭐⭐⭐⭐ — The theoretical framework generalizing EDM to arbitrary noise is valuable, though mathematically the extension primarily concerns the covariance matrix.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three representative tasks cover diverse noise patterns, but validation on generative tasks is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and rigorous; propositions are well-organized; the appendix is thorough.
- Value: ⭐⭐⭐⭐ — Provides a unified and efficient theoretical framework for diffusion-based image restoration, though the scope of applicability warrants further investigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models (EDA)](elucidating_the_design_space_of_arbitrary-noise-based_diffusion_models.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[CVPR 2026\] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)
- [\[CVPR 2026\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
