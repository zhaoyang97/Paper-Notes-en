---
title: >-
  [Paper Note] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Arbitrary Noise Diffusion] The EDA framework is proposed to extend the design space of EDM from pure Gaussian noise to arbitrary noise patterns. Flexible noise diffusion is achieved through SDEs driven by multivariate Gaussian distributions and multiple independent Wiener processes, proving that increased noise complexity introduces no additional sampling overhead. With only 5 sampling steps, it achieves performance comparable to or better than 1…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Arbitrary Noise Diffusion"
  - "EDM Unified Framework"
  - "SDE Design Space"
  - "Medical Image Denoising"
  - "Shadow Removal"
date: 2026-05-08
content_hash: ad511b1e48a35c67
---

# Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models

**Conference**: CVPR 2026  
**arXiv**: [2507.18534](https://arxiv.org/abs/2507.18534)  
**Code**: [https://github.com/PerceptionComputingLab/EDA](https://github.com/PerceptionComputingLab/EDA)  
**Area**: Diffusion Models / Image Restoration  
**Keywords**: Arbitrary Noise Diffusion, EDM Unified Framework, SDE Design Space, Medical Image Denoising, Shadow Removal  

## TL;DR
The EDA framework is proposed to extend the design space of EDM from pure Gaussian noise to arbitrary noise patterns. Flexible noise diffusion is achieved through SDEs driven by multivariate Gaussian distributions and multiple independent Wiener processes, proving that increased noise complexity introduces no additional sampling overhead. With only 5 sampling steps, it achieves performance comparable to or better than 100-step Refusion and specialized methods on MRI bias field correction, CT metal artifact removal, and natural image shadow removal.

## Background & Motivation
EDM (Karras et al., NeurIPS 2022) unifies the design space of most diffusion models, providing flexible noise scheduling and training objective choices. However, its closed-form perturbation kernel $p_t(x_t|x_0) = \mathcal{N}(x_t; s(t)x_0, s^2(t)\sigma^2(t)\mathbf{I})$ is restricted to diffusing pixel-wise independent Gaussian noise. Recently, methods like Flow Matching, MeanFlow, and Cold Diffusion have broken the Gaussian constraint to support flexible noise distributions but either degrade to ODE diffusion (sacrificing robustness and diversity provided by randomness) or lack a rigorous theoretical foundation.

For image restoration tasks, the EDM framework faces two key defects: (1) forced injection of Gaussian noise destroys task-related information in the degraded image; (2) it artificially increases restoration distance and complexity, as the model must start from $P_{LQ} + N_{Gaus}$ rather than directly from the degraded image $P_{LQ}$. Allowing the model to directly diffuse task-specific noise patterns could shorten the restoration path and reduce task difficulty.

## Core Problem
How to expand the diffusion noise pattern from pure Gaussian to arbitrary patterns while maintaining the flexibility of EDM structural parameters (noise scheduling, optional training objectives), thereby establishing a unified SDE-based diffusion design space? Furthermore, it must be proved that such a generalization incurs no additional computational overhead.

## Method

### Overall Architecture

EDA resolves the limitation where EDM only diffuses pixel-wise independent Gaussian noise. For restoration tasks, forcefully injecting Gaussian noise destroys task information and pushes the restoration starting point from the degraded image $P_{LQ}$ to $P_{LQ}+N_{Gaus}$, unnecessarily lengthening the restoration path. The core of EDA is to generalize the diffusion noise from an isotropic Gaussian with scalar variance to a multivariate Gaussian controlled by an arbitrary basis function set $H_{x_0}$. The forward process gradually perturbs the data into the degraded form according to the task, while the reverse process uses PFODE deterministic sampling to recover directly from the degraded image in just 5 steps.

### Key Designs

**1. Generalized Forward Process: Replacing Gaussian noise with multivariate Gaussian driven by arbitrary basis functions and providing it with rigorous SDEs**

To diffuse "task-specific noise" instead of generic Gaussian noise, EDA defines noise as $N = \sum_{m=1}^{M} \frac{\eta + \epsilon_m}{\eta + 1} h_{m,x_0}$, where $H_{x_0} = [h_{1,x_0}, \ldots, h_{M,x_0}]$ is a pre-defined basis function set and $\epsilon_m \sim \mathcal{N}(0,1)$ are independent Gaussian variables. The parameter $\eta \geq 0$ controls randomness ($\eta=0$ for maximum randomness, $\eta \to \infty$ for determinism). The covariance of the perturbation distribution $\Sigma_{x_0} = H_{x_0} H_{x_0}^\top$ is no longer the diagonal matrix $\sigma^2(t)\mathbf{I}$ used in EDM, allowing basis functions to directly capture structured degradation patterns.

The dynamic form of this noise is driven by multiple independent Wiener processes: $dx = [f(t)x + \phi_{x_0}(t)]dt + g(t)\sum_{m=1}^M h_{m,x_0} d\omega_t^{(m)}$, where the drift coefficient, diffusion coefficient, and offset term can be derived analytically from the noise schedule $s(t), \sigma(t)$ and basis functions $H_{x_0}$. This provides a rigorous SDE foundation for arbitrary noise and preserves the robustness and diversity of randomness, unlike Flow Matching which degrades to ODE.

**2. PFODE Simplification and Zero Extra Overhead (Proposition 2)**

This is the most critical theoretical finding: by substituting the score function into the PFODE and approximating it with a denoiser, all extra terms related to basis functions $h_m$ and covariance $\Sigma_{x_0}$ cancel out analytically. The final deterministic sampling formula $\frac{dx}{dt} = (\frac{s'(t)}{s(t)} + \frac{\sigma'(t)}{\sigma(t)})x - \frac{\sigma'(t)s(t)}{\sigma(t)} D_\theta(x;\sigma)$ is identical to EDM. In other words, no matter how complex the noise design is, the sampling process remains unchanged and incurs no extra overhead.

**3. Three Noise Configurations (Proposition 1)**

The choice of basis functions leads to three practical configurations:

- **Case 1 (Unified Basis, Optimal)**: Basis functions are data-independent $H = H_j, \forall j$ (e.g., using low-order Legendre polynomials and trigonometric functions for MRI bias fields).
- **Case 2 (Sample-dependent Basis, Universal)**: $H_{x_0}$ varies with samples, such as $H_{A} = [B - A]$ for CT metal artifacts and shadow removal.
- **Case 3 (Non-Gaussian Noise Discrete Sampling)**: Supports non-Gaussian noise like Poisson noise through discrete spatial distribution matching.

**4. EDM as a Special Case of EDA (Proposition 3)**

When $\eta=0$ and the basis functions are pixel-wise identity matrices $E_{i,j}$, EDA degrades to standard EDM—indicating that EDA is a strict superset of EDM.

### Loss & Training
- Unified training objective: $\mathcal{L} = \mathbb{E}_{x_0 \sim P_{data}} \mathbb{E}_{x \sim P(x_t|y)} \|D_\theta(x;\sigma) - x_0\|^2$
- MRI Bias Field Correction: Operations in the log-domain (turning multiplicative noise to additive), using low-order Legendre and trigonometric basis $H_{3,5}$ with $\eta = 0$.
- CT Metal Artifact Removal: $\eta = 10$, noise is the difference between CT with and without metal, using weighted MSE to balance domains.
- Shadow Removal: $\eta = 10$, noise is the difference between shadow and shadow-free images, based on the ShadowFormer architecture.
- Total diffusion steps $T = 100$, following the DDPM linear $\beta$ schedule.
- Hardware: Single NVIDIA RTX 3090.

## Key Experimental Results

### MRI Bias Field Correction (HCP Dataset)

| Method | SSIM ↑ | PSNR ↑ | COCO ↑ | CV(WM) ↓ |
|------|--------|--------|--------|----------|
| N4 | 0.95 | 25.62 | 0.95 | 7.95 |
| ABCNet | - | - | - | - |
| Refusion (100 steps) | - | - | - | - |
| **EDA (5 steps)** | **Optimal** | **Optimal** | **Optimal** | **Optimal** |

Speed comparison: EDA 0.182 sec/slice vs Refusion 9.665 sec/slice → **~53× acceleration**.

### CT Metal Artifact Removal (DeepLesion)

| Method | Domain | Mean PSNR/SSIM |
|------|-----|----------------|
| InDuDoNet+ | Dual | 41.50/0.9891 |
| DICDNet | Dual | 41.83/0.9923 |
| Refusion | Image | 38.15/0.9793 |
| **EDA (5 steps)** | **Image** | **38.67/0.9823** |

EDA surpasses Refusion using only image-domain information and approaches dual-domain methods.

### Natural Image Shadow Removal (ISTD)

| Method | ALL PSNR ↑ | ALL SSIM ↑ | NS PSNR ↑ |
|------|------------|------------|-----------|
| ShadowFormer | - | - | - |
| Refusion | - | - | - |
| **EDA** | **Optimal** | **Optimal** | **34.31** |

EDA achieves optimal results on the full image and non-shadow regions, with a non-shadow PSNR of 34.31 dB proving precise boundary awareness.

### Ablation Study
- 5-step EDA achieves or exceeds the performance of 100-step Refusion, confirming the effectiveness of shortening the restoration path.
- MeanFlow (ODE method) is significantly weaker than EDA across all tasks, validating the importance of SDE randomness—ODE tends to output blurry average solutions.
- Case 1 basis (data-independent) is theoretically optimal but limited in scope; Case 2 (data-dependent) is more universal and performs excellently in experiments.

## Highlights & Insights
- **Theoretical Elegance**: Proposition 2 proves that arbitrary noise complexity does not increase sampling overhead, as extra terms cancel out in the PFODE—a significant theoretical result.
- **Unified Perspective**: Flow Matching, Cold Diffusion, and EDM can all be understood within the EDA framework, providing a broader theoretical basis for future research.
- **Practical Value**: High-quality restoration in 5 steps enables its use in high-throughput clinical scenarios.
- **Transferable Design**: The concept of encoding task-specific degradation directly as diffusion noise can be extended to other restoration tasks like dehazing, super-resolution, and deblurring.

## Limitations & Future Work
- **Randomness vs. Versatility**: Case 1 offers maximum randomness but requires data-independent basis functions; Case 2-3 trade randomness for universality.
- **Restoration Focus**: The framework has not been validated on generative tasks (e.g., unconditional image generation), so its generative capacity is unclear.
- **Image Domain Only**: In CT metal artifact removal, sinogram-domain information is not utilized, leaving a gap compared to dual-domain SOTA.
- **Basis Selection**: Designing basis functions requires domain knowledge (e.g., Legendre for MRI, image differences for CT), relying on priors of the degradation pattern.

## Related Work & Insights
- **vs EDM (Karras et al., 2022)**: EDA is a strict superset of EDM (Proposition 3). The core difference is generalizing noise from $\sigma^2\mathbf{I}$ to $\Sigma_{x_0}$ while keeping the sampling formula identical.
- **vs Flow Matching / MeanFlow**: While Flow Matching supports arbitrary noise, it is limited to ODE diffusion and lacks SDE randomness. Experiments show MeanFlow performs worse due to blurry average solutions.
- **vs Refusion**: Unlike the 100-step Gaussian diffusion in Refusion starting from noise-corrupted images, EDA starts directly from the degraded image and completes restoration in 5 steps, achieving 53x speedup with better results.

## Rating
- Novelty: ⭐⭐⭐⭐ Generalizing EDM to an arbitrary noise theoretical framework is valuable, though mathematically focused on covariance generalization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three representative tasks cover different noise patterns, but generative tasks are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and rigorous, with organized propositions and detailed appendices.
- Value: ⭐⭐⭐⭐ Provides a unified and efficient theoretical framework for diffusion-based restoration, though its broader scope requires further verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](dcw_snr_t_bias_diffusion.md)
- [\[CVPR 2026\] What Is It Like to Be a Noise? An Entropy-based Gaussian Noise Regularization for Diffusion Models](what_is_it_like_to_be_a_noise_an_entropy-based_gaussian_noise_regularization_for.md)
- [\[ICLR 2026\] Exploring the Design Space of Transition Matching](../../ICLR2026/image_generation/exploring_the_design_space_of_transition_matching.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] Scale Space Diffusion: Integrating Scale Space into the Diffusion Process](scale_space_diffusion.md)

</div>

<!-- RELATED:END -->
