---
title: >-
  [Paper Note] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Digital Watermarking] This paper proposes Shallow Diffuse, a method that exploits the local linearity and low-rank Jacobian of the posterior mean predictor (PMP) in diffusion models to em…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Digital Watermarking"
  - "Diffusion Models"
  - "Low-Dimensional Subspaces"
  - "DDIM"
  - "Frequency-Domain Embedding"
date: 2026-05-08
content_hash: 1270a9f89284c7c7
---

# Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2410.21088](https://arxiv.org/abs/2410.21088)  
**Code**: Not available  
**Area**: Diffusion Models / Watermarking / AI-Generated Content Detection
**Keywords**: Digital Watermarking, Diffusion Models, Low-Dimensional Subspaces, DDIM, Frequency-Domain Embedding

## TL;DR

This paper proposes Shallow Diffuse, a method that exploits the local linearity and low-rank Jacobian of the posterior mean predictor (PMP) in diffusion models to embed watermarks at intermediate diffusion timesteps. This design decouples the watermark from the generation process, achieving, for the first time, both high fidelity and high robustness simultaneously under both server-side and user-side deployment scenarios.

## Background & Motivation

**Background**: Diffusion model-driven commercial AI-generated content (e.g., Stable Diffusion, DALL-E, Imagen) has raised three major security concerns: (1) AI-generated misinformation threatens social stability; (2) training data memorization leads to copyright infringement; (3) iterative training on AI-generated content causes model collapse. Watermarking is a critical technique for identifying and tracing AI-generated content.

**Limitations of Prior Work**:
- Traditional watermarking methods (DWT, RivaGAN, etc.) are primarily designed for post-processing scenarios and offer insufficient robustness.
- Diffusion model-based methods (Tree-Ring, RingID) embed watermarks in the low-frequency Fourier components of the initial noise, which substantially distorts the Gaussian noise distribution and degrades generation fidelity.
- Existing methods either support only the server-side scenario (requiring control over the initial seed) or only the user-side scenario (post-processing embedding), failing to address both simultaneously.

**Key Challenge**: Robustness demands a strong watermark signal, while fidelity demands minimal modification to the image — these two objectives are fundamentally in tension.

**Key Insight**: The paper exploits the low-rank property of the PMP's Jacobian at intermediate timesteps, where most of the watermark energy falls into the null space of the Jacobian. This decouples the watermark from the sampling process — the watermark leaves the predicted $\hat{x}_0$ nearly unchanged (ensuring fidelity) while being fully preserved in $x_t$ (ensuring detectability).

## Method

### Overall Architecture

Shallow Diffuse injects watermarks at an intermediate timestep $t^* = 0.3T$ rather than in the initial noise $x_T$. The workflow is:
- **Server-side**: $x_T \to \text{DDIM sampling to}\ x_{t^*} \to \text{watermark injection} \to \text{DDIM sampling to}\ x_0^{\mathcal{W}}$
- **User-side**: $x_0 \to \text{DDIM inversion to}\ x_{t^*} \to \text{watermark injection} \to \text{DDIM sampling to}\ x_0^{\mathcal{W}}$
- **Detection**: $\bar{x}_0^{\mathcal{W}} \to \text{DDIM inversion to}\ x_{t^*} \to \text{watermark verification}$

### Key Designs

1. **Decoupling Principle via Low-Rank Structure**: The PMP $\mathbf{f}_{\theta,t}(x_t)$ predicts $\mathbb{E}[x_0|x_t]$, and its Jacobian $\mathbf{J}_{\theta,t}$ is **low-rank** (rank ratio $< 10^{-2}$) for $t \in [0.2T, 0.7T]$, while also exhibiting **local linearity**. Upon injecting a watermark $\lambda\Delta\mathbf{x}$:
    $$\mathbf{f}_{\theta,t}(x_{t^*} + \lambda\Delta\mathbf{x}) \approx \mathbf{f}_{\theta,t}(x_{t^*}) + \lambda\underbrace{\mathbf{J}_{\theta,t}(x_{t^*})\Delta\mathbf{x}}_{\approx \mathbf{0}}$$
   Since $r_{t^*} \ll d$ (rank much smaller than dimensionality), the energy of a random watermark $\Delta\mathbf{x}$ predominantly lies in the null space, yielding $\mathbf{J}\Delta\mathbf{x} \approx 0$. Consequently, the predicted $\hat{x}_0$ remains nearly unchanged, ensuring fidelity.

   **Design Motivation**: $t^* = 0.3T$ is chosen because the Jacobian rank ratio is minimized at this point, while PMP linearity is maximized.

2. **High-Frequency Watermark Design**: Unlike Tree-Ring/RingID, which modify low-frequency components, Shallow Diffuse embeds the watermark in the high-frequency region of the frequency domain:
    $$\lambda\Delta\mathbf{x} = \text{DFT}^{-1}(\text{DFT}(x_{t^*}) \odot (1-\mathbf{M}) + \mathbf{W} \odot \mathbf{M}) - x_{t^*}$$
   where $\mathbf{M}$ is a high-frequency mask (without zero-frequency centering) and $\mathbf{W}$ is a watermark key composed of multi-ring Gaussian values.

   **Design Motivation**: (1) High-frequency components carry low energy, so modifications cause minimal visual distortion; (2) Watermark injection at $x_{t^*}$ (close to the clean image rather than pure noise) makes high-frequency operations more stable.

3. **Watermark Detection**: Given a potentially attacked image $\bar{x}_0^{\mathcal{W}}$, the method recovers $\bar{x}_{t^*}^{\mathcal{W}}$ via DDIM inversion and computes a p-value:
    $$\eta = \frac{\text{sum}(\mathbf{M}) \cdot \|\mathbf{M} \odot \mathbf{W} - \mathbf{M} \odot \text{DFT}(\bar{x}_{t^*}^{\mathcal{W}})\|_F^2}{\|\mathbf{M} \odot \text{DFT}(\bar{x}_{t^*}^{\mathcal{W}})\|_F^2}$$
   Watermarked images yield $\eta \approx 0$, while non-watermarked images yield $\eta > \eta_0$ (threshold).

4. **Extension to T2I Models**: For text-to-image models (e.g., Stable Diffusion), watermark injection uses unconditional DDIM (empty prompt), decoupled from the CFG sampling process. The server-side uses CFG sampling to reach $x_{t^*}$; the user-side uses DDIM inversion to reach $x_{t^*}$.

### Theoretical Guarantees

- **Theorem 1 (Fidelity)**: The prediction deviation induced by the watermark satisfies $\|\hat{x}_{0,t}^{\mathcal{W}} - \hat{x}_{0,t}\|_2 \leq \lambda L h(r_t)$, where $h(r_t) \sim \sqrt{r_t/d}$ depends only on the Jacobian rank $r_t$ ($r_t \ll d$) and is weakly related to the ambient dimension $d$.
- **Theorem 2 (Detectability)**: The watermark recovery error after one DDIM step is proportional only to $h(\max\{r_{t-1}, r_t\})$ and VP schedule parameters, both of which are small.

## Key Experimental Results

### Main Results 1: Server-Side Scenario (Stable Diffusion 2-1-base, 5000 images)

| Method | CLIP↑ | FID↓ | PSNR↑ | SSIM↑ | Clean TPR | Avg. Attack TPR↑ |
|--------|-------|------|-------|-------|-----------|-----------------|
| SD w/o WM | 0.3669 | 25.56 | - | - | - | - |
| Tree-Ring | 0.3645 | 25.82 | 16.61 | 0.64 | 1.00 | 0.77 |
| RingID | 0.3637 | 27.13 | 14.27 | 0.51 | 1.00 | 0.91 |
| Gaussian Shading | 0.3663 | 26.17 | 11.04 | 0.48 | 1.00 | 0.93 |
| **Shallow Diffuse** | **0.3669** | **25.60** | **35.49** | **0.96** | 1.00 | **0.93** |

Shallow Diffuse achieves a PSNR **18.88 dB** higher than Tree-Ring and **21.22 dB** higher than RingID, with a substantial lead in fidelity.

### Main Results 2: User-Side Scenario (COCO Dataset)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Clean TPR | Avg. Attack TPR↑ |
|--------|-------|-------|--------|-----------|-----------------|
| Tree-Ring | 28.22 | 0.57 | 0.41 | 1.00 | 0.84 |
| RingID | 12.21 | 0.38 | 0.58 | 1.00 | 0.96 |
| Gaussian Shading | 10.17 | 0.23 | 0.65 | 1.00 | 0.92 |
| RivaGAN | 40.57 | 0.98 | 0.04 | 1.00 | 0.59 |
| **Shallow Diffuse** | **32.11** | **0.84** | **0.05** | 1.00 | **0.93** |

In the user-side scenario, Gaussian Shading and RingID exhibit extremely poor fidelity (PSNR of only 10–12 dB), while Shallow Diffuse maintains a PSNR above 32 dB.

### Ablation Study

| Configuration | PSNR | TPR@1%FPR (avg) | Notes |
|---------------|------|-----------------|-------|
| $t^* = 0.3T$ (default) | 35.49 | 0.93 | Best trade-off |
| $t^* = 0.1T$ | ~38 | ~0.5 | Closer to clean image but reduced robustness |
| $t^* = 0.5T$ | ~25 | ~0.95 | Slightly higher robustness but reduced fidelity |
| Low-frequency watermark | ~20 | ~0.90 | Low-frequency modification causes visual distortion |
| High-frequency watermark (default) | 35.49 | 0.93 | High-frequency modification minimizes distortion |

### Key Findings

- In the server-side scenario, Shallow Diffuse has negligible impact on generation quality (CLIP and FID are on par with the no-watermark baseline).
- Tree-Ring and RingID exhibit severely degraded fidelity in the user-side scenario, as they rely on modifying the initial noise distribution.
- Across 15 attack types, Shallow Diffuse performs well against both distortion attacks (JPEG, blurring, noise) and adversarial attacks.
- The Jacobian rank ratio is minimized near $t^* = 0.3T$ (rank ratio $< 10^{-2}$), validating the theoretical analysis.

## Highlights & Insights

- Exploiting the low-rank structure of the PMP is an elegant design choice — watermarks residing in the null space neither affect generation nor disappear from the latent state.
- The method is applicable to both server-side and user-side scenarios simultaneously, a flexibility absent in prior work.
- Theoretical analysis (Theorems 1 & 2) provides rigorous guarantees for both fidelity and detectability.
- The entire method is training-free and requires only an off-the-shelf diffusion model.

## Limitations & Future Work

- Watermark capacity is limited (the current design embeds a single key; multi-key identification is discussed in the appendix but with restricted applicability).
- The precision of DDIM inversion affects detection quality, particularly for models with high CFG guidance scales or very long inference schedules.
- Compatibility with non-DDIM samplers (e.g., DPM-Solver, Euler) remains to be validated.
- Against advanced adaptive attacks (e.g., an adversary aware of $t^*$ who re-noises and then denoises), additional defenses may be required.

## Related Work & Insights

- **Key distinction from Tree-Ring Watermarks**: Tree-Ring embeds in the low-frequency domain of the initial noise, tightly coupling the watermark with the sampling process; Shallow Diffuse embeds in the high-frequency domain at an intermediate timestep, decoupling it from sampling.
- **Connection to image steganography**: The low-dimensional subspace structure of diffusion models may inspire the design of higher-capacity steganographic schemes.
- The in-depth exploitation of PMP low-rank properties may also inspire downstream applications such as diffusion-based image editing and style transfer.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Exploiting the low-rank structure of PMP for watermark decoupling is a highly original perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 15 attack types, two deployment scenarios, multiple datasets, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear and intuitive figures; complete theoretical derivations.
- **Value**: ⭐⭐⭐⭐⭐ — Simultaneously resolves the fidelity–robustness trade-off with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] WMCopier: Forging Invisible Image Watermarks on Arbitrary Images](wmcopier_forging_invisible_image_watermarks_on_arbitrary_images.md)
- [\[ICCV 2025\] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](../../ICCV2025/image_generation/invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [\[NeurIPS 2025\] BitMark: Watermarking Bitwise Autoregressive Image Generative Models](bitmark_watermarking_bitwise_autoregressive_image_generative_models.md)
- [\[NeurIPS 2025\] Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems](preconditioned_langevin_dynamics_with_score-based_generative_models_for_infinite.md)

</div>

<!-- RELATED:END -->
