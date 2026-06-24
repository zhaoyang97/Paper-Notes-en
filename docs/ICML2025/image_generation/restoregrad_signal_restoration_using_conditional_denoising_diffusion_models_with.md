---
title: >-
  [Paper Note] RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior
description: >-
  [ICML2025][Image Generation][Diffusion Models] The RestoreGrad framework is proposed to jointly learn the prior distribution of conditional DDPMs (as opposed to a fixed standard Gaussian) using a Prior Net and a Posterior Net. By leveraging the correlation between degraded and clean signals to construct a more informative prior, it achieves 5-10× faster convergence and 2-2.5× fewer inference steps in speech enhancement and image restoration tasks.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Diffusion Models"
  - "Signal Restoration"
  - "Learnable Prior"
  - "VAE-DDPM Integration"
  - "Speech Enhancement"
  - "Image Restoration"
date: 2026-05-08
content_hash: f28a21c50447337d
---

# RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior

**Conference**: ICML2025  
**arXiv**: [2502.13574](https://arxiv.org/abs/2502.13574)  
**Code**: To be confirmed  
**Area**: Diffusion Models / Signal Restoration  
**Keywords**: Diffusion Models, Signal Restoration, Learnable Prior, VAE-DDPM Integration, Speech Enhancement, Image Restoration

## TL;DR
The RestoreGrad framework is proposed to jointly learn the prior distribution of conditional DDPMs (as opposed to a fixed standard Gaussian) using a Prior Net and a Posterior Net. By leveraging the correlation between degraded and clean signals to construct a more informative prior, it achieves 5-10× faster convergence and 2-2.5× fewer inference steps in speech enhancement and image restoration tasks.

## Background & Motivation

Conditional DDPMs have demonstrated outstanding performance in signal restoration (speech enhancement, image restoration), but face two critical bottlenecks:

**Slow Convergence**: Standard DDPMs adopt standard Gaussian $\mathcal{N}(\mathbf{0}, \mathbf{I})$ as the prior distribution. The large gap between this and the real data distribution leads to training requiring a large number of iterations to converge.

**High Number of Inference Steps**: The gap between the prior and the data distribution also means that the reverse process requires more steps to reconstruct the signal.

**Limitations of Prior Work**: PriorGrad proposed a rule-based method to extract prior information from conditional inputs. However, such handcrafted priors require domain knowledge, and suitable rules cannot be easily designed for all tasks.

**Key Insight**: In signal restoration tasks, the degraded signal $\mathbf{y}$ is inherently a corrupted version of the clean signal $\mathbf{x}_0$, exhibiting a strong correlation. Standard Gaussian priors completely discard this correlated information, resulting in a loss of efficiency.

## Method

### Core Idea: VAE + DDPM Integration

RestoreGrad seamlessly integrates DDPM into a VAE framework:

- **DDPM as Decoder**: Leveraging the strong generative capability of diffusion models for signal reconstruction.
- **VAE Encoder for Prior Learning**: Jointly learning a better latent space through a Prior Net ($\psi$) and a Posterior Net ($\phi$).

### Architecture Design

The framework consists of three learnable modules:

| Module | Parameterization | Input | Function |
|------|--------|------|------|
| Conditional DDPM ($\theta$) | Noise estimation network | $\mathbf{x}_t, \mathbf{y}, t$ | Predicts noise, performs reverse denoising |
| Prior Net ($\psi$) | Prior encoder | $\mathbf{y}$ | Provides informative prior during inference |
| Posterior Net ($\phi$) | Posterior encoder | $\mathbf{x}_0, \mathbf{y}$ | Utilizes clean signals to assist learning during training |

Both prior and posterior distributions are modeled as zero-mean Gaussians:

$$p_\psi(\boldsymbol{\epsilon}|\mathbf{y}) = \mathcal{N}(\boldsymbol{\epsilon}; \mathbf{0}, \boldsymbol{\Sigma}_{\text{prior}}(\mathbf{y};\psi))$$

$$q_\phi(\boldsymbol{\epsilon}|\mathbf{x}_0, \mathbf{y}) = \mathcal{N}(\boldsymbol{\epsilon}; \mathbf{0}, \boldsymbol{\Sigma}_{\text{post}}(\mathbf{x}_0, \mathbf{y};\phi))$$

### New ELBO Derivation

By integrating the diffusion process into the VAE framework via Proposition 3.1, a new lower bound for the conditional data log-likelihood is derived (Eq. 9), consisting of two components:
- ELBO of the conditional DDPM (reconstruction term)
- KL divergence between the Prior Net and Posterior Net (prior matching term)

### Loss & Training

The final joint training objective (Eq. 11) consists of three terms:

$$\min_{\theta,\phi,\psi} \;\; \eta \cdot \mathcal{L}_{\text{LR}} + \mathcal{L}_{\text{DM}} + \lambda \cdot \mathcal{L}_{\text{PM}}$$

- **Latent Regularization (LR)**: $\mathcal{L}_{\text{LR}} = \bar{\alpha}_T \|\mathbf{x}_0\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{post}}} + \log|\boldsymbol{\Sigma}_{\text{post}}|$, preventing the posterior covariance from increasing infinitely.
- **Denoising Matching (DM)**: $\mathcal{L}_{\text{DM}} = \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{y}, t)\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{post}}}$, training the DDPM to predict the ground-truth noise, with weighting by the inverse of the posterior covariance.
- **Prior Matching (PM)**: $\mathcal{L}_{\text{PM}} = \log\frac{|\boldsymbol{\Sigma}_{\text{prior}}|}{|\boldsymbol{\Sigma}_{\text{post}}|} + \text{tr}(\boldsymbol{\Sigma}^{-1}_{\text{prior}}\boldsymbol{\Sigma}_{\text{post}})$, aligning prior and posterior distributions.

Where $\eta, \lambda > 0$ are hyperparameters, with $\eta$ controlling the regularization strength and $\lambda$ controlling the prior matching weight.

### Inference Procedure

During inference, the Posterior Net is no longer used (as there is no clean signal $\mathbf{x}_0$), and the DDPM directly samples from the Prior Net: $\boldsymbol{\epsilon} \sim p_\psi(\boldsymbol{\epsilon}|\mathbf{y}) = \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_{\text{prior}})$.

## Key Experimental Results

### Speech Enhancement (VoiceBank+DEMAND)

| Method | Training Epochs | Inference Steps | PESQ↑ | CSIG↑ | CBAK↑ | COVL↑ | SI-SNR↑ |
|------|----------|---------|-------|-------|-------|-------|---------|
| CDiffuSE (Baseline) | 445 | 6 | 2.44 | 3.66 | 2.83 | 3.03 | - |
| + PriorGrad | 96 | 6 | 2.42 | 3.67 | 2.93 | 3.03 | 14.21 |
| + **RestoreGrad** | **96** | **6** | **2.51** | **3.80** | **3.00** | **3.14** | **14.74** |
| + **RestoreGrad** | **96** | **3** | **2.50** | **3.75** | **2.99** | **3.11** | **14.65** |

**Key points**: RestoreGrad surpasses the baseline's 445-epoch result using only 96 epochs; performance is barely degraded when inference steps are halved.

### Image Restoration (AllWeather → Multi-weather Evaluation)

| Method | Snow100K-L PSNR/SSIM | Outdoor-Rain PSNR/SSIM | RainDrop PSNR/SSIM |
|------|---------------------|----------------------|-------------------|
| WeatherDiff (1775 epochs) | 30.09/0.904 | 29.64/0.931 | 30.71/0.931 |
| + RestoreGrad (887 epochs) | 30.82/0.916 | **30.83/0.941** | 31.78/0.939 |
| + RestoreGrad (1551 epochs) | **31.16/0.918** | 30.70/0.942 | **32.26/0.941** |

**Key points**: Halving the training epochs is sufficient to comprehensively outperform the baseline; longer training yields further improvements, comparable to the weather-specific DTPM model.

### Encoder Overhead Analysis

| Encoder Size | PESQ↑ | SI-SNR↑ | Latency Ratio | VRAM Ratio |
|-----------|-------|---------|---------|---------|
| Tiny (24K) | 2.48 | 13.74 | 1.9% | 6.5% |
| Base (93K) | 2.51 | 14.74 | 2.2% | 10.3% |
| Large (370K) | 2.54 | 15.01 | 2.6% | 18.2% |

Encoder overhead is minimal (<3% latency, <19% VRAM), and performance steadily improves as the encoder size increases.

### Posterior Net Ablation Study

| Configuration | PESQ↑ | COVL↑ | SI-SNR↑ |
|------|-------|-------|---------|
| CDiffuSE Baseline | 2.32 | 2.89 | 11.84 |
| + PriorGrad | 2.42 | 3.03 | 14.21 |
| + RestoreGrad (Full) | **2.51** | **3.14** | **14.74** |
| + RestoreGrad w/o Posterior (η=0) | — | Training Diverges | — |
| + RestoreGrad w/o Posterior (η=1) | 2.48 | 3.12 | 13.29 |

The Posterior Net is crucial for stabilizing training and improving performance.

## Highlights & Insights

1. **Theoretical Elegance of VAE-DDPM Integration**: It seamlessly embeds DDPM into the VAE framework through a new ELBO derivation, inheriting both the generation capability of DDPM and the modeling efficiency of VAE.
2. **Extremely Low Overhead**: The encoder parameters are only 0.3%-2% of the DDPM, with a latency of <3%, yet it leads to a notable acceleration in convergence.
3. **Cross-modal Versatility**: The same framework is effective for both speech (1D waveforms) and images (2D). Visualizations of the covariance learned by the Prior Net show that it automatically captures signal structures.
4. **Dual Acceleration in Training and Inference**: 5-10× training acceleration + 2-2.5× reduction in inference steps, demonstrating outstanding practical value.
5. **Clever Design of the Posterior Net**: Ground-truth information is utilized during training to guide prior learning, and the Posterior Net is discarded during inference, incurring zero additional inference cost.

## Limitations & Future Work

1. **Restricted Prior Formulation**: Currently, it assumes a zero-mean Gaussian and only learns the covariance. This might be insufficiently flexible for scenarios with non-zero means or non-Gaussian distributions.
2. **Evaluation Limited to Signal Restoration Tasks**: Broader applications such as conditional generation (e.g., text-to-image, text-to-speech) have not been explored.
3. **Hyperparameter Sensitivity**: Although robust within a wide range of $\eta$, adjustments are still required for $\eta$ and $\lambda$, and the optimal values may vary across different tasks.
4. **Simple Encoder Architecture**: A fixed ResNet-20 is used, without exploring stronger encoders or adaptive architectures.
5. **Lack of Comparison with Recent Methods**: No comparison has been made with newer paradigms such as consistency models or flow matching.

## Related Work & Insights

- **PriorGrad** (Lee et al., 2022): A pioneer in handcrafted priors; RestoreGrad can be regarded as its learnable upgrade.
- **CDiffuSE** (Lu et al., 2022): DDPM baseline for speech enhancement.
- **WeatherDiff** (Özdenizci & Legenstein, 2023): DDPM baseline for multi-weather image restoration.
- **DTPM** (Ye et al., 2024): Diffusion texture prior model, which requires large-scale pre-training.

## Rating
- Novelty: ⭐⭐⭐⭐ — The approach of VAE-DDPM integration to learn priors is novel, accompanied by a complete theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Verified across modalities with comprehensive ablation studies and OOD generalization tests.
- Writing Quality: ⭐⭐⭐⭐ — The mathematical derivations are clear, and the presentation of figures and tables is rich.
- Value: ⭐⭐⭐⭐ — Highly practical plug-and-play acceleration for existing conditional DDPMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Navigating Image Restoration with VAR's Distribution Alignment Prior](../../CVPR2025/image_generation/navigating_image_restoration_with_vars_distribution_alignment_prior.md)
- [\[CVPR 2025\] Hiding Images in Diffusion Models by Editing Learned Score Functions](../../CVPR2025/image_generation/hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)
- [\[NeurIPS 2025\] AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models](../../NeurIPS2025/image_generation/accuquant_simulating_multiple_denoising_steps_for_quantizing.md)
- [\[ACL 2025\] OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching](../../ACL2025/image_generation/ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)

</div>

<!-- RELATED:END -->
