---
title: >-
  [Paper Note] Information Theoretic Learning for Diffusion Models with Warm Start
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Models] This paper proposes a likelihood estimation framework that generalizes the classical KL divergence–Fisher information relationship to arbitrary isotropic noise perturbations, combined with warm-start noise injection and importance sampling to eliminate the train-test gap and achieve tighter likelihood upper bounds, attaining state-of-the-art NLL on ImageNet at multiple resolutions.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion Models
  - Likelihood Estimation
  - information theory
  - Fisher Divergence
  - Warm Start
date: 2026-05-08
content_hash: ebc562cfe2cf1845
---

# Information Theoretic Learning for Diffusion Models with Warm Start

**Conference**: NeurIPS 2025
**arXiv**: [2510.20903](https://arxiv.org/abs/2510.20903)
**Code**: None
**Area**: Image Generation
**Keywords**: Diffusion Models, Likelihood Estimation, information theory, Fisher Divergence, Warm Start

## TL;DR

This paper proposes a likelihood estimation framework that generalizes the classical KL divergence–Fisher information relationship to arbitrary isotropic noise perturbations, combined with warm-start noise injection and importance sampling to eliminate the train-test gap and achieve tighter likelihood upper bounds, attaining state-of-the-art NLL on ImageNet at multiple resolutions.

## Background & Motivation

**Background**: Negative log-likelihood (NLL) is a fundamental metric for evaluating density estimation and generative models, with applications in data compression, anomaly detection, and adversarial purification. Diffusion models represent one of the current state-of-the-art likelihood models: variance-preserving (VP) models are trained via the ELBO, while variance-exploding (VE) models directly estimate likelihood using information-theoretic tools.

**Limitations of Prior Work**:
   - The ELBO for VP models achieves strong performance but converges slowly (requiring millions of iterations).
   - The information-theoretic (IT) bound for VE models converges faster but yields inferior likelihood performance compared to VP.
   - As $\sigma_t^2 \to 0$, the diverging SNR causes numerical instability in diffusion models; in practice, training starts from $t = \epsilon > 0$, introducing a train-test mismatch.
   - Discrete data (e.g., images) requires dequantization (uniform or variational), which introduces additional training stages and noise mismatch.

**Key Challenge**: Score matching (Fisher divergence) is exactly equivalent to MLE only when $\sigma_t^2 \to 0$; under finite noise, a bias exists. Meanwhile, numerical instability arises near $t = 0$.

**Goal**: Obtain tighter likelihood bounds without increasing training cost, eliminate the train-test gap, and support non-Gaussian noise.

**Key Insight**: The diffusion process is viewed as a Gaussian channel; mismatched entropy is introduced to characterize the model-data discrepancy, and a low-variance non-Gaussian warm-start noise interval is incorporated.

**Core Idea**: The relationship between score matching and KL divergence is generalized from Gaussian to arbitrary isotropic noise (Theorem 1), combined with warm-start noise injection to unify training and evaluation.

## Method

### Overall Architecture

The forward process of the diffusion model is $\mathbf{Y}_t = \alpha_t \mathbf{X} + \sigma_t \mathbf{N}$, operating over two variance intervals:
1. **Low-variance interval** $[0, \sigma_0^2)$: Arbitrary isotropic warm-up noise $\Psi$ is injected to produce $\tilde{\mathbf{x}} = \alpha_0 \mathbf{x} + \sigma(0)\mathbf{u}$.
2. **High-variance Gaussian interval** $[\sigma_0^2, \sigma_1^2]$: Standard Gaussian diffusion process.

The same warm-up noise is applied to data during both training and evaluation, eliminating the mismatch.

### Key Designs

#### 1. Generalized Relationship Between Score Matching and KL Divergence (Theorem 1)
- **Function**: Proves that the Fisher divergence is the limit of the first-order derivative of the KL divergence with respect to noise variance, holding for arbitrary isotropic noise.
- **Core Formula**:
  $$\frac{d}{d\sigma_t^2} D_{\text{KL}}(p_{\sigma_t^2} \| q_{\sigma_t^2})\bigg|_{\sigma_t^2 \to 0^+} = -\frac{1}{2} I(p(\mathbf{x}) \| q(\hat{\mathbf{x}}; \theta))$$
  where $I(\cdot \| \cdot)$ denotes the Fisher divergence (the score matching objective).
- **Significance**: Extends the previously Gaussian-restricted result to arbitrary noise distributions including Poisson, Laplacian, and uniform.

#### 2. Mismatched Entropy Decomposition (Proposition 1)
- **Function**: Decomposes the model likelihood exactly into an output distribution loss, a score approximation error, and an irreducible information term.
- **Core Formula**:
  $$\mathcal{H}(p(\mathbf{x}), q(\hat{\mathbf{x}};\theta)) = \mathcal{H}(p(\mathbf{y}_1), \pi) + \mathcal{J}_{\text{DSM}}(\theta) - \frac{1}{2}\int_{\sigma_0^2}^{\sigma_1^2} \mathbb{E}\|\nabla_{\mathbf{y}_t} \log p(\mathbf{y}_t|\mathbf{x})\|^2 d\sigma_t^2 + o(\sigma_0^2)$$
- **Design Motivation**: Provides an information-theoretic decomposition of the likelihood, clarifying the origin of each term.

#### 3. Pointwise Likelihood Upper Bound (Theorem 2)
- **Core Formula**:
  $$-\log q(\hat{\mathbf{x}};\theta) \leq \mathcal{H}(p(\mathbf{y}_1|\mathbf{x}), \pi(\mathbf{x})) + \mathcal{L}_{\text{DSM}}(\sigma_t^2;\theta)$$
  where $\mathcal{L}_{\text{DSM}}$ is the denoising score matching loss and $\mathcal{H}(p(\mathbf{y}_1|\mathbf{x}), \pi)$ can be computed analytically.
- **Significance**: The training objective is directly a pointwise NLL upper bound, which is tighter than the ELBO.

#### 4. Warm-Start Dequantization
- **Function**: Replaces traditional uniform dequantization with arbitrary isotropic noise $\Psi$.
- **Generalized De Bruijn Identity** (Proposition 2): The derivative of differential entropy with respect to noise variance is independent of noise type and depends only on Fisher information.
- **In Practice**: The same warm-up noise is added during both training and testing, eliminating the train-test gap without any additional training stage.

#### 5. Importance Sampling
- A sampling distribution $\rho(\eta) \propto w(\eta)$ is designed in the log-SNR space $\eta = -\log \text{SNR}(t)$.
- A noise prediction model $\hat{\mathbf{n}}(\mathbf{y}_t, \eta_t; \theta)$ is used, with time embeddings replaced by log-SNR embeddings.

### Loss & Training

- Training loss: $\nabla_\theta \|\mathbf{n} - \hat{\mathbf{n}}(\mathbf{y}_t, t; \theta)\|^2$ (standard score matching / denoising).
- Evaluation loss: $\mathcal{L}_{\text{DSM}} = \frac{1}{2}\int_{\sigma_0^2}^{\sigma_1^2} \sigma_t^{-2} \mathbb{E}\|\mathbf{n} - \hat{\mathbf{n}}\|_2^2 d\sigma_t^2$, estimated via importance sampling.
- Architecture: VDM U-Net (ResNet blocks without downsampling), conditioned on $\eta(t)$ rather than $t$.
- Trained for 0.3M iterations without data augmentation on 4 × V100 GPUs.

## Key Experimental Results

### Main Results

| Model | CIFAR-10 NLL↓ | ImageNet-32 NLL↓ | ImageNet-64 NLL↓ | ImageNet-128 NLL↓ | Training Iterations |
|------|--------------|-----------------|-----------------|------------------|---------|
| VDM | 2.65 | 3.72 | 3.40 | — | 10M |
| i-DODE | 2.56 | 3.44 | — | — | 6.2M |
| W-PCDM | 2.35 | 3.32 | 2.95 | 2.64 | 2–10M |
| Flow Matching | 2.99 | 3.53 | 3.31 | 2.90 | |
| **Ours (VP+IS)** | **2.50** | **3.01** | **2.91** | **2.59** | **0.3M** |

- ImageNet-32 NLL improves from 3.32 to **3.01** (9.3% gain).
- ImageNet-128 NLL improves from 2.64 to **2.59**.
- Training requires only **0.3M iterations**, 33× fewer than VDM's 10M.

### Ablation Study on Warm-Up Noise

| Noise Type | CIFAR-10 (Ours) | ImageNet-32 (Ours) |
|---------|-----------------|-------------------|
| Gaussian | **2.50** | **3.00** |
| Laplace | 2.51 | 3.01 |
| Logistic | 2.51 | 3.03 |
| Uniform | 2.53 | 3.09 |

Heavy-tailed exponential family noise (Gaussian, Laplace) outperforms light-tailed noise (Uniform), consistent with theoretical expectations.

### Lossless Compression

| Model | CIFAR-10 Compression Rate (bits/dim) |
|------|--------------------------|
| VDM | 2.72 |
| W-PCDM | 2.37 |
| **Ours** | **2.57** |

### Key Findings

- Increasing warm-up noise improves NLL but slightly degrades FID — indicating a trade-off between likelihood accuracy and perceptual quality.
- The gap between the ELBO and the IT bound widens as noise increases, because the pixel-independence assumption in the ELBO becomes less accurate under high noise.
- The VP schedule slightly outperforms the SP (Straight-Path) schedule; the VE schedule lags significantly behind (3.27 bits/dim).

## Highlights & Insights

- **Theoretical Contribution**: The Shannon–Fisher relationship is generalized from Gaussian to arbitrary isotropic noise (Theorem 1 + Proposition 2), unifying the score matching–MLE equivalence across diverse noise distributions.
- **Practical Value**: State-of-the-art NLL is achieved in only 0.3M iterations without data augmentation — reducing training cost by an order of magnitude.
- **Dequantization Without Extra Training**: Warm-start noise replaces traditional variational dequantization, eliminating the train-test gap with zero additional training overhead.
- **Log-SNR Parameterization**: Improved time embedding enhances the efficiency of importance sampling.

## Limitations & Future Work

- The paper focuses solely on likelihood estimation and **does not construct a generative process under non-Gaussian noise** — precluding use for sampling or generation.
- The optimal choice of dequantization strategy and variance schedule remains insufficiently explored.
- FID performance is suboptimal, as the optimization target is likelihood rather than perceptual quality.
- Validation at higher resolutions (ImageNet-256+) is absent.
- Hyperparameter and architecture tuning is constrained by available computational resources.

## Related Work & Insights

- **VDM (Kingma et al.)**: Variational diffusion model trained via the ELBO; achieves CIFAR-10 NLL of 2.65 but requires 10M iterations.
- **ScoreFlow (Song et al.)**: First work to connect score matching with MLE, but restricted to Gaussian noise.
- **i-DODE**: State-of-the-art under the IT framework, but converges slowly and requires data augmentation.
- **W-PCDM**: Cascaded diffusion model with strong NLL but more complex training.
- **Insights**: The information-theoretic perspective provides deeper understanding of diffusion model training; the warm-start technique is potentially applicable to other generative models.

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical contributions are solid (generalizing the KL–Fisher relationship to arbitrary noise), the practical value is high (SOTA NLL in 0.3M iterations), and the method is concise and compatible with standard diffusion training. The primary limitation is that the work addresses only likelihood estimation and does not extend to the sampling or generation process.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)
- [\[NeurIPS 2025\] MMG: Mutual Information Estimation via the MMSE Gap in Diffusion](mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[NeurIPS 2025\] EVODiff: Entropy-aware Variance Optimized Diffusion Inference](evodiff_entropy-aware_variance_optimized_diffusion_inference.md)

<!-- RELATED:END -->
