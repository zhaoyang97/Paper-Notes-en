---
title: >-
  [Paper Note] ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model
description: >-
  [NeurIPS 2025][Image Generation][Discrete Diffusion] This paper proposes ItDPDM (Information-Theoretic Discrete Poisson Diffusion Model)…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Discrete Diffusion"
  - "Poisson Process"
  - "Information Theory"
  - "Likelihood Estimation"
  - "Symbolic Music"
date: 2026-05-08
content_hash: 4b03aceedf3910c8
---

# ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model

**Conference**: NeurIPS 2025
**arXiv**: [2505.05082](https://arxiv.org/abs/2505.05082)
**Code**: Available (link provided in the paper)
**Area**: Image Generation
**Keywords**: Discrete Diffusion, Poisson Process, Information Theory, Likelihood Estimation, Symbolic Music

## TL;DR

This paper proposes ItDPDM (Information-Theoretic Discrete Poisson Diffusion Model), which achieves exact likelihood estimation for non-negative discrete data via a Poisson noise channel and a Poisson Reconstruction Loss (PRL), eliminating ELBO approximation and dequantization. The model outperforms existing discrete diffusion models in likelihood estimation on synthetic data, CIFAR-10, and MIDI music.

## Background & Motivation

**Background**: Diffusion models have achieved remarkable success in the continuous domain (images), but face fundamental limitations when handling inherently discrete data (symbolic music, count data):
- Continuous models require dequantization to map discrete data into continuous space, introducing quantization gaps and train-test mismatch.
- Existing discrete diffusion models (D3PM, LTJ) rely on the evidence lower bound (ELBO), which is not an exact likelihood estimate.

**Limitations of Prior Work**:
- Continuous DDPM fails to accurately learn the PMF of bimodal or skewed discrete distributions (e.g., missing the second mode of the NYC Taxi distribution).
- LTJ employs binomial thinning and an ELBO loss—the ELBO is not an exact likelihood, and the number of denoising steps $T$ requires careful tuning.
- ELBO introduces two layers of approximation: (1) the relaxation of the variational bound itself, and (2) Monte Carlo integration.

**Key Challenge**: Modeling discrete data requires direct manipulation of probability mass functions (PMFs), rather than indirect treatment via probability density functions (PDFs).

**Goal**: Design an information-theoretically exact diffusion model for non-negative discrete data, eliminating ELBO approximation.

**Key Insight**: Drawing an analogy to the I-MMSE identity on the Gaussian channel (derivative of mutual information = minimum mean squared error), this work establishes the I-MPRL identity on the Poisson channel (derivative of mutual information = minimum Poisson reconstruction loss).

**Core Idea**: The I-MPRL identity precisely connects the Poisson denoising loss to the data likelihood: $-\log P(x) = \int_0^\infty \text{mprl}(x, \gamma)\, d\gamma$.

## Method

### Overall Architecture

- **Poisson Noise Channel**: Given a non-negative input $x \geq 0$, the output is $z_\gamma \sim \mathcal{P}(\gamma x)$, with SNR parameter $\gamma$.
- **Forward Process**: The clean image ($\gamma \to \infty$) is progressively degraded to a black image ($\gamma \to 0$, zero photons), in sharp contrast to the white noise starting point of Gaussian diffusion.
- **Reverse Process**: Recovery from the black image is performed stepwise via $z_{\gamma_{t-1}} \sim \text{Poisson}(\gamma_{t-1} \hat{x}_0)$.

### Key Designs

#### 1. Poisson Reconstruction Loss (PRL)
- **Function**: Replaces mean squared error (MSE) as the denoising loss.
- **Core Formula**:
  $$l(x, \hat{x}) = x \log\frac{x}{\hat{x}} - x + \hat{x}$$
  This is a form of Bregman divergence and the convex conjugate of the log-moment generating function of the Poisson distribution.
- **Properties** (Lemma 1):
    - Non-negativity: $l(x, \hat{x}) \geq 0$, with equality if and only if $x = \hat{x}$.
    - Convexity: Convex in both $\hat{x}$ and $x$.
    - Asymmetric penalty: $l \to \infty$ as $\hat{x} \to 0$, appropriate for non-negative data.
    - The optimal estimator is the conditional expectation $E[X|Z_\gamma]$.
- **Design Motivation**: MSE assumes continuous output and introduces quantization error when modeling discrete PMFs; PRL directly corresponds to the mass function of the PMF.

#### 2. I-MPRL Identity (Core Theory)
- **Function**: Establishes the differential relationship between mutual information and MPRL.
- **Core Formula**:
  $$\frac{d}{d\gamma} I(x; z_\gamma) = \text{mprl}(\gamma)$$
  Integrating yields the exact likelihood:
  $$-\log P(x) = \int_0^\infty \text{mprl}(x, \gamma)\, d\gamma$$
- **Comparison with Gaussian I-MMSE**: $\frac{d}{d\gamma} I(x; z_\gamma) = \frac{1}{2}\text{mmse}(\gamma)$. The Poisson counterpart in ItDPDM is structurally analogous.
- **Significance**: Eliminates the first layer of approximation in ELBO—the relationship between the PRL loss and the true likelihood is exact (not variational).

#### 3. NLL Upper Bound and Importance Sampling
- **NLL Upper Bound**:
  $$E[-\log P(x)] = \int_0^\infty \text{mprl}(\gamma)\, d\gamma \leq \int_0^\infty \mathbb{E}[l(X, \hat{X})]\, d\gamma$$
  The looseness of the bound arises solely from the suboptimality of the denoiser $\hat{X}$ (not from a variational approximation).
- **log-SNR Parameterization**: $\alpha = \log \gamma$; the integral becomes $\int_{-\infty}^\infty e^\alpha \text{mprl}(\alpha)\, d\alpha$.
- **Tail Bounds**: Analytically derived upper bounds for the integral outside $[\alpha_0, \alpha_1]$ ensure that truncation errors are controllable.
- **Importance Sampling**: The Logistic distribution is used to sample $\alpha$, yielding MC estimation error of $O(1/\sqrt{n})$.

#### 4. Input Normalization
- Poisson noise is non-additive ($z_\gamma \sim \mathcal{P}(\gamma x)$ rather than $z_\gamma = x + n$), and both the mean and variance grow with $\gamma$.
- Normalization: $\tilde{Z}_\gamma = Z_\gamma / (1 + \gamma)$, ensuring inputs remain in the range $[0, X]$.

### Loss & Training

- Training loss: $L = \sum_{i \in B} \text{PRL}(x_i, \hat{x}_i) / q(\alpha)$, where $q(\alpha)$ is the importance sampling weight.
- Denoiser: Conditional MLP (synthetic data) or U-Net/ConvTransformer (real data).
- Sampling: Starting from $z_{\gamma_T} = \mathbf{0}$ (black image), SNR is progressively increased during sampling.

## Key Experimental Results

### Synthetic Discrete Data (Wasserstein-1 Distance ↓)

| Distribution | DDPM | LTJ | **ItDPDM** |
|---|---|---|---|
| PoissMix | 3.76 | 1.21 | **0.99** |
| ZIP | 2.31 | 0.69 | **0.56** |
| NBinomMix | 4.89 | **1.15** | 1.39 |
| Zipf | 1.51 | 0.73 | **0.48** |
| Yule-Simon | 0.32 | 0.17 | **0.14** |

ItDPDM achieves the best Wasserstein distance on 4 out of 6 discrete distributions, with notable advantages on Zipf (heavy-tailed) and PoissMix.

### Real Data NLL (↓ lower is better)

**CIFAR-10 Subset**:

| Noise + Loss | DDPM backbone | IDDPM backbone |
|---|:---:|:---:|
| Gaussian + MSE | 0.44 | 0.48 |
| Gaussian + PRL | 0.27 | 0.32 |
| Poisson + MSE | 0.23 | 0.22 |
| **Poisson + PRL (ItDPDM)** | **0.18** | **0.17** |

**Lakh MIDI (Symbolic Music)**: ItDPDM NLL = **4.61×10⁻⁵** vs. Gaussian+MSE NLL = 0.51.

### Generation Quality

| Method | FID (dB↓) | SSIM ↑ | FAD ↓ | Consistency ↑ |
|---|---|---|---|---|
| DDPM | **0** | **0.93** | 0.89 | 0.91 |
| LTJ | 0.30 | 0.90 | 0.66 | 0.92 |
| D3PM | 2.93 | 0.86 | **0.61** | **0.98** |
| **ItDPDM** | 0.18 | 0.91 | 0.64 | 0.94 |

### Ablation Study

| Setting | NLL (DDPM) |
|---|---|
| Gaussian + MSE | 0.44 |
| Gaussian + PRL | 0.27 |
| Poisson + MSE | 0.23 |
| **Poisson + PRL** | **0.18** |

PRL also outperforms MSE under Gaussian noise (0.27 vs. 0.44), demonstrating the independent value of PRL for discrete data.

### Key Findings

- The I-MPRL identity enables exact (non-variational) likelihood estimation, eliminating the theoretical relaxation inherent in ELBO.
- PRL training converges faster and achieves lower loss in the low-SNR regime.
- Poisson diffusion generalizes well to non-Poisson distributions (Zipf, Yule-Simon), and is not restricted to Poisson-type data.
- MC estimation error is on the order of $10^{-2}$ for $n > 1000$.

## Highlights & Insights

- **Exact Likelihood**: The I-MPRL identity establishes an exact equality (neither an upper nor lower bound) between the PRL loss and the true NLL—a fundamental advantage over all ELBO-based methods.
- **Natural Choice for Non-Negative Integers**: Poisson noise is the natural perturbation for count data, avoiding any mapping to a continuous space.
- **Information-Theoretic Elegance**: The I-MPRL identity on the Poisson channel is a perfect dual of the I-MMSE identity on the Gaussian channel.
- **Generality of PRL**: Even when paired with Gaussian noise, PRL outperforms MSE for discrete data.

## Limitations & Future Work

- **Proof-of-Concept Stage**: Due to computational constraints, state-of-the-art generation quality (FID) on CIFAR-10 and MIDI has not yet been achieved.
- Only unconditional generation is evaluated; conditional generation and out-of-distribution generalization remain unexplored.
- Logistic sampling parameters are fixed without hyperparameter search.
- Validation on large-scale, high-resolution datasets is absent.
- Poisson noise is non-additive and non-source-separable, making reverse denoising more complex than in the Gaussian case.

## Related Work & Insights

- **IT Gaussian Diffusion (Kong et al., 2023)**: Uses the I-MMSE identity for information-theoretic likelihood estimation on continuous data. ItDPDM is its discrete Poisson dual.
- **LTJ (Chen & Zhou, 2023)**: A discrete diffusion model based on binomial thinning and ELBO, which is irreversible and relies on variational approximation.
- **D3PM (Austin et al., 2021)**: A discrete state-space diffusion model using masking/uniform transition matrices.
- **Blackout Diffusion / Beta Diffusion**: Employ irreversible priors without tractable likelihoods.
- **Insights**: Information-theoretic tools (mutual information, channel coding) provide a powerful analytical framework for generative models, with potential for extension to broader discrete domains.

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical contribution is outstanding—the I-MPRL identity achieves exact likelihood estimation, and the method forms an elegant dual to Gaussian diffusion. The synthetic experiments provide thorough validation. The primary limitations are the restricted generation quality and scalability on real-world data, which the authors explicitly acknowledge as a proof-of-concept. Future work with larger models and longer training schedules is expected to yield substantial improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [\[NeurIPS 2025\] MMG: Mutual Information Estimation via the MMSE Gap in Diffusion](mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)

</div>

<!-- RELATED:END -->
