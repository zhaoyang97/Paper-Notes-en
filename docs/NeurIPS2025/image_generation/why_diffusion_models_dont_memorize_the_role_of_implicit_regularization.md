---
title: >-
  [Paper Note] Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training
description: >-
  [NeurIPS 2025][Image Generation][diffusion-model] This paper reveals, through both numerical experiments and theoretical analysis, an **implicit dynamical regularization** mechanism in diffusion model training: the gap between the timescale for generating high-quality samples $\tau_\text{gen}$ and the timescale for memorization $\tau_\text{mem}$ grows linearly with training set size $n$, providing theoretical justification for early stopping.
tags:
  - NeurIPS 2025
  - Image Generation
  - diffusion-model
  - memorization
  - generalization
  - implicit-regularization
  - score-matching
date: 2026-05-08
content_hash: a90d7b12d5372dda
---

# Why Diffusion Models Don't Memorize: The Role of Implicit Regularization

**Conference**: NeurIPS 2025
**arXiv**: [2505.17638](https://arxiv.org/abs/2505.17638)
**Code**: None
**Area**: Image Generation
**Keywords**: diffusion-model, memorization, generalization, implicit-regularization, score-matching
# Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training

## TL;DR

This paper reveals, through both numerical experiments and theoretical analysis, an **implicit dynamical regularization** mechanism in diffusion model training: the gap between the timescale for generating high-quality samples $\tau_\text{gen}$ and the timescale for memorization $\tau_\text{mem}$ grows linearly with training set size $n$, providing theoretical justification for early stopping.

## Background & Motivation

1. **Empirical success of diffusion models**: Diffusion models (DMs) achieve state-of-the-art performance in image, audio, and video generation, yet the mechanism by which they avoid memorizing training data under heavily overparameterized settings remains poorly understood.
2. **Memorization nature of the empirical score**: Theoretically, if a model perfectly learns the empirical score, the generation process will exactly reproduce training samples; avoiding this requires $n$ to grow exponentially with dimension $d$, a condition far from satisfied in practice.
3. **Insufficiency of existing regularization explanations**: Architectural bias, limited parameter capacity, and finite learning rate have been shown to suppress memorization, yet the memorization–generalization transition consistently appears even when all these factors are present, suggesting the core mechanism lies elsewhere.
4. **Insight from spectral bias**: Deep networks tend to learn low-frequency functions first, and the empirical score contains a low-frequency component close to the population score and a dataset-dependent high-frequency component at low noise levels; this frequency dependence has not been systematically exploited.
5. **Need for analytically tractable theoretical models**: Prior theoretical work on memorization has focused primarily on asymptotic behavior under infinite training time, without systematically characterizing the emergence and separation of two timescales during training dynamics.
6. **Practical motivation**: Understanding when early stopping can safely prevent memorization has direct implications for model training in data-scarce scenarios.

## Method

### Core Finding: Separation of Two Timescales

The authors identify two critical timescales during training:

- **$\tau_\text{gen}$ (generalization timescale)**: The time at which the model begins generating high-quality samples, approximately 100K SGD steps, **independent of training set size $n$**.
- **$\tau_\text{mem}$ (memorization timescale)**: The time at which the model begins memorizing training data, which **grows linearly with $n$** ($\tau_\text{mem} \propto n$).

A "generalization window" $[\tau_\text{gen}, \tau_\text{mem}]$ exists between the two timescales; early stopping within this interval yields high-quality, non-memorizing generations. This window expands linearly with $n$.

### Numerical Experiment Design

- **Dataset**: CelebA grayscale $32\times32$ images, training set size $n$ ranging from 128 to 32768.
- **Model**: U-Net architecture (DDPM), three resolution levels, base width $W\in\{8,16,32,48,64\}$, parameter counts $p\in\{0.26\text{M}, 1\text{M}, 4\text{M}, 9\text{M}, 16\text{M}\}$.
- **Optimizer**: SGD with momentum, fixed batch size $\min(n, 512)$.
- **Evaluation metrics**: FID for generation quality; memorization score $f_\text{mem}$ based on nearest-neighbor ratio (threshold $k=1/3$); train/test loss at fixed diffusion time $t=0.01$.

### Effect of Model Capacity

- Larger $W$ (more parameters) advances both $\tau_\text{gen}$ and $\tau_\text{mem}$, while the scaling relationship is preserved: $\tau_\text{gen} \propto W^{-1}$, $\tau_\text{mem} \propto nW^{-1}$.
- A phase diagram is constructed in the $(n, p)$ plane: the generalization window opens when $n > n_\text{gm}(p)$; when $n > n^*(p)$, the model lacks the capacity to memorize even under infinite training (architectural regularization regime).

### Theoretical Analysis: Random Feature Neural Networks (RFNN)

The score function is parameterized as a two-layer random feature network $s_A(x) = \frac{A}{\sqrt{p}}\sigma\!\left(\frac{Wx}{\sqrt{d}}\right)$, where $W$ is fixed and $A$ is learned.

In the high-dimensional limit $d, p, n \to \infty$ (with $\psi_p = p/d$ and $\psi_n = n/d$ fixed):

- **Training dynamics are linear**: The gradient flow equations can be solved exactly; timescales are determined by the eigenvalues of matrix $U$.
- **Theorem 3.1**: The Stieltjes transform equation for the spectral density $\rho(\lambda)$ of $U$ is derived via the replica method.
- **Theorem 3.2**: In the overparameterized regime $\psi_p > \psi_n \gg 1$, the spectrum splits into two separated bulks:
  - **$\rho_2$** (high-eigenvalue bulk): corresponds to $\tau_\text{gen}$, independent of the specific training set, determined by the population covariance $\Sigma$;
  - **$\rho_1$** (low-eigenvalue bulk): corresponds to $\tau_\text{mem}$, scales as $\psi_p/\psi_n$, causing $\tau_\text{mem} \propto n$ as $n$ increases.
  - An additional $\delta$-peak $\delta(\lambda - s_t^2)$ (weight $1 - (1+\psi_n)/\psi_p$) whose eigenvectors do not affect generation quality.

### Three-Region Phase Diagram

Three regions are delineated in the $(n, p)$ plane:
1. **Memorization regime**: Small $n$; the model begins memorizing at $\tau_\text{gen}$.
2. **Dynamical regularization regime**: Moderate $n$; early stopping in $[\tau_\text{gen}, \tau_\text{mem}]$ yields generalization.
3. **Architectural regularization regime**: $n > n^*(p)$; the model lacks sufficient expressivity to memorize even under infinite training.

## Key Experimental Results

### Table 1: Key Metrics at Different Training Set Sizes (U-Net on CelebA, $W=32$)

| Training set size $n$ | Best FID (↓) | $\tau_\text{gen}$ (K steps) | $\tau_\text{mem}/n$ (rescaled) | $f_\text{mem}(\tau_\text{max})$ |
|---|---|---|---|---|
| 128 | ~60 | ~100K | ~300 | High |
| 512 | ~35 | ~100K | ~300 | Medium |
| 1024 | ~25 | ~100K | ~300 | Low |
| 4096 | ~18 | ~100K | ~300 | Very low |
| 32768 | ~15 | ~100K | ~300 | ~0 |

**Key observation**: The normalized memorization curves for all $n$ collapse at $\tau/n \approx 300$, confirming $\tau_\text{mem} \propto n$.

### Table 2: Timescale Scaling at Different Model Widths ($n=1024$)

| Base width $W$ | Parameters $p$ (M) | $W\tau_\text{gen}$ (rescaled) | $\tau_\text{mem}$ scaling |
|---|---|---|---|
| 8 | 0.26 | ~$3\times10^6$ | $\propto nW^{-1}$ |
| 16 | 1 | ~$3\times10^6$ | $\propto nW^{-1}$ |
| 32 | 4 | ~$3\times10^6$ | $\propto nW^{-1}$ |
| 48 | 9 | ~$3\times10^6$ | $\propto nW^{-1}$ |
| 64 | 16 | ~$3\times10^6$ | $\propto nW^{-1}$ |

**Key observation**: The collapse of $W\tau_\text{gen} \approx 3\times10^6$ confirms $\tau_\text{gen} \propto W^{-1}$, independent of $n$.

## Highlights & Insights

- **A clear theoretical picture of implicit regularization in training dynamics**: The separation of two timescales provides rigorous justification for early stopping and reveals the core mechanism underlying generalization in diffusion models.
- **Dual validation through theory and experiment**: The same phenomenon is confirmed on both a realistic U-Net with CelebA and an exactly solvable random feature model, strengthening the credibility of the conclusions.
- **Physical intuition from spectral analysis**: Using the replica method and random matrix theory, generation and memorization are mapped to two separated spectral bulks, providing an elegant and actionable theoretical framework.
- **Practical value of the phase diagram**: The $(n, p)$ phase diagram can guide practitioners in selecting safe training durations based on dataset size and model scale.

## Limitations & Future Work

- **SGD optimizer only**: Practical diffusion models commonly use Adam; while the appendix demonstrates that the two timescales persist under Adam, the precise scaling relations may differ.
- **Unconditional generation setting**: Main experiments focus on unconditional DDPM; the specific dependence of $\tau_\text{gen}$ and $\tau_\text{mem}$ under conditional generation (e.g., classifier-free guidance) remains an open question.
- **Limited parameter range**: Numerical experiments cover 1M–16M parameters; larger models (e.g., >100M) are not explored, precluding a complete $(n, p)$ phase diagram.
- **Simplifying assumptions in the theoretical model**: Assumptions such as a fixed first layer in the RFNN and Gaussian data distribution differ from practical U-Net architectures and real image data; the theoretical analysis is also restricted to a fixed diffusion time $t$.

## Related Work & Insights

| Dimension | Ours | Biroli et al. (2024) / George et al. (2025) |
|---|---|---|
| **Focus** | Implicit regularization in training dynamics and the two timescales | Asymptotic behavior of the memorization–generalization transition in diffusion models |
| **Core contribution** | Discovery that $\tau_\text{gen}$ is independent of $n$ and $\tau_\text{mem} \propto n$, with spectral theory explanation | Computing train/test loss at $\tau\to\infty$ in the RFNN framework |
| **Methodological difference** | Focuses on finite-time training dynamics and emphasizes the role of early stopping | Primarily concerns asymptotic behavior under infinite training time |
| **Scope** | Covers both practical U-Net and theoretical RFNN | Mainly theoretical analysis |

| Dimension | Ours | Li et al. (2024) / Zhang et al. (2023) |
|---|---|---|
| **Focus** | Memorization avoidance via training dynamics | Architectural bias and network capacity constraints on memorization |
| **Mechanism type** | Implicit dynamical regularization (timescale separation) | Architectural regularization (limited expressivity) |
| **Applicable regime** | Intermediate region of the $(n, p)$ phase diagram | Large-data region $n > n^*(p)$ of the $(n, p)$ phase diagram |
| **Complementarity** | The two are complementary: architectural regularization defines $n^*(p)$; dynamical regularization enables memorization avoidance via early stopping when $n < n^*(p)$ | Does not account for time-dependent behavior during training dynamics |

## Rating

| Dimension | Score | Remarks |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | First systematic characterization of the separation between generalization and memorization timescales in diffusion model training |
| Theoretical Depth | ⭐⭐⭐⭐⭐ | Replica method combined with random matrix theory provides rigorous spectral analysis; Theorems 3.1/3.2 yield exact analytical results |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Systematic experiments across multiple $n$ and $p$ on CelebA, but limited to low-resolution grayscale images and relatively small models |
| Value | ⭐⭐⭐⭐ | Phase diagram and scaling laws directly inform early stopping strategies, particularly useful in data-scarce scenarios |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](why_diffusion_models_dont_memorize_the_role_of_implicit_dynamical_regularization.md)
- [\[NeurIPS 2025\] A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)
- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)
- [\[NeurIPS 2025\] Moment- and Power-Spectrum-Based Gaussianity Regularization for Text-to-Image Models](moment-_and_power-spectrum-based_gaussianity_regularization_for_text-to-image_mo.md)
- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)

<!-- RELATED:END -->
