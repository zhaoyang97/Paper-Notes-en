---
title: >-
  [Paper Note] Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training
description: >-
  [NeurIPS 2025][Image Generation][Diffusion models] Through numerical experiments and theoretical analysis, this paper identifies two critical timescales in diffusion model training — a generalization time $\tau_{\text{ge…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion models"
  - "memorization"
  - "generalization"
  - "implicit regularization"
  - "training dynamics"
  - "random features"
  - "early stopping"
date: 2026-05-08
content_hash: 5994a2cf54639aa2
---

# Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training

**Conference**: NeurIPS 2025 Oral  
**arXiv**: [2505.17638](https://arxiv.org/abs/2505.17638)  
**Authors**: Tony Bonnaire, Raphaël Urfin, Giulio Biroli, Marc Mézard (LPENS/PSL Paris, Bocconi University)
**Code**: Not released  
**Area**: Image Generation
**Keywords**: Diffusion models, memorization, generalization, implicit regularization, training dynamics, random features, early stopping

## TL;DR

Through numerical experiments and theoretical analysis, this paper identifies two critical timescales in diffusion model training — a generalization time $\tau_{\text{gen}}$ and a memorization time $\tau_{\text{mem}}$ — where the latter scales linearly with training set size $n$ while the former remains constant. The resulting implicit dynamical regularization enables early stopping to prevent memorization even in heavily overparameterized regimes.

## Background & Motivation

Diffusion models have achieved state-of-the-art performance across diverse generative tasks including images, audio, and video, yet a fundamental question remains insufficiently understood: **why do heavily overparameterized diffusion models not memorize training data?**

- **Theoretical limit of the empirical score**: If a model perfectly learns the empirical score, it would in theory reproduce training samples during generation, unless $n$ grows exponentially in the dimension $d$.
- **Contradiction with empirical observations**: In practice, memorization only occurs when $n$ is small and disappears well before $n$ reaches exponential scale.
- **Incompleteness of existing explanations**: Architectural inductive biases and dynamical regularization from finite learning rates have been shown to play a role, but none of these mechanisms fully explains the core driver of the memorization-to-generalization transition.

The central hypothesis is that **training dynamics itself exhibits an implicit bias** toward solutions with better generalization, and this bias is directly linked to training set size.

## Method

### Core Finding: Two Timescales

Through systematic experiments, the authors identify three phases during training:

1. **Underfitting phase** ($\tau < \tau_{\text{gen}}$): The model has not yet learned sufficient information; generation quality is poor.
2. **Generalization phase** ($\tau_{\text{gen}} \leq \tau \leq \tau_{\text{mem}}$): The model generates high-quality and diverse novel samples.
3. **Memorization phase** ($\tau > \tau_{\text{mem}}$): The model begins reproducing training samples.

Key quantitative laws:
- $\tau_{\text{gen}}$ is **independent of** training set size $n$, depending only on model capacity.
- $\tau_{\text{mem}} \propto n$, scaling linearly with training set size.
- The generalization window $[\tau_{\text{gen}}, \tau_{\text{mem}}]$ expands linearly with $n$.

### Experimental Setup (U-Net on CelebA)

- **Dataset**: CelebA grayscale images downsampled to $32 \times 32$; $n$ ranging from 128 to 32768.
- **Model**: U-Net architecture with three resolution levels; base channel width $W \in \{8, 16, 32, 48, 64\}$; parameter count $p \in \{0.26, 1, 4, 9, 16\} \times 10^6$.
- **Training**: SGD with momentum, DDPM framework, batch size $\min(n, 512)$.
- **Evaluation**: FID (generation quality) + memorization fraction $f_{\text{mem}}$ (based on the nearest-neighbor/second-nearest-neighbor distance ratio between generated samples and the training set).

### Memorization Criterion

A generated sample $\mathbf{x}_\tau$ is classified as memorized when:

$$\mathbb{E}_{\mathbf{x}_\tau}\left[\frac{\|\mathbf{x}_\tau - \mathbf{a}^{\mu_1}\|_2}{\|\mathbf{x}_\tau - \mathbf{a}^{\mu_2}\|_2}\right] < k = \frac{1}{3}$$

where $\mathbf{a}^{\mu_1}$ and $\mathbf{a}^{\mu_2}$ denote the nearest and second-nearest neighbors in the training set, respectively.

### Theoretical Analysis: Random Feature Neural Networks (RFNN)

To establish a tractable theoretical framework, the score function is parameterized using a random feature neural network:

$$\mathbf{s}_{\mathbf{A}}(\mathbf{x}) = \frac{\mathbf{A}}{\sqrt{p}} \sigma\left(\frac{\mathbf{W}\mathbf{x}}{\sqrt{d}}\right)$$

where the first-layer weights $\mathbf{W}$ are frozen and only the second layer $\mathbf{A}$ is trained. In the high-dimensional limit $d, p, n \to \infty$ (with $\psi_p = p/d$ and $\psi_n = n/d$ held fixed):

- The timescales of training dynamics are determined by the eigenvalue spectrum of matrix $\mathbf{U}$.
- **Theorem 3.2** (core theoretical result): In the overparameterized regime ($\psi_p > \psi_n \gg 1$), the eigenvalue spectrum of $\mathbf{U}$ decomposes into two well-separated components:
    - $\rho_2$ (large-eigenvalue bulk): corresponds to the generalization timescale, of order $\psi_p$, independent of $\psi_n$.
    - $\rho_1$ (small-eigenvalue bulk): corresponds to the memorization timescale, of order $\psi_p / \psi_n$.
    - From this, the paper derives $\tau_{\text{mem}} \propto \psi_n / \Delta_t \propto n$.

### Phase Diagram Analysis

Three regions are identified in the $(n, p)$ plane:
1. **Memorization region**: When $n$ is sufficiently small, the model memorizes at $\tau_{\text{gen}}$.
2. **Dynamical regularization region**: $n_{\text{gm}}(p) < n < n^*(p)$; generalization is achieved via early stopping.
3. **Architectural regularization region**: $n > n^*(p)$; the model lacks sufficient capacity to memorize even as $\tau \to \infty$.

## Key Experimental Results

### Table/Figure 2: Effect of Training Set Size on Memorization (U-Net, $p = 4 \times 10^6$, $W = 32$)

| Training set $n$ | $\tau_{\text{gen}}$ (SGD steps) | Onset of $f_{\text{mem}}$ | Collapse after $\tau/n$ normalization |
|---|---|---|---|
| 128 | ~100K | ~30K | Yes |
| 256 | ~100K | ~80K | Yes |
| 512 | ~100K | ~150K | Yes |
| 1024 | ~100K | ~300K | Yes |
| 2048 | ~100K | ~600K | Yes |
| 4096 | ~100K | ~1.2M | Yes |
| 32768 | ~100K | >11M (not observed) | — |

Key findings:
- **$\tau_{\text{gen}} \approx 100$K remains constant across all $n$.**
- The normalized memorization fraction $f_{\text{mem}}(\tau)/f_{\text{mem}}(\tau_{\max})$ rises uniformly at $\tau/n \approx 300$, **confirming $\tau_{\text{mem}} \propto n$**.
- At $n = 32768$, the test loss remains close to the training loss even after 11M training steps.

### Table/Figure 3: Effect of Model Capacity (U-Net with varying width $W$)

| Width $W$ | Parameters $p$ | Scaling of $\tau_{\text{gen}}$ | Scaling of $\tau_{\text{mem}}$ |
|---|---|---|---|
| 8 | 0.26M | baseline | baseline |
| 16 | 1M | $\propto W^{-1}$ | $\propto nW^{-1}$ |
| 32 | 4M | $\propto W^{-1}$ | $\propto nW^{-1}$ |
| 48 | 9M | $\propto W^{-1}$ | $\propto nW^{-1}$ |
| 64 | 16M | $\propto W^{-1}$ | $\propto nW^{-1}$ |

Key findings:
- Larger-capacity networks reach both generalization and memorization faster, but the **ratio $\tau_{\text{mem}}/\tau_{\text{gen}}$ still scales linearly with $n$**.
- The critical training set size $n_{\text{gm}}(p)$ (where $\tau_{\text{mem}} = \tau_{\text{gen}}$) is approximately independent of $p$ for $W > 8$.
- The $(n, p)$ phase diagram clearly shows the dynamical regularization region expanding as $\tau$ increases.

### Theoretical Validation: RFNN Model

- On the RFNN, the separation between training and test loss also scales linearly with $\psi_n$.
- The optimal score error $\mathcal{E}_{\text{score}}$ in the generalization phase decays at rate $\mathcal{O}(\psi_n^{-0.59})$.
- Heatmaps of generalization loss $\mathcal{L}_{\text{gen}}$ in the $(n, p)$ plane at different $\tau$ exhibit phase transition behavior consistent with the U-Net results.

## Highlights & Insights

- **A unified theoretical picture**: This work is the first to attribute the memorization-to-generalization transition in diffusion models to the separation of two timescales in training dynamics, establishing a complete chain from empirical observation to theory.
- **Concise scaling laws**: $\tau_{\text{gen}} = \text{const}$ and $\tau_{\text{mem}} \propto n$ — remarkably simple and powerful quantitative laws that directly inform practical early stopping strategies.
- **Physical intuition**: The empirical score is highly irregular (high-frequency) at low noise levels, while the spectral bias of neural networks causes them to learn low-frequency components first (generalization) before fitting high-frequency ones (memorization), consistent with the spectral bias literature.
- **Ruling out simple explanations**: Full-batch experiments confirm that delayed memorization does not arise from differences in sample repetition frequency, but from an intrinsic $n$-dependence of the loss landscape.
- **Clear phase diagram**: Three distinct regions in the $(n, p)$ plane provide a global perspective for model design and training.

## Limitations & Future Work

- **Only SGD validated**: Although the appendix shows similar behavior for Adam, the effect of different optimizers on the absolute values of the two timescales is not systematically studied.
- **Primarily unconditional generation**: The main experiments are based on unconditional diffusion models; conditional generation (e.g., classifier-free guidance) is only briefly validated on synthetic data.
- **Limited parameter range**: $p$ covers only 1M–16M, insufficient to map a complete large-scale $(n, p)$ phase diagram.
- **Simplified theoretical model**: The RFNN model differs substantially from practical U-Nets (single trainable layer, fixed diffusion time $t$); theoretical extensions to richer data distributions and architectures remain open problems.
- **Low-resolution experiments**: CelebA at $32 \times 32$ grayscale is far from the scale of industrial diffusion models.

## Related Work & Insights

- **Empirical studies of memorization**: Carlini et al. (2023) demonstrate that Stable Diffusion/DALL-E can reproduce training data; Gu et al. (2023) and Chen et al. (2024) study the relationship between memorization and data distribution/model configuration.
- **High-dimensional theoretical analysis**: Biroli et al. (2024) analyze dynamic regimes under the empirical score; Cui et al. (2024, 2025) and George et al. (2025) study asymptotic properties of score learning under different model classes.
- **Architectural regularization**: Kadkhodaie et al. (2024) explain generalization through geometrically adaptive harmonic representations; Li et al. (2024) show that finite capacity limits memorization.
- **Spectral bias**: Rahaman et al. (2019) and others find that deep networks preferentially learn low-frequency functions; this paper connects this phenomenon to the generalization-memorization separation in score learning.
- **Positioning of this work**: Between existing architectural and learning-rate regularization, this paper reveals the important role of training dynamics itself as a third regularization mechanism.

## Rating

- Novelty: ⭐⭐⭐⭐ — The separation of two timescales and their scaling laws are novel and insightful findings, though the broad idea of using early stopping to prevent memorization is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual validation with U-Net and RFNN, systematic variation of $n$ and $p$, but validated only on low-resolution grayscale images.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theory and experiments are tightly integrated; phase diagram visualizations are intuitively clear; notation is consistent throughout.
- Value: ⭐⭐⭐⭐ — Provides theoretical grounding for early stopping in diffusion model training, with practical implications for data-scarce domains such as scientific data generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Understanding Flatness in Generative Models: Its Role and Benefits](../../ICCV2025/image_generation/understanding_flatness_in_generative_models_its_role_and_benefits.md)
- [\[NeurIPS 2025\] Moment- and Power-Spectrum-Based Gaussianity Regularization for Text-to-Image Models](moment-_and_power-spectrum-based_gaussianity_regularization_for_text-to-image_mo.md)
- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[NeurIPS 2025\] Enhancing Diffusion Model Guidance through Calibration and Regularization](enhancing_diffusion_model_guidance_through_calibration_and_regularization.md)

</div>

<!-- RELATED:END -->
