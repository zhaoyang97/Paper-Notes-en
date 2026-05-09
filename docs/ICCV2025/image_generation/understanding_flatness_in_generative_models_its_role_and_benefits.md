---
title: >-
  [Paper Note] Understanding Flatness in Generative Models: Its Role and Benefits
description: >-
  [ICCV 2025][Image Generation][flat minima] This paper presents the first systematic study of loss landscape flatness in generative models, particularly diffusion models. It theoretically demonstrates that flat minima enhance robustness to perturbations in the prior distribution, and empirically shows that SAM effectively promotes flatness in diffusion models, leading to improved generation quality, reduced exposure bias, and greater quantization robustness.
tags:
  - ICCV 2025
  - Image Generation
  - flat minima
  - diffusion models
  - SAM
  - generalization
  - quantization robustness
  - exposure bias
date: 2026-05-08
content_hash: d8de1814a43e7942
---

# Understanding Flatness in Generative Models: Its Role and Benefits

**Conference**: ICCV 2025
**arXiv**: [2503.11078](https://arxiv.org/abs/2503.11078)
**Area**: Diffusion Models · Image Generation
**Keywords**: flat minima, diffusion models, SAM, generalization, quantization robustness, exposure bias

## TL;DR

This paper presents the first systematic study of loss landscape flatness in generative models, particularly diffusion models. It theoretically demonstrates that flat minima enhance robustness to perturbations in the prior distribution, and empirically shows that SAM effectively promotes flatness in diffusion models, leading to improved generation quality, reduced exposure bias, and greater quantization robustness.

## Background & Motivation

Flat minima have been extensively studied in discriminative learning, where they are known to improve generalization and robustness to distribution shift. However, the role of flatness in generative models remains largely unexplored.

A central question arises: **do flat minima in diffusion models force the model to generate similar outputs regardless of input variation?** If so, this would conflict with generative diversity. If not, how does flatness influence generative modeling?

The authors find that:

1. **Diffusion models are inherently quite flat** (likely due to training across multiple noise levels), making standard SAM perturbation magnitudes nearly ineffective.
2. **Significantly stronger regularization** is required to induce meaningful flatness in diffusion models.
3. The benefits of flatness include: **improved FID**, **reduced exposure bias**, and **enhanced quantization robustness**.

## Method

### Theoretical Framework

**Definition 1 (Δ-flat minima)**: A minimum $\theta^*$ is Δ-flat if the loss remains invariant within $\|\delta\|_2 \leq \Delta$.

**Core Theorem (Theorem 1)**: A parameter perturbation $\theta + \delta$ is equivalent to a perturbation of the prior distribution:

$$\hat{p}(\mathbf{x}) = e^{-I(\mathbf{x},\delta)} p(\mathbf{x})$$

where $I(\mathbf{x},\delta) = \frac{1}{2}\mathbf{x}^\top(\delta\mathbf{W}^\top)\mathbf{x} + \mathbf{x}^\top\delta(\mathbf{U}^\top\mathbf{e}) + C$.

**Corollary 1 (Diffusion model version)**: For the noise prior $\epsilon \sim \mathcal{N}(0,\mathbf{I})$, the perturbed prior remains Gaussian: $\hat{\epsilon} = \mathcal{N}(\boldsymbol{\mu}_\delta, \Sigma_\delta)$.

**Theorem 2 (Bridging flatness and distributional robustness)**: A Δ-flat minimum achieves $\mathcal{E}$-distributional gap robustness, where:

$$\mathcal{E} \leq \frac{1}{2}\left[\sum_{i=1}^d(\sigma_i - \log\sigma_i) - d + \frac{\sigma_d}{m^2}\|\mathbf{U}^\top\mathbf{e}\|_2^2 \Delta^2\right]$$

### Two Practical Benefits

1. **Reduced exposure bias**: Flat minima suppress the loss increase caused by perturbed estimates, thereby mitigating error accumulation during iterative sampling.
2. **Quantization robustness**: Quantized parameters $\hat{\theta} = \theta + \Delta$ can be viewed as parameter perturbations; flatness ensures loss stability under such perturbations.

### Comparison of Optimization Methods

Four methods for promoting flatness are compared:
- **EMA**: Exponential moving average of weights; indirectly promotes flatness.
- **SWA**: Stochastic weight averaging; directly targets flat regions.
- **IP**: Input perturbation; promotes flatness indirectly via Lipschitz conditions.
- **SAM**: Explicitly minimizes the sharpness-aware loss $\max_{\|p\|_2 \leq \rho}[L(\theta+p) - L(\theta)]$; most effective.

## Key Experimental Results

### Main Results: FID Comparison

| Method | CIFAR-10 (20 steps) | CIFAR-10 (100 steps) | LSUN Tower (100 steps) | FFHQ (100 steps) |
|--------|--------------------|--------------------|----------------------|-----------------|
| ADM | 34.47 | 8.80 | 8.57 | 7.53 |
| +EMA | 10.63 | 4.06 | 2.49 | 6.19 |
| +SAM | 9.01 | 3.83 | 4.79 | 5.29 |
| +SAM+EMA | **7.00** | **3.18** | **2.30** | **5.04** |
| +SAM+SWA | 7.27 | **2.96** | **2.27** | 4.17 |

The SAM+EMA/SWA combinations consistently achieve the best FID across all three datasets.

### Quantization Robustness

| Method | FID (32-bit) | FID (8-bit) | Degradation |
|--------|-------------|------------|-------------|
| ADM | 34.47 | 48.02 | +13.65 |
| +SAM | 9.01 | 8.94 | **-0.07** |

Models trained with SAM exhibit virtually no FID degradation upon direct 32-bit to 8-bit quantization (−0.07 vs. +13.65), demonstrating remarkably strong quantization robustness.

### Exposure Bias Analysis

| Method | $\|\epsilon_\theta\|^2$ gap |
|--------|--------------------------|
| ADM | +11.39 |
| +SAM | +3.32 |

The $\|\epsilon\|^2$ gap of SAM-trained models is approximately one-third that of ADM, validating that flatness effectively reduces exposure bias.

## Highlights & Insights

1. This work presents the **first systematic study** of flatness in generative models, backed by theoretical guarantees.
2. Diffusion models are found to be **inherently flat** — standard SAM perturbation strengths yield negligible effects, necessitating stronger regularization.
3. The theory elegantly establishes an equivalence between **parameter-space perturbations and prior distribution perturbations**.
4. The practical benefits of flatness are concrete: improved FID, reduced exposure bias, and enhanced quantization robustness.

## Limitations & Future Work

- The theoretical analysis relies on a simplified random feature model, which may not fully reflect the behavior of deep networks in practice.
- Experiments are conducted only on unconditional DDPM; conditional generation (e.g., text-to-image) is not evaluated.
- SAM requires two gradient computations per step, approximately doubling training cost.
- Only 32-bit to 8-bit quantization is tested; finer-grained quantization analysis is absent.

## Related Work & Insights

- Flatness and generalization: SAM, SWA, and related optimizer literature.
- Exposure bias in diffusion models: Input Perturbation (IP) method.
- Model quantization: PTQ4DM, Q-Diffusion, and other diffusion model quantization techniques.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Depth | 5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| **Overall** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [\[NeurIPS 2025\] Why Diffusion Models Don't Memorize: The Role of Implicit Regularization](../../NeurIPS2025/image_generation/why_diffusion_models_dont_memorize_the_role_of_implicit_regularization.md)
- [\[NeurIPS 2025\] Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](../../NeurIPS2025/image_generation/why_diffusion_models_dont_memorize_the_role_of_implicit_dynamical_regularization.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](../../NeurIPS2025/image_generation/understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)

</div>

<!-- RELATED:END -->
