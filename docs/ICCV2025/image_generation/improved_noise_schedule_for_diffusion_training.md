---
title: >-
  [Paper Note] Improved Noise Schedule for Diffusion Training
description: >-
  [ICCV 2025][Image Generation][Noise schedule] This paper proposes a unified framework for analyzing and designing noise schedules in diffusion models from a probability distribution perspective. It finds that a Laplace n…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Noise schedule"
  - "diffusion model training"
  - "importance sampling"
  - "Laplace distribution"
  - "logSNR"
date: 2026-05-08
content_hash: 54d9203a7044671b
---

# Improved Noise Schedule for Diffusion Training

**Conference**: ICCV 2025
**arXiv**: [2407.03297](https://arxiv.org/abs/2407.03297)
**Code**: None
**Area**: Diffusion Models / Image Generation
**Keywords**: Noise schedule, diffusion model training, importance sampling, Laplace distribution, logSNR

## TL;DR

This paper proposes a unified framework for analyzing and designing noise schedules in diffusion models from a probability distribution perspective. It finds that a Laplace noise schedule—which concentrates sampling probability near $\log\text{SNR}=0$ (the signal–noise transition point)—improves FID by 26.6% over the standard cosine schedule under the same training budget, outperforming all loss-weighting adjustment methods.

## Background & Motivation

### Problem Definition

Diffusion models must learn denoising across different noise levels during training, yet the fundamental question of *where to allocate computational resources across noise levels* has long been neglected. Timesteps are typically sampled uniformly as $t \sim \mathcal{U}[0,1]$, which implicitly induces a non-uniform distribution over noise intensities.

### Limitations of Prior Work

**Architectural improvements**: Methods such as DiT's AdaLN, MM-DiT's modality-separated weights, and U-shaped skip connections are effective but do not address the noise schedule itself.

**Loss-weighting adjustments**: Methods such as Min-SNR, Soft-Min-SNR, and P2 accelerate convergence by reweighting the loss at different noise levels, but are fundamentally equivalent to importance sampling over noise intensity. In practice, directly modifying the sampling distribution $p(\lambda)$ is more effective than scaling the loss weight $w(\lambda)$.

**Ad hoc noise schedule design**: Schedules such as linear, cosine, and EDM lack a unified theoretical foundation, and their selection is often heuristic.

### Core Motivation

**Key insight**: Within the unified diffusion training objective, adjusting the loss weight $w(\lambda)$ and modifying the noise sampling distribution $p(\lambda)$ are theoretically equivalent. However, under a fixed computational budget, directly concentrating compute (FLOPs) on intermediate noise levels ($\log\text{SNR} \approx 0$) is more efficient than upweighting the loss in that region. This implies that noise schedule design is fundamentally a **probability distribution design problem**.

## Method

### Overall Architecture

The noise schedule is reformulated as a probability distribution $p(\lambda)$ over $\log\text{SNR}$. New noise schedules are derived by selecting different distribution families, directing the model's focus toward the critical transition point between signal and noise.

### Key Designs

#### 1. **Unified Framework for Noise Schedules from a Probabilistic Perspective**

- **Function**: Establishes a bidirectional mapping between a noise schedule $\lambda(t)$ and a sampling distribution $p(\lambda)$ over noise intensity.
- **Mechanism**: When $t$ is drawn from a uniform distribution, the sampling probability of noise intensity $\lambda = \log\text{SNR}$ is:
  $$p(\lambda) = -\frac{dt}{d\lambda}$$
  Conversely, any probability distribution $p(\lambda)$ induces a noise schedule:
  $$t = 1 - \int_{-\infty}^{\lambda} p(\lambda)\, d\lambda = \mathcal{P}(\lambda), \quad \lambda = \mathcal{P}^{-1}(t)$$
  where $\mathcal{P}(\lambda)$ is the cumulative distribution function of $\lambda$.
- **Design Motivation**: This framework reframes noise schedule design as a distribution design problem—rather than specifying how noise evolves over time, one optimizes how sampling resources are allocated across noise intensities.

#### 2. **Laplace Noise Schedule**

- **Function**: Proposes a new noise schedule using the Laplace distribution as $p(\lambda)$.
- **Mechanism**: The probability density function of the Laplace distribution is:
  $$p(\lambda) = \frac{1}{2b} \exp\left(-\frac{|\lambda - \mu|}{b}\right)$$
  The corresponding noise schedule is:
  $$\lambda = \mu - b \cdot \text{sgn}(0.5 - t) \cdot \ln(1 - 2|t - 0.5|)$$
  Default parameters are $\mu = 0$ (centered at $\log\text{SNR} = 0$) and $b = 0.5$ (sharp peak to concentrate more sampling at intermediate noise levels).
- **Design Motivation**: The Laplace distribution has a simple exponential decay and symmetric form; its sharp peak at $\lambda = 0$ corresponds precisely to the signal–noise equilibrium point. Experiments confirm this critical region is the most important for diffusion model training.

#### 3. **Unified Training Objective and Practical Setup**

- **Function**: Analyzes, within the VDM++ unified training framework, why modifying $p(\lambda)$ is superior to modifying $w(\lambda)$.
- **Mechanism**: The unified training loss is:
  $$\mathcal{L}_w(\theta) = \frac{1}{2} \mathbb{E}_{\mathbf{x},\boldsymbol{\epsilon},\lambda\sim p(\lambda)} \left[\frac{w(\lambda)}{p(\lambda)} \|\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_\lambda;\lambda) - \boldsymbol{\epsilon}\|_2^2\right]$$
  Although adjusting $w(\lambda)$ and $p(\lambda)$ are theoretically equivalent, modifying $p(\lambda)$ directly allocates more forward and backward computation to critical noise levels, whereas adjusting $w(\lambda)$ only rescales gradient magnitudes without increasing compute in that region.
- **Design Motivation**: Under a fixed compute budget, *where* computation is spent matters more than *how gradients are weighted*. This explains why loss-weighting methods such as Min-SNR are effective but inferior to directly modifying the noise schedule.

### Loss & Training

- Standard MSE loss supporting $\epsilon$-, $\mathbf{x}_0$-, and $\mathbf{v}$-prediction targets.
- DDIM with 50 steps at inference; sampling SNR is aligned to the cosine schedule for fair comparison.
- For the Laplace schedule, inference begins from $t_{\max} = 0.99$ due to insufficient training at extreme noise levels.
- 500K iterations, batch size 256, 8×V100 GPUs.

## Key Experimental Results

### Main Results

**Comparison of noise schedules and loss-weighting methods on ImageNet-256 (FID-10K)**:

| Method | Type | CFG=1.5 | CFG=2.0 | CFG=3.0 |
|--------|------|---------|---------|---------|
| Cosine | Baseline schedule | 17.79 | 10.85 | 11.06 |
| EDM | Schedule | 26.11 | 15.09 | 11.56 |
| FM-OT | Schedule | 24.49 | 14.66 | 11.98 |
| Min-SNR | Loss weight | 16.06 | 9.70 | 10.43 |
| Soft-Min-SNR | Loss weight | 14.89 | 9.07 | 10.66 |
| Cosine Scaled | Ours (schedule) | 12.74 | **8.04** | 11.02 |
| Cauchy | Ours (schedule) | 12.91 | 8.14 | 11.02 |
| **Laplace** | **Ours (schedule)** | 16.69 | 9.04 | **7.96 (−2.89)** |

### Ablation Study

**Robustness across prediction targets (FID-10K, ImageNet-256)**:

| Prediction Target | Schedule | 100K | 200K | 300K | 400K | 500K |
|-------------------|----------|------|------|------|------|------|
| $\mathbf{x}_0$ | Cosine | 35.20 | 17.60 | 13.37 | 11.84 | 11.16 |
| $\mathbf{x}_0$ | Laplace | 21.78 | 10.86 | 9.44 | 8.73 | **8.48** |
| $\mathbf{v}$ | Cosine | 25.70 | 14.01 | 11.78 | 11.26 | 11.06 |
| $\mathbf{v}$ | Laplace | 18.03 | 9.37 | 8.31 | 8.07 | **7.96** |
| $\boldsymbol{\epsilon}$ | Cosine | 28.63 | 15.80 | 12.49 | 11.14 | 10.46 |
| $\boldsymbol{\epsilon}$ | Laplace | 27.98 | 13.92 | 11.01 | 10.00 | **9.53** |

**Higher-resolution ImageNet-512**: Laplace ($b=0.75$) reduces FID from 11.91 (Cosine) to **9.09** (↓23.7%).

### Key Findings

1. **Concentrating mass near $\log\text{SNR}=0$ consistently yields the best performance**: Across Laplace, Cauchy, and Cosine Scaled distributions, peak density at $\lambda = 0$ uniformly achieves the best results.
2. **Modifying the schedule outperforms modifying the loss weight**: The Laplace schedule (FID 7.96) significantly surpasses Soft-Min-SNR (FID 9.07) under equivalent compute.
3. **Faster convergence**: The Laplace schedule substantially outperforms Cosine at early training stages (100K–200K iterations).
4. **Generalization across prediction targets**: Consistent improvements over the baseline are observed for all three prediction targets ($\epsilon$, $\mathbf{x}_0$, $\mathbf{v}$).
5. **Sensitivity of $b$ in the Laplace schedule**: $b = 0.5$ is optimal for ImageNet-256; too small a value leads to insufficient training at extreme noise levels, while too large a value degenerates toward near-uniform sampling.

## Highlights & Insights

1. **Unified probabilistic framework**: Noise schedule design is elegantly recast as a distribution selection problem, enabling different schedules to be compared and designed within a common coordinate system.
2. **"Where to compute" matters more than "how to weight"**: This finding carries implications for all iteratively trained models, not just diffusion models.
3. **High practical value**: The Laplace schedule requires only a few lines of code (pseudocode provided in the appendix) and can be used as a drop-in replacement for existing schedules.
4. **Complementary to SD3's logit-normal sampling**: The appendix analyzes how SD3's sampling scheme also embodies the principle of concentrating mass at intermediate timesteps.

## Limitations & Future Work

1. **Validation limited to DiT-B scale**: To control variables, experiments are not conducted at XL scale; effectiveness in large-model training remains to be verified.
2. **Hyperparameters require resolution-specific tuning**: Different $b$ values are needed for 512 vs. 256 resolution (0.75 vs. 0.5), with no adaptive scheme proposed.
3. **Inference-side scheduling not addressed**: The paper notes that noise allocation during inference also warrants optimization, but this is left for future work.
4. **FID-10K evaluation**: FID estimates based on small sample sizes may exhibit variance; evaluation at larger scale would be more reliable.
5. **No direct quantitative comparison with recent flow matching methods** (e.g., SD3's logit-normal) within a unified framework.

## Related Work & Insights

- **Relation to Min-SNR**: Min-SNR effectively reduces $w(\lambda)$ at high-SNR regions by truncating the loss weight, whereas this work directly reduces the sampling frequency at high/low SNR regions.
- **Relation to SD3's logit-normal**: SD3's logit-normal sampling achieves a similar effect of concentrating mass at intermediate timesteps within the flow matching framework, corroborating the same underlying principle.
- **Broader implication**: The paradigm of optimizing training by redesigning the sampling distribution is extensible to larger-scale settings such as video diffusion models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified probabilistic framework is insightful, though the core idea of concentrating on intermediate noise levels was implicitly suggested by prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic comparisons across schedules, prediction targets, and resolutions, though model scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, with well-coordinated equations, tables, and figures.
- **Value**: ⭐⭐⭐⭐ — Highly practical; a few lines of code suffice to improve training efficiency, making widespread adoption feasible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improved Training Technique for Shortcut Models (iSM)](../../NeurIPS2025/image_generation/improved_training_technique_for_shortcut_models.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[CVPR 2026\] TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](../../CVPR2026/image_generation/taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)
- [\[ICCV 2025\] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization](dmq_dissecting_outliers_of_diffusion_models_for_post-training_quantization.md)

</div>

<!-- RELATED:END -->
