---
title: >-
  [Paper Note] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection
description: >-
  [ICLR 2026][Image Generation][Model Evidence] This paper proposes DiME, a model evidence estimator that integrates along the temporal marginals of the diffusion posterior. DiME requires neither prior scores nor density evaluations, and accurately estimates model evidence under diffusion model priors using as few as 20 posterior samples, enabling prior selection and model validation.
tags:
  - ICLR 2026
  - Image Generation
  - Model Evidence
  - Diffusion Priors
  - Posterior Sampling
  - Model Selection
  - Black Hole Imaging
date: 2026-05-08
content_hash: 5e22248997106a3d
---

# Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection

**Conference**: ICLR 2026
**arXiv**: [2602.20549](https://arxiv.org/abs/2602.20549)
**Code**: —
**Area**: Bayesian Inference / Diffusion Models
**Keywords**: Model Evidence, Diffusion Priors, Posterior Sampling, Model Selection, Black Hole Imaging

## TL;DR

This paper proposes DiME, a model evidence estimator that integrates along the temporal marginals of the diffusion posterior. DiME requires neither prior scores nor density evaluations, and accurately estimates model evidence under diffusion model priors using as few as 20 posterior samples, enabling prior selection and model validation.

## Background & Motivation

In Bayesian inverse problems, the prior distribution $p(\boldsymbol{x})$ fundamentally determines the posterior $p(\boldsymbol{x}|\boldsymbol{y})$. An ill-chosen prior can severely bias reconstruction results. The principled approach is to evaluate competing prior models via the model evidence $p(\boldsymbol{y}|M)$.

However, computing model evidence directly for diffusion model priors is intractable:
- It requires integrating over the full prior: $\log p(\boldsymbol{y}|M) = \log \int p(\boldsymbol{y}|\boldsymbol{x}) p(\boldsymbol{x}|M) d\boldsymbol{x}$
- Existing methods (SMC, AIS, nested sampling) require the clean prior score $\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x})$ or an unnormalized density
- Diffusion models learn scores of intermediate noisy priors, making clean prior scores inaccurate
- Density estimation methods suffer from high variance, requiring thousands of posterior samples

## Method

### Core Formula

**DiME Estimator** (integrating along the standard marginals):

$$\log p(\boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_0 \sim p(\boldsymbol{x}_0|\boldsymbol{y})}[\log p(\boldsymbol{y}|\boldsymbol{x}_0)] - D_{\text{KL}}(p(\boldsymbol{x}_0|\boldsymbol{y}) \| p(\boldsymbol{x}_0))$$

The KL divergence is estimated by integrating over temporal marginals of the reverse diffusion:

$$D_{\text{KL}} \approx \sum_{i=1}^N c_{t_i} \Delta t_i \mathbb{E}_{\boldsymbol{x}_{t_i} \sim p(\boldsymbol{x}_{t_i}|\boldsymbol{y})} \|\nabla_{\boldsymbol{x}_{t_i}} \log p(\boldsymbol{y}|\boldsymbol{x}_{t_i})\|^2$$

where $c_{t_i} = \sigma_{t_i}' \sigma_{t_i} - \sigma_{t_i}^2 \frac{a_{t_i}'}{a_{t_i}}$.

### Key Design 1: Unbiased Likelihood Score Estimation

Direct computation of $\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t)$ is intractable. Leveraging DAPS posterior samples $\tilde{\boldsymbol{x}}_0 \sim p(\boldsymbol{x}_0|\boldsymbol{x}_t, \boldsymbol{y})$, two unbiased estimators are designed:

**High-noise estimator** (low variance at high noise levels):

$$\Theta_{\text{high}}(\tilde{\boldsymbol{x}}_0) = \frac{a_t}{\sigma_t^2}(\tilde{\boldsymbol{x}}_0 - \mathbb{E}[\boldsymbol{x}_0|\boldsymbol{x}_t])$$

**Low-noise estimator** (low variance at low noise levels):

$$\Theta_{\text{low}}(\tilde{\boldsymbol{x}}_0) = \frac{a_t}{\sigma_t^2}(\boldsymbol{\Sigma}_{\boldsymbol{x}_0|\boldsymbol{x}_t} \nabla_{\tilde{\boldsymbol{x}}_0} \log p(\boldsymbol{y}|\tilde{\boldsymbol{x}}_0))$$

Two independent samples $\tilde{\boldsymbol{x}}_0^{(1)}, \tilde{\boldsymbol{x}}_0^{(2)}$ are drawn for each $\boldsymbol{x}_t$ to obtain an unbiased estimate of the squared score.

### Key Design 2: Improved Posterior Covariance

The DAPS covariance heuristic $\sigma_t^2$ overestimates variance at high noise levels. DiME introduces a prior-informed approximation:

$$\boldsymbol{\Sigma}_{\boldsymbol{x}_0|\boldsymbol{x}_t} = \left[\boldsymbol{\Sigma}_0^{-1} + \frac{a_t^2}{\sigma_t^2}\mathbf{I}\right]^{-1}$$

where $\boldsymbol{\Sigma}_0$ is estimated empirically from training data.

### Implementation

DiME operates in concert with the DAPS posterior sampling method, reusing intermediate samples naturally produced during sampling without incurring additional computational cost.

## Key Experimental Results

### Gaussian Mixture Prior Benchmark

| Method | In-distribution $\boldsymbol{x}^*$ Rel. Error↓ | OOD Rel. Error↓ | Saddle Point Rel. Error↓ |
|------|------|------|------|
| Naive MC (1000) | 2451% | 2357% | 2299% |
| Original DAPS Heuristic | 146% | 3.3% | 7.3% |
| TI | 3.2% | 5.6% | 1.2% |
| SMC | 2.6% | 1.2% | **0.7%** |
| **DiME** | **1.5%** | **0.6%** | 0.8% |

DiME achieves accuracy comparable to SMC without requiring prior scores.

### MNIST Model Selection

Given a single noisy measurement, DiME selects the correct prior from 10 diffusion models corresponding to different digit classes, while baseline methods fail.

### M87* Black Hole Imaging

- DiME identifies the GRMHD prior as having higher likelihood than RIAF, spatial image, face, and MNIST priors.
- Prior predictive checks confirm that M87* observations are statistically consistent with the GRMHD prior.

### Key Findings

- Accurate estimates are achievable with as few as 20 posterior samples.
- The automatic switching strategy between high- and low-noise estimators effectively reduces variance.
- The improved covariance approximation substantially reduces bias at high noise levels.
- DiME generalizes to model evidence estimation under arbitrary annealing schedules.

## Highlights & Insights

- The first diffusion model evidence estimator that does not rely on prior scores or densities.
- Exceptionally sample-efficient (20 samples vs. thousands required by baseline methods).
- Theoretically elegant derivation that exploits intermediate samples naturally arising during diffusion sampling.
- Validated on a real scientific application (black hole imaging), demonstrating practical utility.

## Limitations & Future Work

- Relies on the Gaussian approximation $p(\boldsymbol{x}_0|\boldsymbol{x}_t) \approx \mathcal{N}$, which may be inaccurate for multimodal priors.
- Coupled to a specific posterior sampling method (DAPS); generalization to other samplers requires additional derivation.
- The diagonal covariance approximation may limit accuracy in complex high-dimensional settings.
- Estimator variance may increase with problem dimensionality.

## Related Work & Insights

- **Evidence Estimation**: SMC, AIS, nested sampling, harmonic mean estimator, etc.
- **Diffusion Posterior Sampling**: DAPS, DPS, PnP-DM, and related methods.
- **Model Selection**: Bayes factors, cross-validation, and alternative frameworks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introduces a fundamentally new paradigm for diffusion evidence estimation.
- Theory: ⭐⭐⭐⭐⭐ — Rigorous derivations supported by multiple lemmas.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation from toy to real scientific applications.
- Value: ⭐⭐⭐⭐ — Direct practical value for scientific imaging and model selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[ICLR 2026\] Monocular Normal Estimation via Shading Sequence Estimation](monocular_normal_estimation_via_shading_sequence_estimation.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[NeurIPS 2025\] HollowFlow: Efficient Sample Likelihood Evaluation using Hollow Message Passing](../../NeurIPS2025/image_generation/hollowflow_efficient_sample_likelihood_evaluation_using_hollow_message_passing.md)
- [\[ICLR 2026\] DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)

</div>

<!-- RELATED:END -->
