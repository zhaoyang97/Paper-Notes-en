---
title: >-
  [Paper Note] RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories
description: >-
  [CVPR 2025][Image Generation][Diffusion acceleration] The RayFlow diffusion framework is proposed, which designs a unique diffusion trajectory (pointing to an instance-specific target distribution) for each sample and optimizes training via Time Sampler importance sampling, maintaining generation diversity and stability while minimizing sampling steps.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion acceleration"
  - "flow matching"
  - "instance-aware trajectories"
  - "importance sampling"
  - "sampling stability"
date: 2026-05-08
content_hash: acc5a0d695cb05a4
---

# RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories

**Conference**: CVPR 2025  
**arXiv**: [2503.07699](https://arxiv.org/abs/2503.07699)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Diffusion acceleration, flow matching, instance-aware trajectories, importance sampling, sampling stability

## TL;DR

The RayFlow diffusion framework is proposed, which designs a unique diffusion trajectory (pointing to an instance-specific target distribution) for each sample and optimizes training via Time Sampler importance sampling, maintaining generation diversity and stability while minimizing sampling steps.

## Background & Motivation

The slow generation speed of diffusion models remains a core challenge. Existing acceleration methods suffer from the following issues:

1. **Three issues of traditional diffusion**: (1) The denoising expectations at different timesteps differ, and compressing steps inevitably degrades quality; (2) All samples converge to the same standard Gaussian, leading to path overlap and significant sampling randomness; (3) The results of adjacent sample points can vary drastically, causing sampling instability.
2. **Limitations of Rectified Flow**: Although it uses straight-line ODEs for sampling, there is a large gap between the trajectory and the actual ODE path, which severely restricts generation diversity and lacks proof of theoretical optimality.
3. **Limitations of distillation methods**: High computational overhead, complex training, and difficulty in preserving guidance capability.

This paper proposes that each sample diffuses along a unique trajectory to an instance-specific target distribution $\mathcal{N}(\epsilon_\mu, \sigma^2 I)$, rather than a uniform standard Gaussian.

## Method

### Overall Architecture

RayFlow modifies the target distribution of the diffusion process: from a standard Gaussian $\mathcal{N}(0, I)$ to an instance-specific distribution $\mathcal{N}(\epsilon_\mu, \sigma^2 I)$, where $\epsilon_\mu = \mathbb{E}_t[\mathbb{E}[\bar{\epsilon}_t]]$ is the unified noise expectation of the pretrained model, and $\sigma \to 0$. This ensures that the diffusion trajectories of different samples do not overlap, making reverse sampling more stable.

### Key Designs

**1. RayFlow Forward/Backward Process**

- **Function**: Constructing a unique diffusion trajectory for each sample to maximize path probability.
- **Mechanism**: The forward process is defined as $\psi_t(\cdot|\epsilon) = \sqrt{\bar{\alpha}_t} x_0 + (1-\sqrt{\alpha_t})\epsilon_\mu + \sqrt{1-\bar{\alpha}_t}\epsilon$, which adds a shift term $(1-\sqrt{\alpha_t})\epsilon_\mu$ on top of the traditional VP representation. It is theoretically proven that the optimal parameters are $\epsilon_\mu^* = \mathbb{E}_t[\mathbb{E}[\bar{\epsilon_t}]]$ and $\sigma^* \to 0$, meaning the variance of the target distribution approaches zero.
- **Design Motivation**: To ensure all timesteps share a unified noise expectation, resolving the inconsistency of expectations across different steps in traditional diffusion. A target distribution variance approaching zero implies that the trajectories are nearly deterministic, thereby maximizing path probability.

**2. Time Sampler Importance Sampling**

- **Function**: Adaptively selecting key timesteps during training to reduce computational redundancy.
- **Mechanism**: The optimal sampling distribution is $q^*(t|x_0, \epsilon_\mu) \propto \xi_t(x_0, \epsilon_\mu) p(t)$, where $\xi_t$ measures the model's prediction error at timestep $t$. A neural network based on Stochastic Stein Discrepancies (SSD) is used to approximate this optimal distribution.
- **Design Motivation**: Uniform timestep sampling leads to significant computational waste on timesteps that the model has already mastered. Importance sampling focuses on key timesteps with larger prediction errors, thereby reducing the variance of the training loss.

**3. Fast One-Step Sampling Variant**

- **Function**: Supporting single-step generation to achieve the fastest possible inference.
- **Mechanism**: Since the trajectory of each sample in RayFlow is more deterministic (with target distribution variance approaching zero), recovering $x_0$ directly from the target mean $\hat{\epsilon}_\mu^*$ in a single step becomes feasible: $x_0 \approx \frac{\hat{\epsilon}_\mu - (1-\sqrt{\bar{\alpha}_T})\epsilon_\mu}{\sqrt{\bar{\alpha}_T}}$.
- **Design Motivation**: Non-overlapping trajectories combined with a unified expectation leads to a significantly enhanced single-step sampling quality.

### Loss & Training

Conditional loss based on the Flow Matching framework:

$$\mathcal{L}_{CFM} = \mathbb{E}_{t, p(x_t|\epsilon), p(\epsilon)} [\|v_\theta(x_t, t) - u(x_t|\epsilon)\|_2^2]$$

This is equivalent to a weighted noise prediction loss, where the weights are determined by the signal-to-noise ratio.

## Key Experimental Results

### Text-to-Image Generation (SDXL Backbone)

| Method | FID↓ | Steps | CLIP Score↑ |
|------|------|------|-----------|
| SDXL (Original) | 23.4 | 50 | 0.32 |
| Rectified Flow | 28.1 | 4 | 0.30 |
| Lightning | 25.6 | 4 | 0.31 |
| **RayFlow** | **22.8** | **4** | **0.32** |
| **RayFlow (1-step)** | **25.1** | **1** | **0.31** |

### Ablation Study

| Component | FID↓ |
|------|------|
| Baseline (RF) | 28.1 |
| + Instance-aware target | 25.4 |
| + Time Sampler | 23.6 |
| + Full RayFlow | **22.8** |

### Key Findings

- RayFlow with 4 steps outperforms the original SDXL with 50 steps (FID 22.8 vs. 23.4) while maintaining controllability.
- The 1-step generation achieves an FID of only 25.1, significantly outperforming other acceleration methods.
- The Time Sampler contributes to approximately a ~2-point improvement in FID.
- The instance-aware trajectory design effectively avoids trajectory overlapping, reducing sampling randomness.

## Highlights & Insights

1. **Thorough Theoretical Analysis**: The optimal parameters are derived from path probability maximization rather than heuristic designs.
2. **Simplicity of Unified Expectation**: Computing $\epsilon_\mu$ via a pretrained model is straightforward and requires no additional training.
3. **High Versatility of Time Sampler**: The SSD-based importance sampling method can be extended to other diffusion training processes.

## Limitations & Future Work

- The computation of $\epsilon_\mu$ depends on the pretrained model, and $\epsilon_\mu$ varies across different models.
- The Time Sampler introduces additional training overhead for the neural network.
- The actual efficacy of the "path probability maximization" assumption in high-dimensional spaces requires more verification.

## Related Work & Insights

- **Rectified Flow**: A pioneer of straight-line sampling, but its trajectory constraints are excessively strong.
- **Consistency Models**: Another few-step generation method, but requiring complex training.
- **SD-Lightning/Turbo**: Distillation-based methods with high computational overhead.

## Rating

⭐⭐⭐⭐ — Solid theoretical derivation and novel instance-aware trajectory design. It achieves excellent performance in 4-step or even 1-step generation, and the Time Sampler is also a practical contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion](../../ICML2025/image_generation/multidimensional_adaptive_coefficient_for_inference_trajectory_optimization_in_f.md)
- [\[CVPR 2026\] Few-Step Diffusion Sampling Through Instance-Aware Discretizations](../../CVPR2026/image_generation/few-step_diffusion_sampling_through_instance-aware_discretizations.md)
- [\[ICML 2025\] SADA: Stability-guided Adaptive Diffusion Acceleration](../../ICML2025/image_generation/sada_stability-guided_adaptive_diffusion_acceleration.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[CVPR 2025\] ILIAS: Instance-Level Image Retrieval At Scale](ilias_instance-level_image_retrieval_at_scale.md)

</div>

<!-- RELATED:END -->
