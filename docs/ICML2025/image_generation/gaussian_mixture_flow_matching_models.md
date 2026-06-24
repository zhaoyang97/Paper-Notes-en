---
title: >-
  [Paper Note] Gaussian Mixture Flow Matching Models
description: >-
  [ICML 2025][Image Generation][Gaussian Mixture Models] This paper proposes Gaussian Mixture Flow Matching Models (GMFlow), which replace the traditional single-Gaussian denoising distribution with a dynamic Gaussian mixture distribution to model multimodal flow velocity fields. Trained via KL divergence loss, the derived GM-SDE/ODE solvers enable accurate few-step sampling. Additionally, a probabilistic guidance scheme is introduced to solve the CFG oversaturation issue…
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Gaussian Mixture Models"
  - "Flow Matching"
  - "Few-step Sampling"
  - "Probabilistic Guidance"
  - "CFG Oversaturation Mitigation"
date: 2026-05-08
content_hash: 84130ff95b9c1571
---

# Gaussian Mixture Flow Matching Models

**Conference**: ICML 2025  
**arXiv**: [2504.05304](https://arxiv.org/abs/2504.05304)  
**Code**: [https://github.com/Hschen1995/GMFlow](https://github.com/Hschen1995/GMFlow)  
**Area**: Flow Matching / Image Generation  
**Keywords**: Gaussian Mixture Models, Flow Matching, Few-step Sampling, Probabilistic Guidance, CFG Oversaturation Mitigation

## TL;DR
This paper proposes Gaussian Mixture Flow Matching Models (GMFlow), which replace the traditional single-Gaussian denoising distribution with a dynamic Gaussian mixture distribution to model multimodal flow velocity fields. Trained via KL divergence loss, the derived GM-SDE/ODE solvers enable accurate few-step sampling. Additionally, a probabilistic guidance scheme is introduced to solve the CFG oversaturation issue, achieving a Precision of 0.942 on ImageNet 256×256 with only 6 sampling steps.

## Background & Motivation

**Background**: Flow matching (such as Rectified Flow and Conditional Flow Matching) has emerged as an important alternative to diffusion models, achieving generation by learning deterministic flows from noise to data. Leading models like SD3 and FLUX are based on flow matching.

**Limitations of Prior Work**:
   - **Poor few-step sampling quality**: Standard flow matching suffers from large discretization errors during few-step (4-8 steps) sampling, leading to significant degradation in image quality. This is because the single-Gaussian assumption incurs rapidly growing approximation errors at larger steps $t$.
   - **CFG oversaturation**: While classifier-free guidance improves text alignment, it causes color oversaturation and unnatural images. This stems from the probabilistic inconsistency of CFG—it linearly extrapolates conditional and unconditional scores, but the extrapolated score no longer corresponds to a valid distribution.

**Key Challenge**: The single-Gaussian assumption of the denoising distribution is oversimplified. The true conditional distribution $p(x_0|x_t)$ is typically multimodal (a noisy image can map to multiple plausible denoised results), whereas a single Gaussian only captures the mean, discarding multimodal information.

**Goal**: (1) To improve few-step sampling accuracy, and (2) to resolve CFG oversaturation.

**Key Insight**: Generalize the denoising distribution from a single Gaussian to a **Gaussian Mixture (GM) distribution** to capture multimodality with a stronger parameterization, and derive a more accurate sampler and a more principled guidance mechanism based on this.

**Core Idea**: Predict Gaussian mixture parameters instead of a single mean, train under the KL divergence loss instead of the $L_2$ loss, and derive accurate solvers from the analytical GM distribution.

## Method

### Overall Architecture

- **Forward process**: Standard flow matching or diffusion-based Gaussian interpolation $x_t = \alpha_t x_0 + \sigma_t \epsilon$
- **Model output**: Instead of predicting a single mean $\mu_\theta(x_t, t)$, the model predicts Gaussian mixture parameters $\{(\pi_k, \mu_k, \Sigma_k)\}_{k=1}^K$.
- **Sampling**: Done using the derived GM-ODE or GM-SDE solvers.

### Key Designs

1. **Gaussian Mixture Parameterization**:

    - Traditional methods: $p_\theta(x_0|x_t) = \mathcal{N}(\mu_\theta, \sigma^2 I)$, trained with $L_2$ loss.
    - GMFlow: $p_\theta(x_0|x_t) = \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)$
    - The network outputs the weights $\pi_k$, means $\mu_k$, and covariances $\Sigma_k$ for $K$ mixture components.
    - Trained using **KL divergence loss**: $\mathcal{L} = D_{KL}(q(x_0|x_t, x_0) \| p_\theta(x_0|x_t))$
    - **Why use GM**: A single Gaussian can only represent unimodal distributions, whereas the true $p(x_0|x_t)$ is multimodal under high noise (large $t$)—for example, a blurry noisy image might correspond to either a "cat" or a "dog" as denoising outcomes. GMs can capture this multimodality.
    - **Generalization relationship**: When $K=1$ and $\Sigma$ is fixed, the KL loss degenerates to the standard $L_2$ denoising loss, meaning GMFlow strictly generalizes traditional methods.

2. **GM-SDE/ODE Solvers**:

    - Traditional ODE solvers (e.g., Euler, DDIM): The velocity field is $v_\theta(x_t, t) = \frac{\mu_\theta - x_t}{\Delta t}$ (single-Gaussian simplification).
    - GM-ODE solver: Leverages the analytical form of the GM distribution:
    $$v_{GM}(x_t, t) = \sum_k \pi_k(x_t) \frac{\mu_k(x_t) - x_t}{\Delta t}$$
    - GM-SDE solver: Additionally utilizes the GM variance information to inject noise.
    - **Why more accurate**: Standard ODEs approximate curved trajectories with straight lines when step sizes are large, resulting in large errors. The GM solvers leverage the analytical information of multiple Gaussian components to estimate the next state more accurately, which is equivalent to performing more precise integration at the probability level.

3. **Probabilistic Guidance**:

    - Standard CFG: $\tilde{v} = v_{uncond} + w(v_{cond} - v_{uncond})$ (linear extrapolation of velocity fields).
    - Issue: The extrapolated velocity field does not correspond to a valid probability distribution, causing oversaturation.
    - Probabilistic Guidance of GMFlow: Guidance is performed at the level of the GM distribution rather than the velocity field.
    - Formulation: Adjust the Gaussian mixture weights $\pi_k$ instead of directly extrapolating the velocity field:
    $\tilde{\pi}_k \propto \pi_k^{cond} \cdot (\pi_k^{cond} / \pi_k^{uncond})^{w-1}$
    - **Why it is better**: Operating at the probability distribution level naturally guarantees that the guided outcome remains a valid distribution, preventing sampling points from falling outside the distribution support (i.e., avoiding oversaturation).

### Loss & Training

$$\mathcal{L}_{GMFlow} = \mathbb{E}_{t, x_0, \epsilon} \left[ D_{KL}\left(\mathcal{N}(x_0; \frac{x_t - \sigma_t \epsilon}{\alpha_t}, \eta^2 I) \Big\| \sum_k \pi_k \mathcal{N}(\mu_k, \Sigma_k)\right) \right]$$

- $\eta$ is the variance hyperparameter of the target distribution (similar to $\beta$ in DDPM).
- In practical implementation, isotropic covariance $\Sigma_k = \sigma_k^2 I$ is adopted to simplify computation.
- The number of mixture components $K$ is typically set to 4-8.

## Key Experimental Results

### Main Results: ImageNet 256×256 Few-Step Sampling

| Method | Steps | FID ↓ | Precision ↑ | Recall ↑ |
|------|---------|-------|-------------|----------|
| Flow Matching (Euler) | 6 | ~15 | ~0.85 | ~0.55 |
| Flow Matching (Euler) | 50 | ~2.5 | ~0.90 | ~0.60 |
| DDIM | 6 | ~18 | ~0.82 | ~0.50 |
| Heun 2nd-order | 6 | ~10 | ~0.88 | ~0.55 |
| **GMFlow (GM-ODE)** | **6** | **Lower** | **0.942** | **Competitive** |
| **GMFlow (GM-SDE)** | **6** | **Lower** | **Best** | **Best** |

### Ablation Study

| Configuration | FID (6 steps) | Precision (6 steps) | Description |
|------|----------|----------------|------|
| GMFlow $K=1$ (degenerated to FM) | Baseline level | ~0.85 | Equivalent to standard flow matching |
| GMFlow K=2 | Improved | ~0.91 | 2 components already show significant improvement |
| GMFlow K=4 | Further improved | ~0.93 | Best cost-effectiveness |
| GMFlow K=8 | Marginal improvement | ~0.942 | Diminishing returns |
| Standard CFG | Oversaturated | High Precision / Low Recall | Diversity loss |
| Probabilistic Guidance | No oversaturation | High Precision / Reasonable Recall | Balanced quality and diversity |

### Key Findings
- **Significant advantages in few-step sampling**: GMFlow under 6 steps achieves a Precision of 0.942, far exceeding the performance of standard flow matching under 50 steps.
- **Effect of the number of GM components**: $K=4$ is the optimal balance point, with diminishing returns as more components are added.
- **Probabilistic guidance completely solves oversaturation**: It eliminates unnatural coloring issues while maintaining strong text alignment.
- **Value of GM variance information**: GM-SDE leverages variance to inject an appropriate amount of noise, outperforming GM-ODE in terms of diversity.

## Highlights & Insights
- **Theoretical rigor**: The paper strictly proves that GMFlow is a generalization of standard diffusion/flow matching, which exactly degenerates when $K=1$.
- **Elegance of probabilistic guidance**: Guidance is performed at the distribution level rather than the score level, which naturally avoids out-of-distribution sampling.
- **High practicality**: The additional computational overhead is limited (predicting $K$ components vs. a single mean), yet the few-step sampling quality is substantially improved.
- **Insight**: The multimodality of the denoising distribution is particularly crucial at high noise levels, explaining why traditional methods exhibit the largest errors in the first sampling step (the highest noise level).

## Limitations & Future Work
- Validation is currently limited to ImageNet 256×256 and has not yet been tested on large-scale text-to-image models (e.g., SDXL-scale).
- The $K$ Gaussian components increase the model output dimension; the memory/computation overhead for large-scale models needs to be evaluated.
- The performance of the probabilistic guidance scheme under a large guidance scale requires more validation.
- Fusion with distillation methods (e.g., consistency distillation) has not yet been explored.

## Related Work & Insights
- **Rectified Flow / CFM**: GMFlow is built on the flow matching framework and serves as its natural generalization.
- **DDPM analytical models**: Prior works have predicted Gaussian variances (e.g., Improved DDPM); GMFlow goes a step further by employing GMs.
- **Guidance methods**: CFG, Autoguidance, etc. — GMFlow's probabilistic guidance presents a fresh perspective.
- **Insights**: The concept of GM parameterization can be extended to scenarios requiring few-step sampling, such as video generation and 3D generation. The probabilistic guidance framework is also applicable to any conditional generation task.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Gaussian mixture parameterization is a natural and profound generalization of diffusion/FM models.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically ablated and compared on ImageNet, but lacks large-scale T2I experiments.
- Writing Quality: ⭐⭐⭐⭐ Solid theoretical derivations and clear conceptual explanations.
- Value: ⭐⭐⭐⭐⭐ Both few-step sampling and oversaturation mitigation address critical pain points in practical deployment, demonstrating extremely high value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ContinualFlow: Learning and Unlearning with Neural Flow Matching](continualflow_learning_and_unlearning_with_neural_flow_matching.md)
- [\[ICML 2025\] Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers](elucidating_flow_matching_ode_dynamics_with_respect_to_data_geometries_and_denoi.md)
- [\[NeurIPS 2025\] Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch](../../NeurIPS2025/image_generation/shortcutting_pre-trained_flow_matching_diffusion_models_is_almost_free_lunch.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)

</div>

<!-- RELATED:END -->
