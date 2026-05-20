---
title: >-
  [Paper Note] Learning to Integrate Diffusion ODEs by Averaging the Derivatives
description: >-
  [NeurIPS 2025][Image Generation][Diffusion model acceleration] This paper proposes the **Secant Losses** family, which learns to integrate diffusion ODEs via Monte Carlo integration and Picard iteration…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "secant loss"
  - "ODE integration"
  - "Monte Carlo integration"
  - "Picard iteration"
date: 2026-05-08
content_hash: 9b8512737024d6e3
---

# Learning to Integrate Diffusion ODEs by Averaging the Derivatives

**Conference**: NeurIPS 2025
**arXiv**: [2505.14502](https://arxiv.org/abs/2505.14502)  
**Code**: [GitHub](https://github.com/poppuppy/secant-expectation)  
**Area**: Diffusion Models / Image Generation
**Keywords**: Diffusion model acceleration, secant loss, ODE integration, Monte Carlo integration, Picard iteration

## TL;DR

This paper proposes the **Secant Losses** family, which learns to integrate diffusion ODEs via Monte Carlo integration and Picard iteration, progressively extending the tangent of a diffusion model into a secant. The approach achieves an excellent balance between training stability and few-step inference.

## Background & Motivation

Diffusion models produce high-quality generations but typically require hundreds to thousands of function evaluations (NFEs) per sample, severely limiting practical deployment. Existing acceleration methods fall into two categories:

**Fast samplers** (DPM-Solver, UniPC, etc.): Performance degrades sharply when NFE < 10, as numerical solvers lack sufficient accuracy under extremely few steps.

**Distillation methods** (consistency models, adversarial distillation, etc.): While enabling few-step generation, they often introduce complex training pipelines, training instability, model collapse, or over-smoothing artifacts.

**Key Challenge**: Fast samplers fail under very few steps; distillation methods are too complex and unstable. A simple, stable, and effective intermediate solution is lacking.

**Key Insight**: From a geometric perspective, diffusion models learn the **tangent** of the PF-ODE (instantaneous rate of change), whereas what is actually needed is the **secant** between two time points (average rate of change). The secant is precisely the average of all tangents over an interval — a relationship that can be approximated via Monte Carlo integration and resolved during training via Picard iteration.

## Method

### Overall Architecture

Given the PF-ODE $\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{v}(\boldsymbol{x}_t, t)$, transitioning from $\boldsymbol{x}_t$ to $\boldsymbol{x}_s$ conventionally requires stepwise numerical integration. Instead, this paper proposes directly modeling a **secant function** $\boldsymbol{f}_\theta(\boldsymbol{x}_t, t, s)$ with a neural network such that:

$$\boldsymbol{x}_s = \boldsymbol{x}_t + (s-t) \boldsymbol{f}_\theta(\boldsymbol{x}_t, t, s)$$

The secant function is defined as the expectation of all tangents over the interval: $\boldsymbol{f}(\boldsymbol{x}_t, t, s) = \mathbb{E}_{r \sim U(t,s)} \boldsymbol{v}(\boldsymbol{x}_r, r)$.

### Key Designs

1. **Secant Expectation Loss**:
   The core observation is that the secant equals the uniform sampling expectation of the tangent. A loss can thus be constructed as $\mathcal{L} = \|\boldsymbol{f}_\theta(\boldsymbol{x}_t, t, s) - \boldsymbol{v}(\boldsymbol{x}_r, r)\|^2$, where $r \sim U(t,s)$. However, during training only one of $\boldsymbol{x}_t$ or $\boldsymbol{x}_r$ is accessible at a time.

2. **Picard Iteration Estimation**:
   Inspired by Picard iteration, the model itself is used to estimate the missing point. Two strategies are proposed:
   - **Estimate Interior (EI)**: Sample $\boldsymbol{x}_t$; estimate the interior point as $\hat{\boldsymbol{x}}_r = \boldsymbol{x}_t + (r-t)\boldsymbol{f}_{\theta^-}(\boldsymbol{x}_t, t, r)$; evaluate $\boldsymbol{v}(\hat{\boldsymbol{x}}_r, r)$ via a teacher model as the target.
   - **Estimate Endpoint (EE)**: Sample $\boldsymbol{x}_r$; back-propagate via the model to estimate $\hat{\boldsymbol{x}}_t$; use the ground-truth $\alpha_r'\boldsymbol{x}_0 + \sigma_r'\boldsymbol{z}$ directly as the target.

3. **Four Variants**:
    - **SDEI** (Distillation + Estimate Interior): Requires a teacher model; 3 forward + 1 backward passes.
    - **STEI** (Training + Estimate Interior): No teacher required; incorporates diffusion loss regularization; 4 forward + 2 backward passes.
    - **SDEE** (Distillation + Estimate Endpoint): Requires a teacher model; 3 forward + 1 backward passes.
    - **STEE** (Training + Estimate Endpoint): No teacher required; the lightest variant; only 2 forward + 1 backward passes.

4. **Target Stability Advantage**:
   Compared to consistency models, the targets in the secant losses are either identical to the diffusion loss ($\alpha_t'\boldsymbol{x}_0 + \sigma_t'\boldsymbol{z}$) or the diffusion model output $\boldsymbol{v}(\boldsymbol{x}_t, t)$ itself, without involving the model-dependent derivative term $\frac{d}{dt}\boldsymbol{f}_{\theta^-}$. This yields substantially superior training stability over consistency models.

### Loss & Training

- **Diffusion model initialization**: Pre-trained weights are loaded so that $\boldsymbol{f}_\theta(\boldsymbol{x}_t, t, t) = \boldsymbol{v}(\boldsymbol{x}_t, t)$, greatly accelerating convergence.
- **Balancing factor in STEI**: $\lambda=1$ is found optimal for balancing the diffusion loss and secant loss.
- **CFG integration**: Distillation variants embed CFG directly into $\boldsymbol{v}$ within the loss; STEE follows standard diffusion training with random label dropout and CFG at inference.
- **Uniform step sampling**: Uniform step sizes $(t,s) = (i/N, (i-1)/N)$ are used at inference.
- **Only 1% of original training cost**: 50K–100K iterations vs. 7M iterations for SiT.

## Key Experimental Results

### Main Results — CIFAR-10 Unconditional Generation

| Method | FID↓ | Steps | Category |
|--------|------|-------|----------|
| EDM (Teacher) | 1.97 | 35 | Diffusion model |
| DPM-Solver-v3 | 2.51 | 10 | Fast sampler |
| LD3 | 2.38 | 10 | Fast sampler |
| sCD | 2.52 | 2 | Consistency distillation |
| ECT | 2.11 | 2 | Consistency training |
| IMM | 1.98 | 2 | Training/fine-tuning |
| **SDEI (Ours)** | **2.14** | **10** | Fine-tuning |

### Main Results — ImageNet 256×256 Class-Conditional Generation

| Method | FID↓ | IS↑ | Steps |
|--------|------|-----|-------|
| SiT-XL/2 (Teacher) | 2.15 | 258.09 | 250 |
| IMM (4 steps) | 2.51 | - | 4 |
| IMM (8 steps) | 1.99 | - | 8 |
| **STEI (4 steps)** | **2.78** | **269.87** | **4** |
| **STEI+guid.int. (4 steps)** | **2.27** | **273.76** | **4** |
| **STEE (8 steps)** | **2.33** | **274.47** | **8** |
| **STEE+guid.int. (8 steps)** | **1.96** | **275.81** | **8** |
| **STEI (1 step)** | **7.12** | **241.75** | **1** |

### Ablation Study

| Configuration | FID↓ | Note |
|---------------|------|------|
| $\lambda=0.1$ | 3.96 | Diffusion loss weight too small |
| $\lambda=0.5$ | 3.15 | Suboptimal |
| $\lambda=1.0$ | **2.84** | Optimal balance |
| $\lambda=2.0$ | 3.96 | Diffusion loss begins to degrade |
| Discrete $t$ sampling, generation only | **3.23** | Best fixed-step performance |
| Continuous $t$ sampling, generation only | 4.29 | Flexible steps but reduced performance |
| Continuous $t$ sampling, generation + inversion | 5.47 | Model capacity is divided |

### Key Findings

- Consistency models diverge during training on ImageNet-256, whereas the secant losses consistently converge stably and rapidly.
- EI variants generally outperform EE variants, as the inputs are cleaner and the error propagation path is shorter.
- Training from scratch for 3000K iterations achieves an 8-step FID of 2.68, validating scalability.
- At 1-step generation, STEI (FID 7.12) surpasses IMM and Shortcut Models at 4 steps.

## Highlights & Insights

- **Theoretical elegance**: The loss function is naturally derived from Monte Carlo integration and Picard iteration, with clear geometric intuition (tangent → secant).
- **Highly stable training**: By sharing the same targets as the diffusion model, the problematic derivative terms found in consistency models are entirely avoided.
- **Simple implementation**: The approach is largely parallel to diffusion model training, requiring no additional discriminators, score distillation, or other complex components.
- **High training efficiency**: Only 1% of the teacher model's training budget is required.

## Limitations & Future Work

- A significant performance gap remains between 1-step and 8-step generation; large-step transitions remain challenging.
- ImageNet performance relies on CFG; the theoretical relationship between CFG and secant losses is not fully explored.
- Training data is required, which may be limiting in data-scarce scenarios.
- Secant accuracy guarantees are local; global extension relies on bootstrapping.

## Related Work & Insights

- The proposed method stands in a "differentiation vs. integration" duality with consistency models, using integration to circumvent the instabilities introduced by differentiation.
- The approach is related to the multi-time training strategy of Rectified Flow, but secant losses place greater emphasis on local accuracy.
- The method can be viewed as an effective intermediate solution between fast samplers and distillation methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Reframes the few-step generation problem from an integration perspective, forming an elegant duality with consistency models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Thorough evaluation on CIFAR-10 and ImageNet, though text-to-image validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Geometric intuition, theoretical derivation, and experimental validation are presented in a clear, well-structured progression.
- **Value**: ⭐⭐⭐⭐ — Provides a stable and concise few-step diffusion solution that is training-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](../../ICCV2025/image_generation/golden_noise_for_diffusion_models_a_learning_framework.md)

</div>

<!-- RELATED:END -->
