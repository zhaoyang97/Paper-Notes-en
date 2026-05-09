---
title: >-
  [Paper Note] Test-Time Iterative Error Correction for Efficient Diffusion Models
description: >-
  [ICLR 2026][Image Generation][iterative error correction] This paper proposes IEC (Iterative Error Correction), a plug-and-play test-time method that iteratively corrects inference errors in efficient diffusion models, reducing error accumulation from exponential to linear growth.
tags:
  - ICLR 2026
  - Image Generation
  - iterative error correction
  - test-time enhancement
  - quantized diffusion
  - feature caching
  - error propagation
date: 2026-05-08
content_hash: 946f9abedc17f4d3
---

# Test-Time Iterative Error Correction for Efficient Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2511.06250](https://arxiv.org/abs/2511.06250)
**Code**: [GitHub](https://github.com/zysxmu/IEC)
**Area**: Diffusion Models / Model Efficiency / Test-Time Optimization
**Keywords**: iterative error correction, test-time enhancement, quantized diffusion, feature caching, error propagation

## TL;DR

This paper proposes IEC (Iterative Error Correction), a plug-and-play test-time method that iteratively corrects inference errors in efficient diffusion models, reducing error accumulation from exponential to linear growth.

## Background & Motivation

### State of the Field

**Background**: Efficient diffusion models (via quantization, feature caching, etc.) face significant challenges after deployment:

**Approximation errors are unavoidable**: Errors introduced by quantization and caching accumulate exponentially across timesteps.

**Post-deployment models cannot be modified**:

### Limitations of Prior Work

**Limitations of Prior Work**: Memory constraints and deployment strategies render model parameters immutable.

### Root Cause

**Key Challenge**: The original full-precision weights may no longer be accessible.

### Solution Direction

**Solution Direction**: Re-executing the efficiency pipeline is computationally expensive.

**Existing methods are pre-deployment solutions**: Timestep-level quantization parameters and non-uniform caching strategies require re-running the entire efficiency pipeline.

**Core Problem**: Can the performance of an already-deployed diffusion model be improved without repeating the model efficiency pipeline?

## Method

### 1. Error Propagation Analysis

The DDIM sampling process: $x_{t-1} = A_t x_t + B_t \epsilon_\theta(x_t, t)$

Error recurrence after introducing efficient approximations:

$$\delta_{t-1} = (A_t + B_t J_t) \delta_t + B_t \epsilon_\theta^\delta$$

Final accumulated error:

$$\delta_0 = \sum_{i=1}^T \left(\prod_{j=i+1}^T (A_j + B_j J_j)\right)(B_i \epsilon_\theta^\delta)$$

**Key Findings**: Experiments confirm that $\|A_t + B_t J_t\| > 1$ holds at all timesteps, implying **exponential amplification** of errors.

### 2. Iterative Error Correction (IEC)

A correction iteration is introduced at each timestep:

$$x_{t-1}^{(k+1)} = x_{t-1}^{(k)} + \lambda (A_t x_t + B_t \epsilon_\theta(x_{t-1}^{(k)}, t) - x_{t-1}^{(k)})$$

This is equivalent to a fixed-point iteration $x_{t-1}^* = G(x_{t-1}^*)$, where:

$$G(x) = (1-\lambda)x + \lambda(A_t x_t + B_t \epsilon_\theta(x, t))$$

### 3. Convergence Proof

Using the Banach fixed-point theorem:

- Jacobian of the mapping $G$: $\nabla G(x) = (1-\lambda)I + \lambda B_t J_t$
- Lipschitz constant: $L = \|(1-\lambda)I + \lambda B_t J_t\|$
- Since $B_t < 0$, an appropriate positive $\lambda$ ensures $L < 1$
- Experiments verify that $\|\nabla G(x)\| < 1$ holds for all timesteps when $\lambda \in [0.1, 0.7]$
- In practice, $\lambda = 0.5$ is used

### 4. Error Suppression Effect

After IEC convergence, the per-step error is bounded: $\|\delta_{t-1}^{(\infty)}\| \leq \frac{C}{1-L}$

Crucially, IEC eliminates dependence on the error from the previous timestep, reducing total accumulated error from exponential to linear growth: $\delta_0^{\text{IEC}} = \sum_{j=1}^T \delta_j^x$

### 5. Practical Usage

- Maximum iterations: $K=1$ (only one additional forward pass required in practice)
- Threshold: $\tau = 10^{-5}$
- Can be selectively applied to a subset of timesteps
- For quantization methods: applied at every timestep
- For caching methods: applied only at non-cached timesteps

## Experiments

### Setup

- Models: DDPM, LDM, Stable Diffusion
- Efficiency techniques: quantization (W4A8/W8A8), DeepCache, CacheQuant
- Datasets: CIFAR-10, LSUN-Churches, LSUN-Bedrooms, ImageNet, MS-COCO
- Metrics: FID, IS, CLIP Score

### Main Results (Quantization + IEC)

### Main Results

| Dataset | Precision | Baseline FID | +IEC FID | Improvement |
|---------|-----------|-------------|---------|-------------|
| CIFAR-10 | W8A8 | High | Significantly reduced | Substantial |
| CIFAR-10 | W4A8 | Very high | Noticeably reduced | Substantial |
| LSUN-Churches | W8A8 | High | Reduced | Improved |
| LSUN-Bedrooms | W8A8 | High | Reduced | Improved |

### DeepCache + IEC

### Ablation Study

| Dataset | Cache Strategy | Baseline FID | +IEC FID |
|---------|---------------|-------------|---------|
| CIFAR-10 | N=10 | High | Reduced |
| ImageNet | N=10 | High | Reduced |

### CacheQuant + IEC

IEC also proves effective on hybrid efficiency schemes combining quantization and caching.

### Results on Stable Diffusion

- Both FID and CLIP Score improve on MS-COCO
- Applying IEC only at the first timestep already yields substantial gains

### Flexibility Analysis

| Strategy | Effect |
|----------|--------|
| No IEC | Baseline performance |
| A steps at head and tail | Adjustable quality–efficiency trade-off |
| Applied at all steps | Maximum quality improvement |

By controlling the number of timesteps at which IEC is applied, users can achieve fine-grained control over the efficiency–quality trade-off.

## Highlights & Insights

1. **Theoretical rigor**: A complete theoretical chain from error propagation analysis to convergence proof.
2. **Plug-and-play**: No retraining, no architectural modifications, and no access to the original model required.
3. **Broad applicability**: Effective across different efficiency techniques (quantization, caching, and hybrid approaches).
4. **Flexible and controllable**: Users can freely adjust the degree of application to balance efficiency and quality.
5. **A new perspective on test-time methods**: Draws inspiration from test-time scaling ideas and applies them to generative models.

## Limitations & Future Work

1. Each IEC iteration requires an additional forward pass, increasing inference time.
2. The theoretical analysis is based on DDIM; applicability to other samplers (e.g., DPM-Solver) requires further validation.
3. The optimal value of $\lambda$ may vary across models and datasets.
4. For extremely low-bit quantization with large errors (e.g., W2), the improvement from IEC may be limited.
5. The relationship with test-time training methods is not discussed.

## Related Work & Insights

- **Diffusion model quantization**: PTQ4DM, Q-Diffusion, TDQ
- **Feature caching**: DeepCache, CacheQuant
- **Test-time scaling**: TTT (Snell 2024), REPA
- **Efficient sampling**: DDIM, DPM-Solver, consistency models

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Clear theoretical contribution in reducing error growth from exponential to linear.
- **Practicality**: ⭐⭐⭐⭐⭐ — Post-deployment optimization; genuinely plug-and-play.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across models, techniques, and datasets.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous theoretical derivations and well-designed experimental setup.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[CVPR 2026\] Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction](../../CVPR2026/image_generation/physics-consistent_diffusion_for_efficient_fluid_super-resolution_via_multiscale.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
