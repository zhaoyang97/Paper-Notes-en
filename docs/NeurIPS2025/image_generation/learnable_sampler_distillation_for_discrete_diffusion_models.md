---
title: >-
  [Paper Note] Learnable Sampler Distillation for Discrete Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][discrete diffusion models] This paper proposes LSD and LSD+, which distill the intermediate score trajectory knowledge of a high-fidelity teacher sampler into a few-step student sampler via learnable sampling coefficients and non-uniform time scheduling, enabling efficient and high-quality sampling for discrete diffusion models.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "discrete diffusion models"
  - "sampling acceleration"
  - "distillation"
  - "learnable coefficients"
  - "time scheduling"
date: 2026-05-08
content_hash: 6d83d24aee057ba6
---

# Learnable Sampler Distillation for Discrete Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.19962](https://arxiv.org/abs/2509.19962)  
**Code**: [GitHub](https://github.com/feiyangfu/LSD)  
**Area**: Diffusion Models / Discrete Diffusion Acceleration
**Keywords**: discrete diffusion models, sampling acceleration, distillation, learnable coefficients, time scheduling

## TL;DR

This paper proposes LSD and LSD+, which distill the intermediate score trajectory knowledge of a high-fidelity teacher sampler into a few-step student sampler via learnable sampling coefficients and non-uniform time scheduling, enabling efficient and high-quality sampling for discrete diffusion models.

## Background & Motivation

Discrete diffusion models (DDMs) have demonstrated strong capabilities in generating discrete data such as text and molecules, but their low sampling efficiency remains the primary bottleneck for practical deployment — typically requiring 1024+ neural function evaluations (NFEs).

Directly reducing the number of sampling steps severely amplifies two types of errors:

**Compounding Decoding Error**: DDMs adopt a factorized parameterization for computational efficiency, predicting the denoising state of each token independently and ignoring inherent inter-token dependencies. Reducing the number of steps degrades this approximation quality.

**Discretization Error**: Numerical methods such as Euler or τ-leaping cannot accurately approximate the reverse dynamics under large step sizes.

These two types of errors accumulate along the sampling trajectory and severely degrade generation quality at low NFEs.

Key challenges faced by existing acceleration methods:
- Exact simulation methods (e.g., Gillespie) are computationally expensive and non-parallelizable.
- Approximate methods such as τ-leaping are only first-order accurate and require small step sizes to ensure quality.
- The JYS method only optimizes *when to sample* (timestep selection), while still employing standard large-step approximations at each timestep.
- Distillation methods for continuous diffusion (e.g., S4S) rely on final-sample comparison and cannot be directly applied to DDMs, as the non-differentiability of discrete sampling blocks gradient flow.

## Method

### Overall Architecture

LSD adopts a teacher–student distillation framework: the teacher sampler uses $N$ steps (small step size, high fidelity), while the student sampler uses only $M \ll N$ steps. The core innovation lies in aligning **intermediate score trajectories** rather than final outputs, thereby circumventing the non-differentiability of discrete sampling. LSD+ further learns a non-uniform time schedule.

### Key Designs

1. **Learnable Coefficients**

   Learnable time-dependent coefficients $\Phi(t_k)$ are introduced into the update rule of the standard Euler sampler:
    $p(x^i_{t_{k+1}}|x^i_{t_k}) = \delta_{x^i_{t_k}}(x^i_{t_{k+1}}) + \Delta t \cdot Q_{t_k}(x^i_{t_k}, x^i_{t_{k+1}}) \cdot (\Phi(t_k) s_\theta(x_{t_k}, t_k))_{i, x^i_{t_{k+1}}}$

   The coefficient $\Phi(t_k)$ adaptively modulates the influence of the concrete score at each step, compensating for the accumulated error introduced by large step sizes. It is optimized by minimizing the score discrepancy between the student and teacher at each timestep:
    $\mathcal{L}_k(\Phi(t_k)) = \mathbb{E}_{x_{t_0}\sim\pi} \left[ d(s^*_k, \Phi(t_k) s_k) \right]$

   **Design Motivation**: Directly minimizing the final output distance $d(x_\epsilon, x^*_\epsilon)$ is infeasible due to the non-differentiability of discrete sampling, whereas aligning intermediate score trajectories provides a differentiable optimization path.

2. **Learnable Non-Uniform Time Scheduling (LSD+)**

   Building upon LSD, LSD+ additionally learns non-uniform step sizes $\{\kappa_k\}_{k=1}^M$ (initialized as $\Delta t$), yielding learned timesteps $\tau_k = T - \sum_{\ell=1}^k \kappa_\ell$. The step sizes are updated by aligning the **effective transition terms** of the student and teacher in the reverse process:
    $\tilde{\mathcal{L}}_k(\kappa_k) = \mathbb{E}_{x_{t_0}\sim\pi} \left[ d\left(\kappa_k s_\theta(x_{\tau_k}, \tau_k), \frac{T-\epsilon}{N} s_\theta(x^*_{t_k}, t_k)\right) \right]$

   **Design Motivation**: The reverse diffusion dynamics vary substantially across different time intervals; adaptively allocating step sizes better captures this variation and further reduces accumulated error.

3. **Relaxed Objective**

   The student sampler is allowed to match the teacher trajectory starting from a perturbed initial point $\tilde{x}_{t_0}$ within Hamming distance $\zeta$ of the original $x_{t_0}$:
    $d_H(x_{t_0}, \tilde{x}_{t_0}) \leq \zeta$

   $\zeta$ is set to approximately 5% of the sequence length. The relaxed objective is:
    $\mathcal{L}_{\text{relaxed},k}(\Phi(t_k)) = \mathbb{E}_{x_{t_0}, \tilde{x}_{t_0}} \left[ d(s_\theta(x^*_{t_k}, t_k), \Phi(t_k) s_\theta(\tilde{x}_{t_k}, t_k)) \right]$

   **Design Motivation**: A capacity-constrained student sampler cannot strictly match the teacher output; the relaxation makes optimization more feasible. The original, unperturbed input is still used at inference time.

### Loss & Training

The training process is efficient (approximately 5 minutes on an RTX 4090), and the learned parameters introduce no additional computational overhead at inference:
- **LSD**: Initialize $\Phi(t_k)=1$ and optimize the coefficients at each timestep sequentially using SGD.
- **LSD+**: Jointly optimize both the coefficients and step sizes $\kappa_k$.
- The distance metric $d$ uses the standard L2 norm.

## Key Experimental Results

### Main Results

Text generation (generative perplexity ↓, SEDD-small backbone, 1024 tokens × 1024 samples):

| Sampler | NFE=8 | NFE=16 | NFE=32 | NFE=64 |
|--------|-------|--------|--------|--------|
| Euler | 423.1 | 215.5 | 72.8 | 56.2 |
| Tweedie | 404.9 | 177.5 | 64.3 | 50.2 |
| JYS-Euler | 308.1 | 125.3 | 55.8 | 32.9 |
| LSD+-Euler | **128.4** | **51.8** | **36.8** | **20.7** |
| LSD+-Tweedie | 137.9 | 61.0 | 38.2 | 20.5 |

The gains of LSD+ are even more pronounced under the RADD backbone: at NFE=8, perplexity is 89.8 versus 671.0 for Euler, a reduction of approximately 87%.

### Ablation Study

| Configuration | NFE=32 (Perplexity) | Notes |
|------|----------------|------|
| LSD+ w/o relaxed objective | 34.98 | Strict matching is harder to optimize |
| LSD+ w/ relaxed objective | **33.23** | Relaxation improves feasibility |

Ablation on Hamming distance threshold (SEDD-small, Euler, 32 steps):

| Threshold | 0% | 1% | 5% (selected) | 10% | 20% |
|------|----|----|-----------|-----|-----|
| Perplexity ↓ | 35.98 | 32.15 | **31.24** | 39.97 | 51.52 |

5% yields the optimal trade-off; both larger and smaller values are detrimental.

### Key Findings

- LSD and LSD+ substantially outperform all baselines across all three backbones (SEDD-small, SEDD-medium, RADD) and all NFE settings.
- LSD+ consistently surpasses LSD, confirming the value of non-uniform time scheduling.
- The improvement is most dramatic at 8 steps (extremely low NFEs): LSD+-Euler reduces perplexity by approximately 70% relative to Euler.
- The generality of the method is validated on image generation and synthetic tasks, with significant improvements in FID on CIFAR-10 and error rates on the countdown task.

## Highlights & Insights

- The core insight — aligning intermediate score trajectories rather than final outputs — elegantly circumvents the fundamental obstacle of non-differentiable discrete sampling.
- The learnable coefficients endow the sampler with the ability to adaptively compensate for accumulated errors, representing an extremely lightweight yet effective enhancement.
- The relaxed objective leverages Hamming distance as a natural metric in discrete space, yielding a conceptually elegant design.
- Training cost is minimal (5 minutes) with zero inference overhead, making the approach highly practical.

## Limitations & Future Work

- The learned coefficients and time schedules are currently global (input-agnostic); input-conditioned adaptation may yield further improvements.
- The Hamming distance threshold for the relaxed objective requires manual tuning.
- No theoretical guarantees are provided on the distributional gap between the teacher and student sampler outputs.
- Performance on larger-scale DDM models remains to be validated.

## Related Work & Insights

- Conceptually related to LD3 and S4S for continuous diffusion, but addresses the DDM-specific challenge of non-differentiability.
- JYS optimizes only *when to sample*; LSD simultaneously optimizes *how to sample* (coefficients) and *when to sample* (time scheduling).
- For the field of discrete diffusion model acceleration, this paper identifies intermediate trajectory alignment as a more tractable direction than final-sample matching.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Intermediate score trajectory alignment combined with learnable coefficients constitutes a genuinely novel paradigm for DDM sampling acceleration.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three backbones, three task types (text / image / synthetic), and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Technical presentation is clear, with well-articulated distinctions from related work.
- **Value**: ⭐⭐⭐⭐⭐ Minimal training cost, zero inference overhead, and substantial improvements in low-step generation quality make this work highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[ICML 2025\] Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)](../../ICML2025/image_generation/distillation_of_discrete_diffusion_through_dimensional_correlations.md)
- [\[NeurIPS 2025\] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[CVPR 2025\] MixerMDM: Learnable Composition of Human Motion Diffusion Models](../../CVPR2025/image_generation/mixermdm_learnable_composition_of_human_motion_diffusion_models.md)

</div>

<!-- RELATED:END -->
