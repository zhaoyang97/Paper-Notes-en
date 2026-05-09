---
title: >-
  [Paper Note] DistillKac: Few-Step Image Generation via Damped Wave Equations
description: >-
  [ICLR 2026][Image Generation][damped wave equation] This paper replaces the Fokker-Planck equation with the damped wave equation (telegrapher's equation) and its stochastic Kac representation as the probabilistic flow foundation for generative models, enabling finite-speed propagation. An endpoint distillation method is proposed for few-step generation, achieving FID=4.14 in 4 steps and FID=5.66 in 1 step on CIFAR-10.
tags:
  - ICLR 2026
  - Image Generation
  - damped wave equation
  - Kac process
  - finite-speed flow
  - endpoint distillation
  - few-step generation
date: 2026-05-08
content_hash: 04396f3b3693cd2c
---

# DistillKac: Few-Step Image Generation via Damped Wave Equations

**Conference**: ICLR 2026
**arXiv**: [2509.21513](https://arxiv.org/abs/2509.21513)
**Code**: None
**Area**: Diffusion Models / Few-Step Generation / Novel PDE Framework
**Keywords**: damped wave equation, Kac process, finite-speed flow, endpoint distillation, few-step generation

## TL;DR
This paper replaces the Fokker-Planck equation with the damped wave equation (telegrapher's equation) and its stochastic Kac representation as the probabilistic flow foundation for generative models, enabling finite-speed propagation. An endpoint distillation method is proposed for few-step generation, achieving FID=4.14 in 4 steps and FID=5.66 in 1 step on CIFAR-10.

## Background & Motivation

**Background**: Diffusion models are grounded in the Fokker-Planck equation (a parabolic PDE), whose reverse velocity field becomes stiff near the terminal time, as the diffusion process permits infinite propagation speed.

**Limitations of Prior Work**: The velocity norm of the reverse ODE can grow unboundedly as $t \to T$, causing numerical instability near the endpoint and requiring a large number of steps to guarantee accuracy. During distillation, student models tend to deviate from teacher trajectories under large step sizes.

**Key Challenge**: Infinite propagation speed → stiff velocity field → unstable sampling → many steps required. Can this problem be addressed at the PDE level?

**Goal**: Introduce a hyperbolic PDE (damped wave equation) as an alternative, exploiting its finite-speed propagation to obtain more stable few-step generation.

**Key Insight**: The damped wave equation generalizes the Fokker-Planck equation — diffusion emerges as the limit when both damping and speed tend to infinity. The Kac process naturally imposes a velocity upper bound $c$, guaranteeing globally bounded kinetic energy and Lipschitz regularity in Wasserstein space.

**Core Idea**: Finite-speed probabilistic flow ensures that endpoint matching automatically guarantees proximity along the entire path, thereby stabilizing few-step distillation.

## Method

### Overall Architecture

- **PDE Replacement**: Replaces the Fokker-Planck equation with the damped wave equation $\partial_{tt} p + \lambda \partial_t p = c^2 \nabla^2 p$
- **Stochastic Kac Representation**: Particles move at finite speed $c$, with velocity direction reversed (1D) or resampled (high-dimensional) according to a Poisson process
- **Guided Kac Flow**: Incorporates CFG in the velocity space while preserving square-integrability
- **Endpoint Distillation**: The student matches the teacher's output at the endpoints of each time interval

### Key Designs

1. **Finite-Speed Probabilistic Flow**:

    - Function: Replaces the diffusion SDE with the Kac process, yielding a globally bounded velocity field
    - Mechanism: In the Kac process, particle position $X_t$ and velocity $V_t$ evolve jointly, with $|V_t| \leq c$, so trajectories remain within the causal cone and cannot propagate infinitely
    - Design Motivation: Bounded velocity → no terminal stiffness → more stable numerical integration → more robust few-step sampling

2. **Endpoint Distillation + Path Stability Theorem (Theorem 8)**:

    - Function: Proves that endpoint matching guarantees proximity along the entire trajectory
    - Mechanism: By exploiting the Lipschitz regularity of the Kac flow, if the student and teacher agree at endpoints $t_k$, they remain close throughout the interval $[t_{k+1}, t_k]$, with errors decaying at $O(M^{-1})$ (Euler student)
    - Design Motivation: This is an advantage unique to finite-speed flows — infinite-speed diffusion flows cannot guarantee such stability

3. **Velocity-Space CFG**:

    - Function: Applies classifier-free guidance to the Kac velocity field
    - Mechanism: $u_{\text{guided}} = (1+w) u_\theta^{\text{cond}} - w u_\theta^{\text{uncond}}$, proved to preserve square-integrability under mild conditions
    - Design Motivation: Conventional CFG operates in score space and may violate the finite-speed constraint; operating in velocity space naturally preserves it

### Loss & Training

- UNet backbone; evaluated on CIFAR-10, CelebA-64, and LSUN Bedroom-256
- Teacher: 100-step Guided Kac Flow (AB-2 integrator)
- Distillation: iterative schedule 100→20→4→2→1 steps
- Endpoint MSE loss

## Key Experimental Results

### Main Results

| Method | NFE | FID (CIFAR-10) | FID (CelebA-64) |
|--------|-----|----------------|-----------------|
| Guided Kac Flow (100 steps, AB-2) | 100 | 3.58 | 3.50 |
| **DistillKac** | **20** | **3.72** | **3.42** |
| DistillKac | 4 | 4.14 | 4.36 |
| DistillKac | 2 | 4.68 | 5.66 |
| DistillKac | 1 | 5.66 | 7.45 |
| DDIM (100 steps) | 100 | 4.16 | 6.53 |
| DDIM (20 steps) | 20 | 6.84 | 13.73 |
| Progressive Distillation | 4 | 3.00 | — |
| iCT | 2 | 2.46 | — |

### Key Findings
- Distilling from 100 to 1 step increases FID by only 2.08 (3.58→5.66), demonstrating the endpoint stability advantage of finite-speed flow
- At 20 steps, DistillKac (3.72) substantially outperforms DDIM (6.84); the gap widens further at 4 steps (4.14 vs. unavailable)
- The AB-2 integrator achieves the best efficiency: second-order accuracy with only one function evaluation per step
- Absolute FID values remain below EDM (1.79) and iCT (2.46), indicating that the base Kac flow model's fitting capacity requires further improvement

## Highlights & Insights
- **PDE-Level Innovation**: Extending generative models from parabolic PDEs (Fokker-Planck) to hyperbolic PDEs (damped wave equation) represents a fundamental paradigm shift. The taxonomy in Table 1 — mapping three PDE classes (parabolic/elliptic/hyperbolic) to three families of generative models — is highly illuminating.
- The **endpoint-to-path stability theorem** is the core theoretical contribution: the geometric properties of finite-speed flows allow endpoint supervision to yield path consistency "for free," providing the theoretical foundation for the distillation design.
- Potential impact: If the base Kac flow model quality can be further improved (e.g., via DiT), the stability advantages of finite-speed propagation may become even more pronounced at scale.

## Limitations & Future Work
- Absolute generation quality falls short of SOTA (FID 3.58 vs. EDM 1.79); the base Kac flow model requires improvement
- Validation is limited to small-scale datasets (CIFAR-10, CelebA-64); experiments on ImageNet or high-resolution benchmarks are absent
- The speed bound $c$ and damping rate $\lambda$ of the Kac process require tuning, increasing the hyperparameter burden
- The efficiency of the directional resampling mechanism for Kac processes in high dimensions is not thoroughly analyzed
- Comparison with consistency models (iCT, sCT) is insufficiently comprehensive

## Related Work & Insights
- **vs. Kac Flow (Duong et al., 2026)**: DistillKac builds on this work by adding CFG and distillation, reducing FID from 6.42 to 3.58 (100 steps) and 5.66 (1 step)
- **vs. Progressive Distillation**: Conceptually similar, but with a distinct theoretical foundation — DistillKac provides formal guarantees via the endpoint-to-path stability theorem
- **vs. Flow Matching / Rectified Flow**: Both are ODE-based flows, but the Kac process imposes a finite-speed constraint that may yield greater stability near the terminal time

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering hyperbolic PDE framework for generative models; outstanding theoretical contribution
- Experimental Thoroughness: ⭐⭐⭐ Limited to small datasets; absolute performance is not sufficiently strong
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and elegant theoretical derivations; the mapping from PDEs to generative models is clearly presented
- Value: ⭐⭐⭐⭐ Opens a new direction (hyperbolic generative models), but further work is needed to validate large-scale feasibility

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](../../CVPR2026/image_generation/uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)
- [\[CVPR 2026\] Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning](../../CVPR2026/image_generation/refining_few-step_text-to-multiview_diffusion_via_reinforcement_learning.md)
- [\[ICCV 2025\] Learning Few-Step Diffusion Models by Trajectory Distribution Matching](../../ICCV2025/image_generation/learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)
- [\[ICLR 2026\] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)

<!-- RELATED:END -->
