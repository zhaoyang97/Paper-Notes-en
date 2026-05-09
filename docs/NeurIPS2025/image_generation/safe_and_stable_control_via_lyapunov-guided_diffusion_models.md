---
title: >-
  [Paper Note] Safe and Stable Control via Lyapunov-Guided Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Diffusion models] This paper proposes S²Diff, a model-based diffusion planning framework that leverages Control Lyapunov Barrier Functions (CLBF) to guide diffusion sampling for generating trajectory-level control policies. Without requiring control-affine assumptions or quadratic programming, S²Diff simultaneously guarantees safety and stability on a variety of nonlinear dynamical systems, achieving an average safety rate of 98.75%.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion models
  - Lyapunov stability
  - safe control
  - certificate functions
  - Almost Lyapunov theory
date: 2026-05-08
content_hash: 56aae182e2776175
---

# Safe and Stable Control via Lyapunov-Guided Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.25375](https://arxiv.org/abs/2509.25375)
**Code**: None
**Area**: Image Generation / Control
**Keywords**: Diffusion models, Lyapunov stability, safe control, certificate functions, Almost Lyapunov theory

## TL;DR

This paper proposes S²Diff, a model-based diffusion planning framework that leverages Control Lyapunov Barrier Functions (CLBF) to guide diffusion sampling for generating trajectory-level control policies. Without requiring control-affine assumptions or quadratic programming, S²Diff simultaneously guarantees safety and stability on a variety of nonlinear dynamical systems, achieving an average safety rate of 98.75%.

## Background & Motivation

Real-world control problems (e.g., robotics, aerospace) require not only cost minimization but also simultaneous guarantees of **safety** (trajectories avoid dangerous regions) and **stability** (the system converges to a target state). Satisfying both simultaneously remains an open problem in control theory.

Core limitations of existing approaches:

**Model Predictive Control (MPC)**: Minimizes cumulative cost over a receding horizon via convex optimization. However, resulting policies are often suboptimal, computational complexity grows sharply with horizon length, and the problem frequently becomes infeasible for high-dimensional nonlinear systems.

**Certificate-function-based methods (CLBF-QP)**: Learn a CLBF and solve for control policies via quadratic programming (QP). Three fundamental issues arise: (a) QP formulation requires control-affine dynamics; (b) greedy step-wise optimization leads to globally inconsistent behavior, necessitating slack variables that alter the optimization objective; (c) joint learning of CLBF and policy destabilizes the QP feasible region, causing training collapse.

**Existing diffusion planning methods** (e.g., Diffuser, SafeDiffuser): Perform well in long-horizon decision-making but focus solely on cost minimization, leaving safety and stability guarantees unaddressed. SafeDiffuser additionally requires hand-crafted CBF priors, limiting practical applicability.

**Root Cause**: Gradient-based methods are constrained by control-affine assumptions, slack variable introduction, and joint training instability; sampling-based methods lack safety and stability guarantees.

**Core Idea**: Use CLBF as a guidance function for diffusion models, replacing QP solving with diffusion sampling, and establish the intrinsic connection between diffusion sampling and Almost Lyapunov theory—even when the Lie derivative condition is violated in a small region, the global system can still maintain approximate exponential decay.

## Method

### Overall Architecture

S²Diff employs an iterative two-stage framework: (1) use the current CLBF as a guidance function to generate trajectory-level control policies satisfying safety and stability constraints via diffusion sampling; (2) update CLBF parameters using the sampled trajectories. The two stages alternate until convergence.

### Key Designs

1. **Probabilistic Modeling and CLBF-Guided Sampling**:

    - **Function**: Reformulates the constrained control optimization problem as a probabilistic sampling problem.
    - **Mechanism**: Defines the target trajectory distribution $p(U) \propto p_{\text{safe}}(U) \cdot p_{\text{stable}}(U) \cdot p_{\text{cost}}(U)$, where the safety term is a product of indicator functions for $V(x_t) \leq c$, the stability term takes a **soft-constraint** form $p_{\text{stable}} \propto \exp(-\frac{1}{\gamma_2}\sum_t \|[\mathcal{L}_f V(x_t) + \lambda V(x_t)]^+\|^2)$, and the cost term is an exponential of accumulated cost.
    - **Design Motivation**: The soft constraint corresponds to Almost Lyapunov theory—allowing the Lie derivative condition to be violated over a small-probability region. When temperature $\gamma_2$ is sufficiently small, the sampled policy can guarantee safety and stability at the trajectory level. Unlike QP methods that introduce slack variables and alter the optimization objective, soft constraints preserve the original problem and avoid high rejection rates.

2. **Monte Carlo-Based Diffusion Sampling**:

    - **Function**: Generates control trajectories via a DDPM-style forward-reverse diffusion process.
    - **Mechanism**: The forward process adds Gaussian noise to trajectories; the reverse process estimates the score function $\nabla \log p(U^i)$ via Sequential Monte Carlo (SMC) and denoises using an unbiased estimate of the posterior expectation $\mathbb{E}[U^0 | U^i]$.
    - **Design Motivation**: The sampling approach requires no control-affine structure, eliminates the need for slack variables, and directly handles general differentiable nonlinear dynamics. The generated clean trajectory $U^0$ directly yields the control policy $u_{1:T}$.

3. **Iterative CLBF Update**:

    - **Function**: Updates the neural network-parameterized CLBF using trajectory data obtained from diffusion sampling.
    - **Mechanism**: The loss function comprises six terms that respectively enforce: zero value at the equilibrium, positive definiteness, safe-set sublevel set containment, unsafe-set superlevel set containment, continuous-time Lie derivative decrease (via automatic differentiation), and discrete-time Lyapunov decrease.
    - **Design Motivation**: CLBF is parameterized by a general multi-layer neural network $\hat{V} = W_N \sigma_{N-1}(\cdots \sigma_1(W_1 x))$ rather than a traditional quadratic form, enabling handling of non-convex safe sets. Combining continuous and discrete Lie derivative constraints provides complementary benefits that improve learning quality.

### Theoretical Guarantees

**Theorem 3.1 (Almost Lyapunov Safety-Stability Guarantee)**: If CLBF $V$ satisfies $\min_u \mathcal{L}_f V(x) < -\lambda V(x)$ everywhere on the compact state space $\mathcal{X}$ except on a connected set $\Omega$ of volume less than $\epsilon$, then under the diffusion sampling policy the system satisfies $V(x_t) \leq \exp(-\lambda_1 t) V(x_0) + M \epsilon^{1/n}$. That is, the violation region introduces only an additive buffer term of order $O(\epsilon^{1/n})$, while the overall system maintains approximate exponential decay.

## Key Experimental Results

### Main Results

Comparison with rCLBF-QP, MPC, and MBD on 8 nonlinear dynamical systems (including both control-affine and non-control-affine):

| Task | Method | Safety Rate | ‖x−x⋆‖ | Inference Time (ms) |
|------|--------|-------------|---------|---------------------|
| Segway | rCLBF-QP | 90% | 0.11 | 5.2 |
| Segway | S²Diff | **100%** | 0.23 | 21.8 |
| Neural Lander | rCLBF-QP | 55% | 0.13 | 12.7 |
| Neural Lander | S²Diff | **100%** | **0.06** | 35.4 |
| 2D Quad | rCLBF-QP | 70% | 0.19 | 18.6 |
| 2D Quad | S²Diff | **95%** | **0.11** | 82.4 |
| F-16 (non-affine) | MBD | 100% | 68.34 | 611.3 |
| F-16 (non-affine) | S²Diff | **100%** | **47.61** | **257.2** |
| **Average** | rCLBF-QP | 78.75% | 0.384 | 10.06 |
| **Average** | S²Diff | **98.75%** | **0.226** | 45.64 |

### Ablation Study

| Configuration | Safety Rate | ‖x−x⋆‖ | Notes |
|--------------|-------------|---------|-------|
| γ₂=0.5 | 35% | 0.18 | Temperature too high; stability constraint too loose |
| γ₂=0.1 | **100%** | **0.06** | Optimal balance |
| γ₂=0.01 | 100% | 0.12 | Temperature too low; overly strict constraint leads to suboptimality |
| Discrete Lie derivative only (α₁=0) | 85% | 0.21 | Absence of auto-differentiation constraint reduces safety rate |
| Continuous Lie derivative only (α₂=0) | 100% | 0.15 | Absence of discrete constraint reduces precision |
| Dual Lie derivatives (α₁=α₂=1) | **100%** | **0.06** | Complementary constraints yield optimal performance |

### Key Findings

- The CLBF learned via diffusion sampling exhibits a larger contraction region than that learned via QP, providing stronger stability guarantees.
- CLBF guidance substantially improves diffusion sampling efficiency: the unguided MBD achieves only a 73.75% safety rate, which rises to 98.75% with CLBF guidance.
- Empirical violation rates are extremely low—Segway 0.5%, Neural Lander 1.1%, F-16 2.4%—validating the Almost Lyapunov theory.
- S²Diff is the only certificate-based method capable of handling the non-control-affine F-16 system (16-dimensional state space).

## Highlights & Insights

- This work is the first to establish a theoretical connection between diffusion sampling and Almost Lyapunov theory, providing a rigorous theoretical foundation for the empirical phenomenon of locally violating Lie derivative conditions during sampling.
- The framework is elegantly self-reinforcing: CLBF guides diffusion sampling, while the sampled trajectories in turn improve the CLBF.
- The method eliminates dependence on control-affine assumptions, slack variables, and QP solvers, substantially broadening its applicability.

## Limitations & Future Work

- Inference speed (average 45.6 ms) is faster than MPC (249.5 ms) but slower than QP (10 ms); policy distillation could be explored for acceleration.
- Validation is currently limited to model-based settings with known dynamics; extension to model-free settings is a natural future direction.
- Neural network parameterization of CLBF may face scalability challenges in extremely high-dimensional systems.

## Related Work & Insights

- The intersection of diffusion planning (Diffuser, SafeDiffuser) and certificate functions (CLBF, CBF) is a promising research direction.
- Almost Lyapunov theory provides a theoretical tool for "allowing occasional violations while guaranteeing global performance," transferable to other constrained optimization settings.
- For the model-based RL community: learned Lyapunov functions can serve as general-purpose safety guidance signals.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](training-free_constrained_generation_with_stable_diffusion_models.md)
- [\[NeurIPS 2025\] KLASS: KL-Guided Fast Inference in Masked Diffusion Models](klass_kl-guided_fast_inference_in_masked_diffusion_models.md)
- [\[NeurIPS 2025\] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting](one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)
- [\[ICLR 2026\] SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions](../../ICLR2026/image_generation/safeflowmatcher_safe_and_fast_planning_using_flow_matching_with_control_barrier_.md)

<!-- RELATED:END -->
