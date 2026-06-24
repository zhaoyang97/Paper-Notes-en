---
title: >-
  [Paper Note] How to Build a Consistency Model: Learning Flow Maps via Self-Distillation
description: >-
  [NeurIPS 2025][Model Compression][Flow Map] This paper proposes a unified self-distillation framework for directly learning flow maps (the generalized form of consistency models). By exploiting the tangent condition, any distillation scheme is converted into a direct training algorithm that requires no pretrained teacher. Three algorithm families are derived (Eulerian / Lagrangian / Progressive), among which the Lagrangian method avoids spatial gradients and bootstrapping…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Flow Map"
  - "Consistency Model"
  - "Self-Distillation"
  - "Accelerated Inference"
  - "Flow Matching"
date: 2026-05-08
content_hash: a84042576342eb9a
---

# How to Build a Consistency Model: Learning Flow Maps via Self-Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2505.18825](https://arxiv.org/abs/2505.18825)  
**Code**: [nmboffi/flow-maps](https://github.com/nmboffi/flow-maps)  
**Area**: Image Generation
**Keywords**: Flow Map, Consistency Model, Self-Distillation, Accelerated Inference, Flow Matching

## TL;DR
This paper proposes a unified self-distillation framework for directly learning flow maps (the generalized form of consistency models). By exploiting the tangent condition, any distillation scheme is converted into a direct training algorithm that requires no pretrained teacher. Three algorithm families are derived (Eulerian / Lagrangian / Progressive), among which the Lagrangian method avoids spatial gradients and bootstrapping, achieving the most stable training and best performance.

## Background & Motivation
**Background**: Flow- and diffusion-based generative models have achieved state-of-the-art results in vision, language, and protein structure prediction, but inference requires repeated ODE/SDE solving, incurring substantial computational cost that limits real-time applications.

**Need for Accelerated Inference**: Consistency models learn flow maps (solution operators of probability flow equations) to enable one- or few-step sampling, accelerating inference by 10–100×. However, existing methods lack a unified mathematical formulation and suffer from training instability and high engineering complexity.

**Distillation vs. Direct Training**: Distillation methods (e.g., progressive distillation) require a two-stage pipeline—first training a teacher, then a student—where student performance is bounded by teacher quality. Direct training methods (e.g., consistency training) face optimization instability and require carefully designed annealing schedules and gradient clipping.

**Key Challenge**: The absence of a unified mathematical framework for efficiently learning flow maps causes existing methods to appear as disconnected algorithms with unclear design principles.

**Key Insight**: The paper exploits the tangent condition—that the time derivative of the flow map on the diagonal $s=t$ recovers the velocity field $b_t$—to convert distillation schemes into self-distillation, training a single network to jointly learn the velocity field and the flow map.

## Core Problem
- **Does a principled methodology for flow map training exist that is as clean as standard flow matching?**
- Can flow maps be trained efficiently and stably without a pretrained teacher?
- How do different mathematical representations (Eulerian / Lagrangian / Semigroup) affect training stability and performance?

## Method

### Theoretical Foundation: Three Equivalent Characterizations of Flow Maps

Let $X_{s,t}(x)$ denote the flow map of the probability flow $\dot{x}_t = b_t(x_t)$, satisfying the jump condition $X_{s,t}(x_s) = x_t$. Three equivalent characterizations are given:

1. **Lagrangian condition** (ODE perspective):
   $$\partial_t X_{s,t}(x) = v_{t,t}(X_{s,t}(x))$$
   The rate of change of the flow map in the $t$ direction equals the velocity field at the endpoint—an ODE in $t$.

2. **Eulerian condition** (PDE perspective):
   $$\partial_s X_{s,t}(x) + \nabla X_{s,t}(x)\, v_{s,s}(x) = 0$$
   Involves the spatial Jacobian $\nabla X_{s,t}$ of the flow map—a PDE in $s$.

3. **Semigroup condition** (composability):
   $$X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)$$
   Two successive jumps can be replaced by a single jump.

### Tangent Condition: Linking the Velocity Field and the Flow Map

The key mathematical insight is:
$$\lim_{s \to t} \partial_t X_{s,t}(x) = b_t(x)$$

That is, the tangent direction of the flow map on the diagonal $s=t$ exactly recovers the velocity field. This implies that the flow map implicitly encodes a velocity model.

Building on this, the following parameterization is adopted:
$$\hat{X}_{s,t}(x) = x + (t-s)\, \hat{v}_{s,t}(x)$$

where $v_{s,t}$ has the geometric interpretation of the "slope" of the chord from $x_s$ to $x_t$ along the ODE trajectory, satisfying $v_{t,t}(x) = b_t(x)$.

### Self-Distillation Framework

The total loss consists of two components:
$$\mathcal{L}_{\text{SD}}(\hat{v}) = \mathcal{L}_b(\hat{v}) + \mathcal{L}_D(\hat{v})$$

- **Diagonal loss** $\mathcal{L}_b$: learns the velocity field via standard flow matching at $s=t$
  $$\mathcal{L}_b = \int_0^1 \mathbb{E}_{x_0,x_1}\left[|\hat{v}_{t,t}(I_t) - \dot{I}_t|^2\right] dt$$

- **Off-diagonal loss** $\mathcal{L}_D$: learns the flow map at $s \neq t$ via a distillation target, with three variants:

| Method | Loss Source | Spatial Gradients | Bootstrapping |
|--------|-------------|-------------------|---------------|
| **LSD** (Lagrangian) | Lagrangian ODE residual | ❌ | ❌ |
| **ESD** (Eulerian) | Eulerian PDE residual | ✅ (Jacobian) | ❌ |
| **PSD** (Progressive) | Semigroup consistency | ❌ | ✅ (small → large steps) |

### LSD (Lagrangian Self-Distillation) — Best-Performing Method

$$\mathcal{L}_{\text{LSD}} = \int_0^1 \int_0^t \mathbb{E}\left[|\partial_t \hat{X}_{s,t}(I_s) - \hat{v}_{t,t}(\hat{X}_{s,t}(I_s))|^2\right] ds\, dt$$

- Compares the time derivative of the flow map against the velocity field prediction at the flow map endpoint
- **No spatial gradients required** (avoids Jacobian computation), yielding smoother training gradients
- **No bootstrapping** (does not compose small steps into large ones), preventing error accumulation
- Applies stop-gradient so that information flows unidirectionally from the diagonal (velocity field) to the off-diagonal (flow map)

### Key Engineering Details

1. **Adaptive loss weighting**: Introduces learnable weights $w_{s,t}$ (extending EDM2 to two time dimensions):
   $$\mathcal{L} = \mathbb{E}_{p_{s,t}}\left[e^{-w_{s,t}} \cdot \ell_{s,t} + w_{s,t}\right]$$
   $w_{s,t}$ estimates the log-variance of the loss, normalizing gradient contributions across different $(s,t)$ pairs.

2. **Time sampling**: Uses a mixture distribution $p_{s,t} = \eta\, U_d + (1-\eta)\, U_{od}$, with $\eta=0.75$ allocating 75% of the batch to diagonal flow matching and 25% to off-diagonal distillation. Since distillation targets are more expensive (requiring multiple network evaluations and JVPs), $\eta$ also controls training cost.

3. **PSD scaling preconditioning**: The PSD loss is reformulated to eliminate the $(t-s)^2$ factor, avoiding large effective learning rate disparities across time steps.

### Unification of Existing Methods

By selecting different distillation targets and teachers, the framework recovers:
- **Consistency Models** (Song et al., 2023) → special case of the Eulerian perspective
- **Consistency Trajectory Models** (Kim et al., 2024) → two-time Eulerian distillation
- **Shortcut Models** (Frans et al., 2024) → discretized semigroup condition
- **Progressive Distillation** (Salimans & Ho, 2022) → distillation variant of PSD
- **Mean Flow** / **Align Your Flow** → specific instances of the Eulerian formulation

## Key Experimental Results

LSD, ESD, PSD-M (midpoint), and PSD-U (uniform) are compared on Checkerboard (2D synthetic), CIFAR-10, CelebA-64, and AFHQ-64 under a fixed training budget:

| Dataset | Method | 1-step | 2-step | 4-step | 8-step | 16-step |
|---------|--------|--------|--------|--------|--------|---------|
| CIFAR-10 (FID↓) | **LSD** | **8.10** | **4.37** | **3.34** | **3.33** | **3.57** |
| | PSD-M | 12.81 | 8.43 | 5.96 | 5.07 | 4.64 |
| | PSD-U | 13.61 | 7.95 | 6.03 | 5.32 | 5.16 |
| CelebA-64 (FID↓) | **LSD** | **12.22** | **5.74** | **3.18** | **2.18** | **1.96** |
| | PSD-M | 19.64 | 11.75 | 7.89 | 6.06 | 5.09 |
| AFHQ-64 (FID↓) | **LSD** | **11.19** | **7.78** | **7.00** | **5.89** | **5.61** |
| | PSD-U | 14.50 | 10.73 | 10.99 | 12.02 | 11.47 |

- **ESD is unstable on image datasets**: gradient norms far exceed those of LSD/PSD, eventually diverging, precluding FID reporting
- **LSD achieves the best results across all datasets and all step counts**
- On CIFAR-10, LSD reaches FID 3.34 at 4 steps, nearly matching the 8-step FID of 3.33, indicating early saturation
- On Checkerboard, LSD saturates in KL divergence (0.07) at $N \geq 4$, while other methods require 16 steps to approach this level

## Theoretical Guarantees

Wasserstein bounds are proved for both LSD and ESD:
- **LSD**: $W_2^2(\hat{\rho}_1, \rho_1) \leq 4 e^{1+2\hat{L}} \varepsilon$
- **ESD**: $W_2^2(\hat{\rho}_1, \rho_1) \leq 2e(1+e^{2\hat{L}}) \varepsilon$

where $\varepsilon$ is the loss value and $\hat{L}$ is the Lipschitz constant of $v_{t,t}$. Due to error accumulation and distribution shift, no analogous guarantee can be established for PSD.

## Highlights & Insights
1. **Unified framework**: Subsumes consistency models, progressive distillation, and shortcut models into a single mathematical system, revealing shared design principles
2. **Self-distillation eliminates teacher dependence**: The tangent condition enables a single network to jointly learn the velocity field and flow map, removing the need for two-stage training
3. **Superiority of the Lagrangian method**: LSD avoids spatial gradients and bootstrapping, achieving the best stability and performance across all experiments, and reveals that the root cause of consistency training instability is the spatial Jacobian introduced by the Eulerian perspective
4. **Theory-practice alignment**: LSD enjoys tighter Wasserstein bounds and empirically outperforms alternatives; PSD lacks theoretical guarantees and is correspondingly weaker in practice
5. **Plug-and-play**: The framework extends directly to conditional generation and CFG guidance

## Limitations & Future Work
1. **Computational cost not fully explored**: Due to resource constraints, a systematic ablation over all design choices (parameterization, architecture, stop-gradient schemes) was not conducted
2. **Unconditional generation only**: Validation in class-conditional or text-to-image settings is absent; CFG guidance is left to future work
3. **Limited resolution**: Experiments reach at most 64×64; validation at 256/512 resolution or in latent space has not been performed
4. **Single-model capacity pressure**: Using one network to represent both the velocity field ($s=t$) and the flow map ($s \neq t$) may require larger model capacity; dual-model and higher-order parameterizations are discussed but not experimentally evaluated
5. **Room for improvement in few-step performance**: The 1-step FID on CIFAR-10 (8.10) still lags behind state-of-the-art consistency models

## Related Work & Insights

| Method | Teacher Required | Mathematical Basis | Training Stability | Multi-step Inference |
|--------|-----------------|-------------------|-------------------|----------------------|
| Consistency Model (CT) | ❌ | Eulerian PDE | Poor (requires careful engineering) | ❌ (single time) |
| Consistency Distillation (CD) | ✅ | Eulerian PDE | Moderate | ❌ |
| Progressive Distillation | ✅ | Semigroup | Moderate | ✅ |
| Shortcut Models | ❌ | Discrete Semigroup | Moderate | ✅ |
| **Ours (LSD)** | **❌** | **Lagrangian ODE** | **Best** | **✅** |

The key contribution is revealing that the instability of the Eulerian perspective (used by consistency models) stems from spatial Jacobian computation, a difficulty entirely avoided by the Lagrangian perspective.

## Loss & Training

### Training Procedure (Algorithm 1)

1. Each mini-batch is split according to ratio $\eta$ into diagonal and off-diagonal samples
2. **Diagonal part**: Sample $M_d = \lfloor \eta M \rfloor$ pairs $(x_0, x_1)$ and times $t$; compute interpolant $I_t$ and target $\dot{I}_t$; train $v_{t,t}$ with the standard flow matching loss
3. **Off-diagonal part**: Sample $M_o = M - M_d$ pairs and time pairs $(s,t)$; compute interpolant $I_s$; train $v_{s,t}$ with the chosen distillation target (LSD/ESD/PSD)
4. Both loss components are summed and back-propagated jointly; adaptive weights $w_{s,t}$ are updated simultaneously

### Stop-Gradient Strategy

- In LSD, stop-gradient is applied to the velocity field prediction $\hat{v}_{t,t}(\hat{X}_{s,t}(I_s))$, ensuring that information flows unidirectionally from the diagonal (which has external supervision $\dot{I}_t$) to the off-diagonal
- This is equivalent to treating the model's own diagonal predictions as a "frozen teacher," hence the term "self-distillation"
- In PSD, stop-gradient is applied to the "teacher" side of the composed step ($\hat{X}_{u,t}(\hat{X}_{s,u}(I_s))$)
- In ESD, stop-gradient is applied to $v_{s,s}(I_s)$

### Inference: Flexible Multi-Step Sampling

- **1-step**: $\hat{x}_1 = \hat{X}_{0,1}(x_0) = x_0 + v_{0,1}(x_0)$, completing sampling in a single forward pass
- **N-step**: $[0,1]$ is partitioned into $N$ uniform segments and the maps $\hat{X}_{t_i, t_{i+1}}$ are composed sequentially, trading compute for quality
- Multi-step inference requires no additional training; the model natively supports arbitrary step counts—a core advantage of the two-time parameterization $v_{s,t}$

## My Notes

### Why Lagrangian Outperforms Eulerian — A Gradient-Level Intuition

The Eulerian condition $\partial_s X_{s,t} + \nabla X_{s,t} \cdot v_{s,s} = 0$ requires computing the spatial Jacobian $\nabla X_{s,t}$ of the flow map. In a neural network, this Jacobian is a $d \times d$ matrix (where $d$ is the data dimension); even when computed efficiently via JVPs, its gradients still involve second-order derivatives (Hessian-vector products), leading to unstable gradient norms. In the CIFAR-10 experiments, ESD gradient norms exceed those of LSD by several orders of magnitude, ultimately causing training to diverge.

By contrast, the Lagrangian condition $\partial_t X_{s,t}(x) = v_{t,t}(X_{s,t}(x))$ requires only the time derivative of the flow map in $t$ (computed efficiently via JVP) and an evaluation of the velocity field at the flow map endpoint—spatial Jacobians are completely absent. This fundamentally explains why consistency training (Eulerian perspective) demands careful engineering while LSD works out of the box.

### Deep Connection with Shortcut Models

Shortcut Models (Frans et al., 2024) are essentially a discretization of the semigroup condition: training is restricted to a finite set of step sizes $\{0, \Delta, 2\Delta, \ldots\}$, with large steps bootstrapped from small ones via self-consistency. PSD in this paper is the continuous generalization; however, experiments show that PSD underperforms LSD because the semigroup condition incurs error accumulation—the accuracy of large steps depends on that of small steps, creating a chain dependency.

This suggests that **avoiding bootstrapping is key to training stability**. In LSD, each $(s,t)$ pair is trained independently without relying on the prediction quality at other step sizes, making it more robust.

### Single-Model vs. Dual-Model Trade-off

This paper uses a single network $v_{s,t}$ to simultaneously represent the velocity field ($s=t$) and the flow map ($s \neq t$). The advantage is parameter sharing and training efficiency, but it requires the network to have sufficient capacity to jointly fit two qualitatively different functions. The paper discusses a dual-model scheme and a higher-order parameterization $X_{s,t}(x) = x + (t-s)b_s(x) + \frac{1}{2}(t-s)^2 \psi_{s,t}(x)$ (analogous to a second-order ODE solver expansion), but neither is experimentally evaluated. This is a direction worth exploring, particularly when model capacity becomes a bottleneck in high-resolution generation.

### Potential Extensions

- **Latent space application**: Combining with Latent Diffusion / SDXL to train flow maps in latent space, potentially enabling 1–4-step high-quality text-to-image generation
- **CFG distillation**: The paper derives the theoretical formula for the CFG flow map $v_{t,t}(x;\alpha,c) = q_t(x;\alpha,c)$ but does not experiment with it; incorporating guidance scale $\alpha$ as an additional conditioning variable is a straightforward extension
- **Comparison with InstaFlow / SDXL-Turbo**: These methods achieve few-step generation via progressive distillation; LSD offers a direct training alternative that eliminates dependence on a teacher model

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Unified framework + three algorithm families + novel Lagrangian method; outstanding theoretical depth and breadth
- Experimental Thoroughness: ⭐⭐⭐⭐ — Fair multi-dataset comparison, but limited resolution and absence of conditional generation experiments
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous mathematical derivations, strong systematicity, clear illustrations
- Value: ⭐⭐⭐⭐⭐ — Provides principled design guidelines for the consistency model family with lasting impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency](robustifying_learning-augmented_caching_efficiently_without_compromising_1-consi.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)
- [\[NeurIPS 2025\] SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning](scan_self-denoising_monte_carlo_annotation_for_robust_process_reward_learning.md)
- [\[ICML 2026\] FedSDR: Federated Self-Distillation with Rectification](../../ICML2026/model_compression/fedsdr_federated_self-distillation_with_rectification.md)
- [\[NeurIPS 2025\] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models](vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)

</div>

<!-- RELATED:END -->
