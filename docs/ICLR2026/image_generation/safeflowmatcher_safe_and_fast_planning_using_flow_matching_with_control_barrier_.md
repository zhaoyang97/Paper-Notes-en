---
title: >-
  [Paper Note] SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions
description: >-
  [ICLR 2026][Image Generation][Flow Matching] This paper proposes SafeFlowMatcher, a safe planning framework that integrates flow matching with Control Barrier Functions (CBF). Through a predictor-corrector (PC) integrato…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Flow Matching"
  - "Control Barrier Functions (CBF)"
  - "Safe Planning"
  - "Predictor-Corrector Integrator"
  - "Finite-Time Convergence"
date: 2026-05-08
content_hash: d877e79377712296
---

# SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions

**Conference**: ICLR 2026
**arXiv**: [2509.24243](https://arxiv.org/abs/2509.24243)  
**Code**: See project page  
**Area**: Image Generation
**Keywords**: Flow Matching, Control Barrier Functions (CBF), Safe Planning, Predictor-Corrector Integrator, Finite-Time Convergence

## TL;DR

This paper proposes SafeFlowMatcher, a safe planning framework that integrates flow matching with Control Barrier Functions (CBF). Through a predictor-corrector (PC) integrator, it decouples trajectory generation from safety certification, providing formal safety guarantees while preserving the efficiency of flow matching.

## Background & Motivation

### State of the Field

**Background**: Generative model-based path planning faces two major challenges:

**Lack of Safety**: The sampling dynamics of diffusion/flow matching models are governed by implicitly learned rules, which may produce trajectories that violate safety constraints.

**Efficiency Bottleneck**: Diffusion models require multi-step denoising, making real-time planning computationally expensive.

Issues with existing safe planning methods:

### Limitations of Prior Work

**Limitations of Prior Work**: Safety guidance methods **(e.g., classifier guidance)** rely on data-driven surrogates and cannot provide strong safety guarantees.

### Root Cause

**Key Challenge**: Certification methods **(e.g., SafeDiffuser)** impose constraints on intermediate latent states, leading to **semantic misalignment**—certification concerns the executed trajectory, yet constraints are applied to intermediate states that are never executed. This **distorts the learned flow** and creates **local traps** (trajectories become stuck near obstacle boundaries and fail to reach the goal).

## Method

### Overall Architecture

SafeFlowMatcher = Prediction Phase (unconstrained FM) + Correction Phase (CBF safety certification)

### Key Design 1: Predictor-Corrector (PC) Integrator

**Prediction Phase**: Starting from pure noise $\boldsymbol{\tau}_0^p \sim \mathcal{N}(0, I)$, Euler integration is applied to obtain candidate trajectories (typically $T^p = 1$ step):

$$\boldsymbol{\tau}_1^p = \Psi_{0 \to 1}^{(T^p)}(\boldsymbol{\tau}_0^p) = \boldsymbol{\tau}_1^\star + \varepsilon$$

**Correction Phase**: Starting from $\boldsymbol{\tau}_0^c = \boldsymbol{\tau}_1^p$, the trajectory is refined through two mechanisms:

(i) **Vanishing-Time Flow Dynamics (VTFD)**: Reduces discretization error.

$$\frac{d\boldsymbol{\tau}_t^c}{dt} = \alpha(1-t) v_t(\boldsymbol{\tau}_t^c; \theta) \triangleq \tilde{v}_t(\boldsymbol{\tau}_t^c; \theta)$$

The factor $(1-t)$ progressively suppresses the vector field as $t \to 1$, producing a contractive effect.

**Lemma 3** proves error decay: $\mathbf{e}_t = O((1-t)^2) + (\varepsilon + O(1))e^{-\alpha t}$

(ii) **CBF Safety Constraint**: Introduces a minimal perturbation $\Delta\mathbf{u}_t$:

$$\frac{d\boldsymbol{\tau}_t^c}{dt} = \tilde{v}_t(\boldsymbol{\tau}_t^c; \theta) + \Delta\mathbf{u}_t$$

### Key Design 2: Barrier Certificate

Define the robust safe set $\mathcal{C}_\delta = \{\boldsymbol{\tau}^{c,k} \in \mathcal{D} \mid b(\boldsymbol{\tau}^{c,k}) \geq \delta\}$.

**Theorem 1 (Forward Invariance)**: A control $\mathbf{u}_t$ satisfying the following barrier certificate guarantees finite-time flow invariance:

$$\nabla b(\boldsymbol{\tau}_t^{c,k})^\top \mathbf{u}_t^k + \epsilon \cdot \text{sgn}(b(\boldsymbol{\tau}_t^{c,k}) - \delta)|b(\boldsymbol{\tau}_t^{c,k}) - \delta|^\rho + w_t^k r_t^k \geq 0$$

**Proposition 1 (Finite Convergence Time)**:

$$T \leq t_w + \frac{(\delta - b(\boldsymbol{\tau}_{t_w}^{c,k}))^{1-\rho}}{\epsilon(1-\rho)}$$

### CBF Quadratic Program

Each waypoint independently solves a QP:

$$\mathbf{u}_t^{k*} = \arg\min_{\mathbf{u}_t^k} \|\mathbf{u}_t^k - \tilde{v}_t^k\|^2 \quad \text{s.t. CBF constraint}$$

## Experiments

### Experimental Setup

- Maze navigation (Maze2D)
- Locomotion control
- Robot manipulation

### Main Results

Compared to baseline methods, SafeFlowMatcher achieves:
- **Faster** inference: flow matching generates high-quality trajectories in one or few steps.
- **Smoother** trajectories: the PC integrator avoids path distortion caused by intermediate-state constraints.
- **Safer** behavior: CBF provides formal safety guarantees.

### Ablation Study

The individual contributions of the PC integrator and the barrier certificate are validated:
- Removing the correction phase → degraded safety.
- Removing VTFD → degraded trajectory quality.
- Applying constraints directly on intermediate states → local trap problem.

### Key Findings

- **Imposing safety constraints only on the executed trajectory** is the critical design choice—it avoids distribution shift and local traps.
- The decay factor in VTFD effectively reduces prediction error.
- The relaxation weight $w_t^k$ in CBF provides numerical stability during the early correction phase.

## Highlights & Insights

- Elegantly combines flow matching (efficiency) with CBF (safety certification), with solid theoretical guarantees.
- The decoupled design of the PC integrator fundamentally resolves the local trap problem present in existing methods.
- Strong theoretical contributions: forward invariance theorem and finite convergence time guarantee.
- High framework generality: applicable to scenarios where system dynamics and cost maps are unknown.

## Limitations & Future Work

- Construction of the CBF depends on the definition of the barrier function $b$; applicability to complex environments warrants further investigation.
- Computational cost of QP solving increases with the number of waypoints and constraints.
- Assumes the flow matching model has been reasonably pre-trained.
- Validation is limited to specific tasks; high-dimensional state spaces are not addressed.

## Related Work & Insights

- **Flow matching for planning**: FlowPolicy, EquiBot, and related works applying FM to robot control.
- **Safe diffusion planning**: SafeDiffuser applies CBF constraints on intermediate states.
- **CBF methods**: Finite-time convergence CBF, learned CBF, and related approaches.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The decoupled design of the PC integrator is elegant and addresses a fundamental problem.
- Theoretical Rigor: ⭐⭐⭐⭐⭐ — Complete proofs of forward invariance and convergence.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-task validation with thorough ablation studies.
- Value: ⭐⭐⭐⭐ — Direct applicability to safety-critical robot deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICML 2026\] LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](../../ICML2026/image_generation/lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[ICLR 2026\] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)
- [\[ICLR 2026\] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)

</div>

<!-- RELATED:END -->
