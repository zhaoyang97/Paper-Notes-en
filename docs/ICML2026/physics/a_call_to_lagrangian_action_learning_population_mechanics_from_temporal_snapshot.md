---
title: >-
  [Paper Note] A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots
description: >-
  [ICML 2026][Physics & Scientific Computing][Population Dynamics] Starting from the principle of least action, this paper proposes the Wasserstein Lagrangian Mechanics (WLM) framework to learn second-order population dyna…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Population Dynamics"
  - "Wasserstein Mechanics"
  - "Least Action Principle"
  - "Second-order Dynamics"
  - "Neural Potential"
date: 2026-05-08
content_hash: b05bb49a5ce27294
---

# A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots

**Conference**: ICML 2026  
**arXiv**: [2605.08550](https://arxiv.org/abs/2605.08550)  
**Code**: https://github.com/guanton/WLM  
**Area**: Diffusion Models / Dynamics Learning  
**Keywords**: Population Dynamics, Wasserstein Mechanics, Least Action Principle, Second-order Dynamics, Neural Potential

## TL;DR
Starting from the principle of least action, this paper proposes the Wasserstein Lagrangian Mechanics (WLM) framework to learn second-order population dynamics rather than traditional first-order gradient flows. This enables the capture of richer collective phenomena such as periodicity and rotation, and allows for interpolation and future forecasting without requiring a reference process.

## Background & Motivation

**Background**: Traditional population dynamics modeling (ranging from molecular diffusion to cell differentiation and collective biological behavior) commonly adopts the Wasserstein gradient flow paradigm. Starting from a free energy functional, these models characterize purely dissipative evolutionary processes.

**Limitations of Prior Work**: Gradient flows are inherently first-order systems that eventually progress toward an equilibrium state. However, real-world population dynamics often exhibit non-equilibrium phenomena such as periodicity, rotation, and oscillation—for instance, vortices or Boids flocking behaviors—which transcend the energy minimization framework.

**Key Challenge**: While the mathematical foundation of gradient flows is solid and optimization algorithms are mature, their expressive power is constrained. Given only temporal snapshots of marginal distributions (where individual trajectories cannot be tracked), how can broader second-order dynamics be inferred without pre-specifying a Lagrangian?

**Goal**: To revisit population models from the principle of least action and replace first-order gradient flows with second-order systems, thereby creating a unified framework covering classical mechanics, quantum mechanics, and gradient flows.

**Key Insight**: By defining population "coordinates" and a Lagrangian in the Wasserstein-2 distance space $\mathcal{P}_2(\mathbb{R}^d)$, equations of motion can be derived through the variational principle, establishing a mechanical viewpoint where the "population drives its own evolution."

**Core Idea**: Parameterize second-order dynamics using a population potential functional $\mathcal{U}[\rho_t]$ and a damping coefficient $\gamma$. This allows the framework to encompass gradient flows (overdamped limit $\gamma\to\infty$), conservative classical mechanics ($\gamma=0$), and quantum mechanics.

## Method

### Overall Architecture

The method consists of a theoretical layer and an algorithmic layer.

**Theoretical Layer**: Population dynamics in $\mathcal{P}_2(\mathbb{R}^d)$ are represented via the continuity equation $\dot{\rho}_t = -\nabla\cdot(\rho_t \nabla s_t)$, where $s_t$ is a time-varying potential function. By introducing a damped Wasserstein Lagrangian $\mathcal{L}[\rho_t, s_t, t] = e^{\gamma t}(\frac{1}{2}\int\|\nabla s_t\|^2 \rho_t dx - \mathcal{U}[\rho_t])$ and applying the variational principle, particle-level equations of motion are derived: $\frac{d}{dt}x_t = \nabla s_t(x_t)$ and $\frac{d}{dt}v_t = -\nabla\frac{\delta\mathcal{U}[\rho_t]}{\delta\rho_t}(x_t) - \gamma v_t$. This constitutes a population version of "generalized Newton's laws," where particle acceleration is driven by the gradient of the population potential and velocity damping.

**Algorithmic Layer**: The neural mechanics model directly learns the potential functional $\Psi_\theta$. Proposition 3.1 converts functional derivatives into neural network gradients with respect to particle coordinates. Time integration utilizes a Leapfrog integrator, and differences between predicted and observed margins are measured using Sinkhorn divergence, optimized via end-to-end backpropagation.

### Key Designs

1. **Hamiltonian Framework with Population Potential Parameterization**:
    - **Function**: Converts the non-computable functional gradient $\frac{\delta\mathcal{U}[p]}{\delta p}(x)$ into a differentiable neural network parameter gradient.
    - **Mechanism**: For an empirical distribution $\hat{p} = \frac{1}{N}\sum_i\delta_{x^{(i)}}$, the identity $\nabla_{x^{(j)}}\Psi(x^{(1)},\ldots,x^{(N)}) = \frac{1}{N}\nabla_x\frac{\delta\mathcal{U}[p]}{\delta p}(x^{(j)})\big|_{p=\hat{p}}$ holds. Thus, the potential gradient is obtained directly via automatic differentiation without explicit functional derivation.
    - **Design Motivation**: Bypasses the analytical difficulty of functional derivatives in Wasserstein space by leveraging the discrete structure of empirical measures, allowing the learned scalar function $\Psi_\theta$ to naturally define the potential field.

2. **Mixed Batches and Reference-Free Learning**:
    - **Function**: Flexibly handles multi-source data while preserving second-order dynamics without pre-specifying a reference SDE or OT plan.
    - **Mechanism**: Starting from arbitrary initial conditions $(p_0, v_0)$, a single Lagrangian $(\Psi_\theta, \gamma)$ covers all observation times. The loss function accumulates Sinkhorn divergence between each pair of predicted and observed margins. Unlike AM, JKOnet, or Flow Matching, this method does not rely on any reference process.
    - **Design Motivation**: Provides a general modeling framework for scenarios lacking domain priors, enabling application to unknown physical or biological systems.

3. **Learnable Damping and Unified Expression**:
    - **Function**: Automatically determines the nature of the dynamics between gradient flows and classical mechanics by learning the damping coefficient $\gamma$.
    - **Mechanism**: $\gamma=0$ corresponds to a conservative system (classical mechanics); $\gamma\to\infty$ corresponds to the overdamped limit (gradient flow); intermediate values $\gamma>0$ correspond to second-order systems with viscosity but retained inertia. On gradient flow data, the model automatically learns $\gamma\geq 500$ (the strong damping limit).
    - **Design Motivation**: A single learnable parameter allows the model to adaptively switch between various mechanical paradigms without manual selection.

### Loss & Training

For each observation interval $t_i\to t_{i+1}$, the system is advanced via Leapfrog to obtain the predicted margin $\hat{p}_{t_{i+1}}$, and the Sinkhorn divergence $\sum_i \mathcal{S}_\epsilon(\hat{p}_{t_{i+1}}, p_{t_{i+1}})$ is accumulated against the observed margin. Optimization variables include the potential network parameters $\theta$, the initial velocity field $v_0$, and the damping coefficient $\gamma$.

## Key Experimental Results

### Main Results

| Task | Method | Metric | Result | Note |
|------|------|------|------|------|
| Gradient Flow SDE (paired) | nn-APPEX | Forecast $W_1$ | 0.131±0.006 | Traditional gradient flow method |
| Gradient Flow SDE (paired) | WLM (learnable $\gamma$) | Forecast $W_1$ | 0.137±0.012 | Similar performance without priors |
| Gradient Flow SDE (unpaired) | JKOnet* | Train $W_1$ | 0.236±0.040 | Performance fails when paired assumption is broken |
| Gradient Flow SDE (unpaired) | WLM (learnable $\gamma$) | Train $W_1$ | 0.068±0.004 | **Significantly outperforms first-order methods** |
| Gulf Vortex Interpolation | WLM | Multi-step $W_1$ | 0.060–0.068 | Outperforms AM/UAM/sAM without priors |
| Gulf Vortex Forecast | WLM | Forecast $W_1$ | 0.567±0.014 | Only reference-free method capable of forecasting |
| Embryonic scRNA | WLM | Interpolation $W_1$ | Better than GF+OT | Effective on high-dimensional real data |
| Boids Collective Behavior | WLM | Forecasting | Wins overall | Captures collective oscillations |

### Ablation Study

| Configuration | Unpaired SDE Forecast $W_1$ | Vortex Forecast $W_1$ | Note |
|------|------|------|------|
| Full WLM ($\Psi_\theta$+Learnable $\gamma$) | 0.246±0.026 | 0.567±0.014 | Complete model |
| WLM ($\gamma=0$ Conservative) | 0.346±0.045 | 0.689±0.120 | Poor performance without damping |
| WLM (Fixed excessive $\gamma$) | 0.298±0.031 | 0.612±0.025 | Overdamping deviates from gradient flow |

### Key Findings
- **2nd-order vs 1st-order**: WLM reduces $W_1$ from 1.618 to 0.246 on unpaired data, demonstrating that second-order frameworks are more robust to confounded trajectories.
- **Adaptability of Learnable Damping**: The model automatically converges to the overdamped limit on gradient flow data, while maintaining moderate damping to capture rotation in vortex and Boids data.
- **Forecasting vs Interpolation**: WLM is the only method that can forecast future vortex states without using a reference process.
- **High-Dimensional Scalability**: Performance remains effective on scRNA data with thousands of dimensions, confirming the universality of neural potential parameterization.

## Highlights & Insights
- **Theoretical Depth**: Fully derives a Hamiltonian framework in Wasserstein space, unifying gradient flows, classical mechanics, and quantum mechanics.
- **Methodological Ingenuity**: Proposition 3.1 transforms non-computable functional gradients into neural network parameter gradients, making the learning process feasible within standard autograd frameworks.
- **Reference-Free Learning**: No requirement for pre-specified reference SDEs or OT plans ensures true universality for unknown systems.
- **True Forecasting**: By learning the underlying dynamical equations, the model can extrapolate beyond the training time window, representing a qualitative advancement over interpolation-only methods.

## Limitations & Future Work
- Assumes snapshots represent the same population at multiple times; cannot handle matching snapshots from entirely different populations.
- The boundaries of the expressive power of neural potential networks lack theoretical characterization.
- Particle scales in experiments are relatively small (~1000); scalability for large-scale applications (millions of particles) has not been verified.
- The sample complexity of potential learning in high-dimensional scenarios may be high, requiring more efficient approximations (e.g., inducing points, sparse representations).

## Related Work & Insights
- **vs Gradient Flow Methods (JKOnet*, nn-APPEX)**: Gradient flows are a special case where $\gamma\to\infty$; WLM captures rotation and oscillation through a general second-order framework.
- **vs Flow Matching / Diffusion Interpolation**: Flow matching performs interpolation but relies on reference processes; WLM is reference-free and enables forecasting, at the cost of needing more data to learn the potential.
- **vs Action Matching (AM/UAM/sAM)**: While similar in spirit, AM is limited to deterministic dynamics; WLM unifiedly handles both stochastic and deterministic cases.
- **Insight**: The potential learning paradigm has transfer value for physical simulation, molecular dynamics, and social systems, offering better interpretability than direct SDE learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rewriting population dynamics from the principle of least action with a second-order, reference-free approach is a significant paradigm extension.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers synthetic SDEs, physical vortices, biological scRNA, and collective behavior, with clear ablations, though sample scales remain small.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivation, complete algorithmic pseudocode, and highly expressive visualizations.
- Value: ⭐⭐⭐⭐⭐ Establishes a new framework for scientific modeling and dynamics learning with high potential for interdisciplinary impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ANTIC: Adaptive Neural Temporal In-situ Compressor](antic_adaptive_neural_temporal_in-situ_compressor.md)
- [\[ICML 2026\] BALLAST: Bayesian Active Learning with Look-ahead Amendment for Sea-drifter Trajectories under Spatio-Temporal Vector Fields](ballast_bayesian_active_learning_with_look-ahead_amendment_for_sea-drifter_traje.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](topology-preserving_neural_operator_learning_via_hodge_decomposition.md)
- [\[ICML 2026\] Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs](hermite-ngp_gradient-augmented_hash_encoding_for_learning_pdes.md)
- [\[ICML 2026\] Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting](learning_to_refine_spectral-decoupled_iterative_refinement_framework_for_precipi.md)

</div>

<!-- RELATED:END -->
