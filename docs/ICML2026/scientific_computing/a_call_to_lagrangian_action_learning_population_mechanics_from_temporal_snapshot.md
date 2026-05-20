---
title: >-
  [Paper Note] A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots
description: >-
  [ICML 2026][Scientific Computing][Population Dynamics] Starting from the principle of least action, this paper proposes the Wasserstein Lagrangian Mechanics (WLM) framework to learn second-order population dynamics inste…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Population Dynamics"
  - "Wasserstein Mechanics"
  - "Principle of Least Action"
  - "Second-Order Dynamics"
  - "Neural Potential"
date: 2026-05-08
content_hash: 12ce1a5b45918f21
---

# A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots

**Conference**: ICML 2026  
**arXiv**: [2605.08550](https://arxiv.org/abs/2605.08550)  
**Code**: https://github.com/guanton/WLM  
**Area**: Diffusion Models / Dynamics Learning  
**Keywords**: Population Dynamics, Wasserstein Mechanics, Principle of Least Action, Second-Order Dynamics, Neural Potential

## TL;DR
Starting from the principle of least action, this paper proposes the Wasserstein Lagrangian Mechanics (WLM) framework to learn second-order population dynamics instead of traditional first-order gradient flows. This enables the capture of richer collective phenomena such as periodicity and rotation, and allows interpolation and future prediction without requiring a reference process.

## Background & Motivation

**Background**: Traditional population dynamics modeling (from molecular diffusion to cell differentiation and collective biological behavior) commonly adopts the Wasserstein gradient flow paradigm, starting from a free energy functional to model purely dissipative evolution processes.

**Limitations of Prior Work**: Gradient flows are essentially first-order systems that ultimately converge to an equilibrium state. However, real-world population dynamics often exhibit non-equilibrium phenomena such as periodicity, rotational motion, and oscillations—e.g., vortices, Boids flocking behavior—which go beyond the energy minimization framework.

**Key Challenge**: Although gradient flows have solid mathematical foundations and mature optimization algorithms, their expressive power is limited. When only temporal snapshots of marginal distributions are available (without tracking individual trajectories), how can one infer more general second-order dynamics without pre-specifying the Lagrangian?

**Goal**: To revisit population models from the principle of least action, replacing first-order gradient flows with second-order systems, thereby unifying classical mechanics, quantum mechanics, and even gradient flows within a single framework.

**Key Insight**: Define population "coordinates" and Lagrangian in the Wasserstein-2 distance space $\mathcal{P}_2(\mathbb{R}^d)$, and derive Hamiltonian equations of motion via variational principles, forming a mechanics perspective where the population drives its own evolution.

**Core Idea**: Parameterize second-order dynamics using a population potential functional $\mathcal{U}[\rho_t]$ and damping coefficient $\gamma$, encompassing gradient flows (overdamped limit $\gamma\to\infty$), conservative classical mechanics ($\gamma=0$), quantum mechanics, and more.

## Method

### Overall Architecture

The method consists of a theoretical layer and an algorithmic layer.

**Theoretical Layer**: In $\mathcal{P}_2(\mathbb{R}^d)$, population dynamics are represented as the continuity equation $\dot{\rho}_t = -\nabla\cdot(\rho_t \nabla s_t)$, where $s_t$ is a time-dependent potential function. Introducing a damped Wasserstein Lagrangian $\mathcal{L}[\rho_t, s_t, t] = e^{\gamma t}(\frac{1}{2}\int\|\nabla s_t\|^2 \rho_t dx - \mathcal{U}[\rho_t])$, the variational principle yields particle-level equations of motion: $\frac{d}{dt}x_t = \nabla s_t(x_t)$, $\frac{d}{dt}v_t = -\nabla\frac{\delta\mathcal{U}[\rho_t]}{\delta\rho_t}(x_t) - \gamma v_t$. This is a "generalized Newton's law" for populations: particle acceleration is driven by the gradient of the population potential and velocity damping.

**Algorithmic Layer**: The neural mechanics model directly learns the potential functional $\Psi_\theta$, and via Proposition 3.1, converts functional derivatives into neural network gradients with respect to particle coordinates. Time evolution uses a Leapfrog integrator; the discrepancy between predicted and observed marginals is measured by Sinkhorn divergence, enabling end-to-end backpropagation optimization.

### Key Designs

1. **Population Potential Parameterization in the Hamiltonian Framework**:

    - **Function**: Converts the intractable functional gradient $\frac{\delta\mathcal{U}[p]}{\delta p}(x)$ into a differentiable neural network parameter gradient.
    - **Mechanism**: For the empirical distribution $\hat{p} = \frac{1}{N}\sum_i\delta_{x^{(i)}}$, the identity $\nabla_{x^{(j)}}\Psi(x^{(1)},\ldots,x^{(N)}) = \frac{1}{N}\nabla_x\frac{\delta\mathcal{U}[p]}{\delta p}(x^{(j)})\big|_{p=\hat{p}}$ holds. Thus, the potential gradient can be obtained via automatic differentiation, without explicit functional derivative computation.
    - **Design Motivation**: Circumvents the difficulty of directly solving functional derivatives in Wasserstein space by leveraging the discrete structure of empirical measures, allowing the neural network-learned scalar function $\Psi_\theta$ to naturally define the potential field.

2. **Mixed Batching and Reference-Free Learning**:

    - **Function**: Flexibly handles multi-source data while retaining second-order dynamics, without pre-specifying a reference SDE or OT plan.
    - **Mechanism**: Starting from arbitrary initial conditions $(p_0, v_0)$, a single Lagrangian $(\Psi_\theta, \gamma)$ covers all observation times. The loss function accumulates Sinkhorn divergence over each predicted/observed marginal pair. Unlike AM/JKOnet/Flow Matching, this method does not rely on any reference process, learning the mechanics system directly from data.
    - **Design Motivation**: Provides a general modeling framework in scenarios lacking domain priors, enabling application to unknown physical or biological systems.

3. **Learnable Damping and Unified Expression**:

    - **Function**: By learning the damping coefficient $\gamma$, the model automatically determines the regime between gradient flow and classical mechanics.
    - **Mechanism**: $\gamma=0$ corresponds to conservative systems (classical mechanics); $\gamma\to\infty$ to the overdamped limit (gradient flow); intermediate $\gamma>0$ yields viscous but still inertial second-order systems. On gradient flow data, the model automatically learns $\gamma\geq 500$ (strong damping), accurately recovering gradient flows.
    - **Design Motivation**: A single learnable parameter allows the model to adaptively switch among various mechanics paradigms, eliminating manual selection.

### Loss & Training

For each observed time segment $t_i\to t_{i+1}$, Leapfrog integration yields the predicted marginal $\hat{p}_{t_{i+1}}$, and the Sinkhorn divergence $\sum_i \mathcal{S}_\epsilon(\hat{p}_{t_{i+1}}, p_{t_{i+1}})$ is accumulated between predicted and observed marginals. Optimization variables include the potential network parameters $\theta$, initial velocity field $v_0$, and damping coefficient $\gamma$.

## Key Experimental Results

### Main Results

| Task | Method | Metric | Result | Notes |
|------|------|------|------|------|
| Gradient Flow SDE (paired) | nn-APPEX | Forecast $W_1$ | 0.131±0.006 | Traditional gradient flow method |
| Gradient Flow SDE (paired) | WLM (learnable $\gamma$) | Forecast $W_1$ | 0.137±0.012 | Comparable performance, no prior |
| Gradient Flow SDE (unpaired) | JKOnet* | Train $W_1$ | 0.236±0.040 | Performance collapses without pairing |
| Gradient Flow SDE (unpaired) | WLM (learnable $\gamma$) | Train $W_1$ | 0.068±0.004 | **Significantly outperforms first-order methods** |
| Gulf Vortex Interpolation (small vortex) | WLM | Multi-time $W_1$ | 0.060–0.068 | Outperforms AM/UAM/sAM without prior |
| Gulf Vortex Forecast (large vortex) | WLM | Forecast $W_1$ | 0.567±0.014 | Only reference-free method capable of forecasting |
| Embryonic Development scRNA | WLM | Interpolation $W_1$ | Outperforms gradient flow+OT | Effective on high-dimensional real data |
| Boids Flocking Behavior | WLM | Forecast | Comprehensive win | Captures collective oscillations |

### Ablation Study

| Configuration | Unpaired SDE Forecast $W_1$ | Vortex Forecast $W_1$ | Notes |
|------|------|------|------|
| Full WLM ($\Psi_\theta$+learnable $\gamma$) | 0.246±0.026 | 0.567±0.014 | Complete model |
| WLM ($\gamma=0$ conservative) | 0.346±0.045 | 0.689±0.120 | Lacks damping, especially for interpolation |
| WLM (fixed large $\gamma$) | 0.298±0.031 | 0.612±0.025 | Overdamped, deviates from gradient flow |

### Key Findings
- **Second-Order vs First-Order**: On unpaired data, WLM reduces $W_1$ from 1.618 to 0.246; the second-order framework is more robust to trajectory confusion.
- **Adaptivity of Learnable Damping**: The model automatically converges to the overdamped limit on gradient flow data, and maintains moderate damping on vortex/Boids data to capture rotation.
- **Forecasting vs Interpolation**: WLM is the only method capable of forecasting vortex future states without using a reference process.
- **High-Dimensional Scalability**: Remains effective on scRNA data with thousands of dimensions, confirming the generality of neural potential parameterization.

## Highlights & Insights
- **Theoretical Depth**: Fully derives the Hamiltonian framework in Wasserstein space, unifying gradient flows, classical mechanics, and quantum mechanics.
- **Methodological Ingenuity**: Proposition 3.1 transforms intractable functional gradients into neural network parameter gradients, making the entire learning process feasible within standard autograd frameworks.
- **Reference-Free Learning**: Does not require pre-specified reference SDE or OT plans, enabling true generality on unknown systems.
- **True Forecasting, Not Just Interpolation**: Learns dynamical equations that extrapolate beyond the training time window, representing a qualitative advance over interpolation-only methods.

## Limitations & Future Work
- Still assumes observations are snapshots of the same population at multiple time points, and cannot handle matching snapshots from entirely different populations.
- The expressive limits of neural potential networks lack theoretical characterization.
- Experiments are limited to small particle counts (about 1000); scalability to large-scale applications (millions of particles) remains untested.
- Sample complexity for potential learning in high-dimensional settings may be high, requiring more efficient approximations (inducing points, sparse representations).

## Related Work & Insights
- **vs Gradient Flow Methods (JKOnet*, nn-APPEX)**: Gradient flow is a special case with $\gamma\to\infty$; WLM uses a more general second-order framework to capture rotation/oscillation.
- **vs Flow Matching / Diffusion Interpolation**: Flow matching enables interpolation but relies on a reference process; WLM requires no reference and enables forecasting, at the cost of needing more data to learn the potential.
- **vs Action Matching (AM/UAM/sAM)**: Similar in spirit, but AM is limited to deterministic dynamics; WLM unifies stochastic and deterministic cases.
- **Insights**: The potential learning paradigm is transferable to physical simulation, molecular dynamics, social systems, etc., and is more interpretable than directly learning SDEs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rewrites population dynamics from the principle of least action; second-order + reference-free is a true paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers synthetic SDEs, physical vortices, biological scRNA, and collective behavior; clear ablations, but small sample sizes.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations, complete algorithm pseudocode, and expressive figures/tables.
- Value: ⭐⭐⭐⭐⭐ Opens a new framework for scientific modeling and dynamics learning, with strong interdisciplinary impact potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](../../ECCV2024/image_generation/idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)
- [\[ICML 2026\] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)
- [\[CVPR 2026\] Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition](../../CVPR2026/image_generation/learning_by_neighbor-aware_semantics_deciding_by_open-form_flows_towards_robust_.md)
- [\[CVPR 2026\] Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods](../../CVPR2026/image_generation/training-free_detection_of_generated_videos_via_spatial-temporal_likelihoods.md)
- [\[ICLR 2026\] Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions](../../ICLR2026/image_generation/temporal_concept_dynamics_in_diffusion_models_via_prompt-conditioned_interventio.md)

</div>

<!-- RELATED:END -->
