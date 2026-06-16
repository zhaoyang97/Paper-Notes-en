---
title: >-
  [Paper Note] A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots
description: >-
  [ICML 2026][Physics & Scientific Computing][Paper Note] Starting from the principle of least action, this paper proposes the Wasserstein Lagrangian Mechanics (WLM) framework to learn second-order population dynamics rather than traditional first-order gradient flow dynamics. This enables capturing richer collective phenomena such as periodicity and rotation, and allows for
tags:
  - ICML 2026
  - Physics & Scientific Computing
date: 2026-05-08
content_hash: ac41c8213017b92e
---
# A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.08550](https://arxiv.org/abs/2605.08550)  
**Code**: https://github.com/guanton/WLM  
**Area**: Diffusion Models / Dynamics Learning  
**Keywords**: Population Dynamics, Wasserstein Mechanics, Principle of Least Action, Second-order Dynamics, Neural Potentials

## TL;DR
Starting from the principle of least action, this paper proposes the Wasserstein Lagrangian Mechanics (WLM) framework to learn second-order population dynamics rather than traditional first-order gradient flow dynamics. This enables capturing richer collective phenomena such as periodicity and rotation, and allows for interpolation and future forecasting without requiring a reference process.

## Background & Motivation

**Background**: Traditional population dynamics modeling (ranging from molecular diffusion to cell differentiation and biological collective behavior) commonly adopts the Wasserstein gradient flow paradigm. Starting from free energy functionals, these methods can model purely dissipative evolution processes.

**Limitations of Prior Work**: Gradient flows are essentially first-order systems that eventually converge toward an equilibrium state. However, real-world population dynamics often exhibit non-equilibrium phenomena such as periodicity, rotational motion, and oscillations—e.g., vortices and Boids collective behavior—which transcend the energy minimization framework.

**Key Challenge**: While gradient flows have solid mathematical foundations and mature optimization algorithms, their expressive power is limited. When only temporal snapshots of marginal distributions are available (precluding the tracking of individual trajectories), how can broader second-order dynamics be inferred without pre-specifying a Lagrangian?

**Goal**: To re-examine population models through the principle of least action, replacing first-order gradient flows with second-order systems to form a framework that unifies classical mechanics, quantum mechanics, and gradient flows.

**Key Insight**: Populations "coordinates" and Lagrangians are defined in the Wasserstein-2 distance space $\mathcal{P}_2(\mathbb{R}^d)$. Hamiltonian equations of motion are derived via the variational principle, establishing a mechanical perspective of "populations driving their own evolution."

**Core Idea**: The second-order dynamics are parameterized using a population potential functional $\mathcal{U}[\rho_t]$ and a damping coefficient $\gamma$. This allows the framework to cover gradient flows (overdamped limit $\gamma\to\infty$), conservative classical mechanics ($\gamma=0$), and quantum mechanics.

## Method

### Overall Architecture

The method consists of a theoretical layer and an algorithmic layer.

**Theoretical Layer**: Within $\mathcal{P}_2(\mathbb{R}^d)$, population dynamics are represented by the continuity equation $\dot{\rho}_t = -\nabla\cdot(\rho_t \nabla s_t)$, where $s_t$ is a time-varying potential function. By introducing a damped Wasserstein Lagrangian $\mathcal{L}[\rho_t, s_t, t] = e^{\gamma t}(\frac{1}{2}\int\|\nabla s_t\|^2 \rho_t dx - \mathcal{U}[\rho_t])$ and applying the variational principle, particle-level equations of motion are derived: $\frac{d}{dt}x_t = \nabla s_t(x_t)$ and $\frac{d}{dt}v_t = -\nabla\frac{\delta\mathcal{U}[\rho_t]}{\delta\rho_t}(x_t) - \gamma v_t$. This constitutes a population-based version of "generalized Newton's laws," where particle acceleration is driven by the population potential gradient and velocity damping.

**Algorithmic Layer**: The neural mechanics model directly learns the potential functional $\Psi_\theta$. Through Proposition 3.1, functional derivatives are converted into neural network gradients with respect to particle coordinates. Time integration is performed using a Leapfrog integrator. The difference between predicted and observed marginals is measured using Sinkhorn divergence, optimized via end-to-end backpropagation.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Temporal snapshots {p_t} + Initial velocity v₀"] --> B["Population potential parameterization Ψθ<br/>Compute particle gradients via autograd (Prop 3.1)"]
    G["Learnable damping γ<br/>γ=0: Conservative · γ→∞: Degenerates to Gradient Flow"] --> C["Generalized Newtonian Acceleration<br/>a = −∇Ψθ − γv"]
    B --> C
    C --> D["Leapfrog Advancement<br/>Simulate K-step marginals from (p₀,v₀)"]
    D --> E["Predicted Marginals ↔ Observed Marginals<br/>Sinkhorn Divergence Loss"]
    E -->|End-to-end backprop (DTO) updates θ,v₀,γ| B
    E --> F["Learned Mechanical Equations<br/>Interpolation / Future Forecast"]
```

### Key Designs

**1. Population Potential Parameterization under the Hamiltonian Framework: Turning uncomputable functional derivatives into neural network autograd gradients**

WLM seeks to learn the population potential functional $\mathcal{U}[\rho_t]$, but it appears in the equations of motion as a functional derivative $\frac{\delta\mathcal{U}[p]}{\delta p}(x)$, which is nearly impossible to compute directly in Wasserstein space. The authors utilize the discrete structure of the empirical measure $\hat p=\frac1N\sum_i\delta_{x^{(i)}}$ to provide an identity (Prop. 3.1):

$$\nabla_{x^{(j)}}\Psi(x^{(1)},\dots,x^{(N)})=\frac{1}{N}\,\nabla_x\frac{\delta\mathcal{U}[p]}{\delta p}(x^{(j)})\Big|_{p=\hat p},$$

Thus, by feeding all particle coordinates into a scalar network $\Psi_\theta$ and taking the gradient with respect to a single particle, one obtains the gradient of the potential field. This step is critical for the algorithmic layer, as it translates "variational calculus in probability measure space" into "backpropagation on particle coordinates."

**2. Reference-free Simulation-based Learning: Rolling out from initial conditions to cover all observations without relying on reference processes**

Most gradient flow methods (Flow Matching, JKOnet, AM) require a pre-specified reference SDE or OT plan. If real data violates this prior (e.g., unpaired snapshots), performance collapses. WLM instead starts from initial conditions $(p_0, v_0)$ and uses the same set of $(\Psi_\theta, \gamma)$ to simulate marginals via Leapfrog advancement. In each round, a rollout step $K\sim\text{Unif}[1,M]$ is randomly sampled, and loss is accumulated as Sinkhorn divergences between predicted/observed marginal pairs, backpropagated directly along discrete trajectories (discretize-then-optimize). By not assuming pairings or injecting reference dynamics, it is more robust for unknown physical/biological systems lacking domain priors—demonstrated in experiments where $W_1$ on unpaired SDEs dropped from 0.236 for first-order methods to 0.068.

**3. Learnable Damping and Unified Expression: A single scalar $\gamma$ allowing adaptive switching between gradient flow and classical mechanics**

Whether second-order mechanics should retain inertia or be dissipative/conservative should not be manually selected. Thus, the authors set the damping coefficient $\gamma$ as a learnable parameter. It spans a full spectrum: $\gamma=0$ for conservative classical mechanics, $\gamma\to\infty$ as the overdamped limit (degenerating to Wasserstein gradient flow), and $\gamma>0$ for visceral systems with inertia. The model autonomously finds its position: when fed gradient flow data, it automatically learns $\gamma\ge 500$ to replicate the flow; when fed vortex/Boids data, it maintains moderate damping to capture rotational oscillations. This single scalar unifies multiple mechanical paradigms within one framework.

### Loss & Training

For each observed time interval $t_i\to t_{i+1}$, Leapfrog advancement yields a predicted marginal $\hat{p}_{t_{i+1}}$. The Sinkhorn divergence $\sum_i \mathcal{S}_\epsilon(\hat{p}_{t_{i+1}}, p_{t_{i+1}})$ is accumulated against the observed marginals. Optimization variables include the potential network parameters $\theta$, the initial velocity field $v_0$, and the damping coefficient $\gamma$.

## Key Experimental Results

### Main Results

| Task | Method | Metric | Result | Note |
|------|------|------|------|------|
| Gradient Flow SDE (paired) | nn-APPEX | Forecast $W_1$ | 0.131±0.006 | Traditional gradient flow method |
| Gradient Flow SDE (paired) | WLM (learnable $\gamma$) | Forecast $W_1$ | 0.137±0.012 | Similar performance, but no prior |
| Gradient Flow SDE (unpaired) | JKOnet* | Train $W_1$ | 0.236±0.040 | Paired hypothesis fails, performance collapses |
| Gradient Flow SDE (unpaired) | WLM (learnable $\gamma$) | Train $W_1$ | 0.068±0.004 | **Significantly outperforms first-order methods** |
| Gulf vortex interpolation (small) | WLM | Multi-epoch $W_1$ | 0.060–0.068 | Outperforms AM/UAM/sAM without priors |
| Gulf vortex forecast (large) | WLM | Forecast $W_1$ | 0.567±0.014 | Only reference-free method capable of forecasting |
| Embryonic scRNA | WLM | Interpolation $W_1$ | Outperforms GF+OT | Effective on high-dim real-world data |
| Boids collective behavior | WLM | Forecast | Wins across all metrics | Captures collective oscillations |

### Ablation Study

| Configuration | Unpaired SDE Forecast $W_1$ | Vortex Forecast $W_1$ | Note |
|------|------|------|------|
| Full WLM ($\Psi_\theta$ + learnable $\gamma$) | 0.246±0.026 | 0.567±0.014 | Full model |
| WLM ($\gamma=0$ conservative) | 0.346±0.045 | 0.689±0.120 | Poor without damping, especially interpolation |
| WLM (Fixed large $\gamma$) | 0.298±0.031 | 0.612±0.025 | Overdamping deviates from optimal flow |

### Key Findings
- **Second-order vs. First-order**: WLM reduces $W_1$ from 1.618 to 0.246 on unpaired data; the second-order framework is more robust to ambiguous trajectories.
- **Adaptability of Learnable Damping**: The model automatically converges to the overdamped limit on gradient flow data and maintains moderate damping for rotation in vortex/Boids data.
- **Forecasting vs. Interpolation**: WLM is the only method that can forecast future states of vortices without using a reference process.
- **High-dimensional Scalability**: Effective on scRNA data with thousands of dimensions, confirming the universality of neural potential parameterization.

## Highlights & Insights
- **Theoretical Depth**: Derived the Hamiltonian framework in Wasserstein space, unifying gradient flows, classical mechanics, and quantum mechanics.
- **Methodological Ingenuity**: Proposition 3.1 converts uncomputable functional gradients into neural network parameter gradients, making the learning process feasible within standard autograd frameworks.
- **Reference-free Learning**: Does not require pre-specified reference SDEs or OT plans, ensuring true universality for unknown systems.
- **Forecasting Power**: Learns dynamic equations that extrapolate beyond the training window, representing a qualitative advancement over interpolation-only methods.

## Limitations & Future Work
- Assumes observations are snapshots of the same population; cannot handle matching snapshots from completely different populations.
- Lacks theoretical characterization of the expressive boundaries of neural potential networks.
- Experimental particle counts are relatively small (~1000); scalability for large-scale applications (millions of particles) is unverified.
- Sample complexity for potential learning in high-dimensional scenarios may be high, requiring more efficient approximations (e.g., inducing points, sparse representations).

## Related Work & Insights
- **vs. Gradient Flow Methods (JKOnet*, nn-APPEX)**: Gradient flows are a special case where $\gamma\to\infty$; WLM uses a more general second-order framework to capture rotation/oscillation.
- **vs. Flow Matching / Diffusion Interpolation**: Flow Matching interpolates but relies on reference processes; WLM requires no reference and can forecast, at the cost of needing more data to learn the potential.
- **vs. Action Matching (AM/UAM/sAM)**: Similar intuition, but AM-based methods are limited to deterministic dynamics; WLM unifies stochastic and deterministic cases.
- **Insight**: The potential learning paradigm is valuable for physical simulation, molecular dynamics, and social systems, offering better interpretability than direct SDE learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rewriting population dynamics from the principle of least action; second-order + reference-free is a true paradigm extension.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers synthetic SDEs, physical vortices, biological scRNA, and collective behavior; clear ablations, though sample scales are small.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations, complete algorithm pseudocode, and expressive visualizations.
- Value: ⭐⭐⭐⭐⭐ Opens a new framework for scientific modeling and dynamics learning with high interdisciplinary potential.

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
