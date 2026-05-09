---
title: >-
  [Paper Note] Deep Learning for Continuous-Time Stochastic Control with Jumps
description: >-
  [NeurIPS 2025][Stochastic Control] Two model-based deep learning algorithms (GPI-PINN and GPI-CBU) are proposed to solve finite-horizon continuous-time stochastic control problems with jumps. By iteratively training a policy network and a value network, the approach avoids discretization and simulation of state dynamics, and demonstrates strong performance in high-dimensional settings.
tags:
  - NeurIPS 2025
  - Stochastic Control
  - Jump Diffusion
  - HJB Equation
  - Deep Learning
  - Actor-Critic
date: 2026-05-08
content_hash: fc2621c9c19ce62c
---

# Deep Learning for Continuous-Time Stochastic Control with Jumps

**Conference**: NeurIPS 2025
**arXiv**: [2505.15602](https://arxiv.org/abs/2505.15602)
**Code**: [GitHub](https://github.com/jdupret97/Deep-Learning-for-CT-Stochastic-Control-with-Jumps)
**Area**: Other
**Keywords**: Stochastic Control, Jump Diffusion, HJB Equation, Deep Learning, Actor-Critic

## TL;DR

Two model-based deep learning algorithms (GPI-PINN and GPI-CBU) are proposed to solve finite-horizon continuous-time stochastic control problems with jumps. By iteratively training a policy network and a value network, the approach avoids discretization and simulation of state dynamics, and demonstrates strong performance in high-dimensional settings.

## Background & Motivation

Continuous-time stochastic control problems arise widely in dynamic decision-making scenarios, with the Hamilton-Jacobi-Bellman (HJB) equation at their core. Classical methods face three major challenges:

**Curse of Dimensionality**: Finite difference, finite element, and other classical methods are intractable in high dimensions.

**Difficulty Handling Jumps**: When the system dynamics include stochastic jumps, the HJB equation degenerates into a PIDE (partial integro-differential equation), requiring computation of the jump expectation $\mathbb{E}[V(t, x+\gamma(t,x,Z_1,a))]$ at every spatio-temporal sample point, which is computationally prohibitive.

**Implicit Optimal Control**: When the optimal control cannot be obtained in closed form, it cannot be directly substituted into the HJB equation for simplification, necessitating simultaneous approximation of both the value function and the optimal control.

Existing deep learning methods are either model-free RL (which does not exploit known dynamics and thus yields lower accuracy) or based on temporal discretization (introducing discretization error and poor generalization to unseen regions).

## Method

### Overall Architecture

The control problem under consideration is:

$$\sup_\alpha \mathbb{E}\left[\int_0^T f(t, X_t^\alpha, \alpha_t) dt + F(X_T^\alpha)\right]$$

The controlled process follows jump-diffusion dynamics:

$$dX_t^\alpha = \beta(t,X_t^\alpha,\alpha_t)dt + \sigma(t,X_t^\alpha,\alpha_t)dW_t + \int_E \gamma(t,X_{t-}^\alpha,z,\alpha_t)N^\alpha(dz,dt)$$

where $N^\alpha$ is a Cox process with controllable intensity $\lambda(t,X_{t-}^\alpha,\alpha_t)$.

The core of the approach is to approximate the value function and the optimal control with two neural networks $V_\theta$ and $\alpha_\phi$, respectively, trained iteratively in an Actor-Critic fashion, with correctness guaranteed by the Feynman-Kac formula and the verification theorem.

### Key Designs

#### Algorithm 1: GPI-PINN

The PINN methodology is used to minimize the residual of the controlled HJB equation. A key technique, via **Proposition 3.1**, avoids explicit computation of gradients and Hessians:

An auxiliary function $\psi(h)$ is defined such that:

$$\psi''(0) = \partial_t v(t,x) + \beta^\top(t,x,a)\nabla_x v(t,x) + \frac{1}{2}\text{Tr}[\sigma\sigma^\top(t,x,a)\nabla_x^2 v(t,x)]$$

This replaces gradient and Hessian computations with the evaluation of $\psi''(0)$, a univariate second derivative, at a cost of only a small multiple of $n \cdot \text{cost}(v)$.

**Value network update** (minimizing the PIDE residual):

$$\mathscr{L}_1(\theta, \phi) = \xi_1 \mathbb{E}_{(t,x)\sim\mu} \mathcal{H}^2(t,x,\theta,\phi) + \xi_2 \mathbb{E}_{x\sim\nu}(V_\theta(T,x) - F(x))^2$$

**Control network update** (maximizing the Hamiltonian):

$$\mathscr{L}_2(\theta, \phi) = -\mathbb{E}_{(t,x)\sim\mu} \mathcal{H}(t,x,\theta,\phi)$$

A Residual-based Adaptive Distribution (RAD) strategy is employed to adaptively update the sampling distribution.

**Limitation of GPI-PINN**: Jump expectations must still be computed at each sample point, and the gradient step involves third-order derivatives.

#### Algorithm 2: GPI-CBU

The continuous-time Bellman update rule is employed, introducing an **expectation-free operator** $G_\zeta$:

$$G_\zeta(t,x,z,v,a) = v(t,x) + \zeta[\partial_t v + f + \beta^\top \nabla_x v + \frac{1}{2}\text{Tr}[\sigma\sigma^\top \nabla_x^2 v] + \lambda(v(t,x+\gamma(t,x,z,a)) - v(t,x))]$$

The key point is that $G_\zeta$ requires neither computation of the jump expectation (only a single jump evaluation) nor third-order derivatives.

**Proposition 4.1** guarantees that minimizing $\mathbb{E}[(g(Y_t) - G_\zeta(t,Y_t,Z_1,V^\alpha,\alpha(t,Y_t)))^2]$ recovers the correct value function.

Value network update loss:

$$\mathscr{L}_1^{(k)}(\theta) = \xi_1 \mathbb{E}_{(t,x,z)\sim\mu\otimes\mathcal{Z}}(V_\theta(t,x) - G_\zeta(t,x,z,\theta^{(k)},\phi^{(k)}))^2 + \xi_2 \mathbb{E}_{x\sim\nu}(V_\theta(T,x) - F(x))^2$$

### Loss & Training

Both algorithms adopt iterative Actor-Critic training:
1. **Step 1 (Critic)**: Fix the control network; update the value network to satisfy the HJB equation.
2. **Step 2 (Actor)**: Fix the value network; update the control network to maximize the Hamiltonian.

The DGM (Deep Galerkin Method) architecture is used, requiring $C^2$ activation functions. The hyperparameter $\zeta=1$ provides a good trade-off between speed and accuracy; negative scaling factors cause loss divergence.

## Key Experimental Results

### Main Results

**Linear Quadratic Regulator (LQR) with Jumps** ($d=10$):

Comparison with RL and discrete-time methods:

| Method | Type | Accuracy (log MAE_V) |
|--------|------|----------------------|
| PPO | Model-free RL | Worst |
| SAC | Model-free RL | Second worst |
| Han & E (2016) | Model-based discrete-time | Moderate (discretization error) |
| **GPI-CBU** | **Model-based continuous-time** | **Best** |

GPI-PINN vs. GPI-CBU:
- Without jumps: comparable accuracy; GPI-CBU is slightly faster due to the absence of third-order derivatives.
- **With jumps**: GPI-CBU is significantly faster than GPI-PINN (no jump expectation computation required).
- GPI-PINN converges more stably; GPI-CBU has lower computational cost.

**High-Dimensional LQR with Jumps** ($d=50$):
- GPI-PINN is infeasible (prohibitive computational cost).
- GPI-CBU still achieves high-accuracy approximation of both the value function and the optimal control.
- Results up to $d=150$ are reported in the appendix.

### Ablation Study

**Optimal Consumption-Investment Problem with Jumps** ($n=25$ assets, $d=52$ state variables):
- Incorporates stochastic volatility, stochastic jump intensity, and stochastic interest rates.
- GPI-CBU training loss converges, providing a tractable solution for a realistic economic decision problem.
- In a simplified version (constant coefficients), GPI-CBU results are nearly indistinguishable from Runge-Kutta reference solutions.

### Key Findings

1. **Model information is crucial**: Model-based methods (GPI-PINN/CBU) that exploit known dynamics substantially outperform model-free RL methods (PPO/SAC).
2. **GPI-CBU resolves the core computational bottleneck of jump problems**: The expectation-free operator eliminates the need to integrate over the jump distribution at every sample point.
3. **Advantage of global approximation**: Local methods (Han & E 2016) only learn accurately near optimal trajectories and generalize poorly to unexplored regions.
4. **Trade-off between convergence stability and efficiency**: GPI-PINN achieves greater stability by averaging over multiple jumps, while GPI-CBU is more efficient through single-jump evaluation.

## Highlights & Insights

1. **Elegant mathematical derivations**: Proposition 3.1 converts gradient and Hessian computations into a univariate second derivative, which is both concise and practical; Proposition 4.1 provides the theoretical foundation for the CBU approach.
2. **Advantages of continuous-time formulation**: Direct solution in continuous time avoids discretization error and yields a global solution (queryable at arbitrary spatio-temporal points).
3. **Core innovation of GPI-CBU**: The recursive update rule completely eliminates jump expectations from the loss function, making high-dimensional jump control problems tractable for the first time.
4. **Integration of Actor-Critic with PIDE solving**: The GPI paradigm from RL is elegantly combined with numerical methods for partial differential equations.

## Limitations & Future Work

1. **Requires known dynamics model**: In fields such as economics and finance, the dynamics model typically must be inferred from data; the authors suggest pre-learning via model learning algorithms.
2. **Convergence stability of GPI-CBU**: Single-jump estimation introduces high variance; negative scaling factors $\zeta$ cause loss divergence.
3. **Hyperparameter sensitivity**: The choice of $\xi_1$, $\xi_2$, and $\zeta$ has a significant impact on performance.
4. **Constrained and path-dependent problems not addressed**: The current framework assumes Markovian feedback control and an unconstrained action space.
5. **Network architecture choices**: The suitability of the DGM architecture relative to more modern architectures (e.g., Transformers) has not been examined.

## Related Work & Insights

- **Han & E (2016)**: A deep learning control method based on temporal discretization; the present continuous-time approach avoids the associated discretization error.
- **PINN (Raissi et al.)**: Physics-informed neural networks; GPI-PINN extends this framework to control problems.
- **DGM (Sirignano & Spiliopoulos)**: Provides the network architecture design; empirical evidence suggests improvements over standard PINN performance.
- **Duarte et al. (2024)**: Introduced techniques for avoiding gradient and Hessian computation; the present work adapts these to the finite-horizon setting with terminal conditions.
- The work has significant practical value for scientific computing and quantitative finance: tractability of the 50-dimensional consumption-investment problem and the 150-dimensional LQR problem opens the door to real-world applications.

## Rating

- Novelty: ⭐⭐⭐⭐ (The expectation-free update rule in GPI-CBU is the core innovation, resolving the key computational bottleneck in jump control)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Rigorous theoretical validation with analytical solution comparisons and multi-dimensional scaling, though real-world application scenarios are limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (Mathematically rigorous, algorithmically clear, with tight integration of theory and experiments)
- Value: ⭐⭐⭐⭐ (Provides the first practical deep learning solution for high-dimensional stochastic control problems with jumps)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[NeurIPS 2025\] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [\[NeurIPS 2025\] Continuous Thought Machines](continuous_thought_machines.md)
- [\[NeurIPS 2025\] Uncertainty Estimation by Flexible Evidential Deep Learning](uncertainty_estimation_by_flexible_evidential_deep_learning.md)
- [\[NeurIPS 2025\] Deep Legendre Transform](deep_legendre_transform.md)

</div>

<!-- RELATED:END -->
