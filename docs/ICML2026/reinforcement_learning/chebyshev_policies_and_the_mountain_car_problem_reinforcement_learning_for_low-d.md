---
title: >-
  [Paper Note] Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks
description: >-
  [ICML 2026][Reinforcement Learning][Mountain Car] This paper provides the first analytical solution to the classic Mountain Car optimal control problem (unsolved for 36 years), revealing that the optimal policy has a minimalist form ($\alpha = C \cdot \dot{x}$). It demonstrates that existing RL agents exhibit surprisingly high regret and proposes a policy parameteriza
tags:
  - ICML 2026
  - Reinforcement Learning
  - Mountain Car
date: 2026-05-08
content_hash: d1500debc65906be
---
# Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks

**Conference**: ICML2026 Oral  
**arXiv**: [2605.22305](https://arxiv.org/abs/2605.22305)  
**Code**: [GitHub](https://github.com/2oo1/chebyshev-policies) (Yes)  
**Area**: Reinforcement Learning  
**Keywords**: Chebyshev polynomials, Mountain Car, low-dimensional control, optimal control, policy approximation  

## TL;DR

This paper provides the first analytical solution to the classic Mountain Car optimal control problem (unsolved for 36 years), revealing that the optimal policy has a minimalist form ($\alpha = C \cdot \dot{x}$). It demonstrates that existing RL agents exhibit surprisingly high regret and proposes a policy parameterization method based on multivariate Chebyshev polynomials, which reduces the number of parameters by 277x while decreasing regret by 4.18x.

## Background & Motivation

**Background**: Reinforcement learning has made significant progress in control and decision-making tasks, but faces core challenges in deployment such as low sample efficiency, poor interpretability, insufficient real-time performance, and training instability. Current RL agents commonly use MLP neural networks as function approximators for policies.

**Limitations of Prior Work**: As a classic RL benchmark, Mountain Car has existed for 36 years, yet its optimal solution remained unknown. Consequently, the true gap (regret) between existing algorithms and the optimum could not be evaluated. The best agent in RL Baselines3 Zoo (ARS) achieves an average return of only 96.77, clearly below the upper bound of 100, but it was unknown whether this gap could be closed.

**Key Challenge**: MLP policies are parameter-redundant for low-dimensional control tasks and lack theoretical guarantees—they are neither dense subsets of continuous policy spaces nor do they possess favorable mathematical properties like orthogonality. Using a black-box network with thousands of parameters to fit an essentially minimalist optimal control function is "using a sledgehammer to crack a nut."

**Goal**: (1) Analytically solve the optimal control of Mountain Car to quantify the regret of existing methods; (2) Design a new policy parameterization scheme that is parameter-efficient, interpretable, and possesses universal approximation capabilities from first principles.

**Key Insight**: By transforming the discrete dynamics of Mountain Car into continuous ODEs, the authors derive the analytical form of the optimal policy using energy conservation and the Cauchy-Schwarz inequality. The discovery that optimal control is linearly related only to velocity inspired the idea of replacing neural networks with low-order polynomials.

**Core Idea**: Use multivariate Chebyshev polynomials instead of MLPs as the parameterization model for RL policies. They form a dense subset of the continuous policy space (universal approximation) while possessing excellent mathematical properties such as orthogonality and bounded extrema, making them naturally suited for low-dimensional control tasks.

## Method

### Overall Architecture

The work consists of two parts: (1) Analytical solution of Mountain Car to obtain the optimal policy $\pi_{\text{ana}}$ and quantify the regret of existing methods; (2) Introduction of Chebyshev policies as plug-and-play replacements for MLPs based on the simplicity of the optimal solution. The input is the state vector $s \in \mathbb{R}^n$ from the RL environment, and the output is the action distribution $\pi_\theta(s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$, where both $\mu$ and $\sigma$ are parameterized by multivariate Chebyshev polynomials. This can be directly combined with standard algorithms such as PPO, ARS, and REINFORCE.

### Key Designs

**1. Mountain Car Analytical Solution: Calculating the 36-year unsolved optimal control to quantify regret**

Only by knowing the true optimal solution can one quantify how far existing RL methods are from the optimum and set targets for improvement. The authors solve this in three steps: first, transforming time-domain dynamics $\ddot x = a_{\max}\cdot\alpha - g\cos(3x)$ into spatial-domain form $\ddot x = -U'(x)$, and introducing an "unrolled variable" $\xi$ to flatten oscillations into a monotonically increasing form, making the problem integrable; second, using the Cauchy-Schwarz inequality (Lemma 2.3) under no constraints to minimize loss $\ell=\int\alpha^2\,dt$, proving that the optimal action is $\alpha(t)=C\cdot\dot x(t)$ (Theorem 2.4)—action is proportional to velocity, a minimalist form; third, adding constraints back and enumerating the number of strokes $k$ and wall collisions (single/dual-phase trajectories) to find the global optimal constant $C$. This result is impactful: optimal policies for low-dimensional control tasks are often much simpler than expected, while the regret of current MLP agents is strikingly high.

**2. Multivariate Chebyshev Polynomial Policy Parameterization: Replacing MLPs with orthogonal bases, saving parameters by 277x**

The minimalist form of the optimal solution inspired a counter-intuitive idea: using a black-box network with thousands of parameters to fit an essentially simple control function is wasteful. The authors generalize univariate Chebyshev polynomials $T_k(x)=\cos(k\cdot\arccos(x))$ to multivariate forms $T_{d_1,\dots,d_n}(x_1,\dots,x_n)=\prod_i T_{d_i}(x_i)$ via tensor products, using them as orthogonal bases to expand the policy $\mu(s)=\sum\theta_{i_1,\dots,i_n}T_{i_1,\dots,i_n}(s)$. With a max-degree of $d$, the number of parameters is only $(d+1)^n$—just 16 parameters when $n=2, d=3$, compared to 4355 in an MLP. Chebyshev polynomials were chosen because of three properties: orthogonality allows each basis function to contribute independently, avoiding the redundant coupling of MLP parameters; bounded extrema ($|T_k|\le 1$) ensure numerical stability; and density provides universal approximation for continuous policies, guaranteeing the completeness of the policy class from first principles.

**3. Stochastic Policy Integration and Plug-and-Play Design: Seamless integration into PPO/ARS/REINFORCE**

To make this parameterization practical, it must directly replace MLPs in existing algorithms. The authors use two independent Chebyshev polynomials to parameterize $\mu_\theta(s)$ and $\sigma_\theta(s)$, forming a Gaussian stochastic policy $\pi_\theta(s)=\mathcal N(\mu_\theta(s),\sigma_\theta(s))$. PPO additionally uses a third polynomial to parameterize the critic $v_\pi(s)$, while ARS only needs to train $\mu$. Engineering defaults: $\sigma$ uses a lower degree ($d\le 3$) and is initialized to 1; $\mu$ and $v$ are initialized to small random values ($\pm 10^{-3}$). The entire module is implemented as a differentiable layer in PyTorch, supporting standard gradient optimization. Replacing the policy network requires no changes to the algorithm itself, ensuring low barrier to entry.

## Key Experimental Results

### Main Results (Mountain Car)

| Policy | Mean Return $\overline{R}$ | Regret $r$ | Return Range | Parameters | Mean Arrival Time $t_*$ |
|------|----------------------|-----------|---------|--------|------------------|
| $\pi_{\text{ana}}$ (Optimal) | 99.39 | — | 99.15 – 99.52 | — | 769 |
| CH-3-ARS | 98.74 | 0.65 | 98.95 – 99.11 | ~16 | 471 |
| CH-3-REI | 98.62 | 0.77 | 98.31 – 98.89 | ~16 | 396 |
| CH-3-PPO | 98.10 | 1.29 | 97.61 – 98.42 | ~16 | 469 |
| ARS (MLP) | 96.67 | 2.72 | 92.51 – 97.42 | 4355 | 239 |
| SAC (MLP) | 94.61 | 4.78 | 89.70 – 95.77 | 4355 | 106 |
| PPO (MLP) | 93.91 | 5.48 | 90.86 – 95.23 | 4355 | 298 |

### Cross-task Generalization Experiment

| Environment | CH-ARS | ARS (MLP) | CH-PPO | PPO (MLP) |
|------|--------|-----------|--------|-----------|
| Mountain Car | **98.74** | 96.67 | **98.10** | 93.91 |
| Pendulum | **-150.8** | -218.3 | **-162.8** | -176.2 |
| Aero 2 Sim | **-125.2** | -721.8 | **-49.2** | -84.6 |
| Aero 2 Real | **-164.2** | -718.4 | **-55.8** | -182.0 |

### Key Findings

- **Significant Regret Reduction**: The regret of CH-3-ARS is only 0.65, which is 4.18x lower than the best MLP policy (ARS, 2.72). Even using the simplest REINFORCE to train Chebyshev policies (regret 0.77) far outperforms all MLP policies.
- **Key Defects of MLP Policies**: ARS (MLP) outputs positive actions for negative velocities across large regions of the state space, violating the physical dynamics of Mountain Car, which makes its return extremely sensitive to the initial position $x_0$ (dropping as low as 92.51).
- **Sim-to-real Transfer**: On Aero 2 real hardware, Chebyshev policies not only outperform MLP policies across the board but also exhibit better performance retention from simulation to reality (CH-PPO: -49.2 → -55.8 vs PPO: -84.6 → -182.0).
- **Parameter Efficiency**: Chebyshev policies require only about 16 parameters ($d=3, n=2$), which is 1/277 of an MLP (4355 parameters).

## Highlights & Insights

- **Analytical Solution to a 36-year Classic Problem**: Through spatial variable transformation and the Cauchy-Schwarz inequality, the minimalist form of the optimal policy $\alpha = C \cdot \dot{x}$ is proven. This not only solves a specific problem but also reveals that optimal policies for low-dimensional control tasks are often far simpler than anticipated.
- **Policy Class Design Derived from Optimal Solutions**: The methodology involves analyzing the problem structure first and then designing function approximators from first principles, rather than blindly applying neural networks. This "analysis-driven" approach is worth adopting in other RL tasks.
- **Dimensionality Reduction Advantage of Orthogonal Bases**: The orthogonality of Chebyshev bases means each basis function contributes independently, avoiding the redundant coupling of parameters found in MLPs. This allows a very small number of parameters to efficiently cover the policy space, a concept transferable to any low-dimensional continuous control task.

## Limitations & Future Work

- **Curse of Dimensionality**: The number of parameters grows exponentially with dimension $(d+1)^n$, making it inapplicable to high-dimensional state spaces (e.g., humanoid robots $n > 10$).
- **Limitations of Uniform Approximation**: Chebyshev polynomials provide uniform approximation across the entire domain and cannot allocate different expressive power to different regions of the state space like ReLU MLPs, which is disadvantageous for discontinuous policies like bang-bang or sliding mode control.
- **Future Work**: The authors suggest exploring hybrid MLP + Chebyshev architectures to allow complementarity—Chebyshev layers for global smooth approximation and MLP layers for local nonlinearities. Additionally, sparse Chebyshev bases (keeping only important terms) could be studied to mitigate dimensional growth.

## Related Work & Insights

- **Linear Policies**: Rajeswaran et al. (2017) demonstrated the effectiveness of linear policies on multiple continuous control tasks; the Chebyshev policies in this paper can be seen as their polynomial generalization.
- **Random Fourier Features**: Schulman et al. (2015) used random Fourier features $f(s) = \sin(\langle s, v \rangle + \varphi)$ as policy basis functions but lacked theoretical guarantees for universal approximation.
- **Insight**: This paper shows that for low-dimensional control tasks, "smaller and simpler" models can indeed be better—contrasting with the "bigger is better" inertia of deep learning and reminding us to prioritize structured, interpretable methods when the problem allows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](../../AAAI2026/reinforcement_learning/diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)
- [\[ICML 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICML 2026\] PAC-Bayesian Reinforcement Learning Trains Generalizable Policies](pac-bayesian_reinforcement_learning_trains_generalizable_policies.md)
- [\[ICLR 2026\] Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](../../ICLR2026/reinforcement_learning/helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)
- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](learning_unmasking_policies_for_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
