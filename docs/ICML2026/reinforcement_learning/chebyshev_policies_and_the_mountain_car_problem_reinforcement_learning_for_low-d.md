---
title: >-
  [Paper Note] Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks
description: >-
  [ICML2026][Reinforcement Learning][Chebyshev polynomials] This paper provides the first analytical solution to the classic Mountain Car optimal control problem (unsolved for 36 years)…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Chebyshev polynomials"
  - "Mountain Car"
  - "low-dimensional control"
  - "optimal control"
  - "policy approximation"
date: 2026-05-08
content_hash: d0490183bdad472d
---

# Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks

**Conference**: ICML2026  
**arXiv**: [2605.22305](https://arxiv.org/abs/2605.22305)  
**Code**: [GitHub](https://github.com/2oo1/chebyshev-policies) (Available)  
**Area**: Reinforcement Learning  
**Keywords**: Chebyshev polynomials, Mountain Car, low-dimensional control, optimal control, policy approximation  

## TL;DR

This paper provides the first analytical solution to the classic Mountain Car optimal control problem (unsolved for 36 years), revealing a minimalist optimal policy form ($\alpha = C \cdot \dot{x}$). It demonstrates that existing RL agents exhibit surprisingly high regret and proposes a policy parameterization method based on multivariate Chebyshev polynomials, which reduces parameters by 277x while decreasing regret by 4.18x.

## Background & Motivation

**Background**: Reinforcement Learning (RL) has made significant progress in control and decision tasks but faces core challenges in practical deployment, such as low sample efficiency, poor interpretability, insufficient real-time performance, and training instability. Current RL agents commonly use Multi-Layer Perceptron (MLP) neural networks as function approximators for policies.

**Limitations of Prior Work**: As a classic benchmark in RL for 36 years, the optimal solution for Mountain Car has remained unknown, making it impossible to evaluate the true regret of existing algorithms. The best agent in the RL Baselines3 Zoo (ARS) achieves an average reward of only 96.77, noticeably below the upper bound of 100, yet it was unknown whether this gap could be closed.

**Key Challenge**: MLP policies are parameter-redundant for low-dimensional control tasks and lack theoretical guarantees—they are neither a dense subset of the continuous policy space nor do they possess favorable mathematical properties like orthogonality. Using a black-box network with thousands of parameters to fit an essentially minimalist optimal control function is "using a cannon to kill a mosquito."

**Goal**: (1) Analytically solve for the optimal control of Mountain Car and quantify the regret of existing methods; (2) Design a new policy parameterization scheme from first principles that is parameter-efficient, interpretable, and possesses universal approximation capabilities.

**Key Insight**: By transforming the discrete dynamics of Mountain Car into a continuous ODE and applying energy conservation and the Cauchy-Schwarz inequality, the authors derive the analytical form of the optimal policy. The discovery that optimal control is linearly related to velocity inspired the idea of replacing neural networks with low-order polynomials.

**Core Idea**: Replace MLPs with multivariate Chebyshev polynomials as the parameterization model for RL policies. These constitute a dense subset of the continuous policy space (universal approximation) while offering excellent mathematical properties such as orthogonality and bounded extrema, making them naturally suited for low-dimensional control.

## Method

### Overall Architecture

The work is divided into two parts: (1) Analytical solution of Mountain Car to obtain the optimal policy $\pi_{\text{ana}}$ and quantify the regret of existing methods; (2) Proposing Chebyshev policies as a plug-and-play replacement for MLPs, inspired by the simplicity of the optimal solution. The input is the environment state vector $s \in \mathbb{R}^n$, and the output is the action distribution $\pi_\theta(s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$, where both $\mu$ and $\sigma$ are parameterized by multivariate Chebyshev polynomials. This can be directly combined with standard algorithms like PPO, ARS, and REINFORCE.

### Key Designs

1.  **Mountain Car Analytical Solution (Three-step method)**:
    *   Function: Provides the first analytical optimal control solution for the Mountain Car problem.
    *   Mechanism: First, transform time-domain dynamics $\ddot{x} = a_{\max} \cdot \alpha - g \cos(3x)$ into space-domain form $\ddot{x} = -U'(x)$, introducing an "unfolding variable" $\xi$ to flatten oscillations into monotonic growth. Second, use the Cauchy-Schwarz inequality (Lemma 2.3) to minimize the loss $\ell = \int \alpha^2 \, dt$ without constraints, proving the optimal action $\alpha(t) = C \cdot \dot{x}(t)$ (Theorem 2.4), i.e., action is proportional to velocity. Third, recover constraints to find the global optimal constant $C$ by enumerating stroke count $k$ and wall-collision conditions.
    *   Design Motivation: Only by knowing the true optimal solution can the real regret of existing RL methods be quantified, providing clear guidance for improvement.

2.  **Multivariate Chebyshev Polynomial Policy Parameterization**:
    *   Function: Provides a parameter-efficient policy function class with universal approximation capabilities.
    *   Mechanism: Generalize univariate Chebyshev polynomials $T_k(x) = \cos(k \cdot \arccos(x))$ to multivariate cases via tensor products $T_{d_1,\ldots,d_n}(x_1,\ldots,x_n) = \prod_i T_{d_i}(x_i)$, using them as an orthogonal basis to expand the policy function $\mu(s) = \sum \theta_{i_1,\ldots,i_n} T_{i_1,\ldots,i_n}(s)$. For max-degree $d$, the parameter count is only $(d+1)^n$; e.g., $n=2, d=3$ requires 16 parameters (vs. 4355 for MLP). Implemented as differentiable modules in PyTorch for standard gradient optimization.
    *   Design Motivation: Chebyshev polynomials offer orthogonality (efficient sampling of policy space), bounded extrema ($|T_k| \leq 1$, numerical stability), and denseness (universal approximation of continuous policies), ensuring the completeness of the policy class from first principles.

3.  **Stochastic Policy Integration and Plug-and-Play Design**:
    *   Function: Seamlessly embeds Chebyshev polynomials into standard RL algorithm frameworks.
    *   Mechanism: Use two independent Chebyshev polynomials to parameterize $\mu_\theta(s)$ and $\sigma_\theta(s)$, forming a Gaussian stochastic policy $\pi_\theta(s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$. For PPO, a third polynomial parameterizes the critic $v_\pi(s)$. In practice, $\sigma$ uses a lower degree ($d \leq 3$) initialized to 1; $\mu$ and $v$ are initialized with small random values ($\pm 10^{-3}$). For ARS, only $\mu$ needs training.
    *   Design Motivation: Maintain full compatibility with existing RL algorithms (PPO/ARS/REINFORCE) without modifying the algorithms themselves, lowering the barrier to adoption.

## Key Experimental Results

### Main Results (Mountain Car)

| Policy | Average Reward $\overline{R}$ | Regret $r$ | Reward Range | Parameters | Avg. Time to Reach $t_*$ |
|--------|----------------------|-----------|---------|--------|------------------|
| $\pi_{\text{ana}}$ (Optimal) | 99.39 | — | 99.15 – 99.52 | — | 769 |
| CH-3-ARS | 98.74 | 0.65 | 98.95 – 99.11 | ~16 | 471 |
| CH-3-REI | 98.62 | 0.77 | 98.31 – 98.89 | ~16 | 396 |
| CH-3-PPO | 98.10 | 1.29 | 97.61 – 98.42 | ~16 | 469 |
| ARS (MLP) | 96.67 | 2.72 | 92.51 – 97.42 | 4355 | 239 |
| SAC (MLP) | 94.61 | 4.78 | 89.70 – 95.77 | 4355 | 106 |
| PPO (MLP) | 93.91 | 5.48 | 90.86 – 95.23 | 4355 | 298 |

### Cross-task Generalization Results

| Environment | CH-ARS | ARS (MLP) | CH-PPO | PPO (MLP) |
|-------------|--------|-----------|--------|-----------|
| Mountain Car| **98.74** | 96.67 | **98.10** | 93.91 |
| Pendulum | **-150.8** | -218.3 | **-162.8** | -176.2 |
| Aero 2 Sim | **-125.2** | -721.8 | **-49.2** | -84.6 |
| Aero 2 Real | **-164.2** | -718.4 | **-55.8** | -182.0 |

### Key Findings

- **Significant Regret Reduction**: The regret of CH-3-ARS is only 0.65, which is 4.18x lower than the best MLP policy (ARS, 2.72). Even using the simplest REINFORCE to train Chebyshev policies (regret 0.77) significantly outperforms all MLP policies.
- **Critical Flaws in MLP Policies**: ARS (MLP) outputs positive actions for negative velocities across large regions of the state space, violating the physical dynamics of Mountain Car and causing its reward to be extremely sensitive to initial position $x_0$ (dropping as low as 92.51).
- **Sim-to-real Transfer**: On Aero 2 real hardware, Chebyshev policies not only outperform MLP policies overall but also show better performance retention from simulation to reality (CH-PPO: -49.2 → -55.8 vs. PPO: -84.6 → -182.0).
- **Parameter Efficiency**: Chebyshev policies require only ~16 parameters ($d=3, n=2$), which is 1/277 of the MLP (4355 parameters).

## Highlights & Insights

- **Analytical Solution to a 36-Year-Old Problem**: Through spatial variable transformation and the Cauchy-Schwarz inequality, the paper proves the minimalist form of the optimal policy $\alpha = C \cdot \dot{x}$. This not only solves a specific problem but also reveals that optimal policies for low-dimensional control tasks are often much simpler than expected.
- **Designing Policy Classes from Optimal Solutions**: The methodology of analyzing problem structure first and then designing function approximators from first principles—rather than blindly applying neural networks—is valuable for other RL tasks.
- **Dimensionality Reduction via Orthogonal Bases**: The orthogonality of the Chebyshev basis means each basis function contributes independently, avoiding the redundant coupling of parameters found in MLPs. This allows a very small number of parameters to efficiently cover the policy space, a concept transferable to any low-dimensional continuous control task.

## Limitations & Future Work

- **Curse of Dimensionality**: Parameter count grows exponentially with dimension $(d+1)^n$, making it inapplicable to high-dimensional state spaces (e.g., humanoid robots where $n > 10$).
- **Uniform Approximation Limits**: Chebyshev polynomials approximate uniformly across the domain and cannot allocate representational capacity differently across state space regions like ReLU MLPs, which is disadvantageous for discontinuous policies like bang-bang or sliding mode control.
- **Future Directions**: The authors suggest exploring hybrid MLP + Chebyshev architectures to enable complementarity—Chebyshev layers for global smooth approximation and MLP layers for local nonlinearities. Additionally, sparse Chebyshev bases (keeping only significant terms) could mitigate dimensional growth.

## Related Work & Insights

- **Linear Policies**: Rajeswaran et al. (2017) demonstrated the effectiveness of linear policies on multiple continuous control tasks; the Chebyshev policy in this paper can be viewed as their polynomial generalization.
- **Random Fourier Features**: Schulman et al. (2015) used random Fourier features $f(s) = \sin(\langle s, v \rangle + \varphi)$ as policy basis functions but lacked theoretical guarantees for universal approximation.
- **Insight**: This paper shows that for low-dimensional control, "smaller and simpler" models can be superior. This contrasts interestingly with the "bigger is better" mindset in deep learning, reminding us to prioritize structured, interpretable methods when the problem allows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](../../AAAI2026/reinforcement_learning/diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)
- [\[ICML 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICML 2026\] PAC-Bayesian Reinforcement Learning Trains Generalizable Policies](pac-bayesian_reinforcement_learning_trains_generalizable_policies.md)
- [\[ICLR 2026\] Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](../../ICLR2026/reinforcement_learning/helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)
- [\[ICML 2026\] Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control](plug-and-play_benchmarking_of_reinforcement_learning_algorithms_for_large-scale_.md)

</div>

<!-- RELATED:END -->
