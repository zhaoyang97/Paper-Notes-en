---
title: >-
  [Paper Note] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning
description: >-
  [ICLR 2026  \n][Reinforcement Learning][continuous-time RL] This paper proposes VIP (Value Iteration via PINN), the first framework to apply Physics-Informed Neural Networks (PINNs) for solving HJB PDEs in continuous-time multi-agent reinforcement learning. A Value Gradient Iteration (VGI) module is introduced to iteratively refine value gradients. VIP consistently outperforms both discrete-time and continuous-time baselines on continuous-time MPE and MuJoCo multi-agent tasks.
tags:
  - ICLR 2026  \n
  - Reinforcement Learning
  - continuous-time RL
  - MARL
  - HJB equation
  - PINN
  - value gradient iteration
date: 2026-05-08
content_hash: 80dbe7a7ee5d394a
---

# Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning

**Conference**: ICLR 2026  \n  
**arXiv**: [2509.09135](https://arxiv.org/abs/2509.09135)  
**Code**: Available (GitHub link)  
**Area**: Reinforcement Learning  
**Keywords**: continuous-time RL, MARL, HJB equation, PINN, value gradient iteration  

## TL;DR
This paper proposes VIP (Value Iteration via PINN), the first framework to apply Physics-Informed Neural Networks (PINNs) for solving HJB PDEs in continuous-time multi-agent reinforcement learning. A Value Gradient Iteration (VGI) module is introduced to iteratively refine value gradients. VIP consistently outperforms both discrete-time and continuous-time baselines on continuous-time MPE and MuJoCo multi-agent tasks.

## Background & Motivation
**Background**: Most RL methods operate under a discrete-time framework with fixed time-step Bellman updates, yet many real-world scenarios (autonomous driving, robotic control, trading) are inherently continuous-time, involving high-frequency or irregularly spaced decisions.

**Limitations of Prior Work**: Discrete-time RL faces two fundamental issues when approximating continuous-time processes: (1) coarse time steps lead to non-smooth controllers and suboptimal behavior; (2) fine time steps cause an explosion in the number of states and iteration steps. As $\Delta t \to 0$, the Bellman operator may become ill-conditioned, with TD targets dominated by approximation noise.

**Key Challenge**: Continuous-time RL (CTRL) avoids time-discretization issues by replacing Bellman recursion with HJB PDEs, but existing CTRL work is almost exclusively limited to single-agent settings. In multi-agent scenarios, solving the HJB equation becomes extremely challenging due to the curse of dimensionality (state space grows exponentially with the number of agents) and non-stationarity (other agents learn simultaneously).

**Goal**: How to extend HJB-based continuous-time RL to cooperative multi-agent settings?

**Key Insight**: Approximate the viscosity solution of the HJB equation using PINNs (to overcome the curse of dimensionality), and introduce a VGI module to ensure accurate value gradients (addressing the inability of PINN residual losses alone to guarantee gradient accuracy).

**Core Idea**: A dual-pronged PINN + VGI approach for accurately learning value functions and their gradients in continuous-time multi-agent systems.

## Method

### Overall Architecture
VIP adopts the CTDE (Centralized Training with Decentralized Execution) paradigm. The critic is a PINN trained with three losses: HJB residual loss + TD anchor loss + VGI consistency loss. Actors are decentralized policy networks updated using instantaneous advantage functions derived from the HJB residual. Dynamics model $f_\psi$ and reward model $r_\phi$ are jointly learned to support VGI computation.

### Key Designs

1. **PINN Critic for HJB Solving**:

    - **Function**: Approximates the optimal value function with a neural network $V_\theta(x)$, trained by minimizing the HJB PDE residual.
    - **Mechanism**: The HJB residual is $\mathcal{R}_\theta(x_t) = -\rho V_\theta + \nabla_x V_\theta^\top f(x,u) + r(x,u)$; the PINN learns a value function satisfying the PDE by minimizing $\|\mathcal{R}_\theta\|_1$. A TD-style anchor loss provides supervision on the value magnitude.
    - **Design Motivation**: Traditional numerical methods (dynamic programming, level-set methods) are infeasible beyond 6 dimensions due to the curse of dimensionality; the Monte Carlo nature of PINNs alleviates this issue.

2. **Value Gradient Iteration (VGI)**:

    - **Function**: Iteratively refines the value gradient $\nabla_x V(x)$ rather than relying solely on PINN automatic differentiation.
    - **Mechanism**: The VGI target is $\hat{g}_t = \nabla_{x_t} r \cdot \Delta t + e^{-\rho\Delta t} \nabla_{x_t} f^\top \nabla_{x_{t+\Delta t}} V_\theta(x_{t+\Delta t})$, which constitutes a one-step Bellman expansion in gradient space. The loss $\mathcal{L}_{vgi} = \|\nabla_x V_\theta - \hat{g}_t\|^2$ enforces consistency between PINN automatic-differentiation gradients and VGI targets.
    - **Design Motivation**: HJB residual loss alone cannot guarantee gradient accuracy—in high-dimensional multi-agent settings, small gradient errors are amplified by coupled dynamics. Theorem 3.4 proves that VGI updates constitute a contraction mapping, guaranteeing convergence.

3. **Continuous-Time Instantaneous Advantage Function**:

    - **Function**: Derives a continuous-time advantage function directly from the HJB residual for policy updates.
    - **Mechanism**: $A(x_t, u_t) = -\rho V(x_t) + \nabla_x V^\top f(x_t, u_t) + r(x_t, u_t)$, which equals exactly the HJB residual. Each agent updates its decentralized policy via $\mathcal{L}_{p_i} = -A_\theta \log \pi_{\phi_i}$.
    - **Theoretical Guarantee**: A Policy Improvement Lemma is proven, establishing that Q-values are monotonically non-decreasing after a one-step gradient update.

### Loss & Training
The total critic loss is: $\mathcal{L}_{total} = \mathcal{L}_{res} + \lambda_{anchor}\mathcal{L}_{anchor} + \lambda_g\mathcal{L}_{vgi}$. Training is performed jointly with the dynamics and reward models. Tanh activation (required for smooth differentiability in PINNs) significantly outperforms ReLU. Balancing the three loss weights is critical—imbalance leads to stiffness issues in PINN training.

## Key Experimental Results

### Main Results (Continuous-Time MuJoCo + MPE)

| Environment | VIP (w/ VGI) | VIP (w/o VGI) | HJBPPO | DPI | Discrete MADDPG |
|---|---|---|---|---|---|
| Ant 2×4 | **Highest** | Significant drop | Lower | Lower | Substantially lower |
| HalfCheetah 6×1 | **Highest** | Drop | Lower | Lower | Substantially lower |
| Cooperative Nav | **Highest** | Drop | Lower | — | Comparable |
| Predator Prey | **Highest** | Drop | Lower | — | Comparable |

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| Remove VGI | Significant drop on all tasks | VGI is critical for value gradient accuracy |
| ReLU vs. Tanh | ReLU consistently worse | Smooth activations necessary for PDE solving |
| Imbalanced loss weights | Performance degradation | Stiffness issues in PINN training |
| Variable time-interval test | VIP stable, MADDPG degrades | Continuous-time methods are robust to time-step variation |

### Key Findings
- VGI is the core contribution: removing VGI causes the value function contour maps to deviate severely from the ground truth (analytical LQR solution for coupled oscillators).
- All discrete-time baselines (MATD3, MAPPO, MADDPG) degrade substantially in continuous-time settings, especially on Ant and HalfCheetah.
- VIP performance remains nearly constant across different time intervals, while MADDPG degrades sharply as the interval increases.
- Experiments cover state spaces up to 113 dimensions (Ant 4×2, 6 agents), demonstrating the scalability of PINNs to high-dimensional systems.

## Highlights & Insights
- **First systematic continuous-time MARL framework**: Bridges the gap from single-agent to multi-agent CTRL with complete theoretical and empirical validation.
- **VGI as gradient-space Bellman expansion**: The combination of trajectory-based gradient propagation with global PDE constraints is an elegant design, with convergence guaranteed by the contraction mapping proof.
- **Clear diagnosis of discrete-time method limitations**: Variable time-interval experiments and analytical LQR comparisons intuitively demonstrate the bias introduced by time discretization.

## Limitations & Future Work
- The current framework handles only cooperative settings (based on HJB); competitive or mixed-motive scenarios require HJI equations and are left for future work.
- Deterministic system assumption—stochastic dynamics would require a stochastic HJB (SHJB) formulation.
- PINN training stability still requires careful hyperparameter tuning (activation functions, loss weight balancing).
- The method requires learning both dynamics and reward models (model-based), increasing overall complexity.

## Related Work & Insights
- **vs. HJBPPO (single-agent)**: VIP extends PINN-HJB to multi-agent settings and addresses inaccurate value gradients in multi-agent scenarios via VGI.
- **vs. DPI/IPI (continuous-time, single-agent)**: These methods do not scale to high-dimensional multi-agent scenarios; VIP overcomes the curse of dimensionality through PINNs.
- **vs. MADDPG (discrete-time MARL)**: MADDPG degrades severely in continuous-time settings, while VIP remains stable.

## Supplementary Technical Details

### Why Does Continuous Time Matter?
Many real-world multi-agent systems (e.g., robotic formations, autonomous vehicle platoons) are inherently continuous-time. Time discretization introduces approximation errors, particularly in fast-dynamics scenarios. Modeling directly in continuous time avoids the difficulty of time-step selection and yields smoother value function approximations.

### Role of PINNs in MARL

Physics-Informed Neural Networks (PINNs) are employed here to solve the HJB equation by incorporating PDE residuals into the loss function to constrain neural network outputs.

This avoids the curse of dimensionality inherent to traditional grid-based methods in high-dimensional state spaces, enabling efficient value function approximation over continuous state-time domains. Compared to discrete-time-step RL, the continuous-time framework requires no time-step selection and naturally accommodates dynamics operating at different time scales.

## Rating
- Novelty: ⭐⭐⭐⭐ First complete framework combining continuous-time MARL + PINN + VGI
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two major benchmarks, analytical validation, multi-dimensional ablations, comparison with discrete-time methods
- Writing Quality: ⭐⭐⭐⭐ Complete theoretical derivations and rich experiments
- Value: ⭐⭐⭐⭐ Opens a new direction for continuous-time multi-agent control

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)

</div>

<!-- RELATED:END -->
