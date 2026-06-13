---
title: >-
  [Paper Note] Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form
description: >-
  [ICLR 2026][Reinforcement Learning][Continuous-time RL] This paper proposes the first continuous-time multi-agent RL framework that explicitly handles state constraints. By reformulating the discontinuous constrained val…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Continuous-time RL"
  - "Multi-agent"
  - "Safety Constraints"
  - "HJB Equation"
  - "Epigraph Reformulation"
date: 2026-05-08
content_hash: 3c3b195d854dcc2b
---

# Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form

**Conference**: ICLR 2026
**arXiv**: [2602.17078](https://arxiv.org/abs/2602.17078)  
**Code**: [GitHub](https://github.com/xuefeng-wang/EPI)  
**Area**: Reinforcement Learning
**Keywords**: Continuous-time RL, Multi-agent, Safety Constraints, HJB Equation, Epigraph Reformulation

## TL;DR

This paper proposes the first continuous-time multi-agent RL framework that explicitly handles state constraints. By reformulating the discontinuous constrained value function into a continuous representation via the epigraph form, and combining an improved PINN-based actor-critic method, the framework achieves safe and stable continuous-time multi-agent control.

## Background & Motivation

Most multi-agent reinforcement learning (MARL) algorithms are built upon discrete-time MDPs and Bellman equations, assuming fixed decision intervals. However, many real-world applications—such as autonomous driving, financial trading, and robotic collaboration—are inherently continuous-time control problems, where discrete-time approximations can lead to performance degradation and training instability under high-frequency or irregular time steps.

Existing continuous-time MARL methods are based on the Hamilton-Jacobi-Bellman (HJB) equation and employ physics-informed neural networks (PINNs) to approximate value functions. However, these methods **largely neglect safety constraints** (e.g., collision penalties), because state constraints introduce discontinuities in the value function that PINN-based approximations cannot accurately capture.

The root cause of this challenge is that safe MARL requires handling constraints, but constraints induce value function discontinuities, while PINNs can only approximate smooth functions. This paper addresses the contradiction through **epigraph reformulation**, which converts the discontinuous constrained value into a continuous representation.

## Method

### Overall Architecture

The EPI framework consists of: (1) formalizing safe CT-MARL as a continuous-time constrained MDP (CT-CMDP); (2) an epigraph reformulation that introduces an auxiliary state $z$ to unify return and constraint objectives; and (3) an improved actor-critic architecture based on inner-outer optimization, comprising a PINN-based critic and decentralized actors.

### Key Designs

1. **Epigraph Reformulation (Core Theoretical Contribution)**:

    - Function: Introduces an auxiliary state $z(t)$ to convert constrained optimization into an unconstrained continuous value function.
    - Mechanism: Defines an auxiliary value function $V(x,z) = \min_{u} \max\{\max_\tau c(x(\tau)), \int_t^\infty \gamma^{\tau-t} l(x(\tau),u(\tau))d\tau - z\}$.
    - Lemma 3.1 proves $v(x) = \min\{z \in \mathbb{R} | V(x,z) \leq 0\}$, converting the retrieval of the constrained value $v$ into a zero-level-set search of $V$.
    - Design Motivation: $V(x,z)$ is continuous (Theorem 3.3), unlike the original discontinuous constrained value function, making it amenable to PINN approximation.

2. **Improved Outer Optimization (Computing $z^*$)**:

    - Function: Directly computes the optimal $z^*$ during training rather than sampling it randomly.
    - Mechanism: $z^* = \min\{z | \max\{V_\phi^{\text{cons}}(x), V_\psi^{\text{ret}}(x) - z\} \leq 0\}$.
    - Design Motivation: Prior methods (e.g., EPPO) randomly sample $z$, introducing non-stationary noise that destabilizes policy updates and requiring expensive root-finding at execution time. EPI designs the return and constraint networks to depend only on $x$ (not $z$), enabling direct use of $z^*$ during training and eliminating root-finding at inference.

3. **PINN-based Critic (Triple Loss)**:

    - Function: Trains the value function with three complementary loss terms.
    - Mechanism:
        - **Residual Loss**: Penalizes violations of the HJB PDE — $\mathcal{L}_{\text{Residual}} = (\max\{c(x)-\tilde{V}, \min_u \mathcal{H}\})^2$.
        - **Target Loss**: Trajectory-based numerical targets — $\mathcal{L}_{\text{Target}} = (V_{\text{tgt}} - \tilde{V})^2$, serving as an anchor when boundary conditions are absent in the infinite-horizon setting.
        - **Value Gradient Iteration (VGI)**: Enforces consistency of the constrained value gradient, ensuring accuracy of $\nabla_x V$.
    - Design Motivation: The residual loss alone is insufficient for unbounded problems; value gradients are critical for accurate policy updates.

4. **Decentralized Actor Learning**:

    - Function: Updates decentralized policies using the epigraph advantage function.
    - Mechanism: $A(x_t,z_t^*,u_t) = \max\{c(x_t)-V, \nabla_x V \cdot f(x,u) - \partial_z V \cdot l(x,u) + \ln\gamma \cdot V\}$.
    - Learned dynamics network $f_\xi$ and cost network $l_\phi$ are used in place of unknown ground-truth functions.
    - Design Motivation: Centralized Training with Decentralized Execution (CTDE); each agent requires only local observations at execution time.

### Loss & Training

Total critic loss: $\mathcal{L}_{\text{Critic}} = \lambda_{\text{res}}\mathcal{L}_{\text{Residual}} + \lambda_{\text{tgt}}\mathcal{L}_{\text{Target}} + \lambda_{\text{vgi}}\mathcal{L}_{\text{VGI}}$. Actor loss: $\mathcal{L}_{\text{actor}} = \mathbb{E}[A_\theta(x,z^*,u)]$. Loss weights are determined via grid search.

## Key Experimental Results

### Main Results (Continuous-Time Safe MPE + MuJoCo)

| Method | Approach | Constraint & Cost Performance |
|--------|----------|-------------------------------|
| MACPO | Trust-region constraints | Overly conservative |
| MAPPO-Lag | Lagrangian relaxation | Unstable balance |
| SAC-Lag | Off-policy + Lagrangian | Poor constraint satisfaction |
| EPPO | Random sampling of $z$ | Stuck at suboptimal |
| CBF | Control barrier functions | Conservative but reasonable |
| **EPI (Ours)** | **Direct $z^*$ optimization** | **Near-optimal on both cost and constraints** |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|------------|---------|
| Full EPI | Optimal | Triple loss + $z^*$ optimization |
| w/o Target Loss | Significant degradation | Value function drift in unbounded problems |
| w/o VGI Loss | Severe degradation | Inaccurate value gradients → harmful policy updates |
| w/o Residual Loss | Minor impact | PDE structure less critical when VGI is present |
| Over-weighting any loss (×20) | Degradation | Balanced weights are optimal |

### Key Findings

- EPPO converges to suboptimal solutions due to random sampling of $z$.
- Target and VGI losses are critical for infinite-horizon problems; the residual loss plays a relatively minor role.
- EPI consistently achieves the lowest cost and constraint violations across MPE scenarios (Formation, Line, Target).
- EPI also outperforms baselines in MuJoCo environments (HalfCheetah, Ant).

## Highlights & Insights

- **First to introduce safety constraints into CT-MARL**: Fills a critical gap in continuous-time safe MARL.
- **Elegance of epigraph reformulation**: Converting a discontinuous value function into a continuous one enables PINN-based methods to work effectively.
- **Direct $z^*$ optimization**: Eliminates the source of noise in prior methods and removes runtime overhead at execution.
- **Theoretical guarantees** (Theorem 3.3): Proves the existence and uniqueness of viscosity solutions to the epigraph HJB PDE.

## Limitations & Future Work

- Learning auxiliary dynamics and cost networks ($f_\xi, l_\phi$) increases model complexity.
- Loss weights $(\lambda_{\text{res}}, \lambda_{\text{tgt}}, \lambda_{\text{vgi}})$ are determined via grid search; adaptive schemes could be explored.
- Experiments are limited in scale (2–6 agents); scalability to large agent populations remains to be validated.
- PINN-based methods may face training difficulties in high-dimensional state spaces.

## Related Work & Insights

- Wang et al. (2025) systematically study CT-MARL but neglect safety constraints; EPI directly complements this gap.
- EPPO (Zhang et al., 2025b) introduces the epigraph form but samples $z$ randomly; EPI's improved scheme yields more stable training.
- So and Fan (2023) apply the epigraph form to single-agent safe control; this work extends it to multi-agent RL.
- Insight: The key to PDE-based RL methods lies not in the residual loss but in the accuracy of value gradients.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First unified treatment of safety constraints, continuous time, and multi-agent settings; highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual benchmarks (MPE and MuJoCo) with detailed ablations; agent scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations and clear architectural diagrams.
- Value: ⭐⭐⭐⭐ Opens a new direction in safe CT-MARL with both theoretical and methodological contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning](policyflow_policy_optimization_with_continuous_normalizing_flow_in_reinforcement.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)

</div>

<!-- RELATED:END -->
