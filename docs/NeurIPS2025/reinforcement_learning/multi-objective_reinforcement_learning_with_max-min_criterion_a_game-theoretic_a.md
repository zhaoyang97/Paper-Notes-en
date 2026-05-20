---
title: >-
  [Paper Note] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach
description: >-
  [NeurIPS 2025][Reinforcement Learning][Multi-objective reinforcement learning] This work reformulates entropy-regularized max-min multi-objective reinforcement learning as a two-player zero-sum regularized game…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Multi-objective reinforcement learning"
  - "max-min fairness"
  - "game theory"
  - "mirror descent"
  - "last-iterate convergence"
date: 2026-05-08
content_hash: 0d12fb0e1f50f43a
---

# Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach

**Conference**: NeurIPS 2025
**arXiv**: [2510.20235](https://arxiv.org/abs/2510.20235)  
**Code**: [GitHub](https://github.com/whbyeon/ERAM-ARAM)  
**Area**: Reinforcement Learning
**Keywords**: Multi-objective reinforcement learning, max-min fairness, game theory, mirror descent, last-iterate convergence

## TL;DR

This work reformulates entropy-regularized max-min multi-objective reinforcement learning as a two-player zero-sum regularized game, proposes the ERAM/ARAM algorithms with closed-form weight updates via mirror descent, and establishes global last-iterate convergence, substantially outperforming baselines across multiple MORL benchmarks.

## Background & Motivation

1. **Background**: Multi-objective reinforcement learning (MORL) has broad applications in autonomous driving, resource allocation, and related domains. Utility-function-based approaches are predominant, with the weighted sum being the most common utility function.

2. **Limitations of Prior Work**: Weighted-sum utility functions are ill-suited for fairness-sensitive settings. The max-min criterion ($\max_\pi \min_{k} V_k^\pi$) is more natural for such settings but poses significant optimization challenges: the min operator is non-differentiable, and standard RL methods cannot be directly applied. The existing method of [Park et al.] is computationally expensive, guarantees only average-iterate convergence, and requires substantial memory to store copies of Q-networks.

3. **Key Challenge**: The nonlinearity introduced by the min operation in max-min MORL renders the standard Bellman operator inapplicable; direct optimization causes the weight vector to oscillate sharply among one-hot solutions at the worst-performing dimensions, permitting only average-iterate convergence.

4. **Goal**: To design an efficient max-min MORL algorithm with provable last-iterate convergence guarantees and low computational overhead.

5. **Key Insight**: Leveraging the max-min = min-max identity (which holds under entropy regularization), the problem is recast as finding a Nash equilibrium of a two-player zero-sum game, with entropy regularization applied to the adversary to stabilize training.

6. **Core Idea**: By adding entropy regularization $H(w)$ to the weight vector $w$, the algorithm obtains a closed-form (softmax) update for $w$, while simultaneously preventing weight oscillation and enabling last-iterate convergence.

## Method

### Overall Architecture

The objective $\max_\pi \min_k V_{k,\tau}^\pi$ is reformulated as a two-player zero-sum game $\mathcal{RG}$: the Learner optimizes policy $\pi_\theta$ (maximization), and the Adversary selects weights $w \in \Delta^K$ (minimization). The utility function is $u = \langle w, \mathbf{V}_\tau^{\pi_\theta} \rangle - \tau_w H(w)$.

### Key Designs

**1. Two-Player Zero-Sum Regularized Game Formulation (Theorem 3.1)**

- **Function**: Establishes an equivalence between max-min MORL and solving for the Nash equilibrium of a regularized game.
- **Mechanism**: Proves that $\max_\pi \min_w \langle w, \mathbf{V}_\tau^\pi \rangle = \min_w \max_\pi \langle w, \mathbf{V}_\tau^\pi \rangle$ holds under entropy regularization, and that the policy component of the Nash equilibrium is precisely the solution to the max-min MORL problem.
- **Design Motivation**: Once recast as a zero-sum game, established game-theoretic learning methods (mirror descent) can be directly applied.

**2. ERAM Algorithm: Closed-Form Updates for Both Players**

- **Function**: A single-loop, computationally efficient algorithm.
- **Mechanism**:
    - Learner: Updates $\theta$ via natural policy gradient (NPG); under softmax parameterization, the update admits a closed form: $\pi_{\theta_{t+1}}(a|s) = \frac{1}{Z} (\pi_{\theta_t}(a|s))^\alpha \exp(\frac{1-\alpha}{\tau} Q_{w_t,\tau}^{\pi_{\theta_t}}(s,a))$
    - Adversary: Updates $w$ via modified mirror descent with $-H(w)$ regularization, yielding the closed-form update: $w_{t+1} = \text{softmax}(-\frac{1-\beta}{\tau_w}\mathbf{V}^{\pi_{\theta_t}} + \beta \log w_t)$
- **Design Motivation**: Choosing KL divergence as the Bregman divergence combined with negative entropy regularization yields an analytic solution for $w$ and ensures $\beta \in (0,1)$ holds unconditionally.

**3. ARAM: Adaptive Regularization Enhancement**

- **Function**: Improved joint optimization across multiple objectives.
- **Mechanism**: Replaces $H(w)$ (equivalent to KL distance from the uniform distribution) with the KL distance from a dynamic reference vector $c$, i.e., $-D_{KL}(w\|c)$, where $c_i = \text{softmax}(\mathbb{E}[r_i(s,a) \cdot r_{i'}(s,a)])$ and $i'$ denotes the worst-performing objective in the previous iteration.
- **Design Motivation**: In ERAM, $w$ tends toward the uniform distribution, which may obscure critical underperforming objectives; ARAM directs greater attention to dimensions correlated with the worst objective without exclusively focusing on it.

### Loss & Training

- Policy side: PPO (deep RL) or closed-form NPG (tabular setting).
- Weight side: Closed-form softmax update.
- Theoretical guarantee: Last-iterate exponential convergence at rate $\rho(\eta,\lambda)^t$, with iteration complexity $O(\frac{1}{\epsilon^2}\log\frac{1}{\epsilon_{acc}})$.

## Key Experimental Results

### Main Results

**Traffic Signal Control**

| Environment | ARAM | ERAM | Park et al. | GGF-PPO | GGF-DQN | Avg-DQN |
|-------------|------|------|-------------|---------|---------|---------|
| Base-4 | **-1160** | -1387 | -1681 | -1731 | -1838 | -2774 |
| Asym-4 | **-2696** | -2732 | -3510 | -3501 | -3053 | -4245 |
| Asym-16 | **-15043** | -17334 | -23663 | -21663 | -17792 | -27499 |

### Ablation Study

| Metric | ERAM | Park et al. |
|--------|------|-------------|
| Convergence type | Last-iterate ✓ | Average-iterate |
| Model parameters (Base-4) | 13,704 | 274,084 |
| Parameter reduction | — | **95%** |
| Training time Base-4 (min) | **111±2.6** | 346±14 |
| Training time Asym-16 (min) | **356±27** | 1125±95 |

### Key Findings

- ARAM achieves consistent state-of-the-art performance across all environments; ERAM ranks second with a simpler architecture.
- Compared to Park et al., the proposed method reduces parameter count by approximately **95%** and training time by roughly **3×**.
- ERAM exhibits last-iterate convergence (monotonically approaching the optimum), whereas Park et al. displays oscillatory behavior.
- The advantage is more pronounced in the 16-objective setting (Asym-16), demonstrating superior scalability in high-dimensional objective spaces.

## Highlights & Insights

- Strong alignment between theory and practice: the last-iterate convergence theory directly informs the algorithm design.
- Entropy regularization serves a dual purpose: it addresses policy uncertainty while simultaneously yielding a closed-form update for $w$.
- The elegant reformulation of MORL fairness as a game-theoretic problem enables the direct reuse of a rich body of game-theoretic tools.
- ARAM's adaptive reference vector represents a principled middle ground between uniform attention to all objectives (ERAM) and exclusive focus on the worst objective (GGF-PPO).

## Limitations & Future Work

- Theoretical analysis of ARAM is left for future work; convergence proofs currently cover ERAM only.
- Tabular theory limitation: convergence proofs rely on closed-form NPG under softmax policy parameterization; theoretical guarantees for the deep RL setting remain unestablished.
- The lower bound condition on $\tau_w$ is conservative ($\tau_w \geq O(K/\tau(1-\gamma)^4)$), potentially requiring more careful hyperparameter tuning in practice.
- The paper focuses exclusively on the min utility function; extensions to other nonlinear utilities (e.g., Nash social welfare) are not discussed.

## Related Work & Insights

- The convex reformulation approach of Park et al. [2024] is the direct predecessor; the present work substantially simplifies the framework through a game-theoretic perspective.
- Connection to distributionally robust RL: $w \in \Delta^K$ can be interpreted as a distribution over a finite uncertainty set, with entropy regularization corresponding to internalized reward uncertainty.
- GGF-PPO is a special case of the proposed method ($\tau_w = 0$, no Bregman constraint → one-hot $w$).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The game-theoretic reformulation is not entirely novel, but the closed-form $w$ update and last-iterate convergence proof constitute important contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple environments including tabular and traffic signal control settings, with thorough complexity comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ The theorem–algorithm–experiment pipeline is clearly structured; the theoretical exposition is rigorous.
- **Value**: ⭐⭐⭐⭐ Provides the most efficient theoretically grounded algorithm to date for fairness-aware RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning](trico_triadic_game-theoretic_co-training_for_robust_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Solving Neural Min-Max Games: The Role of Architecture, Initialization & Dynamics](solving_neural_min-max_games_the_role_of_architecture_initialization_dynamics.md)
- [\[NeurIPS 2025\] VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play](volleybots_a_testbed_for_multi-drone_volleyball_game_combining_motion_control_an.md)
- [\[NeurIPS 2025\] A Differential and Pointwise Control Approach to Reinforcement Learning](a_differential_and_pointwise_control_approach_to_reinforceme.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)

</div>

<!-- RELATED:END -->
