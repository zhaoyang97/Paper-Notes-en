---
title: >-
  [Paper Note] The World Is Bigger! A Computationally-Embedded Perspective on the Big World Hypothesis
description: >-
  [NeurIPS 2025][Reinforcement Learning][Continual Learning] This paper formalizes the Big World Hypothesis from a computationally-embedded perspective, proves that agents embedded in universal-local environments are inherently capacity-constrained, proposes *interactivity* as a computational measure of continual adaptability, and empirically demonstrates that deep nonlinear networks fail to maintain interactivity while deep linear networks improve interactivity as capacity increases.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Continual Learning
  - Big World Hypothesis
  - Embedded Agents
  - Algorithmic Information Theory
  - Interactivity
date: 2026-05-08
content_hash: b0bdf8e6de1f5a57
---

# The World Is Bigger! A Computationally-Embedded Perspective on the Big World Hypothesis

**Conference**: NeurIPS 2025
**arXiv**: [2512.23419](https://arxiv.org/abs/2512.23419)
**Code**: [GitHub](https://github.com/AlexLewandowski/bigger-world-interactivity)
**Area**: Reinforcement Learning
**Keywords**: Continual Learning, Big World Hypothesis, Embedded Agents, Algorithmic Information Theory, Interactivity

## TL;DR

This paper formalizes the Big World Hypothesis from a computationally-embedded perspective, proves that agents embedded in universal-local environments are inherently capacity-constrained, proposes *interactivity* as a computational measure of continual adaptability, and empirically demonstrates that deep nonlinear networks fail to maintain interactivity while deep linear networks improve interactivity as capacity increases.

## Background & Motivation

### State of the Field

The core motivation behind continual learning is the Big World Hypothesis — the environment is larger than the agent, so agents should continually adapt rather than converge to a fixed solution. Existing formalizations typically impose **explicit constraints** on agents: limiting memory, the expressivity of function approximators, computation, or energy. Kumar et al. (2023, 2024) proposed information-theoretic capacity constraints, but these are difficult to measure and enforce beyond analytically tractable simple systems.

### Root Cause

Explicit constraints suffer from two fundamental problems:

**They limit the effectiveness of scaling**: if constraints are manually imposed, increasing agent capacity to improve performance contradicts the constraint.

**They lack generality**: the choice of constraint is often ad hoc and difficult to generalize across settings.

In conventional RL, agents and environments are treated as **separate entities** (Figure 1a), allowing agents to scale arbitrarily beyond the environment in principle. The AIXI framework (Figure 1b) considers universal environments but is itself uncomputable and does not address learning under finite capacity.

### Starting Point

This paper proposes a fundamentally different perspective: **embedded agents** (Figure 1c). Rather than being an external entity independent of the environment, the agent is a finite subset of the environment's state space, simulated by the environment's transition dynamics. This embedding relation provides an **implicit constraint** — the environment necessarily has greater capacity than any agent it contains. This mirrors the physical world: any physical entity is embedded in the physical world and must be smaller than it.

## Method

### Overall Architecture

The paper constructs a three-level theoretical framework:
1. Defining a **universal-local environment** as the "world" in which agents exist
2. Defining **embedded automata** within it as the formal representation of agents
3. Proposing **interactivity** as a measure of continual adaptability

### Key Designs

1. **Universal-Local Environment**: Defined as an algorithmic Markov process $\mathcal{E} = (\Omega, \Xi, \mathbb{T})$ satisfying two properties:

   - **Computational universality**: the transition function can simulate any algorithm (equivalent to a universal Turing machine)
   - **Uniform locality**: the transition function decomposes into identical local transition functions acting on local regions of the state space

   Conway's Game of Life is a concrete example — it is computationally universal (capable of simulating a universal Turing machine) and uniformly local (each cell's transition rule is identical and depends only on 8 neighbors).

2. **Embedded Automaton**: Defined as $\mathcal{A} := (\Omega|_X, \Omega|_Y, \Omega|_\Theta, u, \pi)$, where $\Omega|_\Theta$ is the internal state space, $\Omega|_X, \Omega|_Y$ are input/output spaces, $u$ is the state update function (corresponding to the learning algorithm), and $\pi$ is the output function (corresponding to the policy). A key result (Proposition 2) proves that when the agent's boundary space coincides with its input/output space, the automaton is equivalent to a stateful policy operating on a POMDP.

   Proposition 3 proves that the capacity of an embedded automaton is upper-bounded by $|\Theta|$, so there necessarily exist input-output behaviors it cannot realize — this is the core of the **implicit constraint**.

3. **Interactivity**: Measures the "predictable complexity" in the agent's future behavior:

$$\mathbb{I}_T(\mathcal{A}|x_t, b_{0:t-1}) := \mathbb{K}_\mathcal{E}(b_{t:t+T-1}|\epsilon) - \mathbb{K}_\mathcal{E}(b_{t:t+T-1}|b_{0:t-1})$$

where $\mathbb{K}$ denotes algorithmic complexity (Kolmogorov complexity) and $b$ denotes the behavior sequence. High interactivity implies: (i) future behavior exhibits high unconditional complexity (complex and diverse), and (ii) past behavior is strongly predictive of future behavior (something has been learned).

   Theorem 1 proves that maximum interactivity is capacity-bounded: $\alpha C(\mathcal{A}) - O(1) < \max_\mathcal{A} \mathbb{I}_T \leq C(\mathcal{A}) + O(1)$

### Loss & Training

Since algorithmic complexity is uncomputable, the paper approximates it using **agent-relative complexity** — measured via temporal-difference (TD) error as a proxy for prediction error:

$$\hat{\mathbb{K}}_\mathcal{A}(b_{t:t+T-1}|b_{0:t-1}) = \sum_{k=0}^{T-1}\delta_{t+k}^2(\theta_{t+k-1})$$

Interactivity is approximated as the difference between static and dynamic TD errors, and the policy is trained to maximize this difference:

$$J(\theta) = \sum_{k=1}^{T}\delta_{t+k}^2(z_t, \theta) - \delta_{t+k}^2(z_{t+k}, \theta)$$

The value function uses linear parameterization $v(b_t; \mathbf{W}_t) := \mathbf{W}_t b_t$, and the policy network uses a deep network (linear or ReLU activations) with outputs normalized via RMSNorm.

## Key Experimental Results

### Behavior Self-Prediction Task (Environment-Free Setting)

| Policy Network Type | Depth | Width | Maintains Interactivity | Behavioral Characteristics |
|---|---|---|---|---|
| Deep ReLU | D=2 | 1000 | No, collapses rapidly | No predictable structure |
| Deep Linear | D=2 | 1000 | Yes, sustained | Non-stationary waveforms, locally predictable |

### Capacity Scaling Experiments (Deep Linear Networks)

| Configuration | Interactivity Level | Trend |
|---|---|---|
| Increasing width | Marginal improvement | Wider → slightly higher interactivity |
| Increasing depth (D=1→4) | Significant improvement | Deeper → higher interactivity + more oscillation |

### Key Findings

- **Failure of deep nonlinear networks**: ReLU networks cannot produce behavior sequences with low prediction error under a dynamic value function but high error under a static one, possibly due to non-stationarity inducing plasticity-loss-like phenomena.
- **Success of deep linear networks**: These produce non-stationary waveform-like behavior sequences that are locally predictable by a linear function but require dynamic updates for global prediction.
- **Verification of Theorem 2 (Big World Theorem)**: Agents maximizing interactivity are suboptimal if they stop learning, and maximum interactivity increases with capacity.

## Highlights & Insights

- **Theoretical elegance**: Capacity constraints arise naturally from the embedded-agent perspective rather than being imposed artificially, providing a first-principles derivation of the continual learning problem.
- **Interactivity vs. curiosity-driven exploration**: Interactivity does not pursue an accurate model of the environment (as curiosity does) but instead seeks complex yet predictable behavior — a closer approximation to measuring continual learning ability.
- **Environment-free benchmark**: Maximizing interactivity itself constitutes a continual learning benchmark that requires neither an external environment nor a dataset.
- **Insight on linear vs. nonlinear networks**: Plasticity loss may be an inherent deficiency of nonlinear networks in non-stationary learning settings.

## Limitations & Future Work

- Experiments are limited to network depth $D=4$ (though effective depth is $T \cdot D = 40$ via meta-gradients).
- The computational complexity of the meta-gradient approach is $O(HD^2)$, posing cubic growth challenges when scaling depth and horizon.
- Validation is currently restricted to the self-prediction task; interactivity has not yet been used as an intrinsic reward in external environments.
- Evaluation criteria require rethinking — fixed-agent evaluation as in conventional ML is incompatible with continual learning.
- The impact of hyperparameter choices on long-term continual learning performance remains unclear.

## Related Work & Insights

- **Relation to AIXI**: AIXI is Bayes-optimal in universal environments but uncomputable; this paper considers computable agents with finite capacity embedded in such environments.
- **Comparison with Empowerment**: Empowerment maximizes control (action–future-state mutual information), whereas interactivity pursues complex and predictable behavior (past–future algorithmic mutual information).
- **Comparison with Free Energy Minimization**: Free energy minimization favors predictable states, whereas interactivity actively avoids low-complexity predictable states.
- The paper implies a conjecture: if an agent can sustain a given level of interactivity, it may also be capable of learning any goal-directed behavior with equal or lower interactivity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reframes the continual learning problem from the computationally-embedded perspective; the theoretical framework is highly original.
- Experimental Thoroughness: ⭐⭐⭐ The experimental scale is constrained by the computational cost of meta-gradients, but suffices to validate the core claims.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is clear and complete, proceeding seamlessly from formal definitions to theorems to experimental validation.
- Value: ⭐⭐⭐⭐⭐ Provides a novel theoretical foundation for continual learning; the interactivity framework has far-reaching implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model](bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Meta-World+: An Improved, Standardized, RL Benchmark](meta-world_an_improved_standardized_rl_benchmark.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[NeurIPS 2025\] Emergent World Beliefs: Exploring Transformers in Stochastic Games](emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)

</div>

<!-- RELATED:END -->
