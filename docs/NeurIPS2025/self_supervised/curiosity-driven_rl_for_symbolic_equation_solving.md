---
title: >-
  [Paper Note] Curiosity-driven RL for Symbolic Equation Solving
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Reinforcement Learning] This work combines curiosity-driven exploration mechanisms (RND, ICM, etc.) with a graph action space based on expression trees…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Reinforcement Learning"
  - "Symbolic Mathematics"
  - "Curiosity-driven Exploration"
  - "PPO"
  - "Expression Tree"
date: 2026-05-08
content_hash: a9ccefc1854d8c2c
---

# Curiosity-driven RL for Symbolic Equation Solving

**Conference**: NeurIPS 2025
**arXiv**: [2510.17022](https://arxiv.org/abs/2510.17022)  
**Code**: None  
**Area**: Self-Supervised
**Keywords**: Reinforcement Learning, Symbolic Mathematics, Curiosity-driven Exploration, PPO, Expression Tree

## TL;DR

This work combines curiosity-driven exploration mechanisms (RND, ICM, etc.) with a graph action space based on expression trees, enabling a PPO agent to solve nonlinear equations involving radicals, exponentials, and trigonometric functions — surpassing prior RL methods that were limited to linear equations.

## Background & Motivation

Symbolic mathematics (e.g., solving algebraic equations, computing integrals) has traditionally relied on hand-crafted rule-based systems (Mathematica, Maple, etc.). Reinforcement learning offers a promising avenue for **autonomously learning transformation strategies** and discovering novel solution procedures. However, applying RL to symbolic mathematics presents two major challenges:

**Combinatorial explosion of the state space**: Each equation can branch into a large number of sub-expressions.

**Large and dynamic action space**: At each step, different algebraic transformations can be applied to different sub-expressions, and the set of valid actions varies with the form of the equation.

Prior work (Poesia et al., 2021) was limited to linear equations, using raw text as state representations and atomic-level operations. Dabelow & Ueda (2024) employed Q-learning but remained confined to linear equations of the form $a_0 + a_1 x = a_2 + a_3 x$. This paper presents the first demonstration of RL-based solving of nonlinear equations.

## Method

### Overall Architecture

Equation solving is formulated as an MDP: the state is an equation represented as a vectorized expression tree, the action is an (operation, term) pair, the reward is based on changes in equation complexity, and the termination condition is $\text{lhs} = x$.

### Key Designs

**State Representation**: Equations are represented as expression trees, vectorized via pre-order traversal and padded to a maximum length of $L=50$. Operators and variables are mapped to integers: $\{\text{add}:1, \text{sub}:2, \text{mul}:3, \ldots, x:5, a:6, \ldots\}$.

**Action Space**: Defined as the Cartesian product $(O \times T)$ along with additional operations:

$$O = \{\text{add, sub, mul, div}\} \cup \{\text{square, sqrt, exp, log, sin, cos, asin, acos}\}$$

$$T = \text{SubExpr}(\text{lhs}) \cup \text{SubExpr}(\text{rhs})$$

$$A = (O \times T) \cup \{(\text{Expand}, \text{None}), (\text{collect}, x), (\text{multiply}, -1)\}$$

The maximum action set size is $|A| = 50$, and illegal actions (e.g., division by zero) are masked.

**Reward Function**: Based on changes in equation complexity (number of nodes plus edges in the expression tree):

$$R(\text{action}) = C(\text{equation}) - C(\text{equation after action})$$

This encourages the agent to take actions that simplify the equation. For example, $C(ax+b) = 5 + 4 = 9$ and $C(ax^2 + bx + c) = 10 + 9 = 19$.

**Curiosity-driven Exploration**: Four curiosity-based methods are combined with PPO — ICM (Intrinsic Curiosity Module), RIDE, NGU (Never Give Up), and RND (Random Network Distillation). Curiosity serves as an intrinsic reward to supplement sparse extrinsic rewards.

### Loss & Training

Standard PPO loss is used together with the intrinsic reward from each curiosity method. Taking RND as an example, the intrinsic reward is:

$$r_{\text{int}} = \|f(s; \theta) - \hat{f}(s; \hat{\theta})\|^2$$

where $f$ is a fixed random network and $\hat{f}$ is a trainable predictor network.

## Key Experimental Results

### Main Results

**Fixed-equation environment** (success rate, $N_{\text{trial}}=10$, $N_{\text{train}}=3\times10^6$):

| Equation | A* | A2C | PPO | PPO-ICM | PPO-RIDE | PPO-NGU | PPO-RND |
|---|---|---|---|---|---|---|---|
| $ax+b$ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| $a/x+b$ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| $c(ax+b)+d$ | 1.00 | 0.20 | 0.10 | 0.70 | 0.60 | 0.40 | **1.00** |
| $c+d/(ax+b)$ | 0.00 | 0.00 | 0.20 | 0.50 | 0.40 | 0.50 | **0.80** |
| $(ax+b)+e(cx+d)$ | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.20 | **0.40** |
| $e+(ax+b)/(cx+d)$ | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.30 | **0.50** |

**Random-equation environment** (large dataset, depth < 5, $10^4/10^3$ train/test split):
- PPO-RND: $\text{test}_{10} \approx 0.8$, $\text{test}_{\text{greedy}} \approx 0.25$
- PPO-RIDE: $\text{test}_{10} \approx 0.8$
- Vanilla PPO: completely fails on the large dataset
- PPO-NGU: unexpectedly the worst performer (possibly due to high memory overhead)

### Ablation Study

- Curiosity method comparison: RND achieves the best performance in both fixed and random environments, with a lightweight computational advantage
- Action set size comparison: $|A| = 40, 70, 100$ yields no significant performance differences
- Solvable equation types: covers radical equations, exponential equations, and trigonometric equations

### Key Findings

1. The expression-tree-based MDP formulation is effective, and curiosity-driven exploration is a **necessary condition** for solving non-trivial equations.
2. RND consistently outperforms all other curiosity-based methods: it is lightweight, efficient, and effective in both fixed and random environments.
3. The current approach is limited to "closed" equations — all operations required for solving are already contained within the original expression.

## Highlights & Insights

1. **First demonstration of RL-based solving of nonlinear symbolic equations**, including equations involving radicals, exponentials, and trigonometric functions.
2. The expression-tree action space design is elegant — sub-expressions serve as operands, allowing the action space to adapt dynamically with the equation structure.
3. The ablation experiments convincingly establish the necessity of curiosity-driven exploration: vanilla PPO/A2C completely fails on complex equations without intrinsic motivation.
4. A clear future benchmark is proposed: quadratic equation solving, which requires "open-ended" reasoning such as completing the square.

## Limitations & Future Work

- Only handles "closed" equations — solving does not require introducing new terms absent from the original equation.
- Unable to solve quadratic equations, which demand "generative" reasoning such as discovering the term $(b/2a)^2$.
- As a workshop paper, experimental scale is limited.
- $\text{test}_{\text{greedy}} \approx 0.25$ indicates insufficient generalization under a greedy policy.

## Related Work & Insights

- Poesia et al. (2021): linear equations + raw text action space + contrastive learning; this paper extends to nonlinear equations + expression trees + PPO with curiosity.
- Dabelow & Ueda (2024): Q-learning framework, still restricted to linear equations.
- A natural open question raised by this work is whether AlphaZero-style planning algorithms could handle "open-ended" equations requiring long-horizon reasoning.

## Rating

- ⭐ Novelty: 4/5 — First demonstration of the feasibility of RL-based nonlinear equation solving
- ⭐ Experimental Thoroughness: 3/5 — Experiments are adequate for a workshop paper but limited in scale
- ⭐ Writing Quality: 4/5 — MDP formalization is clear; limitations are discussed candidly
- ⭐ Value: 3/5 — An exploratory contribution that opens a new direction, though practical applicability remains distant

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[NeurIPS 2025\] Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
