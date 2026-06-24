---
title: >-
  [Paper Note] Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm
description: >-
  [NeurIPS 2025][Robotics][Inverse Reinforcement Learning] This paper proposes the TRRO theoretical framework and the PIRO practical algorithm, which guarantee monotonic improvement of reward function updates in IRL via a Minorization-Maximization procedure, achieving stability guarantees analogous to those of TRPO/PPO in forward RL.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Inverse Reinforcement Learning"
  - "Trust Region"
  - "Reward Learning"
  - "Non-Adversarial IRL"
  - "Monotonic Improvement"
date: 2026-05-08
content_hash: a09a621d697e5628
---

# Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm

**Conference**: NeurIPS 2025
**arXiv**: [2509.23135](https://arxiv.org/abs/2509.23135)  
**Code**: [Available](https://github.com/PolynomialTime/PIRO)  
**Area**: Reinforcement Learning
**Keywords**: Inverse Reinforcement Learning, Trust Region, Reward Learning, Non-Adversarial IRL, Monotonic Improvement

## TL;DR

This paper proposes the TRRO theoretical framework and the PIRO practical algorithm, which guarantee monotonic improvement of reward function updates in IRL via a Minorization-Maximization procedure, achieving stability guarantees analogous to those of TRPO/PPO in forward RL.

## Background & Motivation

Inverse reinforcement learning (IRL) learns reward functions from expert demonstrations. Modern IRL methods fall into two main paradigms:

**Adversarial IRL** (e.g., GAIL, AIRL): models reward learning as a minimax game, alternating between reward and policy optimization. Theoretically elegant but practically unstable and sensitive to hyperparameters.

**Non-adversarial IRL** (e.g., SQIL, IQ-Learn, ML-IRL): couples reward and policy via energy-based models and performs joint updates. Empirically more stable, but lacks principled control over reward updates — with no guarantee that each update step moves in the correct direction.

The paper identifies a key observation: existing non-adversarial IRL methods all essentially **maximize the likelihood of expert behavior** (equivalently, minimize the imitation gap). This unified perspective motivates the core idea: if each update step can be **guaranteed to increase the likelihood**, stable IRL training becomes achievable.

This is a perfect symmetric counterpart to TRPO in forward RL:
- TRPO guarantees monotonic policy improvement under a fixed reward.
- TRRO guarantees monotonic reward improvement given expert behavior.

The paper claims to fill the "right half of this symmetric picture."

## Method

### Overall Architecture

TRRO/PIRO follows the non-adversarial, explicit reward (ER) learning paradigm:
1. **Unified perspective**: proves that SQIL, IQ-Learn, f-IRL, and ML-IRL all optimize expert behavior likelihood.
2. **Theoretical contribution**: the TRRO framework guarantees monotonic improvement in inverse reward optimization via the MM algorithm.
3. **Practical algorithm**: PIRO realizes TRRO through adaptive regularization and approximate policy optimization.

### Key Designs

1. **Equivalent form of the likelihood objective (Proposition 1)**:

    - The log-likelihood of ML-IRL: $\ell(\boldsymbol{\theta}) = \mathbb{E}_{\rho^{\pi_E}}[\log \pi_{\boldsymbol{\theta}}(\mathbf{a}|\mathbf{s})]$
    - Equivalent to the imitation gap: $\ell(\boldsymbol{\theta}) = J(\pi_E, r_{\boldsymbol{\theta}}) - J(\pi_{\boldsymbol{\theta}}, r_{\boldsymbol{\theta}})$
    - The gradient is the difference of reward gradients under two occupancy measures: $\nabla_{\boldsymbol{\theta}} \ell = \mathbb{E}_{\rho^{\pi_E}}[\nabla r_{\boldsymbol{\theta}}] - \mathbb{E}_{\rho^{\pi_{\boldsymbol{\theta}}}}[\nabla r_{\boldsymbol{\theta}}]$
    - This bypasses the inner RL loop, reducing nested optimization to a single-loop procedure.

2. **Trust Region Reward Optimization (TRRO, Theorem 3)**:

    - Introduces a surrogate function $\ell_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta})$: the imitation gap computed using the old policy $\pi_{\text{old}}$ in place of the new policy.
    - Proposition 2 shows that the surrogate matches the original objective to first order at $\boldsymbol{\theta}_{\text{old}}$.
    - Theorem 3 establishes a lower bound: $\ell(\boldsymbol{\theta}_{\text{new}}) \geq \ell_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}_{\text{new}}) - C\epsilon_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}_{\text{new}})$
    - where $\epsilon = \max_{s,a} |r_{\boldsymbol{\theta}_{\text{new}}} - r_{\boldsymbol{\theta}_{\text{old}}}|$ measures the reward change.
    - Maximizing this lower bound guarantees that $\ell$ is monotonically non-decreasing (Corollary 4).
    - This constitutes an MM algorithm: the surrogate minorizes the original objective and is tangent to it at $\boldsymbol{\theta}_{\text{old}}$.

3. **Proximal Inverse Reward Optimization (PIRO)**:

    - The theoretical constant $C$ is too large for direct use; it is replaced by a tunable coefficient $\mu > 0$.
    - The $\ell^\infty$ norm in $\epsilon$ is non-differentiable; it is approximated by an $L^2$ norm estimated over expert data and policy rollouts.
    - Objective: $L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) = \ell_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) - \mu \bar{\epsilon}_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta})$
    - $\mu$ is adapted: if $\bar{\epsilon} > \bar{\epsilon}^{\text{target}} \times x$, then $\mu \leftarrow \mu \times y$ (and vice versa).
    - The policy is approximately optimized via a fixed number of SAC iterations rather than exact solving.

### Loss & Training

PIRO alternates between:
- **Policy update**: $k$ rounds of SAC iteration based on the current reward $r_{\boldsymbol{\theta}_{\text{old}}}$.
- **Reward update**: $n$ gradient ascent steps with gradient $\nabla_{\boldsymbol{\theta}} L = \mathbb{E}_{\hat{D}_E}[\nabla r_{\boldsymbol{\theta}}] - \mathbb{E}_{D_S}[\nabla r_{\boldsymbol{\theta}}] - \mu \nabla \bar{\epsilon}$
- Setting $k=n=1, \mu=0$ recovers standard non-adversarial IRL.

## Key Experimental Results

### Main Results: MuJoCo and Gym Robotics

| Task | Expert | GAIL | AIRL | HyPE | IQ-Learn | ML-IRL | f-IRL | **PIRO** | Gain |
|------|--------|------|------|------|----------|--------|-------|----------|------|
| Ant-v4 | 5926 | 997 | 991 | 2801 | 3590 | 5383 | 980 | **5967** | +585 |
| Humanoid-v4 | 5501 | 508 | 281 | 718 | 1848 | 5573 | 470 | **5955** | +382 |
| Walker2d-v4 | 5525 | 4158 | 73 | 1479 | 3023 | 4795 | 244 | **5644** | +849 |
| AntMaze-UMaze | 35.6 | 5.2 | 4.5 | 11.9 | 3.9 | 4.2 | 3.6 | **25.7** | +13.8 |
| AntMaze-Large | 11.5 | 0.9 | 3.4 | 1.5 | 0.8 | 0.3 | 0.9 | **8.8** | +5.4 |

### Ablation Study

| Analysis Dimension | Result |
|---|---|
| Training stability | PIRO yields the smoothest learning curves; baselines exhibit large variance or performance collapse |
| Sample efficiency | PIRO converges at a rate comparable to the fastest baseline while achieving higher final performance |
| State-only reward recovery | Recovered rewards in a $7\times7$ grid world closely match ground truth |
| Reward transfer | Rewards learned on LunarLander remain effective for training policies under added wind perturbations |
| Hyperparameter sensitivity | Robust within $x, y \in (1, 2)$ and $\bar{\epsilon}^{\text{target}} \in (0.1, 1)$ |

### Key Findings

- PIRO outperforms or matches SOTA on nearly all tasks, with particularly notable advantages on challenging tasks (Humanoid, AntMaze, AdroitHand).
- Training stability is the most prominent advantage — baselines such as ML-IRL frequently suffer performance collapse on complex tasks.
- Although per-step computational cost is slightly higher, stable convergence does not increase total computation.
- The only task where PIRO underperforms a baseline is Hopper-v4 (−173.7), suggesting that the proximal constraint may be overly conservative on simple tasks.

## Highlights & Insights

- The **inverse symmetry with TRPO** perspective is elegant: trust region in forward RL guarantees policy improvement ↔ trust region in inverse RL guarantees reward improvement.
- Unifying multiple non-adversarial IRL methods under a likelihood maximization framework constitutes a significant theoretical contribution.
- PIRO's implementation is concise: only a few additional reward gradient steps are needed on top of SAC, making it engineering-friendly.
- The reward transfer experiment demonstrates the advantage of explicit reward learning over implicit methods — the learned reward is decoupled from environment dynamics.

## Limitations & Future Work

- Theoretical guarantees assume exact policy optimization; in practice, finite-step SAC introduces a gap between theory and implementation.
- Reliance on on-policy sampling may limit scalability to tasks where environment interactions are costly.
- The theoretical constant $C$ is too large for direct use, necessitating adaptive $\mu$ to relax the constraint in practice.
- The framework is potentially extensible to RLHF settings, given the natural connection between reward model learning from human feedback and IRL.

## Related Work & Insights

- The instability of adversarial methods such as GAIL and AIRL is a longstanding challenge in IRL; PIRO offers a principled alternative with formal guarantees.
- PIRO is closely related to ML-IRL and can be viewed as ML-IRL augmented with a trust region constraint.
- The TRPO→PPO simplification pathway inspired the TRRO→PIRO design trajectory.
- The framework has potential applications to reward modeling in LLM alignment, as reward learning in RLHF is fundamentally an IRL problem.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First formal stability guarantee for IRL; the inverse-symmetric perspective on TRPO is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 9 tasks, 13 baselines, and comprehensive analysis of stability, efficiency, transfer, and sensitivity.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous theoretical derivations and a concise practical algorithm.
- Value: ⭐⭐⭐⭐⭐ — Significant contributions to both the theory and practice of IRL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Human-assisted Robotic Policy Refinement via Action Preference Optimization](human-assisted_robotic_policy_refinement_via_action_preference_optimization.md)
- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)
- [\[CVPR 2026\] General Process Reward Modeling for Robotic Reinforcement Learning](../../CVPR2026/robotics/general_process_reward_modeling_for_robotic_reinforcement_learning.md)
- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](../../ICLR2026/robotics/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)

</div>

<!-- RELATED:END -->
