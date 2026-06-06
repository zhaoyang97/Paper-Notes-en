---
title: >-
  [Paper Note] Multi-Environment POMDPs: Discrete Model Uncertainty Under Partial Observability
description: >-
  [NeurIPS 2025][LLM Alignment][POMDP] This paper systematically studies Multi-Environment POMDPs (ME-POMDPs)—a class of POMDP ensembles sharing state, action…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "POMDP"
  - "robust planning"
  - "model uncertainty"
  - "multi-environment"
  - "adversarial belief"
  - "value iteration"
  - "linear programming"
date: 2026-05-08
content_hash: 1838b1e84a21b5f4
---

# Multi-Environment POMDPs: Discrete Model Uncertainty Under Partial Observability

**Conference**: NeurIPS 2025  
**arXiv**: [2510.23744](https://arxiv.org/abs/2510.23744)  
**Code**: [GitHub]([https://github.com/](https://github.com/) (see paper reference [6]))  
**Area**: LLM Alignment  
**Keywords**: POMDP, robust planning, model uncertainty, multi-environment, adversarial belief, value iteration, linear programming

## TL;DR

This paper systematically studies Multi-Environment POMDPs (ME-POMDPs)—a class of POMDP ensembles sharing state, action, and observation spaces but with arbitrarily different transition, observation, and reward functions—with the goal of finding a robust policy that maximizes reward under the worst-case environment. By introducing the Adversarial Belief POMDP (AB-POMDP) as a unified model and establishing its equivalence to one-sided partially observable stochastic games (POSGs), the paper proposes both exact (value iteration + LP) and approximate (AB-HSVI) algorithms.

## Background & Motivation

Partially Observable Markov Decision Processes (POMDPs) are widely used for sequential decision-making in robotics, healthcare, and related domains. However, existing algorithms assume exact knowledge of transition and observation dynamics, which rarely holds in practice:

- **Expert disagreement**: Multiple domain experts may hold different views on model parameters (e.g., in endangered bird conservation, experts disagree on state transition probabilities), giving rise to discrete, multi-environment model uncertainty.

- **Limitations of prior work (Robust POMDP)**: Robust POMDPs under continuous uncertainty assume convexity, rectangularity, and dynamic uncertainty. The discrete uncertainty in ME-POMDPs is inherently non-rectangular, making direct application of RPOMDP methods overly conservative.

- **Fully observable case already studied**: Multi-Environment MDPs (MEMDPs) have been extensively studied, but algorithms for the partially observable setting are absent.

- **Static vs. dynamic uncertainty**: Unlike dynamic uncertainty, where the model may change at each step, this paper considers static uncertainty, where the model remains fixed throughout the entire time horizon.

## Method

### Overall Architecture

**ME-POMDP** is defined as $\mathcal{M} = (S, A, Z, n, \{T_i\}, \{O_i\}, \{R_i\}, \{b_i\}, \gamma, H)$, comprising $n$ POMDP environments sharing $S, A, Z$ but with distinct transition, observation, and reward functions. The objective is:

$$V_{\mathcal{M}}^* = \max_{\pi} \min_{i \in [n]} V_{\mathcal{M}_i}^{\pi}$$

**AB-POMDP** (Adversarial Belief POMDP): Extends the initial belief from a single point to a set $B \subseteq \Delta(S)$:

$$V_{\mathsf{M}}^* = \max_{\pi} \min_{b \in B} V_{\mathsf{M}_b}^{\pi}$$

The key theoretical contribution is establishing the equivalence chain ME-POMDP → AB-POMDP → one-sided POSG.

### Key Designs

**Theorem 1 (AB-POMDP ↔ POSG)**: When the belief set $B = \Delta(Q)$, the AB-POMDP is equivalent to a zero-sum one-sided POSG, where the partially observable agent corresponds to the decision-maker and the fully observable nature implements adversarial behavior by selecting the initial state.

**Theorem 2 (ME-POMDP → AB-POMDP)**: Any ME-POMDP can be mapped to an equivalent AB-POMDP with state space $S \times [n] \times \{1, 2\}$, where nature's belief selection is equivalent to environment selection.

**Theorems 3 & 4 (Restricted Model Reductions)**:
- Any ME-POMDP can be reduced to a **PO-MEMDP** (differing only in observation functions) with polynomial overhead.
- A PO-MEMDP can be further reduced to an **MO-POMDP** (identical transition functions), at the cost of an exponential state space expansion to $|S|^n$.

**LP-based Optimal Policy Computation (Theorem 5)**: Given an $\alpha$-vector set $\Gamma$, the value function minimum is solved via a dual linear program.

Primal LP: $\min_{b, v} v$, subject to $\forall \alpha \in \Gamma: \sum_s \alpha(s) b(s) \leq v$, $b \in \Delta(Q)$

Dual LP: $\max_{y, v} v$, subject to $\forall s \in Q: \sum_\alpha \alpha(s) y(\alpha) \geq v$, $y \in \Delta(\Gamma)$

The dual solution $y$ defines a mixed strategy over $\alpha$-vectors and constitutes the optimal policy for the AB-POMDP.

### Loss & Training

**AB-HSVI Algorithm**: An extension of HSVI (Heuristic Search Value Iteration):
1. Initialize upper bound (Fast Informed Bound) and lower bound (fixed-policy $\alpha$-vectors).
2. Compute the worst-case belief $b$ under the current lower bound via LP (3).
3. Execute HSVI depth-first search from $b$, updating both bounds.
4. Recompute the worst-case belief; terminate if the gap between bounds is $< \epsilon$.

Each iteration effectively restarts HSVI from a new worst-case starting point, progressively tightening the approximation quality.

## Key Experimental Results

### Main Results

**Bird Problem (comparison across model types)**:

| Instance | \|S\| | n | PO-MEMDP Value / Time | MO-POMDP Value / Time | ME-POMDP Value / Time |
|---------|-----|---|------------------|------------------|------------------|
| BP3,3,3 | 3 | 3 | 68.26 / 58.5s | 70.44 / 85.0s | 69.62 / 2039.3s |
| BP3,3,4 | 3 | 4 | 44.44 / timeout (gap=4.33) | 54.85 / 2976.3s | 44.79 / timeout (gap=6.02) |
| BP3,3,5 | 3 | 5 | 74.58 / 3104.3s | 80.01 / 21.1s | 74.59 / timeout (gap=0.61) |

**RockSample Problem (varying scales and configurations)**:

| Instance | \|S\| | n | Rock Near: Value / Time | Rock Far: Value / Time |
|---------|-----|---|------------------|------------------|
| RS2,1,2 | 9 | 2 | 16.53 / 11.7s | 16.53 / 11.7s |
| RS3,1,2 | 19 | 2 | 16.14 / 52.7s | 14.68 / 170.0s |
| RS4,1,2 | 33 | 2 | 15.48 / 130.8s | 13.02 / 1589.0s |
| RS5,1,2 | 51 | 2 | 15.40 / 331.4s | 11.03 / timeout (gap=1.46) |
| RS7,1,2 | 99 | 2 | 14.54 / 1280.7s | — |

### Ablation Study

**Model type comparison (Q3)**: MO-POMDP (uncertain observations) generally converges faster and achieves higher values than PO-MEMDP (uncertain transitions), suggesting that observation uncertainty is easier to handle than transition uncertainty.

**AB-POMDP vs. ME-POMDP formulation**: Across all RockSample instances, the AB-POMDP formulation nearly always converges faster than the direct ME-POMDP formulation, with the gap widening as the number of environments increases.

**Robust vs. non-robust (Q2)**: Robust values are close to the optimal values of individual environments yet substantially better than the worst-case performance of policies that incorrectly assume a fixed environment, demonstrating the necessity of robust planning.

### Key Findings

1. Convergence time scales primarily with the number of environments $n$; RockSample instances typically converge faster than Bird problems due to the presence of terminal states.
2. Environment structure strongly influences difficulty: placing rocks far from the initial position can increase solving time by more than 10×.
3. MO-POMDP is often the most efficient formulation despite the $|S|^n$ state space expansion.
4. Generating challenging benchmarks is non-trivial: only 35 out of 100 randomly generated 3-state PO-MEMDPs are non-trivial (requiring > 30s to solve).

## Highlights & Insights

1. **Theoretically complete framework**: The paper establishes a full equivalence chain ME-POMDP ↔ AB-POMDP ↔ POSG, along with reductions between PO-MEMDP and MO-POMDP (Figure 1).
2. **Elegant LP-based policy construction**: The derivation of mixed strategies over $\alpha$-vectors directly from the dual LP is concise and elegant.
3. **Avoiding over-conservatism**: Compared to applying RPOMDP's rectangularization, the ME-POMDP approach preserves the exact structure of discrete uncertainty.
4. **Model selection guidance**: Experiments reveal distinct performance characteristics of different formulations (PO-MEMDP vs. MO-POMDP), providing practical guidance for model choice.

## Limitations & Future Work

1. **Limited scalability**: Current methods struggle to converge within reasonable time when the number of states exceeds 100 or the number of environments exceeds 5.
2. **Lack of benchmarks**: No standard benchmarks exist for ME-POMDPs; the paper's self-constructed benchmarks offer limited coverage.
3. **State space explosion in MO-POMDP reduction**: The exponential growth to $|S|^n$ restricts applicability to large-scale problems.
4. **No comparison with reinforcement learning**: Only planning methods are considered; the potential of RL approaches for ME-POMDPs remains unexplored.
5. **Absence of real-world applications**: Bird and RockSample serve only as theoretical benchmarks; validation on real-world tasks is lacking.

## Related Work & Insights

- **MEMDP (Raskin et al.)**: The fully observable multi-environment model; this paper extends it to the partially observable setting.
- **Robust POMDP**: Continuous uncertainty combined with the rectangularity assumption renders it overly conservative for ME-POMDPs.
- **HSVI (Smith & Simmons)**: The classical point-based value iteration method; AB-HSVI extends it via an LP-based restart strategy.
- **Inspiration**: The ME-POMDP framework is directly applicable to medical decision-making under expert disagreement and robust robot planning across multiple simulators.
- **Connection to RLHF/alignment**: At a conceptual level, robust optimization over multiple "environments" (human preference models) bears similarity to robust reward modeling in alignment.

## Rating

- ⭐⭐⭐⭐⭐ **Theoretical Contribution**: The equivalence relations, reduction theorems, and LP-based policy construction together form a highly complete theoretical framework.
- ⭐⭐⭐ **Experimental Thoroughness**: Only small-scale benchmarks (states < 100) are evaluated; significant gap remains to practical applications.
- ⭐⭐⭐⭐ **Writing Quality**: Definitions are clear, theorems are rigorously stated, and figures are intuitive.
- ⭐⭐⭐ **Value**: Scalability remains the primary obstacle; large-scale real-world applicability is limited in the near term.

**Overall**: ⭐⭐⭐⭐ (3.5/5) — A theoretically rigorous contribution to the planning literature that establishes a complete framework for ME-POMDPs. The main weaknesses are limited scalability and the absence of real-world validation. The work is an important reference for researchers in robust sequential decision-making theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](../../ICLR2026/llm_alignment/jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[NeurIPS 2025\] LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits](laser_learning_to_adaptively_select_reward_models_with_multi-armed_bandits.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[NeurIPS 2025\] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)

</div>

<!-- RELATED:END -->
