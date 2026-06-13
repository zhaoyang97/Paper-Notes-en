---
title: >-
  [Paper Note] Computational Hardness of Reinforcement Learning with Partial $q^\pi$-Realizability
description: >-
  [NeurIPS 2025][Reinforcement Learning][computational complexity] This paper introduces the notion of "partial $q^\pi$-realizability" and proves that learning a near-optimal policy under this setting is NP-hard when using…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "computational complexity"
  - "linear function approximation"
  - "realizability"
  - "NP-hard"
date: 2026-05-08
content_hash: eb374d2429ff8e14
---

# Computational Hardness of Reinforcement Learning with Partial $q^\pi$-Realizability

**Conference**: NeurIPS 2025
**arXiv**: [2510.21888](https://arxiv.org/abs/2510.21888)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning
**Keywords**: reinforcement learning, computational complexity, linear function approximation, realizability, NP-hard

## TL;DR

This paper introduces the notion of "partial $q^\pi$-realizability" and proves that learning a near-optimal policy under this setting is NP-hard when using a greedy policy class, and requires exponential time under the rETH assumption when using a softmax policy class. These results bridge the theoretical gap between $q^*$-realizability and $q^\pi$-realizability.

## Background & Motivation

In reinforcement learning with linear function approximation, two extreme realizability assumptions have been extensively studied:

**$q^*$-realizability** (weaker assumption): Only the optimal value function is assumed to be linearly realizable. Known to admit quasi-polynomial or even exponential computational complexity lower bounds.

**$q^\pi$-realizability** (stronger assumption): The value functions of all policies are assumed to be linearly realizable. Efficient computation is achievable under generative model access.

This gives rise to two core questions:
- **Q1**: If the policy class $\Pi$ is restricted under $q^\pi$-realizability, can positive results still be obtained?
- **Q2**: By expanding the policy class beyond $\{π^*\}$ in the $q^*$-realizability setting to $\{π^*\} \subsetneq \Pi$, can computational hardness be circumvented?

## Method

### Overall Architecture

**Partial $q^\pi$-Realizability** (Definition 3.1): Given a policy class $\Pi \subset \mathcal{A}^{\mathcal{S}}$ and feature mapping $\phi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^d$, an MDP is partially $q^\pi$-realizable under $\Pi$ if for all $\pi \in \Pi$, there exists $\theta_h \in \mathbb{R}^d$ such that:

$$q_h^\pi(s_h, a_h) = \langle \phi(s_h, a_h), \theta_h \rangle \quad \forall (s_h, a_h) \in \mathcal{S}_h \times \mathcal{A}$$

**Learning Objective** ($\epsilon$-optimality): Find a policy $\pi$ such that

$$\max_{\hat{\pi} \in \Pi} v^{\hat{\pi}}(s_1) - v^{\pi}(s_1) \leq \epsilon$$

### Key Designs

**Greedy Policy Class $\Pi^g$** (Definition 3.3):

Defined via parameterized feature mapping $\phi': \mathcal{S} \times \mathcal{A} \to \mathbb{R}^{d'}$ and weight $\theta' \in \mathbb{R}^{d'}$:

$$\pi_{\theta'}^g(s_h) := \arg\max_{a \in \mathcal{A}} \langle \phi'(s_h, a), \theta' \rangle$$

The policy class is $\Pi^g = \{\pi_{\theta'}^g \mid \theta' \in \mathbb{R}^{d'}\}$.

**Softmax Policy Class $\Pi^{sm}$** (Definition 3.4):

$$\pi_{\theta'}(a | s_h) = \frac{e^{\phi'(s_h, a)^\top \theta'}}{\sum_{i=1}^{\kappa} e^{\phi'(s_h, a_i)^\top \theta'}}$$

The policy class is $\Pi^{sm} = \{\pi_{\theta'} \mid \theta' \in \mathbb{R}^{d'}\}$.

**Core Reduction Mechanism** (two-step approach):

**Step 1: Polynomial MDP Construction**. Given a $\delta$-Max-3SAT instance $\varphi$ with $n$ variables and $k$ clauses, construct MDP $M_\varphi$:
- **States**: $n$-tuples in $\{-1, 0, 1\}^n$; initial state is all $-1$; binary tree structure with $2^{n+1}-1$ states in total
- **Actions**: $\mathcal{A} = \{0, 1\}$ (corresponding to variable assignments False/True)
- **Transitions**: Deterministic; at step $h$, the $h$-th component of the state is set to the chosen action value
- **Rewards**: Granted only at terminal states, defined as the fraction of satisfied clauses $R(s_H) = |\mathcal{C}_{true}(s_H)| / |\mathcal{C}|$

**Step 2: Algorithmic Equivalence**. It is shown that if $\mathcal{A}_{rl}$ returns an $\epsilon$-optimal policy on $M_\varphi$, then one can construct $\mathcal{A}_{sat}$ to solve $\delta$-Max-3SAT.

### Loss & Training

This paper is a purely theoretical work and does not involve training loss functions. The core contribution consists of impossibility results established via computational complexity reductions.

## Key Experimental Results

### Main Results

This paper makes theoretical contributions. The main theorems are:

| Theorem | Policy Class | Complexity Assumption | Result |
|------|--------|-----------|------|
| Theorem 3.1 | Greedy $\Pi^g$ | P ≠ NP | gLinear-$\kappa$-RL is NP-hard ($\epsilon \leq 0.05$) |
| Theorem 3.2 | Softmax $\Pi^{sm}$ | rETH | sLinear-$\kappa$-RL requires $\exp(o(d^{1/3}/\text{polylog}(d^{1/3})))$ time |

**Positioning in the Complexity Spectrum**:

| Realizability Setting | Assumption Strength | Computational Complexity |
|-------------|---------|-----------|
| $q^*$-realizability | Weakest | Exponential lower bound (known) |
| **Partial $q^\pi$-realizability** | **Intermediate** | **NP-hard / exponential lower bound (this paper)** |
| $q^\pi$-realizability | Strongest | Polynomial-time solvable (generative model) |

### Ablation Study

**MDP Construction Parameter Analysis**:

- Construction of the PSP feature vector $\phi' \in \mathbb{R}^n$: the $h$-th component corresponds to variable $x_h$; action True maps to $+1$, False to $-1$
- Construction complexity of the realizability feature vector $\phi \in \mathbb{R}^d$: $O(n^3)$ (dependent on the total number of clauses)
- The entire reduction is completed in polynomial time

**$\delta$-Max-3SAT Instance Analysis** (Example 4.1): Using $\varphi: (x_1 \vee \bar{x}_2 \vee x_3) \wedge (\bar{x}_1 \vee x_2 \vee \bar{x}_3)$ as an example, a 15-state MDP is constructed in which all terminal states receive reward 1 except $(0,1,0)$ and $(1,0,1)$, which receive reward $1/2$.

### Key Findings

1. **Persistent hardness**: Even when the policy class $\Pi$ is expanded to include infinitely many linearly realizable policies, solving under partial $q^\pi$-realizability remains computationally intractable.
2. **Subtle asymmetry**: $\Pi^g \subset \Pi^{sm}$ holds with high probability, yet hardness for the greedy policy class requires only P ≠ NP, whereas the softmax case requires the stronger rETH assumption.
3. **Partial negative answers to Q1 and Q2**: Restricting the policy class does not guarantee computational efficiency (Q1); expanding the policy class does not break hardness (Q2).
4. **Connection to Agnostic RL**: The hardness results also imply inherent computational difficulty for agnostic RL under linear function approximation.

## Highlights & Insights

- **Precise theoretical positioning**: Partial $q^\pi$-realizability elegantly bridges the gap between the two extreme realizability assumptions, providing a more practically motivated modeling framework.
- **Elegant reduction design**: Variable assignments in Max-3SAT are naturally mapped to action sequences in the MDP; the binary-tree MDP structure is clean and transparent.
- **Generalization conditions** (Remark 4.2): Sufficient conditions for extending the hardness results to arbitrary policy classes are given as a triple: (i) the policy is polynomially parameterizable, (ii) the feature vector is polynomially constructible, and (iii) the reduction preserves problem-solving equivalence.
- **Practical implication**: The hardness results should not be viewed as purely negative — they indicate that additional structural assumptions (e.g., constraints on policies or features) are necessary for efficient algorithms.

## Limitations & Future Work

1. The policy parameterization feature $\phi'$ and the realizability feature $\phi$ are distinct vectors; the unified case ($\phi' = \phi$) remains unresolved.
2. The hardness results are based on worst-case analysis; practical RL problems may possess structure that makes them more tractable.
3. Only deterministic transition MDPs are considered; conclusions under stochastic transitions may differ.
4. No positive algorithmic results or sufficient conditions for efficient solvability are provided.
5. The generative model represents the strongest interaction protocol; results under weaker interaction models remain unknown.

## Related Work & Insights

- **Relation to KLL+ (2023)**: Extends the exponential hardness results under $q^*$/$v^*$-realizability into the intermediate assumption regime.
- **Contrast with YHAY+ (2022)**: Full $q^\pi$-realizability admits efficient computation via a generative model, whereas partial realizability does not — demonstrating a dramatic jump in computational complexity resulting from a minor relaxation of assumptions.
- **Connection to Agnostic RL**: Partial realizability is equivalent to agnostic learning over a given policy class; the results of this paper imply inherent difficulty for agnostic RL under linear approximation.
- **Directions for future work**: Investigate the complexity of the unified feature case ($\phi' = \phi$); explore additional structural conditions under which partial realizability becomes tractable.

## Rating

- ⭐ Novelty: 4/5 — The introduction of partial realizability and its precise complexity characterization fills an important theoretical gap.
- ⭐ Value: 2/5 — The negative theoretical results provide guidance for algorithm design but have no direct applications.
- ⭐ Experimental Thoroughness: 2/5 — Purely theoretical work; examples and proofs substitute for experiments.
- ⭐ Writing Quality: 4/5 — Definitions and theorems are rigorously stated and the proof structure is clear, though the notation is heavy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Partial Action Replacement: Tackling Distribution Shift in Offline MARL](../../AAAI2026/reinforcement_learning/partial_action_replacement_tackling_distribution_shift_in_offline_marl.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] Learning to Clean: Reinforcement Learning for Noisy Label Correction](learning_to_clean_reinforcement_learning_for_noisy_label_correction.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](reinforcement_learning_teachers_of_test_time_scaling.md)

</div>

<!-- RELATED:END -->
