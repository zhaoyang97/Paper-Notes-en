---
title: >-
  [Paper Note] Plasticity as the Mirror of Empowerment
description: >-
  [NeurIPS 2025][Plasticity] This paper proposes **Generalized Directed Information (GDI)** as an information-theoretic tool for measuring agent plasticity…
tags:
  - "NeurIPS 2025"
  - "Plasticity"
  - "Empowerment"
  - "Information Theory"
  - "Generalized Directed Information"
  - "Agent Design"
date: 2026-05-08
content_hash: 040ceecc68143a39
---

# Plasticity as the Mirror of Empowerment

**Conference**: NeurIPS 2025
**arXiv**: [2505.10361](https://arxiv.org/abs/2505.10361)  
**Code**: None  
**Area**: Other
**Keywords**: Plasticity, Empowerment, Information Theory, Generalized Directed Information, Agent Design

## TL;DR
This paper proposes **Generalized Directed Information (GDI)** as an information-theoretic tool for measuring agent plasticity, revealing that plasticity is the "mirror" of empowerment — both use the same measure but in opposite directions — and proves a strict tension bound between the two.

## Background & Motivation

**Two fundamental agent capabilities**: Any agent possesses two core capacities — the ability to be shaped by observations (plasticity) and the ability to influence future observations (empowerment). A system lacking either capacity can hardly be considered a genuine agent.

**Empowerment has a mature definition**: Klyubin et al. (2005) first defined empowerment as a measure of an agent's control over future observable states; it has since been widely applied in intrinsic motivation, safety alignment, and skill discovery.

**Plasticity lacks a unified formalization**: Although "plasticity" is discussed across neuroscience (synaptic plasticity), biology (environmental responsiveness), and machine learning (loss of plasticity), definitions remain fragmented across fields, with no universal mathematical definition comparable to that of empowerment.

**Limitations of directed information**: Massey's (1990) directed information requires two sequences of equal length starting from the initial time step, precluding flexible measurement of information flow over arbitrary time windows.

**Loss of plasticity in continual learning**: Work by Dohare et al. (2021, 2024) and Lyle et al. (2023) demonstrates that neural networks readily lose plasticity under continual learning, yet a unified theoretical framework for quantifying this phenomenon is lacking.

**Core goal**: To construct a general, agent-centric measure of plasticity that stands on equal theoretical footing with empowerment, and to reveal the intrinsic relationship between the two.

## Method

### Overall Architecture

The paper builds on a minimal-assumption agent–environment interaction model: agent $\lambda$ and environment $e$ share an interface $(A, O)$, exchanging actions and observations at discrete time steps. The core mechanism is to extend directed information to define both plasticity and empowerment simultaneously, thereby exposing their symmetric structure.

### Key Designs

**Module 1: Generalized Directed Information (GDI)**

Classical directed information $I(X_{1:n} \rightarrow Y_{1:n}) = \sum_i I(X_{1:i}; Y_i \mid Y_{1:i-1})$ applies only to equal-length sequences starting from the initial time step. GDI generalizes this to arbitrary intervals $[a:b]$ and $[c:d]$:

$$\mathbb{I}(X_{a:b} \rightarrow Y_{c:d}) = \sum_{i=\max(a,c)}^{d} \mathbb{I}(X_{a:\min(b,i)}; Y_i \mid X_{1:a-1}, Y_{1:i-1})$$

GDI strictly generalizes directed information (reducing to it when $a=c=1, b=d=n$) while satisfying temporal consistency (GDI of future sequences onto past is zero), interval additivity, and the **conservation law** (Theorem 3.5):

$$\mathbb{I}(X_{a:b}; Y_{c:d} \mid X_{1:a-1}, Y_{1:c-1}) = \mathbb{I}(X_{a:b} \rightarrow Y_{c:d}) + \mathbb{I}(Y_{c:d} \hookrightarrow X_{a:b})$$

**Module 2: Formal Definition of Plasticity**

Using GDI, the plasticity of any agent $\lambda$ over environment set $\mathcal{E}$ and time intervals $[a:b] \rightarrow [c:d]$ is defined as:

$$\mathfrak{P}_{a:b}^{c:d}(\lambda, \mathcal{E}) = \max_{e \in \mathcal{E}} \mathbb{I}(O_{a:b} \rightarrow A_{c:d})$$

Intuitively, the greater the influence of the observation sequence $O$ on the action sequence $A$, the more "plastic" the agent. This definition satisfies a set of desirable properties:

| Property | Description |
|----------|-------------|
| Non-negativity | $\mathfrak{P}(\lambda, \mathcal{E}) \geq 0$ |
| Zero plasticity iff | observations have no influence on actions |
| Deterministic agents can be plastic | deterministic policies can still be shaped by observations |
| Monotonicity in environment set | $\mathcal{E}_{\text{small}} \subseteq \mathcal{E}_{\text{big}} \Rightarrow \mathfrak{P}(\lambda, \mathcal{E}_{\text{small}}) \leq \mathfrak{P}(\lambda, \mathcal{E}_{\text{big}})$ |
| Zero plasticity examples | open-loop agents, constant agents, agents depending only on history length |

**Module 3: Mirror Relationship between Plasticity and Empowerment**

GDI likewise extends the definition of empowerment to $\mathfrak{E}(\Lambda, e) = \max_\lambda \mathbb{I}(A_{a:b} \rightarrow O_{c:d})$. Since agent and environment are mathematically symmetric (swapping $A$ and $O$ interchanges the two), Proposition 4.6 establishes:

$$\mathfrak{E}(\lambda) = \mathfrak{P}(e), \quad \mathfrak{P}(\lambda) = \mathfrak{E}(e)$$

That is: **an agent's empowerment equals the environment's plasticity, and an agent's plasticity equals the environment's empowerment**.

### Core Theoretical Result: Tension Theorem (Theorem 4.8)

For any agent–environment pair $(\lambda, e)$ and time intervals $[a:b]$, $[c:d]$, let $m = \min\{(b-a+1)\log|O|,\, (d-c+1)\log|A|\}$; then:

$$\mathfrak{E}_{a:b}^{c:d}(\lambda, e) + \mathfrak{P}_{c:d}^{a:b}(\lambda, e) \leq m$$

This upper bound is **tight**: there exist extreme cases in which one quantity attains $m$ while the other is zero. This implies that an agent cannot simultaneously maximize empowerment and plasticity over the same time window — increasing control over the environment necessarily reduces the degree to which the agent is shaped by it.

## Key Experimental Results

### Main Results: Plasticity and Empowerment of Q-learning on a Two-Armed Bernoulli Bandit

Experiments use a Monte Carlo estimator to estimate plasticity and empowerment over the interval $[1:3] \rightarrow [2:5]$.

**Experiment 1: Effect of $\varepsilon$-greedy exploration parameter on plasticity**

| $\varepsilon$ | Plasticity Trend | Explanation |
|---------------|-----------------|-------------|
| $\varepsilon = 0$ (pure greedy) | Highest | actions fully determined by observation-driven Q-values |
| $\varepsilon = 0.5$ | Moderate | half of actions random, half driven by observations |
| $\varepsilon = 1$ (pure random) | Zero | actions completely independent of observations |

**Experiment 2: Effect of initial Q-values (optimistic/pessimistic) on plasticity and empowerment**

| Initial Q-value | Plasticity | Empowerment | Sum |
|-----------------|------------|-------------|-----|
| $Q_0 = -1$ (pessimistic) | Higher | Lower | $< m$ |
| $Q_0 = 0$ (neutral) | Moderate | Moderate | $< m$ |
| $Q_0 = 1$ (optimistic) | Lower | Highest | $< m$ |

Key findings: (1) empowerment is generally higher than plasticity in this setting; (2) optimistic initialization increases empowerment (more exploration → more control); (3) the tension bound $m$ is verified across all experiments, with the sum never exceeding $m$.

### Thought Experiment: Corridor Environment

In a corridor of $n+1$ rooms, each containing a switch and a light, the agent's control over the lights increases from $0/n$ (leftmost room) to $n/n$ (rightmost room). The leftmost room (no control) maximizes plasticity; the rightmost room (full control) maximizes empowerment; intermediate rooms interpolate smoothly — providing an intuitive illustration of the Pareto frontier defined by the tension.

## Highlights & Insights

1. **Conceptual breakthrough**: Provides the first general, mathematically precise definition of plasticity that stands on equal footing with empowerment.
2. **Elegant mirror structure**: Plasticity and empowerment share the same measure but differ only in direction, revealing a deep symmetry in agent–environment interaction.
3. **Independent value of GDI**: Generalized Directed Information strictly extends Massey's directed information, preserving all its properties while supporting arbitrary time windows — a contribution of independent interest to information theory and causal inference.
4. **Practical implications of the tension theorem**: Quantitatively reveals the incompatibility of "adaptability vs. control," offering a new constraint perspective for agent design.
5. **Minimal assumptions**: The theory requires only a discrete interface and finite sets, with no assumptions of MDPs, the Markov property, or any specific learning algorithm.

## Limitations & Future Work

1. **Purely theoretical contribution**: Experiments are limited to a simple two-armed bandit; large-scale or realistic RL environment validation is absent.
2. **Zero plasticity under deterministic environments**: The definition yields zero plasticity for deterministic environments, which — though theoretically justified — may conflict with intuition.
3. **No treatment of goals or rewards**: The framework does not address goal-directed behavior, limiting direct guidance for practical RL algorithms.
4. **Computational complexity of GDI**: Monte Carlo estimation of GDI scales exponentially with the state space, making large-scale application infeasible.
5. **No connection to loss of plasticity**: Despite motivating the work through continual learning plasticity loss, the paper does not provide concrete connections to or solutions for that phenomenon.

## Related Work & Insights

| Direction | Representative Work | Connection to This Paper |
|-----------|--------------------|-----------------------------|
| Empowerment | Klyubin et al. (2005), Capdepuy (2011) | Unifies empowerment and plasticity under the GDI framework |
| Loss of plasticity | Dohare et al. (2024), Lyle et al. (2023) | Provides a universal definition of plasticity that could serve as a theoretical foundation for this line of work |
| Stability–plasticity dilemma | Carpenter & Grossberg (1988) | The tension theorem offers a new perspective for formalizing this dilemma |
| Directed information | Massey (1990), Massey & Massey (2005) | GDI strictly generalizes directed information and extends the conservation law |
| Universal agents | Hutter (2004), Abel et al. (2023) | Adopts the same minimal-assumption agent–environment interaction framework |
| Intrinsic motivation | Mohamed & Rezende (2015) | Plasticity can serve as an additional intrinsic drive signal alongside empowerment |

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First formalization of plasticity and revelation of its mirror relationship with empowerment; outstanding conceptual contribution.
- **Experimental Thoroughness**: ⭐⭐ — Only a simple bandit experiment; large-scale empirical validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Argumentation progresses in well-structured layers, from axiomatic requirements through definitions, theorems, and experiments; exceptionally clear.
- **Value**: ⭐⭐⭐⭐ — Theoretically elegant with significant long-term potential, though practical applicability remains distant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Meta-learning three-factor plasticity rules for structured credit assignment with sparse feedback](meta-learning_three-factor_plasticity_rules_for_structured_credit_assignment_wit.md)
- [\[ICLR 2026\] FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff](../../ICLR2026/others/fire_frobenius_isometry_reinitialization.md)
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [\[NeurIPS 2025\] Learning Dynamics of RNNs in Closed-Loop Environments](learning_dynamics_of_rnns_in_closed-loop_environments.md)
- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)

</div>

<!-- RELATED:END -->
