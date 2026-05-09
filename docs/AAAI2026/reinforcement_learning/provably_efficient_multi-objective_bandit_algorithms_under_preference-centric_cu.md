---
title: >-
  [Paper Note] Provably Efficient Multi-Objective Bandit Algorithms under Preference-Centric Customization
description: >-
  [AAAI 2026][Reinforcement Learning][multi-objective bandits] This paper presents the first theoretical study of preference-aware customization in multi-objective multi-armed bandits (MO-MAB) with explicit user preferences. It proposes the PAMO-MAB framework and designs PRUCB-UP and PRUCB-HP algorithms for the "unknown preference" and "hidden preference" settings, respectively. Through a two-component architecture combining preference estimation and preference-aware optimization, both algorithms achieve near-optimal regret bounds. The paper also proves that preference-free algorithms inevitably incur $\Omega(T)$ linear regret when the Pareto front contains conflicting arms.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - multi-objective bandits
  - user preference
  - Pareto optimality
  - UCB
  - regret bound
  - preference estimation
date: 2026-05-08
content_hash: 7685130fc6abcb8f
---

# Provably Efficient Multi-Objective Bandit Algorithms under Preference-Centric Customization

**Conference**: AAAI 2026
**arXiv**: [2502.13457](https://arxiv.org/abs/2502.13457)
**Code**: None
**Area**: Online Learning / Multi-Armed Bandits
**Keywords**: multi-objective bandits, user preference, Pareto optimality, UCB, regret bound, preference estimation

## TL;DR

This paper presents the first theoretical study of preference-aware customization in multi-objective multi-armed bandits (MO-MAB) with explicit user preferences. It proposes the PAMO-MAB framework and designs PRUCB-UP and PRUCB-HP algorithms for the "unknown preference" and "hidden preference" settings, respectively. Through a two-component architecture combining preference estimation and preference-aware optimization, both algorithms achieve near-optimal regret bounds. The paper also proves that preference-free algorithms inevitably incur $\Omega(T)$ linear regret when the Pareto front contains conflicting arms.

## Background & Motivation

### Core Problem

Classical MO-MAB formulations target Pareto optimality and employ a global policy to select arms for all users. However, different users hold distinct preferences over objective dimensions:

- **Illustrative scenario**: In restaurant recommendation, User 1 prioritizes taste while User 2 prioritizes price; the same restaurant on the Pareto front may satisfy one user but not the other.
- **Root Cause**: When the Pareto front contains two or more conflicting arms ($|\mathcal{O}^*| \geq 2$), any algorithm that ignores user preferences will necessarily incur linear regret for some users.

### Limitations of Prior Work

1. **Pareto front estimation methods** (Pareto-UCB, Pareto-TS): Estimate the full front and select arms uniformly, without user-level customization.
2. **Scalarization methods** (S-UCB, GGI): Compress multi-dimensional rewards into a scalar via a fixed scalarization function that is independent of individual users.
3. **Lexicographic MO-MAB**: Imposes a strict hierarchy over objectives and cannot model continuous preference trade-offs across users.

### Paper Goals

- Formalize the Preference-Aware MO-MAB (PAMO-MAB) problem, introducing the first explicit user preference framework.
- Prove an $\Omega(T)$ lower bound for preference-free algorithms (Proposition 1).
- Propose near-optimal algorithms for both the hidden-preference and unknown-preference settings.

## Method

### Problem Formulation

Consider $N$ users, $K$ arms, and $D$ objective dimensions. At each round $t$, user $n$ has a stochastic preference $\boldsymbol{c}_t^n \in \mathbb{R}^D$ with mean $\bar{\boldsymbol{c}}^n$. Upon selecting arm $a_t^n$, the agent observes a $D$-dimensional reward $\boldsymbol{r}_{a_t^n, t}$. The overall reward is defined as the inner product $g_{a_t^n, t} = {\boldsymbol{c}_t^n}^\top \boldsymbol{r}_{a_t^n, t}$, and the objective is to minimize cumulative regret:

$$R(T) = \sum_{t=1}^T \sum_{n=1}^N \bar{\boldsymbol{c}}^{n\top}(\boldsymbol{\mu}_{a_t^{n*}} - \boldsymbol{\mu}_{a_t^n})$$

### Overall Architecture: Preference Estimation + Preference-Aware Optimization

All algorithms share a unified two-component framework:

1. **Preference Estimation**: Infer user preference vectors from bandit feedback.
2. **Preference-Aware Optimization**: Select arms based on preference estimates to align decisions with user intent.

### Setting 1: Hidden Preference (PRUCB-HP)

Users provide per-objective ratings and an overall rating; preferences must be inferred from their relationship. Two unique challenges arise:

#### Challenge 1: Stochastic Mapping Problem

The overall reward satisfies $g_t = (\bar{\boldsymbol{c}} + \boldsymbol{\zeta}_t)^\top \boldsymbol{r}_t$, where the residual noise $\boldsymbol{\zeta}_t^\top \boldsymbol{r}_t$ depends on the input (larger rewards yield larger noise), invalidating standard regression models.

#### Challenge 2: Local vs. Global Exploration Dilemma

- **Local exploration**: Select high-reward arms to identify better arms.
- **Global exploration**: Select diverse arms to learn preferences.
- These two objectives conflict: high-reward arms are actually detrimental to preference estimation due to larger noise.

#### Key Design I: WLS Preference Estimator

Weighted Least Squares (WLS) is used for preference learning, with weights set to the reciprocal of the squared $\ell_2$ norm of the reward:

$$w_t^n = \frac{\omega}{\|\boldsymbol{r}_{a_t^n, t}\|_2^2}$$

- High-reward samples receive small weights → suppresses large residual noise.
- Low-reward samples receive large weights → amplifies their contribution to estimation.
- **Core Lemma**: After the re-weighting transformation, the residual noise becomes input-independent $R'$-sub-Gaussian with $R' = \sqrt{\omega}R$, restoring standard regression conditions.

#### Key Design II: Dual Exploration Strategy

The arm selection policy combines two bonus terms:

$$a_t = \arg\max_{i \in \mathcal{A}_t^n} (\hat{\boldsymbol{c}}_t^n)^\top \hat{\boldsymbol{r}}_{i,t} + B_{i,t}^{n,r} + B_{i,t}^{n,c}$$

- **Reward bonus** $B^{n,r}$: Standard UCB bonus based on Hoeffding's inequality, encouraging local exploration.
- **Preference bonus** $B^{n,c}$: Bonus based on pseudo information gain, $\beta_t \|\hat{\boldsymbol{r}}_{i,t} + \rho_{i,t}^\alpha \boldsymbol{e}\|_{(\boldsymbol{V}_{t-1}^n)^{-1}}$, encouraging global exploration.

### Setting 2: Unknown Preference (PRUCB-UP)

Users explicitly provide preference feedback after arm selection. This setting is considerably simpler than the hidden preference case:

- **Preference estimation**: Directly uses the empirical mean $\hat{\boldsymbol{c}}_{t+1}^n = ((t-1)\hat{\boldsymbol{c}}_t^n + \boldsymbol{c}_t^n) / t$.
- **Optimization strategy**: Only the reward bonus is needed; no preference bonus is required: $a_t^n = \arg\max_{i} (\hat{\boldsymbol{c}}_t^n)^\top (\hat{\boldsymbol{r}}_{i,t} + \rho_{i,t}^\alpha \boldsymbol{e})$.

### Theoretical Analysis

#### Lower Bound (Proposition 1)

When $|\mathcal{O}^*| \geq 2$, any preference-free algorithm incurs $R(T) = \Omega(T)$ for some subset of users. The proof argues that any preference-free algorithm achieving sublinear regret under preference $\boldsymbol{c}_1$ must use the same policy under the opposing preference $\boldsymbol{c}_2$; since the optimal arm changes but the policy does not, linear regret is unavoidable.

#### Upper Bound for PRUCB-HP (Theorem 1)

In the hidden preference setting, for $t \geq M$ (where $M$ is a constant independent of $T$), the asymptotic regret satisfies:

$$R(T) = \tilde{O}(D\sqrt{T})$$

The key technical challenge is bounding the cumulative preference bonus $\sum_t B_{i,t}^c$, which requires handling the inconsistency between the Gram matrix $\boldsymbol{V}_t$ constructed from observed rewards and the estimated rewards used in the bonus. This is resolved by transitioning to the expected Gram matrix $\mathbb{E}[\boldsymbol{V}_t]$.

#### Upper Bound for PRUCB-UP (Theorem 2)

In the unknown preference setting:

$$R(T) = O(KN\delta \log T + KN\delta D^2)$$

The regret attributable to preference estimation error constitutes only a constant term in $D$ and $\delta$, indicating that preference learning has a limited impact on total regret.

## Key Experimental Results

### Main Results: Regret Comparison Across Preference Settings

| Algorithm | Preference Setting | Regret Trend | Sublinear? | Key Feature |
|---|---|---|---|---|
| **PRUCB-HP** | Hidden preference | Sublinear growth | ✓ | WLS preference estimation + dual exploration |
| **PRUCB-UP** | Unknown preference | Sublinear growth | ✓ | Empirical mean preference estimation |
| PRUCB-UP (GT) | Known preference | Sublinear growth | ✓ | Ground-truth preference upper bound |
| S-UCB | Both settings | Linear growth | ✗ | Fixed equal-weight scalarization |
| S-MOSS | Both settings | Linear growth | ✗ | Fixed equal-weight + MOSS policy |
| Pareto-UCB | Both settings | Linear growth | ✗ | Estimates Pareto front |
| Pareto-TS | Both settings | Linear growth | ✗ | Thompson sampling |
| OFUL | Hidden preference | Linear growth | ✗ | Linear bandit baseline |
| UCB / MOSS | Hidden preference | Linear growth | ✗ | Uses only scalar feedback |

### Ablation Study: WLS Preference Estimator vs. Standard Linear Regression

| Sample Size | WLS Estimator Error | Standard LR Error | WLS Advantage |
|---|---|---|---|
| 20 | Lower | Higher | Advantage visible even with few samples |
| 50 | Clearly below LR | Moderate bias | Gap widens |
| 80 | Approaches true value [0.5, 0.5] | Bias remains | WLS converges faster |
| 200 | Nearly unbiased | Bias persists | WLS consistently superior |

> Experimental setup: 2D toy instance; Arm-1 mean [0.2, 0.2] (dominated), Arm-2 mean [0.8, 0.8] (Pareto-optimal). True preference mean [0.5, 0.5], variance 0.05. $T=5000$, averaged over 10 repetitions.

## Highlights & Insights

1. **Significant theoretical contribution**: The paper provides the first proof that preference-free algorithms necessarily fail ($\Omega(T)$ lower bound) when the Pareto front contains conflicting arms, establishing a rigorous theoretical motivation for preference-aware MO-MAB customization.
2. **Counter-intuitive finding**: Pareto-optimal (high-reward) arms are actually detrimental to preference learning — larger rewards amplify the noise introduced by stochastic preferences, overturning the intuition that "better arms carry more information."
3. **Elegant WLS weight design**: Setting $w_t = \omega / \|\boldsymbol{r}_t\|_2^2$ eliminates the input dependence of residual noise, transforming a non-standard regression problem into a canonical sub-Gaussian noise model.
4. **Novel dual exploration mechanism**: The pseudo information gain bonus can estimate the contribution of selecting a given arm to preference learning before observing rewards, effectively balancing local and global exploration.
5. **Strong framework unification**: PRUCB-UP can be viewed as a special case of PRUCB-HP with zero preference noise; both algorithms share the same algorithmic skeleton.

## Limitations & Future Work

1. **Synthetic data only**: Experiments are conducted exclusively on synthetic data, without validation on real recommendation or rating systems.
2. **Linear preference assumption**: The overall reward is defined as an inner product of preference and reward vectors (linear model), which cannot capture nonlinear user satisfaction functions.
3. **Preference independence assumption**: Assumption 3 requires preferences to be independent of rewards, which may not hold in certain scenarios (e.g., price-sensitive users may shift preferences in response to high-priced items).
4. **Preference stationarity**: Each user's preference mean $\bar{\boldsymbol{c}}^n$ is assumed fixed; preference drift is not considered.
5. **Scalability**: The WLS estimator requires maintaining a $D \times D$ Gram matrix and its inverse, incurring high computational cost in high-dimensional objective spaces.
6. **User rotation protocol**: The block-rotation protocol used in experiments is somewhat specific and may not generalize to more realistic recommendation scenarios.

## Related Work & Insights

- **Pareto front estimation MO-MAB**: Pareto-UCB (Drugan & Nowe 2013), Knowledge Gradient (Yahyaa et al. 2014), Thompson Sampling (Yahyaa & Manderick 2015), multi-objective contextual bandits (Turgay et al. 2018; Lu et al. 2019).
- **Scalarization MO-MAB**: GGI optimization (Busa-Fekete et al. 2017), multi-objective optimization for music platforms (Mehrotra et al. 2020), Pareto regret analysis (Xu & Klabjan 2023).
- **Lexicographic preference MO-MAB**: Hüyük & Tekin 2021 (introduction of lexicographic ordering), Cheng et al. 2024 (hybrid Pareto-lexicographic approach).
- **Linear bandit foundational techniques**: OFUL (Abbasi-Yadkori et al. 2011), MOSS (Audibert & Bubeck 2009), variance-dependent bandits (Zhou et al. 2021).

## Rating

| Dimension | Score (1–5) | Notes |
|---|---|---|
| Novelty | 4.5 | First theoretical formalization of user-preference-aware MO-MAB customization |
| Theoretical Depth | 5 | Complete lower and upper bound analysis with near-optimal regret guarantees |
| Experimental Thoroughness | 3 | Synthetic experiments only, but covers diverse settings and ablations |
| Practical Value | 3.5 | Promising for recommendation/rating systems; linear preference assumption is a notable limitation |
| Writing Quality | 4 | Clear structure, intuitive motivating figures, rigorous theoretical derivations |
| **Overall** | **4** | A pioneering work on preference-centric MO-MAB customization with solid theoretical contributions |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](../../NeurIPS2025/reinforcement_learning/thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[AAAI 2026\] Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation](scalable_multi-objective_and_meta_reinforcement_learning_via_gradient_estimation.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](object-centric_latent_action_learning.md)
- [\[AAAI 2026\] Object-Centric World Models for Causality-Aware Reinforcement Learning](object-centric_world_models_for_causality-aware_reinforcement_learning.md)

<!-- RELATED:END -->
