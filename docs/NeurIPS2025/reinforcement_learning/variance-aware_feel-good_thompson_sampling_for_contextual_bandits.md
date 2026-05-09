---
title: >-
  [Paper Note] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits
description: >-
  [NeurIPS 2025][Reinforcement Learning][contextual bandits] This paper proposes the FGTS-VA algorithm, which for the first time achieves a variance-aware contextual bandit algorithm based on Feel-Good Thompson Sampling. The resulting regret bound is optimal in the model dimension $d$, matching the best variance-dependent regret bounds established by UCB-based methods.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - contextual bandits
  - Thompson sampling
  - variance-aware
  - feel-good exploration
  - regret bounds
date: 2026-05-08
content_hash: 6915630233762aaf
---

# Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits

**Conference**: NeurIPS 2025
**arXiv**: [2511.02123](https://arxiv.org/abs/2511.02123)
**Code**: Not available
**Area**: Reinforcement Learning
**Keywords**: contextual bandits, Thompson sampling, variance-aware, feel-good exploration, regret bounds

## TL;DR

This paper proposes the FGTS-VA algorithm, which for the first time achieves a variance-aware contextual bandit algorithm based on Feel-Good Thompson Sampling. The resulting regret bound is optimal in the model dimension $d$, matching the best variance-dependent regret bounds established by UCB-based methods.

## Background & Motivation

Contextual bandits are a fundamental framework for interactive decision-making, in which an agent observes a context, selects an action, and receives a stochastic reward at each step. Standard algorithms, however, are notably ill-suited for handling heteroscedastic noise, i.e., settings where the reward variance varies across time steps.

**Importance of variance-awareness**: The worst-case regret bound for standard linear bandit algorithms is $\tilde{\mathcal{O}}(d\sqrt{T})$, yet when rewards are nearly deterministic, simple exploration suffices to attain $\tilde{\mathcal{O}}(d)$. Variance-aware algorithms can adaptively exploit "benign environments," achieving better performance when noise is low.

**Limitations of Prior Work**:

UCB-based methods have achieved the optimal bound $\tilde{\mathcal{O}}(d\sqrt{\sum_t \sigma_t^2} + d)$ (e.g., Weighted OFUL+ and SAVE), whereas Thompson sampling (TS) based methods lag significantly behind.

The only existing variance-aware TS algorithm, LinVDTS, attains a regret bound of $\tilde{\mathcal{O}}(d^{1.5}\sqrt{\sum_t \sigma_t^2} + d^{1.5})$, which is suboptimal in dimension $d$. This is a common shortcoming of TS algorithms, stemming from crude handling of the posterior covariance in classical TS analyses.

FGTS corrects the suboptimality of standard TS under frequentist regret analysis by incorporating a feel-good exploration bonus, but its existing formulation does not support variance-awareness.

**Core Open Problem**: Can one design a FGTS-based contextual bandit algorithm whose regret bound is simultaneously optimal in $d$ and variance-dependent, analogous to UCB-based algorithms?

## Method

### Overall Architecture

FGTS-VA follows the general Thompson sampling framework: at each step, a reward function estimate is sampled from a posterior distribution, and the action maximizing this estimate is selected. The key innovation lies in the design of the posterior distribution.

### Key Designs

1. **Variance-aware posterior distribution**: FGTS-VA defines the following posterior:

$$p_t(f|S_{t-1}) \propto p_0(f) \exp\left(-\sum_{s=1}^{t-1} \eta_s (r_s - f(x_s, a_s))^2 + \lambda_t \max_{a \in \mathcal{A}_t} f(x_t, a)\right)$$

Two critical parameters appear here:
- $\eta_s = \bar{\sigma}_s^{-2}$: variance-dependent weights that downweight high-noise steps ($\bar{\sigma}_s = \max\{\sigma_s, \alpha\}$, where $\alpha$ prevents degeneracy when variance is zero).
- $\lambda_t = c\sqrt{\Lambda_t}/\bar{\sigma}_t^2$: the feel-good exploration strength, which is amplified when the current step has low noise (more informative feedback warrants more exploration).

The cumulative variance $\Lambda_t = \sum_{s=1}^{t}\bar{\sigma}_s^2$ is used in place of the total variance $\Lambda$, elegantly avoiding dependence on the time horizon $T$.

2. **Type B feel-good exploration**: Unlike the Type A variant of Zhang (2022), which adds an exploration bonus at all historical steps, FGTS-VA adopts Type B, adding the exploration term $\lambda_t \max_a f(x_t, a)$ only at the current step. The authors provide a detailed analysis of the technical obstacles facing Type A in the variance-aware setting: the requirement that $\eta_t$ be an absolute constant leads to an irreducible factor of $T$ in the regret bound.

3. **Generalized Decoupling Coefficient**: This is the central analytical tool of the paper, extending the standard decoupling coefficient of Dann et al. (2021) to accommodate weighted parameters $\beta_t$:

$$\sum_{t=1}^T (f_t(z_t) - f_*(z_t)) \leq \sum_{t=1}^T \frac{\gamma}{\beta_t} \sum_{s=1}^{t-1} \beta_s (f_t(z_s) - f_*(z_s))^2 + \gamma\lambda\sum_{t=1}^T \frac{1}{\beta_t} + \left(1 + \frac{1}{4\gamma}\right) \text{dc}$$

The authors show that the generalized decoupling coefficient is $\tilde{\mathcal{O}}(d)$ for linear function classes, and can be upper-bounded by the generalized Eluder dimension for general function classes.

### Proof Overview

- **Technique 1: Averaging over reward randomness first.** This avoids directly computing expectations of exponential terms over posterior samples—a key obstacle in the Type A analysis—by instead leveraging the sub-Gaussian property of noise to eliminate reward randomness first.
- **Technique 2: KL-regularization optimality.** The Donsker–Varadhan duality is applied twice—once to exploit the inequality itself, and once to leverage the optimality condition—elegantly canceling KL divergence terms and reducing expectations of exponential quantities to ordinary expectations.

## Key Experimental Results

### Theoretical Regret Comparison

| Algorithm | Technique | General Regret | Deterministic Regret | $\sigma_t^2$ Required |
|-----------|-----------|---------------|---------------------|----------------------|
| Weighted OFUL+ | UCB | $\tilde{\mathcal{O}}(d\sqrt{\Lambda}+d)$ | $\tilde{\mathcal{O}}(d)$ | Known |
| SAVE | UCB | $\tilde{\mathcal{O}}(d\sqrt{\Lambda}+d)$ | $\tilde{\mathcal{O}}(d)$ | Unknown |
| LinVDTS | TS | $\tilde{\mathcal{O}}(d^{1.5}\sqrt{\Lambda}+d^{1.5})$ | $\tilde{\mathcal{O}}(d^{1.5})$ | Unknown |
| FGTS | TS | $\tilde{\mathcal{O}}(d\sqrt{T})$ | $\tilde{\mathcal{O}}(d\sqrt{T})$ | N/A |
| **FGTS-VA** | **TS** | $\tilde{\mathcal{O}}(d\sqrt{\Lambda}+d)$ | $\tilde{\mathcal{O}}(d)$ | Known |

### Special Case Analysis

| Setting | FGTS-VA Regret | Remarks |
|---------|---------------|---------|
| Deterministic rewards ($\sigma_t^2=0$) | $\mathcal{O}(\text{dc})$ | Matches lower bound; optimal for general function classes |
| Standard bandits ($\sigma_t^2=1$) | $\tilde{\mathcal{O}}(d\sqrt{T})$ | First FGTS algorithm that does not require knowledge of $T$ |
| Linear bandits | $\tilde{\mathcal{O}}(d\sqrt{\Lambda}+d)$ | Matches the optimal UCB-class bound |

### Key Findings

1. **First to bridge the TS–UCB gap in variance-aware bandits**: FGTS-VA matches the optimal UCB-class bound $\tilde{\mathcal{O}}(d\sqrt{\Lambda}+d)$ in the linear setting, eliminating the extra $d^{0.5}$ factor present in LinVDTS.
2. **No prior knowledge of the time horizon required**: When $\sigma_t^2=1$, FGTS-VA reduces to the standard contextual bandit FGTS and is the first such algorithm that operates without knowledge of $T$.
3. **Extension to general function classes**: Through the connection between the generalized decoupling coefficient and the generalized Eluder dimension, FGTS-VA applies to arbitrary finite function classes beyond linear models.

## Highlights & Insights

- The technical advantage of Type B feel-good exploration in the variance-aware setting is revealed for the first time. The constraint that $\eta_t$ must be an absolute constant under Type A is fundamental and cannot be circumvented by straightforward techniques.
- The generalized decoupling coefficient provides a unified analytical tool for heterogeneously weighted Thompson sampling, potentially of value for other online learning problems.
- Replacing $\Lambda$ with $\Lambda_t$ to avoid dependence on $T$ is a subtle but insightful maneuver that reflects a deep understanding of the proof structure.

## Limitations & Future Work

- The current formulation requires $\sigma_t^2$ to be known to the agent (the "weak adversary + variance revelation" setting); extending results to the unknown variance case is an important direction for future work.
- This is a purely theoretical contribution with no empirical experiments to validate whether FGTS-VA outperforms UCB-based methods in practice.
- The precise characterization of the generalized decoupling coefficient for certain function classes remains incomplete.

## Related Work & Insights

- The paper successfully extends the Type B FGTS framework, originally designed for model-free RL by Dann et al. (2021), to (variance-aware) contextual bandits, demonstrating the broad applicability of this technique.
- The work forms a natural "UCB–TS duality" with Weighted OFUL+ of Zhou and Gu (2022), theoretically establishing the equivalence of the two paradigms in the variance-aware setting.
- The paper offers a new analytical paradigm for designing variance-adaptive algorithms in RL theory.

## Rating

- Novelty: ⭐⭐⭐⭐ The generalized decoupling coefficient and proof techniques are genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical; no empirical experiments.
- Writing Quality: ⭐⭐⭐⭐ The proof exposition is clear and the technical motivation is well-articulated.
- Value: ⭐⭐⭐⭐ Addresses an important open problem in the TS literature with solid theoretical contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown](feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Thompson Sampling in Function Spaces via Neural Operators](thompson_sampling_in_function_spaces_via_neural_operators.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](learning_from_demonstrations_via_capability-aware_goal_sampling.md)

</div>

<!-- RELATED:END -->
