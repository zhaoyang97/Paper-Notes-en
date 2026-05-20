---
title: >-
  [Paper Note] Risk-Averse Total-Reward Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Risk-averse RL] This paper proposes risk-averse Q-learning algorithms (ERM-TRC and EVaR-TRC) for the undiscounted total-reward criterion (TRC). By exploiting the elicitability of ER…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Risk-averse RL"
  - "total-reward criterion"
  - "Q-learning"
  - "entropic risk measure (ERM)"
  - "entropic value-at-risk (EVaR)"
date: 2026-05-08
content_hash: 8e25c613862038a7
---

# Risk-Averse Total-Reward Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2506.21683](https://arxiv.org/abs/2506.21683)  
**Code**: [Available](https://github.com/suxh2019/ERM_EVaR_Q)  
**Area**: Reinforcement Learning / Risk Aversion
**Keywords**: Risk-averse RL, total-reward criterion, Q-learning, entropic risk measure (ERM), entropic value-at-risk (EVaR)

## TL;DR

This paper proposes risk-averse Q-learning algorithms (ERM-TRC and EVaR-TRC) for the undiscounted total-reward criterion (TRC). By exploiting the elicitability of ERM, the Bellman operator is reformulated as a stochastic gradient descent objective, and convergence guarantees are established.

## Background & Motivation

Risk-averse reinforcement learning is critical in high-stakes applications such as autonomous driving, robotic surgery, healthcare, and finance. Existing risk-averse RL methods primarily focus on **discounted infinite-horizon** objectives; however, many practical tasks (e.g., robotics, games) lack a natural justification for discounting—they have absorbing terminal states in which future rewards should not be discounted.

The **total-reward criterion (TRC)** is an objective function that does not discount future rewards, generalizing stochastic shortest-path and longest-path problems. The core challenges of TRC are:

**Distribution estimation**: Risk-averse RL requires evaluating the full return distribution rather than just its expectation.

**Non-contractive Bellman operator**: The Bellman operator for risk-averse TRC may not be a contraction mapping, precluding direct application of conventional model-free methods.

**Boundedness conditions**: Additional boundedness conditions are required for convergence, and the value function may become unbounded when the risk parameter $\beta$ is too large.

Existing model-based methods (e.g., linear programming approaches) are effective but require complete transition probabilities and do not scale to large problems. This paper presents the first model-free Q-learning algorithms for TRC.

## Method

### Overall Architecture

Two Q-learning algorithms are proposed:
- **ERM-TRC Q-learning**: Targets the entropic risk measure (ERM) objective.
- **EVaR-TRC Q-learning**: Targets the entropic value-at-risk (EVaR) objective, solved by decomposing EVaR into a sequence of ERM subproblems.

### Key Designs

#### 1. Elicitability-Based ERM Bellman Operator

**Mechanism**: The elicitability of ERM is exploited to redefine the Bellman operator as the solution to a regression problem:

$$\hat{B}_\beta q(s,a) = \arg\min_{y \in \mathbb{R}} \mathbb{E}^{a,s}[\ell_\beta(r(s,a,\tilde{s}_1) + \max_{a'} q(\tilde{s}_1,a',\beta) - y)]$$

where the loss function is the exponential loss: $\ell_\beta(z) = \beta^{-1}(\exp(-\beta z) - 1) + z$.

**Design Motivation**: Standard Q-learning can be interpreted as stochastic gradient descent on a quadratic loss. By analogy, ERM Q-learning performs gradient descent along the derivative of the exponential loss $\ell_\beta$, enabling direct estimation of the ERM value function from samples in a model-free setting.

#### 2. Update Rule for ERM-TRC Q-learning

The Q-value update follows the exponential loss gradient:

$$\tilde{q}_{i+1}(s,a,\beta) = \tilde{q}_i(s,a,\beta) - \tilde{\eta}_i \cdot (\exp(-\beta \cdot \tilde{z}_i(\beta)) - 1)$$

where the TD residual is $\tilde{z}_i(\beta) = r(s,a,s'_i) + \max_{a'} \tilde{q}_i(s'_i,a',\beta) - \tilde{q}_i(s,a,\beta)$.

**Boundedness Condition**: When the TD residual falls outside $[z_{\min}, z_{\max}]$, this indicates that $\beta$ is too large and the q-values are unbounded; the algorithm returns $-\infty$.

#### 3. EVaR-TRC Q-learning

EVaR does not satisfy dynamic consistency and cannot be directly handled via Bellman equations. This paper decomposes the EVaR optimization problem into a series of ERM problems over a discretized set of $\beta$ values:

$$(\pi^*, \beta^*) \in \arg\max_{(\pi,\beta) \in \Pi \times \mathcal{B}(\beta_0,\delta)} h(\pi, \beta)$$

By constructing a finite set $\mathcal{B}(\beta_0, \delta)$ of $\beta$ values, ERM Q-learning is run for each $\beta$, and the policy maximizing $h(\pi,\beta)$ is selected, yielding a $\delta$-optimal EVaR policy.

### Convergence Analysis

**Theorem 4.2 (ERM-TRC Convergence)**: Under standard assumptions (infinite visitation of every state-action pair, step-size conditions $\sum \eta_i = \infty$, $\sum \eta_i^2 < \infty$) and the TD-residual boundedness condition, ERM Q-learning converges almost surely to the fixed point of the optimal value function.

Three key technical distinctions in the proof:
1. Rather than relying on the contraction mapping property, the proof exploits the **monotonicity** of the Bellman operator.
2. Additional boundedness conditions are required.
3. The strong convexity and Lipschitz continuity of the exponential loss provide convergence guarantees (strong convexity constant $l=\beta\exp(-\beta z_{\max})$, Lipschitz constant $L=\beta\exp(-\beta z_{\min})$).

## Key Experimental Results

### Main Results: Cliff Walking Environment

Experiments are conducted on two tabular domains: Cliff Walking (CW) and Gambler's Ruin (GR).

| Environment | α (risk level) | Return Mean | Return Std | Return Range |
|-------------|---------------|-------------|------------|--------------|
| CW | α=0.2 | 1.92 | 0.228 | (0, 2] |
| CW | α=0.6 | −0.074 | 0.228 | (−1, 2] |

At α=0.2 (more risk-averse), the agent completely avoids falling off the cliff and returns are almost entirely in (0, 2]; at α=0.6, the agent may fall, resulting in negative returns.

### Convergence Experiment (Ablation Study)

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| 6 random seeds, α=0.2 | EVaR std = 0.015 | Algorithm converges to similar solutions across seeds |
| CW domain, ~20,000 samples | EVaR gap → 0 | Q-learning converges to LP optimal value |
| GR domain, ~20,000 samples | EVaR gap → 0 | Converges to LP optimal value similarly |

### Key Findings

1. **Policy visualization**: Different values of α yield interpretable, distinct policies—smaller α leads the agent to take a longer detour away from the cliff, while larger α encourages riskier shortcuts.
2. **Convergence stability**: The standard deviation across 6 random seeds is only 0.015, demonstrating algorithmic reliability.
3. **Consistency with LP baseline**: Q-learning converges to the same optimal value as the linear programming method after approximately 20,000 samples.

## Highlights & Insights

1. **Strong theoretical contribution**: This is the first model-free risk-averse RL algorithm for the undiscounted total-reward criterion, filling an important theoretical gap.
2. **Elegant use of elicitability**: The elicitability of ERM is leveraged to cast the problem as stochastic gradient descent, enabling model-free learning.
3. **Divergence detection mechanism**: Returning $-\infty$ when the TD residual goes out of bounds elegantly detects divergence caused by excessively large $\beta$.
4. **EVaR discretization**: The dynamic-consistency-violating EVaR problem is gracefully decomposed into a series of ERM subproblems.

## Limitations & Future Work

1. **Tabular setting only**: The current algorithms and analyses are limited to tabular representations; function approximation (e.g., deep neural networks) is not addressed.
2. **Prior-dependent parameter selection**: The choice of $z_{\min}$/$z_{\max}$ requires careful balancing—values that are too small may miss good solutions, while values that are too large slow divergence detection.
3. **Selection of $\beta_0$**: The EVaR algorithm requires selecting an initial $\beta_0$, which may necessitate multiple Q-learning runs to determine an appropriate value.
4. **Extension to continuous action spaces**: Extension to continuous state/action spaces is not discussed.

## Related Work & Insights

- Compared to the discounted ERM Q-learning of Hau et al. (2024), this paper addresses the more challenging undiscounted setting.
- The elicitability + SGD paradigm may generalize to other elicitable risk measures (e.g., approximations of CVaR).
- Future work could integrate deep RL methods such as DQN to scale the algorithms to large-scale problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (first model-free risk-averse RL algorithm for TRC)
- Experimental Thoroughness: ⭐⭐⭐ (validated on tabular domains only)
- Writing Quality: ⭐⭐⭐⭐ (theoretically rigorous, clearly structured)
- Value: ⭐⭐⭐⭐ (important theoretical foundation work)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents](risk-averse_constrained_reinforcement_learning_with_optimized_certainty_equivale.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[NeurIPS 2025\] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)

</div>

<!-- RELATED:END -->
