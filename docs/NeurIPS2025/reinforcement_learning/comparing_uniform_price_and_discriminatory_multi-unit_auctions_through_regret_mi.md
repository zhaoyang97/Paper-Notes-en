---
title: >-
  [Paper Note] Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization
description: >-
  [NeurIPS 2025][Reinforcement Learning][multi-unit auctions] Under the online learning and regret minimization framework, this paper systematically compares the learning difficulty of uniform-price auctions and discriminatory auctions, proving that the two formats share identical worst-case regret rates, while under specific structural conditions the uniform-price auction admits faster learning rates.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - multi-unit auctions
  - online learning
  - regret minimization
  - uniform-price auction
  - discriminatory auction
date: 2026-05-08
content_hash: 47ae75bb682e4500
---

# Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.19591](https://arxiv.org/abs/2510.19591)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: multi-unit auctions, online learning, regret minimization, uniform-price auction, discriminatory auction

## TL;DR

Under the online learning and regret minimization framework, this paper systematically compares the learning difficulty of uniform-price auctions and discriminatory auctions, proving that the two formats share identical worst-case regret rates, while under specific structural conditions the uniform-price auction admits faster learning rates.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Uniform-price auctions and discriminatory auctions are the most widely used multi-unit auction mechanisms in electricity markets and treasury auctions. Both share identical allocation rules—goods are allocated to the highest bidders—differing only in their pricing rules.

Traditional comparisons of these two auction formats focus on equilibrium analyses of efficiency, revenue, and social welfare. However, when auctions are repeated, participants can leverage online learning techniques to progressively optimize their strategies. This paper proposes a novel comparative perspective: **measuring the difficulty of learning to bid via regret**, thereby quantifying the intrinsic learning complexity differences between the two auction formats.

Core question: Facing stochastic opponents, how many rounds does a bidder need in each auction format to approximate the optimal strategy?

## Method

### Overall Architecture

The repeated auction is modeled as an online learning problem. A single bidder facing stochastic opponents learns the optimal bidding strategy over $T$ rounds. The central measure is pseudo-regret:

$$R_T = T \sup_{\mathbf{b} \in B} \mathbb{E}_{\boldsymbol{\beta} \sim \mathcal{D}}[u(\mathbf{b}, \boldsymbol{\beta})] - \sum_{t=1}^{T} \mathbb{E}_{\boldsymbol{\beta}^t \sim \mathcal{D}}[u(\mathbf{b}^t, \boldsymbol{\beta}^t)]$$

where $\mathbf{b}^t$ is the bidder's bid in round $t$, $\boldsymbol{\beta}^t$ is the opponents' bids, and $u$ is the utility function.

**Two pricing rules**:
- Uniform-price auction: all winning units are priced at the $(K+1)$-th highest bid
$$p(\mathbf{b}, \boldsymbol{\beta}) = \max(b_{x(\mathbf{b},\boldsymbol{\beta})+1}, \beta_{K-x(\mathbf{b},\boldsymbol{\beta})+1})$$

- Discriminatory auction: each winning unit is priced at its corresponding bid
$$p(\mathbf{b}, \boldsymbol{\beta}) = (b_k)_{k \in [K]}$$

### Key Designs

**Full-information feedback algorithm** (Algorithm 1):

The core idea is to estimate the marginal cumulative distribution functions $(F_k)_{k \in [K]}$ of opponents' bids, rather than directly searching for the optimal bid.

1. After each round, observe the complete opponent bid vector $\boldsymbol{\beta}^t$
2. Construct the empirical CDF: $\hat{F}_k^t(x) = \frac{1}{t-1} \sum_{j=1}^{t-1} \mathbb{1}\{\beta_k^j \leq x\}$
3. Estimate the expected utility: $\hat{u}^t(\mathbf{b}) = U((\hat{F}_k^t), \mathbf{b})$
4. Greedy bidding: $\mathbf{b}^t = \arg\max_{\mathbf{b} \in B} \hat{u}^t(\mathbf{b})$

Key Lemma (Lemma 1): The expected utility in both auction formats can be expressed as a function of the bid vector and marginal CDFs, making CDF-estimation-based algorithms applicable to both formats.

**Regret separation under bandit feedback**:

In the uniform-price auction, when the bidder's demand is low, the feedback structure is richer—partial observations of opponents' order statistics become available. The authors exploit a concentration inequality for partially observed order statistics based on the DKW inequality (which may be of independent research interest) to establish this structural advantage.

### Loss & Training

As an online learning framework, no conventional loss function training is involved. The core objective is to minimize cumulative regret $R_T$. Algorithm design follows the explore-exploit paradigm: implicit exploration via statistical CDF estimation, and exploitation via greedy optimization.

## Key Experimental Results

### Main Results

Comparison of regret rates between the two auction formats under bandit feedback (core theoretical results):

| Setting | Unit demand | 2-unit demand | General demand | $\Delta$-separated | i.i.d. |
|---------|-------------|---------------|----------------|--------------------|--------|
| Discriminatory | $\tilde{\Theta}(T^{2/3})$ | $\tilde{\Theta}(T^{2/3})$ | $\tilde{\Theta}(T^{2/3})$ | — | — |
| Uniform-price | **0** | $\tilde{\Theta}(\sqrt{T})$ | $\tilde{\Theta}(T^{2/3})$ | $\tilde{\Theta}(\sqrt{T})$ | $\tilde{\Theta}(\sqrt{T})$ |

Under full-information feedback: both auction formats achieve $\tilde{\mathcal{O}}(K\sqrt{T})$ (improving the known upper bound for uniform-price auctions by a factor of $\sqrt{K}$).

### Ablation Study

Analysis of key conditions for regret rate separation:

1. **Effect of demand quantity**: The uniform-price auction achieves zero regret under unit demand (truthful bidding is optimal), $\tilde{\Theta}(\sqrt{T})$ under 2-unit demand, both superior to the discriminatory auction's $\tilde{\Theta}(T^{2/3})$
2. **$\Delta$-separation condition**: When instance-dependent parameters ensure sufficient separation between the optimal and suboptimal bids, the uniform-price auction achieves $\mathcal{O}(\sqrt{T})$
3. **Symmetric unit-demand opponents**: Under this structured setting, a dedicated algorithm is provided guaranteeing $\tilde{\mathcal{O}}(\sqrt{T})$ regret for the uniform-price auction

### Key Findings

1. **Worst-case equivalence**: Both auction formats share identical worst-case regret rates under both full-information and bandit feedback
2. **Beyond-worst-case separation**: The uniform-price auction enables faster learning ($\sqrt{T}$ vs $T^{2/3}$) under low demand or structurally favorable instances
3. The first tight lower bound $\Omega(T^{2/3})$ for the uniform-price auction under bandit feedback is established
4. The $\Omega(T^{2/3})$ lower bound for discriminatory auctions holds even on instances where the uniform-price auction achieves $\sqrt{T}$ rates

## Highlights & Insights

- **Core finding**: The difference in learning difficulty between auction formats lies not in the worst case, but in **separation on structured instances**—the feedback structure of the uniform-price auction implicitly carries richer information under certain conditions
- **Technical contribution**: A concentration inequality for partially observed order statistics based on the DKW inequality, which is of independent research interest
- **Methodological innovation**: A deterministic algorithm operating directly over the continuous action space (rather than discretization plus stochastic bandits) is proposed, yielding cleaner analysis

## Limitations & Future Work

1. Opponents' bids are assumed to follow a stochastic (i.i.d.) distribution; strategic opponents and game-theoretic equilibrium analysis are not addressed
2. Theoretical results focus on asymptotic regret rates without finite-time numerical validation
3. The dependence on the number of items $K$ in the regret rates may not be optimal (scaling as $K^{5/3}$ under general demand)
4. Extensions to combinatorial auctions or heterogeneous-item auctions are not discussed
5. Algorithms require knowledge of the time horizon $T$ and are not anytime algorithms

## Related Work & Insights

- **Connection to online learning**: The auction mechanism design problem is cast as a standard online learning problem, leveraging concentration inequalities and CDF estimation tools
- **Relation to algorithmic game theory**: This work complements the traditional equilibrium analysis perspective by providing a new basis for auction format selection from the standpoint of learning complexity
- **Directions for inspiration**: Extending the regret minimization framework to combinatorial auctions; studying learning rate separation under strategic opponents (coarse correlated equilibrium)

## Rating

- ⭐ Novelty: 4/5 — The regret minimization perspective for comparing auction formats is a novel and meaningful analytical framework
- ⭐ Value: 3/5 — Offers theoretical guidance for electricity markets and treasury auctions, but lacks empirical validation
- ⭐ Experimental Thoroughness: 2/5 — Purely theoretical work with no numerical experiments
- ⭐ Writing Quality: 4/5 — Theorem statements are clear, result tables are intuitive, and technical details are well organized

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Simultaneous Swap Regret Minimization via KL-Calibration](simultaneous_swap_regret_minimization_via_kl-calibration.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Dynamic Regret Reduces to Kernelized Static Regret](dynamic_regret_reduces_to_kernelized_static_regret.md)
- [\[AAAI 2026\] Deep (Predictive) Discounted Counterfactual Regret Minimization](../../AAAI2026/reinforcement_learning/deep_predictive_discounted_counterfactual_regret_minimization.md)
- [\[NeurIPS 2025\] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality](improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)

<!-- RELATED:END -->
