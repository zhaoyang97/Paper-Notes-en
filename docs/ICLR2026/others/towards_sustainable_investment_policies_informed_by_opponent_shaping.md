---
title: >-
  [Paper Note] Towards Sustainable Investment Policies Informed by Opponent Shaping
description: >-
   This paper formally proves the conditions under which the InvestESG simulation environment constitutes a social dilemma, and applies the Advantage Alignment opponent shaping algorithm to guide economic agents toward sustainable investment equilibria.
tags:

date: 2026-05-08
content_hash: 9dbb676ee8ef46a1
---

# Towards Sustainable Investment Policies Informed by Opponent Shaping

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2602.11829](https://arxiv.org/abs/2602.11829)
- **Code**: To be released
- **Area**: Other
- **Keywords**: Opponent Shaping, Advantage Alignment, Social Dilemmas, ESG, Climate Risk, InvestESG

## TL;DR
This paper formally proves the conditions under which the InvestESG simulation environment constitutes a social dilemma, and applies the Advantage Alignment opponent shaping algorithm to guide economic agents toward sustainable investment equilibria.

## Background & Motivation

### Core Problem
Addressing climate change requires global coordination, yet rational economic agents typically prioritize immediate gains, giving rise to social dilemmas. The paper investigates how multi-agent reinforcement learning can be used to discover and promote sustainable investment strategies.

### InvestESG Environment
A climate investment simulation driven by Multi-Agent RL:
- **Firm agents**: Allocate capital across mitigation, adaptation, and greenwashing strategies
- **Investor agents**: Reallocate capital based on profitability and ESG scores
- Climate risk is determined over a 100-year horizon by cumulative mitigation investment

### Limitations of Prior Work
- Conventional MARL methods such as IPPO/MAPPO converge to selfish strategies
- Opponent shaping methods such as LOLA and M-FOS suffer from poor scalability or support only discrete action spaces
- Reward summation approaches fail due to credit assignment problems when the number of agents exceeds 4

## Method

### Formalization of Social Dilemmas

**Definition (Price of Anarchy)**:
$$\mathcal{P}_a = \frac{\max_{\pi \in \Pi} \mathcal{W}(\pi; \mu)}{\min_{\pi \in \mathcal{N}} \mathcal{W}(\pi; \mu)}$$

A social dilemma exists when $\mathcal{P}_a > 1$.

### Key Parameter: Mitigation Effectiveness $\alpha$

Climate event probability:
$$P_t^e = \frac{\mu_e t}{1 + \lambda_e U_t} + P_0^e, \quad \lambda_e = \alpha \times \tilde{\lambda}_e$$

**Core Finding**: The parameter $\alpha$ (climate responsiveness to mitigation) determines whether a social dilemma exists.

Three regimes are identified:
1. $\lambda < \lambda_{\text{low}}$: Mitigation always yields net negative returns — no dilemma
2. $\lambda_{\text{low}} \leq \lambda \leq \lambda_{\text{critical}}$: Individual and social gradient signs disagree — **social dilemma**
3. $\lambda > \lambda_{\text{critical}}$: Self-interested agents begin to mitigate — no strong dilemma

### Private Marginal Gradient vs. Social Marginal Gradient

**Private gradient**:
$$\frac{d}{du_t^i}\mathbb{E}[K_{t+1}^i] = -\frac{\mathbb{E}[K_{t+1}^i]}{1-u_t^i} + \mathbb{E}\left[\frac{(K_{t+1}^i)^2}{(1-X_t L_i)^2(1-u_t^i)(1+\gamma)} \sum_e \frac{\lambda_e \mu_e t}{(1+\lambda_e U_t)^2}\right]$$

**Lemma 1**: The social marginal gradient is strictly greater than the private marginal gradient.

### Advantage Alignment

The advantage function in the policy gradient is modified as follows:
$$A^{*,i}(s_t, \mathbf{a}_t) = A^i(s_t, \mathbf{a}_t) + \beta\gamma \sum_{j \neq i}\left(\sum_{k<t} \gamma^{t-k} A^i(s_k, \mathbf{a}_k)\right) A^j(s_t, \mathbf{a}_t)$$

This directly modifies the policy gradient to incentivize actions that are beneficial to both the agent itself and others, and can be directly integrated into the PPO framework.

### Why Advantage Alignment Works

The modified advantage is decomposed as:
$$A_t^{*,i} = \underbrace{A_t^i + \beta\gamma b^i \sum_{j \neq i} A_t^j}_{\text{cooperative bias}} + \beta\gamma \sum_{j \neq i} \underbrace{(\sum_{k<t} \gamma^{t-k} A_k^i - b^i)}_{\text{zero-mean}} A_t^j$$

When $\beta\gamma b^i = 1$, the cooperative term is equivalent to reward summation learning. In early training, the critic network lags behind, yielding $b^i > 0$ and thus an initial cooperative bias. As the critic improves, this bias vanishes.

## Experiments

### Main Results ($\alpha = 70$)

| Metric | PPO (ESG=0) | PPO (ESG=1) | PPO (ESG=10) | AdAlign |
|------|------------|------------|-------------|---------|
| Total Market Wealth | Low | Medium | Medium-High | **Highest** |
| Final Mitigation Investment | Excessive | Medium | Medium | **Lower but more strategic** |
| Final Climate Risk | ~0.48 | ~0.48 | ~0.48 | **~0.48** |

### Scalability

| Number of Agents | AdAlign | PPO+Sum Rewards | IPPO | MAPPO |
|-----------|---------|----------------|------|-------|
| 2 (1+1) | ✓ | ✓ | - | - |
| 4 (2+2) | ✓ | ✓ | - | - |
| 6+ | ✓ | **✗ (collapse)** | - | - |
| 10 (5+5) | ✓ | ✗ | - | - |

### Policy Interpretation

Characteristics of the strategy learned by Advantage Alignment:
1. **Targeted mitigation**: Investment is concentrated at critical moments when climate risk rises, rather than over-investing
2. **Uniform allocation**: Investors maintain an approximately uniform distribution of firm investments (low Gini coefficient)
3. **Coordinated cost-sharing**: Firms coordinate to share mitigation costs

## Highlights & Insights
1. **Theoretical contribution**: Rigorous derivation of the parameter conditions under which InvestESG constitutes a social dilemma
2. **Practicality**: Advantage Alignment guides agents toward cooperative equilibria without requiring government intervention
3. **Scalability**: Remains effective as the number of agents grows, outperforming reward summation approaches
4. **Policy interpretability**: The learned strategies exhibit clear economic intuition

## Limitations & Future Work
1. Simplifying assumptions inherent to the InvestESG simulator (limited numbers of firms and investors, simplified climate model)
2. The choice of $\alpha = 70$ is empirical, with limited discussion of calibration to real-world parameters
3. Only firm and investor agents are considered; government roles are not incorporated
4. Advantage Alignment requires centralized training (CTDE)

## Related Work & Insights
- **Opponent shaping**: LOLA, COLA, M-FOS — limited scalability
- **Climate AI**: RICE-N (international negotiations), AI Economist (carbon trading)
- **Social dilemma RL**: Prisoner's Dilemma, Sequential Social Dilemmas

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of theoretical analysis and algorithmic application is valuable
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Ablation and scalability analyses are thorough
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretically rigorous with clear exposition
- **Value**: ⭐⭐⭐ — Practical implications for real-world financial decision-making require further validation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Sustainable AI Economy Needs Data Deals That Work for Generators](../../NeurIPS2025/others/a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)
- [\[AAAI 2026\] CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data](../../AAAI2026/others/cellstream_dynamical_optimal_transport_informed_embeddings_for_reconstructing_ce.md)
- [\[NeurIPS 2025\] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications](../../NeurIPS2025/others/military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)

</div>

<!-- RELATED:END -->
