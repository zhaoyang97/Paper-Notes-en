---
title: >-
  [Paper Note] Towards Sustainable Investment Policies Informed by Opponent Shaping
description: >-
  [ICLR 2026][Others][Opponent Shaping] This paper formally proves the conditions under which the InvestESG simulation environment constitutes a social dilemma and applies the Advantage Alignment opponent shaping algorithm to guide economic agents toward a sustainable investment equilibrium.
tags:
  - ICLR 2026
  - Others
  - Opponent Shaping
  - Advantage Alignment
  - ESG
  - InvestESG
date: 2026-05-08
content_hash: ec70e9cb80bc8f14
---
# Towards Sustainable Investment Policies Informed by Opponent Shaping

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2602.11829](https://arxiv.org/abs/2602.11829)
- **Code**: To be released
- **Area**: Others
- **Keywords**: Opponent Shaping, Advantage Alignment, Social Dilemma, ESG, Climate Risk, InvestESG

## TL;DR
This paper formally proves the conditions under which the InvestESG simulation environment constitutes a social dilemma and applies the Advantage Alignment opponent shaping algorithm to guide economic agents toward a sustainable investment equilibrium.

## Background & Motivation

### Core Problem
Addressing climate change requires global coordination, but rational economic agents typically prioritize immediate benefits, leading to social dilemmas. How can multi-agent reinforcement learning (MARL) be used to discover and promote sustainable investment strategies?

### InvestESG Environment
A climate investment simulation driven by Multi-Agent RL:
- **Firm Agents**: Allocate capital to mitigation, adaptation, and greenwashing strategies.
- **Investor Agents**: Reallocate capital based on profitability and ESG scores.
- Climate risk is determined by cumulative mitigation investment over a 100-year horizon.

### Limitations of Prior Work
- Traditional MARL methods like IPPO/MAPPO converge to selfish strategies.
- Opponent shaping methods such as LOLA and M-FOS suffer from poor scalability or only support discrete action spaces.
- Sum-of-rewards methods fail due to credit assignment issues when the number of agents exceeds 4.

## Method

### Overall Architecture

Ours follows a two-step approach: first, using game-theoretic tools to characterize the parameters under which InvestESG constitutes a "social dilemma," and then integrating Advantage Alignment—an opponent shaping algorithm—into PPO to enable self-interested firm and investor agents to spontaneously converge to a sustainable investment equilibrium. The first part is the diagnosis—proving the existence conditions of the dilemma and its gradient roots; the second part is the prescription—incorporating "consideration for others" into the policy gradient by modifying the advantage function.

### Key Designs

**1. Formalization of Social Dilemma and the $\alpha$-Dilemma Band**

To determine if a multi-agent environment warrants opponent shaping, one must first prove a dilemma exists and identify its "trigger." This paper adopts the Price of Anarchy $\mathcal{P}_a = \frac{\max_{\pi \in \Pi} \mathcal{W}(\pi; \mu)}{\min_{\pi \in \mathcal{N}} \mathcal{W}(\pi; \mu)}$, the ratio of the global optimal social welfare to the worst Nash equilibrium welfare, as a verifiable scalar criterion. When $\mathcal{P}_a > 1$, the individual rational outcome is strictly inferior to the achievable social optimum, establishing the dilemma.

The emergence of the dilemma is controlled by a specific parameter. Climate event probability is defined as $P_t^e = \frac{\mu_e t}{1 + \lambda_e U_t} + P_0^e$, where $\lambda_e = \alpha \times \tilde{\lambda}_e$, and $\alpha$ represents the responsiveness of climate risk to cumulative mitigation investment $U_t$ (mitigation effectiveness). Scanning along $\lambda$ reveals three regions: when $\lambda < \lambda_{\text{low}}$, mitigation returns are always net negative and no one invests (no dilemma); when $\lambda > \lambda_{\text{critical}}$, mitigation returns are high enough that even self-interested agents invest (no strong dilemma); only in the middle band $\lambda_{\text{low}} \leq \lambda \leq \lambda_{\text{critical}}$, where individual gradients and social gradients have opposite signs, does the environment fall into a dilemma zone requiring intervention. This explains why experiments fix $\alpha = 70$—it pins the environment exactly within the dilemma band.

**2. Misalignment of Private and Social Gradients**

The paper calculates the private marginal gradient of a single firm regarding its own capital expectation:

$$\frac{d}{du_t^i}\mathbb{E}[K_{t+1}^i] = -\frac{\mathbb{E}[K_{t+1}^i]}{1-u_t^i} + \mathbb{E}\left[\frac{(K_{t+1}^i)^2}{(1-X_t L_i)^2(1-u_t^i)(1+\gamma)} \sum_e \frac{\lambda_e \mu_e t}{(1+\lambda_e U_t)^2}\right]$$

The first term represents the immediate loss from allocating capital to mitigation, while the second term represents the capital recovered by reducing climate risk. Lemma 1 proves that the social marginal gradient is strictly greater than the private marginal gradient—individuals only perceive the costs they bear but ignore the external benefits mitigation brings to all. Consequently, collective underinvestment occurs in the dilemma band. This inequality provides the precise target for where Advantage Alignment should "fill the gap."

**3. Advantage Alignment: Embedding Altruism into the Advantage Function**

The algorithm modifies the standard advantage $A^i$ into:

$$A^{*,i}(s_t, \mathbf{a}_t) = A^i(s_t, \mathbf{a}_t) + \beta\gamma \sum_{j \neq i}\left(\sum_{k<t} \gamma^{t-k} A^i(s_k, \mathbf{a}_k)\right) A^j(s_t, \mathbf{a}_t)$$

The additional term multiplies agent $i$'s discounted cumulative past advantages with the current privileges $A^j$ of others. When an action is both beneficial to one's own history and others' current state, its effective advantage is amplified, encouraging pro-collective behavior. $\beta$ adjusts the shaping intensity. Crucially, this is a **purely additive modification** to the advantage, allowing it to be plugged into PPO without higher-order gradients or opponent modeling, bypassing the scalability bottlenecks of LOLA/M-FOS.

The modified advantage can be decomposed as:

$$A_t^{*,i} = \underbrace{A_t^i + \beta\gamma b^i \sum_{j \neq i} A_t^j}_{\text{Cooperation Bias}} + \beta\gamma \sum_{j \neq i} \underbrace{\left(\sum_{k<t} \gamma^{t-k} A_k^i - b^i\right)}_{\text{Zero-mean}} A_t^j$$

The first term is an explicit cooperation bias; when $\beta\gamma b^i = 1$, it reverts to sum-of-rewards learning. The second term has a zero mean and only provides shaping when past advantages deviate from the baseline $b^i$. Early in training, the lag in the critic network results in $b^i > 0$, creating an initial bias that pushes agents out of the selfish equilibrium. As the critic becomes accurate and $b^i$ converges, the bias fades, allowing the policy to stabilize in a cooperative equilibrium rather than being continuously distorted by a fixed external force.

## Experiments

### Main Results ($\alpha = 70$)

| Metric | PPO (ESG=0) | PPO (ESG=1) | PPO (ESG=10) | AdAlign |
| :--- | :--- | :--- | :--- | :--- |
| Total Market Wealth | Lower | Medium | Medium-High | **Highest** |
| Final Mitigation Investment | Excessive | Medium | Medium | **Lower but more strategic** |
| Final Climate Risk | ~0.48 | ~0.48 | ~0.48 | **~0.48** |

### Scalability

| Number of Agents | AdAlign | PPO+Sum Rewards | IPPO | MAPPO |
| :--- | :--- | :--- | :--- | :--- |
| 2 (1+1) | ✓ | ✓ | - | - |
| 4 (2+2) | ✓ | ✓ | - | - |
| 6+ | ✓ | **✗ (Collapse)** | - | - |
| 10 (5+5) | ✓ | ✗ | - | - |

### Key Findings

Characteristics of the strategies learned by Advantage Alignment:
1. **Precision Mitigation**: Investing only at critical moments when climate risk rises, rather than over-investing.
2. **Even Distribution**: Investors maintain a nearly uniform distribution of investment across firms (low Gini coefficient).
3. **Coordinated Burden-sharing**: Firms coordinate to share the costs of mitigation.

## Highlights & Insights
1. **Theoretical contribution**: Rigorous proof of the parameter conditions that make InvestESG a social dilemma.
2. **Utility**: Advantage Alignment guides agents toward cooperative equilibrium without requiring government intervention.
3. **Scalability**: Remains effective as the number of agents grows, outperforming sum-of-rewards methods.
4. **Policy Interpretability**: The learned strategies align with economic intuition.

## Limitations & Future Work
1. Simplified assumptions within the InvestESG simulator (limited firms/investors, simplified climate models).
2. The choice of $\alpha = 70$ is empirical, with limited discussion on real-world parameter calibration.
3. Considers only firm and investor agents, excluding the role of government.
4. Advantage Alignment requires Centralized Training (CTDE).

## Related Work & Insights
- **Opponent Shaping**: LOLA, COLA, M-FOS — limited scalability.
- **Climate AI**: RICE-N (international negotiations), AI Economist (carbon trading).
- **Social Dilemma RL**: Prisoner's Dilemma, Sequential Social Dilemmas.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Valuable combination of theoretical analysis and algorithmic application.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Solid ablation and scalability analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous theory and clear expression.
- **Value**: ⭐⭐⭐ — Practical guidance for real financial decision-making requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Sustainable AI Economy Needs Data Deals That Work for Generators](../../NeurIPS2025/others/a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)
- [\[ECCV 2024\] Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data](../../ECCV2024/others/superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)
- [\[ACL 2025\] Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text](../../ACL2025/others/task-informed_anti-curriculum_by_masking_improves_downstream_performance_on_text.md)
- [\[NeurIPS 2025\] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications](../../NeurIPS2025/others/military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)
- [\[ICLR 2026\] Adaptive Conformal Guidance for Learning under Uncertainty](adaptive_conformal_guidance_for_learning_under_uncertainty.md)

</div>

<!-- RELATED:END -->
