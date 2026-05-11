---
title: >-
  [Paper Note] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management
description: >-
  [AAAI 2026][Reinforcement Learning][Portfolio Management] This paper proposes the MARS framework, which achieves risk-aware portfolio management under dynamic market conditions through a two-level architecture comprising…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Portfolio Management"
  - "Multi-Agent Reinforcement Learning"
  - "Risk Management"
  - "Meta-Learning"
  - "Safety Critic"
date: 2026-05-08
content_hash: 4c307ca5661f2807
---

# MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management

**Conference**: AAAI 2026
**arXiv**: [2508.01173](https://arxiv.org/abs/2508.01173)
**Code**: N/A
**Area**: Reinforcement Learning
**Keywords**: Portfolio Management, Multi-Agent Reinforcement Learning, Risk Management, Meta-Learning, Safety Critic

## TL;DR

This paper proposes the MARS framework, which achieves risk-aware portfolio management under dynamic market conditions through a two-level architecture comprising a Heterogeneous Agent Ensemble (HAE)—where each agent has a distinct risk preference and Safety-Critic—and a Meta-Adaptive Controller (MAC). The framework significantly reduces maximum drawdown and volatility.

## Background & Motivation

Deep reinforcement learning (DRL) has made notable advances in automated portfolio management, yet faces two core challenges:

**Non-stationarity**: The statistical properties of financial markets change over time, violating the stationary environment assumption of MDPs. Models trained under one market regime (e.g., a low-volatility bull market) often fail catastrophically when the regime shifts (e.g., a bear market), as previously learned patterns become obsolete.

**Superficial risk management**: Many DRL models handle risk implicitly through reward shaping (e.g., using the Sharpe Ratio as a reward signal). This approach is inherently *reactive*—penalizing risk only after it materializes—rather than proactively managing risk during the decision-making process as human traders do. Consequently, agents remain vulnerable to tail risks and sudden market shocks.

**Deep entanglement of the two challenges**: An agent unable to adapt to regime changes is also unable to manage risk effectively. Existing monolithic models struggle to address both problems simultaneously.

The core design motivation of MARS is to replace explicit feature engineering with *behavioral diversity* across multiple agents, orchestrating a pool ranging from conservative to aggressive agents dynamically to handle diverse market environments.

## Method

### Overall Architecture

MARS adopts a two-level architecture:
- **Lower level**: Heterogeneous Agent Ensemble (HAE) — $N$ Safety-Critic agents with heterogeneous risk preferences
- **Upper level**: Meta-Adaptive Controller (MAC) — dynamically allocates weights across agents

At each timestep $t$, the market state $s_t$ (comprising current holdings, cash balance, and technical indicators) is fed simultaneously to both the MAC and the HAE. The MAC outputs a weight vector $w_t$; each agent in the HAE outputs an action $a_t^i$; the weighted aggregation then passes through a risk management override layer to produce the final executed action $A_t'$.

The problem is formulated as an MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:
- **State space**: Cash balance + per-asset holdings and feature vectors (price + four technical indicators: MACD, RSI, CCI, ADX)
- **Action space**: Continuous vector $A_t \in [-1,1]^D$ representing the target allocation change for each asset
- **Reward function**: $R_t = \frac{V_{t+1} - V_t}{V_t} - C_t - \rho_t$, where $\rho_t = w_{vol} \cdot \sigma_{30d} + w_{dd} \cdot DD_{30d}$ is a risk penalty based on 30-day rolling volatility and maximum drawdown

### Key Designs

1. **Heterogeneous Agent Ensemble (HAE)**: The ensemble $\mathcal{E} = \{\mathcal{A}_1, ..., \mathcal{A}_N\}$ consists of $N$ independent Safety-Critic agents. Each agent $\mathcal{A}_i$ is characterized by a unique risk parameter pair $(\theta_i, \lambda_i)$—where $\theta_i$ is the risk tolerance threshold and $\lambda_i$ is the risk-aversion penalty weight. Risk preferences are uniformly distributed from "ultra-conservative" to "highly aggressive," creating a behaviorally diverse pool of experts. Each agent is built on a DDPG architecture with three networks:

    - **Actor network**: The policy gradient incorporates a Conditional Safety Penalty (CSP):
      $$\nabla_{\phi_i} J(\phi_i) \approx \mathbb{E}[\nabla_{\phi_i} Q_{\psi_i}(s_t, \pi_{\phi_i}(s_t)) - \lambda_i \cdot \nabla_{\phi_i} \text{ReLU}(C_{\xi_i}(s_t, \pi_{\phi_i}(s_t)) - \theta_i)]$$
      The CSP term penalizes the policy only when the predicted risk exceeds the agent-specific threshold $\theta_i$.
    - **Critic network**: Trained with standard TD error to estimate the state-action value function.
    - **Safety-Critic network**: Predicts an external risk score for a given action. Its training target is the environment risk function $\mathcal{C}_{env}$, which synthesizes a $[0,1]$ risk score along three dimensions—portfolio concentration (30%), leverage ratio (30%), and simulated volatility (40%)—trained via MSE loss.

2. **Meta-Adaptive Controller (MAC)**: The MAC is a neural network $M_\omega$ that learns a meta-policy $\pi_\omega(w_t | s_t)$, dynamically allocating agent weights based on the current market state. Output weights are generated via softmax:
   $$w_t = \text{softmax}(M_\omega(s_t))$$
   The final action is a weighted average:
   $$A_t = \sum_{i=1}^{N} w_t^i \cdot \pi_{\phi_i}(s_t)$$

   The MAC is trained to maximize a risk-adjusted return objective:
   $$\mathcal{L}(\omega) = -\left(\frac{\mathbb{E}[\bar{Q}_t]}{\text{Std}(\bar{Q}_t) + \epsilon} - \lambda_{meta} \cdot \mathbb{E}[\bar{C}_t]\right)$$
   The first term resembles a Sharpe ratio (return-to-risk), while the second term penalizes the ensemble's predicted risk. This encourages the MAC to favor agent combinations that are high-return, low-risk, and stable.

3. **Risk Management Override Layer**: Serving as a final safety net, this layer enforces rule-based constraints—no single asset may exceed 20% of total portfolio value, a cash buffer must be maintained, and short selling is prohibited. Any non-compliant action is adjusted to a compliant action $A_t'$. Note that the Safety-Critic is a training-time module; at deployment, risk is managed entirely by MAC dynamic weighting and the rule-based layer.

### Loss & Training

- All networks are three-layer MLPs (256–128–64, ReLU activation)
- $N=10$ agents; $\theta_i$ ranges from 0.10 to 0.55; $\lambda_i$ ranges from 1.0 to 5.5
- Reward function parameters: $w_{vol}=0.5$, $w_{dd}=2.0$, $\gamma=0.99$
- MAC risk penalty parameter: $\lambda_{meta}=0.5$
- Per-asset feature vector dimension $K=5$ (price + 4 technical indicators)
- Random seed fixed at 42

Training procedure (Algorithm 1): At each episode, HAE agents propose actions → MAC computes weights → weighted aggregation → risk override → execution → experience collection → update each agent's Actor/Critic/Safety-Critic → periodic MAC update.

## Key Experimental Results

### Main Results

Experiments are conducted on DJI (Dow Jones, 50 U.S. stocks) and HSI (Hang Seng Index, 50 Hong Kong stocks), with two test periods: 2022 (bear market) and 2024 (bull market).

| Environment | Model | CR% | AR% | SR | AVol% | MDD% |
|---|---|---|---|---|---|---|
| DJI 2024 | **MARS** | **29.50** | **31.19** | **2.84** | 10.99 | **-5.39** |
| DJI 2024 | MARS-Static | 17.10 | 17.17 | 1.71 | 10.04 | -6.79 |
| DJI 2024 | DeepTrader | 13.30 | 14.01 | 1.18 | 11.92 | -6.84 |
| DJI 2024 | HRPM | 19.11 | 20.16 | 0.99 | 20.43 | -7.90 |
| DJI 2024 | DJI Index | 15.36 | 16.19 | 1.41 | 11.51 | -6.06 |
| DJI 2022 | **MARS** | **-0.86** | -0.93 | -0.05 | 19.83 | **-16.77** |
| DJI 2022 | DeepTrader | -10.70 | -11.43 | -0.46 | 25.07 | -21.32 |
| DJI 2022 | AlphaStock | -36.37 | -38.42 | -1.03 | 37.35 | -46.17 |
| HSI 2024 | **MARS** | 16.24 | 17.84 | **1.49** | 12.00 | -7.38 |
| HSI 2022 | **MARS** | **-14.50** | -14.88 | -0.66 | **22.56** | **-32.72** |
| HSI 2022 | DeepTrader | -26.69 | -27.34 | -0.86 | 31.93 | -48.02 |

Key result: MARS achieves relative Sharpe Ratio improvements of 70.6% and 101.4% over the best baseline on DJI 2022 and DJI 2024, respectively.

### Ablation Study

| Configuration | CR% | SR | MDD% | Description |
|---|---|---|---|---|
| MARS (full) | 29.50 | 2.84 | -5.39 | Complete framework |
| MARS-Static | 17.10 | 1.71 | -6.79 | MAC removed (uniform weights) |
| MARS-Homo | 22.21 | 1.85 | -7.81 | Agent heterogeneity removed (identical risk parameters) |
| MARS-Div5 | 12.02 | 1.08 | -6.19 | Only 5 agents |
| MARS-Div15 | 19.70 | 1.67 | -7.26 | 15 agents (diminishing returns) |

### Key Findings

1. **Bear-market capital preservation**: In 2022, MARS incurs only a 0.86% loss, while AlphaStock loses 36.37% with a drawdown of -46.17%.
2. **Adaptive strategy switching**: In 2022, MAC weights fluctuate significantly with frequent switching between conservative and aggressive agents; in 2024, weights stabilize and the conservative–aggressive negative correlation deepens from -0.788 to -0.968.
3. **Ensemble size**: 10 agents is optimal; 5 agents provide insufficient diversity, while 15 agents yield diminishing returns.
4. **MAC is critical**: Removing the MAC reduces the Sharpe Ratio from 2.84 to 1.71, demonstrating that dynamic orchestration is the core contributor.

## Highlights & Insights

- **Architectural innovation**: Risk management is elevated from a passive reward-penalty mechanism to an active paradigm of multi-agent behavioral diversity combined with meta-controller dynamic orchestration.
- **Unique role of the Safety-Critic**: It is used during training to shape each agent's risk preference, but is not employed at deployment—risk is entirely managed by the MAC and the rule-based layer.
- **Three-dimensional risk signal**: The environment risk function integrates portfolio concentration, leverage ratio, and simulated volatility, offering a more comprehensive signal than simple price-based penalties.
- **Elegant MAC training objective**: A Sharpe-ratio-like formulation enables the MAC to simultaneously pursue high returns and low risk.

## Limitations & Future Work

1. Validation is limited to equity markets; other asset classes (e.g., cryptocurrencies, foreign exchange, commodity futures) are not explored.
2. The environment risk function weights (40%–30%–30%) are fixed via sensitivity analysis and may not generalize to all market conditions.
3. Modeling of transaction costs and slippage is relatively simplified.
4. Only 50 stocks are selected; scalability to larger asset universes remains untested.
5. Comparisons with more recent DRL methods (e.g., Transformer-based policies, diffusion models) are absent.

## Related Work & Insights

- **DeepTrader** (Wang et al. 2021b): Dual-module architecture (market condition embedding + drawdown penalty); the primary direct baseline for MARS.
- **HRPM** (Wang et al. 2021a): Hierarchical RL framework in which a high-level agent sets strategic allocations and a low-level agent optimizes execution costs.
- **EarnHFT** (Qin et al. 2024): Three-level hierarchical high-frequency trading using a meta-controller to select the best expert agent—the closest conceptual precursor to MARS's MAC.
- **MAPS** (Lee et al. 2020): Multi-agent collaboration with a diversification penalty to encourage policy diversity—a key inspiration for MARS's heterogeneous design.

## Rating

- Novelty: ⭐⭐⭐⭐ (The two-level architecture is original; the combination of Safety-Critic and meta-controller is creative.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ablations, multi-market and multi-period validation, though baselines are somewhat limited.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; well-formatted equations.)
- Value: ⭐⭐⭐⭐ (Practically applicable in the financial RL domain.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[AAAI 2026\] BAMAS: Structuring Budget-Aware Multi-Agent Systems](bamas_structuring_budget-aware_multi-agent_systems.md)
- [\[AAAI 2026\] Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation](scalable_multi-objective_and_meta_reinforcement_learning_via_gradient_estimation.md)
- [\[AAAI 2026\] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)

</div>

<!-- RELATED:END -->
