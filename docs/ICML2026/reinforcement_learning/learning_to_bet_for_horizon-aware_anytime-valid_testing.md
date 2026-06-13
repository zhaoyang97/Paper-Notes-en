---
title: >-
  [Paper Note] Learning to Bet for Horizon-Aware Anytime-Valid Testing
description: >-
  [ICML 2026][Reinforcement Learning][Test Martingales] This paper reformulates the design of anytime-valid sequential tests under a strict observation limit $N$ as a finite-horizon optimal control problem with state space…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Test Martingales"
  - "Kelly Betting"
  - "Confidence Sequences"
  - "Phase Portrait"
  - "DQN"
  - "Finite-Horizon Decision Making"
date: 2026-05-08
content_hash: eb409cc0376dbda2
---

# Learning to Bet for Horizon-Aware Anytime-Valid Testing

**Conference**: ICML 2026  
**arXiv**: [2603.19551](https://arxiv.org/abs/2603.19551)  
**Code**: https://github.com/egetaga/learning-to-bet (Available)  
**Area**: Sequential Hypothesis Testing / Anytime-Valid Inference / Finite-Horizon Optimal Control  
**Keywords**: Test Martingales, Kelly Betting, Confidence Sequences, Phase Portrait, DQN, Finite-Horizon Decision Making  

## TL;DR
This paper reformulates the design of anytime-valid sequential tests under a strict observation limit $N$ as a finite-horizon optimal control problem with state space $(t, \log W_t)$. It theoretically proves a three-zone "phase portrait" where Kelly betting is optimal in an "on-schedule" middle band, aggressive betting is required when trailing, and conservative betting is preferred when ahead. A unified DQN agent, trained on various synthetic Beta distributions, automatically learns state-dependent policies consistent with this phase portrait. It achieves higher rejection rates within deadlines and narrower confidence sequences on both synthetic and real data while maintaining anytime-validity via Ville’s inequality.

## Background & Motivation
**Background**: The framework of e-processes and test martingales based on "testing by betting" (Shafer 2019/2021; Waudby 2024) has become the mainstream for constructing power-one tests and confidence sequences. Given $H_0: \mu_X = m$, a predictable bet $\lambda_n(m) \in [-1/(1-m), 1/m]$ is defined such that the wealth process $W_n(m) = W_{n-1}(m)(1 + \lambda_n(m)(X_n - m))$ is a non-negative martingale under $H_0$. By Ville’s inequality $\mathbb P(\exists n: W_n \ge 1/\alpha) \le \alpha$, it follows that $\tau_m = \inf\{n: W_n \ge 1/\alpha\}$ is a level-$\alpha$ stopping time. PrPlEB (Waudby 2024), universal portfolios (Orabona 2023), and STaR-Bets (Voravcek 2025) all refine betting strategies within this framework.

**Limitations of Prior Work**: Most mainstream works assume an infinite stream of observations $\{X_n\}$, but real-world scenarios (online A/B testing, adaptive experiments, resource-constrained research) almost always have a hard deadline $N$. Pure anytime strategies in the style of Doob/Robbins remain too conservative for large $N$, while tests tuned for a fixed $N$ violate type-I error under continuous monitoring. Neither approach directly addresses "continuous monitoring with a deadline."

**Key Challenge**: To maximize rejection probability within the deadline $N$, the required "drift per step" $(b - \log W_t)/(N - t)$ is non-stationary and changes dynamically with current progress and remaining time. However, to maintain anytime-validity, the bet must remain $\mathcal F_{t-1}$-measurable and satisfy the martingale constraint $\lambda \in \Lambda_m$. The only previous work addressing this, STaR-Bets (Voravcek 2025), uses a STaR (Sequential Target-Recalculating) heuristic without optimality characterization or adaptation to distribution shapes.

**Goal**: (1) Formulate "horizon-aware betting" as an optimal control (Dynamic Programming, DP) problem; (2) Provide provable sufficient conditions for when to deviate from Kelly betting and in which direction; (3) Translate this theory into a distribution-agnostic betting strategy that can be learned online.

**Key Insight**: The Bellman state is fully determined by $(t, \log W_t)$, and the action is $\lambda$. This is a typical finite-horizon discrete DP, albeit with a continuous action space and an unknown transition distribution $P_X$. Local analysis of the DP allows partitioning the state plane into a "phase portrait" (Kelly/Aggressive/Conservative zones), which a DQN can then implement by absorbing distribution information from empirical features.

**Core Idea**: Treat anytime-valid testing as an optimal control problem to characterize the phase portrait, then use a cross-distribution DQN to implement the portrait as an executable policy, combining theoretical guarantees with learned strategies.

## Method
The authors first formalize the problem using Bellman recursion, prove three complementary "phase portrait theorems," and then constrain the action space to $\{\widehat\lambda_t/2, \widehat\lambda_t, \lambda_{\max}\}$ to train a universal DQN. Crucially, policy learning is transparent to anytime-validity: validity only requires $\lambda_t \in \Lambda_m$ and predictability, regardless of how the policy is derived.

### Overall Architecture
- **DP State**: $(t, \log W_t)$, with termination conditions $W_t \ge 1/\alpha$ or $t = N$.
- **Action**: $\lambda_t(m) \in \Lambda_m = [-1/(1-m), 1/m]$; discretized into three levels: $\{\widehat\lambda_t/2, \widehat\lambda_t, \lambda_{\max}\}$ ("Half-Kelly / Kelly / All-in").
- **Reward**: $R = \mathbb I\{\max_{1 \le t \le N} \log W_t \ge \log(1/\alpha)\}$ (sparse terminal reward), where $\mathbb E[R] = \mathbb P(\tau_m \le N)$ directly equals the rejection rate within the deadline.
- **Bellman Recursion**: $V_t(y) = \max_{\lambda \in \Lambda_m} \mathbb E_{X \sim P_X} \big[\mathbb I\{y + h_m(\lambda, X) \ge b\} + \mathbb I\{y + h_m(\lambda, X) < b\} V_{t+1}(y + h_m(\lambda, X))\big]$, where $h_m(\lambda, x) = \log(1 + \lambda(x - m))$ and $b = \log(1/\alpha)$.
- **Confidence Sequences**: $C_n = \{m \in [0, 1]: W_n(m) < 1/\alpha\}$, derived automatically from the horizon-aware coverage guarantee of each $\tau_m$.

### Key Designs

1. **$(t, \log W_t)$ Plane Phase Portrait and Three Sufficient Conditions**:
    - **Function**: Characterizes when to deviate from Kelly betting at the optimal control level, turning vague "schedule-based adjustment" intuitions into provable propositions.
    - **Mechanism**: Let $T = N - t$ and $\Delta = TL_{\max} - (b - y)$ (where $L_{\max} = \mathbb E[h_m(\lambda_m^{\text{Kelly}}, X)]$) represent the "surplus drift" relative to Kelly. **Theorem 3.1 (Central Band)**: If $\Delta \ge B\sqrt{8T\log T}$, pure Kelly hits the threshold with probability $\ge 1 - 1/T$. Conversely, if the policy deviates from Kelly by $\ge \delta$ for a proportion $\rho$ of time and $\Delta \le \rho\epsilon T - B\sqrt{8T\log 2}$, the rejection probability is $\le 1/2$. **Proposition 3.4 (Trailing Aggression)**: When $r = (b - y)/T > \max\{L_{\max}, B_K/2\}$ (Kelly drift is insufficient), if an aggressive $\lambda^{\text{agg}} > \lambda^{\text{Kelly}}$ satisfies a specific KL rate condition, its rejection probability strictly exceeds Kelly's. **Proposition 3.6 (Leading Conservatism)**: When $B_K/2 < r < L_{\max}$, a conservative $\lambda^{\text{def}} < \lambda^{\text{Kelly}}$ exists with a lower failure probability than Kelly.
    - **Design Motivation**: Provides a provable structure for the optimal policy—Central Kelly zone + Trailing Aggressive zone + Leading Conservative zone—justifying the DQN action set.

2. **Oracle Phase Portrait (DP Backward Induction)**:
    - **Function**: Computes "theoretical optimal actions" on synthetic distributions where $P_X$ is known to serve as ground truth.
    - **Mechanism**: Discretizes the $(t, \log W_t)$ plane and performs Bellman backward induction for $V_t(y)$. Results confirm the three-zone structure. As the problem difficulty increases ($m$ closer to $\mu_X$), the conservative zone disappears, the Kelly zone narrows, and the aggressive zone expands, matching theory.
    - **Design Motivation**: Validates the theoretical shape and provides a visual benchmark for the DQN "modal action map."

3. **Cross-Distribution Universal DQN Betting Agent**:
    - **Mechanism**: Each test is an episode of length $\le N$ with reward $R = \mathbb I\{\max_t \log W_t \ge b\}$. State features are $\mathcal F_{t-1}$-measurable vectors (empirical moments, time left $N-t$, distance to threshold $b - \log W_t$, $m$, empirical Kelly $\widehat\lambda_t(m)$, etc.). Actions are discretized into three levels: $\{\widehat\lambda_t/2, \widehat\lambda_t, \lambda_{\max}\}$. Trained once on 500,000 synthetic episodes (Beta/Beta-mixtures) across randomized $(\mu_X, m, N)$.
    - **Design Motivation**: Since the phase portrait boundaries depend on unknown $P_X$, the DQN learns to internalize distribution sensitivity into its policy. Validity is guaranteed by Ville’s inequality, remaining agnostic to the policy's source.

### Loss & Training
Standard DQN (Mnih 2015) with sparse terminal reward $R \in \{0, 1\}$. Two reward shaping variants were explored: DQN-EB (reward $= 1 + (1 - t/N)$) and DQN-U ($= 1 - t/N$) to encourage earlier rejections.

## Key Experimental Results

### Main Results
Comparison with STaR-Bets/STaR-Hoeffding (Voravcek 2025) and PrPlEB (Waudby 2024).

| Setting | Metric | DQN | STaR-Bets | PrPlEB |
|------|------|------|------|------|
| Beta-mix conc=6, $N=100$, $m=0.45$ | $\mathbb P(\tau \le N)$ | **Highest** | Second | Lowest |
| Beta-mix conc=1, $N=100$, $m=0.45$ | $\mathbb P(\tau \le N)$ | **Highest** | Second | Lowest |
| Three Beta-mix types, $N=100$ | $C_N$ Width | **Narrowest** | Mid | Widest |
| Real Data (DNA & Humidity, 6 sources) | $\mathbb P(\text{reject})$ | **5/6 Highest** | — | — |

The DQN, trained only on synthetic Beta data, maintains lead performance on OOD logit-normal, Bernoulli, and real-world datasets while keeping type-I error calibrated.

### Ablation Study

| Configuration | Key Observation |
|------|------|
| 3 actions vs 9 actions | Similar performance; 3 actions aligned with theory are sufficient. |
| DQN (Original Reward) | Strongest terminal power; directly optimizes $\mathbb P(\tau \le N)$. |
| DQN-EB ($1 + (1 - t/N)$) | Earlier rejections at the cost of slight terminal power. |
| $N \in \{250, 300, 350\}$ (Unseen) | Generalizes well to longer horizons without retraining. |
| $\alpha = 0.01$ (Retrained) | Successfully improves power; $\alpha$ requires specific training. |

### Key Findings
- The DQN's "modal action map" visually replicates the oracle phase portrait: central Kelly band, upper half-Kelly band, and lower all-in band.
- Early conservatism is correct but counter-intuitive: because $\widehat\lambda_t(m)$ has high variance at small $t$, the DQN avoids aggressive bets that could lead to irrecoverable wealth loss due to estimation noise.
- Cross-distribution transfer is successful: training on Beta generalizes to various OOD families and real data.

## Highlights & Insights
- The "optimal control + DQN" bridge is elegant: all statistical validity comes from the martingale constraint, allowing the policy to be a complex black box without compromising rigor.
- Theoretical guidance (Phase Portrait Theorems) simplifies the RL design (action space discretization).
- Generalization suggests that using geometric features ($t/N$, distance to threshold) captures distributionally robust patterns.

## Limitations & Future Work
- **OOD Sensitivity**: Performance on heavy-tailed or highly discrete distributions remains an open question.
- **Action Space**: Limited to 3 levels; continuous actions or asymmetric negative betting ($\lambda < 0$) could further tighten confidence bounds.
- **Accountability**: DQN is a black box, which may require an additional explanation layer for regulated environments (e.g., clinical trials).
- **Alpha Dependency**: A different $\alpha$ requires retraining; future work could treat $\alpha$ as a state input.
- **Scope**: Current work covers bounded $[0, 1]$ means; extending this to variances, quantiles, or CATE requires new control-theoretic analysis.

## Related Work & Insights
- **vs. STaR-Bets (Voravcek 2025)**: The DQN approach provides a more formal optimality framework and better cross-distribution performance than the STaR heuristic.
- **vs. PrPlEB (Waudby 2024)**: While reusing the same e-process framework, this work corrects PrPlEB's inherent conservatism in finite-horizon scenarios.
- **vs. Universal Portfolios (Orabona 2023)**: Universal portfolios provide strong regret guarantees for anytime-tightness, whereas DQN is optimized for maximizing power within a specific deadline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Offline Reinforcement Learning with Universal Horizon Models](offline_reinforcement_learning_with_universal_horizon_models.md)
- [\[ICML 2026\] InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](inftythink_effective_and_efficient_infinite-horizon_reasoning_via_reinforcement_.md)
- [\[ICML 2026\] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)
- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)
- [\[ICML 2026\] Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory](learning_query-aware_budget-tier_routing_for_runtime_agent_memory.md)

</div>

<!-- RELATED:END -->
