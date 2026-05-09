---
title: >-
  [Paper Note] Routing, Cascades, and User Choice for LLMs
description: >-
  [ICLR 2026][Reinforcement Learning][LLM routing] This paper models LLM routing as a provider-user Stackelberg game, proves that the optimal routing policy is almost always a static, cascade-free threshold rule, reveals user-provider misalignment when quality/cost rankings are inconsistent, and shows that under low churn penalties providers are incentivized to inflate latency via throttling to reduce cost at the expense of user utility.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - LLM routing
  - cascading
  - Stackelberg game
  - user-provider misalignment
  - throttling
date: 2026-05-08
content_hash: 96c6a724b2a2367b
---

# Routing, Cascades, and User Choice for LLMs

**Conference**: ICLR 2026
**arXiv**: [2602.09902](https://arxiv.org/abs/2602.09902)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: LLM routing, cascading, Stackelberg game, user-provider misalignment, throttling

## TL;DR

This paper models LLM routing as a provider-user Stackelberg game, proves that the optimal routing policy is almost always a static, cascade-free threshold rule, reveals user-provider misalignment when quality/cost rankings are inconsistent, and shows that under low churn penalties providers are incentivized to inflate latency via throttling to reduce cost at the expense of user utility.

## Background & Motivation

**Background**: LLM providers allocate user tasks across heterogeneous models through routing and cascading strategies to balance quality, latency, and cost. GPT-5 explicitly adopts routing, switching between an "efficient model" and a "deep reasoning model."

**Limitations of Prior Work**: Existing routing algorithms (Ding et al., 2024; Dekoninck et al., 2025) focus on estimating LLM performance and optimizing quality-latency-cost trade-offs, but treat user response behavior as exogenous. However, the prompt-based interface of LLMs means users may interact repeatedly after a model failure, incurring repeated inference costs.

**Key Challenge**: Optimizing single-pass cost may backfire at the level of user behavior. Users may abandon tasks or cancel subscriptions depending on the value of the task and model latency. Optimizing single-pass cost may be "penny-wise but welfare-foolish."

**Goal**: The paper formalizes a two-level Stackelberg game—the provider selects a routing policy (initial model + cascade probability), and the user decides an abandonment probability based on the observed strategy. By fully characterizing the user's best response and simplifying the provider's problem, the paper derives concise threshold rules.

## Method

### Overall Architecture

Consider a single provider with two models $M_1$ (standard) and $M_2$ (reasoning), satisfying $t_1 < t_2$, $c_1 < c_2$, $0 < p_1 < p_2 < 1$. The provider selects a routing policy $(i, s)$: initial model $i$ and cascade probability $s$. The user selects an abandonment probability $q$.

Define the user's single-pass net value as $\xi_i := Vp_i - t_i$; when $\xi_i > 0$ the model is value-dominated, otherwise latency-dominated.

User utility is the success value minus cumulative latency:

$$U_i(s, q) = V \cdot S_i(s, q) - L_i(s, q)$$

Provider cost is the service cost plus user abandonment penalty:

$$J_i(s, q) = C_i(s, q) + P(1 - S_i(s, q))$$

### Key Design 1: Full Characterization of the User's Best Response

**Function**: Derive the user's optimal abandonment strategy given the provider's policy.

**Mechanism** (Theorems 1–2):
- If routed to $M_2$: $q^* = \mathbb{1}\{\xi_2 < 0\}$ (pure threshold rule)
- If routed to $M_1$ and $\xi_1, \xi_2$ share the same sign: user behavior is static (if both value-dominated, $q^*=0$; if both latency-dominated, $q^*=1$)
- If $\xi_1 < 0 < \xi_2$: there exists a threshold $s_0 = -\xi_1/(\xi_2/p_2 - \xi_1)$; the user stays when $s > s_0$ and abandons otherwise
- If $\xi_1 > 0 > \xi_2$: there exist two thresholds $s_L, s_H$; the user stays when $s \leq s_L$, abandons when $s \geq s_H$, and plays a mixed strategy in between

**Design Motivation**: User behavior is affected by the routing policy only when the two models are differentiated in value. When the two models are homogeneous, routing has no influence on user decisions.

### Key Design 2: Simplification of the Provider's Optimal Routing

**Function**: Reduce the provider's optimization problem to a single-variable problem and derive closed-form solutions.

**Mechanism** (Theorems 3–5):
- **Same-sign case** (Theorem 3): The optimal policy always routes to a single model with no cascading. When $\xi_1, \xi_2 > 0$, the choice is based on cost-of-pass $c_i/p_i$; when $\xi_1, \xi_2 < 0$, it depends on the comparison between penalty $P$ and the incremental cost-of-pass.
- **Differentiated case** (Theorems 4–5): In almost all regimes the optimal policy remains static, $(i^*, s^*) \in \{(1,0), (1,1), (2,0)\}$; cascading is optimal only in a narrow region.

**Design Motivation**: Cascading between undifferentiated models increases cost and variance without benefit. Cascading is valuable only when the two models have different net values and within specific parameter ranges.

### Misalignment and Throttling Analysis

**Provider-User Misalignment** (Proposition 1): When the provider's cost-of-pass ranking is inconsistent with the user's utility ranking, a misalignment gap $\Delta_U > 0$ arises—the provider's cost-optimal policy harms user utility.

**Throttling Risk** (Proposition 2): When the user churn penalty satisfies $P \leq \min\{c_1/p_1, c_2/p_2\}$, the provider is incentivized to artificially inflate latency $\hat{t}_i > Vp_i$, making both models latency-dominated and encouraging users to abandon tasks in order to reduce service cost. Under this condition, user utility is maximally damaged.

## Key Experimental Results

### Main Results: Region Decomposition of the Provider's Optimal Policy

| State of $\xi_1, \xi_2$ | User Behavior | Provider's Optimal Policy | Is Cascading Effective? |
|--------------------------|---------------|--------------------------|------------------------|
| Both value-dominated | Statically stays | Route by $c_i/p_i$, no cascading | Ineffective |
| Both latency-dominated | Statically abandons | Depends on $P$ vs $(c_2-c_1)/(p_2-p_1)$ | Ineffective |
| $\xi_1 < 0 < \xi_2$ | Depends on cascade probability | Almost always routes to $M_1$ unless cost-of-pass gap is large | Only under specific conditions |
| $\xi_1 > 0 > \xi_2$ | Three-phase response | Primarily static; mixed strategy in a very narrow interval | Rarely |

### Ablation Study: Throttling Gains

| Configuration | Effect | Condition |
|---------------|--------|-----------|
| $P < \min\{c_1/p_1, c_2/p_2\}$ | Throttling benefits the provider | Low churn penalty |
| $P > \min\{c_1/p_1, c_2/p_2\}$ | Throttling instead increases provider cost | High churn penalty |
| Throttling gain region | Linear in $P$ | User unsubscription can prevent throttling |

### Key Findings

- Optimal routing degenerates to a simple threshold rule across the vast majority of parameter regimes; the value of cascading is extremely limited.
- User behavior is affected by the routing policy only when the two models are differentiated.
- When user and provider model rankings are inconsistent, misalignment is inevitable.
- The key to preventing throttling is ensuring that the cost of user abandonment (churn penalty) is sufficiently high—users should have the right to unsubscribe.

## Highlights & Insights

- This is the first work to elevate LLM routing from a purely engineering optimization to a game-theoretic framework that accounts for user response behavior.
- The theoretical results offer highly practical guidance: the finding that cascading is rarely optimal has direct implications for the design of routing systems such as GPT-5.
- The throttling analysis reveals moral hazard on the part of providers in LLM subscription models, with implications for policy.
- The paper itself was completed with LLM assistance (documented in detail in Appendix A), constituting a meta-level self-consistent validation.

## Limitations & Future Work

- Only the two-model case is analyzed; real deployments may involve routing across a larger number of models.
- Users are assumed to observe the provider's cascade strategy and to adopt stationary abandonment policies; in practice, routing strategies are opaque to users.
- The per-pass success probability is assumed to be i.i.d., ignoring the effect of user feedback on subsequent attempts.
- The analysis is confined to a fixed subscription pricing framework and does not consider per-call API pricing.
- Empirical experiments to validate the theoretical predictions are absent.

## Related Work & Insights

- **FrugalGPT** (Chen et al., 2023) and **RouteLLM** (Ong et al., 2025): These works focus on routing algorithm design; this paper complements them by incorporating the user behavior dimension from a game-theoretic perspective.
- **Cost-of-Pass** (Mahmood 2024; Erol et al., 2025): This paper directly adopts the cost-of-pass concept as a central metric for routing decisions.
- Implications for LLM subscription service design: allowing users to opt out of routing (opt-out routing) should be considered as a mechanism to prevent throttling.

## Rating

- Novelty: ⭐⭐⭐⭐ Analyzing LLM routing from a game-theoretic perspective is highly novel and fills a gap in user behavior modeling.
- Experimental Thoroughness: ⭐⭐⭐ A purely theoretical work; supported by theorem proofs and visualizations but lacking empirical experiments.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; the guideline summary in Figure 1 is highly practical.
- Value: ⭐⭐⭐⭐ Offers direct practical guidance for LLM service pricing and routing policy design; the throttling analysis carries policy significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/router-r1_teaching_llms_multi-round_routing_and_aggregation_via_reinforcement_le.md)
- [\[ICLR 2026\] ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning](remix_reinforcement_routing_lora.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [\[ICLR 2026\] Reasoning Boosts Opinion Alignment in LLMs](reasoning_boosts_opinion_alignment_in_llms.md)
- [\[ICLR 2026\] How LLMs Learn to Reason: A Complex Network Perspective](how_llms_learn_to_reason_a_complex_network_perspective.md)

</div>

<!-- RELATED:END -->
