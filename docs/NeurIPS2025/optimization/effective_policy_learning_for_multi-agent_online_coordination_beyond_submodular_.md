---
title: >-
  [Paper Note] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives
description: >-
  [NeurIPS 2025][Optimization][multi-agent coordination] This paper proposes two multi-agent online coordination algorithms, MA-SPL and MA-MPL, which leverage a *policy-based continuous extension* technique to surpass the limitations of submodularity. For the first time, both algorithms achieve the optimal $(1 - c/e)$ approximation ratio under submodular and weakly submodular objectives, while supporting time-varying objectives and the practical constraint of local-only feedback.
tags:
  - NeurIPS 2025
  - Optimization
  - multi-agent coordination
  - submodular
  - weakly submodular
  - online optimization
  - policy learning
date: 2026-05-08
content_hash: 29b00708ad379787
---

# Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives

**Conference**: NeurIPS 2025
**arXiv**: [2509.22596](https://arxiv.org/abs/2509.22596)
**Code**: None
**Area**: Optimization
**Keywords**: multi-agent coordination, submodular, weakly submodular, online optimization, policy learning

## TL;DR
This paper proposes two multi-agent online coordination algorithms, MA-SPL and MA-MPL, which leverage a *policy-based continuous extension* technique to surpass the limitations of submodularity. For the first time, both algorithms achieve the optimal $(1 - c/e)$ approximation ratio under submodular and weakly submodular objectives, while supporting time-varying objectives and the practical constraint of local-only feedback.

## Background & Motivation

### State of the Field

**State of the Field**: Multi-agent online coordination (MA-OC) requires multiple distributed agents to collaboratively maximize time-varying set-function objectives (e.g., UAV target tracking, area monitoring) in an online manner. Existing methods are mostly restricted to submodular objective functions.

**Limitations of Prior Work**: (a) The standard multilinear extension guarantees lossless rounding only for submodular functions and fails for non-submodular ones; (b) weakly submodular and $\alpha$-weakly DR-submodular functions have never been studied in online settings.

**Root Cause**: How can provable approximation guarantees be achieved for objectives beyond submodularity under decentralized, local-feedback-only constraints?

**Starting Point**: Design a *policy-based continuous extension* that guarantees lossless rounding for arbitrary set functions (not limited to submodular ones), bypassing the dependence of the multilinear extension on submodularity.

**Core Idea**: Optimize policy parameters in continuous space; convert discrete optimization into continuous optimization via a novel continuous extension, then recover discrete solutions through lossless rounding.

## Method

### Overall Architecture
- **Problem**: $n$ agents each select an action $a_i \in \mathcal{A}_i$ to jointly optimize the time-varying objective $f_t(S)$, where $S = \{a_1, \ldots, a_n\}$.
- **Constraints**: Each agent can only query the marginal contribution of its own action (local feedback) and exchanges information over a communication graph.
- **Two algorithms**: MA-SPL (requires problem parameters) and MA-MPL (parameter-free).

### Key Designs

1. **Policy-based Continuous Extension**

   - **Function**: Converts discrete set-function optimization into continuous policy-parameter optimization.
   - **Mechanism**: Each agent maintains an action probability distribution $\pi_i$; the continuous extension is defined as $\bar{f}(\pi) = \mathbb{E}_{S \sim \pi}[f(S)]$, which guarantees lossless rounding for arbitrary set functions (not limited to submodular ones).
   - **Key Breakthrough**: The lossless rounding of the multilinear extension holds only for submodular functions, whereas the proposed continuous extension holds for arbitrary set functions.

2. **MA-SPL (Single-loop Policy Learning)**

   - Approximation ratio: $(1 - c/e)$ for submodular; $\alpha(1 - 1/e)$ for $\alpha$-weakly DR-submodular; $(1 - e^{-\gamma})\beta$ for $(\gamma, \beta)$-weakly submodular.
   - Dynamic regret: $O(\sqrt{P_T \cdot T / (1 - \tau)})$.
   - Requires knowledge of problem parameters such as $\alpha$ or $(\gamma, \beta)$.

3. **MA-MPL (Multi-loop Parameter-free Policy Learning)**

   - Approximation ratio: identical to MA-SPL.
   - Dynamic regret: $O(\sqrt{P_T \cdot T})$, dependent on the diameter of the communication graph.
   - Requires no problem parameters — adapts automatically via multi-loop search.

### Loss & Training
- Online Frank-Wolfe / projected gradient descent in the policy space.
- Communication: agents propagate policy updates over the graph.
- Feedback: each agent observes only the marginal contribution of its own action, $f_t(S) - f_t(S \setminus \{a_i\})$.

## Key Experimental Results

### Main Results (UAV Target Tracking + Area Monitoring)

| Method | Submodular Approx. Ratio | Weakly Submodular Support | Parameter Dependency |
|--------|--------------------------|---------------------------|----------------------|
| Prev. SOTA | $(1/2)$ or $(1 - 1/e)$ (submodular only) | ✗ | — |
| **MA-SPL** | $(1 - c/e)$ | ✓ | Required |
| **MA-MPL** | $(1 - c/e)$ | ✓ | Not required |

### Key Findings
- The first work to handle weakly submodular objectives in a multi-agent online setting.
- MA-SPL and MA-MPL significantly outperform greedy-based baselines in UAV tracking and area monitoring experiments.
- The $(1 - c/e)$ approximation ratio matches the optimal bound for the single-agent setting.

## Highlights & Insights
- **Breakthrough via continuous extension**: The policy-based continuous extension guarantees lossless rounding for arbitrary set functions, representing the core technical innovation.
- **First treatment of weakly submodular objectives in online multi-agent settings**: This had not been studied even in single-agent online settings prior to this work.
- **Practicality of the parameter-free MA-MPL**: It requires no knowledge of the submodularity parameters of the objective function, making it suitable for black-box optimization scenarios.

## Limitations & Future Work
- **Communication overhead**: The regret of MA-MPL scales with the diameter of the communication graph, which may limit efficiency on large sparse graphs.
- **Discrete action space assumption**: Continuous action spaces require discretization, which may incur precision loss.
- **Static communication graph**: Time-varying communication topologies are not considered.
- **Limited experimental scale**: The UAV tracking and area monitoring scenarios are relatively simple.

## Related Work & Insights
- **vs. Chen et al. (2018, single-agent online submodular)**: Their work handles only submodular objectives; this paper extends to weakly submodular objectives.
- **vs. Grimsman et al. (2018, multi-agent offline submodular)**: Their work addresses offline settings; this paper handles online time-varying objectives.
- **vs. standard multilinear extension methods**: The lossless rounding of the multilinear extension is restricted to submodular functions; the proposed policy-based extension applies to arbitrary set functions.

## Rating
- Novelty: ⭐⭐⭐⭐ The policy-based continuous extension is a key innovation; the extension to weakly submodular objectives carries significant theoretical importance.
- Experimental Thoroughness: ⭐⭐⭐ UAV and area monitoring experiments are included but limited in scale.
- Writing Quality: ⭐⭐⭐⭐ Theory is presented clearly; the comparison between the two algorithms is well organized.
- Value: ⭐⭐⭐⭐ A significant contribution to the distributed optimization and multi-agent coordination communities.

## Additional Notes
- The theoretical analysis framework and technical tools developed in this paper also offer insights for adjacent research areas.
- The core contribution lies in a deep theoretical understanding that provides a foundation for subsequent practical optimization.
- The paper is methodologically complementary to other NeurIPS 2025 papers published concurrently.
- The paper's exposition of problem motivation and technical approach is worth studying.
- Readers are encouraged to consult the appendix for more complete experimental details and proofs.

## Further Reading
- This research direction is closely related to several active topics in the AI community.
- The rigor of the theoretical results provides a solid mathematical foundation for subsequent empirical research.
- The methodology can be extended to broader problem settings.
- Follow-up work from the same group is worth monitoring.
- For beginners in theoretical research, the proof sketch section provides an excellent technical roadmap.
- From a methodological perspective, this paper demonstrates how careful mathematical modeling can reduce complex problems to analyzable frameworks.

## Technical Remarks
- The proofs of the core theorems rely on multi-step concentration inequality analysis and properties of the Bellman equation.
- The choice of pessimistic/optimistic principles in algorithm design is a central consideration in offline RL theory.
- Logarithmic factors hidden in the theoretical bounds are non-negligible in practical applications.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](online_two-stage_submodular_maximization.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[AAAI 2026\] Parametrized Multi-Agent Routing via Deep Attention Models](../../AAAI2026/optimization/parametrized_multi-agent_routing_via_deep_attention_models.md)
- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)

<!-- RELATED:END -->
