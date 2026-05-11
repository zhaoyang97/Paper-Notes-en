---
title: >-
  [Paper Note] A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind
description: >-
  [AAAI 2026][Robotics][Theory of Mind] This paper proposes a game-theoretic framework based on Poisson cognitive hierarchy, achieving computable multi-agent Theory of Mind via Gamma-Poisson conjugate Bayesian updates. The…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Theory of Mind"
  - "Cognitive Hierarchy"
  - "Game Theory"
  - "Bayesian Inference"
  - "Gamma-Poisson Conjugate"
  - "Stochastic Games"
date: 2026-05-08
content_hash: cf14672c8c4cc9c3
---

# A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind

**Conference**: AAAI 2026  
**arXiv**: [2511.22536](https://arxiv.org/abs/2511.22536)  
**Authors**: Fengming Zhu, Yuxin Pan, Xiaomeng Zhu, Fangzhen Lin  
**Code**: None  
**Area**: Robotics  
**Keywords**: Theory of Mind, Cognitive Hierarchy, Game Theory, Bayesian Inference, Gamma-Poisson Conjugate, Stochastic Games

## TL;DR

This paper proposes a game-theoretic framework based on Poisson cognitive hierarchy, achieving computable multi-agent Theory of Mind via Gamma-Poisson conjugate Bayesian updates. The framework supports recursive bounded-rationality decision-making and online belief revision while avoiding the undecidability of POMDPs.

## Background & Motivation

**Interdisciplinary Demand for Theory of Mind**: ToM originates from psychology (Premack & Woodruff 1978) and studies agents' ability to infer others' goals, intentions, and beliefs. It is widely needed in logic, economics, and robotics, yet psychological research lacks a formalizable, automatable computational framework.

**Limitations of Logic-Based Formalizations**: In classical automated planning, goals are represented as sets of fluents and beliefs are modeled as revisable mental states, with extensions to multi-agent settings (e.g., ConGolog, higher-order belief revision). However, these logical paradigms struggle to capture utilitarian behavior — agents should maximize cumulative reward rather than merely satisfy logical constraints.

**Advantages of the Decision-Theoretic/Game-Theoretic Perspective**: Game-theoretic models naturally support (a) utilitarian best response — agents simultaneously optimize against the environment and other agents; and (b) seamless integration with statistical learning and modern ML techniques for scaling to large domains.

**Computability Issues with I-POMDP**: Interactive POMDP (Gmytrasiewicz & Doshi 2005) is a classical framework for multi-agent ToM, but its recursive belief space renders solving undecidable in practice.

**Shortcomings of Existing Cognitive Hierarchy Approaches**: GR2 (Wen et al. 2021) embeds Poisson hierarchy into RL training, but (a) does not update beliefs at execution time — making online adaptation to opponent changes impossible; and (b) computes only action-wise best responses rather than strategy-wise ones.

**Lack of a Unified Computable ToM Framework**: Existing deep learning-based ToM work (e.g., Machine ToM, MMToM-QA) focuses on evaluating LLMs' ToM capabilities or single-agent domains, without proposing a conceptually clear and computable formal framework for multi-agent settings.

## Method

### Overall Architecture

The system is modeled as a **Stochastic Game** $\langle \mathcal{N}, \mathcal{S}, \mathcal{A}, T, R \rangle$:

- $\mathcal{N}$: finite set of $n$ agents
- $\mathcal{S}$: finite set of environment states
- $\mathcal{A} = \mathcal{A}_1 \times \cdots \times \mathcal{A}_n$: joint action space
- $T: \mathcal{S} \times \mathcal{A} \mapsto \Delta(\mathcal{S})$: stochastic state transition
- $R_i: \mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}$: immediate reward for agent $i$

Each agent maximizes cumulative discounted reward $\mathbb{E}[\sum_t \gamma^t R_{i,t}]$, with policy $\pi_i: \mathcal{S} \mapsto \Delta(\mathcal{A}_i)$ serving as the formal notion of "intention." Goals are subsumed by the reward structure, and beliefs are modeled via the cognitive hierarchy.

### Key Designs

**1. Poisson Cognitive Hierarchy Construction**

The belief structure adopts a Poisson($\lambda$) cognitive hierarchy, constructing strategies bottom-up:

- **Level-0**: Random policy or simple rule, $\pi_j|_0(a_j|S) \sim \text{Unif}(\mathcal{A}_j)$
- **Level-$(k+1)$**: Best response to strategies at level $k$ and below

The framework provides two implementations:

| | Implementation 1: Singleton BR | Implementation 2: Mixed BR |
|---|---|---|
| **Assumption** | Opponents all at level-$k$ | Opponents distributed over levels $0 \sim k$ via truncated Poisson |
| **Formula** | $\pi_j\|_{k+1} \in BR(\pi_{-j}\|_k)$ | $\pi_j\|_{k+1} \in BR(\pi_{-j}^{mixed}\|_k)$ |
| **Solving** | Solve induced MDP $\mathcal{M}(\pi_{-j}\|_k)$ | Requires QMDP approximation to avoid POMDP undecidability |
| **Initial Cost** | $\Theta(K_j)$ MDPs | $\Theta(K_j)$ MDPs |
| **Update Cost** | Only belief distribution updated; supporting policies unchanged | Requires re-solving each round (BR changes as $g_\iota$ changes) |
| **Property** | Closed under stationary policies | Closed under stationary policies |

**2. Gamma-Poisson Conjugate Bayesian Update**

The mean reasoning level $\Lambda$ is given a prior $\text{Gamma}(a, b)$. After observing the opponent's reasoning levels $(k_1, \ldots, k_m)$ over $m$ interaction rounds, the posterior updates to:

$$\Lambda | (k_1, \ldots, k_m) \sim \text{Gamma}\left(a + \sum_r k_r, \ b + m\right)$$

Optimal estimate: $\lambda' = \frac{a + \sum_r k_r}{b + m}$

This conjugate structure reduces belief updates to maintaining two scalar parameters $(a, b)$, incurring negligible computational cost.

**3. QMDP Approximation for Guaranteed Computability**

In Implementation 2, $BR(\pi_{-j}^{mixed}|_k)$ in principle requires solving a POMDP (which is undecidable). The paper adopts a QMDP approximation:

$$\pi_j|_{k+1}(S) \in \arg\max_{a_j} \sum_{\iota=0}^{k} g_\iota \cdot Q^*_{\mathcal{M}(\pi_{-j}|\iota)}(S, a_j)$$

where $g_\iota$ are the conditional probability weights of the truncated Poisson distribution and $Q^*$ denotes the optimal Q-function of each induced MDP. This approximation reduces the POMDP to a weighted combination of Q-values from multiple MDPs.

**4. Complete Algorithm**

1. Initialize prior $\text{Gamma}(a, b)$
2. Estimate $\lambda$ and construct strategies at each level
3. Compute the agent's own best-response policy
4. Observe opponent behavior and update parameters $(a, b)$
5. Return to step 2 and iterate

### Relationship to I-POMDP

The proposed framework can be viewed as a computable instantiation of I-POMDP: the Poisson hierarchy constrains the recursive belief space, QMDP approximation ensures tractability, and the core capability for recursive multi-agent reasoning is preserved.

## Key Experimental Results

This paper is a **purely theoretical work** with no empirical evaluation. The authors explicitly state: "due to the page limit, we will focus on elaborating our proposed theoretic framework, leaving experiments to future work," and note plans for experimental validation on human-robot cohabiting systems.

| Aspect | Details |
|---|---|
| Experimental Data | None |
| Baseline Comparison | None |
| Evaluation Metrics | Not defined |
| Target Domain | Human-robot cohabiting systems (planned) |

| Theoretical Comparison | Ours | GR2 (Wen et al. 2021) | I-POMDP |
|---|---|---|---|
| Online Belief Update | ✓ Gamma-Poisson | ✗ Fixed after training | ✓ But intractable |
| Best Response Type | Strategy-wise | Action-wise | Strategy-wise |
| Computability | ✓ QMDP approximation | ✓ | ✗ Generally undecidable |
| Recursive Reasoning | ✓ Hierarchical | ✓ Hierarchical | ✓ Infinite recursion |

## Key Findings

- The Poisson cognitive hierarchy and Gamma prior form a conjugate pair, reducing belief updates from complex posterior inference to incremental updates of two scalars.
- The two BR implementations (singleton vs. mixed) reflect a precision-efficiency trade-off: Implementation 1 incurs low update cost (supporting policies unchanged), while Implementation 2 is more realistic but requires re-computation each round.
- Belief updates exhibit **non-monotonic, non-convergent** behavior — when beliefs deviate from the true distribution, they can be rapidly corrected, consistent with real interaction scenarios.
- The framework is closed over the space of stationary policies: if level-0 policies are stationary, all higher-level policies are also stationary.

## Highlights & Insights

- **Elegant Mathematical Architecture**: Systematically maps core ToM concepts (goals→rewards, intentions→policies, beliefs→cognitive hierarchy) into a game-theoretic framework with conceptual clarity and formal rigor.
- **Clever Use of Gamma-Poisson Conjugacy**: Avoids the high computational cost of general Bayesian inference; belief updates cost O(1).
- **Computability Guarantee**: Elegantly sidesteps POMDP undecidability via QMDP approximation, keeping recursive reasoning within tractable bounds.
- **Generality of Two Implementations**: Singleton BR and mixed BR each suit different application requirements, giving the framework practical flexibility.
- **Theoretical Bridge**: Unifies two research lines — logic-based formalization (automated planning) and decision-theoretic approaches (MDP/game theory).

## Limitations & Future Work

1. **Complete Absence of Experimental Validation**: For an AAAI paper, the lack of any experiment is a significant weakness; the practical effectiveness of the framework remains unknown.
2. **No Quality Guarantee for QMDP Approximation**: QMDP is an optimistic approximation that may severely deviate from optimal in scenarios where information gathering is highly valuable.
3. **Overly Strong Observability Assumption for Opponent Reasoning Level**: The framework assumes that one can observe "the opponent played a level-$k_r$ strategy," yet how to accurately infer reasoning levels from behavior is not discussed.
4. **No Analysis of Level Truncation Effects**: Truncating the hierarchy is necessary in practice, but the approximation error introduced by truncation lacks theoretical characterization.
5. **Fully Observable Environment State Assumed**: The environment state $\mathcal{S}$ is assumed fully observable, limiting applicability in partially observable settings.
6. **Scalability Remains Unclear**: While the paper claims ML techniques can extend the framework to large-scale domains, no concrete explanation or pathway is provided.

## Related Work & Insights

- **Logic-Based ToM**: ConGolog (De Giacomo et al. 2000), higher-order belief revision (Wan et al. 2021), general game playing (Genesereth & Thielscher 2014)
- **Cognitive Hierarchy Models**: Classical Poisson cognitive hierarchy (Camerer et al. 2004); GR2 embeds level-k reasoning into RL training (Wen et al. 2021)
- **I-POMDP**: Multi-agent recursive belief framework (Gmytrasiewicz & Doshi 2005); computability issues are well-known
- **Opponent Modeling**: Dirichlet-Categorical conjugate for matrix games (Boutilier 1996); type-based planners (Albrecht & Ramamoorthy 2015; Zhu & Lin 2025b)
- **Machine ToM**: Machine Theory of Mind (Rabinowitz et al. 2018); MMToM-QA (Jin et al. 2024); MuMA-ToM (Shi et al. 2025); AutoToM (Zhang et al. 2025)
- **Stochastic Game Theory**: Shapley 1953; Markov games (Littman 1994); constant-memory strategies (Zhu & Lin 2025a)

## Rating

- Novelty: ⭐⭐⭐⭐ Reformulates core ToM concepts from a game-theoretic perspective; Gamma-Poisson conjugate belief update is original
- Experimental Thoroughness: ⭐⭐ Purely theoretical with no experiments; framework effectiveness is unvalidated
- Writing Quality: ⭐⭐⭐⭐ Mathematical formalization is rigorous and clear; notation is consistent; appendix proofs are complete
- Value: ⭐⭐⭐⭐ Makes an important theoretical contribution to computable multi-agent ToM frameworks, though experimental validation is needed before practical deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [\[AAAI 2026\] Theory of Mind for Explainable Human-Robot Interaction](theory_of_mind_for_explainable_human-robot_interaction.md)
- [\[AAAI 2026\] Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)
- [\[AAAI 2026\] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](../../NeurIPS2025/robotics/mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)

</div>

<!-- RELATED:END -->
