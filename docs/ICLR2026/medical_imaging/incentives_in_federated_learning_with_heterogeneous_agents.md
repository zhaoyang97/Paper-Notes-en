---
title: >-
  [Paper Note] Incentives in Federated Learning with Heterogeneous Agents
description: >-
  [ICLR 2026][Medical Imaging][Federated Learning] This paper analyzes incentive problems in heterogeneous federated learning from a game-theoretic perspective, proves the existence of pure-strategy Nash equilibria under heterogeneous data distributions and PAC accuracy objectives, and proposes a linear programming-based approximation algorithm to determine optimal contribution levels.
tags:
  - ICLR 2026
  - Medical Imaging
  - Federated Learning
  - Incentive Mechanism
  - Heterogeneity
  - Game Theory
  - PAC Learning
date: 2026-05-08
content_hash: 3617fa8d2a8140c0
---

# Incentives in Federated Learning with Heterogeneous Agents

**Conference**: ICLR 2026
**arXiv**: [2509.21612](https://arxiv.org/abs/2509.21612)
**Code**: None
**Area**: Medical Imaging
**Keywords**: Federated Learning, Incentive Mechanism, Heterogeneity, Game Theory, PAC Learning

## TL;DR
This paper analyzes incentive problems in heterogeneous federated learning from a game-theoretic perspective, proves the existence of pure-strategy Nash equilibria under heterogeneous data distributions and PAC accuracy objectives, and proposes a linear programming-based approximation algorithm to determine optimal contribution levels.

## Background & Motivation

**Background**: Federated learning improves sample efficiency by pooling data across multiple agents, but contributing model updates incurs computational, bandwidth, and privacy costs for each participant.

**Limitations of Prior Work**: Existing FL research primarily focuses on algorithmic aspects (how to aggregate, how to handle heterogeneity) while largely neglecting the strategic behavior of participants — rational agents may choose to free-ride or contribute only minimal data.

**Key Challenge**: In heterogeneous settings, each agent has a different data distribution and cares about model performance on its own data. This means different agents derive different benefits from cooperation — agents with similar distributions enjoy stronger mutual benefit, while those with large distributional divergence may gain little from collaboration.

**Goal**: Under heterogeneous data distributions and PAC accuracy objectives, how should incentive mechanisms be designed so that the FL game admits a stable equilibrium?

**Key Insight**: FL is modeled as a strategic game in which each agent chooses the number of samples to contribute to maximize its own utility (PAC accuracy minus contribution cost); the existence and computational complexity of Nash equilibria are then analyzed.

**Core Idea**: Heterogeneous federated learning is formalized as a game under PAC accuracy objectives; pure-strategy Nash equilibria are shown to exist and to be approximately computable via linear programming.

## Method

### Overall Architecture
FL is modeled as an $N$-player game: each agent $i$ holds dataset $D_i$, selects a contribution level $m_i$, and has utility $u_i(\mathbf{m}) = \mathbb{I}[\text{PAC}(i, \mathbf{m})] - c_i \cdot m_i$, where the PAC condition depends on whether the pooled samples suffice to learn an $(\varepsilon, \delta)$-accurate model.

### Key Designs

1. **Heterogeneous PAC-FL Game Model**:

    - **Function**: Formalizes the incentive problem in heterogeneous FL.
    - **Mechanism**: Agent $i$'s PAC condition is determined by whether the pooled sample set $S = \bigcup_j S_j$ (with $|S_j| = m_j$ drawn i.i.d. from $D_j$) can produce an $(\varepsilon, \delta)$-accurate model on $D_i$. Because $D_i \neq D_j$ in general, the value of agent $j$'s data to agent $i$ depends on the distributional distance between them.
    - **Design Motivation**: The PAC framework transforms the question of whether cooperation is worthwhile into a decidable mathematical condition, rather than a vague notion of "performance improvement."

2. **Equilibrium Existence Analysis**:

    - **Function**: Establishes whether pure-strategy Nash equilibria exist in the heterogeneous PAC-FL game.
    - **Mechanism**: The paper first shows that determining whether a given contribution vector $\mathbf{m}$ satisfies the PAC condition is NP-hard (Theorem 2). Under specific assumptions (e.g., distributional distances satisfying metric properties), pure-strategy equilibria are then shown to exist. The key lemma is that the PAC condition is monotone in contribution levels — increasing any agent's contribution cannot harm other agents.
    - **Design Motivation**: Equilibrium existence guarantees the feasibility of stable cooperative arrangements.

3. **Linear Programming Approximation Algorithm**:

    - **Function**: Efficiently computes approximately optimal contribution allocations.
    - **Mechanism**: The NP-hard exact decision problem is relaxed into linear constraints. For each agent $i$, the PAC condition is approximated as $\sum_j w_{ij} m_j \geq T_i$, where $w_{ij}$ captures the contribution weight of $D_j$ toward $D_i$. This yields a linear feasibility problem solvable in polynomial time.
    - **Design Motivation**: Exact solutions are intractable, but the LP relaxation performs sufficiently well in practice.

### Loss & Training
This is a theoretical paper and does not involve model training. The core mathematical tools are PAC learning theory, game-theoretic equilibrium analysis, and linear programming duality.

## Key Experimental Results

### Main Results

| Setting | # Agents | Equilibrium Found | Social Welfare | Notes |
|---------|----------|-------------------|----------------|-------|
| Homogeneous distribution | 5 | ✓ | Optimal | Symmetric equilibrium |
| Mild heterogeneity | 5 | ✓ | Near-optimal | LP relaxation is tight |
| Severe heterogeneity | 5 | ✓ | With loss | Some agents do not contribute |
| 10 agents | 10 | ✓ | — | Scalable |

### Ablation Study

| Heterogeneity Level | Equilibrium Efficiency | Free-Riding Rate | Notes |
|---------------------|----------------------|------------------|-------|
| Low | >0.95 | 0% | All agents contribute |
| Medium | ~0.85 | ~20% | Partial free-riding |
| High | ~0.70 | ~40% | More heterogeneity leads to more free-riding |

### Key Findings
- Pure-strategy Nash equilibria exist under reasonable assumptions but may not be unique.
- Greater heterogeneity leads to more severe free-riding and larger social welfare losses.
- The LP approximation closely tracks the exact solution in practice.
- Distributional distance is the central factor determining the value of cooperation.

## Highlights & Insights
- **Clarity of the theoretical framework**: The FL incentive problem is precisely formalized as an intersection of game theory and PAC learning, with clean boundaries and rigorous conclusions.
- **NP-hard exact problem vs. tractable LP relaxation**: Although the exact problem is intractable, the relaxation is practically effective, providing a sound bridge between theory and practice.

## Limitations & Future Work
- The analysis assumes that PAC conditions for data contributions can be computed analytically; empirical estimation may be required in practice.
- Dynamic games (in which agents can revise strategies over time) are not considered.
- The impact of privacy constraints on contribution willingness is not addressed.
- Large-scale empirical validation is lacking.

## Related Work & Insights
- **vs. FedAvg / FedProx**: These methods focus on algorithmic design (how to aggregate), whereas this paper focuses on mechanism design (why to cooperate) — the two perspectives are complementary.
- **vs. Shapley-value FL**: Shapley value approaches allocate contribution value ex post; this paper studies ex ante participation incentives.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic analysis of heterogeneous FL incentives under the PAC framework.
- Experimental Thoroughness: ⭐⭐ Primarily a theoretical contribution with limited empirical evaluation.
- Writing Quality: ⭐⭐⭐⭐ Theoretically clear with concise proofs.
- Value: ⭐⭐⭐ Offers theoretical guidance for FL system design.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](../../CVPR2026/medical_imaging/omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](../../AAAI2026/medical_imaging/federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)
- [\[CVPR 2026\] FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning](../../CVPR2026/medical_imaging/fedvg_gradient-guided_aggregation_for_enhanced_federated_learning.md)
- [\[ICLR 2026\] From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)
- [\[ICLR 2026\] Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People](shoot_first_ask_questions_later_building_rational_agents_that_explore_and_act_li.md)

<!-- RELATED:END -->
