---
title: >-
  [Paper Note] Incentives in Federated Learning with Heterogeneous Agents
description: >-
  [ICLR 2026][Optimization][Federated Learning] Analyses incentive problems in heterogeneous Federated Learning (FL) from a game-theoretic perspective, proving the existence of Pure Strategy Nash Equilibrium (PSNE) under heterogeneous data distributions and PAC accuracy targets, and proposes a linear programming-based approximation algorithm to determine optimal contribution levels.
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Federated Learning"
  - "Incentive Mechanism"
  - "Heterogeneity"
  - "Game Theory"
  - "PAC Learning"
date: 2026-05-08
content_hash: 1056b4e3c25a934d
---

# Incentives in Federated Learning with Heterogeneous Agents

**Conference**: ICLR 2026  
**arXiv**: [2509.21612](https://arxiv.org/abs/2509.21612)  
**Code**: None  
**Area**: Medical Image  
**Keywords**: Federated Learning, Incentive Mechanism, Heterogeneity, Game Theory, PAC Learning

## TL;DR
Analyses incentive problems in heterogeneous Federated Learning (FL) from a game-theoretic perspective, proving the existence of Pure Strategy Nash Equilibrium (PSNE) under heterogeneous data distributions and PAC accuracy targets, and proposes a linear programming-based approximation algorithm to determine optimal contribution levels.

## Background & Motivation

**Background**: Federated learning improves sample efficiency by pooling data from multiple agents; however, contributing model updates incurs costs in computation, bandwidth, and privacy for each participant.

**Limitations of Prior Work**: Existing FL research primarily focuses on the algorithmic level (how to aggregate, how to handle heterogeneity) but rarely considers the strategic behavior of participants—rational agents may choose to free-ride or contribute only a minimal amount of data.

**Key Challenge**: In heterogeneous scenarios, each agent has a different data distribution and cares about the model's performance on its own data. This implies that different agents benefit differently from cooperation: agents with similar distributions are highly reciprocal, while those with large differences may gain very little from cooperation.

**Goal**: How to design an incentive mechanism under heterogeneous data distributions and PAC accuracy targets such that a stable equilibrium exists in the FL game?

**Key Insight**: Modeling FL as a strategic game where each agent chooses a sample contribution amount to maximize its own utility (PAC accuracy minus contribution cost), and analyzing the existence and computational complexity of Nash Equilibrium.

**Core Idea**: Formalize heterogeneous federated learning as a game under PAC accuracy targets, proving that a pure strategy Nash equilibrium exists and can be computed via linear programming approximation.

## Method

### Overall Architecture
The paper formulates heterogeneous federated learning as an $N$-player strategic game: each agent $i$ holds data sampled from its own distribution $D_i$ and independently chooses a contribution amount $m_i$. The utility is defined as the benefit of hitting the PAC accuracy target minus the contribution cost, denoted as $u_i(\mathbf{m}) = \mathbb{I}[\text{PAC}(i, \mathbf{m})] - c_i \cdot m_i$. The key asymmetry lies in the fact that the PAC condition is determined by whether the pooled samples $S = \bigcup_j S_j$ (where $|S_j| = m_j$) can learn an $(\varepsilon, \delta)$-accurate model on $D_i$. The value of others' data to an agent depends on the distributional distance. Around this model, the authors sequentially address whether an equilibrium exists, how hard it is to determine, and whether it can be solved efficiently.

### Key Designs

**1. Heterogeneous PAC-FL Game Model: Turning "is cooperation worth it" into a decidable mathematical condition**

Existing FL research often uses vague descriptions like "performance will drop" when discussing heterogeneity, failing to answer whether a rational agent should contribute data. This paper formalizes this using the PAC learning framework: whether agent $i$ is satisfied is equivalent to whether the pooled sample set $S = \bigcup_j S_j$ (containing $m_j$ points independently sampled from $D_j$) can train an $(\varepsilon, \delta)$-accurate model on $D_i$. Since distributions $D_i \neq D_j$, the utility of agent $j$'s samples to agent $i$ is characterized entirely by the distributional distance. This step transforms "whether to cooperate" from empirical intuition into a strictly decidable proposition, laying the groundwork for existence and complexity analysis.

**2. Equilibrium Existence and Decision Complexity: Proving hardness before providing existence guarantees**

The authors first present the bad news: determining whether a given contribution vector $\mathbf{m}$ satisfies the PAC conditions for all agents is NP-hard (Theorem 2), as it essentially requires verifying learnability across exponentially many sample combinations. However, under reasonable assumptions (e.g., distribution distance satisfying metric properties), the existence of a Pure Strategy Nash Equilibrium can still be proven. The key lemma is the monotonicity of PAC conditions relative to contribution amounts—any agent increasing their contribution does not harm the satisfaction of other agents, thereby guaranteeing a stable point where no one wishes to unilaterally deviate. This conclusion of "hard to decide but equilibrium exists" suggests that stable cooperation is theoretically reachable but costly to find directly.

**3. Linear Programming Approximation Algorithm: Relaxing NP-hard determination into polynomial solvability**

Since exact determination is infeasible, the authors relax it into linear constraints. For each agent $i$, the PAC satisfaction condition is approximately written as $\sum_j w_{ij} m_j \geq T_i$, where weight $w_{ij}$ quantifies the marginal contribution of samples from $D_j$ to $D_i$, and $T_i$ is the effective sample threshold required to reach the target accuracy. The constraints of all agents are superimposed into a linear feasibility problem, which can be solved in polynomial time using LP to find the approximate optimal contribution allocation. Experiments show that this relaxation is quite tight: it is almost lossless under mild heterogeneity, bridging the gap between theoretical hardness and engineering utility.

This is a purely theoretical work and does not involve model training. The core tools used are PAC learning theory, game-theoretic equilibrium analysis, and linear programming duality.

## Key Experimental Results

### Main Results

| Setting | No. of Agents | Equilibrium Found | Social Welfare | Description |
|------|---------|---------|---------|------|
| Homogeneous | 5 | ✓ | Optimal | Symmetric Equilibrium |
| Mild Heterogeneity | 5 | ✓ | Near-Optimal | Tight LP relaxation |
| Severe Heterogeneity | 5 | ✓ | Lossy | Some agents do not contribute |
| 10 agents | 10 | ✓ | - | Scalable |

### Ablation Study

| Heterogeneity Level | Equilibrium Efficiency | Free-riding Rate | Description |
|---------|---------|---------|------|
| Low | >0.95 | 0% | Everyone contributes |
| Medium | ~0.85 | ~20% | Partial free-riding |
| High | ~0.70 | ~40% | More heterogeneity leads to more free-riding |

### Key Findings
- Pure Strategy Nash Equilibrium exists under reasonable assumptions but may not be unique.
- Greater heterogeneity leads to more severe free-riding and higher social welfare loss.
- LP approximation is close to the exact solution in practice.
- Distributional distance is the core factor determining the value of cooperation.

## Highlights & Insights
- **Clarity of Theoretical Framework**: Precisely formalizes FL incentive problems as a cross-disciplinary issue of game theory and PAC learning, with clear boundaries and rigorous conclusions.
- **NP-hard vs. LP Solvability**: Although the exact problem is difficult, the relaxation is practical, providing a good bridge between theory and practice.

## Limitations & Future Work
- Assumes that PAC conditions for data contribution can be analytically computed; in practice, empirical estimation may be required.
- Does not consider dynamic games where agents can change strategies over time.
- Does not consider the impact of privacy constraints on the willingness to contribute.
- Lacks large-scale empirical validation.

## Related Work & Insights
- **vs FedAvg/FedProx**: These focus on algorithm design (how to aggregate), whereas this paper focuses on mechanism design (why to cooperate), representing complementary perspectives.
- **vs Shapley Value FL**: Shapley value methods allocate contribution value ex-post, while this paper studies ex-ante participation incentives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic analysis of heterogeneous FL incentives under the PAC framework.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a theoretical work with limited empirical evidence.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theory and refined proofs.
- Value: ⭐⭐⭐⭐ Provides theoretical guidance for the design of FL systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning](beyond_aggregation_guiding_clients_in_heterogeneous_federated_learning.md)
- [\[ICLR 2026\] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](deepafl_deep_analytic_federated_learning.md)
- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](../../CVPR2026/optimization/enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)
- [\[ICLR 2026\] Byzantine-Robust Federated Learning with Learnable Aggregation Weights](byzantine-robust_federated_learning_with_learnable_aggregation_weights.md)

</div>

<!-- RELATED:END -->
