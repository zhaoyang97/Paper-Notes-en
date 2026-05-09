---
title: >-
  [Paper Note] Adaptive Frontier Exploration on Graphs with Applications to Network-Based Disease Testing
description: >-
  [NeurIPS 2025][Robotics][Gittins Index] This paper proposes the Adaptive Frontier Exploration on Graphs (AFEG) framework and designs a Gittins index-based policy that is provably optimal when the graph is a forest. On real-world sexually transmitted disease testing networks, the method identifies nearly all HIV-positive individuals by testing only half the population, substantially outperforming greedy and DQN baselines.
tags:
  - NeurIPS 2025
  - Robotics
  - Gittins Index
  - Branching Bandit
  - Graph Frontier Exploration
  - Disease Testing
  - Markov Random Field
date: 2026-05-08
content_hash: 3257c5e2d50f4da0
---

# Adaptive Frontier Exploration on Graphs with Applications to Network-Based Disease Testing

**Conference**: NeurIPS 2025
**arXiv**: [2505.21671](https://arxiv.org/abs/2505.21671)
**Code**: None
**Area**: Robotics
**Keywords**: Gittins Index, Branching Bandit, Graph Frontier Exploration, Disease Testing, Markov Random Field

## TL;DR
This paper proposes the Adaptive Frontier Exploration on Graphs (AFEG) framework and designs a Gittins index-based policy that is provably optimal when the graph is a forest. On real-world sexually transmitted disease testing networks, the method identifies nearly all HIV-positive individuals by testing only half the population, substantially outperforming greedy and DQN baselines.

## Background & Motivation

**State of the Field**: Network-based disease testing (e.g., HIV contact tracing) requires sequentially selecting nodes for testing on social/sexual contact networks, where test results from observed nodes update infection probabilities for their neighbors. The WHO recommends network-based testing strategies to reach undiagnosed populations.

**Limitations of Prior Work**: Directly modeling the problem as an MDP leads to an exponentially large state space; adaptive submodular optimization is inapplicable since marginal gains in disease testing may be increasing rather than diminishing; existing graph-based active search methods lack frontier constraints and theoretical guarantees.

**Root Cause**: Real-world testing is subject to a "frontier constraint"—only neighbors of already-tested nodes can be tested next (analogous to contact tracing)—yet designing an optimal policy under this constraint is highly non-trivial.

**Paper Goals**: How can graph nodes be adaptively selected under frontier constraints to maximize discounted cumulative reward?

**Starting Point**: AFEG is formulated as a branching bandit problem, leveraging the optimality theory of the Gittins index.

**Core Idea**: AFEG on forest graphs can be exactly reduced to a branching bandit problem, for which the Gittins index policy is provably optimal and efficiently computable.

## Method

### Overall Architecture
The input consists of a graph $\mathcal{G}$, a joint label distribution $\mathcal{P}$ (Markov Random Field), and a discount factor $\beta$. At each step, the agent selects a frontier node, observes its label to obtain a reward, and performs Bayesian updates on neighbors' beliefs. The objective is to maximize the discounted cumulative reward $\sum_t \beta^{t-1} r(X_{\pi(\mathcal{S}_{t-1})}, v)$.

### Key Designs

1. **AFEG → Branching Bandit Reduction (Theorem: Forest Optimality)**:

    - Function: Proves that when $\mathcal{G}$ is a forest, AFEG is exactly equivalent to a branching bandit problem.
    - Mechanism: In a forest, selecting a node makes its children new "arms," and by the Markov property, information from observed nodes propagates only along parent-child chains. This satisfies the independence conditions of branching bandits, making the Gittins index policy optimal.
    - Design Motivation: The optimality theory of branching bandits [KO03] guarantees the Gittins policy's optimality, but no efficient computation method previously existed.

2. **Piecewise-Linear Gittins Index Computation**:

    - Function: Proves that the Gittins index retirement value function is piecewise linear, enabling an efficient algorithm.
    - Mechanism: For discrete label spaces, the value function $\phi_{X,v}(k)$ (retirement value function for subtree $X$ under parent label $v$) is piecewise linear. The Gittins index equals the supremum of the function's slope, which can be efficiently computed by storing breakpoints.
    - Design Motivation: The Gittins index has traditionally been considered computationally intractable; this is the first efficient implementation for discrete branching bandits.

3. **Heuristic Extension to Non-Forest Graphs**:

    - Function: For general (non-forest) graphs, approximates the original graph with a BFS tree and applies the Gittins policy.
    - Mechanism: A BFS tree is constructed for each connected component, back edges are ignored, and the Gittins index is computed on the resulting tree. Although optimality guarantees no longer hold, the method consistently outperforms other baselines in practice.
    - Design Motivation: Real-world sexual contact networks, while not strictly forests, are typically sparse with low treewidth (approximately 2–5 in experiments), making BFS tree approximation effective.

### Loss & Training
- No training phase (non-learning-based method).
- Runtime: $\mathcal{O}(n^2 \cdot |\Omega|^2)$; oracle calls: $\mathcal{O}(n \cdot |\Omega|^2)$; space: $\mathcal{O}(n^2 \cdot |\Omega|)$.
- MRF parameters can be learned from historical disease data (Appendix B).

## Key Experimental Results

### Main Results (Real Sexually Transmitted Disease Networks)

| Disease | Gittins Detection Rate at 50% Population Tested | Greedy | DQN | Random |
|---------|--------------------------------------------------|--------|-----|--------|
| HIV | **≈100%** | ≈80% | ≈80% | ≈55% |
| Gonorrhea | **≈98%** | ≈90% | ≈85% | ≈55% |
| Chlamydia | **≈97%** | ≈88% | ≈82% | ≈55% |
| Syphilis | **≈95%** | ≈85% | ≈80% | ≈55% |
| Hepatitis | **≈96%** | ≈87% | ≈83% | ≈55% |

### Ablation Study (Synthetic Trees)

| Configuration | Description |
|---------------|-------------|
| Forest graph | Gittins = Optimal (validates the optimality theorem) |
| Added edges → non-tree | Gittins advantage diminishes but remains superior to all baselines |
| Noisy distribution $\mathcal{Q}_\varepsilon$ | All policies degrade toward Random as $\varepsilon$ increases, but Gittins degrades most slowly |

### Key Findings
- **Gittins is consistently optimal not only in discounted reward but also under any fixed budget**: Although the theory only guarantees discounted cumulative reward, experiments show that Gittins outperforms all other methods at every time step.
- **Real-world networks have low treewidth (2–5)**: Sexual contact networks are naturally near-tree structures, which is the key reason the Gittins policy is effective in practice.
- **DQN underperforms Greedy**: Deep RL methods struggle to compete on structured problems of this type; the exponential state space impedes effective learning.

## Highlights & Insights
- **Formalizing a public health problem as a branching bandit is a genuinely novel perspective**: This provides a theoretically grounded optimal policy for network-based disease testing, rather than a purely heuristic approach.
- **Characterization of the piecewise-linear Gittins index**: This resolves a long-standing open computational problem in the branching bandit literature, representing an independent theoretical contribution.
- **Substantial practical impact**: WHO-recommended contact tracing strategies can directly benefit from this method, with immediate applicability to the HIV "95-95-95" targets.

## Limitations & Future Work
- **No theoretical guarantees for non-forest graphs**: BFS tree approximation may fail on graphs with high treewidth.
- **Assumes fully known MRF**: In practice, $\mathcal{P}$ must be estimated from data, and estimation errors propagate to policy quality.
- **Scalability constraints**: $\mathcal{O}(n^2 |\Omega|^2)$ complexity may be prohibitive for very large networks.
- **Dynamic networks not addressed**: The graph structure is assumed fixed, whereas real contact networks evolve over time.

## Related Work & Insights
- **vs. Active Search on Graphs**: Active search imposes no frontier constraints and uses Gaussian random field approximations; AFEG enforces frontier constraints and models the distribution exactly via MRF.
- **vs. DQN/RL**: General-purpose RL is inefficient in exponential state spaces; the Gittins index exploits problem structure for direct solution.
- **vs. Influence Maximization**: IM maximizes the spread of influence, whereas AFEG maximizes testing reward—the objectives are fundamentally different.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizes graph frontier exploration as a branching bandit problem and provides an efficient algorithm with solid theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic and real-world data, five diseases, and noise robustness analysis, though larger-scale experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated (HIV testing), theory is rigorous, and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ Has direct implications for public health, with both theoretical and practical significance.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](../../AAAI2026/robotics/adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [\[ICCV 2025\] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding](../../ICCV2025/robotics/adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](../../CVPR2026/robotics/learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[AAAI 2026\] Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](../../AAAI2026/robotics/affordance-guided_coarse-to-fine_exploration_for_base_placem.md)
- [\[ICLR 2026\] Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?](../../ICLR2026/robotics/theory_of_space_can_foundation_models_construct_spatial_beliefs_through_active_e.md)

<!-- RELATED:END -->
