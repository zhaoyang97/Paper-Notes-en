---
title: >-
  [Paper Note] Hierarchical Reinforcement Learning with Targeted Causal Interventions
description: >-
  [ICML2025][Reinforcement Learning][Hierarchical Reinforcement Learning] This paper proposes the HRC framework, which models the relationships between subgoals in hierarchical reinforcement learning as a causal graph. It learns the subgoal structure using a causal discovery algorithm and performs **targeted interventions** based on causal effect prioritization, significantly reducing the training cost for long-horizon sparse reward tasks.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Hierarchical Reinforcement Learning"
  - "Causal Discovery"
  - "Subgoal Structure"
  - "Intervention Sampling"
  - "Long-Horizon Sparse Rewards"
date: 2026-05-08
content_hash: 9e3081f06b62b6ba
---

# Hierarchical Reinforcement Learning with Targeted Causal Interventions

**Conference**: ICML2025  
**arXiv**: [2507.04373](https://arxiv.org/abs/2507.04373)  
**Code**: [GitHub](https://github.com/sadegh16/HRC)  
**Area**: Reinforcement Learning  
**Keywords**: Hierarchical Reinforcement Learning, Causal Discovery, Subgoal Structure, Intervention Sampling, Long-Horizon Sparse Rewards

## TL;DR

This paper proposes the HRC framework, which models the relationships between subgoals in hierarchical reinforcement learning as a causal graph. It learns the subgoal structure using a causal discovery algorithm and performs **targeted interventions** based on causal effect prioritization, significantly reducing the training cost for long-horizon sparse reward tasks.

## Background & Motivation

Traditional RL performs poorly in long-horizon sparse reward tasks. HRL mitigates this issue by decomposing tasks into a hierarchy of subgoals. The core challenge lies in how to efficiently discover the hierarchical structure among subgoals and leverage it to accelerate training.

Existing works (Hu et al., 2022; Nguyen et al., 2024) attempt to infer causal relationships among subgoals using causal discovery algorithms, but suffer from the following limitations:

- Direct adaptation of general-purpose causal discovery algorithms (Ke et al., 2019) without customization for HRL scenarios.
- **Random** selection from the controllable subgoal set for exploration, failing to prioritize based on causal effects.
- Lack of theoretical analysis and performance guarantees.

To address these issues, this paper proposes a systematic causal hierarchical reinforcement learning framework.

## Method

### Overall Architecture: HRC (Hierarchical RL via Causality)

HRC associates environmental "unlocking factor" variables $\mathcal{X} = \{X_1, \ldots, X_n\}$ with subgoals $\Phi = \{g_1, \ldots, g_n\}$ in a one-to-one mapping, where subgoal $g_i$ is achieved when $X_i^t = 1$. The causal relationships among subgoals are represented by the **subgoal structure** $\mathscr{G}$ (a directed graph).

The algorithm maintains two key sets:

- **Controllable Set (CS)**: Subgoals that have already been mastered.
- **Intervention Set (IS)**: Subgoals utilized for causal exploration.

### Algorithm Flow (Algorithm 1)

1. **Initialization**: Train root subgoals (subgoals without parent nodes) and add them to CS.
2. **Loop** (until the final goal is added to IS):
    - Select a subgoal $g_{\text{sel}}$ from CS $\rightarrow$ add to IS.
    - **Intervention Sampling**: Perform interventions on subgoals in IS and collect trajectory data $D_I$.
    - **Causal Discovery**: Infer the subgoal structure $\hat{\mathscr{G}}$.
    - **Identify Reachable Subgoals** (CCS): Subgoals whose parent nodes are all in IS.
    - **Train Reachable Subgoals** $\rightarrow$ add to CS.

### Targeted Strategy

The key innovation lies in how to select $g_{\text{sel}}$ from CS, proposing two causally guided ranking rules:

**Rule 1: Causal Effect Ranking**

$$g_{\text{sel}, t} = \arg\max_{g_i \in CS_{t-1}} \widehat{ECE}_{t^*}^{\Delta}(\{g_i\}, \{\}, g_n)$$

Selects the subgoal with the largest causal effect on the final goal $g_n$. When all subgoals are of AND type, this rule achieves optimality by adding only subgoals with active paths to the final goal into IS.

**Rule 2: Shortest Path Ranking**

Inspired by A* search, it selects the subgoal with the minimum combined cost $\mathsf{f}(g_i) = \mathsf{g}(g_i) + \mathsf{h}(g_i)$. When the subgoal structure is a DAG and exclusively consisting of OR types, this rule perfectly matches the shortest path, yielding the minimum training cost.

### Causal Discovery Algorithm: SSD (Subgoal Structure Discovery)

The state transition of subgoals is modeled as an Abstract Structural Causal Model (A-SCM):

$$X_i^{t+1} = \theta_i(\mathbf{X}^t) \oplus \epsilon_i^{t+1}$$

where for AND subgoals, $\theta_i = \bigwedge_{g_j \in PA_{g_i}} X_j^t$, and for OR subgoals, $\theta_i = \bigvee_{g_j \in PA_{g_i}} X_j^t$.

The parent nodes are recovered by minimizing a loss function with sparsity regularization:

$$\mathcal{L}(\boldsymbol{\beta}) = \mathbb{E}[(\hat{X}_i^{t+1} - X_i^{t+1})^2] + \lambda \|\boldsymbol{\beta}\|_0$$

Theoretical proof (Theorem 8.4): There exists $\lambda > 0$ such that the positive coefficients in the optimal solution $\boldsymbol{\beta}^*$ correspond exactly to the parent nodes of $X_i$.

## Key Experimental Results

### Theoretical Cost Analysis (Table 1)

| Graph Structure | HRC_h (Targeted) | HRC_b (Random) |
|--------|--------------|--------------|
| Tree $G(n,b)$ | $O(\log^2(n) \cdot b)$ | $\Omega(n^2 b)$ |
| Semi-Erdős–Rényi $G(n,p)$ | $O(n^{4/3+2c/3} \log n)$ | $\Omega(n^2)$ |

The targeted strategy achieves **exponential** acceleration on tree structures.

### Synthetic Data Experiments (Figure 5)

- On semi-Erdős–Rényi graphs, both HRC_c and HRC_s significantly outperform HRC_b.
- On tree structures, the causal effect rule is equivalent to the shortest path rule (as there is only one path to the final goal).
- The sparser the graph, the greater the advantage of the targeted strategy.

### 2D-Minecraft Environment (Figure 6)

| Method | Convergence Speed |
|------|---------|
| HRC_h(SSD) | **Fastest** |
| HRC_b(SSD) | Second fastest |
| CDHRL | Slow |
| HAC / HER / OHRL / PPO | Slowest |

### Causal Discovery Accuracy Comparison (Table 2)

| Method | SHD ↓ | Missing Edges | Extra Edges |
|------|-------|------|------|
| SSD (Ours)| **12.3** | 6.0 | **6.3** |
| SDI (Ke et al.) | 19.8 | 4.2 | 15.6 |

The number of extra edges for SSD is far lower than that of SDI, reducing the overall SHD by 38%.

## Highlights & Insights

- **Causal Perspective on HRL**: Equating subgoal achievement to a causal intervention (the do-operator), establishing an elegant bridge between HRL and causal inference.
- **Theoretical Guarantees**: Providing the first training cost complexity analysis for causal HRL, showing a reduction in cost from $\Omega(n^2)$ to $O(\log^2 n)$ on tree structures.
- **Customized Causal Discovery**: SSD is specifically tailored to the AND/OR subgoal structures of HRL, achieving superior accuracy compared to general-purpose algorithms.
- **Complementary Dual-Ranking Rules**: The causal effect rule is suited for AND-type subgoals, while the shortest path rule is suited for OR-type subgoals, in addition to a hybrid rule.

## Limitations & Future Work

- The resource variables are assumed to be **discrete and binary**; continuous or high-dimensional state spaces require additional decoupled representation learning.
- Subgoals are assumed to never be lost once achieved (Assumption 4.2), which does not hold in reversible environments.
- Causal discovery can only recover **discoverable parent nodes** (Def 8.2), meaning some parent-child relationships may be missed.
- Experiments are only validated in 2D-Minecraft; evaluations in more complex 3D environments or robotic manipulation tasks are missing.
- The set of environmental resource variables $\mathcal{X}$ must be predefined in practice; autonomous discovery remains an open problem.

## Related Work & Insights

- **CDHRL** (Hu et al., 2022): The first framework for causal HRL, which utilizes SDI for causal discovery but lacks prioritization.
- **Nguyen et al., 2024**: Performs causal discovery on state-action pairs, which is more challenging in large state spaces.
- **HAC** (Levy et al., 2017): Hierarchical Actor-Critic without causal structures.
- **HER** (Andrychowicz et al., 2017): Hindsight Experience Replay.
- **Insight**: Causal effect prioritization can be extended to other domains requiring prioritized exploration, such as active learning and experimental design.

## Rating

- Novelty: ⭐⭐⭐⭐ (Causal intervention perspective + targeted exploration strategy + specialized causal discovery)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Theoretical validation + synthetic data + 2D-Minecraft, but lacks evaluations in more realistic scenarios)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, with the "craftsman" example consistently integrated for ease of understanding)
- Value: ⭐⭐⭐⭐ (Establishes a theoretical foundation for causal HRL, with the targeted strategy yielding practical acceleration)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](../../NeurIPS2025/reinforcement_learning/confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [\[ICML 2025\] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)
- [\[ACL 2026\] Targeted Exploration via Unified Entropy Control for Reinforcement Learning](../../ACL2026/reinforcement_learning/targeted_exploration_via_unified_entropy_control_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)
- [\[ICLR 2026\] Generalization of RLVR Using Causal Reasoning as a Testbed](../../ICLR2026/reinforcement_learning/generalization_of_rlvr_using_causal_reasoning_as_a_testbed.md)

</div>

<!-- RELATED:END -->
