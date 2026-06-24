---
title: >-
  [Paper Note] Reward-Aware Proto-Representations in Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Successor Representation] This paper systematically develops the theoretical foundations of the Default Representation (DR)—deriving DP and TD learning algorithms, analyzing the feature space structure, and proposing default features for function approximation—and demonstrates DR's reward-aware advantages over the Successor Representation (SR) across four settings: reward shaping, option discovery, exploration, and transfer learning.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Successor Representation"
  - "Default Representation"
  - "Reward-Aware Representation"
  - "Option Discovery"
  - "Transfer Learning"
date: 2026-05-08
content_hash: 8ebe2b4aed9fad52
---

# Reward-Aware Proto-Representations in Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.16217](https://arxiv.org/abs/2505.16217)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Successor Representation, Default Representation, Reward-Aware Representation, Option Discovery, Transfer Learning

## TL;DR
This paper systematically develops the theoretical foundations of the Default Representation (DR)—deriving DP and TD learning algorithms, analyzing the feature space structure, and proposing default features for function approximation—and demonstrates DR's reward-aware advantages over the Successor Representation (SR) across four settings: reward shaping, option discovery, exploration, and transfer learning.

## Background & Motivation

### State of the Field

**Background**: The Successor Representation (SR) encodes temporal relationships between states by capturing transition dynamics, and has been widely applied to reward shaping, exploration, transfer learning, and related areas. However, SR is reward-agnostic—it only encodes the number of transitions to each state.

**Limitations of Prior Work**: In environments containing low-reward regions that agents should avoid, SR cannot distinguish between high-reward and low-reward paths. The Default Representation (DR) proposed by Piray and Daw is reward-aware, but lacks efficient online learning algorithms and rigorous theoretical analysis.

**Key Challenge**: SR encodes $\gamma^{\eta(\tau)}$ (discount-weighted step counts), whereas DR encodes $\exp(r(\tau)/\lambda)$ (the exponential of cumulative rewards)—the latter naturally integrates reward information.

**Core Idea**: Develop a complete theoretical toolkit for DR so that it can be applied to RL as conveniently as SR.

## Method

### Overall Architecture
Under the linearly solvable MDP framework, DR is defined as $\mathbf{Z} = [\text{diag}(\exp(-\mathbf{r}/\lambda)) - \mathbf{P}^{\pi_d}]^{-1}$. The paper advances along three dimensions: learning algorithms, theoretical analysis, and function approximation.

### Key Designs

1. **DP and TD Learning Algorithms**:

    - DP update: $\mathbf{Z}_{k+1} = \mathbf{R}^{-1} + \mathbf{R}^{-1}\mathbf{P}^{\pi_d}\mathbf{Z}_k$, with convergence established via the Neumann series.
    - TD update: $\mathbf{Z}(s,j) \leftarrow \mathbf{Z}(s,j) + \alpha[\exp(r/\lambda)(\mathbb{1}_{s=j} + \mathbf{Z}(s',j)) - \mathbf{Z}(s,j)]$
    - Compared to the SR TD update, the key difference lies in the reward-aware discount factor $\exp(r/\lambda)$ replacing $\gamma$.

2. **Feature Space Analysis**:

    - Theorem 3.1: When rewards are **constant** across all states, SR and DR share the same eigenvectors.
    - When rewards **vary**, DR's eigenvectors reflect the locations of low-reward regions (see Figure 2), whereas SR's eigenvectors only reflect transition distances.

3. **Default Features**:

    - Analogous to Successor Features (SF), the value function is decomposed as: $\exp(\mathbf{v}^*_N/\lambda) = \boldsymbol{\zeta}(s)^\top \mathbf{w}$
    - TD update: $\boldsymbol{\zeta}(s) \leftarrow \boldsymbol{\zeta}(s) + \alpha(\exp(r/\lambda)\boldsymbol{\zeta}(s') - \boldsymbol{\zeta}(s))$
    - Optimal policies under different terminal rewards can be computed without access to transition dynamics.

## Key Experimental Results

### Main Results — Reward Shaping (Four Environments with Low-Reward Regions)

| Environment | DR-pot | SR-pot | SR-pri | No Shaping |
|-------------|--------|--------|--------|------------|
| Grid Task | **Best** | Second | Worst | Slow convergence |
| Four Rooms | **Best** | Suboptimal path | Suboptimal path | Extremely slow |

### Ablation Study / Comparison — Exploration (Count-based, values ×10³)

| Environment | Sarsa | +SR | +DR |
|-------------|-------|-----|-----|
| RiverSwim | 25 | 1,206 | **2,964** |
| SixArms | 265 | 1,066 | **3,518** |

### Key Findings
- DR significantly outperforms SR in environments with low-reward obstacles—SR consistently selects the shortest path (which may pass through low-reward regions), whereas DR navigates around them.
- When all states share the same reward, DR and SR perform nearly identically, consistent with theoretical predictions.
- DR's norm, used as a density model on RiverSwim, substantially outperforms SR—likely because incorporating reward information accelerates learning.

## Highlights & Insights
- **DR is a reward-aware generalization of SR**: it unifies both "where to go" and "whether it is worthwhile to go there."
- Default features enable **transfer learning without access to environment dynamics**—SR-based features are more flexible (supporting arbitrary reward changes) but can only recover performance up to the reference policy level.

## Limitations & Future Work
- The term $\exp(-r/\lambda)$ in DR may produce numerical instability under large negative rewards.
- Validation is limited to tabular environments; extension to deep RL has not been explored.
- DR supports transfer only when terminal rewards change, making it less flexible than SF.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically consolidates a promising yet underappreciated representation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four settings: reward shaping, exploration, option discovery, and transfer.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and experimental results align with theoretical predictions.
- Value: ⭐⭐⭐⭐ Establishes a theoretical foundation for reward-aware representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](risk-averse_total-reward_reinforcement_learning.md)
- [\[ICML 2026\] From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning](../../ICML2026/reinforcement_learning/from_reward-free_representations_to_preferences_rethinking_offline_preference-ba.md)
- [\[NeurIPS 2025\] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](learning_from_demonstrations_via_capability-aware_goal_sampling.md)
- [\[NeurIPS 2025\] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning](finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)

</div>

<!-- RELATED:END -->
