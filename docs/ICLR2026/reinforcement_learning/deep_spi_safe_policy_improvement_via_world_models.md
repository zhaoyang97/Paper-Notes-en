---
title: >-
  [Paper Note] Deep SPI: Safe Policy Improvement via World Models
description: >-
  [ICLR 2026][Reinforcement Learning][safe policy improvement] This paper establishes a theoretical framework for Safe Policy Improvement (SPI) that unifies world models and representation learning with policy update guarantees: an importance-ratio-based neighborhood operator constrains policy updates to ensure monotonic improvement and convergence; local transition/reward losses control world model quality and representation stability. The proposed DeepSPI algorithm matches or surpasses PPO and DeepMDP on the ALE-57 benchmark.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - safe policy improvement
  - world model
  - representation learning
  - PPO
  - importance ratio
date: 2026-05-08
content_hash: 9076b7813cf90e58
---

# Deep SPI: Safe Policy Improvement via World Models

**Conference**: ICLR 2026
**arXiv**: [2510.12312](https://arxiv.org/abs/2510.12312)
**Area**: Reinforcement Learning / Safe Policy Improvement / World Models
**Keywords**: safe policy improvement, world model, representation learning, PPO, importance ratio

## TL;DR

This paper establishes a theoretical framework for Safe Policy Improvement (SPI) that unifies world models and representation learning with policy update guarantees: an importance-ratio-based neighborhood operator constrains policy updates to ensure monotonic improvement and convergence; local transition/reward losses control world model quality and representation stability. The proposed DeepSPI algorithm matches or surpasses PPO and DeepMDP on the ALE-57 benchmark.

## Background & Motivation

**State of the Field**: Safe Policy Improvement (SPI) constrains policy updates to avoid catastrophic degradation, providing formal guarantees. However, classical SPI methods apply only to offline, tabular RL, require exhaustive state-action coverage, and cannot scale to high-dimensional continuous spaces.

**Limitation 1 — Out-of-Training (OOT) Problem**: When a policy deviates from the behavioral policy and the world model's training distribution, the model may hallucinate in unexplored regions, causing policy update failures. For example, the model may incorrectly assign high rewards (e.g., +20) to unvisited states that should yield negative rewards.

**Limitation 2 — Confounded Policy Updates**: When the policy and representation are updated simultaneously, a poor representation may merge distinct states into the same latent representation. Changing the policy based on merged representations may cause catastrophic negative rewards in certain actual states.

**Core Idea**: Use the importance ratio (IR) constraint as a neighborhood operator for policy updates, bounding the deviation between the new policy and the behavioral policy within $[2-C, C]$. Combined with local reward and transition losses, this jointly guarantees: (1) the world model is accurate within the policy neighborhood, and (2) representation learning maintains Lipschitz stability.

## Method

### Overall Architecture

DeepSPI extends PPO by embedding auxiliary transition/reward losses directly into the policy optimization objective rather than treating them as independent auxiliary losses. This ensures that representation updates do not push the policy outside the safe neighborhood.

### Key Design 1: Neighborhood Operator

An IR-based neighborhood $\mathcal{N}^C(\pi)$ is defined to constrain the magnitude of policy updates:

$$\mathcal{N}^C(\pi) = \left\{ \pi' \in \Pi \mid 2 - C \leq D_{\text{IR}}^{\inf}(\pi, \pi') \leq D_{\text{IR}}^{\sup}(\pi, \pi') \leq C \right\}$$

The parameter $C \in (1, 2)$ controls the exploration–exploitation trade-off. The advantage is optimized within this neighborhood:

$$\pi_{n+1} = \arg\sup_{\pi' \in \mathcal{N}^C(\pi_n)} \mathbb{E}_{s \sim \mu_{\pi_n}} \mathbb{E}_{a \sim \pi'} A^{\pi_n}(s, a)$$

**Theorem 1** proves that the sequence $\{V^{\pi_n}\}$ improves monotonically and converges to $V^*$.

### Key Design 2: Local Losses for World Model Quality

Local reward loss $L_R^\mathcal{B}$ and transition loss $L_P^\mathcal{B}$ (based on Wasserstein distance) are defined as:

$$L_R^\mathcal{B} = \mathbb{E}_{s,a \sim \mathcal{B}} |R(s,a) - \bar{R}(\bar{s}, a)|, \quad L_P^\mathcal{B} = \mathbb{E}_{s,a \sim \mathcal{B}} \mathcal{W}(\phi_\sharp P(\cdot|s,a), \bar{P}(\cdot|\phi(s), a))$$

**Theorem 2** proves that when the IR constraint is satisfied, the return discrepancy between the true environment and the world model is linearly bounded by the local losses. **Theorem 3** (the Deep SPI theorem) formally guarantees safe policy improvement.

### Key Design 3: Unified Utility Function

Auxiliary losses are embedded into PPO's advantage function:

$$U^{\pi_n}(s, a, s') = A^{\pi_n}(s, a) - \alpha_R \cdot \ell_R(s, a) - \alpha_P \cdot \ell_P(s, a, s')$$

Replacing all occurrences of $A$ in PPO with $U$ ensures that policy updates automatically account for model losses.

### Representation Learning Guarantee

**Theorem 4** proves that when the losses are sufficiently small, the updated representation maintains approximate Lipschitz continuity:

$$|V^{\bar{\pi}}(s_1) - V^{\bar{\pi}}(s_2)| \leq K_V \cdot \bar{d}(\phi(s_1), \phi(s_2)) + \varepsilon$$

States with similar values thus remain close in latent space, preventing representation collapse.

## Key Experimental Results

### ALE-57 Aggregate Results

| Metric | PPO | DeepMDP | **DeepSPI** |
|--------|-----|---------|-------------|
| Mean | Baseline | Slightly above PPO | **Best** |
| Median | Baseline | On par with PPO | **Best** |
| IQM | Baseline | Slightly above PPO | **Best** |
| Optimality Gap↓ | Baseline | Slightly below PPO | **Lowest** |

### Ablation Study: World Model Quality (Median Loss During Training)

| Metric | DeepMDP | **DeepSPI** |
|--------|---------|-------------|
| Transition Loss $L_P$↓ | Higher | **Lower** |
| Reward Loss $L_R$↓ | Comparable | **Comparable** |

### Toy Maze Validation

| Method | Return from Start State I | ⋆-State Representation Distance |
|--------|--------------------------|----------------------------------|
| PPO | ~4.8 (representation collapse; always selects "right") | ~0 (merged) |
| **DeepSPI** | **~8** (distinguishes top/bottom ⋆; correctly selects "up") | **>0 (separated)** |

### Key Findings

- DeepSPI matches or surpasses PPO and DeepMDP on all aggregate metrics across ALE-57.
- DeepSPI consistently achieves lower transition loss, indicating a more accurate learned world model.
- In the carefully designed Toy Maze, DeepSPI successfully avoids representation collapse, improving return by approximately 67%.
- No competition between transition and reward losses is observed (unlike the offline setting).
- DreamSPI (a purely model-based planning variant) demonstrates feasibility in select environments.

## Highlights & Insights

1. **Theory–Practice Bridge**: Extends the rigorous guarantees of offline SPI to online deep RL with world models and representation learning; four theorems build a complete, layered framework.
2. **Unified Resolution of OOT and Confounded Updates**: Two seemingly distinct problems are simultaneously addressed through the same IR neighborhood constraint mechanism.
3. **Embedded Auxiliary Losses**: Incorporating model losses into the advantage rather than optimizing them independently prevents representation updates from pushing the policy out of the safe neighborhood — a more principled approach than DeepMDP's independent auxiliary losses.
4. **Theoretical Foundation for PPO**: Demonstrates that PPO's clipping mechanism is essentially a relaxed form of the neighborhood constraint, providing an SPI-theoretic explanation for its empirical success.

## Limitations & Future Work

- Differences from PPO/DeepMDP in aggregate results require careful examination of confidence intervals; statistically significant gains may not hold for individual environments.
- Lipschitz constraints require additional architectural choices (e.g., GroupSort networks), increasing implementation complexity.
- DreamSPI (purely model-based planning) underperforms online methods, indicating that on-policy world model learning combined with planning remains challenging.
- Evaluation is limited to Atari environments; applicability to continuous control tasks (e.g., MuJoCo) remains unverified.
- The theory requires assumptions such as $\gamma > 1/2$ and $K_{\bar{P}}^{\bar{\pi}} < 1/\gamma$, which may not always hold in practice.

## Related Work & Insights

- Relation to classical SPI methods (SPIBB, Laroche et al.): extends from offline tabular settings to online deep RL.
- Direct comparison with DeepMDP (Gelada et al., 2019): the latter does not constrain the effect of policy updates and thus provides no SPI guarantees.
- Formal connection to PPO/TRPO: proves that the IR neighborhood constraint is a strict version of the clipping operation.
- Connection to bisimulation (Castro et al.): the representation guarantee aligns with the notion of equivalence in state abstraction.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to extend SPI guarantees to online deep RL with world models and representation learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation on ALE-57, though limited to the Atari domain.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous theoretical derivations and intuitive examples, though mathematical density is high.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a solid theoretical foundation and practical algorithm for safety in deep RL.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] From Observations to Events: Event-Aware World Model for Reinforcement Learning](from_observations_to_events_event-aware_world_model_for_reinforcement_learning.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)
- [\[CVPR 2026\] GeoWorld: Geometric World Models](../../CVPR2026/reinforcement_learning/geoworld_geometric_world_models.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model](../../NeurIPS2025/reinforcement_learning/boundary-to-region_supervision_for_offline_safe_reinforcement_learning.md)
- [\[ICLR 2026\] Understanding and Improving Hyperbolic Deep Reinforcement Learning](understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)

<!-- RELATED:END -->
