---
title: >-
  [Paper Note] Boundary-to-Region Supervision for Offline Safe Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Offline Safe Reinforcement Learning] This paper proposes B2R (Boundary-to-Region), a framework that addresses the symmetric conditioning fallacy of sequence models in offline safe RL by introducing Cost-to-Go (CTG) Realignment. It converts sparse boundary supervision into dense safe-region supervision, satisfying safety constraints on 35 out of 38 safety-critical tasks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Offline Safe Reinforcement Learning
  - Decision Transformer
  - Cost Constraints
  - Asymmetric Conditioning
  - Safe Region Supervision
date: 2026-05-08
content_hash: 2f564445f63ede62
---

# Boundary-to-Region Supervision for Offline Safe Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.25727](https://arxiv.org/abs/2509.25727)
**Code**: [https://github.com/HuikangSu/B2R](https://github.com/HuikangSu/B2R)
**Area**: Reinforcement Learning
**Keywords**: Offline Safe Reinforcement Learning, Decision Transformer, Cost Constraints, Asymmetric Conditioning, Safe Region Supervision

## TL;DR
This paper proposes B2R (Boundary-to-Region), a framework that addresses the symmetric conditioning fallacy of sequence models in offline safe RL by introducing Cost-to-Go (CTG) Realignment. It converts sparse boundary supervision into dense safe-region supervision, satisfying safety constraints on 35 out of 38 safety-critical tasks.

## Background & Motivation

**State of the Field**: Offline safe reinforcement learning aims to learn policies that satisfy safety constraints from static datasets. Decision Transformer (DT)-based methods have achieved promising results by reformulating RL as conditional sequence modeling.

**Limitations of Prior Work**: Existing DT methods (e.g., CDT) treat return-to-go (RTG) and cost-to-go (CTG) as symmetric input tokens, overlooking their fundamental difference: RTG is a flexible performance objective, whereas CTG should serve as a rigid safety boundary.

**Root Cause**: This symmetric treatment leads to two problems: (1) it is difficult to select an appropriate initial CTG value at deployment; (2) trajectories whose cumulative cost falls exactly near the safety threshold are sparse in the dataset, resulting in insufficient supervision signals.

**Paper Goals**: To design an asymmetric conditioning mechanism that treats CTG as a boundary constraint rather than a variable target, decoupling safety guarantees from reward optimization.

**Starting Point**: Uniformly realign the costs of all safe trajectories to the safety threshold, enabling the model to learn diverse behaviors across the entire safe region under a fixed boundary token.

**Core Idea**: CTG Realignment converts sparse "boundary supervision" into dense "region supervision," allowing the model to learn from all behaviors within the safe region rather than only from the rare trajectories whose cost exactly equals the threshold.

## Method

### Overall Architecture
B2R comprises three tightly coupled components: (1) trajectory filtering to remove unsafe samples; (2) CTG Realignment to unify all safe trajectories to the deployment cost threshold; and (3) RoPE positional encoding to improve modeling of temporal dynamics after realignment.

### Key Designs

1. **Trajectory Filtering**:

   - Function: Define the safe region and remove violating trajectories.
   - Mechanism: Retain trajectories with cumulative cost $C(\tau) \leq \kappa$ to form $\mathcal{D}_{\text{safe}}$, ensuring all training data satisfies the deployment constraint.
   - Design Motivation: Prevent unsafe trajectories from negatively influencing the learned policy.

2. **CTG Realignment**:

   - Function: Create dense and uniform supervision signals.
   - Mechanism: Add a constant offset to the CTG sequence of each safe trajectory, $\hat{C}_t' = \hat{C}_t + (\kappa - C(\tau))$, so that the initial CTG is uniformly set to $\kappa$ while preserving the original temporal variation pattern.
   - Design Motivation: Conventional methods learn only from boundary trajectories whose cost is exactly $\kappa$ (sparse). After realignment, the model learns diverse behaviors from the entire safe region under a unified boundary token (dense).

3. **RoPE Positional Encoding**:

   - Function: Improve temporal modeling and adapt to CTG Realignment.
   - Mechanism: Replace the original absolute/learnable positional encoding in DT with Rotary Position Embedding (RoPE), whose relative positional encoding property is better suited to capturing the gradual dynamics of the realigned cost sequence.
   - Design Motivation: Realignment alters the absolute values of the CTG sequence while preserving relative changes; the relative encoding nature of RoPE is naturally aligned with this property.

### Loss & Training
The model is trained with standard behavior cloning loss:
$$\mathcal{L}_{BC}(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}_{\text{safe}}}[-\log \pi_\theta(a_t | \hat{R}_{t-K:t}, \hat{C}'_{t-K:t}, s_{t-K:t}, a_{t-K:t-1})]$$
Training is performed on the CTG-realigned safe dataset, and at deployment the fixed value $\hat{C}_0' = \kappa$ is used.

## Key Experimental Results

### Main Results

| Task Category | Safety Constraints Satisfied | Total Tasks | B2R | CDT Baseline |
|---|---|---|---|---|
| Safety-Critical Tasks | 35 | 38 | Best Reward | ~20 satisfied |

### Ablation Study

| CTG Realignment Strategy | Performance | Description |
|---|---|---|
| Shift (uniform offset) | Best | Preserves temporal profile; simple and effective |
| Avg (averaging) | Second best | Uniformly redistributes surplus cost budget |
| Scale (scaling) | Moderate | Multiplicative normalization |
| Rand (random) | Worst | Random redistribution introduces noise |

### Key Findings
- B2R satisfies safety constraints on 35 out of 38 tasks, significantly outperforming CDT and other baselines.
- The Shift strategy is both the simplest and most effective, as it preserves the temporal variation pattern of the original CTG.
- MetaDrive experiments visually demonstrate the fragility of boundary supervision: a policy trained only at $v=10$ frequently exceeds the speed limit, while B2R learns from diverse speed behaviors and achieves smooth safety-margin control.

## Highlights & Insights
- The identification of the "symmetry fallacy" is a particularly insightful contribution: although RTG and CTG are structurally similar, they carry fundamentally different semantics—one is "a goal to pursue" and the other is "a boundary not to be crossed." This insight generalizes to all objective-constrained optimization problems.
- The elegance of CTG Realignment lies in the fact that it converts sparse supervision into dense supervision purely through data preprocessing, without modifying the model architecture.

## Limitations & Future Work
- Theoretical analysis rests on simplified assumptions; safety guarantees in practical environments may be limited.
- Trajectory filtering may discard a large portion of data, which is disadvantageous in data-scarce settings.
- Future work could explore adaptive cost thresholds and online adjustment strategies.

## Related Work & Insights
- **vs. CDT**: CDT treats RTG and CTG symmetrically; B2R decouples safety from performance through asymmetric design.
- **vs. TraC**: TraC only classifies and discards unsafe trajectories, whereas B2R further transforms safe trajectories to preserve behavioral diversity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The identification of the symmetry fallacy and the region supervision paradigm are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive validation across 38 tasks.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical analysis is clear; intuitive figures are effective.
- Value: ⭐⭐⭐⭐⭐ — Provides a new theoretical and practical foundation for applying sequence models to safe RL.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm](trust_region_reward_optimization_and_proximal_inverse_reward_optimization_algori.md)

<!-- RELATED:END -->
