---
title: >-
  [Paper Note] ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting
description: >-
  [ICLR 2026][Reinforcement Learning][Offline RL] ROMI achieves robust value-aware model learning by converting the dynamics uncertainty set into a state uncertainty set via Wasserstein duality, and employs an implicitly differentiable adaptive weighting mechanism to balance dynamics accuracy against value-awareness. This resolves the Q-value underestimation and gradient explosion issues in RAMBO, achieving state-of-the-art performance among model-based offline RL methods on D4RL and NeoRL.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Offline RL
  - model-based methods
  - adversarial model learning
  - Wasserstein duality
  - bilevel optimization
date: 2026-05-08
content_hash: d44d5bf362dd0976
---

# ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting

**Conference**: ICLR 2026
**arXiv**: [2603.08118](https://arxiv.org/abs/2603.08118)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Offline RL, model-based methods, adversarial model learning, Wasserstein duality, bilevel optimization

## TL;DR
ROMI achieves robust value-aware model learning by converting the dynamics uncertainty set into a state uncertainty set via Wasserstein duality, and employs an implicitly differentiable adaptive weighting mechanism to balance dynamics accuracy against value-awareness. This resolves the Q-value underestimation and gradient explosion issues in RAMBO, achieving state-of-the-art performance among model-based offline RL methods on D4RL and NeoRL.

## Background & Motivation

**State of the Field**: Model-based offline RL augments datasets by learning environment dynamics models and expanding training data through simulated rollouts. RAMBO is a representative adversarial model learning method that produces conservative value estimates by inversely optimizing the dynamics model.

**Limitations of Prior Work**: RAMBO has a critical flaw — its trade-off coefficient $\lambda$ must remain extremely small (3e-4); even a slight increase (0.05–0.1) causes severe Q-value underestimation and gradient explosion, leading to training collapse. This renders RAMBO's conservatism essentially uncontrollable.

**Root Cause**: Model learning must simultaneously satisfy two objectives — (a) dynamics accuracy (fitting the data) and (b) value-awareness (being conservative in inaccurate regions that the policy may exploit). RAMBO balances these via direct gradient-based adversarial updates on the model, but mismatched gradient scales between the two objectives cause instability.

**Starting Point**: Convert "adversarial optimization in model space" into "uncertainty sets in state space" via Wasserstein duality, so that conservatism can be smoothly controlled through the uncertainty set radius $\xi$.

**Core Idea**: Rather than adversarially perturbing the dynamics model itself, conservatism is achieved by adversarially perturbing the predicted states.

## Method

### Overall Architecture
Learn dynamics model $\hat{T}_\psi$ → train via weighted MLE (inner optimization) → update the weighting network via RVL loss (outer optimization, gradients propagated through implicit differentiation) → use the learned model for rollout-based data augmentation → standard CQL policy optimization.

### Key Designs

1. **Robust Value-Aware Model Learning (RVL)**:

    - Function: Achieves conservative value estimation by imposing a Wasserstein ball constraint on predicted states.
    - Mechanism: Leverages Proposition 4.1 to convert the Wasserstein dynamics uncertainty set into a state uncertainty set: $\min_{\hat{T} \in \mathcal{M}_\xi} \mathbb{E}_{s' \sim \hat{T}}\hat{V}(s') = \mathbb{E}_{s' \sim \hat{T}_{\text{MLE}}}[\min_{\hat{s} \in U_\xi(s')}\hat{V}(\hat{s})]$. RVL loss: $\mathcal{L}_{\text{RVL}} = (\mathbb{E}_{\hat{s}' \sim \hat{T}}\hat{V}(\hat{s}') - \min_{\tilde{s}' \in U_\xi(s')}\hat{V}(\tilde{s}'))^2$.
    - Design Motivation: $\xi$ directly controls the degree of conservatism — unlike RAMBO's $\lambda$, it is not sensitive or uncontrollable. Training remains stable across $\xi \in \{0.01, 0.1, 1.0, 10\}$.
    - Proposition 4.2 guarantees bounded Q-values: $Q_{\text{true}} - \frac{\gamma(\epsilon_1 + \epsilon_2)}{1-\gamma} \leq \hat{Q} \leq Q_{\text{true}} + \frac{\gamma\epsilon_1}{1-\gamma}$.

2. **Implicitly Differentiable Adaptive Weighting**:

    - Function: Learns per-sample weights for MLE training data via bilevel optimization, enabling the model to be more accurate in value-relevant regions.
    - Mechanism: The inner loop updates $\psi$ (dynamics model) via weighted MLE: $\min_\psi \mathbb{E}[w_\nu(s,a,s')\log\hat{T}_\psi(s'|s,a)]$. The outer loop updates $\nu$ (weighting network) by minimizing $\mathcal{L}_{\text{RVL}}$. Gradients are propagated from the outer to the inner level via the implicit function theorem, without unrolling the optimization trajectory.
    - Design Motivation: Uniform MLE allocates equal effort to all data; adaptive weighting directs the model to be more accurate on value-sensitive state-action pairs, reducing model exploitation during rollouts.
    - Proposition 4.3 proves a convergence rate of $\mathcal{O}(1/\sqrt{K})$.

### Loss & Training
- Outer level: $\min_\nu \mathcal{L}_{\text{RVL}}(\psi^*(\nu), \nu)$
- Inner level: $\min_\psi \mathcal{L}_{\text{WSL}}(\psi, \nu) = \mathbb{E}[w_\nu \log\hat{T}_\psi]$
- Policy optimization: rollouts with the learned model, standard CQL training

## Key Experimental Results

### Main Results
D4RL MuJoCo (aggregate score over 12 tasks):

| Method | Total Score↑ | Category |
|--------|-------------|----------|
| RAMBO | 804.1 | Model-based adversarial |
| MOBILE | 857.7 | Model-based |
| Count-MORL | 927.5 | Model-based |
| **ROMI** | **953.5** | Model-based |

Highlights: hopper-mr 102.0 (vs. RAMBO 77.2), walker2d-me 113.3 (vs. RAMBO 73.7).

NeoRL (aggregate score over 9 tasks): **472.2** (vs. RAMBO 382.8, vs. CQL 466.3).

### Ablation Study

| Configuration | Effect | Remarks |
|---------------|--------|---------|
| Remove adaptive weighting | Increased multi-step rollout prediction error; performance degrades | Demonstrates importance of dynamics-awareness |
| $\xi$ sensitivity | Stable across 0.01–10 | RAMBO collapses at $\lambda \geq 0.05$ |
| Large $\xi$ | Lower Q-values but no explosion | Conservatism remains controllable |

### Key Findings
- ROMI outperforms RAMBO on 11 out of 12 D4RL tasks (+18.6% aggregate score)
- The controllability of $\xi$ is the core distinction from RAMBO — Q-values neither explode nor collapse
- Adaptive weighting is especially critical in multi-step rollouts, reducing model exploitation in OOD regions

## Highlights & Insights
- **Wasserstein duality transformation** is particularly elegant: it converts unstable "adversarial optimization in model space" into stable "state-space perturbations," reducing conservatism control to a single radius parameter $\xi$. This idea is transferable to other settings requiring robustness control.
- **Bilevel optimization enables value-dynamics co-adaptation**: the outer level drives the model to focus on value-sensitive regions while the inner level preserves dynamics accuracy — two inherently conflicting objectives are decoupled through hierarchical decomposition.
- ROMI remains stable in the hyperparameter regime where RAMBO fails entirely ($\lambda \geq 0.05$), substantially improving practical usability.

## Limitations & Future Work
- Bilevel optimization introduces additional computational overhead (implicit differentiation required at each step)
- $\xi$ must be specified in advance; no runtime adaptive adjustment is provided
- Evaluation is limited to standard benchmarks such as MuJoCo and NeoRL; testing on more complex environments is absent

## Related Work & Insights
- **vs. RAMBO**: ROMI fully resolves RAMBO's sensitivity to $\lambda$, achieving an 18.6% improvement in aggregate score
- **vs. MOBILE**: MOBILE employs model uncertainty estimation (ensemble disagreement), whereas ROMI uses value-aware Wasserstein robustness, yielding superior results
- **vs. Count-MORL**: Count-MORL relies on count-based exploration bonuses; ROMI achieves analogous effects through adaptive weighting in a more principled manner

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Wasserstein duality and implicitly differentiable adaptive weighting is novel, though individual components build on prior work
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of D4RL and NeoRL with thorough stability analysis
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous; the analysis of RAMBO's failure modes is in-depth
- Value: ⭐⭐⭐⭐ Provides a stable and reliable approach to conservatism control in model-based offline RL

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] BA-MCTS: Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL](bayes_adaptive_monte_carlo_tree_search_for_offline_model-based_reinforcement_lea.md)
- [\[ICLR 2026\] Transitive RL: Value Learning via Divide and Conquer](transitive_rl_value_learning_via_divide_and_conquer.md)
- [\[ICLR 2026\] From Observations to Events: Event-Aware World Model for Reinforcement Learning](from_observations_to_events_event-aware_world_model_for_reinforcement_learning.md)
- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)

<!-- RELATED:END -->
