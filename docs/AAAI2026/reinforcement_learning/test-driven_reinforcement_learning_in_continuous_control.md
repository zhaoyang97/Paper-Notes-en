---
title: >-
  [Paper Note] Test-driven Reinforcement Learning in Continuous Control
description: >-
  [AAAI 2026 (Oral)][Reinforcement Learning][Test-driven RL] This paper proposes the Test-driven Reinforcement Learning (TdRL) framework, which replaces a single reward function with multiple test functions — pass-fail tests defining optimality criteria and indicative tests guiding learning — to represent task objectives. A return function is learned via lexicographic-heuristic trajectory comparison, matching or surpassing hand-crafted reward methods on the DeepMind Control Suite while naturally supporting multi-objective optimization.
tags:
  - AAAI 2026 (Oral)
  - Reinforcement Learning
  - Test-driven RL
  - Satisficing Theory
  - Multi-objective Optimization
  - Trajectory Return Function
  - Lexicographic Comparison
date: 2026-05-08
content_hash: 945cf5503b080059
---

# Test-driven Reinforcement Learning in Continuous Control

**Conference**: AAAI 2026 (Oral)  
**arXiv**: [2511.07904](https://arxiv.org/abs/2511.07904)  
**Code**: [https://github.com/KezhiAdore/TdRL](https://github.com/KezhiAdore/TdRL)  
**Area**: Reinforcement Learning / Continuous Control / Task Representation  
**Keywords**: Test-driven RL, Satisficing Theory, Multi-objective Optimization, Trajectory Return Function, Lexicographic Comparison

## TL;DR

This paper proposes the Test-driven Reinforcement Learning (TdRL) framework, which replaces a single reward function with multiple test functions — pass-fail tests defining optimality criteria and indicative tests guiding learning — to represent task objectives. A return function is learned via lexicographic-heuristic trajectory comparison, matching or surpassing hand-crafted reward methods on the DeepMind Control Suite while naturally supporting multi-objective optimization.

## Background & Motivation

In RL, the reward function bears a dual responsibility: (1) defining optimal behavior; and (2) guiding the learning process. This dual role makes reward design extremely difficult, frequently causing the following issues:

- **Reward Hacking**: agents exploit loopholes in the reward function rather than genuinely completing the task
- **Design Bias**: experts tend to evaluate individual state-action pairs in isolation, neglecting their effect on the entire trajectory
- **Multi-objective Dilemma**: real-world tasks (e.g., autonomous driving must simultaneously account for safety, speed, comfort, and regulations) make it extremely difficult to determine weights among objectives
- **Limitations of PbRL**: relies on human preference labels, which are subject to subjective bias
- **Limitations of IRL**: requires large amounts of expert demonstrations and generalizes poorly
- **LLM-generated Rewards**: still require manual domain knowledge and extensive iterative training feedback

Core Insight (Satisficing Theory): In multi-objective settings, humans do not pursue the optimum on a single metric but instead seek a "satisfactory solution" across objectives. For instance, drivers do not blindly minimize travel time but arrive on schedule subject to safety, comfort, and compliance constraints.

## Method

### Overall Architecture

A four-stage iterative pipeline:
1. **Collect Trajectory**: the policy interacts with the environment to collect trajectories
2. **Return Learning**: update the return function $R_\xi^{ind}$ based on trajectory comparison outcomes
3. **Reward Learning**: decompose trajectory returns into a state-action reward function $r_\phi(s,a)$
4. **Policy Optimization**: optimize the policy using SAC/PPO based on the learned reward

### Key Designs

1. **Test Function Taxonomy**:

    - **Pass-fail tests** $z^{pf}: \tau \to \{0, 1\}$: hard constraints defining optimal behavior (e.g., "did the agent reach the goal?" "is the torso upright?"), with binary outcomes
    - **Indicative tests** $z^{ind}: \tau \to \mathbb{R}$: provide continuous metrics to guide learning (e.g., "locomotion speed," "energy consumption"); they do not define optimality but help discriminate between trajectories
    - The two test types have distinct roles: pass-fail tests define "what is correct," while indicative tests facilitate "how to learn efficiently"
    - No pre-specified weights are required for combination — lexicographic ordering handles priority automatically

2. **Theoretical Guarantee (Theorem 1)**:

    - If the trajectory return function $R(\tau)$ assigns higher returns to trajectories closer to the optimal trajectory set $\tilde{\mathcal{T}}$ (distance monotonicity)
    - Then maximum-entropy policy optimization yields policies closer to the optimal policy set $\tilde{\Pi}$
    - Formally: $d(\tau_1, \tilde{\mathcal{T}}) \leq d(\tau_2, \tilde{\mathcal{T}}) \Rightarrow R(\tau_1) \geq R(\tau_2) \Rightarrow d(\pi_2, \tilde{\Pi}) \leq d(\pi_1, \tilde{\Pi})$

3. **Lexicographic Trajectory Comparison**:

    - Rather than directly computing $d(\tau, \tilde{\mathcal{T}})$ (since the optimal trajectory set is unknown), heuristic rules determine which of $\tau_1$ and $\tau_2$ is closer to $\tilde{\mathcal{T}}$
    - Priority rules: (1) both pass all pass-fail tests → tie; (2) the trajectory passing more pass-fail tests is preferred; (3) the trajectory passing harder tests (lower historical pass rate) is preferred; (4) rank by degree of optimization on indicative tests, prioritizing the least-optimized dimension
    - Outputs $\mu \in \{0, 0.5, 1\}$ as preference labels for the Bradley-Terry model

4. **Return Function Learning**:

    - A fully connected network $R_\xi^{ind}$ maps $n$ indicative test results to a scalar return
    - Loss: $\mathcal{L}_R^{Dis}$ (distance-based cross-entropy) + $\mathcal{L}_R^{Penalty}$ (numerical stability term)
    - Decomposition into state-action reward: $\mathcal{L}_r = \sum[R(\tau) - \sum r_\phi(s,a)]^2$
    - Two gradient balancing methods: GN (gradient norm normalization) and ES (early stopping, $K^{ES}=10$)

### Loss & Training

- Return function: $\mathcal{L}_R^{Dis}$ (BT-model cross-entropy) + $\mathcal{L}_R^{Penalty}$ (MSE regularization)
- Reward decomposition: $\mathcal{L}_r = \text{MSE}(R(\tau),\, \sum r_\phi(s,a))$
- Policy optimization: standard SAC objective (with maximum entropy), or PPO

Key hyperparameters: 9,000 steps of unsupervised pre-training; trajectory buffer maximum capacity 100; segment size 50; reward/return network lr = $3 \times 10^{-4}$; reward ensemble = 3

## Key Experimental Results

### Main Results: DM-Control Continuous Control Tasks

| Task | SAC + Oracle Reward | TdRL-GN | TdRL-ES | PPO + Oracle | PPO + TdRL |
|------|:-:|:-:|:-:|:-:|:-:|
| Walker-Stand | ~980 | ≈980 | ≈980 | ~970 | ~960 |
| Walker-Run | ~650 | ≈670 | ≈650 | ~550 | ~480 |
| Cheetah-Run | ~830 | ≈830 | ≈850 | ~750 | ~720 |
| Quadruped-Run | ~780 | ≈800 | ≈770 | ~650 | ~600 |

### Ablation Study

| Variant | Walker-Run Performance | Notes |
|------|:-:|------|
| TdRL-ES ($K^{ES}=10$) | **~650** | Recommended default |
| Without Penalty term | Unstable training | Return values grow unbounded |
| Direct reward learning (no return decomposition) | Unstable training | Requires tanh clipping + repeated rescaling |
| $K^{ES}$ too large | Performance drops | Insufficient Penalty constraint |
| $K^{ES}$ too small | Performance drops | Over-constrained, return learning impeded |

### Multi-objective Analysis (Walker-Run)

| Objective | Oracle SAC | TdRL | Threshold |
|------|:-:|:-:|:-:|
| Torso upright $\cos(\theta)$ | ✓ (satisfied) | ✓ (satisfied) | [0.9, 1.0] |
| Torso height | ✗ (not satisfied) | ✓ (satisfied) | >1.2 |
| X-axis velocity | ~8 | ~8 | 8 |

### Key Findings

- **TdRL matches or surpasses oracle reward**: achieves comparable performance without manual reward engineering
- **Natural multi-objective support**: in Walker-Run, the weight design of the oracle reward leads to "upright but crouched running" (high upright score, low stand height); TdRL's pass-fail tests ensure all three objectives are satisfied
- TdRL's policy stability is slightly lower than that of oracle reward — trajectory-level evaluation inherently has greater variance than state-action evaluation, but this precisely mitigates reward hacking
- Both GN and ES gradient balancing methods are effective; ES is simpler (recommended: $K^{ES}=10$)
- TdRL is also applicable to on-policy methods (PPO), although the theory is limited to the maximum-entropy framework

## Highlights & Insights

- **A new task representation paradigm**: test functions vs. reward functions, with functional separation (defining objectives vs. guiding learning)
- The lexicographic comparison is inspired by human decision-making: hard constraints are checked first, then soft metrics — consistent with the human heuristic of "ensure safety before seeking optimality"
- Complete theoretical contribution: Theorem 1 provides convergence guarantees, rigorously derived from max-entropy RL
- Key distinction from PbRL: TdRL does not require human preference labels but automatically generates trajectory comparisons via test functions
- The conceptual contribution outweighs the engineering contribution — the paper introduces a new way of thinking about RL task design

## Limitations & Future Work

- Theory is grounded solely in the maximum-entropy RL framework; on-policy methods such as PPO lack theoretical guarantees
- Test functions still require manual design; though simpler than reward functions, the process is not fully automated
- Lexicographic comparison is a heuristic; theoretical optimality is not guaranteed
- Early-stage training is slower than with oracle reward (the return function must first be learned)
- Validation is limited to simulated environments; no experiments on real robots or autonomous driving scenarios

## Related Work & Insights

| Comparison Dimension | TdRL | PbRL | IRL | Reward Engineering |
|----------|:-:|:-:|:-:|:-:|
| No preference labels required | ✓ | ✗ | ✓ | ✓ |
| No expert demonstrations required | ✓ | ✓ | ✗ | ✓ |
| Trajectory-level evaluation | ✓ | ✓ | — | ✗ |
| Multi-objective support | Native | Requires design | — | Requires weights |

- Future direction: using LLMs to automatically generate test functions (noted in the paper)
- Broader potential of satisficing theory in RL

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (entirely new paradigm; Oral-level conceptual contribution)
- Technical Depth: ⭐⭐⭐⭐ (theoretical theorem + heuristic algorithm + complete implementation)
- Experimental Thoroughness: ⭐⭐⭐ (only 4 DM-Control tasks; real-world scenarios absent)
- Practical Value: ⭐⭐⭐⭐ (reduces task design complexity in multi-objective RL)
- The DMC benchmark is relatively simple; scalability to more complex tasks remains to be verified
- The lexicographic assumption presupposes a clear priority ordering among tests, which may be more complex in practice
- The relationship with and advantages over RLHF require further clarification

## Related Work & Insights
- vs. traditional reward design: simpler to design; objective definition and learning guidance are decoupled
- vs. RLHF: replaces human comparative feedback with pre-defined test functions
- vs. multi-objective RL: native framework support vs. requiring additional algorithmic adaptation

## Inspiration & Connections
The test-driven task representation approach is applicable to safety constraint definition in autonomous driving. It also offers an alternative perspective on the reward hacking problem.

## Rating ⭐⭐⭐⭐ (4/5)
An Oral paper with novel concepts and solid theoretical contributions. However, the experimental scope is limited (DMC only), and test function design still involves subjective factors.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](../../ICLR2026/reinforcement_learning/scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)
- [\[NeurIPS 2025\] Actor-Free Continuous Control via Structurally Maximizable Q-Functions](../../NeurIPS2025/reinforcement_learning/actorfree_continuous_control_via_structurally_maximizable_qf.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](../../ICLR2026/reinforcement_learning/learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[AAAI 2026\] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)

<!-- RELATED:END -->
