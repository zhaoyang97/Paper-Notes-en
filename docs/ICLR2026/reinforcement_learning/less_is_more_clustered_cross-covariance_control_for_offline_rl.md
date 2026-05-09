---
title: >-
  [Paper Note] Less is More: Clustered Cross-Covariance Control for Offline RL
description: >-
  [ICLR 2026][Reinforcement Learning][Offline reinforcement learning] This paper identifies that the standard squared-error TD objective introduces harmful cross-covariance in offline RL, and proposes C⁴ (Clustered Cross-Covariance Control for TD), which mitigates this effect via partitioned buffer sampling and an explicit gradient-based corrective penalty, achieving up to 30% return improvement in small-dataset and OOD-dominated settings.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Offline reinforcement learning
  - distributional shift
  - TD cross-covariance
  - buffer partitioning
  - conservatism control
date: 2026-05-08
content_hash: b89ad46ad1e2fef0
---

# Less is More: Clustered Cross-Covariance Control for Offline RL

**Conference**: ICLR 2026
**arXiv**: [2601.20765](https://arxiv.org/abs/2601.20765)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Offline reinforcement learning, distributional shift, TD cross-covariance, buffer partitioning, conservatism control

## TL;DR

This paper identifies that the standard squared-error TD objective introduces harmful cross-covariance in offline RL, and proposes C⁴ (Clustered Cross-Covariance Control for TD), which mitigates this effect via partitioned buffer sampling and an explicit gradient-based corrective penalty, achieving up to 30% return improvement in small-dataset and OOD-dominated settings.

## Background & Motivation

The central challenge of offline reinforcement learning is distributional shift: the policy must make decisions on state-action pairs not covered by the offline dataset, where value function estimates are unreliable. This problem is particularly severe in the following scenarios:

**Data scarcity**: The offline dataset is limited in scale and cannot adequately cover the state-action space.

**OOD-dominated settings**: The dataset distribution differs significantly from the target policy distribution, with a large fraction of updates falling in out-of-distribution regions.

**Excessive conservatism**: Existing methods (e.g., CQL) address distributional shift by penalizing Q-values in OOD regions, but this conservative strategy can lead to overly pessimistic value estimates in extreme OOD regions.

The key finding of this paper is that **the standard mean squared error (MSE) TD objective introduces harmful TD cross-covariance**. This cross-covariance effect is amplified in OOD regions, causing the optimization direction to deviate from the optimum and degrading the quality of policy learning. This mechanism has not been sufficiently recognized in prior work.

## Method

### Overall Architecture

The C⁴ method comprises two complementary strategies to counteract the harmful effects of TD cross-covariance:

1. **Partitioned Buffer Sampling**: The experience replay buffer is divided into multiple local partitions, with updates performed within each partition, thereby limiting the scope of cross-covariance interactions.
2. **Gradient-based Corrective Penalty**: At each update step, the gradient bias introduced by cross-covariance is explicitly estimated and canceled.

### Key Designs

1. **TD Cross-Covariance Analysis**:

    - **Problem identification**: In standard TD learning, the gradient of the MSE objective $L = \mathbb{E}[(Q(s,a) - (r + \gamma Q'(s',a')))^2]$ contains a cross-covariance term $\text{Cov}(Q(s,a), Q'(s',a'))$.
    - **Amplification mechanism**: In OOD regions, the estimation errors of $Q$ and $Q'$ are highly correlated, causing the cross-covariance term to become abnormally large.
    - **Harmful effect**: This additional covariance term deflects the gradient direction away from the true TD error descent direction, especially in data-scarce regions.
    - **Design Motivation**: A theoretical analysis precisely identifies the root cause, providing a principled foundation for the proposed solution.

2. **Partitioned Buffer Sampling**:

    - A clustering algorithm (e.g., K-means) is used to partition the transitions in the experience replay buffer according to the features of their state-action pairs.
    - Each update samples a mini-batch exclusively from one partition, ensuring that data within a batch originates from similar state-action regions.
    - **Core effects**:
        - Restricts the computation of cross-covariance to local partitions, suppressing irregular cross-region covariance.
        - Aligns update directions within each partition, improving optimization stability.
        - Prevents high covariance in OOD regions from contaminating in-distribution updates.
    - **Theoretical guarantee**: It is proved that buffer partitioning preserves the lower-bound property of the maximization objective.
    - **Design Motivation**: Simple, effective, and easy to integrate with existing implementations.

3. **Gradient-based Corrective Penalty**:

    - At each gradient update step, the bias term introduced by cross-covariance is explicitly estimated.
    - A corrective gradient is added to cancel this bias: $\nabla L_{\text{corrected}} = \nabla L_{\text{TD}} - \nabla L_{\text{cov-bias}}$.
    - **Core effect**: Precisely eliminates covariance bias at the mini-batch level.
    - **Design Motivation**: Provides more accurate bias correction, complementing partitioned sampling.

4. **Integration with Existing Methods**:

    - C⁴ is designed as a plug-and-play module compatible with any policy-constrained offline RL method.
    - Typical integration targets: CQL, IQL, TD3+BC, etc.
    - Does not alter the core behavior of the base algorithm; only mitigates covariance bias.
    - **Design Motivation**: Maximizes compatibility and practical utility.

### Loss & Training

- Base loss: standard TD error + policy constraint (depending on the base algorithm).
- Additional corrective term: $L_{\text{correction}} = \lambda \cdot \hat{\text{Cov}}(Q, Q')$.
- Partitioning strategy: clustering is performed once at the start of training, followed by round-robin sampling across partitions.
- Key hyperparameters: number of partitions $K$ and correction coefficient $\lambda$.

## Key Experimental Results

### Main Results

The paper conducts extensive evaluations on standard offline RL benchmarks including D4RL:

| Dataset Type | Baseline | Baseline + C⁴ | Gain | Notes |
|---|---|---|---|---|
| Small dataset (1%) | CQL | CQL + C⁴ | Up to 30% | Most significant under data scarcity |
| Small dataset (1%) | IQL | IQL + C⁴ | Significant | C⁴ compatible with multiple baselines |
| OOD-dominated | CQL | CQL + C⁴ | Significant | Outstanding when OOD regions dominate |
| Standard dataset | Various | Various + C⁴ | Consistent | Improvements even under normal conditions |
| Medium-Expert | TD3+BC | TD3+BC + C⁴ | Improvement | Mixed-quality data benefits |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Partitioning only | Significant gain | Partitioning alone effectively suppresses covariance |
| Gradient correction only | Notable gain | Precise correction is effective but less stable than partitioning |
| Partitioning + correction | Best | Two components are complementary |
| Varying partition count $K$ | Optimal $K$ exists | Too small loses partition effect; too large leaves too little data per partition |
| Varying dataset size | Larger gain for smaller datasets | Consistent with theory: less data yields more severe covariance bias |
| Varying OOD ratio | Larger gain for more OOD | Validates C⁴'s specificity to OOD regions |

### Key Findings

1. **Cross-covariance is an overlooked critical factor**: Prior offline RL research has focused primarily on Q-value overestimation and policy constraints, neglecting the harmful effects of cross-covariance.
2. **Small datasets benefit the most**: With only 1% of data, C⁴ yields the largest improvements (up to 30%), confirming that covariance bias is most severe under data scarcity.
3. **Alleviating excessive conservatism**: C⁴ reduces over-pessimism in extreme OOD regions, enabling better policy generalization.
4. **Improved stability**: In addition to return gains, training stability is also substantially improved.
5. **Universal enhancer**: C⁴ serves as a general-purpose plug-in that boosts multiple offline RL baselines, reflecting the broad applicability of its theoretical insight.

## Highlights & Insights

1. **Deep theoretical insight**: The paper reveals the harmful mechanism of cross-covariance in TD learning—a seemingly simple yet long-overlooked issue.
2. **Elegant solution**: Partitioned buffer sampling is remarkably simple, incurring almost no additional computational overhead while yielding significant improvements.
3. **Solid theoretical guarantees**: It is formally proved that partitioning preserves the lower-bound property and that the constraint alleviates over-conservatism without altering core behavior.
4. **Comprehensive experimental validation**: The method's effectiveness is verified across multiple baselines and diverse dataset configurations.
5. **"Less is more" philosophy**: The method name is apt—by restricting the scope of data used at each update (less), a better optimization direction is obtained (more).

## Limitations & Future Work

1. **Clustering quality**: The effectiveness of partitioning depends on clustering quality, which may be inaccurate in high-dimensional state spaces.
2. **Static partitioning**: Clustering is performed once at the start of training and is not dynamically adjusted during the training process.
3. **Online RL extension**: The theoretical analysis primarily targets the offline setting; performance in online or hybrid online/offline settings remains unclear.
4. **Partition count selection**: The choice of $K$ requires tuning, and an adaptive method for determining $K$ is currently lacking.
5. **Computational overhead**: While the sampling phase is lightweight, the initial clustering may be time-consuming for very large replay buffers.
6. **High-dimensional continuous control**: Validation on more complex continuous control tasks (e.g., robotic manipulation) is limited.

## Related Work & Insights

- **CQL (Conservative Q-Learning)**: Addresses distributional shift by penalizing Q-values in OOD regions; C⁴ approaches the same problem from a different angle (covariance control).
- **IQL (Implicit Q-Learning)**: Avoids querying OOD actions via quantile regression; C⁴ can be stacked on top as an enhancement module.
- **TD3+BC**: Adds behavioral cloning constraints on top of TD3; C⁴ is likewise integrable.
- **Prioritized Experience Replay**: Controls the sampling distribution via priority, related in spirit to C⁴'s partitioned sampling but with a different motivation.
- **Insights**:
    - Gradient analysis of TD learning may reveal additional overlooked bias terms worth investigating.
    - Structured buffer management (beyond mere prioritization) may become a new research direction in offline RL.
    - The principle of "restricting scope to improve quality" may generalize to other machine learning problems.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ACL 2026\] LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification](../../ACL2026/reinforcement_learning/less_noise_more_voice_reinforcement_learning_for_reasoning_via_instruction_purif.md)
- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)
- [\[ICLR 2026\] BA-MCTS: Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL](bayes_adaptive_monte_carlo_tree_search_for_offline_model-based_reinforcement_lea.md)

<!-- RELATED:END -->
