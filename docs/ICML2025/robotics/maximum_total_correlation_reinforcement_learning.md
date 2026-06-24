---
title: >-
  [Paper Note] Maximum Total Correlation Reinforcement Learning
description: >-
  [ICML2025][Robotics][Total Correlation] This paper proposes maximizing trajectory Total Correlation as an inductive bias for RL, which encourages the policy to generate simple, compressible trajectories. This significantly enhances zero-shot robustness against observational noise, action noise, and dynamics changes without sacrificing task performance.
tags:
  - "ICML2025"
  - "Robotics"
  - "Total Correlation"
  - "trajectory consistency"
  - "information-theoretic regularization"
  - "SAC"
  - "robustness"
  - "compressible behavior"
date: 2026-05-08
content_hash: b69ce626ed15aedf
---

# Maximum Total Correlation Reinforcement Learning

**Conference**: ICML2025  
**arXiv**: [2505.16734](https://arxiv.org/abs/2505.16734)  
**Code**: [GitHub](https://github.com/BangYou01/MTC)  
**Area**: Reinforcement Learning  
**Keywords**: Total Correlation, trajectory consistency, information-theoretic regularization, SAC, robustness, compressible behavior

## TL;DR

This paper proposes maximizing trajectory Total Correlation as an inductive bias for RL, which encourages the policy to generate simple, compressible trajectories. This significantly enhances zero-shot robustness against observational noise, action noise, and dynamics changes without sacrificing task performance.

## Background & Motivation

Reinforcement learning policies are prone to picking up spurious correlations in high-dimensional perceptual inputs, causing fragile policies to fail under minor state variations. Existing methods introduce simplicity inductive biases from different perspectives:

- **RPC** (Eysenbach et al., 2021): Minimizes the mutual information between the raw state sequence and the embedding sequence, but only focuses on state consistency and utilizes only single-step transition information.
- **LZ-SAC / SPAC** (Saanum et al., 2023): Boosts the predictability of action sequences, but only focuses on action consistency.
- **Domain Randomization**: Improves transferability through diverse training distributions, but cannot guarantee coverage of all real-world scenarios.

**Core Problem**: The aforementioned methods focus either solely on state consistency or solely on action consistency, lacking a unified metric for the complete behavior (the entire trajectory of states and actions).

**Core Idea**: Use Total Correlation from information theory to measure the compressibility of the entire trajectory—a larger total correlation indicates more consistent, more compressible trajectories that tend toward open-loop behaviors (such as clean, periodic gaits), naturally leading to greater robustness.

## Method

### Problem Modeling: MTC-RL

Building upon the standard MDP $\mathcal{M}=(\mathcal{S},\mathcal{A},p,r,T)$, a state encoder $f_\theta(z_t|s_t)$ is introduced to map original states to latent variables $z_t$, and the policy $\pi_\phi(a_t|s_t)$ makes decisions based on the latent representations. The optimization objective of MTC-RL is:

$$\max_{\theta,\phi} \; \mathbb{E}_{\pi_\phi,f_\theta}\left[\sum_{t=1}^T r(s_t,a_t) + \alpha \cdot \mathcal{C}(z_1;a_1;\dots;a_{T-1};z_T)\right]$$

where $\mathcal{C}$ is the total correlation of the latent states and actions along the trajectory:

$$\mathcal{C}(x_1;\dots;x_n) = \mathbb{E}\left[\log \frac{p(x_1,\dots,x_n)}{\prod_{i=1}^n p(x_i)}\right]$$

Intuitive meaning: The amount of information saved by joint encoding compared to independent encoding.

### Variational Lower Bound Derivation

Since the total correlation cannot be directly decomposed into step-wise rewards, two parameterized models are introduced to construct a variational lower bound:

1. **History-based Latent Dynamics Model** $q_\eta(z_{t+1}|z_{1:t},a_{1:t})$: Uses history to predict the next latent state.
2. **History-based Action Prediction Model** $q_\chi(a_t|z_{1:t},a_{1:t-1})$: Uses history to predict the current action.

The lower bound is formulated as:

$$\widetilde{\mathcal{C}} = \mathbb{E}_{\pi,f}\left[\sum_{t=1}^{T-1}\log\frac{q_\eta(z_{t+1}|z_{1:t},a_{1:t})\cdot q_\chi(a_t|z_{1:t},a_{1:t-1})}{f_\theta(z_{t+1}|s_{t+1})\cdot \pi_\phi(a_t|s_t)}\right]$$

**Intuition**: When the latent states and actions are highly predictable from history (large numerator), and the irreducible uncertainty of the encoder/policy is relatively high (large denominator attenuating the contribution), the lower bound value is large, indicating high trajectory consistency.

### Regularized Reward Function

Substituting the lower bound into the objective yields the information-regularized reward:

$$r^*(s_t,a_t,s_{t+1}) = r(s_t,a_t) + \alpha\left(\log\frac{q_\eta(z_{t+1}|z_{1:t},a_{1:t})\cdot q_\chi(a_t|z_{1:t},a_{1:t-1})}{f_\theta(z_{t+1}|s_{t+1})\cdot \pi_\phi(a_t|s_t)}\right)$$

This reward biases the policy toward: (1) generating transitions where the latent states are predictable from history; (2) selecting actions that are predictable from history.

### MTC-SAC Implementation

Implemented on top of SAC, with key modifications:

- **Policy Evaluation**: Only replaces the reward function $r \to r^*$, keeping others (target net, dual Q-net) unchanged.
- **Policy Improvement**: Jointly optimizes the policy $\pi_\phi$, encoder $f_\theta$, dynamics model $q_\eta$, and action prediction model $q_\chi$.
- **Adaptive $\alpha$**: Automatically tuned via dual optimization to minimize $L(\alpha) = \alpha(\widetilde{\mathcal{C}} - I_p)$, where $I_p$ is a target lower bound value.

### Theoretical Connection to RPC

The regularization term of MTC is a generalization of RPC. RPC only uses current-step information for single-step prediction, whereas MTC uses the complete history for sequential prediction and additionally incorporates an action prediction model. The authors prove in the appendix that RPC can be derived from the MTC framework, but not vice-versa.

## Key Experimental Results

### Base Performance (8 DMC tasks, 1M steps, 20 seeds)

| Task | MTC | RPC | LZ-SAC | SPAC | SAC |
|------|-----|-----|--------|------|-----|
| Acrobot Swingup | **184±24** | 132±31 | 100±22 | 110±29 | 154±29 |
| Hopper Stand | **933±12** | 568±96 | 593±88 | 213±69 | 683±114 |
| Finger Spin | **985±2** | 869±19 | 805±38 | 136±121 | 955±18 |
| Walker Walk | **967±2** | 940±21 | 939±26 | 883±76 | 962±7 |
| Cheetah Run | **874±21** | 772±57 | 787±17 | 458±52 | 811±36 |
| Quadruped Walk | **944±5** | 842±77 | 595±110 | 505±185 | 738±93 |
| Walker Run | **790±9** | 778±25 | 732±22 | 347±95 | 767±13 |
| Walker Stand | 983±2 | 980±5 | 977±2 | 931±38 | **985±2** |

MTC achieves the best or highly competitive performance in 7 out of 8 tasks.

### Zero-Shot Robustness

- **Observational Noise**: MTC achieves the best aggregated performance across all noise intensities $\sigma \in [0.02, 0.1]$.
- **Action Noise**: MTC maintains the highest average reward under strong action perturbations.
- **Dynamics Mismatch** (mass scaling $0.25\times$ to $1.75\times$): MTC scores the highest under minor dynamics variations.
- **Spurious Correlation**: After adding uncontrollable Gaussian state dimensions in Walker Stand, MTC significantly outperforms RPC and SAC.

### Trajectory Compressibility

Using bzip2 lossless compression on trajectory files, trajectories generated by the MTC policy result in the smallest compressed file sizes, indicating that its behavior has the most periodic and structured characteristics.

### Manipulation Tasks (MetaWorld) and Visual Tasks

- MetaWorld (3 manipulation tasks): The success rate of MTC is on par with or better than the baselines, demonstrating that the method is not limited to periodic tasks.
- 6 pixel-input DMC tasks (Planet benchmark): MTC outperforms RPC, CURL, SAC-AE, and SAC.

## Highlights & Insights

1. **Unified Perspective**: Total Correlation unifies state consistency and action consistency into a single information-theoretic framework, eliminating the need to handle them separately.
2. **Theoretical Elegance**: By introducing a variational lower bound, the non-decomposable trajectory-level objective is converted into step-wise additive regularized rewards, seamlessly integrating with standard RL pipelines.
3. **No Network Changes Necessary**: Built on top of SAC, it only requires modifying the reward function and adding auxiliary models, making the engineering implementation straightforward.
4. **Adaptive Weighting**: $\alpha$ is automatically adjusted via Lagrangian dual optimization, removing the need for manual hyperparameter tuning.
5. **Interpretability**: Generates visually periodic and clean trajectories that are easier for humans to understand and predict.

## Limitations & Future Work

1. **Lower Bound is Always Negative**: Since it is a sum of negative KL divergences, the actual total correlation value cannot be estimated, serving only as an optimization direction.
2. **History-Dependent Rewards**: The regularized reward depends on the full history, which increases computational overhead; however, ablation studies show that the performance difference is minimal when the policy takes only the current state as input.
3. **Non-stationarity in Rewards**: The reward changes as the policy/encoder parameters evolve, which theoretically could cause training instability, though this was not observed in experiments.
4. **Off-policy Training of Prediction Models**: Training prediction models on the replay buffer deviates from the derivation assumptions, which might widen the lower bound gap.
5. **Diminished Edge under Massive Dynamics Mismatch**: The performance advantage diminishes under extreme dynamics changes (e.g., when mass scaling exceeds $1.5\times$).

## Related Work & Insights

- **RPC** (NeurIPS 2021): A special case of MTC (single-step only, no action prediction); RPC can be derived from MTC.
- **LZ-SAC / SPAC**: Focuses solely on action consistency, whereas MTC optimizes both state and action consistency.
- **InfoMax Principle**: MTC extends this to maximizing trajectory-level mutual dependencies.
- **Representation Learning (e.g., DiMAE, CURL)**: The encoder training objective in MTC optimizes representation consistency and policy performance simultaneously.

## Rating

- Novelty: ⭐⭐⭐⭐ — Total Correlation as an RL regularizer is a fresh perspective that unifies state and action consistency.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Thoroughly evaluated on 8+3+6 tasks, featuring multidimensional robustness tests, compressibility analysis, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — The theoretical derivation is clear and the motivation is well-articulated, although the discussion of the negative lower bound could have been presented earlier.
- Value: ⭐⭐⭐⭐ — Provides a simple, generic information-theoretic tool for RL robustness with strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](../../NeurIPS2025/robotics/reinforcement_learning_with_action_chunking.md)
- [\[ICML 2025\] Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning](gradual_transition_from_bellman_optimality_operator_to_bellman_operator_in_onlin.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/robotics/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[AAAI 2026\] Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity](../../AAAI2026/robotics/robust_out-of-order_retrieval_for_grid-based_storage_at_maximum_capacity.md)

</div>

<!-- RELATED:END -->
