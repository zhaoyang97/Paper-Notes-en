---
title: >-
  [Paper Note] LEAST: The Courage to Stop — Overcoming Sunk Cost Fallacy in Deep RL
description: >-
  [ICML 2025][Reinforcement Learning][Sunk Cost Fallacy] Proposes Learn to Stop (LEAST), a lightweight adaptive episode early stopping mechanism: it maintains buffers of Q-values and gradient magnitudes for the most recent $K$ episodes, and constructs a quality threshold $\epsilon_i$ and a learning potential weight $\omega_i$ using step-level medians. An episode is terminated and reset when the current Q-value is lower than $\omega_i \times \epsilon_i$. It yields significant im…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Sunk Cost Fallacy"
  - "Early Stopping"
  - "Q-Value Threshold"
  - "Gradient Statistics"
  - "Replay Buffer"
date: 2026-05-08
content_hash: 6dfdb4d3bfd31c88
---

# LEAST: The Courage to Stop — Overcoming Sunk Cost Fallacy in Deep RL

**Conference**: ICML 2025  
**arXiv**: [2506.13672](https://arxiv.org/abs/2506.13672)  
**Code**: None  
**Area**: Reinforcement Learning / Sample Efficiency  
**Keywords**: Sunk Cost Fallacy, Early Stopping, Q-Value Threshold, Gradient Statistics, Replay Buffer

## TL;DR

Proposes Learn to Stop (LEAST), a lightweight adaptive episode early stopping mechanism: it maintains buffers of Q-values and gradient magnitudes for the most recent $K$ episodes, and constructs a quality threshold $\epsilon_i$ and a learning potential weight $\omega_i$ using step-level medians. An episode is terminated and reset when the current Q-value is lower than $\omega_i \times \epsilon_i$. It yields significant improvements for TD3, SAC, and REDQ across four MuJoCo tasks (improving normalized scores from 0.65 to over 0.70) and accelerates convergence by approximately 30% on the Finger Turn Hard task in DMC visual RL.

## Background & Motivation

**Background**: Off-policy deep RL relies on replay buffers to reuse historical experiences to improve sample efficiency. However, traditional RL frameworks force the agent to run through each episode completely, even when trajectories have fallen into sub-optimal regions.

**Limitations of Prior Work**: When an agent gets trapped in a low-quality trajectory, (1) continuing the interaction wastes environmental interaction budget; (2) low-quality transitions "pollute" the replay buffer, leading to an extremely high proportion of uninformative samples in the training data (up to 30-40% according to experiments); (3) Parameter-sensitive prioritized sampling methods (like PER) cannot fundamentally resolve the quality issue at the data source.

**Key Challenge**: Traditional RL lacks an autonomous stopping mechanism. The agent cannot determine "whether it is worth continuing to run," similar to the human sunk cost fallacy—being unwilling to abandon a trajectory destined for low returns simply because steps have already been invested.

**Goal**: To equip the agent with an adaptive stopping capability based on historical statistics, actively terminating and resetting when the current trajectory quality is detected to be below the historical median and has low learning value.

**Key Insight**: Utilize Q-values to measure trajectory quality and gradient magnitudes to measure learning potential, combining both to construct a dual-criterion stopping threshold.

**Core Idea**: Teach the agent to "cut losses in time," utilizing existing Q-values and gradient info to determine when to abandon the current episode with zero computational cost.

## Method

### Overall Architecture

LEAST is a plug-and-play module that can be embedded into any off-policy RL algorithm. At each step of every episode, LEAST performs: (1) calculating the Q-value $\hat{Q}_i$ and gradient magnitude $G_i$ of the current $(s_i, a_i)$; (2) comparing them with the step-level median of the historical buffer; (3) terminating the episode and resetting the environment if they are below the adaptive threshold; (4) dynamically adjusting the buffer size through an entropy-aware mechanism and scheduling noise to help escape sub-optimal policies.

### Key Designs

1. **Q-Value Quality Threshold (Step-level Median)**:

    - **Function**: Constructing an independent quality reference baseline for each step $i$ within an episode.
    - **Mechanism**: Maintaining a 2D buffer $\mathcal{B}_Q \in \mathbb{R}^{K \times L}$ (where $K$ is the number of recent episodes, and $L$ is the maximum steps). The threshold for each step is $\epsilon_i = \text{Median}(\mathcal{B}_Q[:, i])$. When $\hat{Q}_i < \epsilon_i$, it indicates that the expected return of the current trajectory is lower than the historical median, triggering termination.
    - **Design Motivation**: Since the scale and behavioral requirements of Q-values vary across different steps, step-independent thresholds are required instead of a globally unified threshold. Using the median rather than the mean avoids threshold fluctuations caused by outliers (experiments verify that the median is more stable than the mean).

2. **Gradient Learning Potential Weight**:

    - **Function**: Modulates the Q-value threshold to balance "quality" and "learning value".
    - **Mechanism**: Maintaining a gradient buffer $\mathcal{B}_G$ and calculating a dynamic weight $\omega_i = \frac{\text{Median}(\mathcal{B}_G[:, i])}{G_i}$. When $\omega_i < 1$ (current gradient > historical median), it suggests that the state is novel and worth exploring, lowering the threshold to encourage continuation; when $\omega_i > 1$, the threshold is tightened. The full stopping criterion is: $\hat{Q}_i < \omega_i \times \epsilon_i$ (when $\epsilon_i \geq 0$) or $\hat{Q}_i < \omega_i^{-1} \times \epsilon_i$ (when $\epsilon_i < 0$).
    - **Design Motivation**: Stopping purely based on Q-values would miss valuable exploration opportunities—a low Q-value coupled with a high gradient implies the agent is learning new knowledge and should not be terminated.

3. **Entropy-Aware Dynamic Buffer + Noise Scheduling**:

    - **Function**: Adaptively adjusting the statistical window size and exploration noise.
    - **Mechanism**: Calculating the entropy $H_t$ of $\mathcal{B}_Q$ every $c$ steps. If $H_t > (1+\gamma) \times \bar{H}$, the buffer is expanded to obtain smoother statistical estimates. Meanwhile, if the agent is frequently terminated early (indicating the policy might be stuck in a sub-optimal state), the exploration noise $\sigma$ is increased via a sigmoid schedule to help escape the current behavioral pattern.
    - **Design Motivation**: Unstable policies cause large variances between adjacent trajectories, requiring a larger window to stabilize the median estimate. Frequent early stopping indicates that the policy needs more stochasticity to discover new behaviors.

### Loss & Training

LEAST does not introduce any additional training losses or network parameters. It purely leverages the existing Q-values and TD losses (as a proxy for gradient magnitudes) computed during RL training, thus incurring almost zero computational overhead. Upon termination, the partially collected trajectory is stored in the replay buffer as usual, and the environment is then reset to start a new episode.

## Key Experimental Results

### Main Results (MuJoCo-v4, 5 seeds)

| Algorithm | Ant | Walker2d | HalfCheetah | Humanoid | Normalized Mean |
|------|-----|----------|-------------|----------|-----------|
| TD3 | ~4500 | ~4000 | ~11000 | ~5000 | 0.58 |
| TD3+LEAST | **~6800** | **~5200** | **~12000** | **~5800** | **0.70** |
| SAC | ~5200 | ~4500 | ~11500 | ~5200 | 0.63 |
| SAC+LEAST | **~6500** | **~5500** | **~12500** | **~5800** | **0.71** |
| REDQ | ~6000 | ~5000 | ~12000 | ~5500 | 0.68 |
| REDQ+LEAST | **~6800** | **~5500** | **~12500** | **~6000** | **0.73** |

### Visual RL (DeepMind Control Suite, DrQv2 Backbone)

| Method | Finger Turn Hard | Quadruped Run | Normalized Mean |
|------|-----------------|---------------|-----------|
| DrQv2 | ~700 | ~900 | 0.724 |
| CURL | ~720 | ~880 | 0.71 |
| A-LIX | ~740 | ~910 | 0.73 |
| DrQv2+LEAST | ~780 | ~953 | 0.756 |
| TACO | **~800** | **~960** | **0.78** |

LEAST converges approximately 30% faster than DrQv2 on Finger Turn Hard, achieving a normalized score close to TACO.

### Ablation Study

| Configuration | Normalized Mean | Description |
|------|-----------|------|
| Q-value threshold only (no $\omega$) | 0.65 | Lacks learning potential modulation, terminating too aggressively |
| Q-value + $\omega$ modulation | 0.68 | Dual criteria significantly outperforms single criterion |
| + Dynamic buffer + noise scheduling | **0.70** | Full LEAST is the most stable |
| Mean instead of median | 0.62 | Outliers lead to unstable thresholds |

| Hyperparameter | Optimal Range | Sensitivity |
|--------|---------|--------|
| $\omega$ weight coefficient | [0.3, 0.6] | Medium (SAC is more sensitive than TD3) |
| Noise upper bound $\bar{\sigma}$ | TD3: [0.25, 0.35], SAC: [0.15, 0.25] | Low |
| Startup time | MuJoCo: 10-20% training, DMC: 5-15% | Low |
| Initial buffer size | 250 episodes | Medium |
| Entropy overflow rate $\gamma$ | [0, 0.1] | Low |

### Key Findings

- LEAST is consistently effective across three types of off-policy algorithms (deterministic policy TD3, stochastic policy SAC, and ensemble REDQ).
- Replay buffer analysis: LEAST significantly reduces "white zone" samples (uninformative transitions with low Q-value + low loss) and increases "black zone" samples (high-quality learning signals with high Q-value + high loss).
- Sample efficiency: TD3+LEAST reduces the number of steps required to reach the final performance of vanilla TD3 by 30-50% on average.
- In visual RL, it approaches the performance of methods requiring representation learning modifications like TACO, without needing extra networks.
- $\omega \in [0.3, 0.6]$ is robust on both TD3 and SAC; SAC is insensitive to noise scheduling because the stochastic policy itself provides diversity.

## Highlights & Insights

- **The analogy of "sunk cost fallacy" in RL** is highly inspiring—forcing the agent to finish an episode in conventional RL is indeed analogous to the irrational behavior of 'finishing a terrible movie just because you bought the ticket'.
- **Zero additional computational cost**—by utilizing only the Q-values and TD losses already computed during training, it requires no extra neural networks or training.
- **The choice of median vs. mean** carries practical value—in high-variance scenarios like reinforcement learning, the median exhibits far better robustness to anomalous trajectories than the mean.
- **Complementary to curiosity-driven exploration**—while curiosity encourages exploring new states, LEAST encourages abandoning familiar low-quality trajectories. Being orthogonal, they can be effectively combined.

## Limitations & Future Work

- Although the termination threshold is not highly sensitive, there are still multiple hyperparameters ($\omega$ coefficient, buffer size, startup time, etc.) lacking automated selection methods.
- It has only been validated on off-policy algorithms; its applicability to on-policy methods (e.g., PPO) remains unexplored.
- In extremely sparse reward environments, the Q-value signal itself is unreliable, which may cause LEAST's termination decisions to fail.
- After resetting, the agent might fall back into the same sub-optimal trajectory because the policy has not changed—noise scheduling only partially alleviates this.
- LEAST may increase score variance (box plots show greater dispersion), and training stability needs improvement.
- The "blue zone" samples (high loss but low Q-values) in the replay buffer are not yet effectively addressed.

## Related Work & Insights

- **vs. PER (Prioritized Experience Replay)**: PER filters low-quality transitions on the sampling side, whereas LEAST avoids generating low-quality transitions at the generation source. They are complementary.
- **vs. ICM (Curiosity-Driven Exploration)**: ICM encourages exploring new states, while LEAST encourages abandoning old, bad states. Their directions are complementary.
- **vs. DroQ / REDQ**: These methods improve efficiency through better Q-function fitting, whereas LEAST improves efficiency through better data collection strategies. Orthogonal combinations are effective.
- **vs. DrQv2 / TACO**: Visual RL methods improve efficiency via encoder modifications, whereas LEAST achieves comparable performance at a lower cost through improved data strategies.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The analogy to the sunk cost fallacy in RL is novel, and the method is simple and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ MuJoCo 4 tasks × 4 algorithms + DMC 4 tasks + detailed ablation study, providing comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, precise analogies, step-by-step argumentation, and an opening quote that adds charm.
- Value: ⭐⭐⭐⭐ A plug-and-play general-purpose improvement that directly contributes to off-policy RL practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](../../NeurIPS2025/reinforcement_learning/deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[NeurIPS 2025\] Adaptive Cooperative Transmission Design for URLLC via Deep RL](../../NeurIPS2025/reinforcement_learning/adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)
- [\[ICLR 2026\] GAS: Enhancing Reward-Cost Balance of Generative Model-assisted Offline Safe RL](../../ICLR2026/reinforcement_learning/gas_enhancing_reward-cost_balance_of_generative_model-assisted_offline_safe_rl.md)
- [\[ICML 2025\] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning](network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)
- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)

</div>

<!-- RELATED:END -->
