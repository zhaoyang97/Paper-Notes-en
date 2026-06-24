---
title: >-
  [Paper Note] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks
description: >-
  [ICML 2025][Reinforcement Learning][Parallel data collection] This work systematically investigates the impact of two dimensions of parallel data collection in on-policy RL (the number of parallel environments $N_{\text{envs}}$ vs. rollout length $N_{\text{RO}}$) on PPO performance. It is found that under a fixed data budget, increasing the number of parallel environments is more effective than increasing the rollout length, and larger datasets improve network plasticity and…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Parallel data collection"
  - "PPO"
  - "network plasticity"
  - "sample efficiency"
  - "bias-variance trade-off"
  - "Atari"
date: 2026-05-08
content_hash: a8ad1ddbe21ba5ff
---

# The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks

**Conference**: ICML 2025  
**arXiv**: [2506.03404](https://arxiv.org/abs/2506.03404)  
**Authors**: Walter Mayor, Johan Obando-Ceron, Aaron Courville, Pablo Samuel Castro  
**Area**: Reinforcement Learning  
**Keywords**: Parallel data collection, PPO, network plasticity, sample efficiency, bias-variance trade-off, Atari  

## TL;DR

This work systematically investigates the impact of two dimensions of parallel data collection in on-policy RL (the number of parallel environments $N_{\text{envs}}$ vs. rollout length $N_{\text{RO}}$) on PPO performance. It is found that under a fixed data budget, increasing the number of parallel environments is more effective than increasing the rollout length, and larger datasets improve network plasticity and optimization stability.

## Background & Motivation

**Background**: Parallel data collection is a standard technique in modern RL algorithms. GPU-accelerated simulators (such as Isaac Gym, EnvPool, PGX) enable running thousands of parallel environments on a single device.

**Limitations of Prior Work**: Although parallelization is widely used, how to choose the two key parameters of parallel collection (the number of parallel environments $N_{\text{envs}}$ and rollout length $N_{\text{RO}}$) lacks systematic research. Prior work (Singla et al., 2024) found that simply increasing the data volume leads to diminishing returns, but did not analyze the impact of data structure.

**Key Challenge**: In PPO, the data batch size is $|B| = N_{\text{envs}} \times N_{\text{RO}}$. Different combinations of these two factors trigger different trade-offs:
   - $N_{\text{envs}}$ affects data **diversity** (state-action space coverage)
   - $N_{\text{RO}}$ triggers a **bias-variance trade-off** (longer rollouts have lower bias but higher variance)
   - The number of training epochs needs to balance sample efficiency and overfitting

**Goal**: Systematically analyze the interaction between parallel data collection strategies, network plasticity, learned representation, and sample efficiency to provide practical guidance.

**Key Insight**: With a fixed total number of environment interaction steps (100M), $N_{\text{envs}}$ and $N_{\text{RO}}$ are systematically varied on PPO and PQN. Optimization stability is analyzed using metrics like weight norm and gradient kurtosis.

## Method

### Overall Architecture

This work does not propose a new algorithm, but rather provides a systematic empirical analysis of the parallel data collection strategy in PPO. Core variables:

- **$N_{\text{envs}}$ (Number of parallel environments)**: The number of environment instances running concurrently, affecting the diversity of states and actions in a single collection phase.
- **$N_{\text{RO}}$ (Rollout length)**: The number of steps collected per environment per collection phase, affecting the bias-variance of return estimation.
- **$|B| = N_{\text{envs}} \times N_{\text{RO}}$**: Data batch size.
- **Number of Epochs**: The number of training passes over the same batch. More epochs improve sample efficiency but can cause overfitting.

The research design is divided into three dimensions:

1. **Fixed Data Budget**: Keeping $|B|$ constant (e.g., 1024), varying the combination of $(N_{\text{envs}}, N_{\text{RO}})$ (e.g., 8×128 vs 128×8) to analyze which allocation is superior.
2. **Scaling Data Volume**: Increasing $N_{\text{envs}}$ or $N_{\text{RO}}$ (without keeping the other constant) to analyze the performance differences between the two scaling methods.
3. **Multi-epoch Training**: Varying the number of epochs under different dataset scales to analyze the mitigation effect of data volume on overfitting/loss of plasticity.

### Key Designs

#### 1. Allocation Strategy Under Fixed Budget

- Default PPO configuration: $N_{\text{envs}}=8, N_{\text{RO}}=128$ (batch = 1024)
- Comparison configuration: $N_{\text{envs}}=128, N_{\text{RO}}=8$ (batch = 1024)
- **Key Finding**: Under the same data volume, prioritizing the increase of $N_{\text{envs}}$ is superior to increasing $N_{\text{RO}}$.
- **Mechanism**: More parallel environments imply more independent samples of the initial state distribution, providing higher state-action coverage; whereas long rollouts originate from the same Markov chain, resulting in stronger intra-rollout correlation.

#### 2. Optimization Stability Analysis

The impact of parallel strategies on network training is quantified by monitoring the following metrics:

- **Weight Norm**: Larger $N_{\text{envs}}$ leads to lower weight norm, indicating more stable parameter updates.
- **Gradient Kurtosis**: High kurtosis indicates heavy-tailed/peaky gradient distributions (unstable signals). Increasing $N_{\text{envs}}$ significantly reduces kurtosis.
- **Plasticity Metrics**: Large $N_{\text{envs}}$ mitigates dormant neurons and feature rank collapse.

#### 3. Multi-Epoch and Data Volume Interaction

- Increasing the number of epochs in PPO improves sample efficiency but typically leads to performance collapse (overfitting to old data).
- **Key Finding**: Larger dataset sizes delay or even prevent performance degradation caused by multiple epochs.
- Increasing $N_{\text{envs}}$ mitigates epoch-induced performance collapse more effectively than increasing $N_{\text{RO}}$.

## Key Experimental Results

### Experimental Setup

| Configuration Item | Setting |
|-------|------|
| Algorithm | PPO (CleanRL implementation) + PQN |
| Evaluation Environment | Atari-10 (Arcade Learning Environment) |
| Total Interaction Steps | 100M environment steps |
| Number of Random Seeds | 5 |
| Evaluation Metrics | Human-normalized IQM + 95% bootstrap CI |
| Default $N_{\text{envs}}$ | 8 |
| Default $N_{\text{RO}}$ | 128 |
| Hardware | NVIDIA Tesla A100 GPU |
| Training Time per Run | Approx. 2-3 days |

### $N_{\text{envs}}$ vs $N_{\text{RO}}$ Comparison Under Fixed Budget

| Configuration | $N_{\text{envs}}$ | $N_{\text{RO}}$ | Batch Size | IQM Performance |
|------|------|--------|------------|----------|
| Default | 8 | 128 | 1024 | Baseline |
| High Parallelism | 128 | 8 | 1024 | Significantly outperforms baseline |
| High Rollout | 8 | High | Large | Limited gain |
| Scaled Parallelism | 64/128/256 | 128 | Large | Continuous gain |

### Network Plasticity and Stability Metrics

| Metric | Low $N_{\text{envs}}$ | High $N_{\text{envs}}$ | Trend |
|------|---------------------|---------------------|------|
| Weight Norm | High | Low | Improved stability |
| Gradient Kurtosis | High (heavy-tailed) | Low (near-normal) | Smoother optimization |
| Dormant Neurons | More | Fewer | Maintained plasticity |
| Multi-Epoch Performance Degradation | Severe | Slight | Enhanced overfitting resistance |

## Highlights & Insights

- **"Not all data are created equal"**: Under the same data volume, data from more independent environments is more valuable than data from longer rollouts because independent environments provide better state-space coverage and lower sample correlation.
- The **asymmetry between parallel environment count and rollout length** reveals a deep mechanism: $N_{\text{envs}}$ scales data diversity (different initial states), whereas $N_{\text{RO}}$ only scales temporal extension within a rollout, which is limited by intra-Markov chain correlation.
- The **connection to plasticity is highly inspiring**: Parallel data collection is not only a computational acceleration tool, but also a means to improve network optimization health—lower weight norm and gradient kurtosis are directly linked to better long-term learning capability.
- **"Safe scaling" of epochs**: With a sufficiently large dataset size, the number of training passes can be safely increased without worrying about performance collapse, providing practical guidance for hyperparameter tuning in PPO.
- The experiments are validated on both PPO and PQN algorithms and extended to different network architectures, enhancing the generalizability of the conclusions.

## Limitations & Future Work

- **Limited to on-policy algorithms**: The conclusions are based on PPO (and PQN). Whether they apply to off-policy algorithms (SAC, TD3) remains unverified.
- **Limited environments**: Tested only on Atari-10; validation on continuous control (MuJoCo) and 3D environments is insufficient.
- **Incomplete coverage of later sections**: Architectural analysis, hyperparameter sensitivity, and continuous control experiments in the latter half of the paper (Sections 5-7) are not fully covered due to context limitations.
- **Lack of computational cost analysis**: The wall-clock time and GPU memory overhead of increasing $N_{\text{envs}}$ are not reported in detail.
- **Limited theoretical explanation**: While $N_{\text{envs}}$ is observed to outperform $N_{\text{RO}}$, a rigorous theoretical proof is lacking.
- **Extreme parallel scenarios not investigated**: Whether diminishing returns or negative effects exist when $N_{\text{envs}}$ is extremely large (e.g., 4096+) has not been studied.

## Related Work & Insights

- **vs. A3C/IMPALA**: Early parallel RL focused on distributed architecture design, whereas this work focuses on the impact of parallel strategies on learning dynamics.
- **vs. Singla et al. (2024)**: The former observed diminishing returns in data scaling, while this work further dissects the differences between $N_{\text{envs}}$ and $N_{\text{RO}}$.
- **vs. Plasticity Studies (Lyle 2023, Moalla 2024)**: Loss of plasticity is a core challenge in deep RL. This work suggests parallel data collection serves as a "natural mitigation" approach.
- **Practical Implications**: When tuning PPO, priority should be given to increasing the number of parallel environments rather than the rollout length; ensure the dataset is large enough before increasing epochs.

## Rating

- Novelty: ⭐⭐⭐ Focused on empirical analysis with no new algorithm, but the systematic analysis is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Atari-10 + PPO/PQN + various metrics + architectural analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rational experimental design.
- Value: ⭐⭐⭐⭐ Direct guidance for practitioners tuning parallel strategies in RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC](beyond_the_rainbow_high_performance_deep_reinforcement_learning_on_a_desktop_pc.md)
- [\[ICML 2025\] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning](network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](../../NeurIPS2025/reinforcement_learning/confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [\[ICML 2025\] Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources](heterogeneous_data_game_characterizing_the_model_competition_across_multiple_dat.md)
- [\[NeurIPS 2025\] Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mind_the_gap_the_challenges_of_scale_in_pixel-based_deep_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
