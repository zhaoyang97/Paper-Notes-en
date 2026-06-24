---
title: >-
  [Paper Note] Benchmarking Quantum Reinforcement Learning
description: >-
  [ICML 2025][Reinforcement Learning][Quantum Reinforcement Learning] Proposes a rigorous benchmarking methodology for quantum reinforcement learning (QRL)—introducing a statistical estimator based on sample complexity and the concept of "surpassing" defined by statistical significance. Conducts the largest-scale (100 seeds) comparison of QRL vs. classical RL to date on a newly designed 6G beam management environment, revealing that prior claims regarding QRL superiority need t…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Quantum Reinforcement Learning"
  - "benchmark"
  - "sample complexity"
  - "variational quantum circuits"
  - "statistical testing"
date: 2026-05-08
content_hash: 760d48a08c795db5
---

# Benchmarking Quantum Reinforcement Learning

**Conference**: ICML 2025  
**arXiv**: [2501.15893](https://arxiv.org/abs/2501.15893)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Quantum Reinforcement Learning, benchmark, sample complexity, variational quantum circuits, statistical testing

## TL;DR
Proposes a rigorous benchmarking methodology for quantum reinforcement learning (QRL)—introducing a statistical estimator based on sample complexity and the concept of "surpassing" defined by statistical significance. Conducts the largest-scale (100 seeds) comparison of QRL vs. classical RL to date on a newly designed 6G beam management environment, revealing that prior claims regarding QRL superiority need to be treated with greater caution.

## Background & Motivation

### Background

**Background**: QRL replaces neural networks in classical RL with variational quantum circuits (VQCs), aiming to achieve a quantum advantage in sample complexity. Some studies claim that QRL outperforms classical RL on certain tasks.

**Limitations of Prior Work**: QRL research generally suffers from reproducibility issues—(a) claiming superiority using only 5 seeds; (b) inconsistent statistical coverage; (c) increased difficulty in comparison due to the additional randomness introduced by quantum computing (shot noise, hardware imperfections); (d) lack of flexible and scalable benchmarking environments.

**Key Challenge**: There is no widely accepted statistical methodology to determine whether QRL significantly outperforms classical RL.

**Goal**: To establish a rigorous evaluation methodology for QRL.

**Key Insight**: (a) Define statistical estimators based on sample complexity; (b) design benchmarking environments with flexibly adjustable complexity; (c) conduct large-scale computational experiments using 100 seeds.

**Core Idea**: Statistical significance testing + a sufficient number of seeds is the only reliable way to assess quantum advantage.

## Method

### Overall Architecture
1. Define a sample complexity estimator $\hat{S}$: the number of environment interactions required for an agent to reach a performance threshold $(1-\varepsilon)$.
2. Perform hypothesis testing based on the distribution of $\hat{S}$ to define statistical "surpassing".
3. Compare DDQN and quantum DDQN on the newly designed BeamManagement6G environment.

### Key Designs

1. **Sample Complexity Statistical Estimator**:

    - **Function**: Given a performance threshold, estimate the distribution of the number of samples required for the algorithm to reach that threshold.
    - **Mechanism**: Conduct N=100 independent training runs, record the number of steps to first reach the threshold for each run $\rightarrow$ obtain the empirical distribution of $\hat{S}$.
    - **Design Motivation**: Point estimators are unreliable; distribution-level comparisons are necessary.

2. **Statistical Surpassing Definition**:

    - **Function**: Use hypothesis testing (Mann-Whitney U test) to determine if one algorithm significantly outperforms another.
    - **Mechanism**: If the sample complexity distribution of algorithm A is significantly lower than that of algorithm B ($p < 0.05$), then A surpasses B.
    - **Design Motivation**: To avoid incorrect conclusions caused by looking only at mean values.

3. **BeamManagement6G Benchmark Environment**:

    - **Function**: A beam management task based on 6G wireless communication, with highly adjustable complexity.
    - **Mechanism**: Keep state/action spaces small while maintaining adjustable task complexity, making it suitable for quantum algorithms (due to limited qubit counts).
    - **Design Motivation**: Standard environments like Atari have state spaces too large for current quantum hardware.

### Loss & Training
- DDQN (classical) and DDQN+VQC (hybrid quantum)
- 100 independent training runs per configuration
- Fair tuning of hyperparameters

## Key Experimental Results

### Main Results

| Algorithm Configuration | Number of Parameters | Sample Complexity $\hat{S}$ (Median) | Statistical Test |
|---------|--------|---------------------------|---------|
| Classical DNN (Small, 387 params) | 387 | High | Baseline |
| Quantum VQC (437+101 params) | 538 | Medium | Significantly outperforms small classical |
| Classical DNN (Large, 4611 params) | 4611 | Low | Comparable to quantum |

### Ablation Study

| Configuration | Finding | Explanation |
|------|------|------|
| Low complexity task | Quantum $\approx$ Classical | Task is too simple to require quantum |
| Medium complexity task | Quantum > Small Classical | Quantum shows advantages but lags behind large classical |
| High complexity task | Inconclusive results | Expressive power of quantum circuits is limited |
| 5 seeds vs 100 seeds | Conclusions may reverse | Validates the necessity of statistical rigor |

### Key Findings
- Quantum VQC consistently outperforms small classical networks with a comparable number of parameters.
- However, it is barely competitive compared to large classical networks with 10× the number of parameters.
- Conclusions of prior studies using only 5 seeds are unreliable—re-evaluating with 100 seeds yields more conservative conclusions.
- Quantum advantage is more likely in specific problem classes with small state/action spaces.

## Highlights & Insights
- **Methodological contributions outweigh algorithmic ones**—establishing a rigorous evaluation standard for QRL research.
- Benchmarking with 100 seeds is unprecedented in the quantum computing literature.
- The cautious attitude toward "quantum advantage" is worth adopting by the entire QRL community.

## Limitations & Future Work
- Only DDQN/PPO were compared, without covering a wider range of RL algorithms.
- While the BeamManagement6G environment is practically inspired, it remains a simplified version.
- Implementation errors on real quantum hardware were not considered (simulation-only).
- More complex architectures such as quantum actor-critic were not discussed.

## Related Work & Insights
- **vs. prior QRL studies**: Most used only 5 seeds, lacking statistical rigor.
- **vs. classical RL benchmarking**: This work introduces best practices from classical RL into QRL.
- Offers methodological insights for the broader benchmarking of quantum machine learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Significant contribution at the methodological level.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 100 seeds $\times$ multiple configurations, unprecedented.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition with rigorous statistical methods.
- **Value**: ⭐⭐⭐⭐ Setting a standard for QRL research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search](../../NeurIPS2025/reinforcement_learning/tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)
- [\[ICML 2026\] Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control](../../ICML2026/reinforcement_learning/plug-and-play_benchmarking_of_reinforcement_learning_algorithms_for_large-scale_.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](../../ICLR2026/reinforcement_learning/verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[NeurIPS 2025\] Near-Optimal Quantum Algorithms for Computing (Coarse) Correlated Equilibria of General-Sum Games](../../NeurIPS2025/reinforcement_learning/near-optimal_quantum_algorithms_for_computing_coarse_correlated_equilibria_of_ge.md)
- [\[ICML 2025\] Sliding Puzzles Gym: A Scalable Benchmark for State Representation in Visual Reinforcement Learning](sliding_puzzles_gym_a_scalable_benchmark_for_state_representation_in_visual_rein.md)

</div>

<!-- RELATED:END -->
