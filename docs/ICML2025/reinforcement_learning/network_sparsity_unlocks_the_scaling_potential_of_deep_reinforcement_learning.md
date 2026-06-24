---
title: >-
  [Paper Note] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning
description: >-
  [ICML 2025][Reinforcement Learning][Network Sparsity] This work discovers that simple one-shot random pruning can unlock the scaling potential of deep RL—sparse networks achieve higher parameter efficiency, stronger plasticity preservation, and less gradient interference than dense networks equipped with SOTA architectures.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Network Sparsity"
  - "Deep RL Scaling"
  - "Random Pruning"
  - "Loss of Plasticity"
  - "Gradient Interference"
date: 2026-05-08
content_hash: bce9b05729981e27
---

# Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning

**Conference**: ICML 2025  
**arXiv**: [2506.17204](https://arxiv.org/abs/2506.17204)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: Network Sparsity, Deep RL Scaling, Random Pruning, Loss of Plasticity, Gradient Interference

## TL;DR
This work discovers that simple one-shot random pruning can unlock the scaling potential of deep RL—sparse networks achieve higher parameter efficiency, stronger plasticity preservation, and less gradient interference than dense networks equipped with SOTA architectures.

## Background & Motivation

### Background

**Background**: Scaling up network size in deep RL has been notoriously difficult, which stands in stark contrast to the scaling laws of supervised learning.

**Limitations of Prior Work**: Naively increasing the number of network parameters in RL often fails to bring performance gains. Instead, it leads to performance degradation due to training pathologies such as plasticity loss and gradient interference. Existing solutions, such as periodic resets and Layer Normalization, function as ad-hoc fixes.

**Key Challenge**: Larger networks $\neq$ better RL performance—pathological phenomena in training dynamics exacerbate as the network scales.

**Key Insight**: Rather than pursuing more complex architectural modifications, this work investigates the simplest intervention—static network sparsity.

**Core Idea**: By randomly removing a certain percentage of weights in a one-shot manner before training, the resulting sparse network exhibits better scalability than dense networks with equivalent parameter counts.

### Proposed Approach

**Goal**: ### Overall Architecture
Input: Standard DRL network (e.g., MLP/CNN) → one-shot random pruning prior to training (removing a predefined ratio of weights) → normal RL training → output sparse policy network.


## Method

### Overall Architecture
Input: Standard DRL network (e.g., MLP/CNN) → one-shot random pruning prior to training (removing a predefined ratio of weights) → normal RL training → output sparse policy network.

### Key Designs

1. **One-shot Random Pruning**:

    - Randomly removes a predefined percentage (e.g., 50%, 80%, 90%) of weight connections after initialization but before training.
    - Fixes the sparse structure after pruning without restoration or modification.
    - Design Motivation: The simplest sparsification strategy, eliminating the interference of structural search.

2. **Parameter Efficiency Analysis**:

    - Sparse networks achieve higher network expressiveness with fewer active parameters.
    - Because the sparse structure reduces parameter redundancy, making each parameter more "useful".
    - Design Motivation: To understand why sparsity is effective—not just as a regularization effect.

3. **Resistance to Training Pathologies Analysis**:

    - Plasticity loss: Sparse networks better maintain the ability to learn new information.
    - Gradient interference: Gradients of different objectives are more prone to conflict in dense networks.
    - Design Motivation: Explains the benefits of sparsity from the perspective of optimization dynamics.

### Loss & Training
- Employs standard RL algorithms (such as PPO, SAC, etc.) without modifying the training pipeline.
- The only modification is performing a one-shot random pruning prior to training.

## Key Experimental Results

### Main Results

| Environment Type | Metric | Sparse Network | Dense Network (SOTA arch) | Results |
|---------|------|---------|-------------------|------|
| DMControl | Return | Higher | Baseline | Sparse outperforms dense |
| Atari (Visual RL) | Score | Higher | Baseline | Consistent improvement |
| Streaming RL | Performance | More stable | Degradation | Sparsity mitigates forgetting |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Different Sparsity Rates | Performance Curve | Moderate sparsity rates (50-80%) are typically optimal |
| Random vs. Structured Pruning | Performance | Random pruning is already sufficiently effective |
| With/Without LayerNorm | Performance | Sparsity + LN yields further improvements |
| Plasticity Metrics | Effective Rank | Sparse networks maintain a higher effective rank |

### Key Findings
- Random pruning, as the simplest operation, enables deep RL networks to maintain performance gains during scaling.
- Sparse networks exhibit stronger gradient consistency (reducing interference).
- The effects consistently persist in both visual RL and streaming RL.
- Sparsity is complementary to existing methods such as LayerNorm and periodic reset.

## Highlights & Insights
- **Counterintuitive Conclusion**: Complex methods are unnecessary; simple sparsification can resolve the deep RL scaling dilemma.
- **Unified Explanation**: Three perspectives—parameter efficiency, plasticity preservation, and reduced gradient interference—unifiedly explain the advantages of sparsity.
- **High Practicality**: Effective with just a single-line code modification.

## Limitations & Future Work
- A fixed sparse structure might not be optimal—could dynamic sparsification achieve further improvements?
- Performance may degrade under extremely high sparsity rates, and the optimal sparsity rate is task-dependent.
- Theoretical explanations remain predominantly empirical, lacking rigorous proof.

## Related Work & Insights
- Lottery Ticket Hypothesis (Frankle & Carlin 2019) identifies sparse subnetworks.
- Dormant Neuron (Sokar et al. 2023) analyzes the loss of plasticity in RL.
- Insights: The difficulty of scaling in RL may not stem from networks being too small, but rather from over-parameterization causing optimization bottlenecks.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple yet profound discovery: random pruning unlocks deep RL scaling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation across multiple environment types + mechanistic analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear analysis and compelling arguments.
- Value: ⭐⭐⭐⭐⭐ Provides the most concise solution for deep RL scaling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)
- [\[ICML 2025\] Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC](beyond_the_rainbow_high_performance_deep_reinforcement_learning_on_a_desktop_pc.md)
- [\[ICML 2025\] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)
- [\[ICLR 2026\] Finite-Time Analysis of Actor-Critic Methods with Deep Neural Network Approximation](../../ICLR2026/reinforcement_learning/finite-time_analysis_of_actor-critic_methods_with_deep_neural_network_approximat.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)

</div>

<!-- RELATED:END -->
