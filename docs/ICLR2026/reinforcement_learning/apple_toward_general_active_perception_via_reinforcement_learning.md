---
title: >-
  [Paper Note] APPLE: Toward General Active Perception via Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][active perception] This paper proposes APPLE, a general active perception framework that combines reinforcement learning with supervised learning. Active perception is formulated as a…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "active perception"
  - "reinforcement-learning"
  - "POMDP"
  - "supervised learning"
  - "off-policy"
  - "ViViT"
  - "CrossQ"
date: 2026-05-08
content_hash: 52522e34b5ba2055
---

# APPLE: Toward General Active Perception via Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2505.06182](https://arxiv.org/abs/2505.06182)
**Area**: Active Perception / Reinforcement Learning
**Keywords**: active perception, reinforcement-learning, POMDP, supervised learning, off-policy, ViViT, CrossQ

## TL;DR

This paper proposes APPLE, a general active perception framework that combines reinforcement learning with supervised learning. Active perception is formulated as a POMDP, with the reward defined as the RL reward minus the prediction loss. The gradient naturally decomposes into a policy gradient term and a prediction loss gradient term. Built upon off-policy algorithms (SAC/CrossQ) and a shared ViViT backbone, the framework is validated across 5 diverse task benchmarks. The CrossQ variant requires no per-task hyperparameter tuning and achieves a 53% improvement in training efficiency.

## Background & Motivation

**Core challenge of active perception**: Active perception requires an agent to actively control sensors (e.g., moving camera viewpoints, performing tactile exploration) to gather information while simultaneously completing perceptual prediction tasks, necessitating joint optimization of "how to perceive" and "how to predict."

**Fragmentation of existing methods**: Current active perception approaches are typically designed for specific tasks and sensing modalities (e.g., active object recognition, active tactile perception), lacking a unified framework applicable to diverse tasks.

**Coupling difficulty between RL and prediction tasks**: Pure RL methods require manually designed proxy rewards to indirectly evaluate perceptual quality, making direct optimization of prediction performance difficult; pure supervised learning methods cannot learn perception policies.

**Failure of on-policy methods**: Experiments show that on-policy methods such as REINFORCE and PPO completely fail on active perception tasks due to insufficient exploration efficiency and sparse reward signals.

**Hyperparameter sensitivity**: Existing methods often require careful per-task hyperparameter tuning, limiting their practical generalizability.

**Computational efficiency requirements**: Real-world deployment scenarios demand efficient training and inference, requiring reduced computational overhead without sacrificing performance.

## Method

### Overall Architecture

APPLE formulates active perception as a POMDP, where at each timestep the agent selects an action (sensor control) based on historical observations, then updates its prediction upon receiving new observations. The reward function is defined as $r = r_{RL} - \mathcal{L}_{pred}$, whose gradient naturally decomposes into a policy gradient (optimizing the perception policy) and a prediction loss gradient (optimizing the prediction model).

### Key Designs

1. **Unified reward–loss design**

    - **Function**: The reward is defined as the RL task reward minus the supervised prediction loss: $r_t = r_t^{RL} - \mathcal{L}_{pred}(\hat{y}_t, y)$
    - **Mechanism**: The gradient $\nabla J$ decomposes into $\nabla_\theta J_{RL} + \nabla_\phi \mathcal{L}_{pred}$, where policy parameters $\theta$ are optimized via policy gradient and prediction parameters $\phi$ via supervised loss
    - **Design Motivation**: This avoids the difficulty of manually designing proxy rewards in pure RL frameworks while preserving direct supervised optimization of the prediction model

2. **Off-policy algorithm selection (SAC/CrossQ)**

    - **Function**: Two variants are proposed: APPLE-SAC and APPLE-CrossQ
    - **Mechanism**: Off-policy methods enable efficient exploration through experience replay; CrossQ eliminates a critical hyperparameter by replacing the target network with batch normalization
    - **Design Motivation**: On-policy methods (REINFORCE/PPO) completely fail in active perception; CrossQ is more robust than SAC and removes the need to tune target network update frequency

3. **Shared ViViT backbone**

    - **Function**: A Video Vision Transformer (ViViT)-style architecture serves as a shared feature extractor for both the policy network and prediction network
    - **Mechanism**: Historical observation sequences are treated as video sequences, with spatiotemporal attention mechanisms capturing information aggregation across timesteps
    - **Design Motivation**: A shared backbone reduces parameter count, and sequential modeling naturally accommodates the incremental observation accumulation characteristic of active perception

## Key Experimental Results

### Main Results

| Task | APPLE-SAC | APPLE-CrossQ | Best Baseline | Baseline Method |
|------|-----------|-------------|--------------|----------------|
| MHSB (classification) | 94.2% | **95.1%** | 89.7% | InfoGain |
| CircleSquare (detection) | 0.82 IoU | **0.84 IoU** | 0.76 IoU | Random |
| TactileMNIST (recognition) | 92.8% | **93.5%** | 88.3% | Coverage |
| Volume (estimation) | 0.031 MSE | **0.028 MSE** | 0.045 MSE | Heuristic |
| Toolbox (6DoF) | 78.5% | **80.2%** | 71.4% | AcTPa |

### Ablation Study

| Method/Variant | Avg. Rank | Training Time (relative) | Hyperparameter Tuning |
|---------------|-----------|--------------------------|----------------------|
| APPLE-CrossQ | **1.2** | **1.0x** | Low |
| APPLE-SAC | 1.8 | 1.53x | Medium |
| REINFORCE | 4.5 | 0.8x | High (poor performance) |
| PPO | 4.8 | 1.1x | High (poor performance) |
| Supervised only (no RL) | 3.2 | 0.6x | Low |

### Key Findings

1. **On-policy methods completely fail**: REINFORCE and PPO fail to learn effective policies on all 5 benchmarks, confirming the necessity of off-policy methods for active perception.
2. **CrossQ consistently outperforms SAC**: It achieves higher average rankings across tasks, trains 53% faster, and eliminates the need to tune target network hyperparameters.
3. **Generalizability validated**: The same framework and hyperparameter configuration achieves state-of-the-art or near-state-of-the-art performance across 5 substantially different tasks.
4. **RL + supervision outperforms supervision alone**: Removing the RL component leads to significant performance degradation, demonstrating the importance of learning an active perception policy.

## Highlights & Insights

1. **Unified framework**: The first general active perception framework applicable to diverse sensing modalities and task types.
2. **Elegant gradient decomposition**: The reward–loss design naturally separates policy gradients from prediction gradients, yielding a theoretically clean formulation.
3. **Significant negative result**: The complete failure of on-policy methods provides an important reference for the active perception community.
4. **Practical applicability**: The CrossQ variant requires virtually no hyperparameter tuning, substantially lowering the barrier to real-world adoption.

## Limitations & Future Work

1. **Discrete action spaces**: All experiments use discrete actions; performance in continuous action spaces (e.g., continuous viewpoint control) remains unverified.
2. **Simulation-dominated evaluation**: All 5 benchmarks are simulation environments; generalization to real physical settings has yet to be validated.
3. **Computational requirements**: The computational overhead of the ViViT backbone may become a bottleneck on resource-constrained embedded platforms.
4. **Long observation sequences**: Current experiments involve relatively short perception horizons (5–20 steps); performance trends over longer sequences remain unexplored.

## Related Work & Insights

- **Active perception**: Bajcsy et al. (2018) survey on active perception; AcTPa (Liang et al., 2025) on tactile active perception
- **Off-policy RL**: SAC (Haarnoja et al., 2018), CrossQ (Bhatt et al., 2024) for efficient off-policy learning
- **Vision Transformers**: ViViT (Arnab et al., 2021) architecture for video understanding
- **POMDP solving**: Kaelbling et al. (1998) theoretical framework for POMDPs

## Rating

- Novelty: ⭐⭐⭐⭐ Unified framework and gradient decomposition design are novel
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 benchmarks covering diverse modalities and task types
- Writing Quality: ⭐⭐⭐⭐ Clear framework presentation with rigorous experiments
- Value: ⭐⭐⭐⭐ Strong practical application potential as a general active perception framework

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](../../NeurIPS2025/reinforcement_learning/real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[ICLR 2026\] Spotlight on Token Perception for Multimodal Reinforcement Learning](spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)
- [\[ICLR 2026\] Learning to Orchestrate Agents in Natural Language with the Conductor](learning_to_orchestrate_agents_in_natural_language_with_the_conductor.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
