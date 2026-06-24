---
title: >-
  [Paper Note] Sliding Puzzles Gym: A Scalable Benchmark for State Representation in Visual Reinforcement Learning
description: >-
  [ICML 2025][Reinforcement Learning][Visual RL] This paper proposes Sliding Puzzles Gym (SPGym), a benchmark that transforms the classic 8-puzzle into visual RL tasks. By independently adjusting the image pool size, it precisely controls the complexity of visual representation learning. Experiments reveal fundamental memorization limitations of current methods as visual diversity increases.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Visual RL"
  - "Benchmark"
  - "Representation Learning"
  - "Sliding Puzzle"
  - "Generalization"
date: 2026-05-08
content_hash: 89d7edb016f56982
---

# Sliding Puzzles Gym: A Scalable Benchmark for State Representation in Visual Reinforcement Learning

**Conference**: ICML 2025  
**arXiv**: [2410.14038](https://arxiv.org/abs/2410.14038)  
**Code**: [bryanoliveira/sliding-puzzles-gym](https://github.com/bryanoliveira/sliding-puzzles-gym)  
**Area**: Visual Reinforcement Learning / Representation Learning Benchmark  
**Keywords**: Visual RL, Benchmark, Representation Learning, Sliding Puzzle, Generalization  

## TL;DR

This paper proposes Sliding Puzzles Gym (SPGym), a benchmark that transforms the classic 8-puzzle into visual RL tasks. By independently adjusting the image pool size, it precisely controls the complexity of visual representation learning. Experiments reveal fundamental memorization limitations of current methods as visual diversity increases.

## Background & Motivation

Visual reinforcement learning (Visual RL) requires agents to extract task-relevant features from raw pixel inputs for decision-making. Evaluating representation learning capabilities is a key challenge, but existing benchmarks have core limitations:

**Coupling Issue**: Performance metrics of classic benchmarks like Atari and DM Control **confound representation learning, policy optimization, and environment dynamics**, failing to evaluate representation learning in isolation.

**ProcGen**: Varies both visual and task difficulty simultaneously, making it impossible to isolate the impact of representation learning.

**Distracting Control Suite**: Introduces task-irrelevant visual distractors, which agents can safely ignore.

**COOM**: Focuses on catastrophic forgetting rather than single-task representation learning.

Key requirement: A benchmark that can **precisely and independently scale visual complexity** while keeping task dynamics invariant.

## Method

### Overall Architecture

SPGym frames the classic sliding puzzle as a POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \mathcal{S}_0, \Omega, \mathcal{O})$:
- **State Space** $\mathcal{S}$: All solvable puzzle configurations (approximately $1.81 \times 10^5$ for a 3x3 grid).
- **Action Space** $\mathcal{A}$: Four directions (up, down, left, right).
- **Observation Function** $\mathcal{O}(s, i)$: Selects image $i$ from the image pool $\mathcal{I}$ and arranges the puzzle pieces according to state $s$.
- The agent **cannot directly access the state** and must reconstruct the original image from 84x84 pixels.

### Key Designs: Dual Orthogonal Scaling Mechanism

1. **Visual Diversity Scaling**: Adjusts the image pool size $p$ (from 1 to 100+ images), varying $|\Omega|$ while keeping $\mathcal{P}$, $\mathcal{A}$, and $\mathcal{R}$ unchanged.
2. **Search Complexity Scaling**: Adjusts the grid dimensions (3x3 $\rightarrow$ 4x4), varying $|\mathcal{S}|$ and visual complexity.

### Reward Function

Dense reward based on normalized Manhattan distance:

$$\mathcal{R}(s) = \begin{cases} -D & \text{if action valid} \\ -1 & \text{if action invalid} \\ +1 & \text{if solved} \end{cases}$$

$$D = \frac{\sum_{i,j} |u_{i,j} - u^*_{i,j}| + |v_{i,j} - v^*_{i,j}|}{\sum_{i,j} \max(i, H-i) + \max(j, W-j)}$$

## Key Experimental Results

### Main Results: Sample Efficiency (Million Steps to Reach 80% Success Rate)

| Agent | Pool 1 | Pool 5 | Pool 10 |
|-------|--------|--------|---------|
| PPO | 1.75±0.44 | 7.80±1.08 | 9.73±0.36 |
| PPO + PT (ID) | **0.95±0.21** | **5.55±1.22** | **9.17±1.10** |
| SAC | 0.33±0.07 | 0.91±0.12 | 2.03±0.38 |
| SAC + RAD | **0.24±0.03** | **0.42±0.06** | **0.82±0.18** |
| SAC + CURL | 0.46±0.10 | 1.56±0.31 | 5.24±1.92 |
| SAC + SPR | 2.09±0.81 | 3.68±1.68 | 10.00±0.00 |
| DreamerV3 | **0.42±0.06** | **1.23±0.20** | **1.44±0.58** |
| DreamerV3 w/o dec | 1.13±0.12 | 1.79±0.61 | 2.57±0.91 |

### Comparison of Degradation Regimes

| Agent | Degradation Threshold (Pool Size) | Complete Failure Threshold |
|-------|-----------------|------------|
| PPO | 10 | 20 |
| SAC | 30 | 50 |
| DreamerV3 | 50 | 100 |

### Easy OOD Generalization Results (Success Rate)

| Agent | Pool 1 | Pool 5 | Pool 10 |
|-------|--------|--------|---------|
| SAC + AE | 0.78 | 0.64 | 0.55 |
| SAC + SB | **0.89** | 0.65 | 0.06 |
| SAC + RAD | 0.62 | 0.42 | 0.30 |
| SEFA | 0.76 | 0.44 | 0.37 |

### Hard OOD Generalization Results

All methods exhibit **nearly 0% success rate** on completely unseen images—revealing that end-to-end RL methods rely on memorization rather than true visual understanding.

### Effect of Grid Size (Million Steps, Pool Size 1)

| Grid | PPO | SAC | DreamerV3 |
|------|-----|-----|-----------|
| 3×3 | 1.75 | 0.33 | 0.42 |
| 4×4 | 24.46 | 8.14 | **2.26** |

### Ablation Key Findings

1. **Strong Correlation between Representation Quality and Performance**: Pearson correlation between linear probing accuracy and sample efficiency is $r = -0.81$, $p = 1.1 \times 10^{-13}$.
2. **Strong Correlation between Easy OOD Success Rate and Sample Efficiency**: $r = -0.81$, $p = 2.5 \times 10^{-12}$.
3. **Data Augmentation (RAD) is Consistently Optimal**: Complex auxiliary objectives (CURL, SPR, DBC, VAE) are often outperformed by simple data augmentation.
4. **DreamerV3's World Model is the Most Robust**: Performance drops when removing decoder gradients, proving the importance of reconstruction objectives.
5. **Results on DiffusionDB are Consistent with ImageNet**: Rules out dataset specificity.

## Highlights & Insights

1. **Precise Isolation of Representation Learning**: SPGym is ingeniously designed—visual understanding is a **prerequisite** for task success (necessary to solve the puzzle), rather than a ignorable distractor.
2. **Counter-intuitive Finding**: Larger training pools lead to worse Easy OOD generalization—models trained on smaller pools learn stronger task-specific invariances.
3. **Memorization Exposed**: The complete failure under Hard OOD conditions demonstrates that existing end-to-end RL methods essentially **memorize visual patterns** rather than learning generalized representations.
4. **Mismatch in Method Assumptions**: The implicit assumptions of methods like CURL (contrastive learning requiring similar augmented observations) and DBC (requiring visual similarity to equal dynamical similarity) do not match the discrete, highly variable observations in SPGym.
5. **Continued Training Significantly Improves Solution Quality**: With extended training, DreamerV3's average steps drop from 126 to 23 steps, approaching the theoretical optimum of 22 steps.

## Limitations & Future Work

1. **Insufficient Tuning**: Focusing on "out-of-the-box" settings, which may not exhibit the peak performance of each method.
2. **Statistical Robustness**: Only evaluated on 5 random seeds, which may be insufficient given the high stochasticity of image pools.
3. **Task Simplicity**: The 3x3 puzzle itself is a relatively simple search problem, with the primary difficulty stemming from visual representation.
4. **Discrete Action Space**: Fundamentally different from visual RL in continuous control (e.g., DM Control).

## Related Work & Insights

- **Traditional RL Benchmarks**: Atari, DM Control Suite, DeepMind Lab, CARLA
- **Specialized Visual RL Benchmarks**: ProcGen, Distracting Control Suite, COOM
- **Puzzle Benchmarks**: Estermann et al. 2024 (discrete state space)
- **Sliding Puzzle Solvers**: A*, IDA*, DRL methods (Agostinelli et al., Moon & Cho)
- **Representation Learning Methods**: RAD, CURL, SPR, DBC, DreamerV3, SAC-AE

## Rating

⭐⭐⭐⭐ (4/5)

SPGym is ingeniously designed, successfully isolating the challenge of visual representation learning. The finding that "all methods are memorizing" serves as an important warning. However, the benchmark itself is relatively simple (puzzle tasks), and out-of-the-box evaluations may underestimate the potential of some methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](../../ICLR2026/reinforcement_learning/reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)
- [\[ICML 2025\] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)
- [\[CVPR 2026\] Saliency-Guided Representation with Consistency Policy Learning for Visual Unsupervised Reinforcement Learning](../../CVPR2026/reinforcement_learning/saliency-guided_representation_with_consistency_policy_learning_for_visual_unsup.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../NeurIPS2025/reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](../../ICLR2026/reinforcement_learning/stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
