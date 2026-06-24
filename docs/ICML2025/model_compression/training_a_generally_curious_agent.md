---
title: >-
  [Paper Note] Training a Generally Curious Agent (Paprika)
description: >-
  [ICML 2025][Model Compression][in-context RL] Proposes the Paprika framework, which fine-tunes LLMs on diverse text-based decision-making tasks, enabling the model to learn general information-gathering and decision-making capabilities and transfer zero-shot to completely unseen tasks.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "in-context RL"
  - "curious agent"
  - "curriculum learning"
  - "sequential decision making"
  - "Paprika"
date: 2026-05-08
content_hash: 4d234be454f21027
---

# Training a Generally Curious Agent (Paprika)

**Conference**: ICML 2025  
**arXiv**: [2502.17543](https://arxiv.org/abs/2502.17543)  
**Code**: -  
**Area**: Model Compression  
**Keywords**: in-context RL, curious agent, curriculum learning, sequential decision making, Paprika  

## TL;DR

Proposes the Paprika framework, which fine-tunes LLMs on diverse text-based decision-making tasks, enabling the model to learn general information-gathering and decision-making capabilities and transfer zero-shot to completely unseen tasks.

## Background & Motivation

- LLMs acting as autonomous agents must interact with environments and collect information to achieve goals.
- Direct deployment in the real world to collect data is highly costly and risky.
- Synthetic data generation cannot cover all tasks, but the **in-context learning** capability of LLMs supports learning generalized policies from a small set of tasks.
- **Mechanism**: Instead of training the model to solve all tasks, train the model to **learn the general process of performing tasks** (i.e., in-context RL).
- This is analogous to the SFT/RLHF phases where only a small number of examples are needed to yield models capable of generalizing to diverse queries.

## Method

### 1. Task Design (10 Task Groups)

All tasks are text-based, multi-turn, and partially observable:

| Task Group | Number of Training Tasks | Max Turns | Environment Feedback |
|--------|----------|---------|---------|
| 20 Questions | 1499 | 20 | LLM-generated |
| Guess My City | 500 | 20 | LLM-generated |
| Wordle | 1515 | 6 | Hardcoded |
| Cellular Automata | 1000 | 6 | Hardcoded |
| Customer Service | 628 | 20 | LLM-generated |
| Murder Mystery | 203 | 20 | LLM-generated |
| Mastermind | 1000 | 12 | Hardcoded |
| Battleship | 1000 | 20 | Hardcoded |
| Minesweeper | 1000 | 20 | Hardcoded |
| Bandit Best Arm | 81 | 21 | Hardcoded |

### 2. Dataset Construction

- Diverse trajectories are generated using Min-p sampling (temperature 1.5, p=0.3).
- Each task generates $n_\text{sample}=20$ trajectories.
- Preference pairs $(h_w, h_l)$ are constructed: $h_w$ is the highest-scoring trajectory, while $h_l$ is randomly sampled from low-scoring trajectories.

### 3. Optimization Objectives

**SFT**: Maximizing likelihood on winning trajectories:

$$\mathcal{L}_\text{SFT} = -\mathbb{E}\left[\frac{1}{\sum_t |a_t^w|}\sum_t \log \pi_\theta(a_t^w | h_{:t}^w)\right]$$

**Multi-turn DPO**:

$$\mathcal{L}_\text{DPO} = -\mathbb{E}\left[\log\sigma\left(\sum_t \beta\log\frac{\pi_\theta(a_t^w|h_{:t}^w)}{\pi_\text{ref}(a_t^w|h_{:t}^w)} - \sum_t \beta\log\frac{\pi_\theta(a_t^l|h_{:t}^l)}{\pi_\text{ref}(a_t^l|h_{:t}^l)}\right)\right]$$

Loss is computed only on the agent's action tokens (excluding environment-generated tokens).

**RPO**: Combines SFT + DPO to mitigate the "undesired disalignment" of DPO:

$$\mathcal{L}_\text{RPO} = \mathcal{L}_\text{DPO} + \alpha \mathcal{L}_\text{SFT}$$

### 4. Curriculum Learning: Scalable Task Selection

**Learning Potential Metric**: Measures the learning signal strength of tasks using the coefficient of variation:

$$\nu_\pi(\tau) = \frac{\sqrt{\sigma^2_\pi(\tau)}}{R_\pi(\tau)}$$

High variance -> higher likelihood of sampling both good and bad trajectories -> gradient signal for DPO is present; normalization by the average reward ensures cross-task comparability.

**UCB Algorithm for Task Group Selection**: Treats each task group as an arm and utilizes UCB to balance exploration and exploitation.

## Experimental Results

### Main Results: All-Task Training

Paprika improves the average success rate of Llama-3.1-8B-Instruct by **47%** (relative gain) across all 10 task groups, using only approximately 22,500 trajectories.

### Zero-Shot Transfer (Leave-One-Out)

| Task Group | Baseline | Paprika (LOO) | Paprika (All) | Single-Task Training |
|--------|------|--------------|--------------|----------|
| Bandit | 42.25% | **62.25%** | 65.0% | 58.0% |
| 20 Questions | ~25% | ~38% | ~40% | ~35% |
| Murder Mystery | ~15% | ~28% | ~32% | ~30% |

- The LOO model outperforms the initial model on **9 out of 10 task groups**.
- On **7 out of 10 task groups**, training on all tasks outperforms single-task training.
- This indicates that cross-task policy transfer indeed occurs.

### Curriculum Learning Effects

- Curriculum learning improves the average success rate by **1.4%** and pass@4 by **3.3%** compared to uniform sampling.
- The benefits are mainly observed in moderately difficult tasks.

## Highlights & Insights

- Demonstrates that LLMs can learn **transferable, general exploration strategies** from text-based decision-making tasks.
- Eliminates the need for known optimal algorithms (such as UCB) to generate training data; the model's own diverse sampling suffices.
- The coefficient of variation serves as an intuitive and effective metric for learning potential.
- The task designs exhibit good diversity, spanning reasoning, search, and planning strategies.
- The connection established with meta-RL builds a theoretical framework for LLM agent training.

## Limitations & Future Work

- Data generation remains the main bottleneck (requiring more resources than model updates).
- Negative transfer is observed in Wordle, indicating that not all strategies are transferable across tasks.
- Experiments are conducted only on 8B and 12B models; the effectiveness on larger or smaller models remains unexplored.
- The environments partially rely on LLMs to generate feedback, which may introduce noise.
- Comparison with online RL methods is missing (only offline DPO is utilized).

## Rating

⭐⭐⭐⭐ — An exciting research direction, demonstrating that general decision-making capabilities can be acquired through synthetic data training and transferred zero-shot. However, the bottleneck lies in data generation efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Curious Case of In-Training Compression of State Space Models](../../ICLR2026/model_compression/the_curious_case_of_in-training_compression_of_state_space_models.md)
- [\[ICML 2025\] MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles](moragent_parameter_efficient_agent_tuning_with_mixture-of-roles.md)
- [\[ICML 2025\] Towards an Optimal Control Perspective of ResNet Training](towards_an_optimal_control_perspective_of_resnet_training.md)
- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[ICML 2025\] Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation](merge-friendly_post-training_quantization_for_multi-target_domain_adaptation.md)

</div>

<!-- RELATED:END -->
