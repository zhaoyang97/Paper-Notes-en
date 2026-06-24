---
title: >-
  [Paper Note] Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples
description: >-
  [ICML2025][Robotics][GFlowNet] Flow of Reasoning (FoR) is proposed to model multi-step LLM reasoning as a Markov flow on a DAG. By fine-tuning LLMs with the trajectory balance objective of GFlowNets, the model can sample multiple high-quality and diverse reasoning paths with probabilities proportional to rewards, using only a minimal number of training examples (e.g., 15).
tags:
  - "ICML2025"
  - "Robotics"
  - "GFlowNet"
  - "Divergent Reasoning"
  - "Diverse Sampling"
  - "Few-Shot Fine-Tuning"
  - "Markov Flow"
  - "Trajectory Balance"
date: 2026-05-08
content_hash: 527847e2bbe3ae6f
---

# Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples

**Conference**: ICML2025  
**arXiv**: [2406.05673](https://arxiv.org/abs/2406.05673)  
**Code**: [Yu-Fangxu/FoR](https://github.com/Yu-Fangxu/FoR)  
**Area**: Reinforcement Learning  
**Keywords**: GFlowNet, Divergent Reasoning, Diverse Sampling, Few-Shot Fine-Tuning, Markov Flow, Trajectory Balance

## TL;DR

Flow of Reasoning (FoR) is proposed to model multi-step LLM reasoning as a Markov flow on a DAG. By fine-tuning LLMs with the trajectory balance objective of GFlowNets, the model can sample multiple high-quality and diverse reasoning paths with probabilities proportional to rewards, using only a minimal number of training examples (e.g., 15).

## Background & Motivation

- **Importance of Divergent Reasoning**: One of the core hallmarks of human intelligence is the ability to generate multiple distinct solutions to the same problem (divergent reasoning), which is crucial for enhancing robustness (e.g., self-consistency voting) and assisting scientific discovery.
- **Limitations of Prior Work**:
    - **Inference-Time Methods** (CoT/ToT/RAP): Highly dependent on the base model's capabilities, search-based reasoning incurs high computational overhead, and diversity is limited by decoding strategies.
    - **SFT**: Requires a vast amount of labeled data to cover the diversity of solutions, leading to high annotation costs.
    - **Reward-Maximizing RL** (PPO): The goal is to find a single solution with the highest reward, which naturally ignores solution diversity.
- **Key Challenge**: There is a lack of a data-efficient method that targets the discovery of multiple distinct correct reasoning paths while maintaining reasoning quality.

## Method

### Core Idea: Reasoning as Markov Flow

The multi-step reasoning problem is modeled as a flow network on a directed acyclic graph (DAG):

- **State Nodes** $s_t$: Intermediate states in the reasoning process (e.g., block configurations in BlocksWorld)
- **Edges (Actions)** $s_t \to s_{t+1}$: A single reasoning step
- **Complete Trajectories** $\tau = (s_0 \to s_1 \to \cdots \to s_n)$: A complete reasoning path from the initial state to the terminal state
- **Goal**: To learn a forward policy $P_F(s_t | s_{t-1}; \theta, g)$ such that the probability of sampling a trajectory is proportional to the terminal reward $R(s_n)$

Unlike traditional RL (PPO) which pursues reward maximization, the goal of FoR is to sample multiple distinct paths proportionally to their rewards.

### Flow Decomposition and Forward Policy

By the Markov assumption, the trajectory probability is decomposed into step-by-step conditional probabilities:

$$P(\tau) = \prod_{t=1}^{n} P_F(s_t | s_{t-1})$$

The forward policy is parameterized by the LLM: $P_F(s_{t+1} | s_t; \theta, g) = P_{\text{LLM}}(a_t | s_t; \theta, g, c)$.

### Trajectory Balance Objective

The core training constraint:

$$Z(s_0, g) \prod_{t=1}^{n} P_F(s_t | s_{t-1}; \theta, g) = R(s_n) \prod_{t=1}^{n} P_B(s_{t-1} | s_t)$$

Where $P_B$ is the backward policy, defined as a uniform distribution $P_B(s_{t-1}|s_t) = 1/|\text{Pa}(s_t)|$.

### Log-Variance Loss

To avoid directly learning $\log Z$, a log-variance approximation is adopted:

$$\Phi(\tau; \theta) = \log R(s_n) + \sum_{t=1}^{n} \log P_B(s_{t-1}|s_t) - \sum_{t=1}^{n} \log P_F(s_t|s_{t-1}; \theta, g)$$

The final loss function:

$$\mathcal{L}_V(\tau; \theta) = \left(\Phi(\tau; \theta) - \mathbb{E}_\tau[\Phi(\tau; \theta)]\right)^2$$

Minimizing the variance of $\Phi$ across different trajectories forces the flow to satisfy the condition that terminal flow equals the reward.

### Efficient Exploration Strategies

- **On-policy**: Sample trajectories using the current policy $P_F$ and its temperature variants.
- **Off-policy**: Prioritized replay buffer (prioritizing high-reward trajectories) + $\epsilon$-sampling.
- **Local Search**: Select the highest-reward trajectory in a batch, truncate the second half, and reconstruct it using a random policy $P_U$ to efficiently explore the neighborhood of high-reward regions.

### Key Differences from GFN-CoT

FoR models at the **reasoning step level** (where each step corresponds to one reasoning operation) rather than the token level, thereby amortizing the computational overhead of search-based reasoning into the training phase.

## Key Experimental Results

**Base Model**: Llama-3-8B, with all fine-tuning methods using the same dataset.

### BlocksWorld (Embodied Reasoning)

| Method | 2-step Acc(%) | 4-step Acc(%) | 6-step Acc(%) | 6-step Diversity | 6-step Creativity(%) |
|------|-----------|-----------|-----------|--------------|------------------|
| CoT (1-shot) | 48.88 | 28.57 | 15.82 | 1.05 | 0.00 |
| CoT (GPT-4o) | 93.33 | 54.76 | 67.67 | 1.06 | 0.79 |
| RAP | 100.00 | 92.86 | 69.70 | - | - |
| O1-mini | 100.00 | 100.00 | 93.93 | 1.05 | 2.38 |
| SFT (α=1.0) | 44.44 | 42.06 | 34.68 | 1.04 | 4.76 |
| SFT + PPO | 46.66 | 44.44 | 24.58 | 1.08 | 3.17 |
| **FoR** | **100.00** | **98.41** | **78.44** | **1.33** | **9.52** |

### Main Results

- **Game24** (Math Puzzle): FoR discovers 3+ distinct correct solutions, whereas baseline methods (SFT/CoT) only repeatedly generate 1.
- **Rubik's Cube** (Spatial Reasoning): FoR significantly outperforms in both accuracy and diversity.
- **1D-ARC** (Abstract Reasoning): FoR shows outstanding performance.
- **GSM8K** (Mathematical Reasoning) & **ProntoQA** (Logical Reasoning): FoR significantly enhances solution diversity while maintaining high accuracy.
- Overall improvement over baselines is **20%–85%**, using only **15 training examples**.

### Ablation Study

- Local search contributes significantly to both training efficiency and final performance.
- Hybrid exploration (on-policy + off-policy + local search) achieves the best results.
- In the reward design, assigning high rewards to correct solutions and low rewards to incorrect ones is crucial for diversity.

## Highlights & Insights

1. **Extreme Data Efficiency**: Fine-tuning LLMs for diverse reasoning requires only 15 training examples, which is far superior to SFT's demand for large amounts of annotation.
2. **Elegant Principles**: Combining GFlowNet's flow matching theory with LLM reasoning naturally encourages diversity via reward-proportional sampling.
3. **Reasoning Step-Level Modeling**: Unlike token-level GFlowNets, modeling at the granularity of reasoning steps aligns better with the structural characteristics of multi-step reasoning.
4. **Amortizing Search Costs at Training Time**: Search costs are handled during the training phase, requiring only forward sampling during inference, which is much more efficient than ToT/RAP.
5. **Comprehensive Validation across Six Tasks**: Covers various reasoning types, including embodied, mathematical, spatial, abstract, and logical reasoning.
6. **Novel Creativity Metric**: Proposes a metric to measure the proportion of correct solutions uniquely discovered by a method, evaluating not only diversity but also "creativity."

## Limitations & Future Work

1. **DAG Structure Assumption**: Demands that the reasoning process can be modeled as a DAG, which might not apply to free-form, open-ended reasoning (e.g., creative writing).
2. **Reliance on Reward Functions**: Requires an explicit terminal reward $R(s_n)$, whereas many practical reasoning tasks lack clear, automated reward signals.
3. **State Transition Function**: Some tasks require an external environment simulator or LLM assistance to determine state transitions $T(s_t, a_t)$.
4. **Base Model Scale**: Validated only on Llama-3-8B; performance on larger-scale models remains unknown.
5. **Scalability**: As the number of reasoning steps grows, the trajectory space expands exponentially, potentially reducing exploration efficiency.
6. **Integration with RLHF/DPO**: The potential integration with alignment methods has not been explored.

## Related Work & Insights

- **GFlowNet** (Bengio et al., 2021): Theoretical foundation for reward-proportional sampling.
- **CoT/ToT/RAP**: Representative inference-time search methods; FoR amortizes the search into training.
- **GFN-CoT** (Hu et al., 2023): Token-level GFlowNet + LLM; FoR scales this to the reasoning-step level.
- **Insight**: The "sampling proportional to reward" paradigm of GFlowNets provides a theoretically elegant solution for LLM reasoning diversity, which can be extended to structured reasoning scenarios such as code generation and theorem proving.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The first step-level combination of GFlowNet and LLM reasoning, featuring an elegant framework)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Broad coverage with 6 tasks and solid ablations, but evaluated on only a single base model)
- Writing Quality: ⭐⭐⭐⭐⭐ (The flow analogy is intuitive and easy to understand, with clear formula derivations)
- Value: ⭐⭐⭐⭐⭐ (Fills an important gap in diverse LLM reasoning with extremely high data efficiency)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](../../NeurIPS2025/robotics/nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[CVPR 2026\] Obstruction Reasoning for Robotic Grasping](../../CVPR2026/robotics/obstruction_reasoning_for_robotic_grasping.md)
- [\[NeurIPS 2025\] EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT](../../NeurIPS2025/robotics/egothinker_unveiling_egocentric_reasoning_with_spatiotempora.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](../../NeurIPS2025/robotics/robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[CVPR 2026\] Parse, Search, and Confirmation: Training-Free Aerial Vision-and-Dialog Navigation with Chain-of-Thought Reasoning and Structured Spatial Memory](../../CVPR2026/robotics/parse_search_and_confirmation_training-free_aerial_vision-and-dialog_navigation_.md)

</div>

<!-- RELATED:END -->
