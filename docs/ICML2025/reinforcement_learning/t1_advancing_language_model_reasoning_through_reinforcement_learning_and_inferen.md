---
title: >-
  [Paper Note] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling
description: >-
  [ICML 2025][Reinforcement Learning][Reasoning Ability] T1 scales up open-source LLMs' reasoning performance by utilizing synthetic CoT data for initialization, combined with oversampling and entropy rewards during RL training to encourage exploration. This enables open-source models to exhibit inference-time scaling behavior, outperforming QwQ-32B-Preview on challenging mathematical reasoning benchmarks such as MATH500 and AIME2024.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Reasoning Ability"
  - "Inference-time Scaling"
  - "Chain-of-Thought"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: 60443aefecccd885
---

# T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling

**Conference**: ICML 2025  
**arXiv**: [2501.11651](https://arxiv.org/abs/2501.11651)  
**Code**: [https://github.com/THUDM/T1](https://github.com/THUDM/T1)  
**Area**: LLM Reasoning / Reinforcement Learning  
**Keywords**: Reinforcement Learning, Reasoning Ability, Inference-time Scaling, Chain-of-Thought, Mathematical Reasoning

## TL;DR

T1 scales up open-source LLMs' reasoning performance by utilizing synthetic CoT data for initialization, combined with oversampling and entropy rewards during RL training to encourage exploration. This enables open-source models to exhibit inference-time scaling behavior, outperforming QwQ-32B-Preview on challenging mathematical reasoning benchmarks such as MATH500 and AIME2024.

## Background & Motivation

1. **Background**: Large language models have demonstrated outstanding capabilities on complex reasoning tasks. However, existing methods heavily rely on imitation learning (e.g., distilling CoT data) and perform poorly in test-time scaling.

2. **Limitations of Prior Work**: Although reinforcement learning (RL) holds promise for improving reasoning capabilities through autonomous exploration and feedback learning, recent RL implementations have yielded only limited improvements on complex reasoning tasks. The core problem lies in the fact that models easily fall into local optima, suffer from insufficient sampling diversity, and experience unstable policy optimization during RL training.

3. **Key Challenge**: In theory, RL can enable models to learn superior reasoning strategies through trial and error. However, in practice, the instability of RL training and insufficient exploration limit its gains on LLM reasoning. How to scale up RL training has become a key bottleneck.

4. **Goal**: To effectively enhance the reasoning capabilities of LLMs through RL while achieving inference-time scaling—meaning that increasing the inference computation budget yields better performance.

5. **Key Insight**: Encourage RL exploration from three dimensions: (1) cold-starting with synthetic CoT data that integrates trial-and-error and self-verification, (2) increasing sampling diversity during RL training using oversampling, and (3) stabilizing training with an entropy reward combined with dynamic anchor regularization.

6. **Core Idea**: Enable open-source LLMs to acquire o1-like reasoning capabilities and inference-time scaling behavior through carefully designed SFT initialization and RL training techniques.

## Method

### Overall Architecture

The training of T1 consists of two phases:

1. **SFT Initialization Phase**: Supervised fine-tuning is performed on the base LLM using synthetic CoT data to equip it with basic "long thinking" capabilities.
2. **RL Training Phase**: Reinforcement learning is employed to further optimize the model's reasoning strategies and encourage more exploration.

Input: Mathematical reasoning problem → T1 generates a long CoT containing trial-and-error, backtracking, and self-verification → Output: Final answer.

### Key Designs

1. **Synthetic CoT Data (SFT Initialization)**:
    - **Function**: Construct synthetic CoT training data that integrates trial-and-error and self-verification.
    - **Mechanism**: Differing from traditional concise CoTs, the training data of T1 simulates human thought processes, incorporating attempts, errors, backtracking, and verification. This data teaches the model to actively self-check and self-correct during the reasoning process.
    - **Design Motivation**: Traditional CoT data usually presents a "one-shot correct" solution path, where the model only learns to imitate correct trajectories without acquiring autonomous error-correction capabilities. Introducing trial-and-error and verification steps provides the model with a superior starting point for the RL phase.

2. **Oversampling Strategy**:
    - **Function**: Generate more candidate responses for each query during RL training (increasing the sample size).
    - **Mechanism**: Increase sampling diversity by selecting reward signals from $K$ candidates for policy updates. More samples translate to a larger exploration space.
    - **Design Motivation**: Insufficient sampling diversity during RL training is a primary cause of models trapping in local optima. Oversampling directly expands the search space for policy optimization.

3. **Entropy Bonus**:
    - **Function**: Incorporate policy distribution entropy into the RL reward function as an auxiliary loss.
    - **Mechanism**: Total reward = task reward + $\beta \cdot H(\pi_\theta)$, where $H(\pi_\theta)$ is the policy entropy, and $\beta$ controls exploration intensity.
    - **Design Motivation**: Prevent the policy from converging prematurely to a specific type of solution during RL training, thereby preserving the model's ability to generate diversified reasoning paths.

4. **Dynamic Anchor Regularization**:
    - **Function**: Introduce a dynamically updated reference policy as an anchor for KL divergence regularization.
    - **Mechanism**: Regularize using $D_{KL}(\pi_\theta \| \pi_{anchor})$, where $\pi_{anchor}$ is not a fixed initial policy but is updated dynamically along with the training.
    - **Design Motivation**: Conventional PPO/RLHF utilizes a fixed SFT policy as the KL anchor, which restricts how far the RL optimizer can go. Dynamic anchors allow for larger policy update steps while maintaining training stability.

### Loss & Training

- Base Models: Qwen2.5 series (7B/14B/32B) and GLM-4-9B
- SFT Phase: Utilizing synthetic CoT data (released on HuggingFace)
- RL Phase: Policy optimization based on correct-answer rewards of mathematical problems
- Inference-time Scaling Strategy: Performance scales by increasing the inference token budget, without requiring an external verifier

## Key Experimental Results

### Main Results

| Model | MATH500 | AIME2024 | Omni-math-500 | GPQA |
|------|---------|----------|---------------|------|
| GPT-4o | 76.6 | 9.3 | 26.8 | 53.6 |
| Claude-3.5-sonnet | 78.3 | 16.0 | - | 65.0 |
| Llama-3.3-70B-Instruct | 73.9 | 24.2 | 27.9 | 50.5 |
| Qwen2.5-Math-7B-Instruct | 82.7 | 16.7 | 29.7 | 36.9 |
| o1-preview | 85.5 | 44.6 | - | 72.3 |
| QwQ-32B-Preview | 90.6 | 50.0 | 46.6 | 58.2 |
| **T1-SFT (Qwen2.5-32B)** | 83.4 | 24.9 | 34.6 | 49.5 |
| **T1 (Qwen2.5-32B)** | **92.4** | **50.6** | **49.6** | 56.1 |

### Ablation Study

| Configuration | MATH500 | AIME2024 | Description |
|------|---------|----------|------|
| T1-SFT (GLM-4-9B) | 60.2 | 4.1 | SFT-only, small model |
| T1 (GLM-4-9B) | 65.8 | 9.2 | Significant improvement after +RL |
| T1-SFT (Qwen2.5-14B) | 77.2 | 10.3 | SFT-only, medium model |
| T1 (Qwen2.5-14B) | 87.4 | 30.5 | 3x improvement on AIME after +RL |
| T1-SFT (Qwen2.5-32B) | 83.4 | 24.9 | SFT-only, large model |
| T1 (Qwen2.5-32B) | 92.4 | 50.6 | Outperforms QwQ across the board after +RL |

### Key Findings

1. **Huge Gains from RL Training**: Moving from SFT to RL, the 14B model's performance on AIME2024 surges from 10.3 to 30.5 (approx. 3x), and the 32B model climbs from 24.9 to 50.6 (approx. 2x). This indicates that the improvement in exploratory reasoning capability brought by RL training far exceeds SFT.
2. **Inference-time Scaling Behavior**: Increasing the inference token budget consistently improves T1's performance **without requiring additional verifiers** (such as majority voting or reward model reranking), achieving a native inference scaling behavior similar to o1.
3. **Competitiveness of Open-Source Models**: T1 (32B) outperforms QwQ-32B-Preview and o1-preview on MATH500 and AIME2024, demonstrating the potential of open-source RL training methods.

## Highlights & Insights

- **The Crucial Leap from SFT to RL**: SFT provides a blueprint of "how to think", while RL acquires higher-quality reasoning strategies through autonomous exploration. This two-phase paradigm could become a standard pipeline for training reasoning models.
- **Democratization of Inference-time Scaling**: Previously, only closed-source models (such as o1) exhibited inference-time scaling. T1 proves that open-source models can also achieve this capability through RL training.
- **Systematic Design of Exploration Encouragement Mechanisms**: Oversampling, entropy rewards, and dynamic anchors work in unison to address the issue of insufficient exploration in RL training.

## Limitations & Future Work

1. Currently validated only on mathematical reasoning tasks; the generalization capability on other reasoning tasks such as code generation and logical reasoning remains unknown.
2. RL training incurs high computational costs (due to extensive sampling). Lowering the training overhead is crucial for practical deployment.
3. The impact of the updating strategy of the dynamic anchor on final performance requires more detailed ablation studies.
4. Performance on non-mathematical tasks like GPQA during inference-time scaling has not been fully verified.

## Related Work & Insights

- **OpenAI o1/o3**: Pioneers of closed-source reasoning models; T1 can be considered their open-source alternative.
- **DeepSeek-R1**: Another attempt to enhance reasoning capabilities through RL training.
- **QwQ-32B-Preview**: A reasoning model by the Qwen team; T1 builds upon and further enhances it.
- **Insights**: The design of exploration mechanisms in RL training may be more critical than model scale—a 14B model equipped with sound RL training can approach the SFT performance of a 72B model.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically addresses the exploration issue in RL training, though individual components (entropy reward, oversampling) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple model scales and benchmarks, but lacking non-mathematical reasoning tasks.
- Writing Quality: ⭐⭐⭐⭐ Clearly presents the methodology and experiments with a well-structured layout.
- Value: ⭐⭐⭐⭐⭐ Inference-time scaling for open-source models carries significant practical value and advances the open-source community's progress in reasoning models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] BRITE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning](brite_bootstrapping_reinforced_thinking_process_to_enhance_language_model_reason.md)
- [\[ICCV 2025\] R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](../../ICCV2025/reinforcement_learning/r1-onevision_advancing_generalized_multimodal_reasoning_through_cross-modal_form.md)
- [\[ICML 2025\] Optimizing Language Models for Inference Time Objectives using Reinforcement Learning](optimizing_language_models_for_inference_time_objectives_using_reinforcement_lea.md)
- [\[ICLR 2026\] Structured In-context Environment Scaling for Large Language Model Reasoning](../../ICLR2026/reinforcement_learning/structured_in-context_environment_scaling_for_large_language_model_reasoning.md)
- [\[ICML 2025\] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning](network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
