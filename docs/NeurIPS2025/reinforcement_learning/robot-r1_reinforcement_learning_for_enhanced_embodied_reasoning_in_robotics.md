---
title: >-
  [Paper Note] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics
description: >-
  [NeurIPS 2025][Reinforcement Learning][Embodied Reasoning] Robot-R1 proposes training large vision-language models (LVLMs) via reinforcement learning (GRPO) for embodied reasoning. By casting next keystate prediction as…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Embodied Reasoning"
  - "Robot Control"
  - "GRPO"
  - "Large Vision-Language Models"
date: 2026-05-08
content_hash: 960689c17ed43502
---

# Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics

**Conference**: NeurIPS 2025
**arXiv**: [2506.00070](https://arxiv.org/abs/2506.00070)
**Code**: [GitHub](https://github.com/hiyouga/EasyR1)
**Area**: Reinforcement Learning
**Keywords**: Embodied Reasoning, Reinforcement Learning, Robot Control, GRPO, Large Vision-Language Models

## TL;DR

Robot-R1 proposes training large vision-language models (LVLMs) via reinforcement learning (GRPO) for embodied reasoning. By casting next keystate prediction as multiple-choice questions and optimizing reasoning paths with RL, a 7B-parameter model surpasses GPT-4o on low-level control reasoning tasks.

## Background & Motivation

LVLMs have demonstrated considerable potential in robotics, capable of jointly processing visual inputs and natural language instructions for high-level robot control. However, current LVLMs still face challenges in **fine-grained embodied reasoning**:

**Limitations of SFT datasets**: Existing embodied reasoning datasets are largely constructed heuristically; their language descriptions fail to capture the precise quantitative information required for low-level robot control, and they are not optimized for actual robot action prediction.

**Catastrophic forgetting**: SFT approaches suffer severe performance degradation on out-of-distribution input-output formats, forgetting previously acquired general conversational capabilities.

**Insufficient generalization**: SFT-trained models show limited improvement in spatial and motion reasoning—capabilities critical for robot control—and struggle to generalize to novel tasks.

Inspired by DeepSeek-R1, the authors argue that **reinforcement learning can more effectively elicit and reinforce reasoning paths**, offering superior generalization and sample efficiency compared to SFT. The core insight is that if a model autonomously explores how to reason about next keystate prediction through RL, this reasoning capability naturally transfers to other embodied reasoning tasks.

## Method

### Overall Architecture

Robot-R1 follows a three-stage pipeline: (1) extracting metadata and keyframes from expert demonstrations to construct training data; (2) reformulating continuous state prediction as multiple-choice question answering (MCQA); and (3) training the LVLM with GRPO to generate and optimize reasoning chains. The framework also includes a purpose-built evaluation benchmark, Robot-R1 Bench.

### Key Designs

1. **Metadata-conditioned data generation**:
   Since LVLMs cannot directly infer low-level states from images, Robot-R1 introduces environment metadata $M$ comprising three components: (a) fixed environmental reference points (e.g., table center); (b) a 3D coordinate system definition with positive axis directions; and (c) end-effector dimensions for scale estimation. This metadata is provided as conditional input in the question prompt, helping the model build a quantitative understanding of the scene.

2. **Three MCQA task designs**:

   - **Waypoint prediction QA** (primary task): Given current observation $o_t$, state $s_t$, and metadata $M$, predict the next keyframe state $s_{k^*}$. Each question contains the correct answer and 3 distractors randomly sampled from the valid state space.
   - **Current state prediction QA** (auxiliary task): Identify the current state $s_t$ from visual observations, enhancing the model's understanding of its own state.
   - **Motion prediction QA** (auxiliary task): Predict the motion direction from the current state to the next keyframe (e.g., "move upward," "slightly backward"). Motion labels are extracted from 3D Cartesian displacements via rule-based heuristics.

   The core motivation for discretizing continuous state prediction into multiple-choice questions is that the continuous action space is too large for effective RL exploration; discretization narrows the action space, making learning more efficient.

3. **GRPO for reasoning path optimization**:
   During training, the model outputs its reasoning in `<think>...</think>` tags and the final answer in `<answer>...</answer>` tags. The GRPO algorithm generates $G$ distinct responses per query, using within-group relative advantage $A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$ to update the policy, with a KL divergence penalty to prevent excessive policy drift.

### Loss & Training

The reward signal consists of two components:
- **Format reward $r_f$**: Encourages the model to adhere to the prescribed output structure (`<think>`/`<answer>` tags).
- **Answer correctness reward $r_a$**: A rule-based binary reward granted when the model's answer exactly matches the correct multiple-choice option.

Total reward $R = r_f + r_a$. Training uses batch size 128, 5 epochs, learning rate $1.0 \times 10^{-6}$, and 5 sampled responses per prompt.

### Robot-R1 Bench

A purpose-designed open-ended QA benchmark evaluating four embodied reasoning capabilities: planning, high-level action reasoning, motion reasoning, and spatial reasoning. It comprises 215 questions manually created by experienced researchers, with GPT-4o serving as judge on a 0–3 scoring scale.

## Key Experimental Results

### Main Results: Robot-R1 Bench Low-Level Control Reasoning

| Model | Motion(In) | Motion(Out) | Motion(Avg) | Spatial(In) | Spatial(Out) | Spatial(Avg) |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 0.92 | 0.52 | 0.72 | 1.70 | 1.07 | 1.43 |
| Gemini-2.0-Flash | 0.52 | 0.40 | 0.46 | 1.76 | 1.14 | 1.49 |
| Qwen2.5-VL-7B-Ins | 0.64 | 0.52 | 0.58 | 1.62 | 1.11 | 1.40 |
| Direct SFT | 0.12 | 0 | 0.06 | 0.08 | 0.04 | 0.06 |
| CoT SFT | 0.84 | 0.56 | 0.70 | 0.46 | 0.07 | 0.29 |
| **Robot-R1 (Ours)** | **0.96** | **0.56** | **0.76** | **1.76** | **1.18** | **1.51** |

Robot-R1 achieves the highest scores across all low-level control reasoning metrics, surpassing all commercial models.

### EmbodiedBench Manipulation Benchmark

| Model | Base | Common | Complex | Spatial | Visual | Avg |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-7B-Ins | 6.3 | 6.3 | 6.3 | 14.6 | 11.1 | 8.92 |
| Direct SFT | 0 | 0 | 0 | 0 | 0 | 0 |
| CoT SFT | 0 | 0 | 0 | 0 | 0 | 0 |
| **Robot-R1 (Ours)** | **12.5** | **8.3** | **6.3** | **14.6** | **16.7** | **11.68** |

Robot-R1 improves approximately 31% over the baseline, while SFT-based methods entirely fail to complete any task.

### Ablation Study

| RL Algorithm | Planning | High-Level Action | Motion | Spatial |
|--------------|:---:|:---:|:---:|:---:|
| GRPO | 1.44 | 1.30 | 0.76 | 1.51 |
| RLOO | 1.56 | 1.54 | 0.68 | 1.52 |
| REINFORCE++ | 1.40 | 1.08 | 0.62 | 1.38 |

GRPO and RLOO perform comparably; REINFORCE++ underperforms due to higher variance introduced by batch-level reward normalization.

### Key Findings

- Training exclusively on low-level control tasks yields significant gains in high-level action reasoning—a transfer that SFT cannot achieve.
- Robot-R1 improves approximately 40% on quantitative and 60% on qualitative metrics on the SpatialRGPT benchmark.
- RL algorithms with variance-reduction mechanisms (GRPO/RLOO) are better suited for Robot-R1 training.
- Experiments across three random seeds confirm the robustness of the training results.

## Highlights & Insights

- **Elegant continuous-to-discrete reformulation**: Casting continuous state prediction as multiple-choice questions directly addresses the core challenge of RL exploration in large action spaces.
- **Natural transfer of reasoning capability**: Training only on low-level state prediction yields improvements in high-level reasoning—a compelling demonstration of emergent transfer.
- **Empirical evidence for RL over SFT**: SFT (including CoT-SFT) performs poorly or collapses entirely when transferred to out-of-distribution tasks, whereas RL-trained models exhibit substantially stronger generalization.

## Limitations & Future Work

- Training data is drawn from only 5 tabletop manipulation tasks in RLBench, limiting scene diversity.
- Planning capability shows a slight decline, as the training objective focuses on next-keyframe prediction rather than long-horizon planning.
- Only 7B-scale models are evaluated; the effectiveness of larger models remains unknown.
- The impact of the number of answer choices and distractor generation strategies on training performance is insufficiently explored.

## Related Work & Insights

- **DeepSeek-R1**: The paradigm for RL-based reasoning model training; Robot-R1 extends this approach to the embodied domain.
- **ARM (James et al.)**: The source of the keyframe extraction methodology.
- **EmbodiedBench, SpatialRGPT**: External evaluation benchmarks used to validate generalization.
- Insight: RL-trained embodied reasoning capability may be a critical pathway toward general-purpose robot intelligence.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Transferring the R1-style RL reasoning training paradigm to robotic embodied reasoning is novel and meaningful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three benchmark evaluations + ablations + seed robustness + human agreement validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-articulated motivation.
- **Value**: ⭐⭐⭐⭐ Introduces an effective RL-based reasoning training paradigm for robotics with strong practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)
- [\[NeurIPS 2025\] Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning](router-r1_teaching_llms_multi-round_routing_and_aggregation_via_reinforcement_le.md)
- [\[NeurIPS 2025\] Learning Memory-Enhanced Improvement Heuristics for Flexible Job Shop Scheduling](learning_memory-enhanced_improvement_heuristics_for_flexible_job_shop_scheduling.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
