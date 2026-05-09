---
title: >-
  [Paper Note] Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress
description: >-
  [CVPR 2026][Multimodal VLM][task progress estimation] This paper proposes R²VLM, a recurrent reasoning framework that processes local video segments sequentially, maintains a dynamically updated CoT record tracking task decomposition and completion status, and leverages a multi-dimensional RL reward scheme to achieve state-of-the-art performance in long-horizon embodied task progress estimation. The framework additionally supports downstream applications including policy learning, reward modeling, and proactive assistance.
tags:
  - CVPR 2026
  - Multimodal VLM
  - task progress estimation
  - embodied intelligence
  - recurrent reasoning
  - Chain-of-Thought
  - reinforcement learning
date: 2026-05-08
content_hash: 7b25653e9863f7a0
---

# Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress

**Conference**: CVPR 2026
**arXiv**: [2603.17312](https://arxiv.org/abs/2603.17312)
**Code**: [HuggingFace](https://huggingface.co/)
**Area**: Multimodal VLM
**Keywords**: task progress estimation, embodied intelligence, recurrent reasoning, Chain-of-Thought, reinforcement learning

## TL;DR
This paper proposes R²VLM, a recurrent reasoning framework that processes local video segments sequentially, maintains a dynamically updated CoT record tracking task decomposition and completion status, and leverages a multi-dimensional RL reward scheme to achieve state-of-the-art performance in long-horizon embodied task progress estimation. The framework additionally supports downstream applications including policy learning, reward modeling, and proactive assistance.

## Background & Motivation
**State of the Field**: Embodied agents require accurate estimation of execution progress over multi-step, long-horizon tasks to support long-range planning and context-aware decision-making.

**Limitations of Prior Work**:
   - Methods such as GVL and ROVER exploit VLMs' video understanding capabilities and large context windows while neglecting their reasoning potential.
   - Processing long video trajectories (often thousands of frames) incurs prohibitive computational overhead, making real-time deployment impractical.
   - Tasks involving multiple temporally dependent subtasks demand reasoning capabilities to align visual observations with logical dependencies.

**Root Cause**: Full-video processing is computationally prohibitive, whereas local segment processing lacks global context; video understanding alone is insufficient to handle complex temporal-logical dependencies.

**Paper Goals**: Efficient, accurate, and interpretable estimation of long-horizon embodied task progress.

**Starting Point**: Emulating human cognition — "observe a segment, reason about it, and retain key information" — by recurrently processing video segments while maintaining structured memory.

**Core Idea**: Recurrent reasoning combined with a dynamic CoT as a cross-timestep memory carrier, avoiding full-video processing while preserving global context.

## Method

### Overall Architecture
The input video is segmented into short clips $v_t$ (4s/2s). At each reasoning step, the model receives the current clip $v_t$ and the historical CoT $c_{t-1}$, producing an updated CoT $c_t$ and a progress estimate $p_t$: $c_t, p_t = f_\theta(\tau, v_t, c_{t-1})$.

### Key Designs

1. **Recurrent Reasoning Framework**:

    - Initial iteration: the VLM leverages commonsense knowledge to generate an initial CoT $c_0$ representing task decomposition.
    - Subsequent iterations: task decomposition is dynamically refined (merging/splitting/reordering steps) based on new video segments, with completion status updated accordingly.
    - Three core advantages: (1) CoT improves accuracy and interpretability; (2) historical CoT provides global context; (3) inheriting prior-round reasoning ensures logical consistency.
    - Design Motivation: eliminates redundant computation from processing full-length videos by propagating global information through the CoT.

2. **CoT Structure**:

    - Three components: (i) task decomposition (listing subtasks); (ii) key step analysis (completed/pending); (iii) progress estimation based on the proportion of completed steps.
    - The decomposition can be dynamically adjusted each round, as the environment is partially observable and the actual execution may not fully align with the initial decomposition.

3. **Multi-Dimensional RL Reward System (PPO)**:

    - **Format Reward ($R_{fmt}$)**: verifies output format (think/answer tags); compliant = 1.
    - **Bin Reward ($R_{bin}$)**: checks whether the predicted progress falls within the correct step interval; correct = 1.0, adjacent = 0.25.
    - **MAE Reward ($R_{mae}$)**: $\max(1 - |p_t - p_t^{gt}|/\delta_1, 0)$, providing fine-grained supervision.
    - **Improvement Reward ($R_{imp}$)**: encourages each round's prediction error to be lower than the previous round, reflecting the self-correction capability of recurrent reasoning.
    - **Finish Reward ($R_{fin}$)**: rewards correct judgment of task completion.
    - Overall reward: $R_{overall} = R_{fmt} \cdot (R_{bin} \cdot R_{mae} + \alpha R_{imp} + \beta R_{fin})$
    - PPO over GRPO: in the multi-turn setting, $c_{t-1}$ differs across trajectories, violating GRPO's requirement of identical inputs for generating multiple candidates.

### Loss & Training
Two-stage training: (1) SFT to learn the reasoning pattern; (2) multi-turn PPO reinforcement learning initialized from the cold-start SFT checkpoint.

## Key Experimental Results

### Main Results

| Model | Size | ALFRED $p_{mae}$↓ | ALFRED $bin$↑ | Ego4D $p_{mae}$↓ | Ego4D $bin$↑ |
|------|------|:---------:|:---------:|:---------:|:---------:|
| GPT-5 | - | 18.35 | 0.505 | 25.04 | 0.259 |
| Gemini-2.5-Pro | - | 16.27 | 0.481 | 28.22 | 0.217 |
| Qwen2.5-VL-72B | 72B | 24.88 | 0.342 | 26.88 | 0.254 |
| **R²VLM (SFT+RL)** | **7B** | **6.34** | **0.758** | **11.88** | **0.526** |

### Ablation Study

| Configuration | ALFRED $p_{mae}$↓ | Note |
|------|:---------:|------|
| SFT only | 7.52 | Basic supervised fine-tuning |
| + RL (w/o $R_{imp}$) | 6.89 | Missing cross-round improvement signal |
| + RL (w/o $R_{bin}$) | 7.11 | Missing coarse-grained step constraint |
| **Full R²VLM** | **6.34** | All rewards combined, best performance |

### Key Findings
- R²VLM at 7B comprehensively outperforms GPT-5 and Gemini-2.5-Pro, reducing MAE by over 65%.
- The Improvement Reward contributes significantly to multi-turn reasoning, demonstrating the value of self-correction in recurrent reasoning.
- Strong generalization is observed across three downstream tasks: progress-augmented policy learning, reward modeling, and proactive assistance.
- Recurrent reasoning avoids full-video processing, yielding substantially faster inference compared to global methods.

## Highlights & Insights
- **Recurrent Reasoning + CoT as Memory**: the CoT is extended from a one-shot reasoning tool to a structured cross-timestep memory carrier, maintaining global consistency while avoiding long-video computation. This paradigm is transferable to any VLM task requiring long-span temporal reasoning.
- **Improvement Reward Design**: directly measures the model's self-correction capability by rewarding cross-round error reduction — a design unique to multi-turn reasoning settings.
- **Automated Data Generation Pipeline**: expert trajectories from ALFRED/Ego4D are automatically converted into video segment + CoT training pairs, including a strategy for generating distractor task descriptions.
- **Multi-Downstream Validation**: beyond progress estimation, the framework demonstrates value as an RL reward model and a proactive assistance system.

## Limitations & Future Work
- The quality of CoT step decomposition heavily depends on the VLM's commonsense reasoning, which may be inaccurate for complex novel tasks.
- ALFRED is a simulation environment; performance on real-world data (Ego4D) still exhibits a notable gap.
- Clip length is fixed (4s/2s), with no mechanism for dynamically adjusting segment granularity.
- Experiments are limited to Qwen2.5-VL-7B; the effectiveness on larger or stronger base models remains unverified.

## Related Work & Insights
- **vs. GVL / ROVER**: these methods rely on VLM in-context learning and large context windows without explicit reasoning, limiting performance on complex long-horizon tasks. R²VLM achieves substantial improvements through recurrent reasoning and RL.
- **vs. Hierarchical Reward Methods**: conventional approaches require manually designed task hierarchy decompositions, whereas R²VLM automatically learns decomposition and reasoning strategies through training.

## Supplementary Analysis
- Progress is defined as the proportion of completed steps rather than elapsed time, better reflecting long-horizon task structure given the large variance in per-step duration.
- The distractor task generation strategy is elegant: the first $n_r$ steps are forced to match the original task while subsequent steps differ, enabling precise control over ground-truth progress.
- Human-verified benchmark retention rates of 93% (ALFRED) and 74% (Ego4D) indicate high quality of the automatically generated data.
- The technical rationale for choosing PPO over GRPO is that GRPO requires generating multiple candidates from the same input, which is incompatible with the recurrent setting where $c_{t-1}$ differs across trajectories.
- The asymmetric range [-1, 0.8] of the Improvement Reward amplifies the penalty for error increases, encouraging conservative yet stable progress estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ — the recurrent reasoning + CoT memory framework is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — two datasets, four metrics, and three downstream applications.
- Writing Quality: ⭐⭐⭐⭐ — clear structure with detailed method descriptions.
- Value: ⭐⭐⭐⭐⭐ — significant implications for progress estimation and reward modeling in embodied AI.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)
- [\[CVPR 2026\] Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation](training_high-level_schedulers_with_execution-feedback_reinforcement_learning_fo.md)
- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)
- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)

<!-- RELATED:END -->
