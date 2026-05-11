---
title: >-
  [Paper Note] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge
description: >-
  [ACL 2026][Multimodal VLM][MLLM-as-a-Judge] This paper proposes MT-RL-Judge, a multi-task reinforcement learning framework that jointly optimizes multiple evaluation tasks via GRPO to train a unified MLLM-as-a-Judge mode…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "MLLM-as-a-Judge"
  - "multi-task reinforcement learning"
  - "GRPO"
  - "unified evaluation"
  - "out-of-distribution generalization"
date: 2026-05-08
content_hash: 9462c8a8564a86f5
---

# Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge

**Conference**: ACL 2026
**arXiv**: [2603.11665](https://arxiv.org/abs/2603.11665)
**Code**: None
**Area**: Multimodal VLM / Automatic Evaluation
**Keywords**: MLLM-as-a-Judge, multi-task reinforcement learning, GRPO, unified evaluation, out-of-distribution generalization

## TL;DR

This paper proposes MT-RL-Judge, a multi-task reinforcement learning framework that jointly optimizes multiple evaluation tasks via GRPO to train a unified MLLM-as-a-Judge model. The framework consistently outperforms SFT baselines across six benchmarks covering text-image alignment, safety compliance, and visual quality assessment, and demonstrates strong out-of-distribution generalization on the unseen MJ-Bench pairwise comparison format (82.23% on Safety vs. 49.40% for SFT-Unified).

## Background & Motivation

**Background**: The MLLM-as-a-Judge paradigm has become the dominant approach for large-scale multimodal content evaluation. Existing methods fall into two categories: (1) prompt-based judges that directly apply off-the-shelf MLLMs with evaluation guidelines injected via prompt engineering; and (2) fine-tuned judges trained on specific evaluation datasets via SFT or RL.

**Limitations of Prior Work**: (1) Judges that rely solely on prompt engineering perform poorly on complex tasks and require task-specific training. (2) Existing trainable judges typically focus on a single task (e.g., safety compliance or image quality) and fail to generalize across diverse evaluation scenarios. (3) SFT-trained judges tend to overfit to specific instruction formats—models trained on pointwise evaluation cannot handle pairwise comparison tasks, which is highly impractical in industrial settings where requirements frequently change. (4) Deploying multiple specialized judge models incurs high maintenance costs and inference overhead.

**Key Challenge**: The maximum likelihood estimation objective of SFT encourages models to memorize surface-level statistical associations between inputs and outputs rather than internalizing evaluation reasoning logic—leading to strong in-distribution performance but catastrophic failure under minor changes in format or task.

**Goal**: To construct a unified, RL-enhanced MLLM-as-a-Judge framework capable of handling diverse evaluation tasks simultaneously while preserving generalization to unseen task formats.

**Key Insight**: Leveraging RL (specifically GRPO) to encourage the model to generate reasoning steps before rendering a judgment, thereby internalizing evaluation logic rather than memorizing surface patterns. Multi-task training exposes the model to shared criteria across different evaluation dimensions, further enhancing generalization.

**Core Idea**: Multi-task RL + reasoning-first = a unified judge that is both accurate and generalizable.

## Method

### Overall Architecture

MT-RL-Judge adopts a two-stage training pipeline: (1) an optional SFT warm-up stage; and (2) a multi-task GRPO reinforcement learning stage. Training data are drawn from a unified dataset $D_{unified} = \bigcup_{k=1}^{K} D_k$ aggregated from six evaluation benchmarks, covering text-image alignment (SeeTRUE, ImageReward), safety compliance (UnsafeBench), and visual quality (three AGIN subsets). The model is built on the Qwen3-VL-30B-A3B-Instruct backbone.

### Key Designs

1. **Composite Reward Function**:

    - Function: Jointly optimizes output format compliance and judgment accuracy.
    - Mechanism: The total reward is a linear combination of an accuracy reward and a format reward: $R_{total} = (1-\alpha) \cdot R_{Acc} + \alpha \cdot R_{For}$. $R_{Acc}$ equals 1.0 when the prediction is correct and 0.0 otherwise; $R_{For}$ equals 1.0 when the output follows the reasoning-first structure and 0.0 otherwise. $\alpha$ controls the balance between the two components.
    - Design Motivation: The format reward ensures the model generates a reasoning process before rendering a judgment (analogous to CoT), improving both judgment quality and interpretability; the accuracy reward directly optimizes prediction correctness.

2. **GRPO Multi-Task Training Objective**:

    - Function: Jointly optimizes a global policy across multiple evaluation tasks without requiring a separate value function.
    - Mechanism: For each prompt in the unified dataset, $G=20$ outputs are sampled from the current judge model, and within-group relative rewards are computed for policy optimization. The optimization objective is to maximize the expected reward over the entire unified dataset: $\theta^* = \arg\max_\theta \mathbb{E}_{(x,p,y) \sim D_{unified}} [R_{total}(M_\theta(x))]$. GRPO replaces the explicit value function with the within-group mean reward, simplifying training.
    - Design Motivation: Unlike single-task RL training, multi-task RL exposes the model to shared criteria and latent associations across different evaluation domains, preventing overfitting to specific prompt templates.

3. **Reasoning-First Evaluation Paradigm**:

    - Function: Requires the judge to explicitly generate a reasoning process before producing its final judgment.
    - Mechanism: RL prompts instruct the model to first output a reasoning trace and then produce the final prediction. The format reward $R_{For}$ enforces this structure. By elaborating the analytical logic through reasoning, the model more accurately approximates the evaluation logic aligned with human preferences.
    - Design Motivation: SFT tends to imitate the statistical mapping from inputs to outputs rather than learning the underlying reasoning process, resulting in poor generalization. The reasoning-first paradigm compels the model to "think" before it "judges"—a capability that is especially critical when facing unseen task formats.

### Loss & Training

The SFT stage uses the AdamW optimizer with a learning rate of $1.0 \times 10^{-5}$, a batch size of 256, and full-parameter fine-tuning. The RL stage uses GRPO with rollout $N=20$, a global batch size of 256, and a rollout batch size of 512. All models are stopped upon validation performance plateau, and the best checkpoint is selected.

## Key Experimental Results

### Main Results

**Macro-F1 Comparison across Six Evaluation Tasks**

| Method | AGIN-Nat | AGIN-Tech | AGIN-Rat | SeeTRUE | ImageReward | UnsafeBench |
|--------|----------|-----------|----------|---------|-------------|-------------|
| Off-the-shelf | 67.99 | 63.24 | 64.77 | 80.01 | 55.07 | 72.78 |
| SFT-Single | 78.64 | 77.04 | 78.08 | 80.41 | 64.95 | 90.28 |
| SFT-Unified | 81.75 | 81.22 | 81.31 | 82.32 | 63.34 | 89.49 |
| RL-Single | 80.50 | 80.77 | 82.71 | 83.41 | 65.07 | 86.92 |
| **MT-RL-Judge** | 81.63 | **81.37** | 81.58 | **83.67** | 64.97 | 85.22 |

### Ablation Study

**Out-of-Distribution Generalization on MJ-Bench (Unseen Pairwise Format)**

| Method | Image-text Alignment | Safety Judge |
|--------|---------------------|--------------|
| Off-the-shelf | 59.41 | 73.07 |
| SFT-Unified | 55.82 | 49.40 |
| **MT-RL-Judge** | **60.59** | **82.23** |

### Key Findings

- RL consistently outperforms SFT: RL-Single surpasses SFT-Single on 5 out of 6 tasks, with particularly pronounced gains on reasoning-intensive tasks (SeeTRUE +3.0, AGIN-Rat +4.63).
- Unified training improves rather than degrades performance: SFT-Unified outperforms SFT-Single on most tasks, indicating that multi-task exposure enables the model to learn cross-domain shared evaluation criteria.
- SFT severely overfits to format: SFT-Unified collapses to 49.40% on MJ-Bench Safety—far below the zero-shot baseline of 73.07%—despite having been trained on safety evaluation tasks, failing entirely merely because the input changes from a single image to an image pair.
- MT-RL-Judge demonstrates strong generalization: On the completely unseen pairwise format, it achieves 82.23% on Safety (surpassing SFT-Unified by 32.83 percentage points), confirming that RL-driven reasoning capabilities transfer to new formats.

## Highlights & Insights

- The catastrophic degradation of SFT-Unified on MJ-Bench constitutes the most compelling evidence in the paper, clearly exposing the fundamental flaw of SFT in memorizing format rather than learning evaluation logic.
- The "1+1>2" effect of multi-task RL: joint training not only avoids sacrificing single-task performance but enhances overall results through cross-task knowledge sharing.
- The reasoning-first design philosophy aligns with the trend of recent reasoning models (e.g., DeepSeek-R1, QwQ), and its application to the evaluation setting is a first.
- The experimental design is precise: the four-way comparison of SFT-Single vs. SFT-Unified vs. RL-Single vs. MT-RL-Judge cleanly disentangles the individual contributions of unified training and RL.

## Limitations & Future Work

- Experiments are conducted primarily on binary classification evaluation tasks, without covering multi-level scoring or open-ended evaluation.
- The backbone model is Qwen3-VL-30B-A3B (MoE); effectiveness on other architectures has not been validated.
- The six training tasks offer limited coverage; incorporating more evaluation dimensions may further strengthen generalization.
- The quality and diversity of reasoning traces are not analyzed; whether the reasoning process genuinely reflects evaluation logic requires further investigation.

## Related Work & Insights

- **vs. SFT-based Judges (MLLM-as-a-Judge)**: SFT judges overfit to prompt templates, whereas MT-RL-Judge internalizes evaluation logic through RL-driven reasoning.
- **vs. Mr. Judge / Flex-Judge**: Existing work is predominantly single-task; MT-RL-Judge is the first to realize a unified multi-task RL judge.
- **vs. Self-Consistency Methods**: Sampling multiple candidates and voting is computationally expensive and may aggregate similar errors; MT-RL-Judge improves judgment accuracy through reasoning quality.

## Rating

- Novelty: ⭐⭐⭐⭐ First unified multi-task RL MLLM-as-a-Judge framework; the finding that SFT overfits to format provides a deep and insightful observation.
- Experimental Thoroughness: ⭐⭐⭐ Six task benchmarks are sufficient, but more OOD generalization tests and reasoning quality analyses are lacking.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear and experimental comparisons are elegantly designed, though methodological details are somewhat concise.
- Value: ⭐⭐⭐⭐ Provides a unified and generalizable solution for industrial-grade multimodal evaluation; the MJ-Bench generalization results are convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](../../NeurIPS2025/multimodal_vlm/srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](../../AAAI2026/multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](../../CVPR2026/multimodal_vlm/explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[ACL 2026\] Targeted Exploration via Unified Entropy Control for Reinforcement Learning](targeted_exploration_via_unified_entropy_control_for_reinforcement_learning.md)
- [\[ACL 2026\] From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models](from_heads_to_neurons_causal_attribution_and_steering_in_multi-task_vision-langu.md)

</div>

<!-- RELATED:END -->
