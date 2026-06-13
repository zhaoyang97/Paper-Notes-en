---
title: >-
  [Paper Note] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge
description: >-
  [ACL 2026][LLM Evaluation][MLLM-as-a-Judge] This paper proposes MT-RL-Judge, a multi-task reinforcement learning framework that jointly optimizes multiple evaluation tasks using GRPO to train a unified MLLM-as-a-Judge mo…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "MLLM-as-a-Judge"
  - "Multi-task Reinforcement Learning"
  - "GRPO"
  - "Unified Evaluation"
  - "Out-of-distribution Generalization"
date: 2026-05-08
content_hash: d684e27feafe8faa
---

# Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge

**Conference**: ACL 2026  
**arXiv**: [2603.11665](https://arxiv.org/abs/2603.11665)  
**Code**: None  
**Area**: Multimodal VLM / Automated Evaluation  
**Keywords**: MLLM-as-a-Judge, Multi-task Reinforcement Learning, GRPO, Unified Evaluation, Out-of-distribution Generalization

## TL;DR

This paper proposes MT-RL-Judge, a multi-task reinforcement learning framework that jointly optimizes multiple evaluation tasks using GRPO to train a unified MLLM-as-a-Judge model. It consistently outperforms SFT baselines across six benchmarks including text-to-image alignment, safety compliance, and visual quality assessment, while demonstrating strong out-of-distribution (OOD) generalization on the unseen MJ-Bench pairwise comparison format (82.23% on Safety vs. 49.40% for SFT-Unified).

## Background & Motivation

**Background**: The MLLM-as-a-Judge paradigm has become a mainstream solution for large-scale multimodal content evaluation. Existing methods are categorized into: (1) Prompt-based Judges—directly using off-the-shelf MLLMs with injected evaluation guidelines via prompt engineering; (2) Fine-tuned Judges—improving evaluation capabilities through SFT or RL on specific evaluation datasets.

**Limitations of Prior Work**: (1) Prompt-based judges perform poorly on complex tasks and require task-specific training; (2) Existing trainable judges usually focus on a single task (e.g., safety or quality) and fail to generalize to diverse scenarios; (3) SFT-trained judges easily overfit to specific instruction formats—models trained on pointwise evaluation fail to handle pairwise tasks, which is impractical for industrial scenarios with frequent demand changes; (4) Deploying multiple specialized judges leads to high maintenance and inference overhead.

**Key Challenge**: The maximum likelihood estimation objective of SFT encourages models to memorize surface statistical associations between inputs and outputs rather than internalizing evaluation reasoning logic—causing models to perform well on the training distribution but collapse when faced with minor format or task changes.

**Goal**: To build a unified, RL-enhanced MLLM-as-a-Judge framework capable of handling diverse evaluation tasks simultaneously while maintaining generalization to unseen task formats.

**Key Insight**: Leveraging RL (specifically GRPO) to encourage the model to generate reasoning steps before providing a judgment, thereby internalizing evaluation logic instead of memorizing surface patterns. Multi-task training exposes the model to shared standards across different evaluation dimensions, further enhancing generalization.

**Core Idea**: Multi-task RL + reasoning-first approach = an accurate and generalizable unified judge.

## Method

### Overall Architecture

MT-RL-Judge adopts a two-stage training process: (1) An optional SFT pre-training stage; (2) A multi-task Group Relative Policy Optimization (GRPO) reinforcement learning stage. Training data is derived from a unified dataset $D_{unified} = \bigcup_{k=1}^{K} D_k$ synthesized from six evaluation benchmarks, covering text-image alignment (SeeTRUE, ImageReward), safety compliance (UnsafeBench), and visual quality (three subsets of AGIN). The model is based on the Qwen3-VL-30B-A3B-Instruct backbone.

### Key Designs

1.  **Compound Reward Function**:
    -   **Function**: Simultaneously optimizes output format compliance and judgment accuracy.
    -   **Mechanism**: The total reward is a linear combination of an accuracy reward and a format reward: $R_{total} = (1-\alpha) \cdot R_{Acc} + \alpha \cdot R_{For}$. $R_{Acc}$ is 1.0 for correct predictions and 0.0 otherwise; $R_{For}$ is 1.0 if the output follows the "reasoning-first" structure and 0.0 otherwise. $\alpha$ controls the balance between the two.
    -   **Design Motivation**: The format reward ensures the model generates a reasoning process (akin to CoT) before giving a judgment. This improves judgment quality and enhances interpretability; the accuracy reward directly optimizes for correctness.

2.  **GRPO Multi-task Training Objective**:
    -   **Function**: Jointly optimizes a global policy across multiple evaluation tasks without requiring a separate value function.
    -   **Mechanism**: For each prompt in the unified dataset, $G=20$ outputs are sampled from the current judge, and policy optimization is performed using inner-group relative rewards. The objective is to maximize the expected reward over the unified dataset: $\theta^* = \arg\max_\theta \mathbb{E}_{(x,p,y) \sim D_{unified}} [R_{total}(M_\theta(x))]$. GRPO replaces explicit value functions with group-averaged rewards, simplifying training.
    -   **Design Motivation**: Unlike RL on a single task, multi-task RL allows the model to capture shared standards and latent correlations across evaluation domains, preventing overfitting to specific prompt templates.

3.  **Reasoning-First Paradigm**:
    -   **Function**: Forces the judge to explicitly generate a reasoning process before the final judgment.
    -   **Mechanism**: The RL prompt requires the model to output a reasoning trace followed by the final prediction. The format reward $R_{For}$ enforces this structure. By unfolding analytical logic in the reasoning trace, the model more accurately approximates evaluation logic aligned with human preferences.
    -   **Design Motivation**: SFT tends to mimic statistical mappings rather than learning reasoning, leading to poor generalization. A reasoning-first approach forces the model to "think" before "judging," a capability crucial for handling unseen task formats.

### Loss & Training

The SFT stage uses the AdamW optimizer with a learning rate of $1.0 \times 10^{-5}$, batch size 256, and full-parameter fine-tuning. The RL stage uses GRPO with a rollout size $G=20$, global batch size 256, and rollout batch size 512. Training stops for all models at the plateau of validation performance to select the best checkpoint.

## Key Experimental Results

### Main Results

**Macro-F1 Comparison across Six Evaluation Tasks**

| Method | AGIN-Nat | AGIN-Tech | AGIN-Rat | SeeTRUE | ImageReward | UnsafeBench |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Off-the-shelf | 67.99 | 63.24 | 64.77 | 80.01 | 55.07 | 72.78 |
| SFT-Single | 78.64 | 77.04 | 78.08 | 80.41 | 64.95 | 90.28 |
| SFT-Unified | 81.75 | 81.22 | 81.31 | 82.32 | 63.34 | 89.49 |
| RL-Single | 80.50 | 80.77 | 82.71 | 83.41 | 65.07 | 86.92 |
| **MT-RL-Judge** | 81.63 | **81.37** | 81.58 | **83.67** | 64.97 | 85.22 |

### Ablation Study

**MJ-Bench OOD Generalization (Unseen Pairwise Format)**

| Method | Image-text Alignment | Safety Judge |
| :--- | :--- | :--- |
| Off-the-shelf | 59.41 | 73.07 |
| SFT-Unified | 55.82 | 49.40 |
| **MT-RL-Judge** | **60.59** | **82.23** |

### Key Findings

-   RL consistently outperforms SFT: RL-Single exceeds SFT-Single in 5 out of 6 tasks, particularly in reasoning-intensive tasks (SeeTRUE +3.0, AGIN-Rat +4.63).
-   Unified training provides gains: SFT-Unified outperforms SFT-Single in most tasks, suggesting that multi-task exposure enables the model to learn cross-domain shared evaluation standards.
-   SFT severely overfits format: SFT-Unified collapses to 49.40% on MJ-Bench Safety, much lower than the zero-shot baseline of 73.07%—despite seeing safety tasks during training, it fails completely when the input changes from single-image to dual-image.
-   MT-RL-Judge exhibits significant generalization: On completely unseen pairwise formats, Safety reaches 82.23% (32.83 percentage points higher than SFT-Unified), verifying that RL-driven reasoning capability transfers to new formats.

## Highlights & Insights

-   The catastrophic degradation of SFT-Unified on MJ-Bench is the paper's most compelling evidence—clearly demonstrating the fundamental flaw of SFT memorizing formats instead of learning logic.
-   The "1+1>2" effect of multi-task RL: Joint training does not sacrifice single-task performance; instead, it improves overall performance through cross-task knowledge sharing.
-   The reasoning-first design aligns with recent trends in reasoning models (e.g., DeepSeek-R1, QwQ), but its application in evaluation scenarios is a first.
-   Exquisite experimental design: The four-way comparison (SFT-Single vs SFT-Unified vs RL-Single vs MT-RL-Judge) clearly isolates the respective contributions of unified training and RL.

## Limitations & Future Work

-   Experiments are primarily conducted on binary classification evaluation tasks; multi-level scoring or open-ended evaluations are not addressed.
-   The base model is Qwen3-VL-30B-A3B (MoE); effects on other architectures have not been verified.
-   The coverage of six training tasks is still limited; adding more evaluation dimensions might yield stronger generalization.
-   Quality and diversity of reasoning traces were not analyzed; whether the reasoning process truly reflects evaluation logic requires further validation.

## Related Work & Insights

-   **vs SFT-based Judges (MLLM-as-a-Judge)**: SFT Judges overfit prompt templates; MT-RL-Judge internalizes evaluation logic through RL-based reasoning.
-   **vs Mr. Judge / Flex-Judge**: Existing works mostly involve single-task training; MT-RL-Judge is the first to implement a unified multi-task RL judge.
-   **vs Self-Consistency Methods**: Methods that sample multiple candidates and vote are computationally expensive and may include similar errors; MT-RL-Judge improves judgment accuracy through reasoning quality.

## Rating

-   Novelty: ⭐⭐⭐⭐ First unified multi-task RL MLLM-as-a-Judge framework; profound insights regarding SFT's format overfitting.
-   Experimental Thoroughness: ⭐⭐⭐ Sufficient benchmarks across six tasks, but lacks more extensive OOD generalization tests and reasoning quality analysis.
-   Writing Quality: ⭐⭐⭐⭐ Clear motivation and sophisticated experimental comparison design, though some methodological details are brief.
-   Value: ⭐⭐⭐⭐ Provides a unified and generalizable solution for industrial-grade multimodal evaluation; MJ-Bench generalization results are convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)
- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)
- [\[ICML 2026\] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](../../ICML2026/llm_evaluation/agent_world_model_infinity_synthetic_environments_for_agentic_reinforcement_lear.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ACL 2026\] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases](reasoning_model_is_superior_llm-judge_yet_suffers_from_biases.md)

</div>

<!-- RELATED:END -->
