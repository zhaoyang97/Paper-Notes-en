---
title: >-
  [Paper Note] LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation
description: >-
  [ACL 2025 (XLLM Workshop)][Model Compression][Structured Reasoning] This paper proposes the *Less is More* framework. Under the extreme low-resource conditions of only 24 annotated samples, it distills high-quality structured reasoning data to fine-tune a LLaMA3-8B multi-agent system via three stages: reverse-prompt induction, GPT-4o-enhanced retrieval-augmented reasoning synthesis, and dual-stage reward-guided filtering. It achieved third place in the XLLM@ACL2025 Shared Tas…
tags:
  - "ACL 2025 (XLLM Workshop)"
  - "Model Compression"
  - "Structured Reasoning"
  - "Multi-agent Framework"
  - "Knowledge Distillation"
  - "Low-resource Learning"
  - "Quality-guided Filtering"
date: 2026-05-08
content_hash: 91711527987fa1df
---

# LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation

**Conference**: ACL 2025 (XLLM Workshop)  
**arXiv**: [2504.16408](https://arxiv.org/abs/2504.16408)  
**Code**: [GitHub](https://github.com/JhCircle/Less-is-More)  
**Area**: Model Compression / LLM Reasoning  
**Keywords**: Structured Reasoning, Multi-agent Framework, Knowledge Distillation, Low-resource Learning, Quality-guided Filtering

## TL;DR

This paper proposes the *Less is More* framework. Under the extreme low-resource conditions of only 24 annotated samples, it distills high-quality structured reasoning data to fine-tune a LLaMA3-8B multi-agent system via three stages: reverse-prompt induction, GPT-4o-enhanced retrieval-augmented reasoning synthesis, and dual-stage reward-guided filtering. It achieved third place in the XLLM@ACL2025 Shared Task.

## Background & Motivation

**Background**: Structured reasoning tasks—such as decomposing questions into logical constraints and verifying each step of a reasoning chain—require LLMs to generate explainable, step-by-step reasoning processes. The XLLM@ACL2025 Shared Task-III targets this challenge, requiring participants to learn from only 24 annotated samples across four sub-tasks: Question Parsing (QP), CoT Parsing (CP), CoT Statement (CS), and Verification (CV).

**Limitations of Prior Work**: (1) Annotated data is extremely scarce, preventing direct fine-tuning of high-capacity models; (2) Maintaining step-level consistency and logical coherence across multiple reasoning modules is highly challenging. Existing CoT prompting methods usually rely on large-scale instruction tuning or heuristic prompts, which perform poorly in scenarios where annotations are scarce and structured granularity is required.

**Key Challenge**: The tension between data quality and quantity—it is impossible to obtain a large amount of high-quality annotations in low-resource environments, but the model needs sufficient and high-quality training signals to learn structured reasoning.

**Goal**: Starting from minimal seed data, to generate high-quality training data through a controllable data distillation pipeline, enabling small models to perform structured reasoning.

**Key Insight**: The authors hypothesize that "less is more" (quality over quantity)—through quality-guided filtering, truly useful training signals can be extracted from a large amount of synthetic data.

**Core Idea**: Automatically induce task prompts with reverse thinking, generate a large volume of reasoning data using GPT-4o, and guarantee data quality via a dual-stage (structural + reward) filtering system. Ultimately, data quality, rather than quantity, drives the improvement in model performance.

## Method

### Overall Architecture

The input is a natural language logical reasoning problem (from the LogiQA dataset), and the output is a structured reasoning process including question parsing, CoT decomposition, and step-by-step verification. The entire pipeline is divided into a training phase (prompt induction $\rightarrow$ data synthesis $\rightarrow$ quality filtering $\rightarrow$ model fine-tuning) and an inference phase (cascading processing by three specialized agents).

### Key Designs

1. **Reverse-Prompt Induction**:

    - **Function**: Automatically induces optimal task-specific prompts from a small set of seed samples.
    - **Mechanism**: Inspired by "Reverse-of-Thought" (RoT), given the seed data $\{(x_i, y_i)\}$, LLMs are prompted with a reverse prompt instruction $\mathcal{P}_{\text{reverse}}$ to infer "what instruction could generate these outputs." A candidate prompt set $\Pi$ is generated, and the optimal prompt $\pi_t^* = \arg\max_{\pi}[S_{\text{gen}}(\pi) + S_{\text{pref}}(\pi)]$ is selected based on a joint generation score $S_{\text{gen}}$ and preference score $S_{\text{pref}}$.
    - **Design Motivation**: To avoid the subjectivity and inefficiency of manual prompt engineering by automating the discovery of optimal instruction templates from seed data.

2. **Retrieval-Augmented Reasoning Synthesis**:

    - **Function**: Utilizes GPT-4o to generate structured reasoning annotations for unlabeled LogiQA data.
    - **Mechanism**: For each unlabeled question $x$, a pre-trained encoder computes the embedding $\mathbf{h}_x$, and $k$ semantically nearest neighbors are retrieved from the seed set as few-shot demonstrations. Two prompts, QP and UCoT, are formulated to call GPT-4o for generating structured JSON outputs containing CoT steps, textual evidence, and verification labels.
    - **Design Motivation**: Leveraging powerful closed-source models (GPT-4o) to generate supervision signals, and retrieving the most similar demonstrations to ensure contextual consistency.

3. **Dual-Stage Reward-Based Filtering**:

    - **Function**: Filters high-quality samples from synthetic data for downstream fine-tuning.
    - **Mechanism**: The first stage is structural filtering, which removes outputs with improper formats (e.g., JSON parsing failure, reasoning steps fewer than two). The second stage uses a LLaMA3-based reward model to score each data item under both few-shot and zero-shot prompts: $s_{\text{avg}} = \frac{1}{2}(s_{\text{few}} + s_{\text{zero}})$, retaining only samples with $\mathcal{S}(x) > 0$. This dual-prompt scoring strategy balances contextual coherence and general quality.
    - **Design Motivation**: Synthetic data inevitably contains noise, requiring strict filtering to guarantee fine-tuning effectiveness.

### Loss & Training

The three sub-task models are independently fine-tuned from Meta-Llama-3-8B-Instruct using LoRA+ (rank=16, $\alpha=32$, lorap_lr_ratio=16). They are trained for 5 epochs via the ms-swift framework with a learning rate of $2 \times 10^{-5}$, batch size of 4, gradient accumulation of 4 steps, warmup ratio of 0.03, on two NVIDIA A100-80G GPUs.

## Key Experimental Results

### Main Results

| Filtering Strategy | Ques._F1 | Stmt._F1 | Evid._F1 | Reason._F1 |
|--------------------|----------|----------|----------|------------|
| Structural Filtering Only | 56.87 | 36.72 | 10.80 | 5.20 |
| Zero-shot Reward Filtering | 62.76 | 38.05 | 12.79 | 7.15 |
| Few-shot Reward Filtering | 65.89 | 38.26 | 14.45 | 7.70 |
| Average Reward Filtering | **66.71** | **39.21** | **14.92** | **8.98** |

### Ablation Study (Training Data Volume)

| Configuration | QP Count | CP Count | CV Count | Description |
|---------------|----------|----------|----------|-------------|
| Original LogiQA | 7,376 | - | - | Unfiltered |
| Structural Filtering | 1,940 | 1,940 | 13,818 | Format-compliant |
| Zero-shot Filtering | 1,309 | 1,309 | 9,434 | Quality Filtering |
| Few-shot Filtering | 1,377 | 1,377 | 9,858 | Quality Filtering |
| Average Filtering | 1,346 | 1,346 | 9,688 | Best Balance |

### Key Findings

- The average reward filtering strategy consistently achieves the best performance across all metrics, with Reasoning F1 improving from 5.20 to 8.98 (+72.7%).
- Even though Question F1 does not directly participate in the reward calculation, it improves from 56.87 to 66.71—indicating that high-quality intermediate supervision signals can indirectly enhance the model's global structural understanding.
- Performance is actually better after reducing the data volume from 7,376 to ~1,346, directly proving the core thesis of "Less is More".
- Few-shot filtering is slightly superior to zero-shot filtering, suggesting that contextual information helps in more accurate quality assessment.

## Highlights & Insights

- **Empirical evidence of "quality-driven" data**: Utilizing only ~18% of the filtered data significantly outperforms using the full structurally filtered data, providing valuable guidance for LLM fine-tuning under low-resource scenarios.
- **Reverse-prompt induction** serves as an elegant cold-start strategy—backward-inducing prompts from target outputs, avoiding the burden of manual prompt engineering. This technique can be transferred to any task requiring learning from few demonstrations.
- **Unexpected cross-module transfer**: Reward signals are only applied to the CoT module, yet the QP module also achieves a significant boost. This implies an intrinsic association between structured reasoning sub-tasks.

## Limitations & Future Work

- Scoring third place in the competition instead of first indicates room for improvement, especially regarding step-level verification (Reason_F1 is only 8.98).
- Reliance on GPT-4o for data synthesis incurs high costs and a closed-source dependency.
- Evaluation is only conducted on the LogiQA dataset; generalization to other structured reasoning tasks (e.g., mathematical proof, code debugging) remains to be verified.
- The choice of reward model greatly impacts filtering quality, but the paper does not delve deeply into comparing different reward models.
- Future work could explore iterative distillation (using the model trained on filtered data to generate synthetic data again) to further improve quality.

## Related Work & Insights

- **vs LIMA (Less is More for Alignment)**: Both share the "quality > quantity" philosophy, but this paper targets structured reasoning rather than general alignment, introducing automated quality filtering mechanisms.
- **vs Standard CoT Prompting**: Standard CoT heavily relies on manual design, whereas this paper achieves fully automatic structured CoT generation via reverse induction + RA-ICL.
- **vs Knowledge Distillation Method**: Traditional distillation focuses on soft labels, while this paper distills explicit reasoning traces and ensures purity through quality filtering.

## Rating

- Novelty: ⭐⭐⭐⭐ Each component (reverse-prompting, RA-ICL, reward filtering) is not entirely new, but the combination is highly effective in low-resource structured reasoning scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations sufficiently demonstrate the effect of each filtering strategy, but evaluations are limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear methodology, intuitive framework diagrams, and standardized formulas.
- Value: ⭐⭐⭐⭐ High reference value as a shared task solution; the conclusion "quality > quantity" is inspiring for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](../../NeurIPS2025/model_compression/less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[ICML 2026\] Critique-Guided Distillation for Robust Reasoning via Refinement](../../ICML2026/model_compression/critique-guided_distillation_for_robust_reasoning_via_refinement.md)
- [\[CVPR 2025\] Less is More: Efficient Model Merging with Binary Task Switch](../../CVPR2025/model_compression/less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [\[ICML 2026\] Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning](../../ICML2026/model_compression/beyond_tokens_enhancing_rtl_quality_estimation_via_structural_graph_learning.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)

</div>

<!-- RELATED:END -->
