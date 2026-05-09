---
title: >-
  [Paper Note] DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning
description: >-
  [ICCV 2025][Multimodal VLM][Compositional Visual Reasoning] This paper proposes the DWIM framework, which employs a discrepancy-aware workflow generation strategy to curate high-quality training data and an instruct-masking fine-tuning strategy to clone only effective actions, endowing LLMs with tool-aware capability for compositional visual reasoning and achieving state-of-the-art results on multiple VR benchmarks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Compositional Visual Reasoning
  - Tool-awareness
  - Workflow Generation
  - Instruct-Masking Fine-tuning
  - LLM
date: 2026-05-08
content_hash: bc1a9cd02c6bf951
---

# DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning

**Conference**: ICCV 2025
**arXiv**: [2503.19263](https://arxiv.org/abs/2503.19263)
**Code**: None
**Area**: Multimodal VLM / Visual Reasoning
**Keywords**: Compositional Visual Reasoning, Tool-awareness, Workflow Generation, Instruct-Masking Fine-tuning, LLM

## TL;DR

This paper proposes the DWIM framework, which employs a discrepancy-aware workflow generation strategy to curate high-quality training data and an instruct-masking fine-tuning strategy to clone only effective actions, endowing LLMs with tool-aware capability for compositional visual reasoning and achieving state-of-the-art results on multiple VR benchmarks.

## Background & Motivation

**Background**: Visual Reasoning (VR) is a key task for enabling human-like visual understanding in models. Compositional VR methods have become mainstream in recent years, leveraging the reasoning capabilities of LLMs to decompose complex visual problems into multiple sub-steps and invoke external tools (e.g., object detectors, OCR, depth estimators) for stepwise solving, outperforming end-to-end approaches.

**Limitations of Prior Work**: Frozen LLMs lack awareness of visual reasoning tools—they do not know what tools can do, when to invoke which tool, or whether the results returned by tools are reliable. This leads to workflows containing numerous invalid or erroneous tool calls, constituting a significant performance bottleneck.

**Key Challenge**: Although fine-tuning can teach LLMs tool usage in other domains (e.g., code generation, mathematical reasoning), visual reasoning presents a triple challenge: (1) limited training data—high-quality visual reasoning workflow data is scarce; (2) imperfect tools—external tools themselves introduce errors, reducing data collection efficiency; and (3) noisy workflows—collected workflows contain numerous erroneous steps, and direct fine-tuning causes the model to learn bad habits.

**Goal**: To endow LLMs with tool-aware capability, enabling correct tool usage in visual reasoning while addressing poor training data quality and high workflow noise.

**Key Insight**: The authors observe that not every step in a workflow is worth learning—some tool calls are correct and effective, while others are erroneous. The key lies in distinguishing "good actions" from "bad actions" within a workflow and learning only the effective portions.

**Core Idea**: A two-stage solution is proposed—first, a discrepancy-aware strategy generates high-quality training workflows (filtering out unreliable ones); then, instruct-masking fine-tuning ensures the model clones only effective actions (skipping erroneous steps in the workflow).

## Method

### Overall Architecture

DWIM consists of two core components: (1) Discrepancy-aware Workflow Generation, responsible for selecting high-quality workflows during data collection for training; and (2) Instruct-Masking Tuning, responsible for restricting the model to learn only effective steps within a workflow during training. The input is a visual question (image + question), and the output is a reasoning workflow containing tool calls that ultimately yields an answer.

### Key Designs

1. **Discrepancy-aware Workflow Generation**:

    - Function: Selects high-quality training samples from automatically collected workflows.
    - Mechanism: Exploits the discrepancy between tool execution results and expectations to assess workflow quality. Specifically, for each candidate workflow, the tool calls are executed and intermediate results are checked for plausibility—if the tool outputs are consistent with the question context (low discrepancy), the workflow is deemed viable; otherwise, it is filtered out. This evaluation avoids the pitfall of relying solely on final answer correctness to judge workflow quality (the final answer may be coincidentally correct despite erroneous intermediate steps).
    - Design Motivation: Workflows generated directly by LLMs are too noisy, and filtering solely by final answer correctness retains workflows that arrive at correct answers through incorrect reasoning. By evaluating the discrepancy at each tool-call step, training data can be curated with greater precision.

2. **Instruct-Masking Fine-tuning**:

    - Function: Restricts the model during fine-tuning to learn only effective actions in the workflow, ignoring invalid or erroneous steps.
    - Mechanism: Each step in the workflow is labeled—effective tool-call steps retain their training loss, while invalid steps are masked during loss computation. This prevents the model from being forced to imitate erroneous tool usage patterns, cloning only those actions that are genuinely effective.
    - Design Motivation: Even after discrepancy-aware filtering, training workflows may still contain suboptimal individual steps (since an overall viable workflow may contain locally suboptimal steps). The instruct-masking mechanism provides fine-grained step-level control, ensuring the model learns exclusively correct tool usage patterns.

3. **Tool-aware Reasoning Mechanism**:

    - Function: Enables the fine-tuned LLM to generate reasonable tool-calling workflows at inference time.
    - Mechanism: Through the above two-stage training, the LLM learns when to invoke which tool, how to parse tool-returned results, and how to continue reasoning based on intermediate outputs. At inference time, the model autoregressively generates a step sequence containing tool calls, while an external executor runs the tools and feeds results back to the model.
    - Design Motivation: Existing methods rely on the zero-shot capability or few-shot prompting of frozen LLMs, resulting in limited tool usage capability. Fine-tuning on high-quality data endows the model with internalized tool-aware capability.

### Loss & Training

Training adopts the standard autoregressive language modeling loss, with invalid-step tokens masked via instruct-masking, computing cross-entropy loss only over tokens corresponding to effective actions. This ensures that gradient updates are guided exclusively by correct tool usage patterns.

## Key Experimental Results

### Main Results

DWIM is evaluated on multiple visual reasoning benchmarks spanning diverse task types:

| Dataset | Task | Ours | Prev. SOTA | Gain |
|---------|------|------|------------|------|
| GQA | Visual QA | SOTA | — | Significant |
| VQAv2 | Visual QA | SOTA | — | Significant |
| NLVR2 | Visual Reasoning | SOTA | — | Significant |
| RefCOCO | Referring Expression | SOTA | — | Significant |

Experiments demonstrate that DWIM achieves optimal performance across various VR tasks, with particularly pronounced advantages on complex reasoning tasks requiring multi-step tool invocation.

### Ablation Study

| Configuration | Performance Change | Note |
|---------------|--------------------|------|
| Full DWIM | Best | Complete model |
| w/o Discrepancy-aware | Degraded | No workflow quality filtering; noisy training data |
| w/o Instruct-Masking | Degraded | No masking of invalid steps; model learns erroneous patterns |
| Final-answer-only filtering | Suboptimal | Retains workflows with coincidentally correct answers |

### Key Findings

- Both components contribute significantly: discrepancy-aware workflow generation addresses data quality, and instruct-masking fine-tuning addresses residual noise—neither is dispensable.
- DWIM's advantage is more pronounced on complex tasks requiring multi-step reasoning, demonstrating that tool-aware capability is critical for complex reasoning.
- Compared to prompt-engineering-based methods, the fine-tuning approach yields superior accuracy and consistency in tool usage.

## Highlights & Insights

- **Granularity of discrepancy-aware evaluation**: Rather than simply checking whether the final answer is correct, the method evaluates whether each intermediate tool call is reasonable. This idea is transferable to other multi-step reasoning scenarios (e.g., code generation, mathematical proof) for selecting high-quality training trajectories.
- **Elegant design of instruct-masking**: Introducing token-level selective learning within sequence-level supervised learning preserves the contextual information of the complete workflow while avoiding learning from erroneous steps—a general technique for "learning from noisy data."
- **Automated data pipeline**: The entire process requires no human annotation—workflows are automatically generated, evaluated, and masked, yielding strong scalability.

## Limitations & Future Work

- Performance is bounded by the quality and coverage of external tools—if the available tool set is insufficient or tool error rates are high, the method's effectiveness degrades.
- Discrepancy-aware evaluation requires executing tool calls, increasing the computational overhead of data preparation.
- Current evaluation is primarily on static benchmarks; performance in dynamic, open-world visual reasoning scenarios remains to be validated.
- Future work could explore enabling the model to adaptively discover and integrate new tools rather than relying on a fixed tool set.

## Related Work & Insights

- **vs. VisProg/ViperGPT**: These classical compositional VR methods use frozen LLMs to generate programs/workflows for tool invocation without fine-tuning the LLM. DWIM endows the LLM with internalized tool-aware capability through fine-tuning, yielding better performance.
- **vs. Chameleon**: Chameleon also performs tool-augmented LLM reasoning, but its tool selection strategy is rule-based. DWIM learns tool usage patterns in a data-driven manner, offering greater flexibility.
- **vs. InstructBLIP/LLaVA**: These end-to-end multimodal models do not use external tools and underperform compositional methods on tasks requiring precise perception. DWIM demonstrates that the "LLM reasoning + external tools" paradigm still has significant headroom.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of discrepancy-awareness and instruct-masking is novel, though the overall framework falls within the compositional VR paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear; method description is well-structured.
- Value: ⭐⭐⭐⭐ Provides a practical solution to data quality and training strategy challenges in compositional visual reasoning.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](../../NeurIPS2025/multimodal_vlm/learning_to_instruct_for_visual_instruction_tuning.md)
- [\[ICCV 2025\] orderchain towards general instruct-tuning for stimulating the ordinal understan](orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](attention_to_the_burstiness_in_visual_prompt_tuning.md)

<!-- RELATED:END -->
