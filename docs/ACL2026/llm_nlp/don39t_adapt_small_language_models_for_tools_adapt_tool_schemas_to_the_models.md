---
title: >-
  [Paper Note] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models
description: >-
  [ACL 2026][LLM/NLP][Small Language Models] This paper proposes PA-Tool, a training-free tool schema optimization method that leverages a "peakedness" signal borrowed from data contamination detection to identify naming patterns familiar to a model from pretraining. By renaming tool components to align with the internalized knowledge of small language models (SLMs), PA-Tool achieves up to 17% improvement on MetaTool and RoTBench, with an 80% reduction in schema misalignment errors.
tags:
  - ACL 2026
  - LLM/NLP
  - Small Language Models
  - Tool Calling
  - Schema Alignment
  - Pretrained Knowledge
  - Training-free Optimization
date: 2026-05-08
content_hash: bca214b13bc5b2aa
---

# Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models

**Conference**: ACL 2026
**arXiv**: [2510.07248](https://arxiv.org/abs/2510.07248)
**Code**: [GitHub](https://github.com/holi-lab/PA-Tool)
**Area**: LLM/NLP
**Keywords**: Small Language Models, Tool Calling, Schema Alignment, Pretrained Knowledge, Training-free Optimization

## TL;DR

This paper proposes PA-Tool, a training-free tool schema optimization method that leverages a "peakedness" signal borrowed from data contamination detection to identify naming patterns familiar to a model from pretraining. By renaming tool components to align with the internalized knowledge of small language models (SLMs), PA-Tool achieves up to 17% improvement on MetaTool and RoTBench, with an 80% reduction in schema misalignment errors.

## Background & Motivation

**Background**: Tool-augmented language models have become core components of modern AI systems. With the growth of multi-agent architectures, there is increasing demand to deploy small language models (SLMs, typically ≤8B parameters) for subtasks such as tool selection (identifying the correct API) and parameter identification (supplying correct arguments).

**Limitations of Prior Work**: SLMs perform substantially worse than large models on tool-use tasks. A common failure mode is *schema misalignment*: even when the correct tool is provided in context, the model hallucinates plausible-sounding but nonexistent tool names. This suggests that when confronted with unfamiliar schemas, models fall back on naming conventions internalized during pretraining.

**Key Challenge**: Existing approaches either train models to accommodate arbitrary schemas—requiring large amounts of data and risking catastrophic forgetting—or improve tool documentation and interaction history indirectly without addressing the fundamental naming mismatch at the schema level. Training-based methods are costly and unscalable, while training-free methods do not tackle the root cause of naming misalignment.

**Goal**: To propose a training-free method that adapts tool schemas to match a model's pretrained knowledge, rather than training the model to adapt to schemas.

**Key Insight**: The paper borrows the concept of *peakedness* from data contamination detection—patterns seen frequently during pretraining produce highly concentrated output distributions. This signal is used to identify naming patterns that are "familiar" to the model.

**Core Idea**: Rather than training SLMs to adapt to unfamiliar tool schemas, adapt the schemas to align with the model's pretrained knowledge—by generating multiple candidate names, computing peakedness, and selecting the candidate with the highest peakedness as the one most familiar to the model.

## Method

### Overall Architecture

PA-Tool applies a three-stage renaming process to each component (tool names and parameter names) in a tool schema: (1) **Candidate Generation** — the SLM generates multiple candidate names for a component based on its description; (2) **Peakedness Computation** — measuring how many similar candidates cluster around each candidate name; (3) **Schema Selection** — the candidate with the highest peakedness is selected as the new name. This process is applied iteratively to all components in the schema, constructing a mapping dictionary from original names to pretraining-aligned names.

### Key Designs

1. **Candidate Generation**:

    - **Function**: Explores diverse naming patterns the model may have encountered during pretraining.
    - **Mechanism**: Given a component description $d$, $N$ candidate names $\mathcal{C} = \{s_1, s_2, \ldots, s_N\}$ are sampled at temperature $t \in (0,1]$. Temperature-controlled sampling goes beyond a single greedy path to reveal diverse naming patterns learned by the model. A reference name $s_{\text{ref}}$ is also generated via greedy decoding ($t=0$) for tie-breaking.
    - **Design Motivation**: Greedy decoding yields only a single candidate, limiting exploration of the schema space (experiments show that greedy decoding can sometimes underperform the base model).

2. **Peakedness Computation**:

    - **Function**: Identifies strongly memorized patterns that appeared frequently during pretraining.
    - **Mechanism**: For each candidate $s_i$, the peakedness score is computed as $\phi(s_i) = \sum_{j \neq i} \mathbb{I}(d_{\text{edit}}(s_i, s_j) \leq \tau)$, where $d_{\text{edit}}$ denotes character-level edit distance and the threshold $\tau = \alpha \cdot \ell_{\max}$ ($\ell_{\max}$ is the maximum candidate length and $\alpha$ controls similarity strictness). High peakedness indicates that the model generated a highly concentrated distribution around that naming pattern.
    - **Design Motivation**: Inspired by the CDD data contamination detection method—patterns seen frequently during training yield concentrated output distributions across multiple samples. The length-adaptive threshold ensures consistent similarity criteria across names of varying lengths.

3. **Schema Selection and Conflict Resolution**:

    - **Function**: Selects the best pretraining-aligned name and resolves cross-tool naming conflicts.
    - **Mechanism**: The candidate with the highest peakedness is selected: $s^* = \arg\max_{s_i \in \mathcal{C}} \phi(s_i)$. In case of ties, the candidate with the smallest edit distance from the reference name is chosen: $s^* = \arg\min_{s_i \in \mathcal{C}^*} d_{\text{edit}}(s_i, s_{\text{ref}})$. When different tools produce conflicting names due to similar descriptions, a priority locking mechanism is applied.
    - **Design Motivation**: The highest-peakedness candidate represents the naming convention most deeply internalized by the model. The one-time schema mapping can be reused across inference calls without regeneration.

### Loss & Training

PA-Tool is entirely training-free. Only a one-time schema mapping is required, using 32 candidates, a temperature of 0.4, and $\alpha = 0.2$. Greedy decoding (temperature 0) is used during inference to ensure reproducibility. The mapping dictionary can be reused whenever the tool set remains unchanged, with no model modification, retraining, or risk of catastrophic forgetting.

## Key Experimental Results

### Main Results

**MetaTool Tool Selection (Accuracy %)**

| Model | Method | Similar | Scenario | Reliability | Multi-tool |
|------|------|------|------|------|------|
| Qwen2.5-7B | Base | 59.6 | 74.4 | 78.3 | 78.3 |
| Qwen2.5-7B | PA-Tool | 64.1 | 78.4 | **88.2** | 84.9 |
| Llama3.1-8B | Base | 61.5 | 73.9 | 53.5 | 78.7 |
| Llama3.1-8B | PA-Tool | **70.4** | **79.9** | 66.0 | **88.3** |
| Llama3.2-3B | Base | 55.0 | 58.6 | 43.6 | 79.1 |
| Llama3.2-3B | PA-Tool | 65.7 | 67.7 | 60.6 | 80.5 |

**RoTBench Tool Selection and Parameter Identification**

| Model | Method | Single-turn Tool Selection | Single-turn Param. ID | Multi-turn Tool Selection | Multi-turn Param. ID |
|------|------|------|------|------|------|
| Llama3.1-8B | Base | 58.1 | 17.1 | 42.8 | 34.3 |
| Llama3.1-8B | PA-Tool | **68.6** | 18.1 | **48.6** | **35.7** |

### Ablation Study

| Configuration | Single-turn Tool Selection | Single-turn Param. ID | Description |
|------|------|------|------|
| Base | 58.1 | 17.1 | No alignment |
| Tool-only | 62.9 | 14.3 | Tool names aligned only |
| Param-only | 56.2 | 17.1 | Parameter names aligned only |
| Both (PA-Tool) | **68.6** | **18.1** | Joint alignment performs best |

**Error Type Analysis (Llama3.1-8B, MetaTool)**

| Error Type | Base | PA-Tool | Reduction |
|------|------|------|------|
| Schema misalignment errors | — | — | **−80.0%** |
| Functional confusion errors | — | — | −24.0% |
| Contextual understanding errors | — | — | −18.8% |

### Key Findings

- PA-Tool yields the largest gains on the Reliability subtask (up to 17%), which requires the model to recognize when no suitable tool exists.
- The Multi-tool subtask sees gains of up to 9.6% (Llama3.1-8B: 78.7→88.3%), as schema misalignment errors compound when selecting tool combinations.
- PA-Tool alone surpasses supervised fine-tuning models on multiple subtasks, and combining the two yields further improvements.
- Peakedness validation experiments confirm that as training progresses, model peakedness consistently increases (up to +25.8%), supporting its use as a familiarity signal.
- Consistent improvements are also observed on the API-Bank and τ-Bench end-to-end benchmarks.

## Highlights & Insights

- The inversion of the conventional assumption is highly instructive: "Do not adapt the model to the tools; adapt the tools to the model"—this perspective generalizes to other model-interface interaction scenarios.
- The cross-domain transfer from data contamination detection to tool schema optimization is elegant: peakedness is repurposed from "detecting contamination" to "leveraging pretrained knowledge."
- The one-time mapping design makes deployment cost negligible, and the approach is orthogonal to and composable with fine-tuning, retrieval, and constrained decoding methods.
- PA-Tool enables Llama3.1-8B to surpass Claude-Sonnet-4.5 on the Multi-tool subtask (88.3% vs. 85.1%), demonstrating that schema alignment can close the performance gap between large and small models.

## Limitations & Future Work

- The method relies on the model's comprehension of component descriptions to generate candidate names; poor description quality may degrade performance.
- Only tool names and parameter names are renamed; alignment of schema structure (e.g., parameter types, nested structures) is not considered.
- Gains are smaller on closed-source large models, where schema misalignment is inherently less severe.
- The peakedness signal may be less stable for very short or very long tool names.

## Related Work & Insights

- **vs. Supervised Fine-tuning (SFT)**: SFT requires training data and suffers from generalization issues (adding more data can degrade RoTBench performance); PA-Tool is training-free and complementary to SFT.
- **vs. EasyTool (description augmentation)**: EasyTool improves tool descriptions without modifying names; PA-Tool modifies names without altering descriptions. The two are orthogonal and can be combined.
- **vs. Constrained Decoding**: Constrained decoding eliminates formatting errors but does not address naming preference mismatches; PA-Tool resolves naming issues at the source.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The inversion of conventional assumptions is distinctive; the cross-domain transfer of the peakedness signal is particularly clever.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation across four benchmarks (MetaTool, RoTBench, API-Bank, τ-Bench), with error analysis, peakedness validation, component ablations, and combination experiments with SFT and training-free baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, the method is intuitive, and the analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ Provides a zero-cost approach to improving SLM tool-use capabilities, with significant practical value for multi-agent system deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Perception Programs: Unlocking Visual Tool Reasoning in Language Models](../../CVPR2026/llm_nlp/perception_programs_visual_tool_reasoning.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](../../ICLR2026/llm_nlp/predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](../../NeurIPS2025/llm_nlp/nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)

</div>

<!-- RELATED:END -->
