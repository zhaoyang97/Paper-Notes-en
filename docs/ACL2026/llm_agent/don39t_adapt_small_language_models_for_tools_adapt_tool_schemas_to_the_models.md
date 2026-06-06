---
title: >-
  [Paper Note] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models
description: >-
  [ACL 2026][LLM Agent][Small Language Models] This paper proposes PA-Tool, a training-free tool schema optimization method. It utilizes the "peakedness" signal, borrowed from data contamination detection…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Small Language Models"
  - "Tool Use"
  - "Schema Alignment"
  - "Pre-training Knowledge"
  - "Training-free"
date: 2026-05-08
content_hash: 269ed12a68f070cd
---

# Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models

**Conference**: ACL 2026  
**arXiv**: [2510.07248](https://arxiv.org/abs/2510.07248)  
**Code**: [GitHub](https://github.com/holi-lab/PA-Tool)  
**Area**: LLM/NLP  
**Keywords**: Small Language Models, Tool Use, Schema Alignment, Pre-training Knowledge, Training-free

## TL;DR

This paper proposes PA-Tool, a training-free tool schema optimization method. It utilizes the "peakedness" signal, borrowed from data contamination detection, to identify naming patterns familiar to the model from pre-training. By renaming tool components to align with the internalized knowledge of small language models (SLMs), it achieves up to a 17% improvement on MetaTool and RoTBench, while reducing schema misalignment errors by 80%.

## Background & Motivation

**Background**: Tool-augmented language models have become core components of modern AI systems. With the development of multi-agent architectures, there is an increasing demand to deploy SLMs (typically $\leq$ 8B) for sub-tasks, including tool selection (identifying the correct API) and parameter identification (providing correct parameters).

**Limitations of Prior Work**: SLMs perform significantly worse than large models on tool-use tasks. A common failure mode is "schema misalignment": even when the correct tool is provided in the context, models still hallucinate plausible-looking but non-existent tool names. This suggests that the model reverts to naming conventions internalized during pre-training when faced with unfamiliar schemas.

**Key Challenge**: Existing methods either adapt models to arbitrary schemas through training (requiring large amounts of data and potentially causing catastrophic forgetting) or indirectly improve performance through tool documentation or interaction history (without addressing the fundamental mismatch at the schema level). Training methods are costly and non-scalable, while previous training-free methods do not touch the root of naming misalignment.

**Goal**: To propose a training-free method that adjusts tool schemas to match the model's pre-training knowledge, rather than training the model to adapt to the schema.

**Key Insight**: Borrow the concept of "peakedness" from the field of data contamination detection—patterns frequently seen by a model during pre-training lead to highly concentrated output distributions. This signal is used to identify naming patterns "familiar" to the model.

**Core Idea**: Instead of training small models to adapt to unfamiliar tool schemas, the schema is adjusted to align with the model's pre-training knowledge by generating multiple candidate names, calculating peakedness, and selecting the candidate with the highest peakedness to find the model's most familiar naming.

## Method

### Overall Architecture

PA-Tool performs a three-stage renaming process for each component in the tool schema (tool names, parameter names): (1) Candidate Generation—allowing the SLM to generate candidate names multiple times based on the component description; (2) Peakedness Computation—measuring how many similar candidates surround each candidate name; (3) Schema Selection—selecting the candidate with the highest peakedness as the new name. This process is iteratively executed for all components in the schema to build a mapping dictionary from original names to pre-training aligned names.

### Key Designs

1.  **Candidate Generation**:
    *   **Function**: Explores diverse naming patterns the model may have encountered during pre-training.
    *   **Mechanism**: Given a component description $d$, $N$ candidate names $\mathcal{C} = \{s_1, s_2, \ldots, s_N\}$ are sampled at temperature $t \in (0,1]$. Temperature-controlled sampling goes beyond a single greedy path to reveal varied naming patterns learned by the model. Simultaneously, a reference name $s_{\text{ref}}$ is generated using greedy decoding ($t=0$) for tie-breaking.
    *   **Design Motivation**: Greedy decoding only generates a single candidate, limiting the exploration of the schema space (experiments show Greedy can sometimes be inferior to the Base version).

2.  **Peakedness Computation**:
    *   **Function**: Identifies strong memory patterns frequently encountered during model pre-training.
    *   **Mechanism**: For each candidate $s_i$, the peakedness score is calculated as $\phi(s_i) = \sum_{j \neq i} \mathbb{I}(d_{\text{edit}}(s_i, s_j) \leq \tau)$, where $d_{\text{edit}}$ is the character-level edit distance, and the threshold $\tau = \alpha \cdot \ell_{\max}$ ($\ell_{\max}$ is the maximum candidate length, and $\alpha$ controls similarity strictness). High peakedness implies the model produces a highly concentrated distribution for that naming pattern.
    *   **Design Motivation**: Inspired by the CDD data contamination detection method—patterns that appear frequently in training produce concentrated output distributions during multiple samplings. The length-adaptive threshold ensures consistent similarity standards across names of different lengths.

3.  **Schema Selection and Conflict Resolution**:
    *   **Function**: Selects the best pre-training aligned name and handles cross-tool conflicts.
    *   **Mechanism**: The candidate with the highest peakedness is selected: $s^* = \arg\max_{s_i \in \mathcal{C}} \phi(s_i)$. In case of a tie, the candidate with the smallest edit distance to the reference name is chosen: $s^* = \arg\min_{s_i \in \mathcal{C}^*} d_{\text{edit}}(s_i, s_{\text{ref}})$. When different tools produce name conflicts due to similar descriptions, a priority locking mechanism is used to resolve them.
    *   **Design Motivation**: The candidate with the highest peakedness represents the model's most deeply internalized naming convention. The one-time schema mapping can be reused without regeneration during every inference.

### Loss & Training

PA-Tool is completely training-free. It requiring only a one-time schema mapping: using 32 candidates, temperature 0.4, and $\alpha = 0.2$. Temperature 0 is used during inference to ensure reproducibility. The mapping dictionary can be reused as long as the toolset remains unchanged, requiring no model modification, retraining, or risk of catastrophic forgetting.

## Key Experimental Results

### Main Results

**MetaTool Tool Selection (Accuracy %)**

| Model | Method | Similar | Scenario | Reliability | Multi-tool |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-7B | Base | 59.6 | 74.4 | 78.3 | 78.3 |
| Qwen2.5-7B | PA-Tool | 64.1 | 78.4 | **88.2** | 84.9 |
| Llama3.1-8B | Base | 61.5 | 73.9 | 53.5 | 78.7 |
| Llama3.1-8B | PA-Tool | **70.4** | **79.9** | 66.0 | **88.3** |
| Llama3.2-3B | Base | 55.0 | 58.6 | 43.6 | 79.1 |
| Llama3.2-3B | PA-Tool | 65.7 | 67.7 | 60.6 | 80.5 |

**RoTBench Tool Selection and Parameter Identification**

| Model | Method | Single-turn Selection | Single-turn Param | Multi-turn Selection | Multi-turn Param |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Llama3.1-8B | Base | 58.1 | 17.1 | 42.8 | 34.3 |
| Llama3.1-8B | PA-Tool | **68.6** | 18.1 | **48.6** | **35.7** |

### Ablation Study

| Config | Single-turn Selection | Single-turn Param | Description |
| :--- | :--- | :--- | :--- |
| Base | 58.1 | 17.1 | No alignment |
| Tool-only | 62.9 | 14.3 | Tool name alignment only |
| Param-only | 56.2 | 17.1 | Parameter name alignment only |
| Both (PA-Tool) | **68.6** | **18.1** | Joint alignment is best |

**Error Type Analysis (Llama3.1-8B, MetaTool)**

| Error Type | Base | PA-Tool | Gain |
| :--- | :--- | :--- | :--- |
| Schema Misalignment Error | — | — | **-80.0%** |
| Functional Confusion Error | — | — | -24.0% |
| Context Understanding Error | — | — | -18.8% |

### Key Findings

*   PA-Tool shows the largest improvement in the Reliability sub-task (up to 17%), which requires the model to identify when no suitable tool exists.
*   The improvement in the Multi-tool sub-task reaches 9.6% (Llama3.1-8B: 78.7% $\rightarrow$ 88.3%) because schema misalignment accumulates during the selection of multiple tool combinations.
*   The standalone use of PA-Tool can surpass supervised fine-tuning (SFT) models on several sub-tasks, and further improvements are observed when combining both.
*   Peakedness validation experiments confirm that as the number of training epochs increases, the model's peakedness consistently rises (up to +25.8%), supporting the hypothesis that it acts as a familiarity signal.
*   Consistent improvements are also demonstrated on end-to-end benchmarks like API-Bank and $\tau$-Bench.

## Highlights & Insights

*   The reverse thinking is highly inspiring: "Don't make the model adapt to the tool; make the tool adapt to the model"—this approach can be generalized to other model-interface interaction scenarios.
*   Cross-domain transfer from data contamination detection to tool schema optimization: transforming peakedness from "detecting contamination" to "utilizing pre-training knowledge," turning waste into treasure.
*   The one-time mapping design makes deployment costs extremely low, and it is orthogonal to and combinable with methods like fine-tuning, retrieval, and constrained decoding.
*   PA-Tool allows Llama3.1-8B to outperform Claude-Sonnet-4.5 on the Multi-tool sub-task (88.3% vs. 85.1%), proving that schema alignment can bridge the gap between small and large models.

## Limitations & Future Work

*   Performance depends on the model's understanding of component descriptions to generate candidate names; poor description quality may affect results.
*   Currently, only tool names and parameter names are renamed, without considering the alignment of schema structures (e.g., parameter types, nested structures).
*   The effect is smaller on closed-source models (as the schema misalignment problem is inherently less severe in large models).
*   The peakedness signal may be less stable when tool names are very short or very long.

## Related Work & Insights

*   **vs SFT**: SFT requires training data and faces generalization issues (performance on RoTBench actually decreased after adding data), whereas PA-Tool is training-free and complementary to SFT.
*   **vs EasyTool (Description Enhancement)**: EasyTool improves tool descriptions without modifying names; PA-Tool modifies names without changing descriptions. The two are orthogonal and can be combined.
*   **vs Constrained Decoding**: Constrained decoding eliminates formatting errors but does not resolve naming preference mismatches; PA-Tool addresses naming issues at the root.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ Unique reverse thinking; the cross-domain transfer of the peakedness signal is very clever.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with four benchmarks (MetaTool, RoTBench, API-Bank, $\tau$-Bench), error analysis, peakedness validation, component ablation, and combination experiments with SFT/training-free methods.
*   Writing Quality: ⭐⭐⭐⭐ Clear motivation, intuitive method, and in-depth analysis.
*   Value: ⭐⭐⭐⭐⭐ Provides a practical, zero-cost solution to improve SLM tool-use capabilities, with significant value for multi-agent system deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair](polaris_a_gödel_agent_framework_for_small_language_models_through_experience-abs.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)

</div>

<!-- RELATED:END -->
