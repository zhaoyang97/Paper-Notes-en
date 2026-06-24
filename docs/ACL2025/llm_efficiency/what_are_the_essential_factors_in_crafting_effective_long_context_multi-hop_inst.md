---
title: >-
  [Paper Note] What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices
description: >-
  [ACL 2025][LLM Efficiency][Long Context] A Multi-agent Interactive Multi-hop Generation (MIMG) framework is proposed to systematically synthesize high-quality long-context multi-hop instruction data through four modules: quality validation, single-hop question generation, multi-question sampling, and multi-hop merging. The trained models achieve an average improvement of 7.54%, even surpassing larger-scale human-annotated datasets.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Long Context"
  - "Multi-hop Reasoning"
  - "Instruction Tuning"
  - "Data Synthesis"
  - "Multi-Agent"
date: 2026-05-08
content_hash: 887c9e434fc999d5
---

# What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices

**Conference**: ACL 2025  
**arXiv**: [2409.01893](https://arxiv.org/abs/2409.01893)  
**Code**: [WowCZ/LongMIT](https://github.com/WowCZ/LongMIT)  
**Area**: LLM Efficiency  
**Keywords**: Long Context, Multi-hop Reasoning, Instruction Tuning, Data Synthesis, Multi-Agent

## TL;DR

A Multi-agent Interactive Multi-hop Generation (MIMG) framework is proposed to systematically synthesize high-quality long-context multi-hop instruction data through four modules: quality validation, single-hop question generation, multi-question sampling, and multi-hop merging. The trained models achieve an average improvement of 7.54%, even surpassing larger-scale human-annotated datasets.

## Background & Motivation

**Background**: Long-context LLMs show outstanding performance in tasks such as information extraction, QA, and planning. However, merely expanding the context window is insufficient for effectively utilizing long contexts; high-quality instruction tuning data is required to optimize long-context capabilities.

**Limitations of Prior Work**: Existing Self-Instruct methods generate poor-quality long-context data—taking Qwen2-72B as an example, multi-hop samples account for less than 35%, over 40% are of low quality, and more than 45% exhibit semantic repetition.

**Key Challenge**: The cost of human annotation for long-context instruction data is extremely high, yet automatic synthesis methods struggle to balance multi-hop nature, quality, and diversity simultaneously.

**Goal**: Systematically investigate "what are the essential factors in crafting effective long-context multi-hop instruction datasets"—covering validation, generation, sampling, and merging strategies.

**Key Insight**: Leveraging the multi-agent collaboration paradigm, the data synthesis workflow is decomposed into four specialized Agent modules, with the optimal strategy for each module determined through extensive comparative experiments.

**Core Idea**: Use a multi-agent pipeline to progressively generate high-quality multi-hop instruction data—first generating single-hop questions, then sampling and combining them, and finally merging them into multi-hop questions, with a quality validation Agent overseeing the entire process.

## Method

### Overall Architecture

The MIMG framework comprises four core modules: Quality Validation Agent (QVA), Single-hop Question Generation Agent (SQGA), Multi-question Sampling strategy (MQS), and Multi-hop Question Merging Agent (MQMA). QVA globally supervises data quality at each stage, SQGA generates single-hop QA from a single document, MQS samples semantically related questions across documents, and MQMA merges single-hop questions into coherent multi-hop questions.

### Key Designs

**Module 1: Quality Validation Agent (QVA)**

- **Function**: Evaluates and filters low-quality samples throughout the entire data synthesis pipeline.
- **Mechanism**: Employs a scoring strategy rather than classification, filtering high-quality data via a threshold. The validation conditions include multi-dimensional scoring criteria, auxiliary contextual details, and reasoning rationales.
- **Design Motivation**: Experiments reveal that scoring strategies significantly outperform binary classification in long-context scenarios (yielding higher kappa and precision) and mitigate long-context bias. Incorporating scoring dimensions and rationales further enhances robustness.

**Module 2: Single-hop Question Generation Agent (SQGA)**

- **Function**: Generates multiple high-quality single-hop QA pairs from a single document.
- **Mechanism**: Adopts a "question-first, answer-second" generation sequence instead of generating the entire QA pair in one step. It explores different LLM scales as generation backbones.
- **Design Motivation**: Step-by-step generation significantly improves data quality, particularly for open-source LLMs, which offer high cost-effectiveness for single-hop generation.

**Module 3: Multi-question Sampling Strategy (MQS)**

- **Function**: Samples semantically related questions from single-hop questions of multiple documents, preparing for multi-hop merging.
- **Mechanism**: Constructs a question-document semantic correlation matrix using BGE embedding similarity, sampling based on question similarity (rather than document similarity); both intra-document and inter-document sampling are supported.
- **Design Motivation**: Embedding similarity outperforms BM25 and LDA. Question-based sampling is significantly superior to document-based sampling, as questions provide more precise contextual clues.

**Module 4: Multi-hop Question Merging Agent (MQMA)**

- **Function**: Merges multiple sampled single-hop questions into coherent multi-hop questions.
- **Mechanism**: Performs merging using only QA pairs (without needing the original documents), which open-source LLMs can handle competently; no rationale is used to assist the merging.
- **Design Motivation**: Experiments indicate that including documents does not consistently improve performance and increases token consumption. Utilizing rationales during the merging phase instead leads to model misunderstanding of the reasoning chain.

### Loss & Training

Standard instruction-tuning is conducted based on the synthesized LongMIT dataset, with effectiveness validated across multiple LLMs (InternLM2-1.8B/7B, LLaMA3-8B, etc.).

## Key Experimental Results

### Main Results

QA accuracy evaluated by GPT-4o on LongBench:

| Model | Dataset | NarrativeQA | 2WikiMQA | DuReader | HotpotQA | MultifieldQA-en | MultifieldQA-zh | MuSiQue | Qasper | AVG |
|---|---|---|---|---|---|---|---|---|---|---|
| InternLM2-1.8B | +LongAlign | 25.00 | 33.00 | 25.00 | 49.50 | 76.00 | 67.50 | 24.50 | 44.00 | 43.06 |
| InternLM2-1.8B | **+LongMIT** | **26.00** | **35.50** | **60.00** | **56.00** | **75.33** | **75.50** | **29.00** | **47.50** | **50.60** |
| LLaMA3-8B | +LongAlign | 29.00 | 44.50 | 56.50 | 56.50 | 79.33 | 80.50 | 21.50 | 55.50 | 52.92 |
| LLaMA3-8B | **+LongMIT** | **36.50** | **67.50** | **74.00** | **71.00** | **87.33** | **84.50** | **39.50** | **54.00** | **64.29** |
| InternLM2-7B | +LongAlign | 45.00 | 40.00 | 60.00 | 65.50 | 74.67 | 86.00 | 34.00 | 56.50 | 57.71 |
| InternLM2-7B | **+LongMIT** | **46.50** | **57.00** | **74.00** | **73.00** | **91.33** | **91.00** | **45.00** | **62.00** | **67.48** |

### Ablation Study

Data quality comparison across MIMG modules (human evaluation based on 200 samples generated by Qwen2-72B):

| Strategy | Multi-hop Ratio (%) | High Quality Ratio (%) | Non-repetitive Ratio (%) |
|---|---|---|---|
| Original Self-Instruct | <35 | ~60 | ~55 |
| MIMG (Full) | >85 | >85 | >85 |

### Key Findings

- Models trained on LongMIT show an average improvement of **7.54%**, with particularly significant gains on multi-hop benchmarks (2WikiMQA, MuSiQue, HotpotQA).
- Synthesized high-quality data even outperforms larger-scale human-annotated datasets (e.g., ChatQA2, LongAlign).
- The scoring validation strategy achieves near-perfect precision in long-context scenarios, but a lower kappa—indicating that LLMs are excellent filters rather than good annotators.
- Increasing the number of multi-hops (hop count) correlates positively with model performance.
- The MIMG framework only increases input tokens by about 3k (2.5x) and output tokens by less than 0.5k, yet yields a nearly 4-fold improvement in quality.

## Highlights & Insights

- **Highly Systematic**: Conducts large-scale ablation experiments across 17 strategies, 10 domains, and 5 LLMs to determine the optimal strategy for each step, establishing an exemplary research paradigm.
- **Key Insight**: "LLMs are not good long-context annotators buy are excellent filters"—exhibiting low kappa but high precision in long-context scenarios, which offers key guiding value for long-context data engineering.
- **Multi-hop Merging Does Not Need Rationale**: Unlike single-hop generation, incorporating rationales during the multi-hop merging stage is actually detrimental, revealing the limitations of CoT in complex synthesis scenarios.

## Limitations & Future Work

- Due to the high cost of large-scale distillation training for GPT-4-grade methods, some evaluations relied on human assessment of only 200 samples, which is a limited sample size.
- The framework depends on multiple LLM calls. Although optimized, the end-to-end synthesis cost remains relatively high, and more efficient synthesis workflows can be explored in the future.
- The study focus is limited to the QA task format, leaving other long-context types like summarization and info extraction unexplored.
- Generating long texts by splicing documents can create a distribution discrepancy compared to actual long-document scenarios.

## Related Work & Insights

- Most closely related to Quest (Gao et al., 2024), but while Quest constructs closed-source single-hop QA, this work models document relationships before generating multi-hop QA.
- The multi-agent collaboration mechanism can be extended to other data synthesis scenarios (e.g., multi-turn conversations, code generation data).
- The "decompose-then-combine" multi-hop data construction strategy also provides useful references for data construction in Retrieval-Augmented Generation (RAG).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The framework design of multi-agent collaborative synthesis for multi-hop data is novel, and the systematic investigation of various factors is unique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely comprehensive ablation experiments (17 strategies x 10 domains x 5 LLMs), rendering the conclusions highly credible.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, with each experimental finding well-summarized, though somewhat complex due to numerous mathematical notations.
- **Value**: ⭐⭐⭐⭐ Offers direct practical guidance for long-context data engineering; the LongMIT dataset and best practices are readily reusable by the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] NExtLong: Toward Effective Long-Context Training without Long Documents](../../ICML2025/llm_efficiency/nextlong_toward_effective_long-context_training_without_long_documents.md)
- [\[ACL 2025\] What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](many_shot_attacks_long_context.md)
- [\[ICML 2025\] Long-Short Alignment for Effective Long-Context Modeling in LLMs](../../ICML2025/llm_efficiency/long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [\[ACL 2025\] LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training](adaptive_grouped_pe_context_window.md)
- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)

</div>

<!-- RELATED:END -->
