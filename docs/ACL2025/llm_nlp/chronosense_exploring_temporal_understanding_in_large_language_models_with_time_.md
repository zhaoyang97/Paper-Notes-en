---
title: >-
  [Paper Note] ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events
description: >-
  [ACL2025][LLM (Other)][Temporal Reasoning] This paper proposes the ChronoSense benchmark, which is the first to fully cover all 13 temporal relations of Allen's interval algebra and introduce three types of temporal arithmetic tasks. Through a systematic evaluation of seven LLMs under 0-shot, few-shot, and CoT settings, it reveals that models' temporal understanding capabilities are generally weak and heavily rely on pre-training memory.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Temporal Reasoning"
  - "Allen's Interval Relations"
  - "Benchmark"
  - "Temporal Arithmetic"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 3a4a33bc0c36dee5
---

# ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events

**Conference**: ACL2025  
**arXiv**: [2501.03040](https://arxiv.org/abs/2501.03040)  
**Code**: [duyguislakoglu/chronosense](https://github.com/duyguislakoglu/chronosense)  
**Area**: LLM/NLP  
**Keywords**: Temporal Reasoning, Allen's Interval Relations, Benchmark, Temporal Arithmetic, LLM Evaluation

## TL;DR

This paper proposes the ChronoSense benchmark, which is the first to fully cover all 13 temporal relations of Allen's interval algebra and introduce three types of temporal arithmetic tasks. Through a systematic evaluation of seven LLMs under 0-shot, few-shot, and CoT settings, it reveals that models' temporal understanding capabilities are generally weak and heavily rely on pre-training memory.

## Background & Motivation

1. **Weak Temporal Reasoning in LLMs**: Although LLMs perform exceptionally well on general NLP tasks, they still exhibit significant shortcomings in reasoning, arithmetic, and numerical processing, which directly constrains their temporal reasoning capabilities.
2. **Allen's Interval Algebra Long Ignored as a Foundational Framework**: Allen's interval algebra defines 13 mutually exclusive and exhaustive relations between two time intervals (Before, After, During, Contains, Equals, Overlaps, Overlapped-by, Meets, Met-by, Starts, Started-by, Finishes, Finished-by). Despite being a classic formal tool in temporal reasoning used for over 30 years, **none** of the existing LLM temporal reasoning benchmarks fully cover all 13 relations.
3. **Deficiencies in Existing Benchmarks**: TimeBench focuses on abstract temporal expressions and commonsense reasoning but lacks complete coverage of Allen relations; TGQA only covers three simple event relations; TRAM lacks explicit start and end time information for events; TORQUE lacks clear timestamps. A systematic benchmark is urgently needed to fill this gap.
4. **Indistinguishable Memory vs. Reasoning**: When models encounter real historical events present in their pre-training corpora, they might directly "memorize the answer" instead of performing genuine temporal reasoning. An experimental mechanism is required to isolate this confounding factor.
5. **High Reliability Required in Practical Applications**: Scenarios such as historical analysis, legal AI, and medical timeline management demand temporal relation judgments with accuracy far exceeding the 50% random baseline, necessitating diagnostic tools to pinpoint specific weaknesses in LLMs.

## Method

### Overall Architecture

ChronoSense consists of two main categories of True/False binary classification questions, with the temporal granularity unified to "years":

- **Allen Interval Relation Questions** (Type 1): Given two events and their time intervals, determine whether a specific Allen relation holds true.
- **Temporal Arithmetic Questions** (Type 2): Given the temporal attributes of a single event, determine whether a specific arithmetic derivation conclusion is correct.

Each Allen relation and each arithmetic task contain **4,000 training / 500 validation / 500 test** samples, with a strict 50:50 balance between positive and negative samples.

### Construction of Allen Interval Relation Questions

The data generation pipeline is as follows:

1. **Event Pair Extraction**: Real historical event pairs with explicit start and end years are extracted from Wikidata using SPARQL queries.
2. **Automated Relation Labeling**: The time intervals of the two events are compared to determine the uniquely correct Allen relation.
3. **Natural Language Generation**: The Allen relation is converted into a natural language hypothesis, which is then paired with the event context.
4. **Negative Sample Generation**: The correct Allen relation is replaced with another relation and labeled as False. Since certain relations might be ambiguous under year granularity (e.g., Equals cannot easily use Contains as a negative sample because the exact calendar dates are unknown), exclusion rules are carefully designed to avoid ambiguous cases.
5. **Abstract Version**: Real event names are replaced with "Event A" and "Event B" to detect memory effects.
6. **Multi-Prompt Templates**: Three different natural language phrasings are designed for each question to test prompt sensitivity.

Example: The Context provides "Fourth Cholera Pandemic 1863-1875" and "World War II 1939-1945". The Hypothesis asks, "Did the Fourth Cholera Pandemic occur before World War II with no overlap?" -> Correctness = True.

### Construction of Temporal Arithmetic Questions

Three types of arithmetic tasks:

- **End Timepoint**: Given the start time and duration, determine if the calculated end time is correct (e.g., starts in 1948 + 39 years = 1987?).
- **Next Occurrence**: Given the first occurrence time and frequency period, determine if the next occurrence time is correct (e.g., first in 1773 + every 5 years → occurs in 1779?).
- **Intermediate Timepoint** (First proposed in this paper): Determine whether an event was ongoing during a specific year within its start and end interval.

Due to the scarcity of periodic events in Wikidata, all arithmetic questions utilize synthetic "Event A" names, placing them naturally in an abstract setting.

### Evaluation Setup

- **7 Models**: Gemma2-9B-it, GPT-4o, GPT-4o-mini, Llama3.1-8B, Mistral-7B, Mixtral-8x7B, Phi-3-mini.
- **4 Prompting Strategies**: 0-shot, 1-shot, 3-shot, and Chain-of-Thought (CoT, adding "Let's think step by step." to the end of the prompt).
- **Evaluation Metric**: Accuracy (random baseline is 50%).
- **Generation Constraints**: Maximum 64 tokens for standard settings and 512 tokens for CoT settings.

## Key Experimental Results

### Table 1: Average Accuracy on Allen Relations and Arithmetic Tasks

| Task Type | Setup | Gemma2-9B | GPT-4o | GPT-4o-mini | Llama3.1-8B | Mistral-7B | Mixtral-8x7B | Phi-3-mini |
|---------|------|-----------|--------|-------------|-------------|------------|-------------|-----------|
| Allen | 0-shot | 0.09\* | **0.87** | 0.72 | 0.13\* | 0.50 | 0.54 | 0.56 |
| Allen | 1-shot | 0.75 | **0.93** | 0.75 | 0.01\* | 0.47 | 0.56 | 0.59 |
| Allen | 3-shot | 0.26\* | **0.95** | 0.78 | 0.01\* | 0.49 | 0.58 | 0.66 |
| Allen | CoT | 0.75 | 0.65 | 0.69 | **0.75** | 0.51 | 0.57 | **0.75** |
| Allen (Abstract) | 0-shot | 0.15\* | **0.78** | 0.64 | 0.14\* | 0.23\* | 0.35 | 0.61 |
| Arithmetic | 0-shot | **0.76** | 0.55 | 0.60 | 0.48 | 0.36\* | 0.35 | 0.67 |
| Arithmetic | CoT | 0.94 | **0.99** | **0.99** | 0.92 | 0.70 | 0.75 | 0.98 |

> \* indicates that accuracy is distorted due to the model generating a large number of unclear answers ( can be ≥250 unanswered True/False).

### Table 2: 0-shot Breakdown Performance of GPT-4o on 13 Allen Relations

| Allen Relation | GPT-4o | Mixtral-8x7B | Phi-3-mini | 7-Model Avg |
|-----------|--------|-------------|-----------|----------|
| Before | 0.914 | 0.902 | 0.758 | 0.70 |
| After | 0.956 | 0.780 | 0.566 | 0.64 |
| Contains | 0.884 | 0.472 | 0.652 | 0.47 |
| During | 0.878 | 0.512 | 0.490 | 0.46 |
| Overlaps | 0.884 | 0.430 | 0.648 | 0.47 |
| Overlapped-By | 0.842 | 0.476 | 0.786 | 0.49 |
| Meets | 0.910 | 0.740 | 0.488 | 0.52 |
| Met-By | 0.864 | 0.594 | 0.494 | 0.48 |
| Starts | 0.846 | 0.442 | 0.492 | 0.45 |
| Started-By | 0.896 | 0.578 | 0.474 | 0.46 |
| Finishes | 0.908 | 0.430 | 0.492 | 0.43 |
| Finished-By | 0.926 | 0.398 | 0.486 | 0.45 |
| **Equals** | **0.690** | **0.336** | 0.540 | **0.33** |

### Table 3: Comparison of Three Temporal Arithmetic Subtasks under 0-shot and CoT

| Subtask | Setup | GPT-4o | GPT-4o-mini | Phi-3-mini | Gemma2-9B | 7-Model Avg |
|--------|------|--------|-------------|-----------|-----------|----------|
| End Timepoint | 0-shot | 0.552 | 0.652 | 0.604 | 0.670 | 0.56 |
| End Timepoint | CoT | 0.978 | 0.978 | 0.996 | 0.992 | 0.95 |
| Intermediate | 0-shot | 1.000 | 0.996 | 0.994 | 0.938 | 0.81 |
| Intermediate | CoT | 0.998 | 0.998 | 0.984 | 0.978 | 0.86 |
| Next Occurrence | 0-shot | 0.126\* | 0.158\* | 0.432 | 0.678 | 0.24 |
| Next Occurrence | CoT | **1.000** | **1.000** | 0.962 | 0.874 | 0.88 |

> Next Occurrence is extremely challenging in the 0-shot setting (most models perform far below the random baseline), but under CoT, GPT-4o reaches 1.000, fully demonstrating the value of step-by-step reasoning.

## Key Findings

1. **Low Overall Performance**: Most models perform near or below the random baseline (0.50) on Allen relation tasks. Even the strongest model, GPT-4o, only scores 0.87 in the 0-shot setting, showing significant performance gaps across different relations.
2. **Asymmetrical Handling of Symmetrical Relations**: Despite Before/After and Contains/During being symmetrical relations, models show notable performance differences (Before average of 0.70 vs. After of 0.64; Contains average of 0.47 vs. During of 0.46), indicating that models do not truly comprehend temporal symmetry.
3. **Equals is the Hardest Relation**: The average accuracy under 0-shot and CoT is only 0.33 and 0.49, as it requires comparing two boundaries for exact matches.
4. **Significant Memory Effect**: When switching from real event names to abstract names in Allen relation questions, GPT-4o's accuracy drops from 0.87 to 0.78, and Mixtral's drops from 0.54 to 0.35, confirming that models partially rely on pre-training memory rather than genuine reasoning.
5. **Stunning Effect of CoT on Arithmetic Tasks**: The performance improvement from 0-shot to CoT is significantly larger for arithmetic tasks than for Allen relation tasks (GPT-4o: 0.55 → 0.99), because arithmetic tasks are essentially multi-step calculations well-suited for step-by-step reasoning.
6. **Diverse Typical Error Patterns**: Confounding start and end years, logical reasoning errors, redundant irrelevant calculations, correct explanations but wrong concluded answers, and confusion caused by temporal granularity.

## Highlights & Insights

1. **The first LLM temporal reasoning benchmark to fully cover all 13 of Allen's interval relations**, filling a long-standing gap in this field.
2. **Abstract version design** strictly isolates memory retrieval from structural reasoning, representing a rigorous experimental design.
3. **The Intermediate Timepoint task is proposed for the first time**, testing the model's ability to judge whether an event is currently occurring within a time interval, which is novel and original.
4. **Thoughtful negative sample generation strategy**: Ambiguous Allen relations are carefully excluded as negative samples under year granularity, ensuring unequivocal labels.
5. **Comprehensive evaluation coverage**: 4 prompting strategies × 7 models × 16 tasks × 3 prompt variants, presenting a complete evaluation matrix.

## Limitations & Future Work

1. **Single Temporal Granularity**: The temporal granularity is restricted to "years," which cannot evaluate models' performance at finer scales such as day, month, or hours/seconds, while fine-grained reasoning is more critical in practical scenarios.
2. **Limited Model Coverage**: Only 7 models (including 2 closed-source ones) are evaluated, lacking other prominent model series like Claude, Qwen, and DeepSeek.
3. **Simple Task Formulation**: Questions are limited to True/False binary classification, lacking tests of the models' ability to actively recognize or generate Allen relations.
4. **Lack of Transitive Reasoning**: It only tests pairwise event relation judgments, without exploiting the transitivity properties of Allen's algebra for multi-event chain reasoning.
5. **Ambiguity in Some Event Names**: Wikidata contains exhibitions named after people, which may introduce additional confusion to the models.
6. **Truncation of Outputs**: Responses exceeding the maximum token limit are directly truncated, potentially omitting the correct parsed answers.

## Rating

- Novelty: ⭐⭐⭐⭐ First to fully cover all 13 Allen interval relations; the abstract version design and Intermediate Timepoint tasks are of original value.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 models × 4 strategies × 16 tasks × 3 prompt variants provide a comprehensive matrix, though model coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐ Highly clear problem definitions, closely knit experimental designs, and deep analysis.
- Value: ⭐⭐⭐⭐ Provides a standardized diagnostic tool for LLMs' temporal reasoning capabilities, directly guiding future directions for model improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Does Time Have Its Place? Temporal Heads Where Language Models Recall Time-specific Information](does_time_have_its_place_temporal_heads_where_language_models_recall_time-specif.md)
- [\[ACL 2025\] SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms](synapticrag_enhancing_temporal_memory_retrieval_in_large_language_models_through.md)
- [\[ACL 2025\] DeAL: Decoding-time Alignment for Large Language Models](deal_decoding_time_alignment.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)

</div>

<!-- RELATED:END -->
