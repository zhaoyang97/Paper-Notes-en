---
title: >-
  [Paper Note] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents
description: >-
  [ACL 2025][LLM (Other)][memory evaluation] This work constructs MemBench, the first evaluation benchmark for LLM Agent memory capabilities that simultaneously covers two interaction scenarios (participation/observation), two memory levels (factual/reflective), and four evaluation metrics (accuracy/recall/capacity/efficiency). Evaluations across seven memory mechanisms demonstrate that simple RetrievalMemory performs best under large-scale memory conditions ($100\text{K}$ toke…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "memory evaluation"
  - "LLM agent"
  - "reflective memory"
  - "factual memory"
  - "benchmark"
date: 2026-05-08
content_hash: 0c9487d0a1a80dd8
---

# MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents

**Conference**: ACL 2025  
**arXiv**: [2506.21605](https://arxiv.org/abs/2506.21605)  
**Code**: [https://github.com/import-myself/Membench](https://github.com/import-myself/Membench)  
**Area**: LLM NLP / Agent Memory  
**Keywords**: memory evaluation, LLM agent, reflective memory, factual memory, benchmark

## TL;DR
This work constructs MemBench, the first evaluation benchmark for LLM Agent memory capabilities that simultaneously covers two interaction scenarios (participation/observation), two memory levels (factual/reflective), and four evaluation metrics (accuracy/recall/capacity/efficiency). Evaluations across seven memory mechanisms demonstrate that simple RetrievalMemory performs best under large-scale memory conditions ($100\text{K}$ tokens) with an accuracy of $0.833$, while complex mechanisms (MemGPT, GenerativeAgent) fail to show advantages.

## Background & Motivation

**Background**: The memory module of LLM-based Agents serves as the core infrastructure for storing information and accumulating experience, with several memory mechanisms such as GenAgent, MemGPT, and MemoryBank being proposed. However, how to scientifically evaluate the memory capabilities of these mechanisms remains an open question.

**Limitations of Prior Work**: Existing evaluations suffer from three major shortcomings: (1) Single memory level—they only test factual memory (e.g., "what dishes the user likes") while neglecting reflective memory (e.g., "what the user's taste preferences are"), which requires inductive summarization from multiple interactions; (2) Single interaction scenario—they only cover the participation scenario where the Agent directly dialogues with the user, ignoring the observation scenario where the Agent acts as an observer recording information; (3) Single evaluation metric—focusing only on accuracy while neglecting efficiency (read/write time overhead) and capacity (whether performance collapses as memory size increases).

**Key Challenge**: The design goals of Agent memory mechanisms are multi-dimensional—requiring high accuracy, high recall, rapid processing (efficiency), and large storage (capacity). However, existing benchmarks cannot measure these dimensions simultaneously, making it difficult to comprehensively compare the strengths and bottlenecks of different mechanisms.

**Goal**: (1) Construct an evaluation dataset containing both factual and reflective memory levels; (2) Design both participation and observation interaction scenarios; (3) Propose a comprehensive evaluation benchmark across four dimensions: accuracy, recall, capacity, and efficiency.

**Key Insight**: Real user preference hierarchical relationships are extracted from recommendation system datasets (MovieLens, Food, Goodreads) to construct a mapping dictionary from "low-level specific preferences $\rightarrow$ high-level abstract preferences," serving as the ground truth for reflective memory.

**Core Idea**: The first multi-scenario (participation/observation), multi-level (factual/reflective), and multi-metric (accuracy/recall/capacity/efficiency) Agent memory evaluation benchmark.

## Method

### Overall Architecture
Data generation pipeline: Construct the user relation graph (profile + related entities) $\rightarrow$ Sample high-level preference attributes from recommendation datasets $\rightarrow$ Construct high-to-low level preference mappings $\rightarrow$ Generate multi-turn dialogues containing key evidence conversations $\rightarrow$ Organize dialogues into participation/observation data formats chronologically $\rightarrow$ Insert noisy sessions to control difficulty $\rightarrow$ Evaluate on seven memory mechanisms using four metrics.

### Key Designs

1. **Dual-Scenario Data Construction (Participation + Observation Scenarios)**:

    - **Function**: Distinguish between the Agent's memory needs in "actively participating in dialogues" and "passively observing information streams."
    - **Mechanism**: **Participation Scenario**—The Agent engages in multi-turn dialogues with the user, needing to remember the user's inputs and its own responses (e.g., remembering what was recommended after recommendation). The data is in a multi-session dialogue format. **Observation Scenario**—The Agent only passively receives the user's message stream (e.g., group chat logs) without responding. The data is a list of timestamped messages. To eliminate interference from other modules (such as reasoning) on memory, the Agent's responses in the participation scenario are predefined.
    - **Design Motivation**: Existing benchmarks (e.g., LoCoMo, LongMemEval) only cover participation scenarios, but in practical applications, Agents frequently act as observers (e.g., monitoring group chats, keeping meeting minutes). The memory requirements for these two scenarios are distinct.

2. **Dual-Level Memory Content (Factual + Reflective Memory)**:

    - **Function**: Distinguish between explicit information extraction and implicit preference induction.
    - **Mechanism**: **Factual Memory**—Specifically test concrete attributes extracted directly from dialogues (e.g., kin age, event timestamps), evaluating capabilities such as information extraction, cross-session reasoning, knowledge updating, and temporal reasoning. **Reflective Memory**—Inductively summarize high-level preferences from multiple low-level preference expressions (e.g., "likes Star Wars + likes Blade Runner $\rightarrow$ science fiction movie lover"). Ground truth labels are high-level preference categories extracted using user-item relations from MovieLens, Food, and Goodreads datasets.
    - **Design Motivation**: Factual memory is "remembering what was said," whereas reflective memory is "understanding what it means." The latter is closer to how human memory works, yet it has been entirely unexamined in prior evaluations.

3. **Four-Dimensional Evaluation Metrics (Accuracy / Recall / Capacity / Efficiency)**:

    - **Function**: Comprehensively evaluate the overall performance of memory mechanisms.
    - **Mechanism**: **Accuracy**—All questions are formatted as multiple-choice questions, and answers are compared against the ground truth. **Recall**—For retrieval-based mechanisms, Recall@10 is used to measure whether key evidence dialogues are successfully retrieved. **Capacity**—The volume of memory is increased incrementally ($10\text{K} \rightarrow 100\text{K}$ tokens) to observe whether accuracy drops abruptly. **Efficiency**—The read/write time overhead per message turn is measured (seconds/operation).
    - **Design Motivation**: Mechanisms with high accuracy but an unacceptably high read/write latency (e.g., 6 seconds per operation) are impractical for real-world scenarios. The capacity metric can expose the hard limits of memory windows.

### Dataset Statistics
Participation Scenario: factual memory has 51K sessions / 39K questions, reflective memory has 3.5K sessions / 3.5K questions. Observation Scenario: factual memory has 8.5K message lists / 8.5K questions, reflective memory has 2K / 2K. Average tokens: participation factual is 10,285 TPT, observation factual is 617 TPT.

## Key Experimental Results

### Main Results (Factual Memory, based on Qwen2.5-7B)

| Memory Method | Participation-Acc(10K) | Participation-Acc(100K) | Observation-Acc(1K) | Observation-Acc(100K) | Write Time (s) |
|----------|-------------|---------------|-------------|---------------|-----------|
| FullMemory | 0.647 | 0.489 | 0.786 | 0.631 | <0.001 |
| RecentMemory | 0.639 | 0.422 | 0.800 | 0.512 | <0.001 |
| **RetrievalMemory** | **0.692** | **0.833** | **0.883** | **0.933** | 0.058 |
| GenerativeAgent | 0.478 | 0.455 | 0.779 | 0.476 | 6.116 |
| MemoryBank | 0.442 | 0.456 | 0.721 | 0.488 | 8.047 |
| MemGPT | 0.455 | 0.411 | 0.789 | 0.488 | 0.106 |
| SCMemory | 0.355 | 0.444 | 0.529 | 0.429 | 2.276 |

### Reflective Memory Evaluation

| Memory Method | Participation-Acc(10K) | Participation-Acc(100K) | Observation-Acc(1K) | Observation-Acc(100K) |
|----------|-------------|---------------|-------------|---------------|
| FullMemory | 0.733 | 0.533 | 0.883 | 0.333 |
| RetrievalMemory | 0.692 | 0.833 | 0.883 | 0.933 |
| GenerativeAgent | 0.742 | 0.333 | 0.883 | 0.200 |
| MemoryBank | 0.692 | 0.400 | 0.900 | 0.333 |
| MemGPT | 0.733 | 0.367 | 0.883 | 0.200 |

### Key Findings
- RetrievalMemory significantly outperforms all complex mechanisms in the $100\text{K}$ token scenario (factual accuracy of $0.833$ vs. $0.455$ for GenerativeAgent), counter-intuitively proving that simple retrieval is more reliable than complex memory management.
- FullMemory and RecentMemory severely degrade at $100\text{K}$ (from $0.647 \rightarrow 0.489$ and $0.800 \rightarrow 0.512$, respectively) as key information is pushed out of the context window.
- GenerativeAgent has a write latency of 6.1 seconds/operation and MemoryBank has 8.0 seconds/operation, making them impractical for real-time scenarios.
- Regarding reflective memory, the accuracy of most mechanisms plummets to $0.2 \– 0.4$ at $100\text{K}$, whereas only RetrievalMemory maintains $0.933$ (in the observation scenario), showing that high-level inductive capability almost completely fails under large-scale memory conditions.
- RetrievalMemory's Recall@10 is $0.776$ at $10\text{K}$ and $0.749$ at $100\text{K}$; the stable retrieval quality is key to maintaining its performance.

## Highlights & Insights
- The introduction of the "reflective memory" concept is a core contribution—shifting Agent memory evaluation from "what is remembered" to "what is understood." The method of leveraging recommendation system data to build high-to-low level preference mappings as the ground truth is highly reusable.
- The four-dimensional metrics expose the "false prosperity" of complex memory mechanisms—GenerativeAgent yields moderate accuracy yet suffers from a 6-second write latency, resulting in extremely low cost-effectiveness for practical deployment.
- The capacity test reveals a critical threshold effect: when memory size increases from $10\text{K}$ to $100\text{K}$, window-based methods experience a steep decline, whereas retrieval-based methods actually improve (due to more noise but stable retrieval precision), which provides direct guidance for the architectural design of Agent memory.

## Limitations & Future Work
- Evaluations were conducted solely using Qwen2.5-7B as the base model. Testing stronger LLMs (e.g., GPT-4o, Claude 3.5) might alter the ranking landscape.
- The test size for reflective memory is relatively small (120/60 samples), limiting the statistical power.
- The "forgetting" mechanism of memory is not considered; in real-world scenarios, Agents should be capable of selectively forgetting irrelevant information.
- The noise data consists only of news text, and the domain discrepancy with conversation content might affect the realism of the evaluation.

## Related Work & Insights
- **vs LoCoMo**: LoCoMo only features participation scenarios and factual memory without user profiles, while MemBench incorporates observation scenarios, reflective memory, and user profiles, representing a comprehensive multi-dimensional upgrade.
- **vs LongMemEval**: LongMemEval focuses primarily on factual memory evaluation within long dialogues, whereas MemBench supplements this with reflective memory and multi-metric dimensions.
- **vs MemGPT**: MemGPT's hierarchical memory management performs poorly on MemBench (Acc of $0.411$ @ $100\text{K}$), implying its memory organization strategy might fail under realistic workloads.
- Insight: The architectural design of Agent memory should prioritize retrieval precision over pursuing complex memory management mechanisms.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of reflective memory, observation scenarios, and four-dimensional metrics is a pioneering effort that fills a crucial evaluation gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison of seven memory mechanisms alongside capacity and efficiency analysis, although limited by the usage of a single base model.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework with rich charts and a highly detailed description of the data construction pipeline.
- **Value**: ⭐⭐⭐⭐ Provides direct guidance for Agent memory design—proving the superiority of simple retrieval is a highly valuable finding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] C²LEVA: Toward Comprehensive and Contamination-Free Language Model Evaluation](c2leva_toward_comprehensive_and_contamination-free_language_model_evaluation.md)
- [\[ACL 2025\] TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues](tremu_towards_neuro-symbolic_temporal_reasoning_for_llm-agents_with_memory_in_mu.md)
- [\[ACL 2025\] Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference](can_llms_reason_about_program_semantics_a_comprehensive_evaluation_of_llms_on_fo.md)
- [\[ACL 2025\] Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents](simulating_diverse_students.md)
- [\[ACL 2025\] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](lm_graph_search_supervision.md)

</div>

<!-- RELATED:END -->
