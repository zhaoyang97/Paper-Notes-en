---
title: >-
  [Paper Note] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering
description: >-
  [ACL 2026][LLM Evaluation][Tool-Augmented Reasoning] This paper constructs ReCoQA—a large-scale benchmark containing 29,270 real estate QA pairs that require models to integrate database queries and map API calls for hyb…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Tool-Augmented Reasoning"
  - "Multi-Step Reasoning"
  - "Real Estate QA"
  - "Multi-Agent Framework"
  - "Benchmark Dataset"
date: 2026-05-08
content_hash: 5d098f7db01e364b
---

# ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering

**Conference**: ACL 2026  
**arXiv**: [2604.17944](https://arxiv.org/abs/2604.17944)  
**Code**: [https://github.com/Husky-989/ReCoQA](https://github.com/Husky-989/ReCoQA)  
**Area**: LLM Reasoning  
**Keywords**: Tool-Augmented Reasoning, Multi-Step Reasoning, Real Estate QA, Multi-Agent Framework, Benchmark Dataset

## TL;DR
This paper constructs ReCoQA—a large-scale benchmark containing 29,270 real estate QA pairs that require models to integrate database queries and map API calls for hybrid multi-source reasoning. It proposes a hierarchical multi-agent framework, HIRE-Agent, as a strong baseline, systematically revealing the bottlenecks of existing LLMs in complex reasoning for vertical domains.

## Background & Motivation

**Background**: In real estate decision-making, users must switch between multiple platforms—comparing listings on one site, calculating commute times on a map app, and checking school district information on government websites. This fragmented information acquisition incurs significant time costs and cognitive load. AI Agents are a powerful solution, but existing QA benchmarks cannot effectively evaluate this hybrid reasoning capability.

**Limitations of Prior Work**: Classical datasets like Spider focus only on structured queries, while recent Agent benchmarks evaluate general tool usage, but both treat database queries and external API calls as independent capabilities. Existing benchmarks fail to simulate hybrid workflows in real-world scenarios—for example, where the output of a database query (list of candidate residential areas) is dyamically required as input for an API call (distance calculation). Furthermore, most map-related QA datasets assume geographic information is static, ignoring dynamic values such as commute times.

**Key Challenge**: Real-world vertical domain decision-making requires tight coupling of heterogeneous information sources and multi-step reasoning, yet a benchmark for systematically evaluating this capability is lacking.

**Goal**: (1) Construct an end-to-end benchmark covering static database queries, dynamic API calls, and multi-step reasoning; (2) Establish a hierarchical multi-agent baseline and systematically analyze the bottlenecks of each module.

**Key Insight**: Utilizing real estate purchasing consultation as the entry point—a vertical scenario that naturally requires tight integration of database queries (property attributes) and map APIs (commute distances, surrounding facilities).

**Core Idea**: Design a large-scale benchmark with three progressive question difficulty levels (simple query, joint query, multi-step reasoning) and utilize a hierarchical "Understand-Plan-Execute" Agent architecture as a strong baseline.

## Method

### Overall Architecture
ReCoQA consists of two contributions: the dataset and the HIRE-Agent framework. The dataset covers 8 major Chinese cities, including residential area information, POI data, and location pairing data stored in PostgreSQL, and provides 4 map API functions. HIRE-Agent adopts a three-tier architecture consisting of a "Frontend Agent + Supervisor Agent + Expert Agent."

### Key Designs

1.  **Three-level Progressive Difficulty Question Design**:

    - **Function**: Systematically evaluate reasoning capabilities from simple to complex.
    - **Mechanism**: Type 1 (Simple Query) requires only direct database queries; Type 2 (Joint Query) requires simultaneous use of DB and Map APIs; Type 3 (Multi-step Reasoning) requires chain-of-reasoning—first querying the DB to get coordinates, then calling APIs to calculate distances, and finally performing comparative reasoning based on API results. Each sample is annotated with complete intermediate steps (SLU labels, SQL statements, API call sequences).
    - **Design Motivation**: The progressive difficulty design allows for pinpointing exactly where the model fails in the reasoning chain, and intermediate step annotations enable interpretable evaluation.

2.  **Frontend Agent (SLU Module)**:

    - **Function**: Parses natural language queries from users into structured intent and slot representations.
    - **Mechanism**: Fine-tunes a BERT model for intent detection (16 predefined intents) and slot filling (19 slot types, IOB tagging), parsing "Commute from Tianhe District to Jimanjia Plaza within 30 minutes" into specific locations, transportation modes, and time constraints.
    - **Design Motivation**: Decoupling intent understanding from execution is essential—ablation studies show that without SLU labels, Type 3 accuracy averages only 0.2921, which improves to 0.6535 when added.

3.  **Supervisor Agent (Task Orchestration)**:

    - **Function**: Receives structured input and performs task decomposition and orchestration.
    - **Mechanism**: Uses CoT prompting to generate execution plans, dispatches subtasks to Expert Agents sequentially according to the plan, and dynamically adjusts based on feedback (continues on success, re-plans on failure), with a maximum step limit to prevent infinite loops.
    - **Design Motivation**: Centralized orchestration ensures global consistency in multi-step reasoning, and the re-planning mechanism improves robustness.

### Loss & Training
The SLU module fine-tunes BERT using cross-entropy loss; the LLM Agent components use ICL (5-shot) guidance without additional training. API results are implemented via pre-caching for deterministic and cost-free execution.

## Key Experimental Results

### Main Results

| Model | Method | Type 1 Acc | Type 2 Acc | Type 3 Acc | Overall Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-72B | Standard | 0.8082 | 0.7110 | 0.3973 | 0.5855 |
| Qwen2.5-72B | HIRE-Agent | 0.8862 | 0.6581 | **0.6211** | **0.6989** |
| Qwen3-30B A3B | Standard | 0.7394 | 0.5871 | 0.3512 | 0.5131 |
| Qwen3-30B A3B | HIRE-Agent | 0.7659 | **0.8645** | **0.8371** | **0.8260** |
| Average | Standard | 0.7741 | 0.6090 | 0.3559 | 0.5299 |
| Average | HIRE-Agent | **0.8453** | **0.7658** | **0.6535** | **0.7323** |

### Ablation Study (Bottleneck Analysis)

| Component | Key Metric | Description |
| :--- | :--- | :--- |
| No SLU → With SLU | Type 3: 0.2921 → 0.6535 | SLU module contributes the most, Gain +0.3614 |
| GT SQL Labels | Qwen2.5-72B Gain +0.1407 | SQL generation is the primary bottleneck for this model |
| GT API Labels | Qwen3-8B Gain +0.0660 | Tool invocation is the primary bottleneck for this model |
| All GT Labels | Overall Acc only 0.8864 | Exposes a "composition gap" in global planning and final reasoning |

### Key Findings
- The hierarchical architecture improves overall accuracy by an average of 20.24 percentage points, with the largest gain in Type 3 multi-step reasoning (+29.76 percentage points).
- Even with GT labels for all intermediate steps, accuracy remains only 0.8864, indicating a "composition gap"—where models cannot perfectly integrate results from multiple subtasks.
- Qwen3-30B exhibited an "overthinking" phenomenon: performance was actually worse on simple questions (complex reasoning ability interfered with the direct execution of simple queries).
- Bottlenecks vary across models: Qwen2.5-72B is bottlenecked by SQL generation, while Qwen3-8B is bottlenecked by tool invocation, revealing heterogeneity in model capabilities.

## Highlights & Insights
- The **progressive bottleneck analysis method** is highly valuable—by step-wise replacement with GT labels to locate the performance contribution of each module, this diagnostic approach can be migrated to any multi-module system evaluation.
- The **API caching strategy** achieves benchmark reproducibility and zero-cost usage—by pre-storing real-time API call results as local database queries, it ensures deterministic results and eliminates API fees.
- The discovery of the **"overthinking" phenomenon** in large models—where strong reasoning capabilities become a burden for simple tasks—has important implications for Agent design.

## Limitations & Future Work
- The dataset currently covers real estate scenarios in only 8 Chinese cities, presenting significant geographic and cultural limitations.
- Questions are generated based on 41 templates; though rewritten, pattern repetition may still exist.
- While the API caching strategy improves reproducibility, it does not reflect real-time API latency and error handling.
- Future work could extend to more vertical domains (medical, legal, finance) to verify the framework's universality.

## Related Work & Insights
- **vs Spider**: Spider only evaluates Text-to-SQL without involving external API calls and multi-source fusion; ReCoQA requires tight coupling of databases and APIs.
- **vs RETQA**: RETQA introduces SLU but is limited to static databases and does not support dynamic geographic information queries.
- **vs MACT**: MACT demonstrates complex Agent collaboration but relies on Pandas for data operation, which has memory and scalability issues; HIRE-Agent's combination of SQL + API is more scalable.

## Rating
- Novelty: ⭐⭐⭐⭐ First vertical domain benchmark to systematically evaluate hybrid DB-API reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed with 4 LLMs, multi-layer ablation, bottleneck analysis, and real-world scenario testing.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and in-depth analysis.
- Value: ⭐⭐⭐⭐ The dataset and analysis methodology are of significant reference value to the Agent research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain](diningbench_a_hierarchical_multi-view_benchmark_for_perception_and_reasoning_in_.md)
- [\[ACL 2026\] Idiom Understanding as a Tool to Measure the Dialect Gap](idiom_understanding_as_a_tool_to_measure_the_dialect_gap.md)
- [\[ACL 2026\] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](k-metbench_a_multi-dimensional_benchmark_for_fine-grained_evaluation_of_expert_r.md)

</div>

<!-- RELATED:END -->
