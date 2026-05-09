---
title: >-
  [Paper Note] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering
description: >-
  [ACL 2026][LLM Reasoning][Tool-augmented reasoning] This paper introduces ReCoQA—a large-scale benchmark comprising 29,270 real estate question-answer pairs—that requires models to perform hybrid multi-source reasoning by integrating database queries and map API calls. The authors further propose HIRE-Agent, a hierarchical multi-agent framework serving as a strong baseline, and systematically identify the bottlenecks of existing LLMs in complex reasoning within vertical domains.
tags:
  - ACL 2026
  - LLM Reasoning
  - Tool-augmented reasoning
  - multi-step reasoning
  - real estate QA
  - multi-agent framework
  - benchmark dataset
date: 2026-05-08
content_hash: 1594ae140bd338bf
---

# ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering

**Conference**: ACL 2026  
**arXiv**: [2604.17944](https://arxiv.org/abs/2604.17944)  
**Code**: [https://github.com/Husky-989/ReCoQA](https://github.com/Husky-989/ReCoQA)  
**Area**: LLM Reasoning  
**Keywords**: Tool-augmented reasoning, multi-step reasoning, real estate QA, multi-agent framework, benchmark dataset

## TL;DR
This paper introduces ReCoQA—a large-scale benchmark comprising 29,270 real estate question-answer pairs—that requires models to perform hybrid multi-source reasoning by integrating database queries and map API calls. The authors further propose HIRE-Agent, a hierarchical multi-agent framework serving as a strong baseline, and systematically identify the bottlenecks of existing LLMs in complex reasoning within vertical domains.

## Background & Motivation

**State of the Field**: Real estate decision-making requires users to navigate multiple platforms simultaneously—comparing listings on one website, estimating commute times via a map application, and consulting school district information on government portals. This fragmented information retrieval process imposes significant time costs and cognitive load. AI agents offer a promising solution, yet existing QA benchmarks are inadequate for evaluating such hybrid reasoning capabilities.

**Limitations of Prior Work**: Classic datasets such as Spider focus exclusively on structured queries, while recent agent benchmarks assess general tool use; however, both treat database queries and external API calls as isolated capabilities. Existing benchmarks fail to simulate hybrid workflows encountered in practice—for instance, where the output of a database query (a list of candidate neighborhoods) must dynamically serve as the input to an API call (distance computation). Furthermore, most map QA datasets assume static geographic information, neglecting dynamic values such as commute times.

**Root Cause**: Real-world vertical-domain decision-making demands tight coupling of heterogeneous information sources and multi-step reasoning, yet no benchmark systematically evaluates this capability.

**Paper Goals**: (1) Construct an end-to-end benchmark encompassing static database queries, dynamic API calls, and multi-step reasoning; (2) establish a hierarchical multi-agent baseline and systematically analyze per-module bottlenecks.

**Starting Point**: The paper takes real estate purchase consultation as its entry point—a domain that inherently requires tight integration of database queries (property attributes) and map APIs (commute distances, nearby amenities).

**Core Idea**: Design a large-scale benchmark with three progressively difficult question types (simple query, joint query, multi-step reasoning), and employ a hierarchical "understand–plan–execute" agent architecture as a strong baseline.

## Method

### Overall Architecture
ReCoQA encompasses two contributions: a dataset and the HIRE-Agent framework. The dataset covers eight major Chinese cities and includes neighborhood information, POI data, and location-pair data stored in PostgreSQL, along with four map API functions. HIRE-Agent adopts a three-tier architecture comprising a frontend agent, a supervisor agent, and expert agents.

### Key Designs

1. **Three-Level Progressive Question Type Design**:

    - Function: Systematically evaluate reasoning capabilities ranging from simple to complex.
    - Mechanism: Type 1 (simple query) requires only direct database retrieval; Type 2 (joint query) requires concurrent use of the database and map APIs; Type 3 (multi-step reasoning) requires chained inference—first querying the database for coordinates, then invoking the API to compute distances, and finally performing comparative reasoning based on API results. Each sample is annotated with complete intermediate steps (SLU labels, SQL statements, and API call sequences).
    - Design Motivation: The progressive difficulty design precisely localizes where in the reasoning chain a model fails; intermediate step annotations enable interpretable evaluation.

2. **Frontend Agent (SLU Module)**:

    - Function: Parse natural language user queries into structured intent and slot representations.
    - Mechanism: A fine-tuned BERT model performs intent detection (16 predefined intents) and slot filling (19 slot types with IOB tagging), mapping queries such as "commute to Jimanjia Plaza within 30 minutes in Tianhe District" to specific locations, transportation modes, and temporal constraints.
    - Design Motivation: Decoupling intent understanding from execution is essential—ablation experiments show that without SLU labels, Type 3 accuracy averages only 0.2921, rising to 0.6535 upon inclusion.

3. **Supervisor Agent (Task Orchestration)**:

    - Function: Receive structured input, decompose tasks, and orchestrate their execution.
    - Mechanism: Chain-of-thought prompting generates an execution plan; sub-tasks are dispatched sequentially to expert agents; the supervisor dynamically adjusts based on feedback (continuing on success, replanning on failure); a maximum step limit prevents infinite loops.
    - Design Motivation: Centralized orchestration ensures global consistency in multi-step reasoning; the replanning mechanism improves robustness.

### Loss & Training
The SLU module fine-tunes BERT using cross-entropy loss. LLM agent components are guided via in-context learning (5-shot) without additional training. API results are pre-cached to ensure deterministic and cost-free execution.

## Key Experimental Results

### Main Results

| Model | Method | Type 1 Acc | Type 2 Acc | Type 3 Acc | Overall Acc |
|-------|--------|-----------|-----------|-----------|-------------|
| Qwen2.5-72B | Standard | 0.8082 | 0.7110 | 0.3973 | 0.5855 |
| Qwen2.5-72B | HIRE-Agent | 0.8862 | 0.6581 | **0.6211** | **0.6989** |
| Qwen3-30B A3B | Standard | 0.7394 | 0.5871 | 0.3512 | 0.5131 |
| Qwen3-30B A3B | HIRE-Agent | 0.7659 | **0.8645** | **0.8371** | **0.8260** |
| Average | Standard | 0.7741 | 0.6090 | 0.3559 | 0.5299 |
| Average | HIRE-Agent | **0.8453** | **0.7658** | **0.6535** | **0.7323** |

### Ablation Study (Bottleneck Analysis)

| Component | Key Metric | Description |
|-----------|-----------|-------------|
| w/o SLU → w/ SLU | Type 3: 0.2921 → 0.6535 | SLU module contributes most, +0.3614 gain |
| GT SQL labels | Qwen2.5-72B: +0.1407 | SQL generation is the primary bottleneck for this model |
| GT API labels | Qwen3-8B: +0.0660 | Tool invocation is the primary bottleneck for this model |
| All GT labels | Average accuracy only 0.8864 | Exposes a compositional gap in global planning and final reasoning |

### Key Findings
- The hierarchical architecture improves overall accuracy by an average of 20.24 percentage points, with the largest gain on Type 3 multi-step reasoning (+29.76 percentage points).
- Even when all intermediate ground-truth labels are provided, accuracy reaches only 0.8864, indicating a "compositional gap"—models cannot perfectly integrate the results of multiple sub-tasks.
- Qwen3-30B exhibits an "overthinking" phenomenon: performance on simple questions degrades because strong reasoning capabilities interfere with direct execution of straightforward queries.
- Bottlenecks differ across models: SQL generation is the limiting factor for Qwen2.5-72B, while tool invocation limits Qwen3-8B, revealing heterogeneity in model capabilities.

## Highlights & Insights
- The **progressive bottleneck analysis methodology** is particularly valuable—systematically replacing individual components with ground-truth labels to isolate each module's performance contribution. This diagnostic approach is transferable to the evaluation of any multi-module system.
- The **API caching strategy** ensures benchmark reproducibility at zero cost—pre-storing real-time API call results as local database queries guarantees determinism while eliminating API expenses.
- The paper uncovers an **"overthinking" phenomenon** in large language models—strong reasoning capabilities become a liability on simple tasks, which carries important implications for agent design.

## Limitations & Future Work
- The dataset currently covers real estate scenarios in only eight Chinese cities, imposing notable geographic and cultural limitations.
- Questions are generated from 41 templates and, despite paraphrasing, may still exhibit repetitive patterns.
- The API caching strategy, while enhancing reproducibility, cannot capture the latency and error-handling characteristics of live API calls.
- Future work may extend the framework to additional vertical domains (healthcare, legal, finance) to validate its generalizability.

## Related Work & Insights
- **vs. Spider**: Spider evaluates only Text-to-SQL and does not involve external API calls or multi-source fusion; ReCoQA requires tight coupling between database and API operations.
- **vs. RETQA**: RETQA introduces SLU but is confined to static databases and does not support dynamic geographic information queries.
- **vs. MACT**: MACT demonstrates complex agent collaboration but relies on Pandas for data manipulation, which introduces memory and scalability concerns; HIRE-Agent's SQL + API combination is more scalable.

## Rating
- Novelty: ⭐⭐⭐⭐ First benchmark to systematically evaluate hybrid database–API reasoning in a vertical domain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four LLMs, multi-level ablations, bottleneck analysis, and real-world scenario testing—exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with in-depth analysis.
- Value: ⭐⭐⭐⭐ The dataset and analytical methodology offer significant reference value for the agent research community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](../../CVPR2026/llm_reasoning/step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_temporal_reasoning.md)
- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](../../ICLR2026/llm_reasoning/agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)

<!-- RELATED:END -->
