---
title: >-
  [Paper Note] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering
description: >-
  [ACL 2026][LLM Evaluation][Tool-augmented reasoning] This paper constructs ReCoQA—a large-scale benchmark containing 29,270 real estate QA pairs—which requires models to integrate database queries and map API calls for hybrid multi-source reasoning. A hierarchical multi-agent framework, HIRE-Agent, is proposed as a strong baseline, systematically revealing the bottlenecks of existing LLMs in complex reasoning within vertical domains.
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Tool-augmented reasoning"
  - "Multi-step reasoning"
  - "Real estate QA"
  - "Multi-Agent framework"
  - "Benchmark dataset"
date: 2026-05-08
content_hash: bc1293193b017969
---

# ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering

**Conference**: ACL 2026  
**arXiv**: [2604.17944](https://arxiv.org/abs/2604.17944)  
**Code**: [https://github.com/Husky-989/ReCoQA](https://github.com/Husky-989/ReCoQA)  
**Area**: LLM Reasoning  
**Keywords**: Tool-augmented reasoning, Multi-step reasoning, Real estate QA, Multi-Agent framework, Benchmark dataset

## TL;DR
This paper constructs ReCoQA—a large-scale benchmark containing 29,270 real estate QA pairs—which requires models to integrate database queries and map API calls for hybrid multi-source reasoning. A hierarchical multi-agent framework, HIRE-Agent, is proposed as a strong baseline, systematically revealing the bottlenecks of existing LLMs in complex reasoning within vertical domains.

## Background & Motivation

**Background**: In real estate decision-making, users must switch between multiple platforms—comparing listings on one site, calculating commute times on map apps, and checking school district information on government websites. This fragmented information acquisition results in significant time costs and cognitive load. AI Agents offer a powerful solution, but existing QA benchmarks cannot effectively evaluate such hybrid reasoning capabilities.

**Limitations of Prior Work**: Classical datasets like Spider focus only on structured queries, while recent Agent benchmarks evaluate general tool usage; however, both treat database queries and external API calls as independent capabilities. Existing benchmarks fail to simulate hybrid workflows in practical scenarios—where the output of a database query (e.g., a list of candidate communities) serves as dynamic input for an API call (e.g., distance calculation). Additionally, most map QA datasets assume geolocation information is static, ignoring dynamic values like commute times.

**Key Challenge**: Real-world vertical domain decisions require tight coupling of heterogeneous information sources and multi-step reasoning, yet there is a lack of systematic benchmarks to evaluate this capability.

**Goal**: (1) Build an end-to-end benchmark covering static database queries, dynamic API calls, and multi-step reasoning; (2) Establish a hierarchical multi-agent baseline and systematically analyze the bottlenecks of each module.

**Key Insight**: Utilizing real estate purchasing consultation as the focal point—a vertical scenario naturally requiring the tight integration of database queries (listing attributes) and map APIs (commute distances and surrounding facilities).

**Core Idea**: Design a large-scale benchmark containing three progressive levels of problem difficulty (Simple Query, Joint Query, Multi-step Reasoning) and employ a hierarchical "Understand-Plan-Execute" Agent architecture as a strong baseline.

## Method

### Overall Architecture

ReCoQA delivers both an evaluation benchmark and a strong baseline. The benchmark side covers community, POI, and location pair data across 8 Chinese cities, unified in PostgreSQL with 4 categories of map APIs. Questions are organized into three difficulty levels: "Simple Query → Joint Query → Multi-step Reasoning." Each sample includes SLU labels, SQL statements, and full intermediate annotations of API call sequences. On the baseline side, HIRE-Agent decomposes "natural language query → structured understanding → task orchestration → tool execution → answer" into three layers: Front-end Agent, Supervisor Agent, and Expert Agent. The key lies in allowing database query outputs to be dynamically fed into map APIs, completing chain-of-thought reasoning that requires cross-source coupling.

```mermaid
graph TD
    Q["Natural Language Query<br/>(e.g., 'Communities in Tianhe with commute to Jimmall Plaza ≤ 30 mins')"] --> FE["Front-end Agent<br/>BERT Intent Detection + Slot Filling (SLU)"]
    FE --> SUP["Supervisor Agent<br/>CoT generates execution plan, dispatches sub-tasks step-by-step"]
    SUP -->|Dispatch sub-tasks (DB coordinates dynamically fed to Map API)| EXEC
    subgraph EXEC["Expert Agent (Tool Execution Layer)"]
        direction TB
        DB["Database Expert<br/>Table Caption Retrieval (BM25) → SQL Generation & Execution"]
        MAP["Map Reasoning Expert<br/>Tool Selection → Amap API Call → Reasoning"]
    end
    EXEC -->|Return evidence / Report error| SUP
    SUP -->|Insufficient evidence → Re-plan, terminate if max steps reached| SUP
    SUP -->|Sufficient evidence| ANS["Synthesize Natural Language Answer"]
```

### Key Designs

**1. Progressive Three-Tier Problem Types: Exposing the Reasoning Chain for Evaluation**
Type 1 (Simple Query) requires direct database queries only. Type 2 (Joint Query) necessitates coordination between the database and map APIs. Type 3 (Multi-step Reasoning) demands chained reasoning—first querying the database for coordinates, then calling APIs to calculate distances, and finally comparing results based on API outputs. Every sample is equipped with full intermediate step annotations (SLU labels, SQL statements, API call sequences). This progressive design enables precise localization of which link in the reasoning chain failed, upgrading evaluation from "answer correctness" to "process explainability."

**2. Front-end Agent: Compressing Natural Language into Intent and Slots**
This layer fine-tunes BERT for simultaneous intent detection (16 predefined intents) and slot filling (19 slot types, IOB tagging), parsing queries into structured representations of locations, transport modes, and time constraints. Decoupling intent understanding from execution is crucial: ablation shows that without SLU labels, Type 3 accuracy averages only 0.2921, jumping to 0.6535 with them, indicating that many errors stem from the query not being understood.

**3. Supervisor Agent: Orchestrating Multi-step Execution with CoT and Re-planning**
Upon receiving structured input, the Supervisor Agent generates an execution plan via Chain-of-Thought (CoT) prompting, dispatching sub-tasks to Expert Agents sequentially and adjusting dynamically based on feedback—continuing upon success or re-planning upon failure, with a maximum step limit to prevent infinite loops. Centralized orchestration ensures global consistency in multi-step reasoning, while the re-planning mechanism provides the system a second chance when a single step fails, which is the source of stability for long-chain Type 3 tasks.

**4. Expert Agent: Heterogeneous Tool Execution and Cross-source Coupling**
The Supervisor Agent dispatches sub-tasks to two types of experts. The Database Interactive Expert uses Table Caption Retrieval (TCR) to lock the correct table schema—it prompts the LLM to rewrite the query into an "Ideal Table Caption" summary, then uses BM25 to match real table captions. After localization, the SQL generation module (ICL 5-shot) synthesizes and executes the SQL, returning extracted location names for results containing coordinates. The Map Reasoning Expert selects tools, generates API parameters, and calls the Amap API, performing further reasoning (e.g., comparing distances) if necessary. The "dynamic feeding of database output to map API" emphasized in the framework is realized here: coordinates returned by the Database Expert are transferred by the Supervisor Agent to become inputs for the Map Expert's API call. Ablation localizes the bottleneck precisely: given GT SQL labels, Qwen2.5-72B improves by +0.1407; given GT API labels, Qwen3-8B improves by +0.0660, showing that SQL generation and tool calling are secondary failure points for different models.

### Key Designs: Comprehensive Example
For the Type 3 question "Which communities in Tianhe have a commute to Jimmall Plaza under 30 minutes?": The Front-end Agent first parses it into the intent "Commute Filter" and slots {District=Tianhe, Destination=Jimmall Plaza, Time Limit=30min}. The Supervisor Agent plans: ① Query the database for candidate communities and coordinates in Tianhe → ② Feed the coordinate list into the map API to calculate commute times → ③ Filter communities where time ≤ 30min. The database output here dynamically becomes the API input, finally synthesized into a natural language answer; if any link fails, the re-planning mechanism rolls back and retries.

### Loss & Training
The SLU module is fine-tuned using BERT with Cross-Entropy loss. The LLM Agent components use In-Context Learning (5-shot) guidance without additional training. API results are implemented via pre-caching for determinism and zero-cost execution.

## Key Experimental Results

### Main Results

| Model | Method | Type 1 Acc | Type 2 Acc | Type 3 Acc | Overall Acc |
|------|------|-----------|-----------|-----------|-------------|
| Qwen2.5-72B | Standard | 0.8082 | 0.7110 | 0.3973 | 0.5855 |
| Qwen2.5-72B | HIRE-Agent | 0.8862 | 0.6581 | **0.6211** | **0.6989** |
| Qwen3-30B A3B | Standard | 0.7394 | 0.5871 | 0.3512 | 0.5131 |
| Qwen3-30B A3B | HIRE-Agent | 0.7659 | **0.8645** | **0.8371** | **0.8260** |
| Average | Standard | 0.7741 | 0.6090 | 0.3559 | 0.5299 |
| Average | HIRE-Agent | **0.8453** | **0.7658** | **0.6535** | **0.7323** |

### Ablation Study

| Component | Key Metric | Description |
|------|---------|------|
| W/o SLU → W/ SLU | Type 3: 0.2921 → 0.6535 | SLU module contributes most, gain +0.3614 |
| GT SQL Labels | Qwen2.5-72B Gain +0.1407 | SQL generation is the primary bottleneck for this model |
| GT API Labels | Qwen3-8B Gain +0.0660 | Tool calling is the primary bottleneck for this model |
| All GT Labels | Avg Acc only 0.8864 | Reveals the "Synthesis Gap" in global planning and final reasoning |

### Key Findings
- The hierarchical architecture improves Overall accuracy by 20.24 percentage points on average, with the largest gain in Type 3 multi-step reasoning (+29.76 points).
- Even with all GT labels for intermediate steps, accuracy is only 0.8864, indicating a "Synthesis Gap"—models cannot perfectly integrate results from multiple sub-tasks.
- Qwen3-30B exhibited an "overthinking" phenomenon: performance on simple questions was worse (complex reasoning ability interfered with the direct execution of simple queries).
- Bottlenecks vary by model: Qwen2.5-72B is limited by SQL generation, whereas Qwen3-8B is limited by tool calling, revealing heterogeneity in model capabilities.

## Highlights & Insights
- **Progressive bottleneck analysis** is highly valuable—locating the performance contribution of each module by gradually replacing them with ground truth (GT) labels; this diagnostic method can be migrated to the evaluation of any multi-module system.
- **API caching strategy** achieves benchmark reproducibility and zero-cost usage—pre-storing real-time API call results as local database queries ensures determinism and eliminates API fees.
- Discovery of the LLM **"overthinking" phenomenon**—where strong reasoning capability becomes a burden in simple tasks, which has significant implications for Agent design.

## Limitations & Future Work
- The dataset currently only covers real estate scenarios in 8 Chinese cities, posing geographic and cultural limitations.
- Questions are generated based on 41 templates; although paraphrased, pattern repetition may still exist.
- While the API caching strategy improves reproducibility, it does not reflect the latency and error handling of real-time APIs.
- Future work could extend to more vertical domains (medical, legal, finance) to verify the framework's generality.

## Related Work & Insights
- **vs Spider**: Spider only evaluates Text-to-SQL without external API calls or multi-source integration; ReCoQA requires tight coupling between databases and APIs.
- **vs RETQA**: RETQA introduced SLU but is limited to static databases and does not support dynamic geographical information queries.
- **vs MACT**: MACT demonstrates complex Agent collaboration but relies on Pandas for data manipulation, presenting memory and scalability issues; HIRE-Agent’s combination of SQL + API is more scalable.

## Rating
- Novelty: ⭐⭐⭐⭐ First vertical benchmark to systematically evaluate hybrid DB-API reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed analysis with 4 LLMs, multi-layer ablation, and bottleneck diagnosis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and deep analysis.
- Value: ⭐⭐⭐⭐ The dataset and analysis methods are significant references for the Agent research community.

## Related Papers

- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain](diningbench_a_hierarchical_multi-view_benchmark_for_perception_and_reasoning_in_.md)
- [\[NeurIPS 2025\] DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](../../NeurIPS2025/llm_evaluation/dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)
- [\[ACL 2025\] YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](../../ACL2025/llm_evaluation/yescieval_llm_judge_science.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain](diningbench_a_hierarchical_multi-view_benchmark_for_perception_and_reasoning_in_.md)
- [\[NeurIPS 2025\] DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](../../NeurIPS2025/llm_evaluation/dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)
- [\[ACL 2025\] YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](../../ACL2025/llm_evaluation/yescieval_llm_judge_science.md)

</div>

<!-- RELATED:END -->
