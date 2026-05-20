---
title: >-
  [Paper Note] Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents
description: >-
  [NeurIPS 2025][LLM Agent][Agent caching] This paper proposes Agentic Plan Caching (APC), which extracts structured plan templates from agent execution logs and reuses them via keyword-matching cache hits with a small mod…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Agent caching"
  - "plan templates"
  - "test-time memory"
  - "inference cost optimization"
  - "Plan-Act paradigm"
date: 2026-05-08
content_hash: df0cd4e915a8c9d5
---

# Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents

**Conference**: NeurIPS 2025
**arXiv**: [2506.14852](https://arxiv.org/abs/2506.14852)  
**Code**: None  
**Area**: LLM Agent / LLM Efficiency
**Keywords**: Agent caching, plan templates, test-time memory, inference cost optimization, Plan-Act paradigm

## TL;DR
This paper proposes Agentic Plan Caching (APC), which extracts structured plan templates from agent execution logs and reuses them via keyword-matching cache hits with a small model for adaptation. APC reduces cost by 50.31% and latency by 27.28% on average while retaining 96.61% of accuracy-optimal performance.

## Background & Motivation

**Background**: LLM-based agents adopt a Plan-Act loop in which a large model handles planning while an execution model carries out actions. The planning phase (chain-of-thought reasoning, task decomposition) consumes substantial tokens and expensive API calls.

**Limitations of Prior Work**: (a) **Context caching** (KV cache reuse): model-specific and non-transferable across models, making it unsuitable for multi-model agents; (b) **Semantic caching**: stores (input, output) pairs, but agent outputs depend on external data/environments, so the same query over different data yields different outputs; (c) Neither approach can handle minor query variations adaptively.

**Key Challenge**: Agent planning patterns are semantically highly repetitive (e.g., "analyze dataset" → similar decomposition plans), yet concrete details vary with data and environment. Existing caches cannot decouple "core intent" from "dynamic context."

**Key Insight**: Extract reusable **plan templates** (with context-specific information removed) from completed agent executions, and adapt these templates to new tasks using a small model.

**Core Idea**: Use agent-level plans (rather than query-level responses) as the caching unit. Keyword matching combined with small-model adaptation achieves large-model plan quality at small-model cost.

## Method

### Overall Architecture
Input task query → small model extracts keywords (high-level intent) → exact keyword matching against cache → **Cache hit**: small model adapts template into a concrete plan → **Cache miss**: large model generates a new plan → Actor LM executes → upon completion, extract template and store in cache.

### Key Designs

1. **Plan Template Generation**:

    - Function: Extract reusable structured templates from completed agent execution logs.
    - Mechanism: Two-stage filtering — (a) a rule-based filter extracts key information and removes redundant reasoning steps; (b) a lightweight LLM filter removes context-specific elements (company names, numeric values, etc.) to produce generalized templates.
    - Design Motivation: Raw execution logs are excessively long and contain irrelevant details that small models cannot directly exploit.

2. **Keyword Extraction & Matching**:

    - Function: Use keywords (high-level intent) instead of query similarity for cache lookup.
    - Mechanism: GPT-4o-mini extracts core intent keywords from a query (e.g., "mean calculation"), which are then matched exactly against keywords stored in the cache.
    - Design Motivation: Experiments demonstrate that **query-level semantic similarity is unsuitable for agent caching** — it over-focuses on contextual details (company names, etc.) rather than core intent, leading to high false-positive and false-negative rates. Keywords better capture task intent.
    - Exact rather than fuzzy matching is used to avoid threshold tuning.

3. **Plan Adaptation**:

    - Function: Use a small model to adapt a cached general template into a concrete plan with task-specific context.
    - Mechanism: Upon a cache hit, a small planner LM (e.g., LLaMA-3.1-8B) receives the template plus the current task context and generates an adapted concrete plan.
    - Design Motivation: The template already encodes correct task decomposition logic; the small model only needs to fill in specific details and does not require expensive reasoning capabilities.

4. **Comparison with Full-History Caching**:

    - Full-History Caching (caching complete execution logs as in-context examples) performs poorly — small models struggle to process long, unfiltered execution logs.
    - APC's structured templates are more compact and easier to adapt.

### System Architecture
- Based on the Minion architecture (Plan-Act loop)
- Large planner LM: GPT-4o (used only on cache miss)
- Small planner LM / Actor LM: LLaMA-3.1-8B (used on cache hit)
- Keyword/template extraction: GPT-4o-mini (overhead accounts for only 1.04% of total cost)

## Key Experimental Results

### Main Results (5 Agent Workloads)

| Method | Avg. Cost | Avg. Latency | Accuracy Retention |
|---|---|---|---|
| Accuracy-Optimal (large model always) | 100% | 100% | 100% |
| Cost-Optimal (small model always) | Lowest | Lowest | Low |
| Semantic Caching (80%) | Reduced | Reduced | Unstable |
| Full-History Caching | Reduced | Reduced | Lower |
| **APC** | **−50.31%** | **−27.28%** | **96.61%** |

### Per-Dataset Details

| Dataset | Task Type | Cost Reduction | Accuracy Retention |
|---|---|---|---|
| FinanceBench | Financial data reasoning | ~50% | ~97% |
| QASPER | Long-context reasoning | ~45% | ~96% |
| TabMWP | Mathematical reasoning | ~55% | ~97% |
| AIME 2024/2025 | Competition mathematics | ~40% | ~95% |
| GAIA | Multi-step tool use | ~60% | ~97% |

### Key Findings
- **Semantic caching is unsuitable for agents**: Query similarity fails to capture task intent, yielding high false-positive and false-negative rates across different thresholds.
- **Full-history caching performs poorly**: Small models cannot effectively process long, noisy execution logs.
- **Keywords outperform query similarity**: Exact keyword matching achieves higher precision than semantic similarity at all threshold values.
- **Caching overhead is negligible**: Keyword extraction and template generation account for only 1.04% of total cost.
- **Compatible with existing serving frameworks**: APC can be used alongside vLLM, SGLang, and similar systems.

## Highlights & Insights
- **Paradigm shift from query-level to task-level caching**: Existing caches are designed for chatbots (caching QA pairs), whereas APC is designed for agents (caching plan templates) — this elevation in abstraction level constitutes the core contribution.
- **Empirical analysis of keywords vs. semantic similarity**: The comparison in Figure 3 is compelling — keyword extraction achieves lower false-positive and false-negative rates across all thresholds.
- **Decoupling intent from context**: Separating general planning logic (intent) from specific data (context) enables effective adaptation by small models.
- **Test-time learning**: APC functions as test-time memory — agents automatically accumulate experience during inference without any additional training.

## Limitations & Future Work
- **Exact keyword matching**: May miss semantically similar queries phrased differently; fuzzy matching is left to future work.
- **Dependence on cache correctness**: Only successfully completed plans are cached; inaccurate success detection could introduce erroneous templates.
- **Template generalization**: Adaptation may fail when new tasks differ substantially from cached templates.
- **Future directions**: (1) Hierarchical keywords for multi-granularity matching; (2) Incremental refinement of plan templates; (3) Evaluation across diverse agent architectures.

## Related Work & Insights
- **vs. GPTCache (semantic caching)**: GPTCache stores (query, response) pairs, which is unsuitable for agent outputs that depend on dynamic data.
- **vs. MemGPT**: MemGPT uses memory to enhance agent capabilities (reducing hallucinations), whereas APC uses memory to reduce cost — the objectives are fundamentally different.
- **vs. Case-Based Planning**: Classical CBP relies on manually designed symbolic plans, whereas APC automatically extracts templates from LLM-generated outputs.
- **vs. Context Caching**: KV caching stores internal model states, while APC caches task-level plans — the two operate at different levels of abstraction.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of plan template caching is novel and practical, representing a paradigm shift from chatbot caching to agent caching.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five diverse workloads, comparisons against multiple caching baselines, and detailed overhead analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation analysis (query vs. keyword) is highly convincing; system design is described with clarity.
- Value: ⭐⭐⭐⭐⭐ A 50% cost reduction has substantial practical value for agent deployment, and strong compatibility with existing frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [\[NeurIPS 2025\] CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension](cam_a_constructivist_view_of_agentic_memory_for_llm-based_reading_comprehension.md)
- [\[NeurIPS 2025\] Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading](orchestration_framework_for_financial_agents_from_algorithmic_trading_to_agentic.md)
- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)

</div>

<!-- RELATED:END -->
