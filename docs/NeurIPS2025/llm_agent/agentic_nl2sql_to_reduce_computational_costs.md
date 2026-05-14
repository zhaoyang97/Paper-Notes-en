---
title: >-
  [Paper Note] Agentic NL2SQL to Reduce Computational Costs
description: >-
  [NeurIPS 2025][LLM Agent][NL2SQL] This paper proposes Datalake Agent, an agentic NL2SQL system built on an interactive reasoning loop. Through a hierarchical information retrieval strategy (GetDBDescription → GetTables →…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "NL2SQL"
  - "text-to-SQL"
  - "agentic reasoning"
  - "token reduction"
  - "datalake"
date: 2026-05-08
content_hash: dbe1870191abe9a2
---

# Agentic NL2SQL to Reduce Computational Costs

**Conference**: NeurIPS 2025
**arXiv**: [2510.14808](https://arxiv.org/abs/2510.14808)
**Code**: None
**Area**: Agent
**Keywords**: NL2SQL, text-to-SQL, agentic reasoning, token reduction, datalake

## TL;DR
This paper proposes Datalake Agent, an agentic NL2SQL system built on an interactive reasoning loop. Through a hierarchical information retrieval strategy (GetDBDescription → GetTables → GetColumns → DBQueryFinalSQL), the system enables LLMs to request database schema information on demand rather than receiving it all at once. In a setting with 319 tables, the approach reduces token usage by 87% and cost by 8×, while maintaining superior performance on complex queries.

## Background & Motivation

**Background**: LLMs have demonstrated strong performance in NL2SQL tasks; however, deploying them on enterprise-scale database collections requires injecting schema information from all databases into the prompt, resulting in extremely long prompts and high operational costs.

**Limitations of Prior Work**: (1) Research shows that GPT models degrade to near-random guessing on tabular data when prompts exceed 1,000 tokens; (2) most NL2SQL queries only require a subset of the available schemas, making the majority of meta-information redundant; (3) existing benchmarks (Bird, Spider) provide only relevant schemas, failing to reflect real-world enterprise scenarios.

**Key Challenge**: Providing all schema information to the LLM upfront is both costly and performance-degrading, yet the LLM has no prior knowledge of which databases or tables are relevant.

**Goal**: How can LLMs efficiently retrieve only the necessary schema information when operating over large-scale database collections?

**Key Insight**: Design a hierarchical set of information-retrieval commands (database description → table enumeration → column details) that allow the LLM to progressively narrow its scope within an interactive loop.

**Core Idea**: Replace one-shot schema dumping with an agentic interaction loop, enabling the LLM to act as an investigator that incrementally acquires, refines, and localizes the information it needs.

## Method

### Overall Architecture
User query → LLM enters reasoning loop → incrementally retrieves schema information via predefined commands (GetDBDescription / GetTables / GetColumns) → backtracks to coarser granularity when necessary → constructs SQL query → executes via DBQueryFinalSQL.

### Key Designs

1. **Hierarchical Information Retrieval**:

    - **Function**: Provides four commands that allow the LLM to retrieve schema information at different levels of granularity on demand.
    - **Mechanism**: GetDBDescription (retrieves high-level database summaries) → GetTables (enumerates tables within a specific database) → GetColumns (exposes column-level metadata including names and types) → DBQueryFinalSQL (executes the final SQL query). The hierarchy proceeds from coarse to fine, and the LLM may backtrack to a coarser level at any point.
    - **Design Motivation**: Mirrors the workflow of a human data analyst — first surveying available databases, then examining table structures, and finally focusing on specific columns.

2. **Iterative Refinement**:

    - **Function**: Allows the LLM to flexibly adjust its retrieval strategy throughout the information-gathering process.
    - **Mechanism**: The LLM can revert to a coarser granularity at any point and then re-refine. This feedback-driven loop enables autonomous reasoning over complex database structures.
    - **Design Motivation**: NL2SQL is not a linear process — the LLM may need to explore multiple databases before identifying the correct tables.

### Loss & Training
No training is required. The system uses GPT-4-mini with temperature = 0.1. A benchmark of 100 Table QA tasks spanning 23 databases was manually constructed for evaluation.

## Key Experimental Results

### Main Results

| Setting | Direct Solver Tokens | Datalake Agent Tokens | Agent Savings |
|---------|---------------------|-----------------------|---------------|
| 42 tables | 7,407 | 3,670 | 50% |
| 159 tables | ~18,000 | ~3,900 | ~78% |
| 319 tables | 34,602 | 4,264 | **87%** |

### Ablation Study / Cost Analysis

| Model | Method | Cost per 1,000 Tasks (319 tables) |
|-------|--------|----------------------------------|
| GPT-4-mini | Direct Solver | ~$100 |
| GPT-4-mini | Datalake Agent | ~$12 |
| o1 | Direct Solver | >$500 |
| o1 | Datalake Agent | ~$55 |

### Key Findings
- **Token usage scales nearly independently of table count**: Datalake Agent grows only 16% (from 3,670 to 4,264 tokens) as tables increase from 42 to 319, whereas Direct Solver grows 4.7×.
- **Greater advantage on complex queries**: Direct Solver is marginally better on simple single-table queries, but Datalake Agent outperforms significantly on multi-table join queries.
- **Cost gap amplifies with scale**: At 319 tables, the cost difference reaches 8×; for more expensive models such as o1, the gap exceeds $450 per 1,000 tasks.
- **Limitation — potential for infinite reasoning loops**: The LLM occasionally fails to locate the correct tables, leading to repetitive retrieval requests.

## Highlights & Insights
- **Decoupling token usage from database scale** is the core contribution: regardless of whether there are 42 or 319 tables, the agent retrieves only the information it requires — a property of critical importance in enterprise-scale data lake scenarios.
- **Simple yet effective**: No training is required; significant cost savings are achieved purely through agentic interaction design.
- **Closer to real enterprise scenarios**: Users do not know which database contains the relevant data — a more realistic setting than standard NL2SQL benchmarks that provide pre-filtered schemas.

## Limitations & Future Work
- **Evaluated with GPT-4-mini only**: Effectiveness on stronger or weaker models remains unverified.
- **Infinite loop problem**: Early-stopping or fallback mechanisms need to be incorporated.
- **Small benchmark scale**: Only 100 tasks are used for evaluation.
- **Simulated databases**: 18 of the databases are simulated (schema only, no data), which does not fully represent real-world conditions.

## Related Work & Insights
- **vs. Direct NL2SQL**: Direct approaches inject all schemas into the prompt, causing both cost and performance to degrade as scale increases.
- **vs. Text2API**: Text2API retrieves data through API endpoints but is constrained by API design; NL2SQL offers greater flexibility.
- **vs. Spider/Bird evaluation paradigm**: These benchmarks supply only relevant schemas; the Datalake Agent setting more closely reflects real enterprise deployments.

## Rating
- Novelty: ⭐⭐⭐ — Intuitive concept, though not particularly novel (agentic information retrieval)
- Experimental Thoroughness: ⭐⭐⭐ — Limited to a single model, 100 tasks, and partially simulated data
- Writing Quality: ⭐⭐⭐⭐ — Clear and concise
- Value: ⭐⭐⭐⭐ — Highly practical for enterprise NL2SQL scenarios with significant cost savings

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[NeurIPS 2025\] CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension](cam_a_constructivist_view_of_agentic_memory_for_llm-based_reading_comprehension.md)
- [\[NeurIPS 2025\] Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading](orchestration_framework_for_financial_agents_from_algorithmic_trading_to_agentic.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)
- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)

</div>

<!-- RELATED:END -->
