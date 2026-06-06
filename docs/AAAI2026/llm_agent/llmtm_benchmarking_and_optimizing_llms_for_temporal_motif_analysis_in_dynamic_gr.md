---
title: >-
  [Paper Note] LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs
description: >-
  [AAAI 2026][LLM Agent][Temporal graphs] This paper proposes LLMTM — the first comprehensive benchmark for evaluating LLMs on temporal motif analysis in dynamic graphs. It covers 6 task categories across 9 temporal motif…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "Temporal graphs"
  - "temporal motifs"
  - "dynamic graphs"
  - "structure-aware routing"
  - "LLM benchmark"
date: 2026-05-08
content_hash: 906eaeabd9a1faa8
---

# LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs

**Conference**: AAAI 2026  
**arXiv**: [2512.22266](https://arxiv.org/abs/2512.22266)  
**Code**: [https://github.com/Wjerry5/LLMTM](https://github.com/Wjerry5/LLMTM)  
**Area**: LLM Agent / Graph Analysis  
**Keywords**: Temporal graphs, temporal motifs, dynamic graphs, structure-aware routing, LLM benchmark

## TL;DR
This paper proposes LLMTM — the first comprehensive benchmark for evaluating LLMs on temporal motif analysis in dynamic graphs. It covers 6 task categories across 9 temporal motif types and evaluates 9 models, finding that LLM performance on temporal motif recognition degrades rapidly with increasing motif complexity. A Structure-Aware Dispatcher is further proposed to intelligently route queries to either standard LLM prompting or tool-augmented agents based on graph structural properties and cognitive load, achieving near-peak accuracy while reducing computational cost.

## Background & Motivation

**Background**: LLMs have been applied to various graph reasoning tasks (node classification, link prediction, etc.), but their application to temporal analysis of dynamic graphs remains largely unexplored. Temporal motifs — interaction patterns occurring in a specific order within a short time window — are fundamental building blocks of dynamic graphs and capture real-time anomalies and unique phenomena.

**Limitations of Prior Work**:
   - No benchmark specifically evaluates LLM capabilities on temporal motif analysis
   - It is unclear whether LLMs can understand the temporal ordering of events and structural patterns
   - No query routing strategy tailored to graph structural properties exists — applying heavy agents to simple queries is wasteful

**Key Challenge**: Temporal motif analysis requires simultaneous understanding of graph structure and temporal order, yet LLMs inherently process serialized text — effectively encoding spatiotemporal graph information into text remains a core challenge.

**Goal**: (a) Establish an LLM evaluation benchmark for temporal motifs; (b) Design an efficient query routing strategy.

**Key Insight**: Six task categories covering different aspects of motif analysis (recognition / counting / temporal reasoning / node role identification / evolution patterns / anomaly detection), combined with structure-aware intelligent routing.

**Core Idea**: A 6-task × 9-motif-type benchmark + Structure-Aware Dispatcher = efficient LLM-based analysis on temporal graphs.

## Method

### Overall Architecture
LLMTM benchmark construction: define 9 temporal motif types (various temporal variants of triangles, stars, and chains) → 6 analysis task categories (motif recognition / counting / time-window reasoning / node role identification / motif evolution trends / anomalous motif detection) → evaluate across 9 LLMs. Optimization: a Structure-Aware Dispatcher routes queries to the appropriate reasoning mode based on graph structural properties (density / degree distribution / temporal span) and query complexity.

### Key Designs

1. **6-Category Task Design**:

    - Motif Recognition: given a subgraph, determine whether it constitutes a specific motif
    - Motif Counting: count the number of instances of a given motif type in the graph
    - Time-Window Reasoning: determine the temporal ordering of events within a motif
    - Node Role Identification: identify the functional role of a node within a motif
    - Evolution Trends: analyze how motif frequency changes over time
    - Anomaly Detection: identify motif instances that deviate from normal patterns
    - Design Motivation: progressive difficulty from basic perception to complex reasoning

2. **Structure-Aware Dispatcher**:

    - Function: intelligently routes queries to the appropriate reasoning mode
    - Mechanism: analyzes graph structural features (node count, edge density, temporal span) and query complexity; simple queries → standard LLM prompting (low cost); complex queries → tool-augmented agents (high accuracy but expensive)
    - Design Motivation: not all temporal graph queries require heavyweight tools — standard prompting suffices for motif recognition on sparse, small graphs

3. **Graph-to-Text Serialization**:

    - Function: encode dynamic graphs into text processable by LLMs
    - Mechanism: chronologically sorted edge lists + node attribute descriptions + timestamp annotations
    - Design Motivation: preserves temporal information while aligning with LLMs' text processing conventions

### Loss & Training
- Benchmark evaluation only — no training required
- 9 LLMs evaluated include GPT-4o, Claude-3.5, Llama-3, among others

## Key Experimental Results

### Main Results

| Task | Best Model Accuracy | Difficulty Trend |
|------|---------------------|------------------|
| Motif Recognition | ~85% | Easiest |
| Motif Counting | ~60% | Moderate |
| Time-Window Reasoning | ~55% | Hard |
| Anomaly Detection | ~45% | Hardest |

### Ablation Study: Dispatcher Effectiveness

| Strategy | Accuracy | Computational Cost |
|----------|----------|--------------------|
| All standard prompting | Lower | Lowest |
| All agents | Highest | Highest |
| **Structure-aware routing** | **Near all-agent** | **Significantly lower than all-agent** |

### Key Findings
- **LLM understanding of temporal motifs degrades sharply with complexity**: recognition reaches ~85% but anomaly detection drops to only ~45%
- **Counting is a universal weakness**: all models perform worst on motif counting — enumeration is inherently difficult for LLMs
- **Structure-aware routing is effective**: achieves significant computational cost reduction with only a 1–3% accuracy drop
- **Graph density and size substantially affect performance**: LLMs perform well on sparse, small graphs but degrade sharply on dense, large graphs

## Highlights & Insights
- **The first LLM benchmark for temporal motifs** fills a gap in evaluation of dynamic graph analysis
- The **structure-aware routing** paradigm is generalizable to any graph LLM task — not all graph queries warrant the same level of processing overhead
- The progressive 6-category task design provides fine-grained diagnostics for analyzing LLM temporal reasoning capabilities

## Limitations & Future Work
- Evaluation is restricted to small-scale graphs due to LLM context length constraints
- Temporal motif types are limited to the 9 predefined variants, leaving more complex patterns uncovered
- Feature selection for the dispatcher may require domain knowledge tuning
- The tool interface design of the tool-augmented agent may limit generalizability

## Related Work & Insights
- **vs. NLGraph / GraphQA**: static graph benchmarks. LLMTM extends evaluation to dynamic temporal graphs
- **vs. traditional motif algorithms (SNAP)**: exact but lacking LLM interpretability. LLMTM provides an evaluation framework for LLMs
- Offers reference value for the design of agent systems on temporal graphs

## Rating
- Novelty: ⭐⭐⭐⭐ First temporal motif LLM benchmark + structure-aware routing
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 tasks × 9 motifs × 9 models
- Writing Quality: ⭐⭐⭐⭐ Systematic task taxonomy
- Value: ⭐⭐⭐ Contributes a benchmark for dynamic graph LLM analysis, though application scope is relatively narrow

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)
- [\[AAAI 2026\] Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis](promoting_sustainable_web_agents_benchmarking_and_estimating_energy_consumption_.md)
- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](../../ICLR2026/llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[AAAI 2026\] CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing](causaltrace_a_neurosymbolic_causal_analysis_agent_for_smart_manufacturing.md)
- [\[AAAI 2026\] COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis](coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)

</div>

<!-- RELATED:END -->
