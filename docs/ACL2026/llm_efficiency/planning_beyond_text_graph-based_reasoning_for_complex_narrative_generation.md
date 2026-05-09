---
title: >-
  [Paper Note] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation
description: >-
  [ACL 2026][LLM Efficiency][Narrative Generation] PLOTTER shifts narrative planning from text to graph structure (event graph + character graph) with multi-agent Evaluate-Plan-Revise iterative cycles for causal-level diagnosis and repair.
tags:
  - ACL 2026
  - LLM Efficiency
  - Narrative Generation
  - Graph-Based Reasoning
  - Event Graph
  - Character Graph
  - Multi-Agent Optimization
content_hash: 64e372abbc2c3a79
---

# Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation

**Conference**: ACL 2026
**arXiv**: [2604.21253](https://arxiv.org/abs/2604.21253)
**Code**: N/A
**Area**: LLM Efficiency
**Keywords**: Narrative Generation, Graph-Based Reasoning, Event Graph, Character Graph, Multi-Agent Iterative Optimization

## TL;DR
PLOTTER shifts narrative planning from text representation to graph structure (event graph + character graph), diagnosing and repairing narrative flaws through multi-agent Evaluate-Plan-Revise iterative cycles on graph topology, significantly outperforming existing methods on narrativity, characterization, and dramatic tension.

## Method

### Key Designs

1. **Dual-Graph Narrative Representation**: Event graph $G_e$ with narrative relation labels $\rho(e) \in \{\text{Causal}, \text{Foreshadowing}, \text{Suspense}\}$; character graph $G_c$ with multi-dimensional attributes and evolving relationships.

2. **Multi-Agent Panel + Constrained Graph Editor**: Theme Critic → Character Critic → Plot Critic, with cross-agent verification. Symbolic constraints: (1) causal rationality $\mathcal{K}_C$ — causal subgraph must maintain DAG; (2) narrative completeness $\mathcal{K}_N$ — all nodes reachable from start to end.

3. **Graph-Guided Progressive Script Synthesis**: Deterministic DFS serialization preserving causal topology, state-aware scene generation.

## Key Experimental Results

| Dimension | vs LLM-Plan-Write | vs Dramatron | vs DOC |
|-----------|-------------------|-------------|--------|
| Narrative | 72% | 74% | 92% |
| Characterization | 100% | 76% | 92% |

Three evaluation agents exhibit strong synergy — +29% storyline, +34% script when combined vs individual.

## Highlights & Insights
- Paradigm shift from text to graph for narrative planning — making causal reasoning, foreshadowing relationships, and character dynamics editable symbolic objects
- DAG and connectivity constraints provide deterministic verification independent of LLM reliability

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](../../AAAI2026/llm_efficiency/the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)
- [\[NeurIPS 2025\] On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](../../NeurIPS2025/llm_efficiency/on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)
- [\[AAAI 2026\] InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE](../../AAAI2026/llm_efficiency/intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora.md)
- [\[NeurIPS 2025\] Unmasking COVID-19 Vulnerability in Nigeria: Mapping Risks Beyond Urban Hotspots](../../NeurIPS2025/llm_efficiency/unmasking_covid-19_vulnerability_in_nigeria_mapping_risks_beyond_urban_hotspots.md)
- [\[NeurIPS 2025\] L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](../../NeurIPS2025/llm_efficiency/l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)

<!-- RELATED:END -->
