---
title: >-
  [Paper Note] Which Bird Does Not Have Wings: Negative-Constrained KGQA with Schema-Guided Semantic Matching and Self-Directed Refinement
description: >-
  [ACL 2026][Graph Learning][Knowledge Graph QA] This paper defines the NEST KGQA task and NestKGQA dataset for negation-constrained knowledge graph QA…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Knowledge Graph QA"
  - "Negation Constraint"
  - "Semantic Parsing"
  - "Logical Form"
  - "Schema-Guided"
content_hash: 19e07d32d6165793
---

# Which Bird Does Not Have Wings: Negative-Constrained KGQA with Schema-Guided Semantic Matching and Self-Directed Refinement

**Conference**: ACL 2026
**arXiv**: [2604.14749](https://arxiv.org/abs/2604.14749)  
**Code**: [https://github.com/midannii/CUCKOO](https://github.com/midannii/CUCKOO)  
**Area**: Graph Learning / Knowledge Graph QA
**Keywords**: Knowledge Graph QA, Negation Constraint, Semantic Parsing, Logical Form, Schema-Guided

## TL;DR
This paper defines the NEST KGQA task and NestKGQA dataset for negation-constrained knowledge graph QA, designs PyLF (Python-format logical form) for clear negation expression, and proposes CUCKOO framework with constraint-aware draft generation, schema-guided semantic matching, and self-directed refinement, achieving efficient and precise answers for multi-constraint questions in few-shot settings.

## Method

### Key Designs

1. **PyLF (Python-Format Logical Form)**: Adds a boolean `neg` parameter in JOIN functions to mark negation constraints (e.g., `JOIN('producing', 'Saturn', neg=True)`), leveraging LLMs' familiarity with Python syntax.

2. **Schema-Guided Semantic Matching**: Starting from the subject entity, retrieves candidate entities and their classes, then uses schema-level triples with similarity thresholds to prune invalid combinations, reducing exponential search space to polynomial.

3. **Self-Directed Refinement Module**: Triggered only when query execution returns empty results, diagnosing error types and re-generating drafts without external execution feedback.

## Key Experimental Results

| Dataset | CUCKOO(6) | KB-Coder(6) | KB-BINDER(6) |
|---------|-----------|-------------|--------------|
| GrailQA (Overall) F1 | **64.2** | 56.3 | 54.5 |
| NestKGQA F1 | **26.2** | 24.4 | 4.6 |

## Highlights & Insights
- PyLF's minimal modification approach — adding `neg` boolean to existing JOIN functions — elegantly solves the long-standing negation expression challenge
- Schema-guided matching reduces exponential search space to polynomial through type system pruning

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation](../../AAAI2026/graph_learning/notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)
- [\[ICLR 2026\] Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](../../ICLR2026/graph_learning/pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](../../AAAI2026/graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](../../AAAI2026/graph_learning/self-adaptive_graph_mixture_of_models.md)
- [\[AAAI 2026\] Kernelized Edge Attention: Addressing Semantic Attention Blurring in Temporal Graph Neural Networks](../../AAAI2026/graph_learning/kernelized_edge_attention_addressing_semantic_attention_blurring_in_temporal_gra.md)

</div>

<!-- RELATED:END -->
