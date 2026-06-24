---
title: >-
  [Paper Note] LocAgent: Graph-Guided LLM Agents for Code Localization
description: >-
  [LLM Agent][Code localization] LocAgent parses a codebase into a directed heterogeneous graph (encompassing four relationships: contain/import/invoke/inherit) and designs unified tools (SearchEntity/TraverseGraph/RetrieveEntity) to guide the LLM Agent in multi-hop reasoning. This achieves high-accuracy code localization, reaching a 92.7% accuracy rate at the file level while reducing costs by 86% through fine-tuning open-source models.
tags:
  - "LLM Agent"
  - "Code localization"
  - "graph representation"
  - "software maintenance"
  - "code search"
date: 2026-05-08
content_hash: 76b0a3d6e65962b1
---

# LocAgent: Graph-Guided LLM Agents for Code Localization

| Conference | Area | arXiv | Code |
|------|------|-------|------|
| ACL2025 | LLM Agent / Software Engineering | [2503.09089](https://arxiv.org/abs/2503.09089) | [GitHub](https://github.com/gersteinlab/LocAgent) |

**Keywords**: Code localization, graph representation, LLM Agent, software maintenance, code search

## TL;DR

LocAgent parses a codebase into a directed heterogeneous graph (encompassing four relationships: contain/import/invoke/inherit) and designs unified tools (SearchEntity/TraverseGraph/RetrieveEntity) to guide the LLM Agent in multi-hop reasoning. This achieves high-accuracy code localization, reaching a 92.7% accuracy rate at the file level while reducing costs by 86% through fine-tuning open-source models.

## Background & Motivation

Code localization is a fundamental task in software maintenance, aiming to accurately locate code snippets that require modifications within a codebase based on natural language problem descriptions (such as GitHub Issues). Developers spend up to 66% of debugging time on program comprehension, and automated tools face similar challenges.

Existing methods face three limitations:

**Dense retrieval methods** (e.g., vector embeddings) require maintaining and continuously updating the vector representations of the entire codebase, which incurs high engineering overhead for large and rapidly evolving repositories.

**Large context window LLMs** cannot process the entire codebase at once and must strategically navigate to relevant components.

**Existing Agent approaches** primarily navigate via directory traversal, failing to understand semantic relationships and struggling to perform multi-hop reasoning when facing dense cross-file dependencies.

Core Problem: Issue descriptions often mention symptoms rather than the root cause. For example, "XSS vulnerability in user profile" might require modifying a shared validation utility function that is not explicitly mentioned in the Issue. This implicit mapping between the Issue description and the actual code is difficult for traditional retrieval methods to handle.

## Method

### Overall Architecture

LocAgent consists of three core components:

1. **Graph Construction & Indexing** (Offline): Parses the codebase into a directed heterogeneous graph and constructs a sparse index.
2. **Agent-Guided Search** (Online): The agent autonomously explores and localizes on the graph using unified tools.
3. **Open-Source Model Fine-Tuning**: Reduces costs via trajectory distillation.

### Graph Construction (Graph-based Code Representation)

Construct a directed heterogeneous graph $\mathcal{G}(\mathcal{V}, \mathcal{E}, \mathcal{A}, \mathcal{R})$:

- **Node types** $\mathcal{A}$: directory, file, class, function
- **Edge types** $\mathcal{R}$: contain, import, invoke, inherit

Construction process:
1. All directories and Python files serve as nodes.
2. Recursively parse each file using Abstract Syntax Trees (AST) to extract internal functions and classes as nodes.
3. The function level is the finest node granularity, and the function code content serves as the document for retrieval.
4. The `contain` edges form a tree structure, while `import`/`invoke`/`inherit` edges capture cross-file dependencies.

### Sparse Hierarchical Entity Indexing

Construct a four-level hierarchical index for graph nodes:
1. **Entity ID Index**: Uses fully qualified names (e.g., `src/utils.py:MathUtils.calculate_sum`).
2. **Global Name Dictionary**: Maps entity names to all nodes with duplicate names.
3. **BM25 Inverted Index for Entity IDs**: Handles fuzzy matching.
4. **Code Block Inverted Index**: Covers all potential match scenarios (e.g., global variables).

### Three Unified Tools

| Tool Name | Input | Output |
|--------|------|------|
| SearchEntity | Keywords | Related entities and code snippets (three levels of verbosity: fold/preview/full) |
| TraverseGraph | Source entity ID, direction, steps, entity type, relation type | Traversed subgraph (entities and relations) |
| RetrieveEntity | Entity ID | Full code, file path, line numbers |

Key designs of **TraverseGraph**:
- Supports type-aware BFS search.
- The Agent can select entity types and relationship types, corresponding to generating a meta-path over the heterogeneous graph.
- Outputs utilize an expanded tree format to encode topological relationships via spatial distance.

### Chain-of-Thought Agent Planning

The Agent executes in sequential steps:
1. **Keyword Extraction**: Extracts keywords of different classes from the Issue.
2. **Keyword Linking**: Links to code entities via `SearchEntity`.
3. **Logical Flow Generation**: Identifies entry points, recursively tracing call chains using `TraverseGraph` and `RetrieveEntity`.
4. **Target Entity Localization**: Locates and ranks all suspect code entities based on the logical flow.

### Consistency Confidence Estimation

Uses Reciprocal Rank as the initial confidence score, aggregating it over multiple iterations to obtain the final confidence; highly consistent locations are more likely to be relevant.

### Open-Source Model Fine-Tuning

- Collects 433 successful trajectories from Claude-3.5 and 335 successful trajectories from the fine-tuned Qwen2.5-32B.
- Performs SFT using LoRA.
- Self-improvement loop: Generates new trajectories using the fine-tuned model and filters successful ones for further training.
- Distills the knowledge into a 7B model.

## Key Experimental Results

### New Loc-Bench Benchmark

Addressing the limitations of SWE-Bench (risk of data leakage, category imbalance—85% being bug reports), Loc-Bench is proposed:
- 560 instances covering Bug Reports (242), Feature Requests (150), Security Issues (29), and Performance Issues (139).
- Collects GitHub Issues from after October 2024 to mitigate pre-training data leakage.

### Main Results (SWE-Bench-Lite)

| Method | File Acc@5 | Module Acc@10 | Function Acc@10 |
|------|-----------|------------|------------|
| BM25 | 61.68 | 52.92 | 36.86 |
| CodeRankEmbed | 84.67 | 78.83 | 58.76 |
| Agentless (Claude-3.5) | 79.56 | 68.98 | 58.76 |
| OpenHands (Claude-3.5) | 90.15 | 83.58 | 70.07 |
| SWE-agent (Claude-3.5) | 90.15 | 78.10 | 64.60 |
| **LocAgent (Qwen2.5-32B-ft)** | **92.70** | **87.23** | **77.01** |
| **LocAgent (Claude-3.5)** | **94.16** | **87.59** | **77.37** |

LocAgent consistently leads across all levels.

### Efficiency Analysis

| Method | Model | Cost per Instance | Acc@10/Cost |
|------|------|---------|-------------|
| OpenHands | Claude-3.5 | $0.79 | 0.9 |
| LocAgent | Qwen2.5-7B(ft) | **$0.05** | **13.2** |
| LocAgent | Qwen2.5-32B(ft) | $0.09 | 8.6 |
| LocAgent | Claude-3.5 | $0.66 | 1.2 |

The fine-tuned models achieve an 86% cost reduction.

### Ablation Study

- Removing `TraverseGraph`: Function-level accuracy drops from 71.53% to 66.06%.
- Keeping only the `contain` relationship: 66.42%, indicating that `import`/`invoke`/`inherit` are crucial.
- Restricting hops to 1: 66.79%, showcasing that multi-hop exploration is necessary for deep understanding.
- Removing `SearchEntity`: Drops to 53.28%, confirming that the sparse index is a core contribution.

### Impact on Downstream Tasks

Superior localization directly improves the resolution rate of GitHub Issues:
- Agentless (Claude-3.5) Pass@10: 33.58%
- LocAgent (Claude-3.5) Pass@10: **37.59%** (+12% Gain)

## Highlights & Insights

1. **Core Value of Graph Representation**: Capturing cross-file dependencies via `import`/`invoke`/`inherit` relationships renders modules that are physically distant but logically close as "neighbors" in the graph, which is impossible using traditional directory traversals.
2. **Tool Design Philosophy**: Consolidating all operations into three unified tools instead of numerous fragmented tools is more suited for LLM Agents.
3. **Lightweight and Efficient Indexing**: The indexing process takes only a few seconds and does not require maintaining a vector database, showing strong practicality.
4. **Great Balance of Cost vs. Performance**: Fine-tuning the 7B model costs only $0.05 per instance, yet its performance surpasses most approaches utilizing GPT-4o.

## Limitations & Future Work

1. Focuses solely on Python codebases without validating on other programming languages.
2. Fine-tuning relies heavily on successful trajectories generated by Claude-3.5, indicating a limited data source.
3. Evaluation metrics primary focus on accuracy, lacking fine-grained measurement of localization quality.
4. Although Loc-Bench covers multiple categories, security-related samples remain scarce (29 instances).

## Related Work & Insights

- **Traditional Retrieval Methods**: BM25, E5, Jina-Code, etc., which suffer from high maintenance overhead and lack structural understanding.
- **LLM-based Generative Approaches**: Hierarchical localization in Agentless, and the ACI interface in SWE-Agent.
- **Graph-based Methods**: CodexGraph (Neo4j+Cypher), RepoGraph (subgraph retrieval), RepoUnderstander (MCTS search), and OrcaLoca (priority scheduling). LocAgent's graph representation is the most comprehensive, simultaneously factoring in four node types and four relationship types.

## Rating

⭐⭐⭐⭐ (4/5)

This work addresses a core pain point in code localization—cross-file dependency tracking. The graph representation and unified tool designs are elegant and effective. The logic of fine-tuning open-source models to reduce costs is highly practical. Regrettably, it is limited to Python, and Loc-Bench contains too few samples in the security and performance categories.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GPS: Graph-guided Proactive Information Seeking in Large Language Models](../../ICLR2026/llm_agent/gps_graph-guided_proactive_information_seeking_in_large_language_models.md)
- [\[NeurIPS 2025\] The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](../../NeurIPS2025/llm_agent/the_lighthouse_of_language_enhancing_llm_agents_via_critique-guided_improvement.md)
- [\[NeurIPS 2025\] Distilling LLM Agent into Small Models with Retrieval and Code Tools](../../NeurIPS2025/llm_agent/distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [\[ICML 2026\] Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](../../ICML2026/llm_agent/memory_is_reconstructed_not_retrieved_graph_memory_for_llm_agents.md)
- [\[ACL 2025\] GeAR: Graph-enhanced Agent for Retrieval-augmented Generation](gear_graph-enhanced_agent_for_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
