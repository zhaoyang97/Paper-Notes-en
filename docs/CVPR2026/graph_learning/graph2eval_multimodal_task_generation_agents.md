---
title: >-
  [Paper Note] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs
description: >-
  [CVPR 2026][Graph Learning][knowledge graph] This paper proposes Graph2Eval, a framework that leverages knowledge graphs constructed from heterogeneous data sources as a structured task space. By employing subgraph sampling, task templates, and meta-path strategies, it automatically generates semantically consistent and solvable multimodal agent evaluation tasks, achieving improvements of 20% and 17% in semantic consistency and solvability, respectively.
tags:
  - CVPR 2026
  - Graph Learning
  - knowledge graph
  - agent evaluation
  - task generation
  - multimodal
  - benchmark
date: 2026-05-08
content_hash: e26bc9524ecdaacc
---

# Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs

**Conference**: CVPR 2026
**arXiv**: [2510.00507](https://arxiv.org/abs/2510.00507)
**Code**: [GitHub](https://github.com/YurunChen/Graph2Eval)
**Area**: Agent Evaluation / Knowledge Graphs
**Keywords**: knowledge graph, agent evaluation, task generation, multimodal, benchmark

## TL;DR

This paper proposes Graph2Eval, a framework that leverages knowledge graphs constructed from heterogeneous data sources as a structured task space. By employing subgraph sampling, task templates, and meta-path strategies, it automatically generates semantically consistent and solvable multimodal agent evaluation tasks, achieving improvements of 20% and 17% in semantic consistency and solvability, respectively.

## Background & Motivation

**State of the Field**: As multimodal LLM-driven agents continue to advance in autonomy and generalization, rigorously evaluating their true capabilities has become increasingly important. Existing evaluation approaches rely primarily on static datasets (e.g., GAIA, MiniWoB++, Mind2Web) or environments requiring extensive human annotation (e.g., OSWorld, AndroidWorld).

**Limitations of Prior Work**:
1. Static datasets cannot distinguish genuine generalization from memorization, and scale poorly.
2. Existing LLM-based synthesis methods (e.g., TaskCraft) lack explicit entity-relation modeling, resulting in tasks with poor semantic consistency and solvability.
3. Web interaction task generation methods depend on static data and predefined page relationships, limiting transferability to dynamic web scenarios.

**Root Cause**: How to automatically generate semantically consistent, solvable, and diverse agent evaluation tasks without extensive human annotation.

**Paper Goals**: To construct an automated, scalable, and semantically grounded agent task generation framework covering both document understanding and web interaction scenarios.

**Starting Point**: Treating knowledge graphs (KGs) as structured task spaces, using entities and relations within the graph to constrain task generation and ensure semantic consistency and solvability.

**Core Idea**: Encode entity relationships in data via knowledge graphs, and automatically produce high-quality agent evaluation tasks through subgraph sampling and template-driven generation mechanisms.

## Method

### Overall Architecture

The Graph2Eval dataset generation pipeline comprises five stages: **Data Ingestion** → **Knowledge Graph Construction** → **Subgraph Sampling** → **Task Generation** → **Coverage Optimization**. The overall approach first constructs a structured knowledge graph from document/web data, then extracts local subgraphs via subgraph sampling, and finally generates concrete task instances using task templates and an LLM.

### Key Designs

1. **Data Ingestion**:
    - Applies semantic chunking to document data, segmenting documents into minimal semantic units mapped to graph nodes.
    - Computes $d=384$-dimensional embeddings for each node using all-MiniLM-L6-v2, stored in a vector database.
    - Annotates each node with metadata (file path, title, author, etc.).
    - Collects DOM structures and screenshots from web data via automated URL crawling, integrating simulated human interactions to handle complex page designs.

2. **Knowledge Graph Construction**:
    - Defines the KG as $G = (V, E, R)$, where $V$ is the node set, $E$ is the edge set, and $R$ is the relation type set.
    - **Node Extraction**: Parses paragraphs, headings, hyperlinks, forms, buttons, and other elements from documents/web pages, mapping them to nodes $V = \{v_i \mid v_i \in \text{Elements}(D), \text{type}(v_i) \in \text{NodeTypeSet}\}$.
    - **Node Representation**: Concatenates textual content $c_i^T$ and visual content $c_i^V$ (converted to text descriptions via $\phi_{\text{visual}}$), then encodes them as $h_i = f_{\text{embed}}(c_i^{T+V})$.
    - **Edge Construction**: Builds a heterogeneous edge set $E = E_{\text{text}} \cup E_{\text{web}}$; text edges encode structural (containment/sequence), semantic, contextual, and citation relations; web edges encode navigation, interaction, and layout relations.

3. **Subgraph Sampling**:
    - **Document Understanding Mode**: Selects nodes based on semantic relevance (cosine similarity $\cos(h_i, h_g) > \tau$) and structural matching ($\text{StructMatch}$), retaining only nodes of specified types.
    - **Web Interaction Mode**: Employs a seed-driven strategy, first identifying task-specific seed nodes $S_{\text{seed}}(g)$ (buttons, forms, etc.), then collecting $k$-hop neighbors of seed nodes.
    - The resulting subgraph $G_g = (V_g, E_g, R) \subseteq G$ contains all selected nodes and their internal edges.

4. **Task Generation**:
    - **Document Understanding Tasks**: Maintains a task template library (covering QA, comparison, analysis, reasoning, etc.), extracts template variables from sampled subgraphs, and generates concrete task instances with an LLM.
    - **Web Interaction Tasks**: Proposes a seed-driven subgraph sampling strategy—first identifying key action nodes on the page as "task seeds," then generating concrete task chains via meta-path matching (e.g., Search → Filter → Detail), and finally using an LLM to generate tasks conditioned on the subgraph structure and page context.

5. **Coverage Optimization**:
    - Employs a multi-stage optimization process to ensure task quality, diversity, and representativeness.
    - Iteratively selects tasks using Maximal Marginal Relevance (MMR) to balance coverage and novelty.
    - Coverage dimensions include: node type, edge type, pattern, page level, website type, and difficulty.

### Loss & Training

This work does not involve model training; it is a task generation framework. Task generation and optimization are based on GPT-4o, and evaluation employs multiple models (GPT-4o, Deepseek-V3, Qwen2.5-VL series, Gemini 2.5 Flash, etc.). In terms of generation efficiency, document understanding tasks average 34.87 seconds each, and web interaction tasks average 95.51 seconds each.

## Key Experimental Results

### Main Results

| Model | Setting | F1 | ROUGE-L | LLM Judge |
|------|------|-----|---------|-----------|
| GPT-4o | Single Agent | 0.5766 | 0.4874 | 0.7854 |
| GPT-4o | Multi-Agent | 0.5916 | 0.4873 | 0.7623 |
| Deepseek-V3 | Single Agent | 0.5376 | 0.4518 | **0.8351** |
| Deepseek-V3 | Multi-Agent | 0.5497 | 0.4635 | 0.7984 |
| Qwen2.5-VL-72B | Single Agent | 0.5730 | 0.4837 | 0.7094 |
| Qwen2.5-VL-7B | Single Agent | 0.2093 | 0.1939 | 0.5427 |

Web interaction tasks (Agent S 2.5 overall Success Rate):

| Model | Overall SR |
|------|-----------|
| Gemini 2.5 Flash | **69.20%** |
| Qwen2.5-VL-72B | 38.80% |
| GPT-4o mini | 33.12% |
| UI-TARS-1.5-7B | 7.19% |

### Ablation Study

| Method | Doc Consistency | Doc Solvability | Web Consistency | Web Solvability |
|------|----------------|-----------------|-----------------|-----------------|
| Graph2Eval w/o KG | 0.74 | 0.73 | 0.62 | 0.60 |
| Graph2Eval (Full) | **0.95** (+20%) | **0.93** (+17%) | **0.78** | **0.72** |

Agent evaluation ablation (Qwen2.5-VL-72B):

| Method | Doc Acc | Web SR |
|------|---------|--------|
| w/o KG | 0.68 | 0.12 |
| Graph2Eval | **0.85** | **0.24** |

### Key Findings

1. The incorporation of knowledge graphs substantially improves task semantic consistency (+20%) and solvability (+17%), with KG edge precision reaching 88%.
2. The no-KG baseline generates web tasks largely confined to single-page interactions; multi-page workflows are unsolvable due to the absence of inter-page relation modeling.
3. Graph2Eval-Bench effectively differentiates performance across models of varying scales (e.g., Qwen-72B vs. 7B).
4. Compared to TaskCraft's bottom-up approach, Graph2Eval's top-down paradigm (constructing KG first, then sampling) yields a greater diversity of task types.

## Highlights & Insights

1. The idea of **KG as task space** is highly creative—reformulating task generation as subgraph sampling on a graph naturally guarantees semantic consistency.
2. **A unified framework spanning two scenarios**: document understanding (RAG Agent) and web interaction (Web Agent), realized through a unified graph abstraction.
3. The **seed-driven + meta-path strategy** endows web task generation with compositional flexibility, avoiding rigid all-or-nothing constraints.
4. Results validate that current agents still have substantial room for improvement on dynamically and automatically generated tasks (even the strongest model, Agent S 2.5, achieves only 69% SR).

## Limitations & Future Work

1. KG construction quality is highly dependent on data preprocessing and the accuracy of entity/relation extraction; edge precision is 88% rather than 100%.
2. The current framework covers only document understanding and web interaction, excluding broader agent task types such as tool use and multimodal reasoning.
3. Task generation relies on GPT-4o, which is costly and may introduce model bias.
4. Dynamic update mechanisms for knowledge graphs are not discussed in detail; handling changes in web content remains a challenge.
5. Control over the difficulty distribution of generated tasks is limited, potentially concentrating tasks at medium difficulty.

## Related Work & Insights

- **TaskCraft**: A bottom-up atomic task composition method, but lacking explicit relation modeling.
- **OSWorld / MiniWoB++**: Environment-based benchmarks relying on human annotation, with limited scalability.
- **GAIA / MMBench**: Static QA datasets incapable of evaluating dynamic interaction abilities.
- **Insights**: The KG-driven task generation paradigm can be extended to other agent evaluation domains (e.g., code generation, multi-tool coordination).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[CVPR 2026\] Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning](graph-to-frame_rag_visual-space_knowledge_fusion_for_training-free_and_auditable.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[CVPR 2026\] WSGG: Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](wsgg_spatiotemporal_world_scene_graph.md)
- [\[CVPR 2026\] ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning](viterbiplannet_injecting_procedural_knowledge_via_differentiable_viterbi_for_pla.md)

<!-- RELATED:END -->
