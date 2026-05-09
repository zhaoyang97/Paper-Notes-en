---
title: >-
  [Paper Note] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs
description: >-
  [CVPR 2026][Graph Learning][knowledge graphs] This paper proposes Graph2Eval, a knowledge graph-driven framework for the automatic generation of agent evaluation tasks. By constructing structured knowledge graphs from documents and webpages, performing subgraph sampling, applying LLM-conditioned generation, and employing multi-stage filtering, the framework automatically produces multimodal agent tasks with improved semantic consistency (+20%) and solvability (+17%). The resulting benchmark, Graph2Eval-Bench, comprises 1,319 tasks.
tags:
  - CVPR 2026
  - Graph Learning
  - knowledge graphs
  - automatic task generation
  - agent evaluation
  - document understanding
  - web understanding
  - benchmark construction
date: 2026-05-08
content_hash: 23c5e29fa80add44
---

# Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs

**Conference**: CVPR 2026
**arXiv**: [2510.00507](https://arxiv.org/abs/2510.00507)
**Code**: [github.com/YurunChen/Graph2Eval](https://github.com/YurunChen/Graph2Eval)
**Area**: Human-Machine Understanding / Agent Evaluation
**Keywords**: knowledge graphs, automatic task generation, agent evaluation, document understanding, web understanding, benchmark construction

## TL;DR
This paper proposes Graph2Eval, a knowledge graph-driven framework for the automatic generation of agent evaluation tasks. By constructing structured knowledge graphs from documents and webpages, performing subgraph sampling, applying LLM-conditioned generation, and employing multi-stage filtering, the framework automatically produces multimodal agent tasks with improved semantic consistency (+20%) and solvability (+17%). The resulting benchmark, Graph2Eval-Bench, comprises 1,319 tasks.

## Background & Motivation
Evaluation of multimodal agents (document-understanding agents, web-browsing agents) relies heavily on manually annotated static benchmarks, which suffer from three fundamental limitations:

**Scale bottleneck**: Manual task construction is costly and slow, making it difficult to keep pace with the rapid advancement of agent capabilities.

**Coverage gaps**: Static benchmarks cover only a limited range of task types and difficulty levels, making them susceptible to benchmark overfitting.

**Temporal staleness**: Real-world documents and webpages are continuously updated, causing ground-truth annotations in fixed benchmarks to become outdated.

**Limitations of existing automated approaches**:
- **Pure LLM synthesis** (Self-Instruct, Evol-Instruct): directly prompting LLMs to generate QA pairs from text fragments lacks explicit modeling of inter-entity relations, resulting in questions that reference non-existent entity combinations (**semantic inconsistency**) or require information along unreachable paths (**unsolvability**).
- **Template filling**: template-based methods produce tasks with fixed formats and poor diversity.
- **Random sampling**: randomly extracting document fragments for QA generation lacks structural awareness and frequently yields trivial or ill-formed tasks.

**Core Idea**: Knowledge graphs serve as an intermediate structured representation. Entities and relations are first extracted from documents and webpages to construct a KG $G=(V,E,R)$; semantically coherent context subgraphs are then obtained via subgraph sampling; finally, LLMs generate tasks conditioned on the subgraph constraints. The structural properties of the KG guarantee reachability of entity relations (solvability) and semantic completeness (consistency).

## Core Problem
How to automatically generate semantically consistent, solvable, and diverse multimodal agent evaluation tasks? Key challenges: (1) how to extract structured knowledge from heterogeneous documents and webpages; (2) how to sample subgraphs suitable as task material; (3) how to ensure the quality of generated tasks (freedom from hallucination and genuine completability).

## Method

### Overall Architecture
Graph2Eval is a five-stage pipeline: Data Ingestion → KG Construction → Subgraph Sampling → Task Generation → Multi-stage Filtering.

### Key Designs

1. **Data Ingestion**:

    - **Document mode**: Semantically segments PDF/HTML documents (paragraphs, headings, tables, captions), generates embedding vectors for each chunk, and preserves metadata (page number, hierarchy, context window).
    - **Web mode**: Parses the DOM tree to extract attributes and hierarchical relations of interactive elements (forms, buttons, links, dropdowns), and captures page screenshots for multimodal understanding.
    - Design Motivation: Unifying heterogeneous sources into structured node representations lays the foundation for subsequent graph construction.

2. **KG Construction**:

    - Graph definition: $G = (V, E, R)$
    - **Node types $V$**: paragraph, heading, hyperlink, form_field, table_cell, image_caption, etc.
    - **Edge types**:
        - Textual edges: sequential relations (ordering within a document), semantic relations (embedding similarity above threshold), and citation relations (cross-references, footnotes, hyperlinks).
        - Web edges: navigation relations (link traversal), interaction relations (button↔form, dropdown↔option), and layout relations (DOM parent-child/sibling).
    - Design Motivation: Multi-typed edges capture semantic associations at different granularities—sequential edges preserve local context, semantic edges connect distantly related content, and interaction edges encode agent-executable actions.

3. **Subgraph Sampling**:

    - **Document mode**: Cosine similarity + StructMatch—a seed node is selected, neighbors are expanded by embedding similarity, and StructMatch evaluates the structural diversity of candidate subgraphs (proportion of distinct node/edge types), ensuring that sampled subgraphs are both semantically relevant and structurally rich.
    - **Web mode**: Seed-driven $k$-hop expansion—starting from a seed interactive element, the graph is expanded along navigation/interaction edges for $k$ hops ($k$=2–3) to capture the complete operation path required by an agent to complete a task.
    - Design Motivation: (1) Document tasks require cross-paragraph reasoning → semantic expansion; (2) web tasks require multi-step operations → path expansion. The two sampling strategies are tailored to the respective characteristics of each modality.

4. **Task Generation**:

    - **Template construction**: The sampled subgraph is serialized into a structured prompt (containing node content, edge relations, and metadata) to guide the LLM in constructing task instructions and expected answers grounded in the subgraph.
    - **Meta-path guidance**: Common meta-path patterns are defined (e.g., heading→paragraph→table_cell representing "look up table data based on section description"), and the LLM generates complex QA requiring multi-step reasoning along these meta-paths.
    - Design Motivation: The meta-path mechanism constrains LLM generation—each reasoning step is grounded in entity relations present in the KG, reducing hallucination at the source.

5. **Multi-stage Filtering**:

    - **Stage 1: Node reachability check**—verifies whether all entities referenced in a task answer are reachable from the task starting point within the KG (unreachable → unsolvable → discarded).
    - **Stage 2: LLM quality scoring**—a separate LLM scores each task on clarity, difficulty appropriateness, and answer correctness (1–5); tasks scoring below 3 are discarded.
    - **Stage 3: Similarity-based deduplication**—embedding similarities across tasks are computed, and within clusters of highly similar tasks only the highest-quality instance is retained to ensure overall diversity.
    - Design Motivation: The three-stage filter operates progressively—structural level (reachability) → semantic level (quality) → set level (diversity)—providing layered quality assurance.

### Loss & Training
- **Training-free**: Graph2Eval is a purely inference-time pipeline.
- KG construction uses off-the-shelf embedding models (e.g., text-embedding-3-small).
- Task generation and quality scoring employ GPT-4o and GPT-4-turbo, respectively.
- Average generation time: 34.87 s/task for document tasks and 95.51 s/task for web tasks.

## Key Experimental Results

### Graph2Eval-Bench Dataset Statistics

| Category | Count | Avg. Steps | Node Types Involved |
|----------|-------|-----------|---------------------|
| Document tasks | 1,002 | 2.8 | paragraph, table, heading, image |
| Web tasks | 317 | 4.2 | form, button, link, dropdown |
| **Total** | **1,319** | 3.1 | — |

### Comparison with Baseline Task Generation Methods

| Method | Semantic Consistency ↑ | Solvability ↑ | Diversity ↑ | Hallucination Rate ↓ |
|--------|----------------------|--------------|------------|----------------------|
| Self-Instruct | 0.62 | 0.58 | 0.71 | 18.3% |
| Evol-Instruct | 0.67 | 0.63 | 0.68 | 15.1% |
| Template-based | 0.78 | 0.82 | 0.41 | 5.2% |
| **Graph2Eval** | **0.84** | **0.80** | **0.76** | **4.7%** |

Graph2Eval outperforms the strongest baseline, Evol-Instruct, by +20% in semantic consistency (0.84 vs. 0.67+) and +17% in solvability (0.80 vs. 0.63+).

### Agent Performance on Graph2Eval-Bench

| Agent | Document Task Accuracy | Web Task Success Rate | Overall |
|-------|----------------------|-----------------------|---------|
| GPT-4o | 61.3% | 42.7% | 56.8% |
| Claude-3.5 | 58.9% | 39.2% | 54.1% |
| Gemini-1.5 | 55.2% | 36.8% | 50.5% |
| Open-source best | 41.7% | 28.3% | 38.4% |

Graph2Eval-Bench exhibits sufficient discriminative power—even the strongest model, GPT-4o, achieves only 56.8%, while the best open-source model reaches 38.4%, leaving substantial room for improvement.

### Key Findings
- **KG structure is central**: Replacing the KG with raw text chunks for direct task generation reduces semantic consistency by 22% and solvability by 19%, confirming that explicit entity relation modeling is indispensable.
- **Meta-path guidance is effective**: Tasks generated with meta-path guidance involve more reasoning steps on average (3.4 vs. 2.1) and achieve higher answer accuracy (+8%).
- **Multi-stage filtering is irreplaceable**: Without filtering, approximately 31% of tasks have quality issues (unsolvable or hallucinated); three-stage filtering reduces this to 4.7%.
- **Web tasks are substantially harder**: All agents perform 15–20 percentage points lower on web tasks than on document tasks, indicating that multi-step interactive operations are the primary bottleneck.

## Highlights & Insights
- Using the KG as a **"skeleton for task generation"** is an elegant design choice—it reformulates the unstructured problem of free-form text into a graph-theoretic problem, leveraging graph connectivity to ensure solvability and node content to ensure semantic consistency.
- The unified framework for document and web modalities demonstrates the generalizability of the approach—adapting to a new modality requires only redefining node and edge types.
- The multi-stage filtering design is both practical and efficient—structural reachability checks are computationally cheap, LLM quality scoring is applied only to tasks that pass structural verification, and similarity-based deduplication is performed globally at the final stage.
- The construction of a 1,319-task benchmark constitutes a direct contribution to the agent evaluation community.

## Limitations & Future Work
- KG construction quality depends on the embedding model and threshold settings—general-purpose embeddings may be insufficiently accurate for specialized domains (e.g., medical or legal documents).
- Web tasks number only 317 (vs. 1,002 document tasks), representing an imbalance in scale—DOM parsing and interaction edge extraction for webpages are more complex, making large-scale expansion costly.
- Both task generation and quality scoring rely on GPT-4-level LLMs, incurring high costs and introducing potential preference bias toward specific LLMs.
- Dynamic webpages are not considered—real-world webpage content changes over time, and generated tasks may quickly become invalid.
- The solvability check only verifies node reachability within the KG; practical solvability is also constrained by agent tool capabilities (e.g., inability to operate certain JavaScript controls).
- Meta-path patterns are predefined—new document structures may require manual extension of the pattern library.

## Related Work & Insights
- Graph2Eval is complementary to manually constructed web agent benchmarks such as OSWorld and WebArena, enabling automated rapid generation of evaluation tasks for new websites.
- Compared to DocBench (a document understanding benchmark), the KG-based approach of Graph2Eval can generate more complex tasks requiring cross-paragraph reasoning.
- The KG-driven QA generation paradigm is transferable to the RAG evaluation domain—KG structure can constrain the generation of evaluation questions requiring multi-hop reasoning.
- A broader implication for agent evaluation: transitioning from "manually constructed static benchmarks" to "automated, structure-guided generation" is a scalable and promising direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing knowledge graphs into automatic agent task generation is a novel perspective; the five-stage pipeline is well-designed and complete.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage including multi-baseline comparisons, ablation analysis, multi-agent evaluation, and task quality statistics.
- Writing Quality: ⭐⭐⭐⭐ Clear framework presentation with detailed descriptions of each pipeline stage.
- Value: ⭐⭐⭐⭐⭐ Dual contributions of a benchmark and an automatic generation framework provide direct practical value to the agent evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[CVPR 2026\] WSGG: Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](wsgg_spatiotemporal_world_scene_graph.md)
- [\[CVPR 2026\] ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning](viterbiplannet_injecting_procedural_knowledge_via_differentiable_viterbi_for_pla.md)
- [\[CVPR 2026\] Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning](graph-to-frame_rag_visual-space_knowledge_fusion_for_training-free_and_auditable.md)

</div>

<!-- RELATED:END -->
