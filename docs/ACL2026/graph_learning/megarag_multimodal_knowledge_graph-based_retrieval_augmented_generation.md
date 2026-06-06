---
title: >-
  [Paper Note] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation
description: >-
  [ACL 2026][Graph Learning][Multimodal Knowledge Graph] MegaRAG utilizes MLLMs to extract entity-relations in parallel from each page of long documents and merges them into a Multimodal Knowledge Graph (MMKG). It employs…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Multimodal Knowledge Graph"
  - "RAG"
  - "Cross-modal Reasoning"
  - "Visual Document QA"
  - "MLLM"
date: 2026-05-08
content_hash: f72421f4d56b4c1b
---

# MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2512.20626](https://arxiv.org/abs/2512.20626)  
**Code**: https://github.com/AI-Application-and-Integration-Lab/MegaRAG (Available)  
**Area**: Multimodal RAG / Knowledge Graph  
**Keywords**: Multimodal Knowledge Graph, RAG, Cross-modal Reasoning, Visual Document QA, MLLM

## TL;DR
MegaRAG utilizes MLLMs to extract entity-relations in parallel from each page of long documents and merges them into a Multimodal Knowledge Graph (MMKG). It employs "subgraph-guided" two-round refinement to complete cross-modal and cross-page relations. Combined with dual-path retrieval and two-stage answer generation, it significantly outperforms GraphRAG/LightRAG/VisRAG, achieving 64.85% accuracy on SlideVQA(2k) (compared to the best baseline at 27.66%).

## Background & Motivation
**Background**: RAG providing external knowledge to LLMs has become the de facto standard for long-document QA. Among these, KG-based RAGs like GraphRAG / LightRAG introduce "entity-relation graphs," which excel at multi-hop reasoning compared to chunk-level sparse/dense retrieval and maintain scalability on million-token corpora.

**Limitations of Prior Work**: 1) Existing KG-based RAGs are almost exclusively **text-only**, discarding visual information such as charts, flowcharts, and maps, which are core to visual documents like slide decks and technical reports. 2) Even existing multimodal RAGs (VisRAG, ColPaLi, GME) only perform page-image level dense retrieval **without structural abstraction**, failing to support cross-page multi-hop reasoning. 3) Limited by MLLM context windows, GraphRAG/LightRAG perform independent extraction within chunks and then merge, causing **cross-chunk/cross-page relations** to be severed, resulting in naturally fragmented initial graphs.

**Key Challenge**: Structural abstraction requires per-page/block extraction, while global coverage requires seeing the entire book—both are difficult to satisfy simultaneously within MLLM context windows. Furthermore, utilizing images requires inserting them into prompts, but too many images can lead to modal bias where the model becomes dominated by text.

**Goal**: (i) Automatically and scalably construct MMKGs from visual documents; (ii) Allow each page to "see" the global graph under restricted context windows; (iii) Ensure answer generation absorbs both graph structure and visual evidence without bias.

**Key Insight**: The authors observe that while the initial KG is fragmented, it serves as a "compressed version of global knowledge." Injecting the **subgraph corresponding to the current page** back into the MLLM as a prompt during the second round of refinement preserves global long-range dependencies without overwhelming the prompt.

**Core Idea**: Use a two-round construction process—"page-wise parallel initial construction + subgraph-guided global refinement"—to break the trade-off between "local extraction vs. global coverage." Then, use a two-stage generation process with dual streams (graph-based answer + visual answer) and late fusion to eliminate modal bias.

## Method

### Overall Architecture
The MegaRAG pipeline consists of four steps: (a) **Initial MMKG Construction** — Divide the document into $N$ pages, where each page input is $P_i = \{T_i, F_i, B_i, I_i\}$ (body text, figures, tables, full-page rendering). MLLMs are called in parallel to extract $(E,R)_i^0$, which are merged into $\mathcal{G}^0$. (b) **Graph Refinement** — For each page, use its extracted entity names and relation keywords to retrieve a semantically relevant + one-hop neighborhood subgraph $\mathcal{G}_i^0$ from $\mathcal{G}^0$. This is fed back into the same MLLM along with the original $P_i$ to supplement missing entities and implicit relations, resulting in $\mathcal{G}^1$. (c) **Indexing** — Use GME (Qwen2-VL-2B) to encode page images, entities (name+description), and relations (keywords+source+target+description) into the same dense vector space. (d) **Retrieval + Generation** — The query is decomposed by the MLLM into low-level (specific entities) and high-level (broad concepts) keywords to query the entity and relation databases (top-$k=60$), followed by one-hop expansion. Simultaneously, the query directly searches the page-image database (top-$m=6$). In the generation stage, intermediate answers are produced in parallel using graph evidence and visual evidence, then fused in a second stage for the final answer.

### Key Designs

1.  **Two-round Page-wise MMKG Construction (Core Innovation)**:
    - **Function**: Combines the high efficiency of "page-wise parallel extraction" with the high quality of "global coverage" without exceeding MLLM context windows.
    - **Mechanism**: The first round independently extracts $(E,R)_i^0 = G(P_i)$ for each page, treating charts/tables as single entities (e.g., a "Annual Sales by Vehicle Type" bar chart is a node). The full-page image $I_i$ provides spatial reasoning context but does not produce entities. Merging involves deduplication by entity name and (source, target, type) triplets while aggregating descriptions. The second round refinement is denoted as $(E,R)_i^1 = R(P_i, \mathcal{G}_i^0)$: using first-round entity names and relation keywords as queries, a lightweight subgraph $\mathcal{G}_i^0$ (top-120 semantic + one-hop neighborhood, truncated to 32K tokens) is retrieved from $\mathcal{G}^0$. This allows the MLLM to "refer to the global context" to identify missing entities and implicit relations (e.g., linking the text "Electric vehicle sales increased in 2023" to the graph "Annual Sales by Vehicle Type" with an `illustrates` relation).
    - **Design Motivation**: Feeding the entire $\mathcal{G}^0$ into the prompt exceeds token limits and breaks parallelism. "Viewing the subgraph" in the second round allows recalling cross-page relations (e.g., lines connecting entities in text to line charts spanning multiple pages) while keeping per-page reasoning independent and parallelizable. Unlike the intra-chunk gleaning in GraphRAG/LightRAG, this refinement involves **global information flow back to the page level**.

2.  **Dual-layer Keywords + Unified Retrieval Across Three Vector Stores**:
    - **Function**: Enables a query to simultaneously obtain "local evidence of specific entities" and the "global structure of broad concepts," supplemented by visual page evidence.
    - **Mechanism**: When a query enters, the MLLM extracts two types of keywords (low-level / high-level). These are used to query the entity vector store for top-$k$ and the relation vector store for top-$k$ (each relation also pulls its source/target entities), followed by a one-hop neighborhood expansion for all hit entities. Simultaneously, the query is encoded to search the page-image store for top-$m$ as visual evidence. All three types of vectors (entity, relation, page) share the same GME encoder, enabling cross-modal text→entity / text→relation / text→page retrieval.
    - **Design Motivation**: Entity-only retrieval misses relation context; relation-only retrieval misses isolated entities. The dual-layer keyword approach allows "searching for specific names" and "searching for chapter themes" to coexist. Shared multi-vector space via GME is key—VisRAG / ColPaLi lack KGs, while GraphRAG / LightRAG cannot retrieve images. This encoder unifies both aspects.

3.  **Two-stage Answer Generation (Graph Answer + Visual Answer + Fusion, De-biasing)**:
    - **Function**: Prevents the MLLM from favoring text when provided with both subgraphs (text) and page images (visual), ensuring the final answer incorporates both structural reasoning and visual details.
    - **Mechanism**: The first stage runs two prompts in parallel: (a) Input retrieved page images for the MLLM to generate an intermediate answer $a_v$ based solely on visual evidence; (b) Input subgraphs for the MLLM to generate an intermediate answer $a_g$ based solely on structural knowledge. The second stage uses a fusion prompt to synthesize $(a_v, a_g)$ into the final answer.
    - **Design Motivation**: Authors found that MLLMs exhibit "over-attention to text" (modal bias) when provided with images and KGs simultaneously. Decoupling reasoning followed by late fusion follows a chain-of-experts logic, allowing two independent streams to provide comparable results.

### Loss & Training
**Completely training-free**. All LLM calls (GPT-4o-mini for construction and generation, GPT-4o-mini as judge) use zero-shot prompting with temperature=0. GME-Qwen2-VL-2B uses a pre-trained multimodal encoder running on a local RTX 3090. Document parsing uses MinerU to extract text/figure/table. Text chunks are 1200 tokens with 100 overlap (for baselines only). Retrieval settings: $k=60$, $m=6$. One round of refinement is sufficient.

## Key Experimental Results

### Main Results

**UltraDomain Pure Text Global QA (vs. LightRAG, Win Rate %, higher is better)**:

| Domain | Comprehensiveness | Diversity | Empowerment | Overall |
| :--- | :--- | :--- | :--- | :--- |
| Agriculture | 65.6 vs 4.0 | 70.4 vs 10.4 | 76.0 vs 4.8 | **75.2 vs 4.8** |
| CS | 68.8 vs 3.2 | 72.0 vs 12.8 | 75.2 vs 10.4 | **76.8 vs 4.8** |
| Legal | 54.4 vs 9.6 | 69.6 vs 14.4 | 73.6 vs 12.0 | **72.0 vs 11.2** |
| Mix | 76.8 vs 3.2 | 76.8 vs 11.2 | 80.8 vs 4.0 | **80.0 vs 7.2** |

**Multimodal Local QA (Accuracy %)**:

| Method | SlideVQA(2k) | FinReport | FinSlides | TechReport | TechSlides |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NaiveRAG | 11.34 | 29.66 | 14.64 | 36.63 | 32.94 |
| GraphRAG (L) | 6.80 | 24.50 | 11.98 | 29.60 | 26.81 |
| LightRAG | 27.66 | 31.30 | 13.02 | 42.74 | 31.39 |
| **MegaRAG** | **64.85** | **39.51** | **58.37** | **51.51** | **60.86** |

MegaRAG is **2.3x** better than the strongest baseline LightRAG on SlideVQA(2k); the absolute gain on FinSlides is as high as **+45 pt**.

### Ablation Study (DLCV / World History Multimodal Global QA, Win Rate %)

| Configuration | DLCV Overall | World History Overall | Description |
| :--- | :--- | :--- | :--- |
| MegaRAG (full) | — | — | Full model |
| A1: text-only graph | 14.4 / 57.6 / 28.0 | 1.6 / 78.4 / 20.0 | Remove all visual inputs during graph construction |
| **A2: Page Retrieval Only (No MMKG)** | **0.0 / 100.0 / 0.0** | **0.8 / 91.2 / 8.0** | Remove KG retrieval, rely only on page retrieval |
| A3: Single-stage Fusion | 1.6 / 61.6 / 36.8 | 0.8 / 75.2 / 24.0 | No decoupling of graph and visual answers |

Format: "A_x Win / MegaRAG Win / Tie".

## Key Findings
- **MMKG retrieval is the most beneficial module**: In A2, removing the KG leads to MegaRAG winning nearly 100% of the time, indicating that multi-hop reasoning in visual documents depends almost entirely on the structured graph.
- **Visual entity construction is the second most critical factor**: In A1, removing visual inputs loses chart semantics, with the largest drop seen in slide-heavy data like DLCV—proving that chart nodes are not decorative but provide grounding.
- **Two-stage generation contributes most to Diversity / Empowerment**: In A3, single-prompt fusion causes the MLLM to favor text, leading to answers lacking the diversity provided by visual evidence.

## Highlights & Insights
- **"Subgraph-as-prompt" is a universal design**: MegaRAG does not feed the entire graph back but dynamically recalls the "most relevant small piece of global structure" for each page. This can be transferred to any refinement scenario involving global information + local chunks, such as long document summarization or code repository analysis.
- **Charts as a category of entity**: Traditional KGs restrict nodes to named entities (people/places/orgs). This paper treats a bar chart or a map as an entity linked to text nodes. This "visual-as-entity" ontology extension provides a simple and feasible schema for MMKGs.
- **Two-stage generation = Cheap ensemble**: Paralleling graph and visual answers followed by fusion is essentially an expert ensemble, but the cost is only one additional MLLM call. It is more stable than single multimodal prompts and cheaper than training multi-expert models.

## Limitations & Future Work
- **Limitations**: Experiments were conducted within single books/reports; multi-document cross-doc QA was not performed. Image processing costs are high, and the scale of multimodal data is limited. Treating each figure as a single entity may miss fine-grained objects.
- **Future Directions**: (1) Decompose figures into multiple sub-entities (bars, axes, legends) using methods like set-of-mark; (2) Introduce self-consistency voting during refinement to suppress hallucinatory relations; (3) Develop entity linking/canonicalization for cross-document scenarios to merge MMKGs from multiple books.

## Related Work & Insights
- **vs. GraphRAG (Edge et al., 2024)**: GraphRAG performs community-level global summarization, with costs increasing linearly with the number of communities. MegaRAG uses page-wise subgraph refinement, avoiding repeated global LLM calls and natively supporting multimodality.
- **vs. LightRAG (Guo et al., 2025)**: Follows the dual-layer keyword retrieval of LightRAG but replaces the backbone and adds refinement; construction cost is slightly higher, but QA quality is superior across all benchmarks.
- **vs. VisRAG / ColPaLi / GME**: These are pure visual RAGs that perform page-image retrieval without structural abstraction. MegaRAG reuses GME as an encoder but adds an MMKG layer for cross-page reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ "Page-wise parallel + subgraph-guided refinement" is a clean and effective solution for MMKG, and the visual-as-entity ontology extension is a true multimodal advancement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 datasets (4 text + 4 multimodal) + 5 baselines + full 3-stage ablation + cross-validation with two judge models + overhead analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology diagrams and unified notation.
- Value: ⭐⭐⭐⭐⭐ Training-free, public code, and doubling baseline performance in the industrially valuable visual document QA domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)
- [\[ACL 2026\] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)
- [\[ACL 2026\] LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)
- [\[ACL 2026\] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)
- [\[ACL 2025\] Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)](../../ACL2025/graph_learning/kg_rag_recommendation.md)
- [\[ACL 2026\] LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega.md)

</div>

<!-- RELATED:END -->
