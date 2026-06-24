---
title: >-
  [Paper Note] NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] NeuSym-RAG proposes a hybrid neural-symbolic retrieval framework that parses PDF documents through multiview chunking and simultaneously stores them into a relational database and a vector database. Under this framework, an LLM Agent iteratively interacts with the backends via executable actions (SQL queries, vector retrieval, viewing images, etc.), improving performance on academic paper QA by 17.3% compared to classic RAG.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "PDF QA"
  - "neural-symbolic retrieval"
  - "text-to-SQL"
  - "multiview chunking"
date: 2026-05-08
content_hash: e5514702ea6b89fc
---

# NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering

**Conference**: ACL 2025  
**arXiv**: [2505.19754](https://arxiv.org/abs/2505.19754)  
**Code**: [https://github.com/X-LANCE/NeuSym-RAG](https://github.com/X-LANCE/NeuSym-RAG)  
**Area**: LLM Agent / RAG  
**Keywords**: RAG, PDF QA, neural-symbolic retrieval, text-to-SQL, multiview chunking

## TL;DR
NeuSym-RAG proposes a hybrid neural-symbolic retrieval framework that parses PDF documents through multiview chunking and simultaneously stores them into a relational database and a vector database. Under this framework, an LLM Agent iteratively interacts with the backends via executable actions (SQL queries, vector retrieval, viewing images, etc.), improving performance on academic paper QA by 17.3% compared to classic RAG.

## Background & Motivation

**Background**: RAG is the mainstream solution for knowledge-intensive question answering in LLMs, but typical implementations rely solely on vector-based neural retrieval (embeddings + similarity search). On the other hand, symbolic retrieval methods like Text-to-SQL excel at precise queries but struggle with fuzzy matching.

**Limitations of Prior Work**:
   - **Isolated Study of Neural and Symbolic Retrieval**: Vector retrieval is proficient in semantic fuzzy matching but struggles with aggregation/comparison queries (e.g., "How many tables are in this paper?"). In contrast, symbolic retrieval is suitable for precise execution but fails when facing synonymous variations (e.g., "graph-based RAG" vs "GraphRAG").
   - **Fixed-length Chunking Ignores PDF Structure**: Research papers possess rich intrinsic structures (sections, tables, figures, formulas), which simple sequential split strategies fail to leverage.
   - **Inadequate Realism in Academic QA Datasets**: Most existing datasets are based on single-page content or summaries rather than full-length, multi-document PDF analysis.

**Key Challenge**: A single retrieval paradigm cannot satisfy the diverse nature of real-world user queries (semantic understanding vs. precise computing).

**Goal**: To unify neural and symbolic retrieval into an interactive framework that fully exploits the multiview structural information of PDFs.

**Key Insight**: Parsing PDFs into two parallel backends—a relational database (for precise queries) and a vector store (for semantic matching)—and employing an Agent to adaptively choose the retrieval method based on query types.

**Core Idea**: Utilizing a schema-constrained database to link the multiview chunking of PDFs with vector encoding, and deploying an LLM Agent using the ReAct framework to iteratively interact between the two retrieval backends until sufficient information is collected to generate answers.

## Method

### Overall Architecture
PDF Input → **Stage 1: Multiview Parsing** (parsed into the DuckDB relational database) → **Stage 2: Multimodal Encoding** (encodable column values are vectorized and stored in the Milvus vector store) → **Stage 3: Iterative Agent Interaction** (Agent iteratively retrieves from DB/VS using 5 types of actions to generate answers).

### Key Designs

1. **Multiview Document Parsing**:

    - **Function**: Parsing PDF contents at multiple granularities to populate a relational database.
    - **Mechanism**: (1) Querying the arXiv API to obtain metadata (authors, conference, etc.). (2) Employing PyMuPDF to segment text at page, section, and fixed-length granularities. (3) Using the OCR model MinerU to extract tables and figures. (4) Generating summaries of various elements using LLMs/VLMs.
    - **Design Motivation**: Different queries require different granularities—"how many tables are in this paper" needs table-level granularity, while "details of a specific method" needs section-level granularity. Multiview corresponds to multiple granularities.

2. **Multimodal Vector Encoding**:

    - **Function**: Vectorizing encodable column values in the database, establishing a one-to-one mapping between DB and VS.
    - **Mechanism**: Column types of varchar representing long texts in the DB schema are marked as "encodable". They are encoded using 3 text encoders (BM25, MiniLM, BGE) and 1 image encoder (CLIP). Each vector is appended with a (table_name, column_name, primary_key) triplet to map back to the DB.
    - **Design Motivation**: The DB schema serves as a "skeleton" to organize vectors in the VS, with different encoders catering to diverse matching demands.

3. **Agent Interaction with 5 Executable Actions**:

    - **RetrieveFromVectorstore**: The Agent rewrites queries, selects encoding models and views (table name + column name), supporting metadata filtering.
    - **RetrieveFromDatabase**: The Agent generates SQL queries to perform precise retrieval.
    - **ViewImage**: The Agent specifies coordinates to crop PDF pages and feeds the crop into a VLM for inference.
    - **CalculateExpr**: Executes Python mathematical expressions to mitigate mathematical hallucinations.
    - **GenerateAnswer**: Termination action, delivering the final answer.
    - **Design Motivation**: The Agent can freely combine both retrieval mechanisms—e.g., executing SQL filtering followed by vector matching, or performing vector retrieval followed by SQL refinement.

### Hybrid Retrieval Collaborative Mode
- **DB → VS**: SQL is first employed to filter rows matching structured criteria. Then, primary keys are extracted and passed to the vector search as filters for semantic matching.
- **VS → DB**: Vector search is first utilized to locate semantically relevant entries, which are then converted into temporary tables or SQL conditions for further precision querying.
- **ReAct Framework**: In each turn, the Agent outputs thought → action → observation, iterating until GenerateAnswer is triggered.

## Key Experimental Results

### Main Results (AirQA-Real Dataset)

| Method | Text | Table | Image | Formula | Metadata | AVG |
|------|------|-------|-------|---------|----------|-----|
| Classic-RAG (GPT-4o-mini) | 12.3 | 11.9 | - | - | - | ~25 |
| HybridRAG | - | - | - | - | - | ~30 |
| GraphRAG | - | - | - | - | - | ~28 |
| **NeuSym-RAG (GPT-4o-mini)** | - | - | - | - | - | **~42** |
| NeuSym-RAG vs Classic-RAG | | | | | | **+17.3%** |

Cross-dataset results:

| Dataset | Classic-RAG | NeuSym-RAG | Gain |
|--------|-----------|-----------|------|
| AirQA-Real | 25.0 | **42.3** | +17.3 |
| M3SciQA | 39.2 | **47.5** | +8.3 |
| SciDQA | 44.1 | **49.8** | +5.7 |

### Ablation Study

| Configuration | AirQA-Real AVG |
|------|---------------|
| Full NeuSym-RAG | 42.3 |
| w/o DB (Vector retrieval only) | 35.1 (-7.2) |
| w/o VS (SQL retrieval only) | 30.8 (-11.5) |
| w/o Multiview (Single chunking) | 37.6 (-4.7) |
| w/o ViewImage | 39.1 (-3.2) |

### Key Findings
- **Hybrid retrieval significantly outperforms single paradigms**: Removing either the DB or the VS leads to a clear drop in performance, validating their complementary nature.
- **Multiview chunking contributes 4.7%**: This suggests that splitting text at various granularities is effective, as different queries necessitate different perspectives.
- **Model scale is crucial for Agent retrieval**: GPT-4o > GPT-4o-mini > open-source models, because the Agent demands stronger planning and SQL generation capabilities.
- **Greatest gains are observed in table and formula-type questions**: This is precisely where traditional vector retrieval is weakest, and symbolic retrieval brings precise querying capability.

## Highlights & Insights
- **Designing the DB schema as the "organizational skeleton" for the VS is highly elegant**: Each vector maps back to the DB via a (table, column, pk) triplet, enabling seamless bridging of the two systems. This can be adapted to any structured-document RAG scenario (such as legal documents, financial reports, etc.).
- **Multiview partitioning effectively improves upon "one-size-fits-all chunking"**: Page-level, section-level, and fixed-length segmentations of the same document serve different query types, and combined with table/figure-level parsing, they ensure comprehensive coverage.
- **Adaptive retrieval strategy of the Agent**: Letting the LLM decide when to use SQL and when to use vector search is far more flexible than hardcoded hybrid retrieval.

## Limitations & Future Work
- **Dependency on PDF parsing quality**: Errors in OCR and table extraction propagate directly to retrieval results.
- **AirQA-Real contains only 553 labeled samples**: The small dataset size limits statistical significance.
- **Agent interaction turns increase inference cost**: Each ReAct turn requires calling the LLM, introducing higher latency during multi-turn interactions.
- **Lack of direct comparison with the latest long-context LLMs**: Such as feeding the complete PDF directly to GPT-4o-128k.

## Related Work & Insights
- **vs Classic RAG**: Leverages only vector retrieval and fails to handle precise queries; Ours achieves +17.3%.
- **vs GraphRAG (Edge et al., 2024)**: Utilizes knowledge graphs to organize information but lacks vector semantic matching; it is suitable for global summarization but struggles with detailed queries.
- **vs HybridRAG (Sarmah et al., 2024)**: Simply merges graphing and vector retrieval without an adaptive selection mechanism for the Agent.
- **vs TAG (Biswal et al., 2024)**: A pure Text-to-SQL approach, which exhibits weakness in semantic fuzzy matching.

## Rating
- Novelty: ⭐⭐⭐⭐ Unifies neural and symbolic retrieval into an Agent interaction framework for the first time; the DB schema bridging design is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets + ablation + newly labeled dataset, although AirQA-Real is relatively small in scale.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagram, and highly detailed action space definition.
- Value: ⭐⭐⭐⭐ High practical value for academic paper QA and structured-document RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries](logical_consistency_is_vital_neural-symbolic_information_retrieval_for_negative-.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [\[ACL 2025\] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](hybgrag_hybrid_rag_skb.md)
- [\[ACL 2025\] GRAF: Graph Retrieval Augmented by Facts for Romanian Legal Multi-Choice Question Answering](graf_graph_retrieval_augmented_by_facts_for_romanian_legal_multi-choice_question.md)
- [\[ACL 2025\] ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry](comrag_retrieval-augmented_generation_with_dynamic_vector_stores_for_real-time_c.md)

</div>

<!-- RELATED:END -->
