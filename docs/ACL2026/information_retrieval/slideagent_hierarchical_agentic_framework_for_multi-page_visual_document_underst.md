---
title: >-
  [Paper Note] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding
description: >-
  [ACL 2026][multi-page document understanding] This paper proposes SlideAgent, a hierarchical agentic framework that constructs structured knowledge representations via three dedicated agents operating at the global, page, and element levels, achieving significant improvements in fine-grained understanding of multi-page visual documents, particularly presentation slides.
tags:
  - ACL 2026
  - multi-page document understanding
  - hierarchical agents
  - visual document QA
  - slide comprehension
  - element-level reasoning
date: 2026-05-08
content_hash: 3b83544b46cd2fc6
---

# SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding

**Conference**: ACL 2026
**arXiv**: [2510.26615](https://arxiv.org/abs/2510.26615)
**Code**: [SlideAgent](https://SlideAgent.github.io/)
**Area**: Information Retrieval / Document Understanding
**Keywords**: multi-page document understanding, hierarchical agents, visual document QA, slide comprehension, element-level reasoning

## TL;DR

This paper proposes SlideAgent, a hierarchical agentic framework that constructs structured knowledge representations via three dedicated agents operating at the global, page, and element levels, achieving significant improvements in fine-grained understanding of multi-page visual documents, particularly presentation slides.

## Background & Motivation

**Background**: Multi-page visual documents — such as financial reports, academic presentations, and technical manuals — are pervasive in high-stakes domains including finance, science, and education. These documents convey information not only through text but also through page layout, icons, color coding, and cross-page references.

**Limitations of Prior Work**: Current multimodal large language models (MLLMs) face three key challenges when processing multi-page visual documents: (1) **Insufficient fine-grained reasoning** — MLLMs tend to process each page holistically, overlooking element-level details such as specific data segments within charts; (2) **Lack of domain-specific visual semantics** — pretraining predominantly on natural images leaves models ill-equipped to interpret specialized charts, icon semantics, and spatial layouts in documents; (3) **Metadata dependency** — many systems rely on clean document metadata (e.g., annotated chart positions, hierarchical labels) that is frequently absent or corrupted in practice.

**Key Challenge**: MLLMs may fail when reasoning over a full page holistically (e.g., miscounting categories in a chart), yet succeed when the relevant chart is cropped and presented in isolation — indicating that the model possesses the requisite reasoning capacity but lacks an effective mechanism for fine-grained information extraction.

**Goal**: To construct a general-purpose agentic framework that requires no document metadata, handles multi-page multimodal documents, and achieves precise document understanding through hierarchical knowledge construction and selective agent activation.

**Core Idea**: Inspired by human information processing, the framework decomposes document understanding into three levels — global (overall theme), page (per-page features and cross-page relations), and element (fine-grained parsing of charts, text blocks, and icons) — with dedicated agents at each level operating collaboratively across knowledge construction and reasoning phases.

## Method

### Overall Architecture

SlideAgent operates in two stages: (1) **Knowledge Construction** — a top-down process that builds a hierarchical, query-agnostic knowledge base $\mathcal{K}=\{\mathcal{K}_g, \mathcal{K}_p, \mathcal{K}_e\}$; (2) **Reasoning** — the user query is classified, the corresponding agent levels are selectively activated, and multi-level retrieval and answer synthesis are performed. The entire framework is model-agnostic and compatible with various backbone models such as GPT-4o and InternVL3-8B.

### Key Designs

1. **Hierarchical Knowledge Construction**:

    - Function: Constructs structured, query-agnostic knowledge representations for the document.
    - Mechanism: The global agent $\mathcal{M}_g$ samples the first three pages to generate a document-level summary and thematic overview; the page agent $\mathcal{M}_p$ processes each page sequentially, conditioned on global knowledge and the preceding page's knowledge to produce page-level knowledge $\mathcal{K}_p^i = \mathcal{M}_p(v_i, \mathcal{K}_g^{(0)}, \mathcal{K}_p^{i-1})$; the element agent $\mathcal{M}_e$ decomposes each page into constituent elements — text blocks, charts, icons — via layout parsing and generates semantic role and functional descriptions for each element.
    - Design Motivation: Global knowledge provides a thematic framework, page knowledge supplies sequential context and cross-page associations, and element knowledge delivers fine-grained spatial and content information — the three levels are complementary and mutually indispensable.

2. **Query Classification & Selective Activation**:

    - Function: Activates the requisite subset of agents based on query type, avoiding unnecessary computation and noise introduction.
    - Mechanism: Queries are classified into four categories — global understanding (activates global agent only), factual queries (activates page and element agents), multi-hop reasoning (activates all agents), and layout/visual relation queries (activates element agent). Queries that cannot be classified default to full activation.
    - Design Motivation: Different queries require information at different levels of granularity; over-activating agents increases computational cost and may introduce noise, whereas selective activation strikes a balance between efficiency and accuracy.

3. **Subquery Generation & Multi-level Retrieval**:

    - Function: Expands the original query into multiple subqueries to enable precise retrieval at both page and element levels.
    - Mechanism: Key entities are extracted from the query to generate subqueries; the original query and subqueries are concatenated for joint retrieval of top-$k$ pages and their associated elements. Sparse retrievers (BM25), dense retrievers (SFR), and multimodal retrievers (COLPALI) are all supported.
    - Design Motivation: Original queries are typically short, making direct retrieval noisy; subquery generation broadens semantic coverage, particularly benefiting multi-hop reasoning scenarios.

### Loss & Training

This paper adopts a training-free approach — all agents are implemented via prompt engineering on existing MLLMs, requiring no additional training or fine-tuning. During the knowledge construction stage, carefully designed prompt templates guide each agent to produce structured knowledge. Global knowledge undergoes a refinement step (a single full-field rewrite) to ensure comprehensive integration of information from all pages and to reduce bias toward the first few pages.

## Key Experimental Results

### Main Results

| Dataset | Metric | SlideAgent (GPT-4o) | GPT-4o | Gain |
|--------|------|------|----------|------|
| SlideVQA | Overall | 84.9 | 77.0 | +7.9% |
| TechSlides | Overall | 70.9 | 63.4 | +7.5% |
| FinSlides | Overall | 85.5 | 80.0 | +5.5% |
| InfoVQA | Overall | 79.6 | 69.0 | +10.6% |
| SlideVQA (InternVL3) | Overall | 72.7 | 63.0 | +9.8% |

### Ablation Study

| Configuration | Key Metric (Overall) | Notes |
|------|---------|------|
| w/o Page Agent | −6.3 (GPT-4o) | Largest drop; page-level reasoning is critical for cross-page coherence |
| w/o Element Agent | −4.6 (GPT-4o) | Fine-grained reasoning is especially crucial for numerical questions |
| w/o Global Agent | −2.8 (GPT-4o) | Smallest drop, as lower-level agents partially embed global context |
| w/o Subquery | −5.0 (GPT-4o) | Impact is particularly pronounced in retrieval-dependent scenarios |

### Key Findings
- Hierarchical knowledge construction not only improves QA performance but also substantially enhances page-level retrieval (text retriever SFR gains +6.4 MRR).
- Multi-hop reasoning queries achieve the largest improvements (+9.8%), confirming the value of structured knowledge guidance for complex reasoning.
- A +7.7% gain is observed even in the oracle setting where ground-truth pages are provided, demonstrating that element-level retrieval has independent value.
- Only 12.5% of errors are attributable to OCR/parsing failures; the majority stem from question ambiguity and annotation issues.

## Highlights & Insights
- **Hierarchical divide-and-conquer strategy**: The three-level global–page–element processing model draws on human cognition, is both systematic and intuitive, and lends itself naturally to modular extension in practice.
- **Training-free plug-and-play design**: Built entirely on prompt engineering and existing MLLMs, the framework is directly applicable to any backbone model without modification.
- **Necessity of element-level reasoning**: The motivating case study in Figure 1 compellingly demonstrates MLLM failure in holistic page reasoning versus success after element-level cropping.
- **Dual benefit of knowledge construction**: The structured knowledge generated (page descriptions and subqueries) serves both as a signal for QA and as an augmented retrieval signal, yielding two gains simultaneously.
- **Model agnosticism**: Consistent and significant improvements are obtained across two substantially different backbone models, GPT-4o and InternVL3-8B.

## Limitations & Future Work
- Element boundary detection relies on OCR and layout parsing tools, and parsing quality may vary across tools.
- Global knowledge initialization samples only the first three pages, which may be insufficiently representative for long documents; content-based page selection strategies warrant future exploration.
- The framework primarily employs text-based retrieval (SFR); the potential of multimodal retrievers remains to be further investigated.
- Multi-turn dialogue scenarios are not addressed; extension to interactive document QA represents an important future direction.
- The knowledge construction stage incurs substantial computational overhead, requiring a separate MLLM call for each page.

## Related Work & Insights
- **vs. ViDoRAG**: ViDoRAG also adopts a multi-agent architecture, but SlideAgent's three-level hierarchical design and element-level parsing are considerably more fine-grained, outperforming it comprehensively across all datasets.
- **vs. VDocRAG**: VDocRAG integrates retrieval and reasoning but lacks element-level decomposition; SlideAgent's advantage is especially pronounced on numerical reasoning (Num).
- **vs. COLPALI**: As a purely multimodal retrieval approach, COLPALI is matched or surpassed by SlideAgent's combination of text retrieval and structured knowledge, demonstrating the efficacy of this hybrid strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of hierarchical agents and element-level reasoning is relatively novel within the document understanding field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, 15+ baseline models, and comprehensive ablation and error analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive case studies, and rigorous method descriptions.
- Value: ⭐⭐⭐⭐ The framework is highly generalizable and has direct applicability to enterprise-scale document understanding scenarios.
To be supplemented after a thorough reading of the paper.

## Background & Motivation
To be supplemented after a thorough reading of the paper.

## Method
To be supplemented after a thorough reading of the paper.

## Key Experimental Results
To be supplemented after a thorough reading of the paper.

## Highlights & Insights
To be supplemented after a thorough reading of the paper.

## Limitations & Future Work
To be supplemented after a thorough reading of the paper.

## Related Work & Insights
To be supplemented after a thorough reading of the paper.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ICLR 2026\] RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](../../ICLR2026/information_retrieval/ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[AAAI 2026\] Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](../../AAAI2026/information_retrieval/multimodal_deepresearcher_generating_text-chart_interleaved_.md)

</div>

<!-- RELATED:END -->
