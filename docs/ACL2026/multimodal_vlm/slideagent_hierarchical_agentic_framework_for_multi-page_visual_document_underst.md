---
title: >-
  [Paper Note] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding
description: >-
  [ACL 2026][Multimodal VLM][Multi-page document understanding] SlideAgent is proposed as a hierarchical agentic framework that constructs structured knowledge representations through three specialized levels—global, page…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multi-page document understanding"
  - "hierarchical agent"
  - "visual document QA"
  - "slide understanding"
  - "element-level reasoning"
date: 2026-05-08
content_hash: 322b33ab6f169909
---

# SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding

**Conference**: ACL 2026  
**arXiv**: [2510.26615](https://arxiv.org/abs/2510.26615)  
**Code**: [SlideAgent](https://SlideAgent.github.io/)  
**Area**: Information Retrieval / Document Understanding  
**Keywords**: Multi-page document understanding, hierarchical agent, visual document QA, slide understanding, element-level reasoning

## TL;DR

SlideAgent is proposed as a hierarchical agentic framework that constructs structured knowledge representations through three specialized levels—global, page, and element. It significantly enhances fine-grained understanding of multi-page visual documents, particularly slides.

## Background & Motivation

**Background**: Multi-page visual documents (e.g., financial reports, academic presentations, technical manuals) are prevalent in high-stakes fields such as finance, science, and education. These documents rely not only on text but also on layout, icons, color coding, and cross-page references to convey information.

**Limitations of Prior Work**: Current Multi-modal Large Language Models (MLLMs) face three challenges in multi-page visual document understanding: (1) **Insufficient fine-grained reasoning** — MLLMs tend to process pages holistically, ignoring element-level details (e.g., specific data segments in charts); (2) **Lack of domain-specific visual semantics** — pre-training based on natural images results in poor understanding of professional charts, icon meanings, and spatial layouts; (3) **Metadata dependency** — many systems rely on clean document metadata (e.g., chart bounding boxes, hierarchical labels), which is often missing or corrupted in real-world scenarios.

**Key Challenge**: MLLMs may fail when reasoning about a full page (e.g., miscounting categories in a chart) but can correctly identify information once relevant charts are cropped. This indicates that models possess reasoning capabilities but lack effective fine-grained information extraction mechanisms.

**Goal**: To build a universal agent framework capable of processing multi-page multi-modal documents without relying on metadata, achieving precise document understanding through hierarchical knowledge construction and selective agent activation.

**Core Idea**: Drawing from human information processing, document understanding is decomposed into three levels: global (overall theme), page (single-page features and cross-page relationships), and element (fine-grained parsing of charts, text blocks, and icons). Specialized agents at each level collaborate during both knowledge construction and reasoning phases.

## Method

### Overall Architecture

SlideAgent operates in two phases: (1) **Knowledge Construction** — a hierarchical, query-agnostic knowledge base $\mathcal{K}=\{\mathcal{K}_g, \mathcal{K}_p, \mathcal{K}_e\}$ is built top-down; (2) **Reasoning** — specialized agents are selectively activated based on user query classification for multi-level retrieval and answer synthesis. The framework is model-agnostic and can be paired with various backbones such as GPT-4o or InternVL3-8B.

### Key Designs

1.  **Hierarchical Knowledge Construction**:

    - **Function**: Constructs a structured, query-agnostic knowledge representation for the document.
    - **Mechanism**: The global agent $\mathcal{M}_g$ generates document-level summaries and themes by sampling the first few pages; the page agent $\mathcal{M}_p$ processes pages sequentially, generating page-level knowledge $\mathcal{K}_p^i = \mathcal{M}_p(v_i, \mathcal{K}_g^{(0)}, \mathcal{K}_p^{i-1})$ conditioned on global and previous page context; the element agent $\mathcal{M}_e$ decomposes each page into elements (text blocks, charts, icons) via layout parsing and generates semantic roles and functional descriptions for each.
    - **Design Motivation**: Global knowledge provides the thematic framework, page knowledge provides sequential context and cross-page associations, and element knowledge provides fine-grained spatial and content information—the three are complementary and essential.

2.  **Query Classification & Selective Activation**:

    - **Function**: Precisely activates the required subset of agents based on the query type to avoid unnecessary computation and noise.
    - **Mechanism**: Queries are classified into four categories: global understanding (global agent only), factual queries (page + element agents), multi-hop reasoning (all agents), and layout/visual relations (element agent). All agents are activated by default for unclassifiable queries.
    - **Design Motivation**: Different queries require different levels of granularity. Excessive agent activation increases computational overhead and may introduce noise; selective activation balances efficiency and accuracy.

3.  **Subquery Generation & Multi-level Retrieval**:

    - **Function**: Expands the original query into multiple subqueries for precise retrieval at the page and element levels.
    - **Mechanism**: Key entities are extracted from the query to generate subqueries. The original and subqueries are concatenated for joint retrieval of top-k pages and their elements. Sparse (BM25), dense (SFR), and multi-modal (COLPALI) retrievers are supported.
    - **Design Motivation**: Original queries are often short, leading to noisy retrieval; subquery generation enhances semantic coverage, particularly benefiting multi-hop reasoning scenarios.

### Loss & Training

Ours utilizes a training-free approach—all agents are implemented via prompt engineering on existing MLLMs without additional training or fine-tuning. During knowledge construction, carefully designed prompt templates guide the agents to generate structured knowledge. Global knowledge incorporates a refine step (a single full-field rewrite) to synthesize global information from all pages and reduce bias toward initial pages.

## Key Experimental Results

### Main Results

| Dataset | Metric | SlideAgent (GPT-4o) | GPT-4o | Gain |
| :--- | :--- | :--- | :--- | :--- |
| SlideVQA | Overall | 84.9 | 77.0 | +7.9% |
| TechSlides | Overall | 70.9 | 63.4 | +7.5% |
| FinSlides | Overall | 85.5 | 80.0 | +5.5% |
| InfoVQA | Overall | 79.6 | 69.0 | +10.6% |
| SlideVQA (InternVL3) | Overall | 72.7 | 63.0 | +9.8% |

### Ablation Study

| Configuration | Key Metric (Overall) | Description |
| :--- | :--- | :--- |
| w/o Page Agent | -6.3 (GPT-4o) | Largest drop; page-level reasoning is critical for cross-page coherence. |
| w/o Element Agent | -4.6 (GPT-4o) | Fine-grained reasoning is vital for numerical questions. |
| w/o Global Agent | -2.8 (GPT-4o) | Smallest drop; low-level agents partially embed global context. |
| w/o Subquery | -5.0 (GPT-4o) | Impact is particularly significant in retrieval scenarios. |

### Key Findings
- Hierarchical knowledge construction improves not only QA performance but also page-level retrieval (text retriever SFR achieved a +6.4 MRR gain).
- Multi-hop reasoning queries saw the largest improvement (+9.8%), demonstrating the value of structured knowledge for complex reasoning.
- In the oracle setting (providing ground-truth pages), a +7.7% gain was still achieved, indicating the independent value of element-level retrieval.
- Only 12.5% of errors were attributed to OCR/parsing failures, with most errors originating from question ambiguity and answer labeling issues.

## Highlights & Insights
- **Hierarchical Divide-and-Conquer**: Adopting a "global-page-element" processing model inspired by human cognition is both systematic and intuitive, facilitating modular extension.
- **Training-free Plug-and-Play**: The framework is entirely based on prompt engineering and existing MLLMs, allowing direct application to any backbone model.
- **Necessity of Element-level Reasoning**: The visual case in Figure 1 demonstrates MLLM failure in whole-page reasoning vs. success after element-level cropping, providing compelling evidence.
- **Knowledge Construction Gains for Retrieval**: Generated structured knowledge (page descriptions and subqueries) serves both QA and as enhancement signals for retrieval.
- **Model Agnosticism**: Consistent and significant improvements across diverse backbones like GPT-4o and InternVL3-8B.

## Limitations & Future Work
- Element boundaries depend on OCR and layout parsing tools; quality may vary based on the choice of tools.
- Initial global knowledge sampling (first three pages) may lack representation for long documents; content-based page selection could be explored.
- Primarily utilizes text retrievers (SFR); the potential of multi-modal retrievers requires further investigation.
- Multi-turn dialogue scenarios remain unaddressed; extension to interactive document QA is a key direction.
- Knowledge construction incurs high computational overhead, requiring individual MLLM calls per page.

## Related Work & Insights
- **vs ViDoRAG**: While ViDoRAG also uses multi-agent architectures, SlideAgent's hierarchical design and element-level parsing are more detailed, achieving superior performance across all datasets.
- **vs VDocRAG**: VDocRAG combines retrieval and reasoning but lacks element-level decomposition; SlideAgent shows a distinct advantage in numerical reasoning (Num).
- **vs COLPALI**: As a pure multi-modal retrieval method, COLPALI is compared to SlideAgent, which demonstrates that combining text retrieval with structured knowledge can match or exceed multi-modal retrieval performance.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of hierarchical agents and element-level reasoning is novel in the document understanding field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 4 datasets, 15+ baselines, and detailed ablation and error analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive case studies, and rigorous methodology descriptions.
- Value: ⭐⭐⭐⭐ High generalizability with direct application potential for enterprise-level document understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction](texocr_advancing_document_ocr_models_for_compilable_page-to-latex_reconstruction.md)
- [\[ACL 2026\] HierVA: Hierarchical Visual Agent — Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning](hierarchical_visual_agent_managing_contexts_in_joint_image-text_space_for_advanc.md)
- [\[ACL 2026\] A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends](a_survey_on_mllm-based_visually_rich_document_understanding_methods_challenges_a.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](../../AAAI2026/multimodal_vlm/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)

</div>

<!-- RELATED:END -->
