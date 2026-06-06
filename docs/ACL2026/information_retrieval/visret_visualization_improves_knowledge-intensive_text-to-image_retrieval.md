---
title: >-
  [Paper Note] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval
description: >-
  [ACL 2026][Information Retrieval & RAG][Text-to-Image Retrieval] This paper proposes Visualize-then-Retrieve (VisRet), a new paradigm that visualizes text queries into images via T2I generation models before performing r…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Text-to-Image Retrieval"
  - "Visualized Query"
  - "Cross-modal Alignment"
  - "Retrieval-Augmented Generation"
  - "Modality Projection"
date: 2026-05-08
content_hash: b5d3ca68bb43dd0c
---

# VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval

**Conference**: ACL 2026  
**arXiv**: [2505.20291](https://arxiv.org/abs/2505.20291)  
**Code**: [GitHub](https://github.com/xiaowu0162/Visualize-then-Retrieve)  
**Area**: Image Generation  
**Keywords**: Text-to-Image Retrieval, Visualized Query, Cross-modal Alignment, Retrieval-Augmented Generation, Modality Projection

## TL;DR

This paper proposes Visualize-then-Retrieve (VisRet), a new paradigm that visualizes text queries into images via T2I generation models before performing retrieval within the image modality. It achieves an average nDCG@30 improvement of 0.125 (CLIP) and 0.121 (E5-V) across four benchmarks, with a 15.7% increase in downstream VQA accuracy on Visual-RAG-ME.

## Background & Motivation

**Background**: Text-to-image (T2I) retrieval is a critical component for knowledge-intensive applications. Common methods rank candidate images by their similarity to text queries in a shared embedding space. While cross-modal embedding models (e.g., CLIP, E5-V) continue to improve, cross-modal similarity alignment still faces fundamental challenges.

**Limitations of Prior Work**: Cross-modal embeddings often behave as "bags of concepts," failing to capture structured visual relationships such as pose, perspective, and spatial layout. For instance, for a query like "a bar-headed goose with wings spread," embedding models may match the species but fail to recognize subtle visual features like wing posture or low-angle perspective. Current improvements (query rewriting, multi-stage re-ranking) remain constrained by the inherent difficulty of cross-modal similarity alignment.

**Key Challenge**: Text is inherently limited in its ability to exhaustively describe complex visual-spatial relationships, and cross-modal retrievers have an intrinsic weakness in identifying subtle visual-spatial features. Encoding all visual requirements into a text query may even harm retrieval performance due to embedding quality constraints.

**Goal**: To propose a retrieval paradigm that bypasses the weaknesses of cross-modal similarity matching by projecting text queries into the image modality, leveraging the superior performance of retrievers in single-modality retrieval.

**Key Insight**: Visualization provides a more intuitive and expressive medium than text for expressing compositional concepts (entity + pose + spatial relations). Performing retrieval within the image modality avoids the pitfalls of cross-modal retrievers and exploits their stronger capabilities in single-modality tasks.

**Core Idea**: Decomposing T2I retrieval into two stages: "text-to-image modality projection" and "image-to-image in-modality retrieval." The text query is visualized via a T2I generation model, and image-to-image retrieval is performed directly using the generated images.

## Method

### Overall Architecture

VisRet consists of two stages: (1) Modality Projection: An LLM transforms the original text query into T2I instructions, and a T2I generation model generates $m$ visualization images $\{v_1,\ldots,v_m\} \equiv \mathcal{T}(q)$; (2) In-modality Retrieval: Each generated image is used independently for retrieval, and results are aggregated via Reciprocal Rank Fusion (RRF) to produce the final ranked list.

### Key Designs

1.  **Modality Projection**:
    *   **Function**: Converts text queries into image queries to make key visual-spatial requirements explicit.
    *   **Mechanism**: Given an original query $q$, an LLM drafts T2I instructions $q'$ in the text space, describing images that satisfy the latent features of $q$. A T2I generation method (e.g., Stable Diffusion) then projects $q'$ into $m$ images $\{v_1,\ldots,v_m\}$. Diversity is introduced through multiple samplings.
    *   **Design Motivation**: Visualized queries can simultaneously depict required entities, poses, and perspectives, which are limited by cross-modal matching quality when encoded solely in text.

2.  **In-modality Retrieval and RRF Aggregation**:
    *   **Function**: Completes retrieval within the image modality and aggregates results from multiple visualizations.
    *   **Mechanism**: Each generated image $v_i$ independently retrieves a ranked list $\mathcal{R}(v_i, \mathcal{I})$. These $m$ lists are fused using RRF: $\text{score}_{\text{RRF}}(r) = \sum_{i=1}^{m} \frac{1}{\lambda + \text{rank}_i(r)}$, where $\lambda$ controls the impact of low-ranking items. The top-$k$ results with the highest scores are selected.
    *   **Design Motivation**: Operating entirely within the image modality avoids cross-modal retriever weaknesses. Multi-image aggregation increases query diversity.

3.  **Visual-RAG-ME Benchmark Construction**:
    *   **Function**: Provides a retrieval evaluation benchmark for comparing visual features of multiple entities.
    *   **Mechanism**: Extends Visual-RAG by constructing questions that compare visual features of two biologically similar entities (e.g., which one has a lighter color or smoother surface). Candidates are identified via BM25, comparison questions are manually constructed, and labels are retrieved from iNaturalist annotations. It contains 50 high-quality queries.
    *   **Design Motivation**: Existing benchmarks primarily evaluate single-entity retrieval and lack scenarios requiring reasoning across visual features of multiple entities, which is a significant challenge for T2I retrieval.

### Loss & Training

VisRet is a training-free, plug-and-play method that does not require modifications to the retriever or pre-computed image embedding indices. It only requires a one-time use of an LLM for T2I instructions and a T2I model for image generation.

## Key Experimental Results

### Main Results

**nDCG@30 across four benchmarks (CLIP Retriever)**

| Method | Visual-RAG | Visual-RAG-ME | INQUIRE-Rerank-Hard | COCO-Hard |
| :--- | :--- | :--- | :--- | :--- |
| Original Query | 0.385 | 0.435 | 0.412 | 0.042 |
| LLM Rewriting | 0.395 | 0.572 | 0.407 | 0.093 |
| Corpus Captioning (BLIP) | 0.271 | 0.371 | 0.401 | 0.153 |
| VISA Reranking | 0.388 | 0.457 | 0.000 | 0.000 |
| **VisRet** | **0.438** | **0.605** | **0.455** | **0.108** |

**nDCG@30 across four benchmarks (E5-V Retriever)**

| Method | Visual-RAG | Visual-RAG-ME | INQUIRE-Rerank-Hard | COCO-Hard |
| :--- | :--- | :--- | :--- | :--- |
| Original Query | 0.407 | 0.486 | 0.407 | 0.178 |
| LLM Rewriting | 0.391 | 0.566 | 0.412 | 0.182 |
| **VisRet** | **0.461** | **0.622** | **0.425** | **0.205** |

### Ablation Study

**Impact of T2I Generation Models on Visual-RAG-ME Performance (CLIP Retriever)**

| T2I Model | N@1 | N@10 | N@30 |
| :--- | :--- | :--- | :--- |
| Stable Diffusion 3.5 | 0.270 | 0.467 | 0.484 |
| FLUX.1-dev | 0.320 | 0.501 | 0.494 |
| DALL-E 3 | 0.346 | 0.554 | 0.553 |
| gpt-image-1 (high quality) | **0.460** | **0.632** | **0.605** |

**Multi-image Aggregation vs. Single-image (CLIP Retriever)**

| Benchmark | 3 Images N@30 | 1 Image N@30 |
| :--- | :--- | :--- |
| Visual-RAG | 0.438 | 0.425 |
| Visual-RAG-ME | 0.605 | 0.602 |

### Key Findings

*   VisRet achieves an average nDCG@10 improvement of 0.109 (38%↑) for the CLIP retriever and 0.078 (23%↑) for E5-V.
*   Downstream VQA accuracy: Top-1 retrieval gains 3.8% on Visual-RAG and 15.7% on Visual-RAG-ME.
*   T2I generation quality is a key bottleneck: gpt-image-1 significantly outperforms Stable Diffusion 3.5. Failure modes include lack of focus, factual errors, and poor instruction following.
*   Single-image visualization only slightly reduces performance; the benefit of multi-image aggregation comes from increased query diversity.
*   While visualized queries improve retrieval, they cannot replace real images as independent knowledge sources.

## Highlights & Insights

*   Novel and Practical: Bypasses fundamental cross-modal alignment difficulties through "visualize-then-retrieve," offering a simple yet powerful perspective.
*   Training-free and Plug-and-play: No retraining of retrievers or modification of existing infrastructure is required, allowing direct use of existing image embedding indices.
*   Visual-RAG-ME benchmark fills the gap in evaluating retrieval for multi-entity visual feature comparison.
*   VisRet's actual latency is lower than VISA re-ranking (approx. 5× faster) because VISA requires an LVLM to process top-k candidates.

## Limitations & Future Work

*   Performance highly depends on T2I generation quality; gains from weaker models (e.g., Stable Diffusion) are limited.
*   Generated images may contain factual errors (e.g., inaccurate species appearance), affecting retrieval quality.
*   Evaluation is primarily focused on the natural species domain; effectiveness in other knowledge-intensive fields (e.g., medicine, architecture) remains to be verified.
*   The computational cost of T2I generation is higher than simple query rewriting.

## Related Work & Insights

*   **vs LLM Query Rewriting**: Query rewriting still matches in the text-image cross-modal space, whereas VisRet shifts completely into the image modality to avoid cross-modal weaknesses.
*   **vs VISA Reranking**: VISA relies on LVLM to process top-k candidates, with costs scaling linearly with $k$ and performance limited by initial retrieval quality; VisRet fundamentally changes the query modality.
*   **vs Corpus Captioning**: Converting images to text results in information loss, which is particularly detrimental in knowledge-intensive scenarios.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ Reframing T2I retrieval as "visualization + image-to-image retrieval" is a unique and elegant perspective.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across four benchmarks, two retrievers, multiple ablations, and downstream VQA.
*   Writing Quality: ⭐⭐⭐⭐ Clear motivation, concise methodology, and intuitive charts.
*   Value: ⭐⭐⭐⭐ Provides a new paradigm for knowledge-intensive T2I retrieval, with practical utility enhanced by its training-free nature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ACL 2026\] A Survey of Reasoning-Intensive Retrieval: Progress and Challenges](a_survey_of_reasoning-intensive_retrieval_progress_and_challenges.md)
- [\[ACL 2026\] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](../../AAAI2026/information_retrieval/knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](../../ICLR2026/information_retrieval/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)

</div>

<!-- RELATED:END -->
