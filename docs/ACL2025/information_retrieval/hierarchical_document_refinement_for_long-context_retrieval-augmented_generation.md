---
title: >-
  [Paper Note] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] This paper proposes LongRefiner, a plug-and-play long-document refinement system. Through three stages—dual-level query analysis, hierarchical document structuring, and adaptive refinement—it outperforms full-text input on 7 QA datasets using **only 1/10 of the token budget**, with latency running at just 1/10 of the strongest baseline.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Document Refinement"
  - "Hierarchical Structure"
  - "XML Format"
  - "Long Context"
  - "Multi-Task LoRA"
date: 2026-05-08
content_hash: ee5e87df157f6af3
---

# Hierarchical Document Refinement for Long-context Retrieval-augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2505.10413](https://arxiv.org/abs/2505.10413)  
**Code**: [GitHub](https://github.com/ignorejjj/LongRefiner)  
**Area**: Retrieval-augmented Generation / Long-context Processing  
**Keywords**: RAG, Document Refinement, Hierarchical Structure, XML Format, Long Context, Multi-Task LoRA  

## TL;DR

This paper proposes LongRefiner, a plug-and-play long-document refinement system. Through three stages—dual-level query analysis, hierarchical document structuring, and adaptive refinement—it outperforms full-text input on 7 QA datasets using **only 1/10 of the token budget**, with latency running at just 1/10 of the strongest baseline.

## Background & Motivation

**Background**: RAG systems enhance the knowledge coverage and factual accuracy of LLMs by retrieving external documents. In practical applications, search engines often return full-length documents that contain a vast amount of query-irrelevant information.

**Limitations of Prior Work**:
   - **Low Signal-to-Noise Ratio**: Valuable information in long documents is overwhelmed by a large amount of irrelevant content, making it difficult for models to focus.
   - **High Computational Overhead**: Processing full-length long documents significantly increases the input context length, leading to high inference costs.
   - **Deficiencies of Existing Refinement Methods**: Chunk-based methods can only process short text fragments, lacking a global perspective; perplexity-based methods assess token relevance through perplexity, which is overly coarse-grained.

**Key Challenge**: While full-length documents contain the required information, they introduce too much noise. Conversely, chunk-based methods reduce length but lose the overall structure and contextual relationships of the document.

**Goal**: How to refine long documents **efficiently and faithfully**, preserving critical information while significantly reducing token count.

**Key Insight**: Full documents naturally contain rich structural information (such as logical connections and content organization hierarchies). Leveraging these structures enables more precise information extraction than chunk-based methods.

**Core Idea**: Transform unstructured long documents into hierarchical document trees, adaptively refine them according to query demands, and compress the representation drastically using the XML format.

## Method

### Overall Architecture

LongRefiner integrates three capabilities into a single base model (learned via multi-task LoRA): (1) Dual-Level Query Analysis—determining whether a query requires local or global information; (2) Hierarchical Document Structuring—transforming unstructured documents into document trees; and (3) Adaptive Refinement—selecting which parts of the document to preserve based on query demands.

### Key Designs

1. **Dual-Level Query Analysis**:

    - Define two information scopes: **Local Level** (requiring local knowledge by locating specific paragraphs) and **Global Level** (requiring comprehensive context to attain a global understanding).
    - Use a teacher LLM to annotate queries in the training set with binary labels.
    - Fine-tune the refiner model to predict special tokens (`[Local]` or `[Global]`), and apply softmax on the generation probabilities of both to derive a continuous query scope score $r_q$.

2. **Hierarchical Document Structuring**:

    - Model the document as a **document tree** $D_{\text{str}} = (\mathcal{N}, \mathcal{R})$, where nodes represent sections/subsections/paragraphs and edges denote hierarchical containment.
    - **XML Representation**: Design four XML tags—`<section>`, `<subsection>`, `<skip>`, and `<br>`—to flatten the tree structure into text. The `<skip>` tag omits middle content within paragraphs (keeping only the first and last $k$ tokens), compressing the output token count to approximately **1/10** of the original text.
    - Train the model to learn the mapping from the original text $D$ to $D_{\text{xml}}$. During inference, use the original text and a parser algorithm to reconstruct the complete document tree.
    - The generation process consists of two steps: first generating the hierarchical structure (section headings), and then filling in the content (using `<skip>` to bypass intermediate parts).

3. **Adaptive Refinement**:

    - Compute relevance scores between each paragraph/section and the query based on the document tree structure.
    - Adaptively determine the amount of information to preserve based on the query scope score $r_q$ obtained from query analysis.
    - For Local-type queries, retain less but highly precise content; for Global-type queries, preserve more context.

4. **Wikipedia Label Collection**: Utilize the natural hierarchical structures of Wikipedia articles (titles, subtitles, paragraphs) as ground truth to train the document structuring model.

### Training & Inference Optimization

- **Multi-task LoRA Learning**: The three tasks share the same base model, distinguished via LoRA adapters.
- **Offline/Online Decoupling**: Document structuring is executed offline (preprocessing the corpus), whereas the online phase only requires query analysis and adaptive refinement. This setup processes very few input/output tokens, lowering latency to only 25% of the standard configuration.

## Key Experimental Results

### Experimental Settings

- **Datasets**: 7 QA datasets—Single-hop (NQ, TriviaQA, PopQA), Multi-hop (HotpotQA, 2WikiMultiHopQA), Long-form (ASQA, ELI5).
- **Generator**: LLaMA-3.1-8B-Instruct (64K context).
- **Refiner**: Qwen2.5-3B-Instruct + LoRA.
- **Retrieval**: Retrieve the top-8 full Wikipedia documents for each query.
- **Budget Constraints**: Refined output is constrained to 2K tokens.

### Main Results

The proposed LongRefiner achieves the best performance across all 7 datasets. Key findings:

1. **Performance**: Achieves optimal performance across all datasets, outperforming the perplexity-based method (LongLLMLingua) by over 9%.
2. **Efficiency**: Exhibits latency comparable to standard retrieval methods and is far lower than perplexity-based methods.
3. **Ours vs. Full Text**: Surpasses full-text input performance across 6/7 datasets while using **only 1/10 of the tokens** (except on PopQA, where documents are already short and contain minimal noise).

### Ablation Study

| Method | Single-hop (EM) | Multi-hop (Acc) | Long-form (F1) |
|------|----------------|-----------------|----------------|
| LongRefiner | **62.3** | **37.4** | **30.2** |
| w/o Query Analysis | 60.3 | 36.2 | 29.6 |
| w/o Doc. Structuring | 45.7 | 29.9 | 27.1 |
| w/o Adaptive Refine. | 57.7 | 35.3 | 29.2 |

- The document structuring module is the most critical: removing it drops performance by approximately 20% (degrades to basic chunking).
- Query analysis and adaptive refinement contribute about 2-3% improvement each.

### Further Analysis

- **Model Scale**: Larger structuring models lead to better document structure quality and QA performance.
- **Training Data Size**: When training data is insufficient, the model tends to generate coarse-grained structures (fewer sections, larger content blocks), which might seem to have high recall but offer inaccurate structural details.
- **Document Length**: For short documents, the advantage of LongRefiner is not prominent (since we already have low noise), but in long documents, it significantly outperforms full-text methods and LongLLMLingua.
- **Scoring Model Choice**: Reranker is optimal, Embedding models follow next with higher efficiency, and BM25 performs the worst.
- **Cross-Generator Validation**: Consistently effective when validated using Qwen2.5-7B-Instruct.

## Highlights & Insights

1. **Document Structure is Key to Long-Context Refinement**: Unlike coarse methods based on perplexity or semantic similarity, leveraging the internal hierarchical structure of documents enables more meaningful information organization and refinement.
2. **Clever Design of XML Format**: Using a few XML tags combined with a `<skip>` bypassing mechanism compresses the document tree representation to 1/10, significantly reducing the model's output token count.
3. **Plug-and-Play**: Serving as an independent refinement module, it can be seamlessly plugged into any RAG pipeline without modifying the generator.
4. **Offline/Online Decoupling**: The high-cost portion of document structuring is executed offline, making online inference extremely lightweight.
5. **Multi-task Sharing**: The three distinct capabilities share the same base model, achieving resource efficiency via LoRA.

## Limitations & Future Work

1. Only handles plain text, falling short on complex documents containing tables, figures, or hyperlinks.
2. Trained on Wikipedia; adapting to vertical domains (e.g., enterprise, finance) might necessitate re-annotation.
3. Errors during XML parsing can result in a certain degree of information loss.
4. Refinement may conversely hurt performance in low-noise scenarios with short documents like PopQA.
5. The refiner introduces extra inference overhead; although minimal, it still needs consideration in scenarios with ultra-low latency requirements.

## Related Work & Insights

- **RAG**: Combining retrieval with generation (e.g., RAG, RETRO, REALM).
- **Knowledge Refinement Methods**: Divided into hard refinement (token deletion/summarization/chunk selection) and soft refinement (vector encoding).
- **Long-context Processing**: Perplexity-based token compression methods like LongLLMLingua.

## Rating

⭐⭐⭐⭐⭐ — The problem is highly important (long-context RAG is a core bottleneck in practical deployment), the methodology is elegantly designed (hierarchical structure + XML compression), the evaluation is comprehensive (7 datasets + multi-dimensional analysis), and it holds strong engineering utility (plug-and-play + offline/online decoupling). This represents a high-quality contribution in RAG knowledge refinement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)
- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)
- [\[ACL 2025\] FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)
- [\[ICML 2026\] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](../../ICML2026/information_retrieval/hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
