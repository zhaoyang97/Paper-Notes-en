---
title: >-
  [Paper Note] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][Retrieval-Augmented Generation] This paper proposes FlexRAG, an open-source RAG framework oriented towards research and prototyping. It supports text, multimodal, and web retrieval, achieving an order-of-magnitude lower resource overhead than similar frameworks (such as FlashRAG) through memory mapping and asynchronous processing.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "RAG Framework"
  - "Dense Retrieval"
  - "Web Retrieval"
  - "Multimodal RAG"
date: 2026-05-08
content_hash: 20519f22018411f4
---

# FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2506.12494](https://arxiv.org/abs/2506.12494)  
**Code**: [ictnlp/FlexRAG](https://github.com/ictnlp/FlexRAG)  
**Area**: NLP / RAG Framework  
**Keywords**: Retrieval-Augmented Generation, RAG Framework, Dense Retrieval, Web Retrieval, Multimodal RAG

## TL;DR

This paper proposes FlexRAG, an open-source RAG framework oriented towards research and prototyping. It supports text, multimodal, and web retrieval, achieving an order-of-magnitude lower resource overhead than similar frameworks (such as FlashRAG) through memory mapping and asynchronous processing.

## Background & Motivation

Retrieval-Augmented Generation (RAG) has become a core technology in LLM applications—compensating for the knowledge limitations of models by dynamically retrieving information from external knowledge sources. Although numerous RAG frameworks exist (such as LangChain, LlamaIndex, FlashRAG, etc.), the authors identify several core issues in existing frameworks through analysis:

**Difficulties in Algorithm Replication and Sharing**: RAG systems involve multiple components and complex environment configurations, making it difficult for researchers to replicate others' work accurately.

**Excessive Engineering Burden**: Building a complete RAG system requires handling substantial engineering issues such as data preprocessing, index construction, and API integration, which distracts from core research efforts.

**Incomplete Technical Coverage**: Most existing frameworks focus solely on retrieval strategies, lacking support for cutting-edge directions like multimodal retrieval, web retrieval, and document parsing/chunking.

**High System Overhead**: High computational costs of retrieval and generation components limit researchers with constrained resources.

FlexRAG aims to be a "researcher-friendly" full-lifecycle RAG development tool.

## Method

### Overall Architecture

FlexRAG comprises 12 core modules, divided into four categories by function:

- **Models**: Encoders, Rerankers, Generators
- **Retrievers**: Web Retrievers, FlexRetriever, API Retrievers
- **System Development**: Preprocessors, Refiners, Assistants
- **Evaluation**: Tasks and Metrics

### Key Designs

1. **FlexRetriever—The Core Retrieval Engine**:

    - Supports **MultiField** and **MultiIndex** retrieval paradigms.
    - Documents can be decomposed into semantic fields such as titles, abstracts, and bodies, with each field indexed independently.
    - Supports sparse retrieval (BM25s) and dense retrieval (Contriever, E5, BGE M3, etc.).
    - Key Optimization: Adopts **Memory Map** and **IVFPQ empirical formulas** as the default configuration, reducing CPU and memory resource consumption to only 1/10 of similar frameworks.
    - Deeply integrated with the HuggingFace Hub, enabling one-click publication and sharing of retrievers.

2. **Web Retriever**:

    - Three-role design: Web Seeker (resource localization) $\rightarrow$ Web Downloader (downloading) $\rightarrow$ Web Reader (content extraction).
    - Built-in SimpleWebRetriever (search engine + page parsing) and WikipediaRetriever.
    - Automatically converts HTML into LLM-friendly formats.

3. **Preprocessors**:

    - Document Parser: Extracts readable content from formats like PDF, DOCX, and HTML.
    - Chunker: Segments content into smaller semantic units.
    - Knowledge Preprocessor: Filters and structurally optimizes extracted content.
    - Addresses the often underestimated but critical stage of "data preparation" in practical RAG.

4. **Refiners**:

    - Prompt Squeezer: Compresses and optimizes input prompts.
    - Context Repacker: Reorganizes retrieval results to prevent key information from being ignored.
    - Context Summarizer: Condenses retrieved contexts to reduce inference overhead.
    - These three modules directly address the problem of "retrieved knowledge being underutilized by LLMs."

5. **Asynchronous Processing and Persistent Cache**:

    - Computationally intensive components utilize asynchronous functions to achieve high throughput.
    - Persistent cache mechanisms reduce redundant retrieval overhead.

### Loss & Training

FlexRAG itself is an engineering framework and does not introduce new training methodologies. However, it supports training and inference for all mainstream retrieval and ranking models, including:
- Dense Encoders: Contriever, E5, BGE M3
- Rerankers: Cross-Encoder, ColBERT, T5-style, GPT-style
- Generators: Qwen2, Llama 3.1, ChatQA2, etc.

## Key Experimental Results

### Main Results: Performance of ModularAssistant on Three Major RAG Tasks (Table)

| Method | PopQA F1 | NQ F1 | TriviaQA F1 | Average F1 |
|------|----------|-------|-------------|---------|
| BM25s | 57.88 | 38.79 | 65.93 | 54.20 |
| Contriever | 64.14 | 49.67 | 70.36 | 61.39 |
| E5 base | 59.74 | 50.05 | 71.66 | 60.48 |
| BGE M3 | 63.65 | 50.98 | 71.92 | 62.18 |
| BGE-reranker-M3 | **66.02** | **50.94** | **74.58** | **63.85** |
| ColBERT-v2 | 65.44 | 47.18 | 72.13 | 61.58 |
| rankGPT | 63.11 | 49.50 | 70.13 | 60.91 |

*BGE-reranker-M3 performs best across all tasks, with the improvement brought by the reranker being consistent and significant.*

### Resource Overhead Comparison (Table)

| Metric | FlexRAG | FlashRAG | Gap |
|------|---------|----------|------|
| Average Wall-Clock Time | Low | Order of magnitude higher | ~10x |
| Total CPU Time | Low | Order of magnitude higher | ~10x |
| Average Memory Usage | Low | Several times higher | ~3-5x |
| Total Memory Usage | Low | Several times higher | ~3-5x |

*FlexRAG consistently outperforms FlashRAG across all batch sizes.*

### Key Findings

1. **Rerankers are the Key Lever of RAG Performance**: Selecting top-10 with a reranker after retrieving top-100 improves average F1 from 61.39 to 63.85, significantly outperforming direct top-10 retrieval.

2. **Approximate Nearest Neighbor Indexing has Little Impact**: The F1 gap among FLAT, Faiss, and ScaNN indexing methods is within 1%, but memory and speed differences are significant.

3. **Generator Choice Matters**: Qwen2-7B and ChatQA2-7B demonstrate similar performance, while Llama 3.1-8B scores slightly lower on EM, indicating that the instruction-following capability of the generator affects the final quality.

4. **FlexRAG is Most Efficient at Batch Size = 1**: Since the Tokenizer runs in single-process mode, it avoids process scheduling overhead.

5. **Memory Mapping is the Core Source of the Performance Advantage**: Combined with index parameter optimization from the ANN-Benchmark toolkit, it achieves low resource consumption.

## Highlights & Insights

- **Research-oriented design** is highly focused: unified configuration management, standardized evaluation, HuggingFace Hub integration, and sample repositories—each feature directly addressing researchers' pain points.
- **Full-lifecycle coverage**: One-stop solution spanning document parsing $\rightarrow$ chunking $\rightarrow$ indexing $\rightarrow$ retrieval $\rightarrow$ reranking $\rightarrow$ context refinement $\rightarrow$ generation $\rightarrow$ evaluation.
- Support for **multimodal and web retrieval** serves as an important differentiator compared to competitors like FlashRAG.
- **Low resource consumption** makes large-scale retrieval experiments feasible on ordinary servers.

## Limitations & Future Work

1. **Basic Experiments**: Only demonstrates the performance of ModularAssistant on three standard QA tasks, lacking actual evaluation on Web RAG and multimodal RAG.
2. **Lack of Comparison with Well-known Frameworks like LangChain and LlamaIndex**: Only resource overhead comparisons with FlashRAG are provided.
3. **Mainly Engineering Contribution**: No new retrieval algorithms or generation strategies are proposed.
4. **Insufficient Scalability Verification**: Not tested on ultra-large-scale knowledge bases (e.g., complete Common Crawl).
5. **GUI is Merely a Prototype**: Still far from a production-grade web interface.

## Related Work & Insights

- **FlashRAG** (Jin et al., 2024): The closest competitor architecturally, but has high resource overhead.
- **EasyRAG** (Feng et al., 2024): Targets network operation and maintenance scenarios.
- **RaLLe** (Hoshi et al., 2023): Focuses on evaluation.
- **AutoRAG-HP** (Fu et al., 2024): Automated hyperparameter tuning.
- The differentiation of FlexRAG lies in its **comprehensiveness** (multi-scenarios), **efficiency** (memory mapping), and **shareability** (HuggingFace integration).

## Rating

- **Novelty**: ⭐⭐⭐ Mainly engineering integration, without methodology-level innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers ablation comparisons of retrievers, indexers, rerankers, and generators, but with somewhat narrow scenarios.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear module descriptions, intuitive architecture diagrams, and precise functional positioning.
- **Value**: ⭐⭐⭐⭐ Possesses decent value as an open-source tool, but its features are not highly distinct, and many competitors exist.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)
- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)
- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)
- [\[ICLR 2026\] When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation](../../ICLR2026/information_retrieval/when_to_use_graphs_in_rag_a_comprehensive_analysis_for_graph_retrieval-augmented.md)
- [\[ACL 2025\] VISA: Retrieval Augmented Generation with Visual Source Attribution](visa_retrieval_augmented_generation_with_visual_source_attribution.md)

</div>

<!-- RELATED:END -->
