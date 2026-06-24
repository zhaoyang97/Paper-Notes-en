---
title: >-
  [Paper Note] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] Proposes EXIT—an extractive context compression framework that evaluates sentence-query relevance in parallel via context-aware sentence-level binary classification, outperforming existing abstractive and extractive compression methods in both QA accuracy and inference latency.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Context Compression"
  - "Extractive Compression"
  - "Sentence Classification"
  - "Inference Efficiency"
date: 2026-05-08
content_hash: 21a465f658c0504b
---

# EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2412.12559](https://arxiv.org/abs/2412.12559)  
**Code**: [ThisIsHwang/EXIT](https://github.com/ThisIsHwang/EXIT)  
**Area**: NLP-Retrieval-Augmented Generation  
**Keywords**: RAG, Context Compression, Extractive Compression, Sentence Classification, Inference Efficiency  

## TL;DR

Proposes EXIT—an extractive context compression framework that evaluates sentence-query relevance in parallel via context-aware sentence-level binary classification, outperforming existing abstractive and extractive compression methods in both QA accuracy and inference latency.

## Background & Motivation

- **Background**: Retrieval-Augmented Generation (RAG) enhances the response quality of LLMs by retrieving external documents. However, retrieval models sometimes fail to rank the most relevant documents at the top, and increasing the number of retrieved documents introduces issues of increased latency and degraded accuracy.
- **Limitations of Prior Work**: (1) **Abstractive compression** (e.g., CompAct, Refiner) compresses context via autoregressive generation, but the compression itself incurs high latency (taking 8+ seconds to process 5 documents); (2) **Extractive compression** (e.g., RECOMP-Extr) is fast but employs fixed, context-independent selection strategies, failing to adapt to variations in query complexity and retrieval quality.
- **Key Challenge**: Abstractive methods are accurate but slow, while extractive methods are fast but perform poorly—there is a need for a compression method that is both fast and accurate.
- **Goal**: Design a method that retains the speed advantage of extractive compression while improving compression quality through context-awareness and adaptive selection.
- **Key Insight**: Redefine context compression as a "sentence-level binary classification problem" and utilize the complete context of the input document to perform parallelized relevance assessment.
- **Core Idea**: Transform RAG context compression into a parallelizable sentence classification task, where each sentence's relevance to the query is determined within the full context of the document.

## Method

### Overall Architecture

EXIT consists of three stages: (1) splitting the retrieved documents into sentences; (2) performing parallel binary classification ("Yes"/"No") on each sentence to evaluate its relevance to the query within the full document context; and (3) reconstructing the selected sentences in their original order.

### Key Designs

#### 1. Sentence-level Decomposition
- **Function**: Split each retrieved document into sentences.
- **Mechanism**: Utilize SpaCy's rule-based sentencizer to decompose document $d_i$ into a set of sentences $S_i = \{s_{i1}, s_{i2}, \dots, s_{in}\}$.
- **Design Motivation**: Sentence-level manipulation avoids the fragmentation of key phrases and corruption of entity relationships typical of token-level compression (e.g., LLMLingua), thereby preserving grammatical coherence and semantic integrity.

#### 2. Context-Aware Relevance Classification
- **Function**: Evaluate the relevance of each sentence utilizing the full document context.
- **Mechanism**: Given a query $q$, document $d_i$, and sentence $s_{ij}$, an evaluation model is used to predict the relevance score:

$$r_{ij} = \frac{P(\text{"Yes"} | q, d_i, s_{ij})}{P(\text{"Yes"} | q, d_i, s_{ij}) + P(\text{"No"} | q, d_i, s_{ij})}$$

Only a single token needs to be predicted (the probability of "Yes"/"No"), allowing all sentences to be processed in parallel.
- **Design Motivation**: (1) Feeding the entire document $d_i$ as context is crucial because understanding a single sentence usually requires the broader document background; (2) Single-token prediction is significantly more efficient than multi-token autoregressive generation.

#### 3. Adaptive Threshold Selection
- **Function**: Dynamically select the number of sentences based on query and retrieval quality.
- **Mechanism**: Set a threshold $\tau = 0.5$ and retain sentences with $r_{ij} > \tau$. The compressed document $D'$ contains a variable number of sentences.
- **Design Motivation**: Different queries have varying information needs—simple questions might require only a few sentences, whereas complex multi-hop reasoning requires more evidence.

### Loss

Binary cross-entropy loss:

$$\mathcal{L} = -\mathbb{1}_{l=\text{"Yes"}} \log P(\text{"Yes"}) - (1 - \mathbb{1}_{l=\text{"Yes"}}) \log P(\text{"No"})$$

Training data consists of three types of samples: positive samples (sentences containing answer evidence), hard negative samples (other sentences in the same document), and random negative samples (sentences from irrelevant documents).

### Training

- Trained solely on the HotpotQA training set (which has sentence-level annotations) but generalizes to NQ, TQA, and 2WIKI.
- The classifier is based on Gemma-2B-it.
- Temperature is set to 0.0 during inference.

## Key Experimental Results

### Main Results: Performance on 4 QA Datasets (Llama3.1-8B-Instruct)

| Compression Method | Type | NQ-EM | TQA-EM | HQA-EM | 2WIKI-EM | Avg-EM | Avg Latency (s) |
|---|---|---|---|---|---|---|---|
| Original Document | - | 34.6 | 58.8 | 28.1 | 16.1 | 34.4 | 1.0 |
| CompAct | Abs | 32.9 | 58.1 | 28.8 | 16.8 | 34.2 | **8.4** |
| RECOMP-Extr | Ext | 34.6 | 56.5 | 23.4 | 11.2 | 31.4 | 0.5 |
| LongLLMLingua | Ext | 30.2 | 59.4 | 28.0 | 21.5 | 34.8 | 0.9 |
| **EXIT** | Ext | **35.9** | **60.8** | **30.6** | **24.2** | **37.9** | **0.8** |

- EXIT outperforms all methods (including uncompressed original documents) on Avg-EM, while maintaining a latency of only 0.8s.

### Performance on 70B Models

| Compression Method | Avg-EM | Avg-F1 | Avg Latency (s) |
|---|---|---|---|
| Original Document | 38.8 | 48.7 | 8.4 |
| **EXIT** | **42.5** | **52.0** | **3.5** |

- The improvement of EXIT is even more pronounced on 70B models: EM +3.7, F1 +3.3, with a 58% reduction in latency.

### Ablation Study

| Configuration | EM | F1 | Token Count |
|---|---|---|---|
| **Full EXIT** | **31.6** | **42.6** | 195.1 |
| Positives + Hard Negatives only | 30.0 | 41.3 | 286.8 |
| Positives + Random Negatives only | 29.8 | 40.9 | 404.6 |
| Fixed 2 sentences chosen | 29.4 | 40.7 | 91.0 |
| Fixed 4 sentences chosen | 30.2 | 41.4 | 166.5 |
| No Document Context | 30.4 | 42.3 | 157.4 |

### Key Findings

1. **Extractive Can Outperform Original Documents**: EXIT demonstrates that precise sentence selection can remove noisy information, thereby improving QA performance.
2. **Decoupling of Latency vs FLOPS**: Though EXIT's TFLOPs (35.44) are higher than some methods, its actual latency is lower because of parallel processing—indicating that efficiency should be measured by parallelism, not just computation volume.
3. **Adaptive Selection is Crucial**: Fixed-sentence selection (2 or 4 sentences) consistently underperforms adaptive thresholding.
4. **Context-Awareness Enhances Accuracy**: Removing document context drops EM by 1.2, confirming that full context is essential for determining sentence relevance.
5. **Robustness**: As the number of retrieved documents $k$ increases from 1 to 30, EXIT's EM continuously improves ($28.2 \rightarrow 33.1$), whereas other methods suffer performance degradation.

## Highlights & Insights

- **Elegance of Problem Redefinition**: Redefining "context compression" as "sentence classification" naturally yields the dual advantages of parallelization and context-awareness.
- A novel analysis perspective on **end-to-end latency**—revealing that "compression time + reading time" is the true efficiency metric, and relying solely on token counts is insufficient.
- **Plug-and-play** design: EXIT is independent of specific retrievers and readers, allowing seamless integration into any RAG pipeline without modification.
- High generalizability: Training solely on HotpotQA generalizes well to out-of-domain datasets (NQ, TQA, 2WIKI).

## Limitations & Future Work

- The threshold $\tau = 0.5$ is fixed; exploring adaptive thresholds (e.g., dynamically adjusting based on retrieval quality) is a potential direction.
- The classifier (Gemma-2B) requires an additional forward pass, which might become a bottleneck under extremely high query volumes.
- Sentence tokenization relies heavily on the quality of the sentencizer, which may perform poorly on non-standard text (e.g., tables, lists).
- The modeling of dependencies between sentences in multi-hop reasoning remains unexplored.
- Training relies exclusively on HotpotQA; more diverse training data could further enhance generalizability.

## Related Work & Insights

- **RECOMP (Xu et al., 2024)**: Proposes both abstractive and extractive variants, serving as the primary baseline.
- **CompAct / Refiner**: Representatives of abstractive compression, where latency is the critical bottleneck.
- **LLMLingua Series**: Token-level compression typical of these methods may compromise semantic coherence.
- **Flamingo / RAG (Lewis et al., 2020)**: Foundational works of the RAG paradigm.
- **Insight**: Sentence-level granularity may represent the optimal balance between "semantic integrity" and "compression efficiency."

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The concept of formulating compression as a classification problem is elegant, and the context-aware design is highly effective.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Evaluated across 4 datasets, 2 reader scales, with detailed ablation and latency analysis.
- **Value** ⭐⭐⭐⭐⭐: Plug-and-play, low latency, and highly effective, offering significant value for industrial deployment.
- **Writing Quality** ⭐⭐⭐⭐: Clear diagrams and highly intuitive latency analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)
- [\[ACL 2025\] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)
- [\[ACL 2025\] FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)
- [\[ACL 2025\] Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation](typed-rag_type-aware_decomposition_of_non-factoid_questions_for_retrieval-augmen.md)

</div>

<!-- RELATED:END -->
