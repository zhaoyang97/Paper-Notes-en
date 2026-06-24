---
title: >-
  [Paper Note] Enhancing Lexicon-Based Text Embeddings with Large Language Models
description: >-
  [ACL 2025][Information Retrieval & RAG][Lexicon-based embeddings] This paper proposes the LENS framework, which is the first to apply LLMs to general lexicon-based text embeddings. By utilizing token embedding clustering to resolve LLM vocabulary redundancy and introducing bidirectional attention to overcome the limitations of causal LLMs, LENS outperforms dense embeddings trained on the same data on MTEB. When combined with dense embeddings, it achieves state-of-the-art (SOT…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Lexicon-based embeddings"
  - "LLM"
  - "text embeddings"
  - "token clustering"
  - "sparse retrieval"
date: 2026-05-08
content_hash: 5feb83269d3a2db6
---

# Enhancing Lexicon-Based Text Embeddings with Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2501.09749](https://arxiv.org/abs/2501.09749)  
**Code**: [https://github.com/Yibin-Lei/LENS](https://github.com/Yibin-Lei/LENS)  
**Area**: Information Retrieval / Text Embeddings  
**Keywords**: Lexicon-based embeddings, LLM, text embeddings, token clustering, sparse retrieval

## TL;DR

This paper proposes the LENS framework, which is the first to apply LLMs to general lexicon-based text embeddings. By utilizing token embedding clustering to resolve LLM vocabulary redundancy and introducing bidirectional attention to overcome the limitations of causal LLMs, LENS outperforms dense embeddings trained on the same data on MTEB. When combined with dense embeddings, it achieves state-of-the-art (SOTA) performance on BEIR.

## Background & Motivation

**Background**: Text embeddings serve as the foundation for tasks such as retrieval, semantic similarity, and classification. The current mainstream approach is dense embedding, which encodes text into low-dimensional real-valued vectors. Although lexicon-based sparse embeddings (e.g., SPLADE) perform exceptionally well on retrieval tasks due to strong exact matching and high interpretability, research has been mostly limited to BERT-scale Masked Language Models (MLMs) and remains largely unexplored in the LLM era.

**Limitations of Prior Work**: Direct application of LLMs to lexicon-based embeddings faces two key challenges: (1) LLM vocabulary redundancy is severe—subword tokenization causes semantically equivalent tokens to appear in multiple forms (e.g., "what"/"What"/" what"), which disrupts the consistency of lexical matching; (2) Causal LLMs use unidirectional attention, meaning each token can only attend to previous tokens, whereas lexicon-based embeddings require aggregating information from the outputs of all tokens.

**Key Challenge**: While the capabilities of LLMs continue to scale, their architectural characteristics (subword vocabularies + unidirectional attention) hinder the generation of high-quality, lexicon-based embeddings.

**Goal**: Design a framework that enables LLMs to generate compact, highly efficient, and general lexicon-based embeddings whose performance is comparable to or surpasses that of dense embeddings.

**Key Insight**: Approach the problem by "modifying the LLM's language modeling head and attention mechanism" directly, rather than using external solutions like prompt engineering.

**Core Idea**: Merge semantically similar tokens via KMeans clustering to reduce dimensionality and noise. Combined with bidirectional attention and max-pooling, this allows LLMs to generate high-quality lexicon-based embeddings with dimensions comparable to dense embeddings.

## Method

### Overall Architecture

LENS is built upon Mistral-7B with three core architectural modifications: (1) condensing the 32K vocabulary into 4000/8000 semantic clusters using KMeans clustering, where cluster centroids replace the token embeddings in the LM head; (2) changing unidirectional attention to bidirectional attention; and (3) applying log-saturation followed by max-pooling on the logits of all tokens to produce the final embeddings. The training strictly replicates the public data and pipeline of BGE-en-ICL.

### Key Designs

1. **Token Embedding Clustering (Vocabulary Compression)**:

    - **Function**: Performs KMeans clustering on the token embedding matrix in the LM head to merge approximately 32K tokens into $k$ semantic clusters ($k$ = 4000 or 8000), substituting the original token embeddings with cluster centroid embeddings.
    - **Mechanism**: After clustering, the output logits represent scores for clusters rather than individual tokens, thereby eliminating vocabulary redundancy (e.g., merging "What"/"what"/" what" into the same cluster) and significantly reducing the embedding dimension. The input token embeddings remain unchanged; only the LM head at the output is modified.
    - **Design Motivation**: Directly using a 32K-dimensional sparse embedding is highly inefficient for non-retrieval tasks (clustering, classification), and existing frameworks like FAISS do not support high-dimensional sparse vectors natively. The clustered 4000-dimensional embeddings can be seamlessly integrated into existing dense pipelines.
    - **Clustering Quality Examples**: {"quickly", "rapid", "rapidly", "swift"} are grouped into the same cluster; {"cannot", "impossible", "Unable"} are grouped into another cluster.

2. **Bidirectional Attention Mechanism**:

    - **Function**: Replaces the LLM's causal attention mask with fully connected (bidirectional) attention during the fine-tuning phase.
    - **Mechanism**: Since lexicon-based embeddings require max-pooling over all token outputs, unidirectional attention prevents early tokens from accessing contextual information from subsequent tokens, severely limiting embedding quality.
    - **Design Motivation**: In contrast to dense embedding literature where "retaining the original unidirectional attention is usually optimal", bidirectional attention is crucial for lexicon-based embeddings. Experiments show that bidirectional attention outperforms unidirectional attention by an average of 3+ points across all pooling strategies.

3. **Representation Generation and Pooling**:

    - **Function**: Produces the final embedding by applying a log-saturation transformation on logits followed by max-pooling.
    - **Mechanism**: The log-saturation transformation $w_{ij} = \log(1 + \text{ReLU}(l_{ij}))$ compresses weights into a non-negative range. Max-pooling extracts the maximum value along the sequence dimension $w_j = \max_{i} w_{ij}$, keeping only the strongest signal for each cluster across the entire text. For queries, only the original query tokens are used (excluding task instruction tokens), and a logit shift is applied (each token uses its left neighbor's logit) to adapt to the autoregressive nature.
    - **Design Motivation**: Experimentally, max-pooling outperforms sum-pooling and last-token pooling for lexicon-based embeddings because it naturally yields sparsity—only the most relevant clusters receive substantial weights.

### Loss & Training

- Uses InfoNCE contrastive loss: $\mathcal{L} = -\log \frac{\exp(\text{sim}(q, p)/\tau)}{\exp(\text{sim}(q, p)/\tau) + \sum_j \exp(\text{sim}(q, p_j^-)/\tau)}$ with temperature $\tau = 0.02$.
- Employs KL divergence distillation of BGE-reranker sorting scores as an auxiliary loss for retrieval tasks.
- LoRA fine-tuning (rank=32, alpha=64) with a learning rate of 1e-4, trained for 1 epoch.
- Batch sizes for different tasks: 512 for retrieval, 256 for other tasks.

## Key Experimental Results

### Main Results: MTEB (56 Datasets, 7 Task Types)

| Model | Dimension | Retrieval | Reranking | Clustering | Pair Classification | Classification | STS | Average |
|------|------|------|------|------|---------|------|-----|------|
| BGE-en-ICL (Dense) | 4096 | 61.67 | 59.66 | 57.51 | 86.93 | 88.62 | 83.74 | 71.24 |
| NV-Embed-v2 (Dense) | 4096 | 62.65 | 60.65 | 58.46 | 88.67 | 90.37 | 84.31 | 72.31 |
| **LENS-4000** | 4000 | 60.76 | 60.86 | 57.92 | 87.93 | 88.13 | 84.35 | 71.22 |
| **LENS-8000** | 8000 | 61.86 | 60.91 | 58.02 | 87.98 | 88.43 | 84.67 | **71.63** |

Among models trained on public data, LENS-8000 achieves the highest average score on MTEB, outperforming its dense counterpart BGE-en-ICL in 6 out of 7 task categories.

### Ablation Study: Attention + Pooling Combination

| Attention | Pooling | Retrieval | Clustering | Classification | STS | Average |
|--------|---------|------|------|------|-----|------|
| Unidirectional | Last-token | 73.84 | 60.46 | 58.66 | 89.26 | 67.73 |
| Unidirectional | Max-pooling | 75.18 | 50.93 | 57.58 | 82.74 | 64.15 |
| **Bidirectional** | **Max-pooling** | **76.19** | **63.05** | **62.30** | **88.92** | **69.07** |

Bidirectional attention + max-pooling is the optimal combination, outperforming unidirectional last-token pooling by 1.34 points on average.

### Key Findings

- LENS demonstrates that lexicon-based embeddings can fully match or even surpass dense embeddings in the LLM era, shattering the conventional assumption that "dense embeddings are inherently superior for general tasks".
- Performance actually improves when the cluster count is compressed from 32K (the original vocabulary) to 8K and 4K, indicating that vocabulary denoising on its own provides a performance boost.
- Top-K pruning (retaining only 256 of the 4000 dimensions) causes almost no performance drop, naturally supporting embedding compression without requiring Matryoshka training.
- Combining LENS with dense embeddings achieves SOTA on the BEIR retrieval subset, highlighting the strong complementarity between lexicon-based and dense embeddings.

## Highlights & Insights

- This work is the first to prove that LLMs can generate high-quality, general-purpose lexicon-based embeddings rather than being limited to retrieval tasks, opening a promising new direction for embedding research.
- Token embedding clustering is a simple and elegant design that simultaneously resolves vocabulary redundancy, dimension explosion, and matching inconsistencies.
- The contrasting results of "bidirectional vs. unidirectional attention" compared to dense embedding literature reveal fundamental architectural differences between lexicon-based and dense embeddings.
- Qualitative analysis demonstrates LENS's deep semantic understanding capabilities, such as assigning the highest weight to the "oxygen" cluster for the query "causes of hypoxia in adults".

## Limitations & Future Work

- LENS-4000 still lags behind dense embeddings on certain tasks (such as AIR-Bench), indicating that too few clusters may lead to information over-compression.
- Experiments are restricted to Mistral-7B; the generalization to other LLM backbones remains unexplored.
- KMeans clustering is static, whereas different tasks might demand varying granularities of clustering.
- Multilingual scenarios have not been investigated; multilingual vocabulary redundancy in LLMs might be a more significant bottleneck.

## Related Work & Insights

- **vs. SPLADE**: SPLADE generates lexicon-based embeddings based on MLMs (BERT) and performs exceptionally well on retrieval, but has never been extended to general-purpose tasks. LENS shows that LLMs can achieve better and more generalizable performance.
- **vs. PromptReps**: PromptReps uses prompt engineering to force LLMs to generate lexicon-based embeddings, but its performance significantly lags behind dense embeddings (MRR of 34.15 vs. 41.86). LENS achieves a fundamental breakthrough via direct architectural modification rather than relying on prompting.
- **vs. BGE-en-ICL**: Using identical training data and pipeline, the lexicon-based embeddings of LENS-8000 outperform the dense embeddings of BGE-en-ICL on MTEB.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to apply LLMs to general lexicon-based embeddings; the token clustering design is simple yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive evaluation across MTEB (56 datasets), AIR-Bench, and detailed ablations (covering cluster count, attention, and pooling).
- **Writing Quality**: ⭐⭐⭐⭐ Solid background summary and fair experimental design (strictly replicating BGE-en-ICL configurations).
- **Value**: ⭐⭐⭐⭐ Demonstrates the competitiveness of lexicon-based embeddings in the LLM era, steering future embedding research in a promising direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Redundancy, Isotropy and Intrinsic Dimensionality of Prompt-Based Text Embeddings](redundancy_isotropy_and_intrinsic_dimensionality_of_prompt-based_text_embeddings.md)
- [\[ACL 2025\] RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](rare_retrieval_augmented_reasoning.md)
- [\[ACL 2025\] Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers](drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)
- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[ACL 2025\] Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](astute_rag_knowledge_conflicts.md)

</div>

<!-- RELATED:END -->
