---
title: >-
  [Paper Note] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval
description: >-
  [ACL 2026][Information Retrieval & RAG][Visual Document Retrieval] Ours proposes Prune-then-Merge, a two-stage training-free multi-vector document compression framework. By first removing low-information patches via adap…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Visual Document Retrieval"
  - "Multi-vector Compression"
  - "Adaptive Pruning"
  - "Hierarchical Aggregation"
  - "ColPali"
date: 2026-05-08
content_hash: 8de22ce92f644e2e
---

# Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval

**Conference**: ACL 2026  
**arXiv**: [2602.19549](https://arxiv.org/abs/2602.19549)  
**Code**: None  
**Area**: Information Retrieval / Document Retrieval  
**Keywords**: Visual Document Retrieval, Multi-vector Compression, Adaptive Pruning, Hierarchical Aggregation, ColPali

## TL;DR

Ours proposes Prune-then-Merge, a two-stage training-free multi-vector document compression framework. By first removing low-information patches via adaptive attention pruning and then merging the remaining high-signal patches through hierarchical clustering, it extends the near-lossless compression range from 50-60% to 60-70% and significantly outperforms single-stage methods at 80%+ high compression ratios across 29 VDR datasets.

## Background & Motivation

**Background**: Visual Document Retrieval (VDR) utilizes LVLMs to treat document pages as images. Multi-vector paradigms (e.g., ColPali) represent each page as a set of patch-level embeddings, achieving precise matching through MaxSim late interaction, which provides optimal performance.

**Limitations of Prior Work**: Multi-vector models incur massive storage and computational overhead, storing hundreds or thousands of vectors per page, making large-scale deployment impractical. Existing optimizations follow two paths: (1) Pruning methods (e.g., DocPruner) are near-lossless at medium compression but drop sharply at high ratios; (2) Merging methods (e.g., Light-ColPali) are more graceful at high ratios but may dilute discriminative features, with unstable near-lossless ranges.

**Key Challenge**: Pruning excels at accurately removing noise but cannot handle high redundancy; merging excels at high-ratio compression but suffers from centroid shifts caused by noise in noisy data. Both methods have limitations and cannot simultaneously satisfy the requirements for near-lossless performance and high compression when used in isolation.

**Goal**: Synergize the two complementary methods—pruning to improve signal-to-noise ratio (SNR) followed by merging to achieve high-ratio compression.

**Key Insight**: Driven by Information Bottleneck (IB) theory, the overall compression is decomposed into two more tractable sub-problems: query-independent information filtering (pruning) and redundancy elimination (merging).

**Core Idea**: Refine then compress—pruning transforms the input from a low-SNR set to a high-SNR set, ensuring subsequent merging operates on high-quality vectors to avoid centroid shifts caused by noise.

## Method

### Overall Architecture

Prune-then-Merge is an offline, query-independent compression framework. Stage 1: Utilize the last-layer attention weights of the LVLM to calculate importance scores for each patch and prune low-information patches using an adaptive threshold. Stage 2: Perform hierarchical agglomerative clustering on the remaining high-quality patches, replacing each cluster with its centroid. Finally, store the compressed sparse vectors for online retrieval.

### Key Designs

1.  **Adaptive Attention Pruning**:
    *   **Function**: Removes low-information patches (e.g., whitespace, decorative elements).
    *   **Mechanism**: Extracts attention weights from the final Transformer layer of the encoder, calculating the average attention of the [EOS] token to each patch as the importance score $I(\mathbf{d}_j) = \bar{\mathbf{A}}^{(L)}_{\text{eos},j}$. The adaptive threshold $\tau_d = \mu_d + k \cdot \sigma_d$ is based on document-level statistics, where the hyperparameter $k$ controls pruning strictness. Only patches with importance exceeding the threshold are retained.
    *   **Design Motivation**: Information density varies across documents; fixed-ratio pruning is suboptimal. The adaptive threshold allows each document to retain an appropriate number of informative patches.

2.  **Hierarchical Aggregation Merging**:
    *   **Function**: Further compresses the remaining high-quality patches into fewer representative vectors.
    *   **Mechanism**: L2-normalizes all embeddings, calculates the cosine distance matrix, and applies hierarchical agglomerative clustering using Ward's method to reach the target cluster count $N_p'' = \max(1, \lfloor N_p' / m \rfloor)$. The centroid (mean) of each cluster serves as the new representative embedding.
    *   **Design Motivation**: Pruned patches may still contain semantic redundancy (e.g., multiple patches describing the same table row). Merging on a high-SNR set prevents centroids from being biased by noise.

3.  **Theoretical Guarantee (IB Decomposition)**:
    *   **Function**: Provides a theoretical explanation for why the two-step approach outperforms single-step methods.
    *   **Mechanism**: Decomposes the IB optimization $\max I(\mathbf{D}''; s(q,\mathbf{D})) - \beta I(\mathbf{D}''; \mathbf{D})$ into $g = g_m \circ g_p$. Pruning $g_p$ handles query-independent information filtering (maximizing global semantic information retention), while merging $g_m$ addresses rate-distortion optimization (minimizing MSE quantization error). Single-stage merging leads to noise-shifted centroids, whereas the two-stage method yields more unbiased centroids.
    *   **Design Motivation**: Empirical methods lack systematic guidance; IB decomposition explains the optimization objectives and synergy of each stage.

### Loss & Training

Prune-then-Merge is a completely training-free post-processing framework. It applies to any multi-vector VDR model. The only hyperparameters are pruning strictness $k$ and merging factor $m$.

## Key Experimental Results

### Main Results

**nDCG@5 across 29 VDR datasets (60% compression ratio)**

| Method | ColQwen2.5 | ColNomic | Jina-v4 |
| :--- | :--- | :--- | :--- |
| No Compression | Baseline | Baseline | Baseline |
| DocPruner (Pruning) | Near-lossless | Slight Drop | Slight Drop |
| Light-ColPali (Merging) | Significant Drop | Significant Drop | Significant Drop |
| **Prune-then-Merge** | **Near-lossless** | **Near-lossless** | **Near-lossless** |

### Ablation Study

| Compression Ratio | Pruning Only | Merging Only | Prune-then-Merge |
| :--- | :--- | :--- | :--- |
| 50% | Near-lossless | Slight Drop | Near-lossless |
| 60% | Begins to Drop | Significant Drop | **Remains Near-lossless** |
| 70% | Sharp Drop | Drop | Slight Drop |
| 80% | Collapse | Large Drop | **Still Usable** |

### Key Findings

*   The near-lossless compression range is extended from [50-60%] to [60-70%], an average Gain of 10 percentage points.
*   At 80%+ compression ratios, pruning-only methods suffer from performance collapse, whereas Prune-then-Merge avoids this cliff.
*   Effectiveness is consistent across three mainstream multi-vector models (ColQwen2.5, ColNomic, Jina-v4).
*   Theoretical predictions align with experimental results—improving SNR before compression indeed reduces centroid shift.

## Highlights & Insights

*   The "refine then compress" stepwise logic is simple yet profound—decomposing a complex problem into two more solvable sub-problems.
*   Completely training-free and model-agnostic, allowing immediate application to any multi-vector retrieval model.
*   The IB theoretical analysis not only explains the efficacy but also provides principled guidance for hyperparameter selection.

## Limitations & Future Work

*   The $O(N^2)$ space complexity of hierarchical clustering may become a bottleneck for extremely large documents.
*   Query-independent pruning might inadvertently remove patches that are critical for specific queries.
*   The selection of merging factor $m$ remains empirical.
*   Future work could explore query-aware adaptive compression and learned merging strategies.

## Related Work & Insights

*   **vs DocPruner**: Pruning only, collapses at high ratios; Prune-then-Merge extends the range via subsequent merging.
*   **vs Light-ColPali**: Merging only, centroids diluted by noise; Prune-then-Merge prunes noise before merging.
*   **vs MetaEmbed**: Requires training and architectural changes; Prune-then-Merge is entirely training-free.

## Rating

*   Novelty: ⭐⭐⭐⭐ The stepwise compression logic is simple and effective; IB analysis adds depth.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 29 datasets, 3 models, comprehensive compression ratio comparisons.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear methodology with tight integration of theory and experiments.
*   Value: ⭐⭐⭐⭐ Provides a plug-and-play compression solution for practical multi-vector retrieval deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[ACL 2026\] A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval](a_picture_is_worth_a_thousand_words_an_empirical_study_of_aggregation_strategies.md)
- [\[ICML 2026\] LEMUR: Learned Multi-Vector Retrieval](../../ICML2026/information_retrieval/lemur_learned_multi-vector_retrieval.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)

</div>

<!-- RELATED:END -->
