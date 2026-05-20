---
title: >-
  [Paper Note] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval
description: >-
  [ACL 2026][Information Retrieval & RAG][Visual Document Retrieval] This paper proposes Prune-then-Merge, a two-stage training-free multi-vector document compression framework. It first removes low-information patches via…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Visual Document Retrieval"
  - "Multi-Vector Compression"
  - "Adaptive Pruning"
  - "Hierarchical Aggregation"
  - "ColPali"
date: 2026-05-08
content_hash: f9d07b6e0a5ce957
---

# Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval

**Conference**: ACL 2026
**arXiv**: [2602.19549](https://arxiv.org/abs/2602.19549)  
**Code**: N/A  
**Area**: Information Retrieval / Document Retrieval
**Keywords**: Visual Document Retrieval, Multi-Vector Compression, Adaptive Pruning, Hierarchical Aggregation, ColPali

## TL;DR

This paper proposes Prune-then-Merge, a two-stage training-free multi-vector document compression framework. It first removes low-information patches via adaptive attention-based pruning, then applies hierarchical clustering to merge the remaining high-signal patches. Evaluated across 29 VDR datasets, the framework extends the near-lossless compression range from 50–60% to 60–70% and significantly outperforms single-stage methods at high compression ratios of 80%+.

## Background & Motivation

**Background**: Visual Document Retrieval (VDR) leverages large vision-language models (LVLMs) to process document pages as images. The multi-vector paradigm (e.g., ColPali) represents each page as a set of patch-level embeddings and achieves fine-grained matching via MaxSim late interaction, yielding state-of-the-art performance.

**Limitations of Prior Work**: Multi-vector models incur substantial storage and computational costs—each page may be indexed by hundreds to thousands of vectors—making large-scale deployment impractical. Existing compression methods fall into two camps: (1) pruning methods (e.g., DocPruner) achieve near-lossless compression at moderate ratios but suffer sharp performance degradation at high ratios; (2) merging methods (e.g., Light-ColPali) degrade more gracefully at high compression ratios but risk diluting discriminative features, with unstable near-lossless behavior.

**Key Challenge**: Pruning excels at precisely removing noise but cannot handle high redundancy; merging handles high compression ratios more gracefully but produces centroids biased by noise when applied to noisy inputs. Neither approach alone satisfies both near-lossless and high-compression requirements simultaneously.

**Goal**: To synergize the two complementary strategies—first pruning to improve the signal-to-noise ratio, then merging to achieve high compression.

**Key Insight**: Drawing from information bottleneck (IB) theory, the overall compression problem is decomposed into two more tractable sub-problems: query-agnostic information filtering (pruning) and redundancy elimination (merging).

**Core Idea**: Refine before compressing—pruning transforms the input from a low-SNR set to a high-SNR set, upon which subsequent merging operates on high-quality vectors, avoiding centroid bias induced by noise.

## Method

### Overall Architecture

Prune-then-Merge is an offline, query-agnostic compression framework. **Stage 1**: Importance scores are computed for each patch using the attention weights from the final layer of the LVLM encoder; patches below an adaptive threshold are pruned. **Stage 2**: Hierarchical agglomerative clustering is applied to the remaining high-quality patches, and each cluster is replaced by its centroid. The resulting compact set of vectors is stored for online retrieval.

### Key Designs

1. **Adaptive Attention Pruning**

    - **Function**: Removes low-information patches (e.g., blank regions, decorative elements).
    - **Mechanism**: Attention weights from the final Transformer layer of the encoder are extracted, and the average attention from the [EOS] token to each patch is used as the importance score: $I(\mathbf{d}_j) = \bar{\mathbf{A}}^{(L)}_{\text{eos},j}$. An adaptive threshold $\tau_d = \mu_d + k \cdot \sigma_d$ is computed from document-level statistics, where the hyperparameter $k$ controls pruning stringency. Only patches whose importance exceeds the threshold are retained.
    - **Design Motivation**: Information density varies considerably across documents, making fixed-ratio pruning inappropriate. The adaptive threshold allows each document to retain an appropriate number of informative patches.

2. **Hierarchical Aggregation Merging**

    - **Function**: Further compresses the remaining high-quality patches into fewer representative vectors.
    - **Mechanism**: All embeddings are L2-normalized, a cosine distance matrix is computed, and Ward linkage hierarchical agglomerative clustering is applied. The target number of clusters is $N_p'' = \max(1, \lfloor N_p' / m \rfloor)$. The centroid (mean) of each cluster serves as the new representative embedding.
    - **Design Motivation**: Even after pruning, remaining patches may exhibit semantic redundancy (e.g., multiple patches describing the same table row). Merging on a high-SNR set prevents centroids from being biased by noise.

3. **Theoretical Grounding (Information Bottleneck Decomposition)**

    - **Function**: Provides a theoretical explanation for why two-stage compression outperforms single-stage compression.
    - **Mechanism**: The IB optimization objective $\max I(\mathbf{D}''; s(q,\mathbf{D})) - \beta I(\mathbf{D}''; \mathbf{D})$ is decomposed into $g = g_m \circ g_p$. The pruning stage $g_p$ addresses query-agnostic information filtering (maximizing retention of globally relevant information), while the merging stage $g_m$ addresses rate-distortion optimization (minimizing MSE quantization error). Single-stage merging yields centroids biased by noise, whereas the two-stage approach produces unbiased centroids.
    - **Design Motivation**: Purely empirical methods lack systematic guidance; the IB decomposition explicates the optimization objective of each stage and their synergistic interaction.

### Loss & Training

Prune-then-Merge is a fully training-free post-processing framework with no model training involved. It is compatible with any multi-vector VDR model. The only hyperparameters are the pruning stringency $k$ and the merging factor $m$.

## Key Experimental Results

### Main Results

**nDCG@5 on 29 VDR datasets at 60% compression ratio**

| Method | ColQwen2.5 | ColNomic | Jina-v4 |
|---|---|---|---|
| No Compression | Baseline | Baseline | Baseline |
| DocPruner (Pruning) | Near-lossless | Slight drop | Slight drop |
| Light-ColPali (Merging) | Noticeable drop | Noticeable drop | Noticeable drop |
| **Prune-then-Merge** | **Near-lossless** | **Near-lossless** | **Near-lossless** |

### Ablation Study

| Compression Ratio | Pruning Only | Merging Only | Prune-then-Merge |
|---|---|---|---|
| 50% | Near-lossless | Slight drop | Near-lossless |
| 60% | Begins to drop | Notable drop | **Still near-lossless** |
| 70% | Sharp drop | Drop | Slight drop |
| 80% | Collapse | Large drop | **Still viable** |

### Key Findings

- The near-lossless compression range is extended from [50–60%] to [60–70%], an average improvement of 10 percentage points.
- At 80%+ compression, pruning-only methods suffer catastrophic performance collapse (cliff effect); Prune-then-Merge avoids this failure mode.
- Consistent effectiveness is demonstrated across three mainstream multi-vector models (ColQwen2.5, ColNomic, Jina-v4).
- Theoretical predictions align with empirical results—improving SNR prior to compression demonstrably reduces centroid bias.

## Highlights & Insights

- The "refine-then-compress" decomposition is both concise and insightful, reducing a complex problem to two more tractable sub-problems.
- The framework is entirely training-free and model-agnostic, and can be directly applied to any multi-vector retrieval model.
- The IB-theoretic analysis not only explains the empirical effectiveness of the method but also provides principled guidance for hyperparameter selection.

## Limitations & Future Work

- The $O(N^2)$ space complexity of hierarchical clustering may become a bottleneck for very large documents.
- Query-agnostic pruning may inadvertently discard patches that are informative for certain specific queries.
- The choice of merging factor $m$ remains empirical.
- Future work could explore query-aware adaptive compression and learned merging strategies.

## Related Work & Insights

- **vs. DocPruner**: Pruning only; performance collapses at high compression ratios. Prune-then-Merge extends the usable compression range via subsequent merging.
- **vs. Light-ColPali**: Merging only; centroids are diluted by noise. Prune-then-Merge first removes noise before merging.
- **vs. MetaEmbed**: Requires training and architectural modifications; Prune-then-Merge is entirely training-free.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The two-stage compression idea is concise and effective; the IB-theoretic analysis adds depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 29 datasets across 3 models with comprehensive compression ratio comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Methodology is clearly presented with tight integration of theory and experiments.
- **Value**: ⭐⭐⭐⭐ Provides a plug-and-play compression solution for practical deployment of multi-vector retrieval systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[CVPR 2026\] NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](../../CVPR2026/information_retrieval/nanovdr_distilling_a_2b_vision-language_retriever_into_a_70m_text-only_encoder_f.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)

</div>

<!-- RELATED:END -->
