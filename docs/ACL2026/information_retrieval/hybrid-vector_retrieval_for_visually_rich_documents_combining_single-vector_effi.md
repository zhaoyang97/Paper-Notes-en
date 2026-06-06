---
title: >-
  [Paper Note] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy
description: >-
  [ACL 2026][Information Retrieval & RAG][Visual Document Retrieval] HEAVEN proposes a plug-and-play two-stage hybrid-vector framework that accelerates single-vector coarse retrieval via Visual Summary Pages (VS-Pages) and…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Visual Document Retrieval"
  - "Hybrid-Vector Retrieval"
  - "Efficiency-Accuracy Trade-off"
  - "Visual Summary Pages"
  - "Query Token Filtering"
date: 2026-05-08
content_hash: 7d0196c1c8e5d482
---

# Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy

**Conference**: ACL 2026  
**arXiv**: [2510.22215](https://arxiv.org/abs/2510.22215)  
**Code**: [https://github.com/juyeonnn/HEAVEN](https://github.com/juyeonnn/HEAVEN)  
**Area**: Information Retrieval  
**Keywords**: Visual Document Retrieval, Hybrid-Vector Retrieval, Efficiency-Accuracy Trade-off, Visual Summary Pages, Query Token Filtering

## TL;DR

HEAVEN proposes a plug-and-play two-stage hybrid-vector framework that accelerates single-vector coarse retrieval via Visual Summary Pages (VS-Pages) and reduces multi-vector re-ranking computation through POS-based query token filtering. It maintains $99.87\%$ of multi-vector $Recall@1$ across four benchmarks while reducing per-query FLOPs by $99.82\%$.

## Background & Motivation

**Background**: Visual Document Retrieval (VDR) is a core component of RAG. Current methods follow two primary paradigms: single-vector retrieval (efficient but coarse) and multi-vector retrieval (accurate but computationally expensive), resulting in a significant efficiency-accuracy trade-off.

**Limitations of Prior Work**:
- Single-vector methods (e.g., DSE) require only one dot product per query but lose fine-grained information, leading to a large $Recall@1$ gap (e.g., $22.5\%$ lower than multi-vector models on ViMDoc).
- Multi-vector methods (e.g., ColQwen2.5) must compute interactions between all query tokens and all page patches, resulting in FLOPs that are hundreds of times higher.
- Existing efficiency optimizations (e.g., patch pooling/pruning) suffer from sharp performance declines as the compression ratio increases.

**Key Challenge**: Single-vector methods are sufficient for coarse-grained retrieval (large $K$), where $Recall@200$ differs by only $0.63\%$, but they fail at fine-grained retrieval. Conversely, multi-vector methods offer high accuracy but incur unacceptable computational costs.

**Goal**: Design a framework that reduces per-query computation by several orders of magnitude while preserving multi-vector retrieval accuracy.

**Key Insight**: Leverage two empirical observations: (1) single-vector retrieval is acceptable for large candidate sets; (2) approximately $70\%$ of query tokens are redundant information ($e.g.$, stop words). Based on these, a two-stage cascaded filtering mechanism is proposed.

**Core Idea**: Use single-vector retrieval combined with VS-Pages to rapidly narrow the candidate range, followed by multi-vector retrieval with filtered key query tokens for fine-grained re-ranking, achieving a plug-and-play hybrid retrieval system.

## Method

### Overall Architecture

HEAVEN is a two-stage hybrid-vector retrieval framework. Stage 1 utilizes a single-vector model (e.g., DSE) to efficiently retrieve candidate page sets using Visual Summary Pages (VS-Pages). Stage 2 employs a multi-vector model (e.g., ColQwen2.5) to perform fine-grained re-ranking on the candidate set using only key query tokens. The entire framework is plug-and-play, allowing encoders for both stages to be swapped independently without additional training.

### Key Designs

1. **Visual Summary Pages (VS-Pages)**:
    - **Function**: Compresses multi-page documents into fewer summary pages to reduce the number of comparisons during the single-vector stage.
    - **Mechanism**: Extracts title layouts from each page using document layout analysis (DocLayout-YOLO) and tiles these layouts from multiple pages (default $r=15$ pages) into a single VS-Page. Retrieval is performed at the VS-Page level first and then expanded back to the original pages.
    - **Design Motivation**: Many pages contain repetitive or uninformative content (logos, headers), whereas layouts such as titles are highly indicative for retrieval. VS-Pages are constructed once during indexing, adding no overhead at query time.

2. **Query Token Filtering**:
    - **Function**: Reduces the computational load during the multi-vector re-ranking stage.
    - **Mechanism**: Utilizes Part-of-Speech (POS) tagging to identify key tokens in the query (nouns, named entities, etc., which account for ~30%). Only these key tokens are used to calculate multi-vector similarity for initial refinement, followed by a final ranking on a smaller refined candidate set using all tokens.
    - **Design Motivation**: Most query tokens are redundant and contribute little to retrieval performance. Experiments demonstrate that using only key tokens achieves performance comparable to using all tokens and is superior to random token selection.

3. **Two-stage Score Fusion**:
    - **Function**: Integrates signals from coarse retrieval and fine-grained re-ranking.
    - **Mechanism**: Stage 1 fuses VS-Page level and page-level single-vector scores (weight $\alpha=0.1$). Stage 2 fuses single-vector and multi-vector scores in the final ranking (weight $\beta=0.3$).
    - **Design Motivation**: Retrieval signals at different granularities are complementary; VS-Pages provide document-level context, page-level scores offer fine-grained localization, and multi-vector scores provide token-level matching.

### Loss & Training

HEAVEN is a training-free, plug-and-play framework. Encoders for both stages use off-the-shelf pre-trained models (DSE for Stage 1, ColQwen2.5 for Stage 2) and can be directly replaced with newer models.

## Key Experimental Results

### Main Results

| Dataset | Metric | HEAVEN | ColQwen2.5 (Multi-vector SOTA) | Performance Maintained / FLOPs Reduction |
|--------|------|--------|----------|------|
| ViMDoc | $Recall@1$ | $71.05\%$ | $71.13\%$ | $99.88\% / -99.88\%$ |
| OpenDocVQA | $Recall@1$ | $71.56\%$ | $72.63\%$ | $98.52\% / -99.89\%$ |
| ViDoSeek | $Recall@1$ | $75.04\%$ | $75.57\%$ | $99.30\% / -98.50\%$ |
| M3DocVQA | $Recall@1$ | $59.31\%$ | $57.99\%$ | **$102.27\%$** / $-99.81\%$ |
| **Average** | $Recall@1$ | $69.24\%$ | $69.33\%$ | **$99.87\% / -99.82\%$** |

### Ablation Study

| Configuration | Key Impact | Description |
|------|---------|------|
| w/o VS-Pages | Significant FLOPs increase | Requires comparison across all original pages |
| w/o Candidate Refinement | Sharp performance drop | VS-Page to page score fusion is critical |
| w/o Query Token Filtering | FLOPs increase | Redundant tokens cause unnecessary computation |
| w/o Re-ranking Refinement | Significant performance drop | Single/multi-vector score fusion is complementary |

### Key Findings
- On M3DocVQA, HEAVEN actually outperforms the full ColQwen2.5 ($+2.27\% Recall@1$) due to the beneficial document-level signals provided by VS-Pages.
- Stage 1 alone outperforms DSE ($+1.74\%$ average $Recall@1$, $-49.31\%$ FLOPs), proving that VS-Pages effectively compress the search space.
- Using only ~30% of key query tokens matches the performance of full-token multi-vector retrieval and is superior to random sampling.
- Compared to efficiency optimizations like patch pooling/pruning, HEAVEN achieves significantly higher accuracy at comparable FLOPs levels.

## Highlights & Insights
- The plug-and-play design is highly practical: no training is required, and encoders can be upgraded independently.
- The VS-Pages concept is innovative: extracting key layouts via layout analysis → tiling into summary pages → reducing the number of retrieval targets; this is a one-time process during indexing.
- Filtering tokens from the query side (rather than compressing patches from the document side) is an effective yet relatively overlooked optimization direction.
- The paper introduces the ViMDoc benchmark, filling a gap in the evaluation of visual retrieval for multi-document and long-document scenarios.

## Limitations & Future Work
- VS-Pages depend on the quality of the layout analysis tool (DocLayout-YOLO); performance may be limited for documents with non-standard layouts.
- Query token filtering is based on POS tagging; its effectiveness for non-English languages requires further verification.
- Default hyperparameters are currently tuned for English documents; cross-lingual and cross-domain generalization remains to be explored.
- The two-stage cascaded design introduces system complexity and several hyperparameters ($\alpha, \beta, p_1, p_2, K, r$).

## Related Work & Insights
- **vs ColQwen2.5 (Pure Multi-vector)**: HEAVEN maintains $99.87\%$ of $Recall@1$ while reducing FLOPs by $99.82\%$.
- **vs DSE (Pure Single-vector)**: HEAVEN surpasses DSE even in Stage 1, with VS-Pages effectively compressing the search space.
- **vs Patch Pooling/Pruning**: While others compress the document side, HEAVEN filters the query side, providing a superior efficiency-accuracy trade-off.

## Rating
- Novelty: ⭐⭐⭐⭐ Unique combination of a two-stage hybrid design, VS-Pages, and query-side filtering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across four benchmarks, including detailed comparisons, ablations, and efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with a strong logical progression from observation to methodology and validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ICML 2026\] LEMUR: Learned Multi-Vector Retrieval](../../ICML2026/information_retrieval/lemur_learned_multi-vector_retrieval.md)
- [\[ICML 2026\] Vector Linking based on Cross-Model Local Isometry Consistency](../../ICML2026/information_retrieval/vector_linking_via_cross-model_local_isometric_consistency.md)
- [\[ACL 2026\] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)
- [\[ACL 2026\] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)

</div>

<!-- RELATED:END -->
