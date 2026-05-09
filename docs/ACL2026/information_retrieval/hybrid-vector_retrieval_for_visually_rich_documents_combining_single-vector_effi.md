---
title: >-
  [Paper Note] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy
description: >-
  [ACL 2026][Visual document retrieval] HEAVEN proposes a plug-and-play two-stage hybrid-vector framework that accelerates coarse retrieval via Visual Summary Pages (VS-Pages) with a single-vector model and reduces multi-vector reranking computation via POS-based query token filtering. Across four benchmarks, the framework retains 99.87% of the multi-vector Recall@1 while reducing per-query FLOPs by 99.82%.
tags:
  - ACL 2026
  - Visual document retrieval
  - hybrid-vector retrieval
  - efficiency-accuracy trade-off
  - visual summary pages
  - query token filtering
date: 2026-05-08
content_hash: 2200d1f6b3808a0b
---

# Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy

**Conference**: ACL 2026
**arXiv**: [2510.22215](https://arxiv.org/abs/2510.22215)
**Code**: [https://github.com/juyeonnn/HEAVEN](https://github.com/juyeonnn/HEAVEN)
**Area**: Information Retrieval
**Keywords**: Visual document retrieval, hybrid-vector retrieval, efficiency-accuracy trade-off, visual summary pages, query token filtering

## TL;DR

HEAVEN proposes a plug-and-play two-stage hybrid-vector framework that accelerates coarse retrieval via Visual Summary Pages (VS-Pages) with a single-vector model and reduces multi-vector reranking computation via POS-based query token filtering. Across four benchmarks, the framework retains 99.87% of the multi-vector Recall@1 while reducing per-query FLOPs by 99.82%.

## Background & Motivation

**Background**: Visual Document Retrieval (VDR) is a core component of RAG pipelines. Current methods fall into two paradigms — single-vector retrieval (efficient but coarse) and multi-vector retrieval (accurate but computationally expensive) — exhibiting a significant efficiency-accuracy trade-off.

**Limitations of Prior Work**:
- Single-vector methods (e.g., DSE) require only one dot product per query but lose fine-grained information, resulting in large Recall@1 gaps (e.g., 22.5% lower than multi-vector on ViMDoc).
- Multi-vector methods (e.g., ColQwen2.5) must compute interactions between all query tokens and all page patches, incurring FLOPs orders of magnitude higher.
- Existing efficiency optimizations (patch pooling/pruning) suffer sharp performance degradation at higher compression ratios.

**Key Challenge**: Single-vector methods are already sufficient for coarse-grained retrieval over large candidate sets (Recall@200 gap is only 0.63%), but fail at fine-grained retrieval; multi-vector methods achieve high accuracy but at unacceptable computational cost.

**Goal**: Design a framework that reduces per-query computation by orders of magnitude with negligible loss in multi-vector retrieval accuracy.

**Key Insight**: Two empirical observations motivate the design: (1) single-vector retrieval is acceptable over large candidate sets; (2) approximately 70% of query tokens are stopwords or otherwise redundant. These observations motivate a two-stage cascaded filtering pipeline.

**Core Idea**: First, use single-vector retrieval over VS-Pages to efficiently narrow the candidate set; then apply multi-vector reranking using only key query tokens for fine-grained reranking — yielding a plug-and-play hybrid retrieval system.

## Method

### Overall Architecture

HEAVEN is a two-stage hybrid-vector retrieval framework. Stage 1 employs a single-vector model (e.g., DSE) to efficiently retrieve candidate pages over Visual Summary Pages (VS-Pages). Stage 2 employs a multi-vector model (e.g., ColQwen2.5) to perform fine-grained reranking over the candidate set using filtered key query tokens. The entire framework is plug-and-play; the encoders in both stages can be independently replaced without additional training.

### Key Designs

1. **Visual Summary Pages (VS-Pages)**:
    - **Function**: Compress multi-page documents into fewer summary pages to reduce the number of comparisons in the single-vector stage.
    - **Mechanism**: Document layout analysis (DocLayout-YOLO) is used to extract heading layouts from each page. Heading regions from multiple pages (default $r=15$) are cropped and concatenated into a single summary page. Retrieval is first performed at the VS-Page level and then expanded to the original pages.
    - **Design Motivation**: Many document pages contain repetitive or uninformative content (logos, headers, etc.); only key layout elements such as headings are useful for retrieval. VS-Pages are constructed once at indexing time and introduce no additional query-time overhead.

2. **Query Token Key Filtering**:
    - **Function**: Reduce computation in the multi-vector reranking stage.
    - **Mechanism**: Part-of-speech (POS) tagging is applied to identify key query tokens (nouns, named entities, etc., approximately 30% of all tokens). Multi-vector similarity is first computed using only these key tokens for preliminary reranking, followed by full-token reranking on the refined small candidate set.
    - **Design Motivation**: The majority of query tokens are stopwords or other redundant terms that contribute little to retrieval. Experiments demonstrate that using only key tokens matches full-token performance and outperforms randomly selecting the same number of tokens.

3. **Two-Stage Score Fusion**:
    - **Function**: Integrate signals from coarse retrieval and fine-grained reranking.
    - **Mechanism**: In Stage 1, VS-Page-level and page-level single-vector scores are fused with weight $\alpha=0.1$. In the final Stage 2 ranking, single-vector and multi-vector scores are fused with weight $\beta=0.3$.
    - **Design Motivation**: Retrieval signals at different granularities are complementary — VS-Pages provide document-level context, page-level scores offer fine-grained localization, and multi-vector scores provide token-level matching.

### Loss & Training

HEAVEN is a training-free plug-and-play framework requiring no additional training. Both stage encoders use off-the-shelf pretrained models (DSE for Stage 1, ColQwen2.5 for Stage 2) and can be directly swapped for stronger models.

## Key Experimental Results

### Main Results

| Dataset | Metric | HEAVEN | ColQwen2.5 (Multi-vector SOTA) | Performance Retained / FLOPs Reduced |
|--------|------|--------|----------|------|
| ViMDoc | R@1 | 71.05% | 71.13% | 99.88% / −99.88% |
| OpenDocVQA | R@1 | 71.56% | 72.63% | 98.52% / −99.89% |
| ViDoSeek | R@1 | 75.04% | 75.57% | 99.30% / −98.50% |
| M3DocVQA | R@1 | 59.31% | 57.99% | **102.27%** / −99.81% |
| **Average** | R@1 | 69.24% | 69.33% | **99.87% / −99.82%** |

### Ablation Study

| Configuration | Key Impact | Notes |
|------|---------|------|
| w/o VS-Pages | Significant FLOPs increase | All original pages must be compared |
| w/o candidate refinement | Severe performance drop | Score fusion from VS-Page to page level is critical |
| w/o query token filtering | FLOPs increase | Redundant query tokens introduce unnecessary computation |
| w/o reranking refinement | Significant performance drop | Single- and multi-vector score fusion are complementary |

### Key Findings
- On M3DocVQA, HEAVEN surpasses the full ColQwen2.5 (+2.27% R@1), as document-level signals from VS-Pages provide additional benefit.
- Stage 1 alone already outperforms DSE (+1.74% average R@1, −49.31% FLOPs), confirming the effectiveness of VS-Page compression.
- Using only ~30% key query tokens matches full-token multi-vector performance and outperforms randomly selecting an equivalent number of tokens.
- Compared to patch pooling/pruning efficiency baselines, HEAVEN achieves substantially higher accuracy at equivalent FLOPs.

## Highlights & Insights
- The plug-and-play design is highly practical: no additional training is required, and both stage encoders can be independently upgraded.
- The VS-Pages approach is elegant: layout analysis extracts key elements → cropped regions are concatenated into summary pages → the number of retrieval targets is reduced, with construction required only once at indexing time.
- Filtering tokens from the query side (rather than compressing patches from the document side) represents a relatively underexplored yet effective optimization direction.
- The paper additionally introduces the ViMDoc benchmark, addressing a gap in evaluation for multi-document and long-document visual retrieval.

## Limitations & Future Work
- VS-Pages rely on the quality of the layout analysis tool (DocLayout-YOLO) and may be less effective on documents with non-standard layouts.
- Query token filtering is based on POS tagging; its effectiveness on non-English languages requires further validation.
- Default hyperparameters are tuned for English documents; cross-lingual and cross-domain generalization remains to be examined.
- The two-stage cascaded design introduces system complexity and multiple hyperparameters ($\alpha$, $\beta$, $p_1$, $p_2$, $K$, $r$).

## Related Work & Insights
- **vs. ColQwen2.5 (pure multi-vector)**: HEAVEN retains 99.87% R@1 while reducing FLOPs by 99.82%.
- **vs. DSE (pure single-vector)**: HEAVEN already surpasses DSE at Stage 1; VS-Pages effectively compress the search space.
- **vs. Patch Pooling/Pruning**: Document-side compression vs. query-side filtering — HEAVEN achieves a superior efficiency-accuracy trade-off.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of two-stage hybrid retrieval, VS-Pages, and query-side filtering is distinctive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four benchmarks, multiple comparisons, ablation studies, efficiency analysis, hyperparameter analysis, and plug-and-play validation are all comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, with a complete logical chain from observations to method to validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ACL 2026\] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ACL 2026\] Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking](region-r1_reinforcing_query-side_region_cropping_for_multi-modal_re-ranking.md)

</div>

<!-- RELATED:END -->
