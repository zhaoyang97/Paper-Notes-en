---
title: >-
  [Paper Note] Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval
description: >-
  [CVPR 2026][Information Retrieval & RAG][multi-condition retrieval] This paper introduces MCMR (Multi-Conditional Multimodal Retrieval), a large-scale benchmark that employs a dual-evidence design — where certain attribu…
tags:
  - "CVPR 2026"
  - "Information Retrieval & RAG"
  - "multi-condition retrieval"
  - "fine-grained matching"
  - "dual-evidence"
  - "MLLM reranking"
  - "cross-modal reasoning"
date: 2026-05-08
content_hash: 4c26c1a9dfdd0d91
---

# Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval

**Conference**: CVPR 2026
**arXiv**: [2603.01082](https://arxiv.org/abs/2603.01082)  
**Code**: [github.com/EIT-NLP/MCMR](https://github.com/EIT-NLP/MCMR)  
**Area**: Information Retrieval
**Keywords**: multi-condition retrieval, fine-grained matching, dual-evidence, MLLM reranking, cross-modal reasoning

## TL;DR

This paper introduces MCMR (Multi-Conditional Multimodal Retrieval), a large-scale benchmark that employs a dual-evidence design — where certain attributes are inferable only from images and others only from text — to ensure retrieval tasks cannot be solved unimodally. The benchmark systematically evaluates 5 retrievers and 7 MLLM rerankers, revealing modality asymmetry and fine-grained reasoning gaps.

## Background & Motivation

**Background**: Multimodal retrieval has evolved from the global semantic alignment of the CLIP era to MLLM-based instruction-conditioned retrieval (e.g., VLM2Vec, GME, MM-Embed), yet evaluation benchmarks remain at the level of coarse-grained or single-condition matching.

**Limitations of Prior Work**:

1. Classic benchmarks (MS-COCO, Flickr30K) evaluate only global image-text alignment without compositional reasoning.
2. Fine-grained benchmarks such as FashionIQ and CIRR revolve around single visual edits, which can in principle be resolved from images alone.
3. Multi-condition benchmarks such as MultiConIR operate exclusively in a text-only setting without cross-modal reasoning.
4. MERIT introduces multimodal interleaved queries but relies on reference image comparisons and does not distinguish between visual and textual attribute sources.

**Key Challenge**: Existing benchmarks are either fine-grained but single-condition, or multi-condition but unimodal — none simultaneously satisfy all three dimensions of fine-grained attributes, multi-condition queries, and cross-modal evidence.

**Goal**: Construct a retrieval benchmark that genuinely tests cross-modal compositional reasoning ability.

**Key Insight**: Enforce a "dual-evidence" constraint — each product must contain at least one attribute inferable only from images and one inferable only from text.

**Core Idea**: The dual-evidence design ensures that neither modality alone can resolve the task, thereby genuinely testing a model's cross-modal compositional reasoning capability.

## Method

### Overall Architecture

The MCMR dataset is constructed from the Amazon Reviews (2023) corpus, covering five product domains (tops, bottoms, jewelry, shoes, furniture), comprising 3,997 natural language queries and 104,981 candidate items. Data is built via a multi-stage pipeline (attribute extraction → quality filtering → query generation → validation), and both retrievers and rerankers are evaluated under a unified protocol.

### Key Designs

1. **Dual-Evidence Data Construction Pipeline**

    - **Image side**: Qwen2.5-VL-32B generates structured visual attribute summaries (color, texture, structural details, etc.) from product images, strictly excluding functional or speculative content.
    - **Text side**: Qwen3-32B extracts JSON-structured profiles from product titles, descriptions, and features; DeepSeek-R1 verifies the absence of cross-modal leakage.
    - **Query generation**: Qwen3-32B generates first-person multi-condition queries from image attributes and text summaries; DeepSeek-R1 independently validates cross-modal coverage and consistency.
    - **Human validation**: A 100-sample double-blind study confirms that generated queries are comparable in quality to human-written ones (4.33 vs. 4.41; preference rate 47% vs. 49%).
    - **Complementarity constraint**: Each item must include at least one image-only attribute and one text-only attribute.

2. **MLLM Pointwise Reranking Mechanism**

    - The top-50 candidates from a first-stage retriever are evaluated pointwise by an MLLM for query–candidate relevance.
    - Input: text query + candidate image + candidate text metadata → Output: normalized logits of True/False as relevance scores.
    - Candidates are reranked by score; ties preserve the original retrieval order.
    - Seven rerankers are compared; lychee-reranker-mm achieves the strongest performance across all cutoffs (nDCG@1 = 92.35).

3. **Modality Ablation over Three Candidate Visibility Settings**

    - Three settings are considered: fused (image + text), image-only, and text-only.
    - Query-side ablation is also conducted: retaining all constraints vs. removing image constraints vs. removing text constraints.
    - Constraint quantity ablation: $k_T = k_I \in \{1,2,3,4,5\}$.

### Dataset Statistics

| Domain | # Queries | # Candidates |
|--------|-----------|--------------|
| Tops | 991 | 29,986 |
| Bottoms | 803 | 29,514 |
| Shoes | 847 | 24,997 |
| Jewelry | 602 | 5,491 |
| Furniture | 754 | 14,993 |
| **Total** | **3,997** | **104,981** |

## Key Experimental Results

### Main Results: Retriever Comparison under Fused Modality

| Model | Params | R@1 | R@10 | R@100 | MRR | nDCG@10 |
|-------|--------|-----|------|-------|-----|---------|
| CORAL | 3B | 26.57 | 53.34 | 77.73 | 34.94 | 39.35 |
| LLaVE | 7B | 24.99 | 53.13 | 78.64 | 33.15 | 37.88 |
| MM-EMBED | 8B | 21.74 | 47.91 | 74.16 | 29.35 | 33.75 |
| GME-Qwen2VL | 7B | 21.23 | 45.74 | 73.52 | 28.35 | 32.48 |
| LamRA | 7B | 17.96 | 43.30 | 73.24 | 25.27 | 29.53 |
| VLM2Vec | 4B | 1.83 | 7.03 | 18.96 | 3.11 | 4.02 |

### MLLM Reranker Comparison (LLaVE top-50 pool)

| Reranker | Params | nDCG@1 | nDCG@5 | nDCG@10 | nDCG@50 |
|----------|--------|--------|--------|---------|---------|
| lychee-reranker-mm | 8B | **92.35** | **93.41** | **94.42** | **94.86** |
| InternVL3 | 8B | 80.28 | 81.95 | 84.66 | 86.61 |
| Qwen3-VL-Reranker | 8B | 78.69 | 80.79 | 83.51 | 85.57 |
| Qwen2.5-VL | 32B | 78.22 | 79.87 | 82.58 | 84.88 |
| Qwen2.5-VL | 7B | 74.16 | 77.26 | 80.26 | 82.84 |

### Ablation Study: Effect of Candidate-Side Modality (R@10)

| Setting | GME | LLaVE | MM-EMBED | CORAL |
|---------|-----|-------|---------|-------|
| Fused | 45.74 | 53.13 | 47.91 | 53.34 |
| Image-only | **51.10** | 3.93 | 35.68 | 33.53 |
| Text-only | 29.60 | 29.43 | 34.50 | 22.88 |

### Key Findings

- R@1 reaches only 18–27% while R@100 can reach 78%: coarse recall is feasible, but fine-grained ranking is extremely challenging.
- Significant modality asymmetry: GME's R@10 actually increases under image-only (51.10 vs. 45.74), while LLaVE collapses from 53.13 to 3.93.
- Text-only consistently underperforms fused and image-only settings, indicating that visual cues are the primary discriminative signal in MCMR.
- MLLM reranking improves nDCG@1 from 26.57 (best first-stage retriever, CORAL) to 92.35 (lychee-reranker), a substantial gain.
- Increasing the number of query constraints (1T+1I → 5T+5I) monotonically improves R@10 with diminishing marginal returns.

## Highlights & Insights

- First multimodal retrieval benchmark to simultaneously satisfy all three dimensions: fine-grained attributes, multi-condition queries, and cross-modal evidence.
- The dual-evidence design ensures that the task cannot be solved unimodally, genuinely testing cross-modal integration ability.
- The large performance gap between first-stage retrievers and rerankers (nDCG@1: 26.57 vs. 92.35) exposes a fundamental limitation of embedding-based global matching.
- A complementary pattern is identified: visual cues dominate top-rank precision, while text metadata stabilizes long-tail ranking.
- Human validation confirms no significant quality difference between automatically generated and human-written queries.

## Limitations & Future Work

- Coverage is limited to the product/e-commerce domain (5 item categories), with no extension to general scenarios (news, medical, scientific literature).
- Queries are entirely text-based; image-text interleaved queries (e.g., "find something similar to this image but in cotton") are not explored.
- The candidate pool (~100K items) is far smaller than real-world e-commerce systems (millions to tens of millions), leaving scalability unverified.
- Pointwise reranking incurs high computational cost and cannot be directly applied to large-scale retrieval; more efficient alternatives are needed.
- Model size does not determine reranking ability: Qwen2.5-VL at 32B underperforms lychee-reranker at 8B, but no causal analysis is provided.

## Related Work & Insights

- **vs. MERIT**: MERIT relies on reference image comparisons, whereas MCMR's pure text queries better reflect real user search behavior; MERIT also does not distinguish attribute sources by modality.
- **vs. MultiConIR**: MultiConIR conducts multi-condition retrieval in a text-only setting; MCMR extends this to cross-modal scenarios.
- **vs. FashionIQ/CIRR**: These single visual-edit benchmarks allow attributes to be verified from images alone; MCMR's dual-evidence design is substantially more challenging.
- Insight: Future work may explore hierarchical retrieval architectures that decompose multi-condition queries into sub-tasks, or incorporate condition-aware sparse attention at the retrieval stage.

## Rating

- Novelty: ⭐⭐⭐⭐ The benchmark design satisfying all three dimensions simultaneously is original, and the dual-evidence constraint is a valuable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation covers 5 retrievers + 7 rerankers, 3 modality settings, candidate-side and query-side ablations, and constraint quantity ablations — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear, experimental analysis is in-depth, and the benchmark comparison table (Tab. 1) is easy to read.
- Value: ⭐⭐⭐⭐ Fills the gap in multi-condition cross-modal retrieval benchmarks; the reranker vs. retriever gap analysis offers meaningful practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[ACL 2026\] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](../../ACL2026/information_retrieval/rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)
- [\[AAAI 2026\] Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score](../../AAAI2026/information_retrieval/beyond_perplexity_let_the_reader_select_retrieval_summaries_via_spectrum_project.md)

</div>

<!-- RELATED:END -->
