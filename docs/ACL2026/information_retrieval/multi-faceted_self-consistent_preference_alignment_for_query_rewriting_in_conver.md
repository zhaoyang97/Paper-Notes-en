---
title: >-
  [Paper Note] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search
description: >-
  [ACL 2026 Findings][conversational query rewriting] This paper proposes MSPA-CQR, which constructs self-consistent preference data across three dimensions—rewriting, retrieval, and response—and trains a query rewriting model via prefix-guided multi-dimensional DPO, achieving significant improvements over existing methods in both in-distribution and out-of-distribution settings.
tags:
  - ACL 2026 Findings
  - conversational query rewriting
  - preference alignment
  - self-consistency scoring
  - multi-dimensional DPO
  - conversational search
date: 2026-05-08
content_hash: 28a02bff7ea95993
---

# Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search

**Conference**: ACL 2026 Findings
**arXiv**: [2604.06771](https://arxiv.org/abs/2604.06771)
**Code**: N/A
**Area**: Information Retrieval
**Keywords**: conversational query rewriting, preference alignment, self-consistency scoring, multi-dimensional DPO, conversational search

## TL;DR

This paper proposes MSPA-CQR, which constructs self-consistent preference data across three dimensions—rewriting, retrieval, and response—and trains a query rewriting model via prefix-guided multi-dimensional DPO, achieving significant improvements over existing methods in both in-distribution and out-of-distribution settings.

## Background & Motivation

**Background**: In conversational question answering (CQA), user queries are often ambiguous (e.g., unclear references, omitted keywords), necessitating conversational query rewriting (CQR) to convert underspecified queries into complete, self-contained ones for downstream retrieval. Early approaches relied on human-annotated rewrites as training targets, but manual annotation is costly and typically optimizes for readability rather than retrieval effectiveness.

**Limitations of Prior Work**: Recent studies have begun incorporating retrieval signals as feedback, yet two issues persist: (1) only the retrieval dimension is considered, while rewriting quality and response quality are neglected; (2) preference data construction depends on human-annotated gold passages, limiting applicability to unannotated data.

**Key Challenge**: A high-quality rewritten query must simultaneously satisfy three criteria—the rewrite itself should be self-contained and complete, it should capture key information to support effective retrieval without redundancy, and the corresponding response should be coherent and accurate. These three dimensions exhibit non-trivial divergence (Kendall-Tau correlations of only 0.36–0.58), making single-dimension alignment insufficient.

**Goal**: (1) Construct multi-dimensional preference data without reliance on manual annotation; (2) design an optimization method that jointly learns preferences from the rewriting, retrieval, and response dimensions.

**Key Insight**: Inspired by the self-consistency strategy, rewrites that are semantically highly consistent across multiple samples are considered more reliable. Based on this intuition, the authors design three distinct self-consistency scoring functions to assess rewriting quality.

**Core Idea**: Multiple candidate rewrites are sampled from an LLM and scored from three perspectives—semantic consistency of rewrites, overlap of retrieved passages, and semantic consistency of generated responses. These scores are used to construct multi-dimensional preference pairs, which are then used to train the model via prefix-guided MDPO to generate optimal rewrites under different preference conditions.

## Method

### Overall Architecture

MSPA-CQR consists of two stages: (1) **Multi-dimensional preference data construction**—$K$ candidate rewrites are sampled from an LLM and scored across three dimensions (rewriting/retrieval/response) using self-consistency metrics to form chosen/rejected pairs; (2) **Prefix-guided multi-dimensional preference optimization**—during DPO training, each sample is prepended with a preference-type prefix (e.g., `[REWRITE]`, `[RETRIEVAL]`, `[RESPONSE]`), enabling the model to distinguish and adapt to different preference dimensions. At inference time, three queries are generated using the three preference prefixes and concatenated for retrieval.

### Key Designs

1. **Three-Dimensional Self-Consistency Scoring**:

    - **Function**: Score each candidate rewritten query to assess its quality across each dimension.
    - **Mechanism**: Given $K$ candidate rewrites $\{rq_i\}$, the rewrite score $RW_i$ is computed as the mean pairwise semantic similarity with all other rewrites (via an NLI model) plus a length penalty; the retrieval score $RT_i$ is computed as the mean size of the intersection of retrieved passages across different rewrites; the response score $RP_i$ is computed as the mean pairwise semantic similarity of the corresponding generated responses (via an NLI model). The highest- and lowest-scoring candidates serve as chosen and rejected samples, respectively.
    - **Design Motivation**: Self-consistency scoring eliminates dependence on human-annotated gold passages. The three scoring functions capture complementary aspects of query quality—rewriting targets self-containedness, retrieval targets key information coverage, and response targets answer-orientedness.

2. **Prefix-Guided Multi-Dimensional DPO (MDPO)**:

    - **Function**: Enable the model to simultaneously learn preference signals from all three dimensions.
    - **Mechanism**: A prefix label set $V = \{[\text{REWRITE}], [\text{RETRIEVAL}], [\text{RESPONSE}]\}$ is defined; each preference sample is prepended with its corresponding label at training time. The training objective follows standard DPO but leverages prefixes to differentiate preference dimensions: $\mathcal{L}_{MDPO} = -\mathbb{E}[\log \sigma(\hat{r}_\theta(pr,x,rq^+) - \hat{r}_\theta(pr,x,rq^-))]$
    - **Design Motivation**: The ranking divergence across the three preference dimensions is substantial (Kendall-Tau as low as 0.36), making naive mixed training infeasible. Prefix conditioning is a lightweight yet effective mechanism that allows a single model to accommodate multiple preference types.

3. **Multi-Query Fusion at Inference**:

    - **Function**: Aggregate rewrites from all three preference dimensions for retrieval at inference time.
    - **Mechanism**: Three rewritten queries are generated using the three preference prefixes respectively, then concatenated into a single query submitted to the retrieval system.
    - **Design Motivation**: Rewrites conditioned on different preferences emphasize complementary information (self-containedness vs. retrieval keywords vs. response orientation); concatenation provides broader and more comprehensive retrieval coverage.

## Key Experimental Results

### Main Results

| Dataset | Retriever | Metric | MSPA-CQR | Prev. SOTA (RETPO) | Gain |
|--------|--------|------|----------|------------------|------|
| TopiOCQA | BM25 | MRR | 30.6 | 28.3 | +2.3 |
| TopiOCQA | BM25 | R@100 | 75.2 | 73.1 | +2.1 |
| QReCC | BM25 | MRR | 57.4 | 50.0 | +7.4 |
| QReCC | BM25 | R@100 | 95.2 | 89.5 | +5.7 |
| TopiOCQA | ANCE | MRR | 41.4 | 30.0 | +11.4 |
| QReCC | ANCE | R@10 | 72.3 | 66.7 | +5.6 |

### Ablation Study

| Configuration | TopiOCQA MRR | QReCC MRR | Notes |
|------|-------------|-----------|------|
| Full MSPA-CQR | 30.6 | 57.4 | Complete model |
| w/o Retrieval Pref | Degraded | Degraded | Removing retrieval preference hurts |
| w/o Response Pref | Degraded | Degraded | Removing response preference hurts |
| w/o Rewrite Pref | Degraded | Degraded | Removing rewrite preference hurts |
| Single Pref (retrieval only) | ~28.3 | ~50.0 | Degenerates to RETPO-like setting |

### Key Findings

- The three preference dimensions exhibit substantial divergence: on TopiOCQA, the Kendall-Tau correlation between rewriting and retrieval dimensions is only 0.36, confirming that single-dimension alignment cannot substitute for multi-dimensional alignment.
- In out-of-distribution (OOD) settings (cross-dataset transfer), MSPA-CQR remains robust, demonstrating that multi-dimensional alignment improves generalization.
- Performance gains are most pronounced under dense retrieval (ANCE), with MRR improving by 11.4 points, suggesting that multi-dimensional rewrites are particularly beneficial for semantic matching.

## Highlights & Insights

- **Self-consistency scoring as a substitute for manual annotation**: The approach elegantly leverages agreement across multiple sampled rewrites to assess quality, entirely eliminating dependence on gold passages and enabling application to any unannotated conversational data.
- **Prefix-controlled multi-preference learning**: A simple prefix label mechanism enables a single model to internalize three distinct preference types, which is far more efficient than training three separate models and allows flexible combination at inference time.
- **Three-query fusion retrieval**: Generating and concatenating three preference-conditioned rewrites at inference time produces an effect analogous to query expansion—simple yet effective.

## Limitations & Future Work

- Inference requires generating three rewritten queries and concatenating them, increasing query length and retrieval latency.
- Evaluation is conducted exclusively on English datasets (TopiOCQA, QReCC); multilingual settings remain unexplored.
- Sampling multiple candidate rewrites from an LLM incurs non-trivial computational cost during preference data construction.
- Dynamic weighting of the three preference dimensions, rather than simple concatenation, warrants further exploration.

## Related Work & Insights

- **vs. RETPO**: RETPO applies DPO alignment using only retrieval-side preference and relies on human-annotated gold passages. MSPA-CQR extends alignment to three dimensions and replaces manual annotation with self-consistency scoring.
- **vs. IterCQR**: IterCQR employs retrieval signals for reinforcement learning but relies on a single signal type. MSPA-CQR's multi-dimensional signals provide substantially richer training supervision.
- **vs. AdaCQR**: AdaCQR performs adaptive rewriting based on T5, whereas MSPA-CQR uses LLaMA-2-7B and achieves stronger generalization through preference alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-dimensional self-consistent preference alignment framework is conceptually novel, though the core techniques (DPO + prefix control) are relatively well-established.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation covers two mainstream datasets, sparse and dense retrieval, and OOD settings; ablation study details could be more complete.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and the method is described with sufficient precision.
- **Value**: ⭐⭐⭐⭐ Represents a meaningful advance for the CQR field; the self-consistency scoring paradigm is transferable to other preference alignment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[AAAI 2026\] ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](../../AAAI2026/information_retrieval/refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)
- [\[ACL 2026\] Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking](region-r1_reinforcing_query-side_region_cropping_for_multi-modal_re-ranking.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)

</div>

<!-- RELATED:END -->
