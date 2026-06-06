---
title: >-
  [Paper Note] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search
description: >-
  [ACL 2026 Findings][Information Retrieval & RAG][Conversational Query Rewriting] This paper proposes MSPA-CQR, which constructs self-consistent preference data across three dimensions—rewriting, retrieval…
tags:
  - "ACL 2026 Findings"
  - "Information Retrieval & RAG"
  - "Conversational Query Rewriting"
  - "Preference Alignment"
  - "Self-Consistency Scoring"
  - "Multi-dimensional DPO"
  - "Conversational Search"
date: 2026-05-08
content_hash: db83e23301061737
---

# Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.06771](https://arxiv.org/abs/2604.06771)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Conversational Query Rewriting, Preference Alignment, Self-Consistency Scoring, Multi-dimensional DPO, Conversational Search

## TL;DR

This paper proposes MSPA-CQR, which constructs self-consistent preference data across three dimensions—rewriting, retrieval, and response—and trains the query rewriting model using prefix-guided multi-dimensional DPO. The method significantly outperforms existing approaches in both in-distribution and out-of-distribution scenarios.

## Background & Motivation

**Background**: In Conversational Question Answering (CQA), user queries often exhibit ambiguity (e.g., coreference and ellipsis), requiring Conversational Query Rewriting (CQR) to transform vague queries into complete, self-contained versions to facilitate downstream retrieval. Early methods relied on human-annotated rewrites, which are costly and often focus on readability rather than retrieval performance.

**Limitations of Prior Work**: Recent studies have introduced retrieval signals as feedback, but two issues remain: (1) they only consider preferences in the retrieval dimension, ignoring feedback from rewrite and response quality; (2) the construction of preference data relies on human-annotated gold passages, hindering generalization to unlabelled data.

**Key Challenge**: A high-quality rewritten query must simultaneously satisfy three requirements: the rewrite itself must be self-contained, the retrieval should include key information without redundancy, and the corresponding response must be reasonable and accurate. Preferences across these three dimensions vary (Kendall-Tau correlation is only 0.36-0.58), making alignment via a single dimension insufficient.

**Goal**: (1) Construct multi-dimensional preference data without relying on human annotations; (2) Design an optimization method capable of learning preferences from rewriting, retrieval, and response dimensions simultaneously.

**Key Insight**: Inspired by self-consistency strategies, if multiple rewriting results are semantically highly consistent, these rewrites are more reliable. Based on this, the authors design three different self-consistency scoring methods to measure rewrite quality.

**Core Idea**: Sample multiple candidate rewrites using an LLM, score and rank them from the perspectives of rewrite semantic consistency, retrieval result intersection, and response semantic consistency to construct multi-dimensional preference pairs. Then, use prefix-guided MDPO to enable the model to generate optimal rewrites under different preference constraints.

## Method

### Overall Architecture

MSPA-CQR consists of two stages: (1) Multi-dimensional preference data construction—sampling $K$ candidate rewritten queries with an LLM and performing self-consistency scoring across rewriting/retrieval/reponse dimensions to select chosen/rejected pairs; (2) Prefix-guided multi-dimensional preference optimization—adding preference type prefixes (e.g., [REWRITE], [RETRIEVAL], [RESPONSE]) to each data point during DPO training to help the model distinguish and adapt to different preference dimensions. During inference, three queries are generated using the three preference labels and concatenated for retrieval.

### Key Designs

1.  **Three-dimensional Self-consistency Scoring**:
    *   Function: Scores each candidate rewritten query to measure its quality across different dimensions.
    *   Mechanism: For $K$ candidate rewrites $\{rq_i\}$, the rewrite score $RW_i$ is calculated using an NLI model as the mean semantic similarity to other rewrites with a length penalty; the retrieval score $RT_i$ calculates the mean intersection size of retrieved passages; the response score $RP_i$ uses an NLI model to calculate the mean semantic similarity between corresponding responses. Samples with the highest and lowest scores are used as chosen and rejected samples, respectively.
    *   Design Motivation: Self-consistency scoring avoids dependence on human-annotated gold passages, and the three scoring methods capture query quality from different angles: rewriting focuses on self-containment, retrieval on key information, and response on answer-orientation.

2.  **Prefix-guided Multi-dimensional DPO (MDPO)**:
    *   Function: Enables the model to learn preference information from three dimensions simultaneously.
    *   Mechanism: Defines a prefix label set $V = \{[REWRITE], [RETRIEVAL], [RESPONSE]\}$ and prepends the corresponding preference label to the input of each preference data point. The training objective is similar to standard DPO but uses prefixes to distinguish preference dimensions: $$\mathcal{L}_{MDPO} = -\mathbb{E}[\log \sigma(\hat{r}_\theta(pr,x,rq^+) - \hat{r}_\theta(pr,x,rq^-))]$$
    *   Design Motivation: Significant ranking differences between the three preference dimensions (Kendall-Tau as low as 0.36) suggest that mixed training is ineffective. Prefix control is a lightweight yet effective way for a single model to adapt to multiple preferences.

3.  **Multi-query Fusion Inference**:
    *   Function: Integrates rewriting results from three preferences for retrieval during inference.
    *   Mechanism: Generates three rewritten queries using the three preference prefixes, then concatenates them into a single long query for the retrieval system.
    *   Design Motivation: Different preference-oriented rewrites emphasize different information (self-containment vs. retrieval keywords vs. response-oriented); concatenation provides a more comprehensive coverage of retrieval needs.

## Key Experimental Results

### Main Results

| Dataset | Retriever | Metric | MSPA-CQR | RETPO (Prev. SOTA) | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TopiOCQA | BM25 | MRR | 30.6 | 28.3 | +2.3 |
| TopiOCQA | BM25 | R@100 | 75.2 | 73.1 | +2.1 |
| QReCC | BM25 | MRR | 57.4 | 50.0 | +7.4 |
| QReCC | BM25 | R@100 | 95.2 | 89.5 | +5.7 |
| TopiOCQA | ANCE | MRR | 41.4 | 30.0 | +11.4 |
| QReCC | ANCE | R@10 | 72.3 | 66.7 | +5.6 |

### Ablation Study

| Configuration | TopiOCQA MRR | QReCC MRR | Description |
| :--- | :--- | :--- | :--- |
| Full MSPA-CQR | 30.6 | 57.4 | Full Model |
| w/o Retrieval Pref | Decrease | Decrease | Performance drops without retrieval preference |
| w/o Response Pref | Decrease | Decrease | Performance drops without response preference |
| w/o Rewrite Pref | Decrease | Decrease | Performance drops without rewrite preference |
| Single Pref (Ret. only) | ~28.3 | ~50.0 | Degrades to RETPO-like performance |

### Key Findings

*   Significant differences exist between the three preference dimensions: Kendall-Tau between rewriting and retrieval on TopiOCQA is only 0.36, indicating a single preference cannot replace multi-dimensional alignment.
*   In OOD scenarios (cross-dataset transfer), MSPA-CQR remains robust, proving that multi-dimensional alignment enhances generalization capacity.
*   The gain is more significant in dense retrieval (ANCE) scenarios (MRR increase of 11.4), suggesting that multi-faceted rewriting is more beneficial for semantic matching.

## Highlights & Insights

*   **Self-consistency scoring replaces human annotation**: Cleverly utilizes the consistency of multiple samples to measure rewrite quality, completely avoiding reliance on gold passages and allowing application to any unlabelled conversational data.
*   **Prefix control for multi-preference learning**: Simple prefix labels allow a single model to distinguish between three preferences, which is much more efficient than training three independent models and allows flexible combinations during inference.
*   **Three-query fusion retrieval**: Generating and concatenating three preference-oriented rewrites during inference acts like query expansion, which is simple yet effective.

## Limitations & Future Work

*   Generating and concatenating three rewritten queries during inference increases query length and retrieval latency.
*   The method was only validated on English datasets (TopiOCQA, QReCC); multilingual scenarios have not been explored.
*   The cost of sampling multiple candidate rewrites using an LLM is high, making the computational overhead of the preference data construction stage non-negligible.
*   Future work could explore dynamic weighting of the three preference dimensions instead of simple concatenation.

## Related Work & Insights

*   **vs RETPO**: RETPO only uses retrieval preferences for DPO alignment and relies on human-annotated gold passages. MSPA-CQR expands to three dimensions and replaces human annotation with self-consistency.
*   **vs IterCQR**: IterCQR uses retrieval signals for reinforcement learning, but the signal is singular. The multi-dimensional signals in MSPA-CQR provide richer training information.
*   **vs AdaCQR**: AdaCQR performs adaptive rewriting based on T5; MSPA-CQR uses LLaMA-2-7B and achieves stronger generalization through preference alignment.

## Rating

*   Novelty: ⭐⭐⭐⭐ The idea of three-dimensional self-consistent preference alignment is novel, though the core techniques (DPO + prefix control) are relatively established.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers two major datasets, sparse/dense retrieval, and OOD evaluation, though ablation details could be more comprehensive.
*   Writing Quality: ⭐⭐⭐⭐ Clear derivation of motivation and complete method description.
*   Value: ⭐⭐⭐⭐ Provides practical advancement to the CQR field; the self-consistency scoring ideal is transferable to other preference alignment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ICML 2026\] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](../../ICML2026/information_retrieval/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)
- [\[AAAI 2026\] ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](../../AAAI2026/information_retrieval/refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](../../CVPR2026/information_retrieval/bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)

</div>

<!-- RELATED:END -->
