---
title: >-
  [Paper Note] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion
description: >-
  [ACL 2026][Information Retrieval & RAG][Multilingual RAG] This paper demonstrates that the apparent "English preference" in multilingual RAG systems is primarily an artifact of structural priors embedded in evaluation be…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Multilingual RAG"
  - "English-centric bias"
  - "language preference"
  - "query fusion"
  - "debiased calibration"
date: 2026-05-08
content_hash: c11e61f895e361d2
---

# Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion

**Conference**: ACL 2026
**arXiv**: [2601.02956](https://arxiv.org/abs/2601.02956)  
**Code**: [GitHub](https://github.com/jeonghyunpark2002/DELTA)  
**Area**: Information Retrieval / Multilingual RAG
**Keywords**: Multilingual RAG, English-centric bias, language preference, query fusion, debiased calibration

## TL;DR

This paper demonstrates that the apparent "English preference" in multilingual RAG systems is primarily an artifact of structural priors embedded in evaluation benchmarks (i.e., gold evidence concentrated in English and cultural priors) rather than an intrinsic model bias. The authors propose a debiased language preference metric, DeLP, which reveals that retrievers actually prefer monolingual alignment. Building on this insight, they design the DELTA query augmentation framework, which consistently outperforms English-pivot strategies on multilingual RAG benchmarks.

## Background & Motivation

**Background**: Multilingual RAG (mRAG) enhances LLMs' cross-lingual question answering capabilities by retrieving evidence from multilingual knowledge sources. The English-pivot strategy—translating non-English queries into English before retrieval—is widely regarded as an effective heuristic.

**Limitations of Prior Work**: (1) The community broadly attributes the effectiveness of English-pivot to LLMs' "English-centric" capabilities, including stronger English-language reasoning and reduced translation noise. (2) However, this paper finds that the apparent "English preference" is primarily driven by structural biases in evaluation benchmarks: in MKQA, 73.3% of gold evidence resides in the English Wikipedia, while other languages account for only 0.5–1.4%. (3) Existing metrics such as MLRS cannot distinguish between a model's genuine preference and external necessity imposed by data distribution.

**Key Challenge**: English-pivot appears effective not because models prefer English, but because correct answers are almost exclusively found in English resources—a consequence of data imbalance rather than model bias. Once these structural confounds are removed, the model's true preference remains unclear.

**Goal**: (1) Identify the true source of "English preference" in mRAG. (2) Design the debiased metric DeLP to measure intrinsic language preference. (3) Leverage the post-debiasing insights to develop superior mRAG strategies.

**Key Insight**: The authors identify three categories of structural priors—exposure prior (high-resource corpora dominating retrieval results), gold availability prior (correct evidence concentrated in English), and cultural prior (region-specific topics bound to particular languages)—and regress these priors out of the raw preference signal via ridge regression.

**Core Idea**: After debiasing, the retrievers' true preference is monolingual alignment (i.e., retrieval performs best when query and document languages match), not English preference. Accordingly, queries should be augmented into multilingual anchors to exploit monolingual alignment, rather than being blindly translated into English.

## Method

### Overall Architecture

**DeLP metric**: Collect raw language preference signals → construct prior feature vectors (exposure / gold availability / cultural / corpus size / passage length) → fit priors via ridge regression → residuals constitute the debiased true preference.
**DELTA framework**: Given a query → use DeLP signals to identify the language set preferred by the model → translate the query into the preferred languages → fuse the original query with translated queries for retrieval → generate the answer.

### Key Designs

1. **DeLP: Debiased Language Preference Metric**

    - **Function**: Disentangle the model's intrinsic language preference from structural confounds.
    - **Mechanism**: Decompose raw preference into a prior-explained component (exposure, gold availability, and cultural priors) and a residual (true preference). Ridge regression fits the priors as $s_e(L_q, L_d) \approx w^\top \phi(L_q, L_d) + \epsilon$; the residual $\epsilon$ constitutes the DeLP score.
    - **Design Motivation**: Existing metrics conflate data distribution effects with model preference. DeLP isolates the model's true preference by explicitly regressing out known structural factors.

2. **Monolingual Alignment Discovery**

    - **Function**: Reveal the retriever's intrinsic language preference pattern.
    - **Mechanism**: Applying DeLP shows that apparent English preference diminishes substantially (from seemingly dominant to moderate), while the monolingual alignment signal strengthens—retrieval performs best when query and document languages match (e.g., a Japanese query retrieving from Japanese Wikipedia).
    - **Design Motivation**: If the model's true preference is monolingual alignment rather than English, the English-pivot strategy merely exploits the abundance of English resources indirectly, rather than being the optimal approach.

3. **DELTA Query Augmentation Framework**

    - **Function**: Guide query augmentation using debiased language preference signals.
    - **Mechanism**: DeLP signals are used to dynamically identify the language set most preferred by the model for a given query. The query is then translated into these preferred languages, and the original and translated queries are fused for retrieval. This preserves the contextual information of the original script while maximizing the benefits of monolingual alignment.
    - **Design Motivation**: Rather than blindly translating into English, DELTA selects the most advantageous languages based on the model's true preference. The framework is lightweight—operating solely at the query level without requiring modification of the retriever or corpus—and adapts dynamically.

### Loss & Training

No model training is involved. Evaluation uses existing retrievers (BGE-m3) and generators (Qwen3-235B, DeepSeek-v3.1, Gemini-2.5-Flash).

## Key Experimental Results

### Main Results

**End-to-end multilingual RAG accuracy (selected languages)**

| Method | ko | zh | ja | ar | Avg. |
|--------|----|----|----|----|------|
| Baseline (original query) | Low | Low | Low | Low | Low |
| English-pivot | Mid | Mid | Mid | Mid | Mid |
| **DELTA** | **High** | **High** | **High** | **High** | **Highest** |

### Ablation Study

**Effect of structural priors on preference measurement**

| Metric | English Preference | Monolingual Alignment Signal |
|--------|-------------------|------------------------------|
| MLRS (raw) | Strong | Weak |
| **DeLP (debiased)** | **Weak** | **Strong** |

### Key Findings

- English Wikipedia covers 73.3% of gold evidence, while other languages account for only 0.5–1.4%; the "effectiveness" of English-pivot primarily stems from this extreme imbalance.
- After debiasing, the English preference diminishes substantially, and monolingual alignment emerges as the dominant preference—retrievers perform best when query and document languages match.
- DELTA consistently outperforms English-pivot, demonstrating that exploiting the model's true preference is more effective than following biased environmental signals.
- Cultural prior is also a significant confound: correct answers to region-specific questions are more likely to reside in the corresponding language's Wikipedia.

## Highlights & Insights

- The systematic deconstruction of the "English preference myth" is the paper's central contribution, exposing a significant blind spot in existing evaluation methodology.
- The design philosophy of DeLP—examining residuals after regressing out known priors—is transferable to any evaluation scenario involving confounding factors.
- DELTA is extremely lightweight, operating solely at the query level without requiring modifications to the model, retriever, or corpus.

## Limitations & Future Work

- The debiasing effectiveness of DeLP depends on the completeness of the identified prior factors; unrecognized confounds may still influence conclusions.
- Validation is conducted only on the MKQA benchmark; findings may differ on other multilingual QA benchmarks.
- The translation step in DELTA introduces additional latency.
- The influence of training-time biases within the retriever itself on language preference is not explored.

## Related Work & Insights

- **vs. English-pivot strategy**: This paper demonstrates that the effectiveness of English-pivot stems from data imbalance rather than model preference.
- **vs. MLRS**: MLRS conflates structural priors with model preference; DeLP reveals the true signal through debiasing.
- **vs. CoPriva**: CoPriva addresses textual privacy preservation, whereas this paper focuses on debiasing language preference.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The deconstruction of the "English preference myth" and the debiased language preference metric constitute significant contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated with three strong LLMs, but limited to the single MKQA benchmark.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Analytical logic is rigorous; the identification and argumentation of structural biases are convincing.
- **Value**: ⭐⭐⭐⭐ — Reframes the understanding of multilingual RAG; both DeLP and DELTA offer direct practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)
- [\[AAAI 2026\] ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](../../AAAI2026/information_retrieval/refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
