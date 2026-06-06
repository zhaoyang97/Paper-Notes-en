---
title: >-
  [Paper Note] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion
description: >-
  [ACL 2026][Information Retrieval & RAG][Multilingual RAG] This paper discovers that "English preference" in multilingual RAG systems is primarily an artifact of structural priors in evaluation benchmarks (gold evidence c…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Multilingual RAG"
  - "English-centric bias"
  - "language preference"
  - "query fusion"
  - "debiased calibration"
date: 2026-05-08
content_hash: ca3c25bda5ed0a41
---

# Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion

**Conference**: ACL 2026  
**arXiv**: [2601.02956](https://arxiv.org/abs/2601.02956)  
**Code**: [GitHub](https://github.com/jeonghyunpark2002/DELTA)  
**Area**: Information Retrieval / Multilingual RAG  
**Keywords**: Multilingual RAG, English-centric bias, language preference, query fusion, debiased calibration

## TL;DR

This paper discovers that "English preference" in multilingual RAG systems is primarily an artifact of structural priors in evaluation benchmarks (gold evidence concentration in English, cultural priors) rather than inherent model bias. It proposes a debiased language preference metric, DeLP, to reveal that retrievers actually prefer monolingual alignment. Based on this, it designs the DELTA query augmentation framework, which consistently outperforms English-pivot strategies in multilingual RAG.

## Background & Motivation

**Background**: Multilingual RAG (mRAG) enhances the cross-lingual answering capabilities of LLMs by retrieving evidence from multilingual knowledge sources. English-pivot (translating non-English queries to English before retrieval) is widely considered an effective heuristic strategy.

**Limitations of Prior Work**: (1) Academia generally attributes the effectiveness of English-pivot to the "English-centric" capabilities of LLMs—stronger English reasoning and less translation noise; (2) However, this paper finds that this "English preference" is mainly driven by structural biases in evaluation benchmarks—73.3% of gold evidence in benchmarks like MKQA exists in English Wikipedia, while other languages account for only 0.5-1.4%; (3) Existing metrics (such as MLRS) cannot distinguish between the model's true preference and the external necessity imposed by data distribution.

**Key Challenge**: English-pivot appears effective not because the model prefers English, but because correct answers exist almost exclusively in English resources—this is data imbalance, not model bias. What is the model's true preference once these structural confounding factors are removed?

**Goal**: (1) Reveal the true source of "English preference" in mRAG; (2) Design a debiased metric, DeLP, to measure the model's inherent language preference; (3) Design better mRAG strategies based on debiased insights.

**Key Insight**: Identify three types of structural priors—exposure prior (high-resource corpora dominating retrieval results), gold availability prior (correct evidence concentrated in English), and cultural prior (regional topics tied to specific languages)—and then regress these priors out from the raw preference signal using ridge regression.

**Core Idea**: After debiasing, it is discovered that the retriever's true preference is monolingual alignment (retrieval is most effective when query and document languages match) rather than English preference. Therefore, queries should be augmented into multilingual anchors to leverage monolingual alignment, rather than blindly translating to English.

## Method

### Overall Architecture

DeLP Metric: Collect raw language preference signals → Construct prior feature vectors (exposure/gold availability/cultural/corpus size/passage length) → Fit priors using ridge regression → Residuals represent the debiased true preference. DELTA Framework: Given a query → Use DeLP signals to identify the set of languages preferred by the model → Translate the query into these preferred languages → Fuse original and translated queries for retrieval → Generate answers.

### Key Designs

1.  **DeLP Debiased Language Preference Metric**:
    - **Function**: Separate the model's inherent language preference from structural confounding factors.
    - **Mechanism**: Decompose the raw preference into a prior-explained part (exposure, gold availability, cultural priors) and a residual (true preference). Use ridge regression $s_e(L_q, L_d) \approx w^\top \phi(L_q, L_d) + \epsilon$ to fit the priors; the residual $\epsilon$ is the DeLP score.
    - **Design Motivation**: Existing metrics conflate data distribution effects with model preference. DeLP reveals the model's true preference by explicitly regressing out known structural factors.

2.  **Monolingual Alignment Discovery**:
    - **Function**: Reveal the retriever's inherent language preference patterns.
    - **Mechanism**: After applying DeLP, it was found that English preference significantly shrinks (from apparent dominance to a moderate level), while the monolingual alignment signal strengthens—performance is highest when the query language and document language match (e.g., Japanese query retrieving from Japanese Wikipedia).
    - **Design Motivation**: If the model truly prefers monolingual alignment rather than English, then the English-pivot strategy only indirectly exploits the richness of English resources rather than being the optimal strategy.

3.  **DELTA Query Augmentation Framework**:
    - **Function**: Utilize debiased language preferences to guide query augmentation.
    - **Mechanism**: Dynamically identify the model's most preferred language set for a given query based on DeLP signals, translate the query into these preferred languages, and then fuse the original and translated queries for retrieval. This preserves the original script's context while maximizing the benefits of monolingual alignment.
    - **Design Motivation**: Instead of blindly translating to English, select the most advantageous languages based on the model's true preferences—it is lightweight (no modification of the retriever or corpus) and dynamically adaptive.

### Loss & Training

No model training is involved. Evaluation is conducted using existing retrievers (BGE-m3) and generators (Qwen3-235B, DeepSeek-v3.1, Gemini-2.5-Flash).

## Key Experimental Results

### Main Results

**End-to-End Multilingual RAG Accuracy (Selected Languages)**

| Method | ko | zh | ja | ar | Average |
|------|-----|-----|-----|-----|------|
| Base (Original Language Query) | Low | Low | Low | Low | Low |
| English-pivot | Mid | Mid | Mid | Mid | Mid |
| **DELTA** | **High** | **High** | **High** | **High** | **Highest** |

### Ablation Study

**Impact of Structural Priors on Preference Metrics**

| Metric | English Preference | Monolingual Alignment Signal |
|------|---------|-----------|
| MLRS (Raw) | Strong | Weak |
| **DeLP (Debiased)** | **Weak** | **Strong** |

### Key Findings

- English Wikipedia covers 73.3% of gold evidence, while other languages range from 0.5-1.4%—the "effectiveness" of English-pivot primarily stems from this extreme imbalance.
- After debiasing, English preference shrinks significantly, and monolingual alignment becomes the dominant preference—retrievers perform best when query and document languages match.
- DELTA consistently outperforms English-pivot—proving that utilizing the model's true preference is more effective than following biased environmental signals.
- Cultural prior is also a significant confounding factor—correct answers for regional topics are more likely to exist in the corresponding language's Wikipedia.

## Highlights & Insights

- The systematic deconstruction of the "English preference myth" is the core contribution of this paper—it reveals a major blind spot in evaluation methodology.
- The design of the DeLP metric (regressing out known priors to examine residuals) is transferable to any evaluation scenario involving confounding factors.
- DELTA is extremely lightweight—it operates only at the query level and requires no modifications to the model, retriever, or corpus.

## Limitations & Future Work

- The debiasing effect of DeLP depends on the completeness of identifying prior factors—unidentified confounders may still influence conclusions.
- Validated only on the MKQA benchmark; conclusions might vary on other multilingual QA benchmarks.
- The translation step in DELTA introduces additional latency.
- Training biases of the retriever itself regarding language preference were not explored.

## Related Work & Insights

- **vs English-pivot Strategy**: This paper proves the effectiveness of English-pivot results from data imbalance rather than model preference.
- **vs MLRS**: MLRS conflates structural priors and model preference, whereas DeLP reveals true signals through debiasing.
- **vs CoPriva**: While CoPriva studies text privacy protection, this paper focuses on debiasing language preferences.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The deconstruction of the "English preference myth" and the debiased language preference metric are significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated with three strong LLMs, though limited to the MKQA benchmark.
- Writing Quality: ⭐⭐⭐⭐⭐ Logic is rigorous; the identification and demonstration of structural bias are convincing.
- Value: ⭐⭐⭐⭐ It shifts the understanding of multilingual RAG; both DeLP and DELTA offer direct practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)
- [\[ACL 2026\] GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models](gift_guided_fine-tuning_and_transfer_for_enhancing_instruction-tuned_language_mo.md)
- [\[ACL 2026\] IF-GEO: Conflict-Aware Instruction Fusion for Multi-Query Generative Engine Optimization](if-geo_conflict-aware_instruction_fusion_for_multi-query_generative_engine_optim.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)

</div>

<!-- RELATED:END -->
