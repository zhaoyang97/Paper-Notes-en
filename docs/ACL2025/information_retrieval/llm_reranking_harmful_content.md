---
title: >-
  [Paper Note] Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms
description: >-
  [ACL 2025][Information Retrieval & RAG][Content moderation] Proposes a pairwise preference re-ranking method based on LLMs to demote harmful content in social media recommendation sequences under zero-shot and few-shot settings. The method significantly outperforms industrial-grade classifiers such as Perspective API and OpenAI Moderation API, while introducing two new evaluation metrics: PP-k and EWN.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Content moderation"
  - "LLM re-ranking"
  - "harmful content"
  - "social media"
  - "pairwise comparison"
date: 2026-05-08
content_hash: 239af302ed8c7bd7
---

# Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms

**Conference**: ACL 2025  
**arXiv**: [2501.13977](https://arxiv.org/abs/2501.13977)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Content moderation, LLM re-ranking, harmful content, social media, pairwise comparison

## TL;DR
Proposes a pairwise preference re-ranking method based on LLMs to demote harmful content in social media recommendation sequences under zero-shot and few-shot settings. The method significantly outperforms industrial-grade classifiers such as Perspective API and OpenAI Moderation API, while introducing two new evaluation metrics: PP-k and EWN.

## Background & Motivation
**Background**: Social media platforms use recommendation algorithms to maximize user engagement, which can lead to user exposure to harmful content (e.g., misinformation, hate speech, addictive content). Existing moderation relies heavily on classifiers trained on large-scale human-annotated data.

**Limitations of Prior Work**: (a) Classifiers require huge amounts of annotated data, offering poor scalability; (b) the forms of harmful content change dynamically (concept drift), making it difficult for classifiers to automatically generalize to new types of harmful content.

**Key Challenge**: Traditional classifiers struggle to balance the need for large-scale processing with the ability to adapt to dynamically changing definitions of harmful content.

**Goal**: Can LLMs' reasoning capabilities be leveraged to reduce the exposure of harmful content via re-ranking under zero-shot and few-shot settings?

**Key Insight**: Instead of performing absolute harmful/safe classification, relative harmfulness is assessed through pairwise comparison. The recommendation sequence is then re-ranked based on this relative comparison to demote harmful content to the end of the sequence (demotion rather than deletion, thus preserving freedom of expression).

**Core Idea**: Using LLMs for pairwise harmfulness comparison and re-ranking is more effective at mitigating exposure to harmful content than trained specialized classifiers.

## Method

### Overall Architecture
The input is a recommendation sequence $X = \{x_i\}_{i=1}^n$ (e.g., YouTube video descriptions). The LLM performs pairwise comparisons on all content pairs $(x_i, x_j)$ to determine which is more harmful. After aggregating the scores, they are sorted in ascending order—harmful content is naturally pushed to the end of the sequence.

### Key Designs
1. **Preferential Pairwise Ranking**:

    - **Function**: Performs pairwise comparison on all content pairs in a sequence using an LLM to determine which is more harmful.
    - **Mechanism**: Improves the scoring function from Qin et al. (2024)—content judged as harmful receives a score of +1; if the LLM considers both contents safe, the comparison is skipped (no points added) to prevent safe content from being erroneously demoted. Finally, ascending sorting is applied based on the scores.
    - **Design Motivation**: Pairwise comparison leverages the LLM's context window, offering higher accuracy than independent classification; demotion rather than deletion preserves freedom of expression.

2. **Three Preference Constraint Settings (Preference Constraints)**:

    - Zero-Shot: Relies purely on the LLM's intrinsic understanding of harmfulness.
    - Zero-Shot + Prompt Engineering: Explicitly defines the criteria and characteristics of "harmfulness" in the prompt.
    - Few-Shot ICL: Provides examples of harmful content, using K-Means clustering to select representative exemplars to avoid bias toward specific types.
    - **Design Motivation**: Gradually provides more information to study the performance of LLMs under different information constraints.

3. **New Evaluation Metrics**:

    - PP-k (Per-Pref-k): Measures what percentage of the sequence a user needs to consume to encounter the $k$-th harmful item; higher values are better.
    - EWN (Exponentially Weighted Normalization): Measures ranking quality using exponentially decaying weights, normalized to $[0,1]$ independently of the proportion of harmful content.

## Key Experimental Results

### Main Results (YouTube Dataset, 30% Harmful Ratio)

| Method | TP-5 | TP-10 | EWN |
|------|------|-------|-----|
| Original | ~0.72 | ~0.70 | 0.487 |
| Perspective API | ~0.78 | ~0.74 | ~0.65 |
| OpenAI Moderation | ~0.80 | ~0.75 | ~0.68 |
| Zero-Shot (GPT-3.5) | ~0.84 | ~0.80 | ~0.73 |
| Zero-Shot + PE | ~0.87 | ~0.83 | ~0.76 |
| Few-Shot ICL | ~0.88 | ~0.84 | ~0.77 |

### Cross-Model Experiments

| LLM | EWN (Zero-Shot) | EWN (Few-Shot) |
|-----|------------------|----------------|
| GPT-3.5-Turbo | Highest | Highest |
| Mistral-7B | Lags by ~10% | Lags by ~10% |
| Llama2-13B | Below OpenAI Mod baseline | Below baseline |

### Key Findings
- Even in zero-shot settings, LLM re-ranking outperforms Perspective API and OpenAI Moderation which are trained on large datasets.
- When the proportion of harmful content increases from 10% to 50%, the performance degradation of the LLM method (~23%) is far less than that of traditional methods (~40%).
- Increasing the number of exemplars in ICL does not always yield better performance; 4 exemplars are sufficient, and too many can degrade performance.
- Although Mistral-7B has only 7B parameters, its performance is only ~10% behind GPT-3.5, making it viable for local deployment.

## Highlights & Insights
- The "demotion instead of deletion" strategy is highly practical for real-world platform moderation, preserving freedom of expression while mitigating exposure.
- The pairwise comparison approach can be adapted to other tasks requiring relative judgments (e.g., content quality ranking, recommendation diversity optimization).
- The clustering-based selection of ICL exemplars ensures representativeness and prevents bias towards specific categories of harmful content.

## Limitations & Future Work
- The time complexity of pairwise comparison is $O(n^2)$, leading to high API costs as sequence length increases.
- Only text descriptions are used, without leveraging multimodal information such as videos or images.
- Llama2-13B performs poorly, indicating that the method relies on a certain level of LLM capability.
- The actual impact on user engagement and retention is not evaluated.

## Related Work & Insights
- **vs Perspective API**: As a specialized toxicity detection tool, it performs worse than the general reasoning capabilities of LLMs.
- **vs Qin et al. (2024)**: Adapts their pairwise ranking framework but improves the scoring function to suit harmful content demotion scenarios.
- **vs Traditional Classifiers**: Eliminates the need for annotated data and training, making it more robust against concept drift.

## Rating
- **Novelty**: ⭐⭐⭐ The idea of pairwise re-ranking is not entirely new, but applying it to harmful content moderation has high practical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough cross-dimensional comparisons across 3 datasets, 3 models, and 3 settings.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with a complete derivation of the new metrics.
- **Value**: ⭐⭐⭐⭐ Direct application value to platform content moderation, with zero-shot capability being a key highlight.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](rare_retrieval_augmented_reasoning.md)
- [\[ACL 2025\] Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers](drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)
- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[ACL 2025\] Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](astute_rag_knowledge_conflicts.md)

</div>

<!-- RELATED:END -->
