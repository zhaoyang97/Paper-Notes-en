---
title: >-
  [Paper Note] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP
description: >-
  [ACL 2026][Multilingual & Machine Translation][Transliteration] This paper systematically reviews the evolving role of transliteration in cross-lingual NLP…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Transliteration"
  - "Cross-script Transfer"
  - "Romanization"
  - "Script Barriers"
  - "Multilingual Language Models"
date: 2026-05-08
content_hash: 0298564e9767a99e
---

# Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.18722](https://arxiv.org/abs/2604.18722)  
**Code**: None  
**Area**: Recommender Systems  
**Keywords**: Transliteration, Cross-script Transfer, Romanization, Script Barriers, Multilingual Language Models

## TL;DR

This paper systematically reviews the evolving role of transliteration in cross-lingual NLP, proposing five major motivation categories (Named Entity/OOV processing, code-mixing, leveraging cross-script similarity, English-centric transfer, unified preprocessing), comparing the strengths and weaknesses of six integration methods, and discussing whether transliteration remains necessary in the context of modern LLMs.

## Background & Motivation

**Background**: Cross-lingual transfer is a core driver of multilingual NLP, but "script barriers"—the lack of lexical overlap between different writing systems—heavily impede knowledge transfer between languages. Transliteration, as a technique to convert one writing system into another, has become a practical solution to mitigate cross-script incompatibility.

**Limitations of Prior Work**: (1) Transliteration research is scattered across different motivations and methodologies, lacking a unified classification framework; (2) There is no systematic summary of when transliteration is beneficial versus harmful (e.g., converting ideographic scripts like Chinese to Latin script loses semantic information); (3) In the LLM era, whether large-scale pre-training has rendered transliteration unnecessary requires re-examination.

**Key Challenge**: Transliteration facilitates transfer by increasing lexical overlap, but it may simultaneously lose critical information such as semantics and morphology. It is necessary to clarify the conditions under which the benefits of transliteration outweigh the costs.

**Goal**: To provide a comprehensive classification framework to guide researchers in selecting the most suitable transliteration strategy based on language, task, and resource conditions.

**Key Insight**: The paper constructs a classification system from two dimensions: motivation (why) and method (how), tracing technical evolution from the statistical MT era to the LLM era.

**Core Idea**: Transliteration is not merely a preprocessing technique but a bridge connecting knowledge transfer across different writing systems. Its effectiveness depends on the interaction between linguistic relatedness, script type, and downstream tasks.

## Method

### Overall Architecture

The survey covers over 50 papers, organized as follows: (1) Five major motivation categories—why transliteration is used; (2) Six integration methods—how to incorporate transliteration into NLP pipelines; (3) Condition analysis—when transliteration is beneficial; (4) Positioning in the LLM era—whether modern large models still require transliteration.

### Key Designs

1.  **Five-fold Motivation Taxonomy**:
    *   Function: Systematizing the understanding of transliteration's role in NLP.
    *   Mechanism: (1) Named Entity and OOV processing—the earliest applications; (2) Code-mixed text processing—handling multiple scripts within the same text; (3) Leveraging cross-script similarity—when related languages use different scripts (e.g., Hindi/Urdu); (4) English-centric transfer—leveraging English-centric pre-trained models; (5) Unified preprocessing—reducing vocabulary size for multilingual models.
    *   Design Motivation: Existing literature classifies by method or task, ignoring the temporal evolution of motivations. Categorizing by motivation reveals a paradigm shift from "patching" to "system design."

2.  **Comparison of Six Integration Methods**:
    *   Function: Guiding practitioners toward the most suitable transliteration strategy.
    *   Mechanism: (1) Total replacement with transliterated corpora; (2) Transliteration augmentation (retaining original + adding transliterated version); (3) Vocabulary augmentation; (4) Embedding concatenation/fusion; (5) Prompting strategies (including transliteration in ICL); (6) Multi-encoder/ensemble architectures.
    *   Design Motivation: Different integration methods vary significantly in complexity, effectiveness, and applicability—simple replacement might be the most effective but also the riskiest.

3.  **Revisiting the LLM Era**:
    *   Function: Assessing whether modern large-scale pre-training makes transliteration redundant.
    *   Mechanism: Even in the LLM era, (1) tokenizer coverage for low-resource scripts remains insufficient (one word may be split into 10+ tokens); (2) Romanization can significantly improve inference efficiency (reducing token counts); (3) For scripts with extremely limited training data, transliteration remains the most practical solution.
    *   Design Motivation: Avoiding the "LLM solves everything" misconception—tokenization bottlenecks persist in multilingual LLMs.

### Loss & Training

A survey paper; does not involve training.

## Key Experimental Results

### Main Results

The survey does not contain its own experiments but summarizes the following key findings:

| Condition | Transliteration Effect | Reason |
| :--- | :--- | :--- |
| Related languages + different scripts | Strongly Positive | Maximizes lexical overlap gains |
| Ideographic scripts (Chinese, etc.) | Negative | Transliteration loses semantic information |
| Code-mixed text | Positive | Unified scripts reduce distribution mismatch |
| Poor LLM tokenizer coverage | Positive | Reduces token fragmentation |

### Key Findings

*   Transliteration is most effective when related languages use different scripts (e.g., Devanagari Hindi aligned with Urdu after Romanization).
*   Romanization has practical value for LLM inference efficiency—Romanizing low-resource languages can reduce token counts by 2-5 times.
*   The main risk of transliteration is information loss—particularly for tonal languages and ideographic scripts.
*   Bilingual augmentation (transliteration + original text) is generally safer than pure replacement.

## Highlights & Insights

*   Organizing the survey by the temporal evolution of motivations provides a thought-provoking perspective—the shift from "patches" to "system design" reveals the maturation of the field.
*   The argument for the continued necessity of transliteration in the LLM era is highly persuasive—the tokenization bottleneck is an easily overlooked but practically impactful issue.
*   Specific practical recommendations provide significant value—researchers can directly consult the summary to select strategies based on language, task, and resource constraints.

## Limitations & Future Work

*   The scope of the survey focuses heavily on Romanization/Latinization, with less discussion on non-Latin target scripts.
*   The discussion on transliteration quality (tool selection, impact of errors) is not sufficiently deep.
*   Lack of a quantitative meta-analysis.

## Related Work & Insights

*   **vs. Multilingual Pre-training Surveys**: Those focus on model architectures and training strategies, while this work focuses on transliteration interventions at the data/input level.
*   **vs. Code-mixing Surveys**: Those cover the full phenomenon of code-mixing, while this work focuses on the orthogonal dimension of cross-script conversion.

## Rating

*   Novelty: ⭐⭐⭐ The organization framework of the survey itself is valuable, though it does not introduce new methods.
*   Experimental Thoroughness: ⭐⭐⭐ Survey paper with no original experiments.
*   Writing Quality: ⭐⭐⭐⭐ Clear taxonomy and useful summary tables.
*   Value: ⭐⭐⭐⭐ Highly practical reference value for multilingual NLP researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources](bhashasutra_a_task-centric_unified_survey_of_indian_nlp_datasets_corpora_and_res.md)
- [\[ACL 2026\] SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams](serm_self-evolving_relevance_model_with_agent-driven_learning_from_massive_query.md)
- [\[AAAI 2026\] NADIR: Differential Attention Flow for Non-Autoregressive Transliteration in Indic Languages](../../AAAI2026/multilingual_mt/nadir_differential_attention_flow_for_non-autoregressive_transliteration_in_indi.md)
- [\[ACL 2026\] SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics](steereval_inference-time_interventions_strengthen_multilingual_generalization_in.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)

</div>

<!-- RELATED:END -->
