---
title: >-
  [Paper Note] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP
description: >-
  [ACL 2026][Multilingual & Translation][Paper Note] This paper systematically reviews the evolving role of transliteration in cross-lingual NLP, proposes a taxonomy of five motivations (Named Entity/OOV handling, code-mixing, leveraging cross-script similarity, English-centric transfer, and unified preprocessing), compares the pros and cons of six integration methods, a
tags:
  - ACL 2026
  - Multilingual & Translation
date: 2026-05-08
content_hash: 5aba0d21fa32fb9d
---
# Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.18722](https://arxiv.org/abs/2604.18722)  
**Code**: None  
**Area**: Recommender Systems  
**Keywords**: Transliteration, Cross-script transfer, Romanization, Script barriers, Multilingual language models

## TL;DR

This paper systematically reviews the evolving role of transliteration in cross-lingual NLP, proposes a taxonomy of five motivations (Named Entity/OOV handling, code-mixing, leveraging cross-script similarity, English-centric transfer, and unified preprocessing), compares the pros and cons of six integration methods, and discusses whether transliteration remains necessary in the context of modern LLMs.

## Background & Motivation

**Background**: Cross-lingual transfer is a core driver of multilingual NLP, but "script barriers"—the lack of lexical overlap between different writing systems—severely hinder knowledge transfer between languages. Transliteration, as a technique for converting one writing system to another, has become a practical solution to mitigate cross-script incompatibility.

**Limitations of Prior Work**: (1) Transliteration research is scattered across different motivations and methodologies, lacking a unified classification framework; (2) There is no systematic summary of when transliteration is beneficial versus harmful (e.g., converting ideographic scripts like Chinese to Latin can lose semantic information); (3) In the LLM era, whether large-scale pre-training has rendered transliteration unnecessary needs re-examination.

**Key Challenge**: Transliteration facilitates transfer by increasing lexical overlap, but it may simultaneously lose critical information such as semantics and morphology. It is necessary to clarify the conditions under which the benefits of transliteration outweigh its drawbacks.

**Goal**: To provide a comprehensive classification framework that guides researchers in selecting the most appropriate transliteration strategy based on language, task, and resource conditions.

**Key Insight**: Construct a classification system along two dimensions: motivation (why) and method (how), tracing technical evolution from the statistical MT era to the LLM era.

**Core Idea**: Transliteration is not merely a preprocessing technique but a bridge connecting knowledge transfer across different writing systems. Its effectiveness depends on the interaction between linguistic affinity, script type, and downstream tasks.

## Method

### Overall Architecture

The survey covers over 50 papers and is organized into the following structure: (1) Five motivation categories—why transliteration is used; (2) Six integration methods—how to incorporate transliteration into NLP pipelines; (3) Conditional analysis—when transliteration is beneficial; (4) Positioning in the LLM era—whether modern large models still require transliteration.

### Key Designs

**1. Five Motivation Categories: Organizing literature by "why" rather than method or task**

Previous reviews mostly segmented by method or downstream tasks, resulting in the same transliteration technique being scattered, which fails to show its evolving role. This paper uses motivation as the axis, identifying five categories: Named Entity and OOV processing (earliest application, transcribing foreign names/places), code-mixed text processing (unifying scripts when multiple are mixed in one sentence), leveraging cross-script linguistic similarity (aligning related languages like Hindi/Urdu written in different scripts), English-centric transfer (using English-centric pre-trained models as a springboard), and unified preprocessing (unifying multiple languages into one script to compress the vocabulary). These five categories roughly follow a timeline, revealing how transliteration evolved from a "patch" for specific issues to a systemic design choice in preprocessing pipelines.

**2. Comparison of Six Integration Methods: Providing a lookup table for system selection**

Knowing why to use it is insufficient; practitioners care about how to integrate it. This paper ranks integration methods into six levels from radical to conservative: complete replacement with transliterated corpora, transliteration augmentation (retaining original text and appending transliterated versions), vocabulary augmentation, embedding layer concatenation/fusion, prompting strategies (adding transliteration in ICL context), and multi-encoder/multi-ensemble architectures. They differ significantly in implementation complexity, performance ceilings, and application conditions—direct replacement is often simplest and potentially most effective but carries the highest risk (loss of semantics in ideographic scripts), while "original + transliterated" augmentation is a safer default.

**3. Re-examination in the LLM Era: Refuting the intuition that "Big Models make transliteration redundant"**

A natural suspicion is whether transliteration is still necessary after large-scale pre-training. This paper provides three counter-examples showing that bottlenecks remain. First, tokenizer coverage for low-resource scripts is still poor, where a single word is often fragmented into 10+ tokens; second, Romanizing low-resource languages can reduce token counts by 2–5 times, directly improving inference efficiency and cost; third, for scripts with extremely limited training data, transliteration remains the most practical access solution. The conclusion is to avoid the "LLM solves everything" misconception—tokenization bottlenecks have not disappeared in multilingual LLMs; instead, transliteration has found a new existence centered on efficiency.

### Loss & Training

A survey paper, no training involved.

## Key Experimental Results

### Main Results

The survey does not include original experiments but summarizes the following key findings:

| Condition | Transliteration Effect | Reason |
| :--- | :--- | :--- |
| Related languages + Different scripts | Strongly Positive | Maximizes lexical overlap gain |
| Ideographic scripts (Chinese, etc.) | Negative | Transliteration loses semantic information |
| Code-mixed text | Positive | Unified scripts reduce distributional mismatch |
| Poor LLM tokenizer coverage | Positive | Reduces token fragmentation |

### Key Findings

- Transliteration is most effective when related languages use different scripts (e.g., Devanagari Hindi $\rightarrow$ Latin alignment with Urdu).
- Romanization has practical value for LLM inference efficiency—Romanizing low-resource languages reduces token counts by 2-5x.
- The primary risk of transliteration is information loss—especially for tonal languages and ideographic scripts.
- Bilingual augmentation (transliteration + original text) is generally safer than pure replacement.

## Highlights & Insights

- Organizing the survey by the temporal evolution of motivations is an insightful perspective—the evolution from "patch" to "system design" reveals the maturation of the field.
- The arguments for the continued necessity of transliteration in the LLM era are very persuasive—tokenization bottlenecks are an easily overlooked but practically impactful issue.
- Specific practical recommendations provide significant value—researchers can directly consult the tables to select strategies based on language, task, and resources.

## Limitations & Future Work

- The scope of the survey emphasizes Romanization/Latinization, with less discussion on non-Latin target scripts.
- The discussion on transliteration quality (choice of tools, impact of errors) is not sufficiently deep.
- Lack of quantitative meta-analysis.

## Related Work & Insights

- **vs. Multilingual Pre-training Surveys**: Those focus on model architectures and training strategies, while Ours focuses on transliteration interventions at the data/input level.
- **vs. Code-mixing Surveys**: Those cover the full phenomenon of code-mixing, while Ours focuses on the orthogonal dimension of cross-script conversion.

## Rating

- **Novelty**: ⭐⭐⭐ The organizational framework of the survey is valuable, though it introduces no new methods.
- **Experimental Thoroughness**: ⭐⭐⭐ Survey paper with no original experiments.
- **Writing Quality**: ⭐⭐⭐⭐ The classification system is clear, and the tabular summaries are useful.
- **Value**: ⭐⭐⭐⭐ Provides practical reference value for multilingual NLP researchers.

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
