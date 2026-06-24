---
title: >-
  [Paper Note] gec-metrics: A Unified Library for Grammatical Error Correction Evaluation
description: >-
  [ACL 2025][Text Generation][Grammatical Error Correction Evaluation] This paper proposes `gec-metrics`, a unified library that integrates 10 grammatical error correction (GEC) evaluation metrics into a single interface. It also provides meta-evaluation functionalities, addressing the issues of fragmentation, non-reproducibility, and limited extensibility in existing GEC evaluation implementations.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Grammatical Error Correction Evaluation"
  - "Unified Framework"
  - "Meta-Evaluation"
  - "GEC"
  - "Evaluation Metrics"
date: 2026-05-08
content_hash: 47920bb4ee65c25b
---

# gec-metrics: A Unified Library for Grammatical Error Correction Evaluation

**Conference**: ACL 2025  
**arXiv**: [2505.19388](https://arxiv.org/abs/2505.19388)  
**Code**: [GitHub](https://github.com/gotutiyan/gec-metrics)  
**Area**: Text Generation  
**Keywords**: Grammatical Error Correction Evaluation, Unified Framework, Meta-Evaluation, GEC, Evaluation Metrics  

## TL;DR

This paper proposes `gec-metrics`, a unified library that integrates 10 grammatical error correction (GEC) evaluation metrics into a single interface. It also provides meta-evaluation functionalities, addressing the issues of fragmentation, non-reproducibility, and limited extensibility in existing GEC evaluation implementations.

## Background & Motivation

**Background**: Grammatical Error Correction (GEC) is the task of automatically correcting spelling, tense, word usage, and other grammatical errors. A variety of evaluation metrics have been developed, such as ERRANT, GLEU, SOME, and IMPARA. These metrics are broadly categorized into reference-based and reference-free metrics.

**Limitations of Prior Work**:
   - **Inconsistent Interfaces**: Existing evaluation metrics utilize different input and output formats, hindering cross-metric evaluation. For instance, evaluation of existing GEC models relies heavily on ERRANT, while metrics like IMPARA—which correlate better with human judgments—are rarely reported.
   - **Lack of Official Resources**: Implementations for Scribendi and LLM-{S,E} have not been publicly released, and pre-trained weights for IMPARA are unavailable. This results in inconsistent scores for the same metric reported across different papers (e.g., the reported Pearson $r$ for Scribendi on GJG15 ranges from 0.303 to 0.951).
   - **Lack of API Support**: Most existing implementations are CLI scripts, which cannot be easily integrated into downstream applications such as reinforcement learning reward functions or minimum Bayes risk (MBR) decoding.

**Key Challenge**: While GEC model architectures have been standardized (e.g., UnifiedGEC), evaluation metrics remain fragmented, restricting model development and fair comparison.

**Goal**: Build a unified GEC evaluation library that supports fair comparison across metrics, meta-evaluation, and extensible development.

**Key Insight**: Leveraging the successful paradigm of HuggingFace Transformers + Evaluate—using a unified framework to accelerate scientific research.

**Core Idea**: Address the fragmentation of GEC evaluation through a unified interface, standard implementation, and a unified meta-evaluation framework.

## Method

### Overall Architecture

The `gec-metrics` system consists of two major types of interfaces:
- **Metric Class**: A unified evaluation interface that supports two granularities: `score_corpus()` and `score_sentence()`.
- **MetaEval Class**: A unified meta-evaluation interface that supports both system-level and sentence-level evaluation.

### Key Designs

#### Supported Evaluation Metrics (10 types)

**Reference-based Metrics**:
- **Edit-level**: ERRANT ($F_\beta$ on edit overlap), PT-ERRANT (BERTScore-weighted edits), GoToScorer (weighted by correction difficulty)
- **N-gram-level**: GLEU (precision-based), GREEN ($F_\beta$ score)

**Reference-free Metrics**:
- **Sentence-level**: SOME (grammaticality + fluency + meaning preservation), Scribendi (perplexity-based), IMPARA (similarity + quality estimation), LLM-S (LLM 5-stage evaluation), LLM-E (edit-sequence evaluation)

#### Meta-Evaluation Framework

Supports two meta-evaluation datasets, GJG15 and SEEDA:
- **System-level**: Pearson ($r$) and Spearman ($\rho$) correlation coefficients
- **Sentence-level**: Accuracy (Acc.) and Kendall ($\tau$) rank correlation coefficients
- SEEDA supports SEEDA-S (sentence-level human evaluation) and SEEDA-E (edit-level human evaluation), with configurations for both Base and +Fluency.

#### Extensibility Design

- All classes inherit from an abstract base class, requiring the implementation of only minimal methods like `score_sentence()` to add new metrics.
- CLI supports YAML configuration inputs to ensure experiment reproducibility.
- A GUI interface (powered by Streamlit) is provided, enabling evaluation without coding.

#### Analysis and Visualization Tools

- **Window analysis**: Analyzes evaluation performance based on rank differences.
- **Pairwise analysis**: Groups and statistics metric agreement rates based on human ranking differences.
- **Edit-level attribution**: Analyzes the types of edit operations that metrics focus on.

### Loss & Training

As a library paper, this work does not involve model training. However, the authors reproduced and released resources for metrics lacking official pre-trained weights:
- IMPARA: Fine-tuned a bert-base-cased model using 3,276 training instances generated from CoNLL-2013, and released the public weights.
- LLM-{S,E}: Provided the first public implementation, supporting the OpenAI API, Gemini API, and HuggingFace causal language models.

## Key Experimental Results

### Main Results

**System-level (SEEDA-E +Fluency, TrueSkill)**:

| Metric | Pearson $r$ | Spearman $\rho$ |
|---|---|---|
| ERRANT | -0.508 | 0.033 |
| GREEN | 0.252 | 0.618 |
| GLEU | 0.232 | 0.569 |
| SOME | **0.943** | **0.969** |
| IMPARA | 0.900 | 0.978 |
| Scribendi | 0.715 | 0.842 |
| GPT-4-S | 0.390 | 0.714 |
| Qwen2.5-S | 0.790 | 0.930 |

**Sentence-level (SEEDA-S Base)**:

| Metric | Accuracy | Kendall $\tau$ |
|---|---|---|
| ERRANT | 0.594 | 0.189 |
| GLEU | 0.672 | 0.343 |
| SOME | **0.778** | **0.555** |
| IMPARA | 0.753 | 0.506 |

### Metric Ensembling Experiment

By building a simple ensemble that averages the rankings of non-LLM metrics, this method achieved the highest system-level Spearman correlation of $\rho = 0.984$ on SEEDA-E.

### Key Findings

- **ERRANT Fails under the +Fluency Setting**: The system-level correlation becomes negative ($r = -0.508$), indicating that edit-level metrics struggle to evaluate fluency improvements.
- **SOME and IMPARA are the Most Robust**: Both perform well across all datasets and settings, including the +Fluency setting.
- **Generalization of LLM Metrics Requires Verification**: LLM-based metrics perform well on SEEDA (with $\rho$ up to 0.930), but perform poorly when evaluated on GJG15 for the first time, indicating limited generalization capability.
- **Pairwise Analysis Insights**: The larger the difference in human ranking, the higher the metric's accuracy. Conversely, capability to distinguish between closely ranked systems remains weak.
- **Window Analysis**: IMPARA's correlation suddenly drops at $x=7$ on SEEDA-S, which aligns with observations in prior literature.

## Highlights & Insights

- **Fills Infrastructure Gap**: Similar to how HuggingFace Evaluate serves NLP evaluation, `gec-metrics` serves GEC evaluation by lowering the barrier to entry and facilitating fair comparisons.
- **First Open-Source Implementation of LLM-{S,E}**: Confirmed details directly with the original authors, providing a valuable resource to the community.
- **Reveals Instability in LLM Evaluation**: GPT-4-E exhibits massive performance discrepancies across different datasets, suggesting the need for further validation.
- **Simple Ensembles are Efficient**: Simply averaging the ranks of multiple metrics yields the highest correlation, demonstrating that different metrics capture complementary dimensions.

## Limitations & Future Work

- Currently only supports English GEC; multi-lingual support needs to be expanded.
- Meta-evaluation datasets are limited (only GJG15 and SEEDA); building new datasets is highly costly.
- LLM-based metrics are expensive to run; the paper uses gpt-4o-mini instead of gpt-4 to control costs.
- IMPARA requires self-reproduction and training, which might introduce minor implementation discrepancies.

## Related Work & Insights

- UnifiedGEC (Zhao et al., 2025) unified GEC models $\rightarrow$ This work unifies GEC evaluation.
- HuggingFace Evaluate (Von Werra et al., 2022) unified evaluation paradigm $\rightarrow$ Directly inspired the design of this work.
- GMEG-Metric (Napoles et al., 2019) ensemble method $\rightarrow$ Simple experiments in this work validate the merit of ensembling.
- mbrs (Deguchi et al., 2024) architectural design $\rightarrow$ Inspired the code structure of `gec-metrics`.

## Rating

- **Novelty**: ⭐⭐⭐ — As a codebase/utility library paper, technical novelty is limited, but it resolves genuine community pain points.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparison covering 10 metrics across multiple meta-evaluation datasets.
- **Writing Quality**: ⭐⭐⭐⭐ — Clarifies problems thoroughly, provides clean code examples, and ensures high reproducibility.
- **Value**: ⭐⭐⭐⭐⭐ — Serving as an infrastructure-level tool, it holds long-term impact for the GEC community and has already been adopted by shared tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?](rethinking_evaluation_metrics_for_grammatical_error_correction_why_use_a_differe.md)
- [\[ACL 2025\] IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator](impara-ged_grammatical_error_detection_is_boosting_reference-free_grammatical_er.md)
- [\[ACL 2025\] Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study](enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study.md)
- [\[ACL 2025\] A Representation Level Analysis of NMT Model Robustness to Grammatical Errors](a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors.md)
- [\[ACL 2025\] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)

</div>

<!-- RELATED:END -->
