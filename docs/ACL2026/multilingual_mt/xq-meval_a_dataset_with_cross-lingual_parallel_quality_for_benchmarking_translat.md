---
title: >-
  [Paper Note] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics
description: >-
  [ACL 2026][Multilingual & Machine Translation][Translation evaluation metrics] The authors construct XQ-MEval, the first translation evaluation benchmark with cross-lingual parallel quality. By generating quality-control…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Translation evaluation metrics"
  - "Cross-lingual scoring bias"
  - "MQM error injection"
  - "Multilingual benchmark"
  - "Metric calibration"
date: 2026-05-08
content_hash: fecc8c5497a79923
---

# XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.14934](https://arxiv.org/abs/2604.14934)  
**Code**: [GitHub](https://github.com/zhiqu22/XQ-MEval)  
**Area**: AI Safety  
**Keywords**: Translation evaluation metrics, Cross-lingual scoring bias, MQM error injection, Multilingual benchmark, Metric calibration

## TL;DR
The authors construct XQ-MEval, the first translation evaluation benchmark with cross-lingual parallel quality. By generating quality-controlled pseudo-translations via semi-automatic MQM error injection, they empirically reveal cross-lingual scoring biases in automatic metrics for the first time and propose an LGN normalization strategy to effectively calibrate multilingual metric evaluations.

## Background & Motivation

**Background**: Evaluation of multilingual translation systems typically relies on automatic metrics (COMET, MetricX, etc.). The standard practice is to average metric scores across various language directions to obtain a system-level score. MQM human evaluation achieves cross-lingual comparability through standardized error categories and hierarchical penalty points.

**Limitations of Prior Work**: The averaging strategy implicitly assumes that metric scores for similar errors across different language pairs share the same scale. However, metrics may exhibit cross-lingual scoring bias—translations of identical quality receiving different scores in different languages. For instance, a translation containing a single "major" error might receive significantly different COMET scores depending on the target language.

**Key Challenge**: There is no benchmark dataset providing cross-lingual parallel quality instances, making it impossible to systematically quantify and verify scoring bias. Furthermore, the extremely high cost of expert annotation limits the scope of language coverage.

**Goal**: (1) Construct a cross-lingual parallel quality benchmark; (2) Quantify cross-lingual scoring bias; (3) Propose calibration strategies to improve the fairness of multilingual evaluation.

**Key Insight**: Automatically inject errors defined by MQM into high-quality reference translations. By controlling the number of errors, quality-controlled pseudo-translations are generated, with reliability ensured through native speaker filtering.

**Core Idea**: By injecting a controlled number of MQM errors into high-quality Flores translations, the authors construct cross-lingual quality-parallel triplets (source, pseudo-translation, reference), allowing cross-lingual comparisons to be established on the basis of identical errors.

## Method

### Overall Architecture
A three-stage pipeline: (1) Phrase-level—GPT-4o is used to inject a single MQM major error into the reference translation, followed by native speaker filtering; (2) Sentence-level—Merging 0–5 errors to generate pseudo-translations with six quality levels; (3) System-level—Assembling triplets (source + pseudo-translation + reference) to construct pseudo-systems, evaluating automatic metrics with predefined scores. It covers 9 translation directions.

### Key Designs

1. **Semi-automatic Error Injection and Filtering**:
    - **Function**: Generate single-error candidates with cross-lingual parallel quality.
    - **Mechanism**: For each translation instance in Flores, GPT-4o injects four types of MQM errors (Addition, Omission, Mistranslation, Untranslated) into the reference, placed separately in either the first or second half of the sentence, producing up to 8 candidates per instance. Two native speakers independently review them, retaining only those with unanimous approval. Purely semantic error types are chosen to ensure cross-lingual comparability.
    - **Design Motivation**: The combination of LLM injection and human filtering balances cost and reliability, being much more economical than pure expert annotation while ensuring semantic equivalence of errors across languages.

2. **Controlled Quality Pseudo-translation Generation**:
    - **Function**: Generate pseudo-translations with predefined MQM scores.
    - **Mechanism**: Non-overlapping error segments from the error pool (0–5 errors) are merged to generate pseudo-translations. Zero errors correspond to a full score (0 penalty), while 5 errors correspond to the lowest quality (-25 points, with each major error penalizing 5 points). Parallel instances exist for every quality level across each language pair, ensuring quality hierarchies are aligned cross-lingually.
    - **Design Motivation**: Implementing a controlled quality gradient via error counts enables direct comparison of "equal quality" translations across different languages.

3. **LGN Normalization Strategy**:
    - **Function**: Eliminate cross-lingual scoring bias and fairify multilingual system evaluation.
    - **Mechanism**: Language-specific Global Normalization—For each language direction, the metric score distribution (mean and standard deviation) is estimated using pseudo-translations from XQ-MEval. Actual evaluation scores are then z-score normalized to map all languages to the same scale before averaging.
    - **Design Motivation**: When directly averaging scores across languages, high scores from high-resource languages can mask low scores from low-resource languages. LGN makes scores comparable across languages.

### Loss & Training
XQ-MEval is an evaluation benchmark rather than a training method and does not involve model training. LGN is a test-time calibration strategy that only requires estimating distribution parameters for each language using XQ-MEval data.

## Key Experimental Results

### Main Results

| Metric | Averaging Strategy Consistency | LGN Consistency | Description |
| :--- | :--- | :--- | :--- |
| COMET-22 | Low | Significant Improvement | One of the regression metrics with the most severe bias |
| MetricX-23 | Low | Improvement | Similar bias issues |
| BLEU | Moderate | Improvement | Smaller bias in sequence-based metrics |
| chrF | Moderate | Improvement | Character-level metrics are relatively robust |

### Ablation Study

| Analysis Dimension | Finding |
| :--- | :--- |
| Bias Manifestation 1: Same Quality, Different Scores | For 1 major error, the COMET score difference between en-zh and en-ja exceeds 0.1 |
| Bias Manifestation 2: Inconsistent Quality Decay | When errors increase from 0 to 5, the slope of score decline varies significantly across languages |
| LGN vs. Direct Averaging | LGN significantly reduces the variance in score ranges across languages |

### Key Findings
- Empirical evidence confirms that automatic translation metrics possess systematic cross-lingual scoring biases, which are most severe in regression-based metrics (COMET, MetricX).
- Bias manifests in two ways: (1) different scores for the same quality; (2) inconsistent quality decay rates across languages.
- Direct averaging strategies show clear inconsistency with MQM human evaluations.
- LGN normalization effectively mitigates bias, improving the fairness and reliability of multilingual evaluation.
- Biases are typically more severe in low-resource languages (e.g., lo, si).

## Highlights & Insights
- The study addresses a previously overlooked but critical issue: cross-lingual scoring bias in translation metrics directly affects the fairness of multilingual system selection. This has practical implications for NMT competition rankings and product decisions.
- The semi-automatic construction method (LLM injection + human filtering) is a clever compromise that makes coverage of 9 languages feasible. This pipeline can be extended to build other benchmarks with cross-lingual quality alignment.
- The LGN strategy, while simple, is highly effective and low-cost to implement, allowing for direct application to existing evaluation workflows.

## Limitations & Future Work
- Pseudo-translations are synthetic and exhibit distributional differences compared to real translation system outputs.
- Only 4 types of MQM errors are covered (representing 46.3% of total errors); other types like fluency are not included.
- Some low-resource languages were reviewed by only one annotator, slightly weakening reliability.
- Each direction in Flores contains only 102 instances, resulting in a limited scale.
- Future work could expand to more languages and error types while exploring more complex calibration methods.

## Related Work & Insights
- **vs. WMT MQM**: WMT MQM uses expert annotations for single language directions and cannot provide cross-lingual parallel quality; this work achieves parallelism through synthesis.
- **vs. COMET/MetricX**: This work reveals systematic biases in these SOTA metrics, indicating that direct averaging of scores may mislead system selection.
- **vs. Von Däniken et al. 2025**: While they found that metrics could be inconsistent even in a single direction, this study expands the analysis to the cross-lingual dimension.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study and quantification of cross-lingual bias in translation metrics.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across 9 languages and 9 metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation and rigorous pipeline design.
- Value: ⭐⭐⭐⭐ Provides direct practical guidance for fairness in NMT evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[ACL 2026\] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation](beyond_literal_mapping_benchmarking_and_improving_non-literal_translation_evalua.md)
- [\[ACL 2026\] IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents](indotabvqa_a_benchmark_for_cross-lingual_table_understanding_in_bahasa_indonesia.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)

</div>

<!-- RELATED:END -->
