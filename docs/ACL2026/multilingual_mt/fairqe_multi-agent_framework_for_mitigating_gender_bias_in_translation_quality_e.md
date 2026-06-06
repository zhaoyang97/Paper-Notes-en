---
title: >-
  [Paper Note] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation
description: >-
  [ACL 2026][Multilingual & Machine Translation][Translation Quality Estimation] The FairQE multi-agent framework is proposed to effectively mitigate systemic gender bias in QE models through gender cue detection…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Translation Quality Estimation"
  - "Gender Bias"
  - "Multi-Agent"
  - "Fairness"
  - "Bias Mitigation"
date: 2026-05-08
content_hash: 4997e66e8a3d4719
---

# FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation

**Conference**: ACL 2026  
**arXiv**: [2604.21420](https://arxiv.org/abs/2604.21420)  
**Code**: None  
**Area**: LLM Agent / Machine Translation Evaluation  
**Keywords**: Translation Quality Estimation, Gender Bias, Multi-Agent, Fairness, Bias Mitigation

## TL;DR

The FairQE multi-agent framework is proposed to effectively mitigate systemic gender bias in QE models through gender cue detection, gender-swapped variant generation, and a dynamic bias-aware score aggregation mechanism, without sacrificing translation quality assessment accuracy.

## Background & Motivation

**Background**: Translation Quality Estimation (QE) aims to automatically assess machine translation quality without reference translations. Models such as COMETKiwi and MetricX have achieved excellent performance in WMT evaluations, becoming essential tools for translation assessment.

**Limitations of Prior Work**: Existing QE models exhibit systemic gender bias—tending to give higher scores to masculine translations in gender-ambiguous contexts and potentially favoring masculine forms even when feminine translations are explicitly required (preference reversal phenomenon). This bias cascades into downstream decisions (model selection, data filtering, etc.).

**Key Challenge**: How to mitigate gender bias while maintaining the assessment accuracy of QE models? Simple debiasing might impair the model's ability to judge translation quality.

**Goal**: Design a model-agnostic framework that can calibrate gender bias in existing QE models in a plug-and-play manner while maintaining or even improving overall assessment performance.

**Key Insight**: Adopt a multi-agent collaborative architecture to decompose bias detection, variant generation, and debiased reasoning into independent modules, combining LLM reasoning capabilities with the quantitative scoring of traditional QE models.

**Core Idea**: Quantify the degree of bias by generating gender-swapped variants and use dynamic weights to perform a soft switch between traditional QE scores and LLM debiased reasoning scores—the greater the bias, the more the system relies on LLM reasoning.

## Method

### Overall Architecture

FairQE consists of four sequential stages: (1) Gender cue detection—identifying gender-related linguistic cues in the source sentence; (2) Gender-swapped variant generation—generating masculine/feminine/neutral translation variants based on cue types; (3) Dual-stream quality assessment—traditional QE models provide quantitative scores while LLM agents perform debiased reasoning; (4) Dynamic bias-aware aggregation—dynamically adjusting the weights of the two scores based on bias severity.

### Key Designs

1.  **Gender Cue Detector ($Agent_{cue}$)**:
    - **Function**: Identifies gender-related linguistic cues in source-target sentence pairs.
    - **Mechanism**: Defines a gender bias cue taxonomy containing 12 fine-grained categories, classifying cues into gender-ambiguous and gender-explicit types, with each cue linked to corresponding segments in the source and target sentences.
    - **Design Motivation**: Different types of gender cues require different debiasing strategies (ambiguous cues require consistency, explicit cues require faithfulness), necessitating precise identification of cue types.

2.  **Gender-swapped Variant Generator ($Agent_{amb}$ + $Agent_{exp}$)**:
    - **Function**: Generates gender-swapped translation variants to quantify bias.
    - **Mechanism**: For gender-ambiguous cues, it generates all valid gender realizations (F/M/N); for gender-explicit cues, it verifies whether the target translation adheres to source gender constraints and generates swapped variants for comparison.
    - **Design Motivation**: Comparing QE scores across different gender variants allows for the quantification of the model's gender preference.

3.  **Dynamic Bias-aware Score Aggregation**:
    - **Function**: Dynamically fuses traditional QE scores and LLM debiasing scores based on bias severity.
    - **Mechanism**: Calculates ambiguous bias $b_{amb}$ (the range of variant scores) and explicit bias $b_{exp}$ (the degree of preference violation), controlling the fusion weight via a soft gate $w = B/(1+B)$. When bias is low, it relies on traditional QE; when bias is high, it leans towards LLM reasoning.
    - **Design Motivation**: Traditional QE models are stronger in fine-grained precision, while LLMs excel in reasoning-intensive tasks; dynamic aggregation leverages their complementary strengths.

### Loss & Training

Ours does not involve training and is a pure inference-time plug-and-play framework. Hyperparameters $\alpha$ and $\beta$ control the weights of ambiguous and explicit bias, respectively.

## Key Experimental Results

### Main Results (Gender-Ambiguous Scenarios - F/M QE Score Ratio)

| Method | ES | FR | IT | AR | DE | HI |
|------|-----|-----|-----|-----|-----|-----|
| COMETKiwi 22 | 0.983 | 0.978 | 0.979 | 0.985 | 0.994 | 0.991 |
| FairQE (w/ COMETKiwi 22) | **0.995** | **0.986** | **0.992** | **0.994** | **0.999** | **0.997** |

### Accuracy in Gender-Explicit Scenarios

| Method | AR | DE | HI |
|------|------|------|------|
| COMETKiwi 22 | 95.0 | 99.2 | 55.3 |
| FairQE (w/ COMETKiwi 22) | **97.3** | **99.7** | 74.0 |

### Key Findings
- FairQE consistently outperforms baseline QE models on gender fairness metrics, with F/M score ratios closer to the ideal value of 1.0.
- In MQM evaluations, FairQE achieves competitive or even superior overall QE performance, proving that debiasing does not sacrifice assessment accuracy.
- The model-agnostic design allows it to be combined with various QE models (COMETKiwi, MetricX).
- In MQM evaluations, the avg-corr reached 0.812 (w/ COMETKiwi 22), surpassing most baselines.

## Highlights & Insights
- The first unified framework to simultaneously address QE bias in both gender-ambiguous and gender-explicit scenarios.
- The dynamic aggregation mechanism elegantly balances fairness and accuracy—degenerating into the original QE model when bias is zero.
- The multi-agent design ensures clear responsibilities for each module, allowing for independent optimization.
- The approach of generating gender-swapped variants can be generalized to other types of bias detection.

## Limitations & Future Work
- High inference costs due to multiple LLM and QE model calls per sample.
- Gender cue detection relies on LLM linguistic understanding, which may be less effective for low-resource languages.
- Focuses only on gender bias; future work could explore extending the framework to other social biases (age, race, etc.).
- Reliance on the quality of LLM gender swapping; improper swaps may introduce noise.
- Optimal values for hyperparameters $\alpha$ and $\beta$ may vary by language pair and QE model, requiring consideration of tuning costs.
- Strategies for handling non-binary gender expressions were not explored.

## Related Work & Insights
- **vs COMETKiwi/MetricX**: These traditional QE models have strong assessment precision but exhibit gender bias; FairQE calibrates bias on top of them.
- **vs GEMBA-MQM**: Pure LLM methods have advantages in reasoning but are less precise than specialized QE models; FairQE combines the strengths of both.
- **vs Debiasing Training Methods**: FairQE is an inference-time plug-and-play solution that does not require retraining QE models.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-agent + dynamic aggregation debiasing framework design is novel, and the bias quantification via gender-swapped variants is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four evaluation settings cover ambiguous/explicit scenarios across multiple language pairs.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear with a high degree of mathematical formalization.
- Value: ⭐⭐⭐⭐ Addresses fairness issues of QE models in practical deployment, offering engineering utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mitigating Extrinsic Gender Bias for Bangla Classification Tasks](mitigating_extrinsic_gender_bias_for_bangla_classification_tasks.md)
- [\[ACL 2026\] TransLaw: A Large-Scale Dataset and Multi-Agent Benchmark Simulating Professional Translation of Hong Kong Case Law](translaw_a_large-scale_dataset_and_multi-agent_benchmark_simulating_professional.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)

</div>

<!-- RELATED:END -->
