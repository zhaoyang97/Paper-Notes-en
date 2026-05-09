---
title: >-
  [Paper Note] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation
description: >-
  [ACL 2026][LLM Agent][Translation Quality Estimation] This paper proposes FairQE, a multi-agent framework that mitigates systematic gender bias in QE models through gender cue detection, gender-flipped variant generation, and dynamic bias-aware score aggregation, without sacrificing translation quality estimation accuracy.
tags:
  - ACL 2026
  - LLM Agent
  - Translation Quality Estimation
  - Gender Bias
  - Multi-Agent
  - Fairness
  - Bias Mitigation
date: 2026-05-08
content_hash: 99a7c37a2d2fd7fe
---

# FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation

**Conference**: ACL 2026
**arXiv**: [2604.21420](https://arxiv.org/abs/2604.21420)
**Code**: None
**Area**: LLM Agent / Machine Translation Evaluation
**Keywords**: Translation Quality Estimation, Gender Bias, Multi-Agent, Fairness, Bias Mitigation

## TL;DR

This paper proposes FairQE, a multi-agent framework that mitigates systematic gender bias in QE models through gender cue detection, gender-flipped variant generation, and dynamic bias-aware score aggregation, without sacrificing translation quality estimation accuracy.

## Background & Motivation

**Background**: Translation Quality Estimation (QE) aims to automatically evaluate machine translation quality without reference translations. Models such as COMETKiwi and MetricX have achieved strong performance in WMT evaluations and have become important tools for translation assessment.

**Limitations of Prior Work**: Existing QE models exhibit systematic gender bias — they tend to assign higher scores to masculine translations in gender-ambiguous contexts, and may still prefer masculine forms even when feminine translations are explicitly required (preference reversal). Such bias can cascade into downstream decisions, including model selection and data filtering.

**Key Challenge**: The core tension lies in mitigating gender bias while preserving QE model accuracy. Naive debiasing may impair the model's ability to judge translation quality.

**Goal**: To design a model-agnostic framework that calibrates gender bias in existing QE models in a plug-and-play manner while maintaining or improving overall evaluation performance.

**Key Insight**: A multi-agent collaborative architecture is adopted, decomposing bias detection, variant generation, and debiased inference into independent modules, combining LLM reasoning capabilities with the quantitative scoring of traditional QE models.

**Core Idea**: Gender-flipped variants are generated to quantify the degree of bias, and dynamic weights are used to softly switch between traditional QE scores and LLM-based debiased reasoning scores — the greater the bias, the more the framework relies on LLM reasoning.

## Method

### Overall Architecture

FairQE consists of four sequential stages: (1) gender cue detection — identifying gender-related linguistic cues in source sentences; (2) gender-flipped variant generation — producing masculine/feminine/neutral translation variants based on cue type; (3) dual-stream quality estimation — traditional QE models provide quantitative scores while an LLM agent performs debiased reasoning; and (4) dynamic bias-aware aggregation — dynamically adjusting the weights between the two score streams according to bias severity.

### Key Designs

1. **Gender Cue Detector ($Agent_{cue}$)**:

   - Function: Identifies gender-related linguistic cues in source–target sentence pairs.
   - Mechanism: Defines a gender bias cue taxonomy comprising 12 fine-grained categories, classifying cues into gender-ambiguous and gender-explicit types, with each cue linked to corresponding spans in both the source and target sentences.
   - Design Motivation: Different types of gender cues require different debiasing strategies — ambiguous cues require consistency, while explicit cues require faithfulness — necessitating precise cue type identification.

2. **Gender-Flipped Variant Generator ($Agent_{amb}$ + $Agent_{exp}$)**:

   - Function: Generates gender-flipped translation variants to quantify bias.
   - Mechanism: For gender-ambiguous cues, all valid gender realizations (F/M/N) are generated; for gender-explicit cues, the framework verifies whether the target translation conforms to the gender constraints of the source sentence and generates flipped variants for contrastive analysis.
   - Design Motivation: Comparing QE scores across gender variants enables quantification of the model's gender preference.

3. **Dynamic Bias-Aware Score Aggregation**:

   - Function: Dynamically fuses traditional QE scores and LLM debiased scores according to bias severity.
   - Mechanism: Ambiguous bias $b_{amb}$ (range of scores across variants) and explicit bias $b_{exp}$ (degree of preference violation) are computed; a soft gate $w = B/(1+B)$ controls the fusion weight. When bias is small, the framework relies on traditional QE; when bias is large, it defers to LLM reasoning.
   - Design Motivation: Traditional QE models are stronger in fine-grained accuracy, while LLMs excel at reasoning-intensive tasks; dynamic aggregation leverages the complementary strengths of both.

### Loss & Training

FairQE involves no training and is a purely inference-time plug-and-play framework. Hyperparameters $\alpha$ and $\beta$ control the weights assigned to ambiguous and explicit bias, respectively.

## Key Experimental Results

### Main Results (Gender-Ambiguous Setting — F/M QE Score Ratio)

| Method | ES | FR | IT | AR | DE | HI |
|--------|-----|-----|-----|-----|-----|-----|
| COMETKiwi 22 | 0.983 | 0.978 | 0.979 | 0.985 | 0.994 | 0.991 |
| FairQE (w/ COMETKiwi 22) | **0.995** | **0.986** | **0.992** | **0.994** | **0.999** | **0.997** |

### Accuracy in Gender-Explicit Setting

| Method | AR | DE | HI |
|--------|------|------|------|
| COMETKiwi 22 | 95.0 | 99.2 | 55.3 |
| FairQE (w/ COMETKiwi 22) | **97.3** | **99.7** | 74.0 |

### Key Findings
- FairQE consistently outperforms baseline QE models on gender fairness metrics, with F/M score ratios closer to the ideal value of 1.0.
- In MQM evaluation, FairQE achieves competitive or superior overall QE performance, demonstrating that debiasing does not sacrifice evaluation accuracy.
- The model-agnostic design enables compatibility with multiple QE models (COMETKiwi, MetricX).
- FairQE achieves an avg-corr of 0.812 on MQM evaluation (w/ COMETKiwi 22), surpassing most baselines.

## Highlights & Insights
- This is the first unified framework addressing QE bias in both gender-ambiguous and gender-explicit scenarios.
- The dynamic aggregation mechanism elegantly balances fairness and accuracy — when bias is zero, the framework degenerates to the original QE model.
- The multi-agent design ensures clear separation of responsibilities across modules, each of which can be independently optimized.
- The idea of generating gender-flipped variants for bias quantification is generalizable to other types of bias detection.

## Limitations & Future Work
- Multiple LLM and QE model calls are required per sample, resulting in relatively high inference cost.
- Gender cue detection relies on the LLM's language understanding capabilities, which may be limited for low-resource languages.
- The framework focuses solely on gender bias; future work could explore extending it to other forms of social bias (e.g., age, race).
- The quality of gender-flipped variants depends on the LLM, and inappropriate flips may introduce noise.
- Optimal values of hyperparameters $\alpha$ and $\beta$ may vary across language pairs and QE models, incurring non-trivial tuning costs.
- Strategies for handling non-binary gender expressions remain unexplored.

## Related Work & Insights
- **vs COMETKiwi/MetricX**: These traditional QE models achieve strong evaluation accuracy but exhibit gender bias; FairQE calibrates bias on top of them.
- **vs GEMBA-MQM**: Pure LLM-based approaches offer stronger reasoning capabilities but lower accuracy than dedicated QE models; FairQE combines the advantages of both.
- **vs Debiasing via Retraining**: FairQE is an inference-time plug-and-play solution that requires no retraining of QE models.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-agent + dynamic aggregation debiasing framework is novel, and the use of gender-flipped variants for bias quantification is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four evaluation settings covering gender-ambiguous and gender-explicit scenarios across multiple language pairs.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly with a high degree of mathematical formalization.
- Value: ⭐⭐⭐⭐ Addresses fairness issues in deployed QE models with practical engineering utility.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering](mata_multi-agent_framework_for_reliable_and_flexible_table_question_answering.md)
- [\[CVPR 2026\] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](../../CVPR2026/llm_agent/nerfify_multiagent_nerf_paper_to_code.md)
- [\[AAAI 2026\] ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](../../AAAI2026/llm_agent/arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](../../CVPR2026/llm_agent/think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)

<!-- RELATED:END -->
