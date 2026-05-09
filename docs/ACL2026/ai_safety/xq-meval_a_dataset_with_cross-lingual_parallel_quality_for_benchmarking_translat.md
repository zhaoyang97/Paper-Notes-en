---
title: >-
  [Paper Note] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics
description: >-
  [ACL 2026][AI Safety][Translation Evaluation Metrics] This paper constructs XQ-MEval, the first translation evaluation benchmark with cross-lingual parallel quality, using semi-automated MQM error injection to generate pseudo-translations with controllable quality, empirically revealing cross-lingual scoring bias in automatic evaluation metrics for the first time, and proposing an LGN normalization strategy that effectively calibrates multilingual metric evaluation.
tags:
  - ACL 2026
  - AI Safety
  - Translation Evaluation Metrics
  - Cross-Lingual Scoring Bias
  - MQM Error Injection
  - Multilingual Benchmark
  - Metric Calibration
date: 2025-05-08
content_hash: cf465b2a4ec3e6ad
---

# XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics

**Conference**: ACL 2026  
**arXiv**: [2604.14934](https://arxiv.org/abs/2604.14934)  
**Code**: [GitHub](https://github.com/zhiqu22/XQ-MEval)  
**Area**: AI Safety  
**Keywords**: Translation Evaluation Metrics, Cross-Lingual Scoring Bias, MQM Error Injection, Multilingual Benchmark, Metric Calibration

## TL;DR

This paper constructs XQ-MEval, the first translation evaluation benchmark with cross-lingual parallel quality, using semi-automated MQM error injection to generate pseudo-translations with controllable quality, empirically revealing cross-lingual scoring bias in automatic evaluation metrics for the first time, and proposing an LGN normalization strategy that effectively calibrates multilingual metric evaluation.

## Background & Motivation

**Background**: Evaluation of multilingual translation systems typically relies on automatic metrics (COMET, MetricX, etc.), with the standard practice of averaging metric scores across language directions as the system-level score. MQM human evaluation achieves cross-lingual comparability through standardized error categories and hierarchical penalty scoring.

**Limitations of Prior Work**: The averaging strategy implicitly assumes that different languages score on the same scale for similar errors, but in practice metrics may exhibit cross-lingual scoring bias — translations of the same quality receive different scores in different languages. For example, translations with the same major error receive significantly different COMET scores across languages.

**Key Challenge**: No benchmark dataset provides cross-lingual parallel quality instances, making it impossible to systematically quantify and verify scoring bias. Expert annotation costs are extremely high, limiting language coverage.

**Goal**: (1) Construct a cross-lingual parallel quality benchmark; (2) quantify cross-lingual scoring bias; (3) propose calibration strategies to improve fairness in multilingual evaluation.

**Key Insight**: Automatically inject MQM-defined errors into high-quality reference translations, generating pseudo-translations with controllable quality through controlling error quantity, with native speaker filtering to ensure reliability.

**Core Idea**: By injecting a controllable number of MQM errors into Flores high-quality translations, construct cross-lingual quality-parallel triplets (source, pseudo-translation, reference) so that cross-lingual comparisons are grounded in identical errors.

## Method

### Overall Architecture

Three-stage pipeline: (1) Phrase-level — uses GPT-4o to inject a single MQM major error into reference translations, with native speaker filtering; (2) Sentence-level — merges 0–5 errors to generate pseudo-translations at six quality levels; (3) System-level — assembles triplets (source + pseudo-translation + reference) to construct pseudo-systems, evaluating automatic metrics with predefined scores. Covers 9 translation directions.

### Key Designs

1. **Semi-Automated Error Injection and Filtering**:

    - Function: Generate cross-lingual quality-parallel single-error candidates
    - Mechanism: For each reference translation in Flores, GPT-4o injects four MQM error types (Addition, Omission, Mistranslation, Untranslated), in both the first and second half of the sentence, producing up to 8 candidates per instance. Two native speakers independently review, retaining only unanimously approved candidates. Error types are chosen to be purely semantic, ensuring cross-lingual comparability
    - Design Motivation: LLM injection + human filtering balances cost and reliability, being far more economical than pure expert annotation while ensuring semantic equivalence of errors across languages

2. **Controllable Quality Pseudo-Translation Generation**:

    - Function: Generate pseudo-translations with predefined MQM scores
    - Mechanism: Merges 0–5 non-overlapping error segments from the error pool to generate pseudo-translations, where 0 errors corresponds to a perfect score (0 penalty) and 5 errors corresponds to the worst (-25 points, -5 per major error). Each language has parallel instances at each quality level, with quality tiers aligned across languages
    - Design Motivation: Controlling error quantity achieves a controllable quality gradient, enabling direct comparison of "same-quality" translations across languages

3. **LGN Normalization Calibration Strategy**:

    - Function: Eliminate cross-lingual scoring bias and equalize multilingual system evaluation
    - Mechanism: Language-specific Global Normalization — for each language direction, estimates the metric score distribution (mean and standard deviation) using pseudo-translation scores from XQ-MEval, then applies z-score normalization to actual evaluation scores, mapping all languages to the same scale before averaging
    - Design Motivation: When directly averaging scores across languages, high scores in high-resource languages can mask low scores in low-resource languages; LGN makes scores comparable across languages

### Loss & Training

XQ-MEval is an evaluation benchmark rather than a training method, involving no model training. LGN is a test-time calibration strategy that only requires XQ-MEval data to estimate score distribution parameters for each language.

## Key Experimental Results

### Main Results

| Metric | Average Strategy Consistency | LGN Consistency | Note |
|--------|----------------------------|-----------------|------|
| COMET-22 | Low | Significantly improved | One of the regression metrics with worst cross-lingual bias |
| MetricX-23 | Low | Improved | Similar bias issues |
| BLEU | Medium | Improved | Sequence metrics show less bias |
| chrF | Medium | Improved | Character-level metrics relatively robust |

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Bias manifestation 1: Same quality, different scores | With 1 major error, COMET score difference exceeds 0.1 between en-zh and en-ja |
| Bias manifestation 2: Inconsistent quality degradation rates | Score decline slopes vary significantly across languages as errors increase from 0 to 5 |
| LGN vs direct averaging | LGN significantly reduces cross-language score range differences |

### Key Findings

- First empirical evidence that automatic translation metrics exhibit systematic cross-lingual scoring bias, most severe in regression metrics (COMET, MetricX)
- Bias manifests in two ways: (1) different scores for same quality; (2) inconsistent quality degradation rates across languages
- The direct averaging strategy shows clear inconsistency with MQM human evaluation
- LGN normalization effectively mitigates bias, improving fairness and reliability of multilingual evaluation
- Low-resource languages (lo, si) typically exhibit more severe bias

## Highlights & Insights

- Raises a previously overlooked but critically important problem — cross-lingual scoring bias in translation metrics directly affects the fairness of multilingual system selection. This has practical implications for NMT competition rankings and product decisions
- The semi-automated construction method (LLM injection + human filtering) is an ingenious compromise, enabling coverage of 9 languages. This pipeline is generalizable to constructing other cross-lingual quality-aligned benchmarks
- The LGN strategy, while simple, is remarkably effective with low implementation cost and can be directly applied to existing evaluation workflows

## Limitations & Future Work

- Pseudo-translations are synthetic, with distributional differences from real translation system outputs
- Only covers 4 MQM error types (accounting for 46.3% of total errors); fluency and other types are not included
- Some low-resource languages have only 1 reviewer, slightly reducing reliability
- Flores has only 102 instances per direction, limiting scale
- Future work could extend to more languages and error types, and explore more sophisticated calibration methods

## Related Work & Insights

- **vs WMT MQM**: WMT MQM is expert annotation for single language directions, unable to provide cross-lingual parallel quality; this paper achieves parallelism through synthesis
- **vs COMET/MetricX**: This paper reveals systematic bias in these SOTA metrics, showing that direct score averaging may mislead system selection
- **vs Von Däniken et al. 2025**: Found that metrics may be inconsistent even within a single direction; this paper extends the analysis to the cross-lingual dimension

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study and quantification of cross-lingual bias in translation metrics
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 languages × 9 metrics — broad coverage
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem motivation, rigorous pipeline design
- Recommendation: ⭐⭐⭐⭐ Direct practical guidance for NMT evaluation fairness

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](../../CVPR2026/ai_safety/generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](../../NeurIPS2025/ai_safety/impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[ACL 2026\] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection](xlsr-mambo_scaling_the_hybrid_mamba-attention_backbone_for_audio_deepfake_detect.md)
- [\[ACL 2026\] When Bigger Isn't Better: A Comprehensive Fairness Evaluation of Political Bias in Multi-News Summarisation](when_bigger_isn39t_better_a_comprehensive_fairness_evaluation_of_political_bias_.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)

<!-- RELATED:END -->
