---
title: >-
  [Paper Note] Your Model is Overconfident, and Other Lies We Tell Ourselves
description: >-
  [ACL 2025][Data Complexity] Through a comprehensive analysis of 29 models on the ChaosNLI and DynaSent datasets, this work reveals a correlated but non-linear and non-monotonic relationship among data complexity metrics such as annotator disagreement, training dynamics, and model confidence, challenging the common assumption that "model uncertainty $\approx$ human disagreement."
tags:
  - "ACL 2025"
  - "Data Complexity"
  - "Annotator Disagreement"
  - "Model Uncertainty"
  - "Calibration Analysis"
  - "Intrinsic Difficulty"
date: 2026-05-08
content_hash: 8e07c946350d0d84
---

# Your Model is Overconfident, and Other Lies We Tell Ourselves

**Conference**: ACL 2025  
**arXiv**: [2503.01235](https://arxiv.org/abs/2503.01235)  
**Code**: None  
**Area**: Other  
**Keywords**: Data Complexity, Annotator Disagreement, Model Uncertainty, Calibration Analysis, Intrinsic Difficulty

## TL;DR

Through a comprehensive analysis of 29 models on the ChaosNLI and DynaSent datasets, this work reveals a correlated but non-linear and non-monotonic relationship among data complexity metrics such as annotator disagreement, training dynamics, and model confidence, challenging the common assumption that "model uncertainty $\approx$ human disagreement."

## Background & Motivation

In NLP evaluation, the intrinsic difficulty of data is a crucial but often overlooked factor. Prior work typically conflates the following three concepts:
1. **Annotator disagreement**: Multiple annotators assigning different labels to the same instance.
2. **Data uncertainty**: The inherent randomness or noise in the data.
3. **Data complexity**: Classification difficulty caused by structural characteristics of the instance.

In the literature, model uncertainty (e.g., predictive entropy, training dynamics) is frequently used as a proxy for human disagreement to evaluate instance difficulty, implicitly assuming that "instances that models find difficult are also difficult for humans." However, does this assumption hold? What is the actual relationship between different complexity metrics? This is the core problem investigated in this paper.

## Method

### Overall Architecture

This paper proposes a systematic experimental framework to define and compare multiple data complexity metrics, categorized into three major groups:
- Human-based metrics: annotator disagreement rate, annotation distribution entropy.
- Reference-free model-based metrics: model pool disagreement/entropy, average model entropy, Conformal Prediction set size.
- Reference-dependent model-based metrics: model fail rate, early layer termination, early training termination, training-fold fail rate, training-fold probability mass.

### Key Designs

1. **Multi-model pool construction strategy**: Two complementary model pools are designed—a heterogeneous training pool (5 different 1B LLMs: OLMo, Pythia, Llama 3.2, Falcon, and BLOOM, each trained on different subsets of the NLI training set, totaling 25 classifiers) and a homogeneous training pool (BERT models of various sizes trained on the full dataset)—to validate the stability of metrics from different perspectives.

2. **Conformal Prediction set size**: Conformal prediction (specifically, the Least-Ambiguous Set-Valued Classifier) is used to quantify the level of ambiguity required for the classifier to achieve statistical guarantees. A larger set indicates a more ambiguous instance. Experiments are conducted under three risk tolerance levels: $\alpha = 0.05$, $0.1$, and $0.2$. Although a labeled calibration set is required, the prediction itself does not need gold labels, thus categorizing it as reference-free.

3. **Early computation/training termination metrics**: Based on the concept by Baldock et al., this assesses at which layer (or at which training checkpoint) the model begins to consistently make correct predictions. An earlier stabilization indicates a simpler instance. For Transformer models, intermediate layer representations are cleverly projected directly onto the classification head instead of using the original kNN method, resolving the issue of inconsistent representation dimensions across layers in sequence classification tasks.

### Loss & Training

- 1B model pool: 5 LLMs $\times$ 5 data subsets = 25 classifiers, trained using standard cross-entropy loss.
- BERT model pool: multiple BERT variants of different sizes fine-tuned on the full NLI dataset.
- All metrics are computed on the test sets (ChaosNLI / DynaSent).

## Key Experimental Results

### Main Results

Spearman correlation coefficients between human metrics and model metrics on SNLI (1B pool):

| Model Metric | ℍ_ent | ℍ_dis |
|---------|-------|-------|
| 𝕄_dis (Model pool disagreement) | 0.244 | 0.218 |
| 𝕄_ent (Model pool entropy) | 0.278 | 0.243 |
| 𝕄_avg_ent (Average model entropy) | 0.390 | 0.349 |
| 𝕄_fail^ref (Model failure rate) | 0.399 | 0.396 |
| 𝕄_1st_ckpt^ref (Early training termination) | 0.436 | 0.424 |
| 𝕄_avg_ckpt_p^ref (Training probability mass) | 0.439 | 0.424 |

On MNLI, the correlation between the heterogeneous 1B pool and human metrics is close to 0 (e.g., 𝕄_dis = -0.002), whereas the homogeneous <1B pool provides a weak positive correlation (~0.14-0.25).

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| Reference-free vs Reference-dependent | Reference-dependent metrics systematically correlate better with human disagreement than reference-free metrics |
| Heterogeneous 1B pool vs Homogeneous <1B pool | The composition of the model pool significantly affects metric behavior; the heterogeneous pool shows almost no correlation on MNLI |
| CP Set Size vs Training Dynamics | The two types of metrics do not align; instances that models deem difficult are not necessarily perceived as difficult by humans |
| SNLI vs MNLI | Trends are not entirely consistent across datasets, indicating that the relationships are highly data-dependent |

### Key Findings

- The correlation between all model metrics and human disagreement is weak (Spearman < 0.45), and the relationship is non-linear and non-monotonic.
- Reference-free metrics (which do not consider prediction correctness) show almost no correlation with human disagreement.
- Conflict arises between different types of model complexity metrics: conformal prediction methods do not align with training dynamics methods.
- The conclusions hold true on the DynaSent sentiment analysis dataset, demonstrating the generalizability of the findings.
- Models can make errors on instances where humans achieve high agreement, and they can also reach consensus on instances with high human disagreement.

## Highlights & Insights

- Systematically deconstructs three commonly conflated concepts (disagreement, uncertainty, and complexity) with a rigorous experimental design.
- Unveils an overlooked yet critical issue: using model uncertainty as a proxy for human disagreement is unreliable.
- The design of multiple model pools (heterogeneous vs. homogeneous) demonstrates the sensitivity of the experimental conclusions.
- Provides important cautionary insights for application scenarios that rely on "model difficulty," such as active learning and data curation.

## Limitations & Future Work

- Evaluated only on NLI and sentiment analysis tasks, without covering generative tasks or a wider range of NLP tasks.
- All models are encoder-based (BERT, 1B LLMs), leaving the behavior of large-scale generative LLMs unexplored.
- The impact of the number of human annotations on the estimation accuracy of ℍ_dis/ℍ_ent is not fully discussed.
- No alternative solutions are proposed—how to better estimate the intrinsic difficulty of instances remains an open question.

## Related Work & Insights

- Dataset Cartography by Swayamdipta et al. (2020) was the first to characterize instance difficulty using training dynamics; this work further validates its limitations.
- The application of Conformal Prediction in NLP is a relatively novel direction.
- Insights: When building evaluation benchmarks, one should not simply assume that annotator disagreement equals model uncertainty; more granular modeling of data complexity is required.

## Rating

- Novelty: ⭐⭐⭐ — More of an analytical work with no new method proposed, but with deep analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — A highly solid systematic comparison across 29 models, multiple datasets, and multiple metrics.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical symbols are clearly defined, and the argumentation logic is rigorous.
- Value: ⭐⭐⭐ — Provides guiding significance for evaluation and annotation practices, but offers no directly usable tools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MagicArticulate: Make Your 3D Models Articulation-Ready](../../CVPR2025/others/magicarticulate_make_your_3d_models_articulation-ready.md)
- [\[ACL 2025\] DeepRTL2: A Versatile Model for RTL-Related Tasks](deeprtl2_a_versatile_model_for_rtl-related_tasks.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](../../ICCV2025/others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)
- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)
- [\[ACL 2025\] EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)

</div>

<!-- RELATED:END -->
