---
title: >-
  [Paper Note] Ranked from Within: Ranking Large Multimodal Models Without Labels
description: >-
  [ICML 2025][Multimodal VLM][Unsupervised Model Ranking] This work systematically investigates whether the relative performance of LMMs can be predicted in label-free scenarios. By evaluating 47 SOTA LMMs across 9 VQA benchmarks, the study reveals that uncertainty metrics based on softmax distributions provide a robust unsupervised model ranking (Spearman correlation $\rho=0.92$ with the ground-truth ranking).
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Unsupervised Model Ranking"
  - "Uncertainty Estimation"
  - "VQA"
  - "Model Selection"
  - "LMM Evaluation"
date: 2026-05-08
content_hash: 8492a80c911eda0d
---

# Ranked from Within: Ranking Large Multimodal Models Without Labels

**Conference**: ICML 2025  
**arXiv**: [2412.06461](https://arxiv.org/abs/2412.06461)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Unsupervised Model Ranking, Uncertainty Estimation, VQA, Model Selection, LMM Evaluation

## TL;DR

This work systematically investigates whether the relative performance of LMMs can be predicted in label-free scenarios. By evaluating 47 SOTA LMMs across 9 VQA benchmarks, the study reveals that uncertainty metrics based on softmax distributions provide a robust unsupervised model ranking (Spearman correlation $\rho=0.92$ with the ground-truth ranking).

## Background & Motivation

### 1. The Challenge of LMM Model Selection

With the continuous emergence of numerous LMMs (e.g., LLaVA, InstructBLIP, Qwen-VL), how can users efficiently select the optimal model for new datasets or tasks? Traditional approaches are equivalent to "designing and grading an exam" — which requires labeled validation data.

### Goal

**Goal**: Annotating evaluation datasets for every new application scenario is highly resource-intensive. Many users (especially in industrial deployment) may only have access to unlabeled data. This calls for a method to "rank models without grading exams".

### 3. Instability of Cross-Benchmark Ranking

A model's ranking on one benchmark does not reliably predict its ranking on another benchmark (as verified by the paper's experiments) — meaning rankings from legacy benchmarks cannot be used to select models for new deployment scenarios.

### 4. Key Insight

When LMMs generate answers, each token is associated with a softmax probability distribution. Leveraging these probability signals allows for the estimation of model uncertainty — models with lower uncertainty typically perform better.

## Method

### Overall Architecture

1. Generate answers from $M$ LMMs on the target data (no labels required).
2. Extract uncertainty signals (softmax probabilities, self-consistency, etc.) from the generation process.
3. Calculate a surrogate ranking score $s_m$ for each model.
4. Measure the alignment between the surrogate scores and actual performance using Spearman/Kendall correlation coefficients.

### Key Designs

#### 1. Three Types of Unsupervised Ranking Signals

**Probability-based Ranking (Most Effective)**:
- Token-level probability: The maximum softmax value of each generated token.
- Sequence-level probability: The joint probability of the entire response sequence.
- Calculation: Averaging the token probabilities across all test samples.

**Self-consistency Ranking**:
- Answer the same question multiple times; higher consistency $\rightarrow$ higher model confidence.
- High cost (requires multiple inferences).

**Labeled Surrogate Set Ranking**:
- Use performance on a small labeled dataset as a surrogate.
- However, cross-domain transfer is highly unstable (a key finding).

#### 2. Evaluation Protocol

- 47 LMMs: Spanning different frameworks such as LLaVA, InstructBLIP, Qwen-VL, etc.
- Different vision encoders (CLIP, SigLIP) and language models (Vicuna, LLaMA).
- 9 Benchmarks: ScienceQA, MMMU, ChartQA, TextVQA, RealWorldQA, etc.
- Covering diverse tasks such as reasoning, OCR, and spatial understanding.

## Key Experimental Results

### Main Results: Comparison of Ranking Methods

| Ranking Method | Spearman $\rho$ (Mean) | Cross-Benchmark Stability | Computational Cost |
|---------|------------------|------------|--------|
| Benchmark A → Benchmark B (Cross-domain surrogate) | ~0.65 | Low | Requires Annotations |
| Self-consistency | ~0.78 | Medium | High |
| Sequence-level probability | ~0.88 | Medium-High | Low |
| **Token-level softmax probability** | **0.92** | **High** | **Low** |

### Cross-Benchmark Correlation Analysis

| Benchmark Pair | Performance Ranking Correlation $\rho$ | Description |
|--------|----------|------|
| ScienceQA ↔ MMMU | ~0.82 | Both are reasoning tasks, high correlation |
| ScienceQA ↔ TextVQA | ~0.51 | Reasoning vs OCR, low correlation |
| ChartQA ↔ RealWorldQA | ~0.43 | Different capability dimensions |
| Mean across all benchmarks | ~0.60 | **Predicting one benchmark from another is unreliable** |

### Key Findings

- Softmax probability-based ranking correlates highly with ground-truth performance across almost all benchmarks ($\rho>0.85$).
- Cross-benchmark ranking transfer is unstable—a model's superiority on one task does not guarantee dominance on another.
- Text prompt similarity is a better predictor of cross-dataset performance correlation than image feature similarity.
- Probability-based methods are effective for both closed-ended (multiple choice) and open-ended (free text) tasks, though slightly weaker on the latter.
- The self-consistency method is costly (requiring 5-10 samples) and unstable for smaller models.

## Highlights & Insights

- **Highly Practical**: Users only need to run a single forward pass on the target data to select models without any annotations.
- **Counter-intuitive Finding**: Selecting models for new scenarios based on legacy benchmark rankings is unreliable — challenging common practices in the community.
- **Uncertainty as a Quality Signal**: Models "know what they do not know," making softmax probability the optimal unsupervised surrogate.
- **Large-scale Validation**: Scaled verification across 47 models and 9 benchmarks ensures the statistical reliability of the conclusions.

## Limitations & Future Work

- Only VQA tasks were tested; applicability to more open-ended generation tasks (specifically dialogue and summarization) remains to be verified.
- Softmax probabilities can be affected by sampling parameters such as temperature/top-p, which were not ablated in the paper.
- Certain models (such as GPT-4V) do not allow access to logits, making the method inapplicable to closed-source API models.
- Hybrid ranking strategies that combine multiple signals (e.g., probability + self-consistency) could be explored in future work.

## Related Work & Insights

- **vs. Accuracy-on-the-Line (Miller et al. 2021)**: AoL assumes ID accuracy predicts OOD accuracy, but this work finds it unreliable in the context of LMMs.
- **vs. Uncertainty Estimation Methods (Kuhn et al. 2023)**: Semantic entropy is used for confidence estimation on individual predictions, whereas this work applies uncertainty to overall model ranking.
- **vs. LMM Benchmarking**: While traditional methods require annotations, this work provides an annotation-free alternative.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic study on label-free ranking of LMMs, with clear insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale validation across 47 models and 9 benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly defined problems and rigorous experimental design.
- Value: ⭐⭐⭐⭐⭐ Directly practical for guidance on LMM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Streamline Without Sacrifice — Squeeze out Computation Redundancy in LMM](streamline_without_sacrifice_--_squeeze_out_computation_redundancy_in_lmm.md)
- [\[ICML 2025\] Robust Multimodal Large Language Models Against Modality Conflict](robust_multimodal_large_language_models_against_modality_conflict.md)
- [\[CVPR 2025\] PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models](../../CVPR2025/multimodal_vlm/parc_a_quantitative_framework_uncovering_the_symmetries_within_vision_language_m.md)
- [\[CVPR 2026\] Towards Multimodal Domain Generalization with Few Labels](../../CVPR2026/multimodal_vlm/towards_multimodal_domain_generalization_with_few_labels.md)
- [\[NeurIPS 2025\] GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization](../../NeurIPS2025/multimodal_vlm/georanker_distance-aware_ranking_for_worldwide_image_geolocalization.md)

</div>

<!-- RELATED:END -->
