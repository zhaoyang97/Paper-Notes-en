---
title: >-
  [Paper Note] VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs
description: >-
  [ICLR 2026][Multimodal VLM][mathematical reasoning benchmark] This paper introduces VisioMath, a benchmark comprising 1,800 K-12 mathematics problems in which all answer choices consist of highly visually similar figures…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "mathematical reasoning benchmark"
  - "multi-image reasoning"
  - "visual similarity"
  - "image-text alignment"
  - "LMM evaluation"
date: 2026-05-08
content_hash: 62c0594ad5bc387f
---

# VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs

**Conference**: ICLR 2026
**arXiv**: [2506.06727](https://arxiv.org/abs/2506.06727)  
**Code**: [GitHub](https://github.com/Nefefilibata/VisioMath)  
**Area**: Multimodal VLM
**Keywords**: mathematical reasoning benchmark, multi-image reasoning, visual similarity, image-text alignment, LMM evaluation

## TL;DR

This paper introduces VisioMath, a benchmark comprising 1,800 K-12 mathematics problems in which all answer choices consist of highly visually similar figures. It reveals a core weakness of LMMs in multi-image–text alignment, and explores three alignment strategies that achieve up to +12.6% accuracy improvement.

## Background & Motivation

Most existing multimodal mathematical reasoning benchmarks focus on single-image settings or text-based answer options, overlooking an important and common problem type: questions in which **all answer choices are figures**. Such problems are prevalent in K-12 mathematics education and require fine-grained comparative reasoning over visually near-identical geometric diagrams, function curves, and similar stimuli.

Existing multi-image benchmarks (e.g., MathVerse-mv, MV-Math) lack systematic consideration of **high visual similarity**. The central observation of VisioMath is that LMMs systematically fail to discriminate among nearly identical figure-based options, with the dominant failure mode being **image-text misalignment**—models rely on positional heuristics rather than textual cues when reasoning.

## Method

### Overall Architecture

VisioMath = a carefully constructed benchmark of 1,800 multiple-choice mathematics problems + comprehensive evaluation + exploration of alignment strategies.

### Key Designs

1. **Benchmark Construction**: 1,800 multiple-choice problems are collected from authentic Chinese high school and college entrance examination papers spanning 2002–2023, containing 8,070 figure-based answer options. Three core design principles:

    - **Representativeness**: Real examination problems covering K-12 topics including geometry, algebraic visualization, numerical comparison, and function pattern recognition.
    - **Reliability**: JSON standardization, LaTeX mathematical notation, manually cropped images (strictly one image per option), and human cross-validation.
    - **High Visual Similarity**: Qwen multimodal-embedding-v1 is used to compute the minimum pairwise cosine similarity among options, $\text{Sim}(Q) = \min_{i \neq j} \cos(f(x_i), f(x_j))$; the full similarity spectrum is retained to avoid selection bias.

2. **Visual Similarity Quantification**: Problems are stratified into four quartile bins (Q1–Q4) by visual similarity, enabling systematic analysis of LMM performance as similarity increases. Approximately 50% of problems also contain figures in the question stem, further increasing the complexity of visual reasoning.

3. **Three Alignment Strategies**:

    - **Image Concatenation** (training-free): Multiple option images are merged into a single composite layout.
    - **Explicit Visual-Text Anchoring** (training-free): Explicit correspondence markers are established between figure regions and their textual option labels.
    - **Alignment-Oriented CoT Fine-tuning**: A multi-image chain-of-thought dataset is constructed for supervised fine-tuning; even a small amount of data yields a +12.6% improvement.

### Loss & Training

This work is primarily a benchmark evaluation study. The alignment-oriented CoT fine-tuning employs standard SFT on a small VisioMath-CoT dataset.

## Key Experimental Results

### Main Results

| Model | VisioMath Avg. | No-figure Stem | Figure Stem |
|-------|---------------|----------------|-------------|
| Human | 91.3 | 92.3 | 89.7 |
| Gemini 2.5 Pro | **80.9** | **86.3** | **75.2** |
| Seed1.6-Thinking | 72.3 | 83.9 | 58.0 |
| GPT-4.1 | 52.6 | 56.1 | 42.8 |
| GLM-4.5V (best open-source) | 53.7 | 61.2 | 37.2 |
| Qwen2.5-VL-72B | 43.7 | 49.8 | 33.0 |
| Vision-R1-7B | 36.7 | 33.7 | 29.2 |
| Random | 25.6 | — | — |

| Visual Similarity Quartile | Q1 (Low) | Q2 | Q3 | Q4 (High) |
|---------------------------|----------|-----|-----|-----------|
| Human | 95.7 | 91.2 | 87.6 | 89.0 |
| Gemini 2.5 Pro | 86.2 | 83.8 | 76.7 | 76.9 |
| GLM-4.5V | 68.7 | 59.3 | 44.2 | 44.7 |
| Qwen2.5-VL-7B | 33.6 | 37.8 | 29.8 | 29.6 |

### Ablation Study

| Strategy | Accuracy | Gain | Notes |
|----------|----------|------|-------|
| Baseline (no strategy) | baseline | — | Standard inference |
| Option Shuffling | −8.7% (Gemini) | significant drop | Confirms reliance on positional heuristics |
| Alignment-Oriented CoT Fine-tuning | +12.6% | largest gain | Effective with only a small CoT dataset |

| Error Analysis (GLM-4.5V, 50 samples) | Proportion | Notes |
|---------------------------------------|-----------|-------|
| Image-text misalignment | **36%** | Primary error source |
| Other reasoning errors | 64% | Includes computation and concept errors |

### Key Findings

- **Figure-stem problems are harder**: Nearly all LMMs show a significant accuracy drop on problems whose stems contain figures (Gemini −11.1%, GLM −24%), indicating that integrating multi-source visual information is a critical bottleneck.
- **High similarity causes severe degradation**: Model accuracy declines by 12–15 percentage points from the lowest to the highest similarity quartile.
- **Image-text misalignment is the primary cause**: 36% of errors stem from alignment failure; LMMs tend to apply positional heuristics instead of semantic reasoning.
- **Human vs. LMM divergence**: Human accuracy declines only slightly at high similarity before stabilizing, suggesting that human errors stem primarily from conceptual misunderstanding, whereas LMM errors arise from perception-alignment failure.
- The best open-source model, GLM-4.5V (53.7%), lags behind humans (91.3%) by 37.6 percentage points, indicating that this task is far from solved.

## Highlights & Insights

- Fills the gap in evaluation of figure-based-option mathematical reasoning, representing the first systematic study of how visual similarity affects multimodal reasoning.
- The option-shuffling experiment elegantly demonstrates that LMMs rely on positional heuristics rather than genuine semantic alignment.
- The visual similarity quantification methodology (minimum cosine similarity + Qwen embeddings) is rigorously validated.
- The +12.6% gain from CoT fine-tuning indicates that the problem can be partially mitigated through data-centric strategies.

## Limitations & Future Work

- Problem sources are limited to Chinese college entrance and high school examinations (with English translations provided), resulting in limited cultural and curricular coverage.
- The benchmark scale of 1,800 problems is relatively modest, and sub-domain sample sizes may be insufficient.
- The alignment strategies explored are preliminary; more systematic architecture-level improvements remain to be investigated.
- Only multiple-choice problems are covered; open-ended figure-based reasoning is not addressed.

## Related Work & Insights

- VisioMath is complementary to MathVista, MathVerse, and similar benchmarks, focusing specifically on fine-grained discrimination among multiple figure-based options.
- Image-text misalignment may be a pervasive issue in any VLM task requiring the processing of multiple images (e.g., document understanding, medical image comparison).
- The findings provide guidance for LMM training: explicit multi-image–text alignment capabilities need to be strengthened.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Fills the gap in figure-based-option reasoning evaluation; visual similarity quantification is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 20+ models (closed-source, open-source, and math-specialized); error analysis and controlled experiments are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorously structured; the observation–analysis–strategy logic is clear, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Reveals a core weakness of LMMs; practical impact as a benchmark paper depends on community adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](../../ACL2026/multimodal_vlm/errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](../../ACL2026/multimodal_vlm/a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)

</div>

<!-- RELATED:END -->
