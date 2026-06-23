---
title: >-
  [Paper Note] VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs
description: >-
  [ICLR 2026][LLM Reasoning][Paper Note] VisioMath is proposed as a benchmark containing 1800 K-12 mathematics problems where all options consist of highly visually similar charts. It reveals a core weakness of LMMs in multi-image-text alignment and explores three alignment strategies achieving a +12.6% gain.
tags:
  - ICLR 2026
  - LLM Reasoning
date: 2026-05-08
content_hash: a4707e38c1bece60
---
# VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs

**Conference**: ICLR 2026  
**arXiv**: [2506.06727](https://arxiv.org/abs/2506.06727)  
**Code**: [GitHub](https://github.com/Nefefilibata/VisioMath)  
**Area**: Multimodal VLM  
**Keywords**: Mathematical reasoning benchmark, Multi-image reasoning, Visual similarity, Image-text alignment, LMM evaluation

## TL;DR

VisioMath is proposed as a benchmark containing 1800 K-12 mathematics problems where all options consist of highly visually similar charts. It reveals a core weakness of LMMs in multi-image-text alignment and explores three alignment strategies achieving a +12.6% gain.

## Background & Motivation

Most existing multimodal mathematical reasoning benchmarks focus on single-image scenarios or text-based options, ignoring a significant and common problem type: **all answer options are charts**. These problems are prevalent in K-12 mathematics education, requiring fine-grained comparative reasoning of geometry, function curves, and other visually similar figures.

Existing multi-image benchmarks (e.g., MathVerse-mv, MV-Math) lack systematic consideration of **high visual similarity**. The core observation of VisioMath is that LMMs systematically fail to distinguish between nearly identical chart options. The primary failure mode is **image-text misalignment**, where models rely on positional heuristics rather than textual cues for reasoning.

## Method

### Overall Architecture

VisioMath formalizes a problem type ignored by previous benchmarks—where all options are charts and highly similar—into a multi-choice mathematical benchmark of 1800 questions. It performs large-scale LMM evaluation and explores alignment strategies to mitigate these shortcomings. The main technical line is "constructing a difficult benchmark targeting visual similarity → quantifying model degradation based on similarity → validating that bottlenecks can be partially resolved via alignment strategies."

### Key Designs

**1. Benchmark construction with options as figures: Controlling difficulty via visual similarity**

In standard math benchmarks, options are text or a single accompanying image, requiring only the understanding of the prompt. VisioMath requires models to identify the correct answer from 8,070 figure options, where figures often differ only by the slope of a curve or the position of a vertex. The authors selected 1800 such multiple-choice questions from Chinese high school and Gaokao exams (2002–2023), covering K-12 topics like geometry, algebraic visualization, numerical comparison, and function pattern recognition. Data is structurally stored in JSON with LaTeX for formulas, and manual cropping ensures a strict one-figure-per-option format with cross-verification to avoid noisy data. The difficulty stems not from the mathematical principles alone but from the controlled variable of "inter-option similarity."

**2. Quantifying visual similarity via minimum cosine similarity: Using similarity as a continuous axis**

To study the impact of similarity on reasoning, a scoring mechanism is required. Each option figure $x_i$ is encoded into a vector $f(x_i)$ via Qwen multimodal-embedding-v1. The similarity of a problem is defined as the minimum pairwise cosine similarity between options: $\text{Sim}(Q) = \min_{i \neq j} \cos(f(x_i), f(x_j))$. The minimum is used instead of the average because a single pair of highly similar distractors is sufficient to confuse the model. The benchmark retains the full similarity spectrum without truncation to avoid selection bias, and questions are divided into four quartiles (Q1–Q4) to observe accuracy degradation as similarity increases. Additionally, about half of the prompts contain images, adding the dimension of multi-source visual integration.

**3. Three alignment strategies: From training-free rewriting to CoT fine-tuning**

Since the primary failure mode is image-text misalignment (failing to map text labels to corresponding figures), the authors design three interventions. The first two are training-free: **Image Merging** combines multiple option figures into a single layout to reduce the burden of cross-image switching; **Explicit Visual-Text Anchors** add clear markers between each figure and its textual label to force image-text binding. The third is **Alignment-oriented CoT Fine-tuning**, where a multi-image Chain-of-Thought dataset is constructed for standard SFT. This encourages models to explicitly describe "which figure corresponds to which option and why," resulting in a +12.6% gain with minimal data. These strategies demonstrate that the bottleneck is addressable through explicit multi-image-text alignment mechanisms.

## Key Experimental Results

### Main Results

| Model | VisioMath Avg. | Prompt without Figure | Prompt with Figure |
|------|-------------|---------|---------|
| Human | 91.3 | 92.3 | 89.7 |
| Gemini 2.5 Pro | **80.9** | **86.3** | **75.2** |
| Seed1.6-Thinking | 72.3 | 83.9 | 58.0 |
| GPT-4.1 | 52.6 | 56.1 | 42.8 |
| GLM-4.5V (Best Open Source) | 53.7 | 61.2 | 37.2 |
| Qwen2.5-VL-72B | 43.7 | 49.8 | 33.0 |
| Vision-R1-7B | 36.7 | 33.7 | 29.2 |
| Random | 25.6 | - | - |

| Visual Similarity Quadrant | Q1 (Low) | Q2 | Q3 | Q4 (High) |
|---------------|---------|-----|-----|---------|
| Human | 95.7 | 91.2 | 87.6 | 89.0 |
| Gemini 2.5 Pro | 86.2 | 83.8 | 76.7 | 76.9 |
| GLM-4.5V | 68.7 | 59.3 | 44.2 | 44.7 |
| Qwen2.5-VL-7B | 33.6 | 37.8 | 29.8 | 29.6 |

### Ablation Study

| Strategy | Accuracy | Gain | Description |
|------|--------|------|------|
| Baseline | Baseline | - | Original reasoning |
| Shuffling | -8.7% (Gemini) | Significant Drop | Proves reliance on positional heuristics |
| Alignment-oriented CoT SFT | +12.6% | Max Gain | Effective with small CoT samples |

| Error Analysis (GLM4.5V, 50 samples) | Percentage | Description |
|---------------------------|------|------|
| Image-Text Misalignment | **36%** | Primary source of error |
| Other Reasoning Errors | 64% | Includes calculation, conceptual errors, etc. |

### Key Findings

- **Prompts with figures are harder**: Accuracy drops significantly for almost all LMMs when the prompt contains an image (Gemini drops 11.1%, GLM drops 24%), highlighting multi-source visual integration as a bottleneck.
- **Severe degradation with high similarity**: Model accuracy decreases by 12–15 percentage points from the lowest to the highest similarity quartiles.
- **Misalignment is the cause**: 36% of errors stem from image-text alignment failure; LMMs tend to use positional heuristics instead of semantic reasoning.
- **Human vs LMM Divergence**: Human accuracy stabilizes after a slight drop in high-similarity quartiles, indicating human errors are conceptual, while LMM errors are perceptual-alignment failures.
- The gap between the best open-source model GLM-4.5V (53.7%) and Humans (91.3%) is 37.6%, showing the task is far from solved.

## Highlights & Insights

- Fills the gap in evaluating math reasoning with chart-based options; first systematic study on the impact of visual similarity on multimodal reasoning.
- Shuffling experiments elegantly prove that LMMs rely on positional heuristics rather than true semantic alignment.
- The visual similarity quantification method (minimum cosine similarity + Qwen embeddings) is rigorously validated.
- The +12.6% gain from CoT fine-tuning suggests the problem can be partially mitigated through data-driven strategies.

## Limitations & Future Work

- Data sources are limited to Chinese high school/Gaokao (despite English translation), resulting in limited cultural and curriculum coverage.
- The benchmark scale of 1800 questions is moderate; samples in sub-fields may be insufficient.
- Alignment strategies are preliminary; more systematic architectural improvements remain to be researched.
- Covers only multiple-choice questions; open-ended chart reasoning is not included.

## Related Work & Insights

- Complementary to MathVista and MathVerse—VisioMath focuses on fine-grained distinction of multi-image options.
- Image-text misalignment may be prevalent in VLM tasks involving multiple images (e.g., document understanding, medical image comparison).
- Provides insights for LMM training: the need to strengthen explicit multi-image-text alignment capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ Fills a gap in chart-option reasoning evaluation; novel similarity quantification.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 20+ models (closed-source, open-source, math-specific) with sufficient error analysis and controls.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous structure; clear logic across observation, analysis, and strategy; intuitive tables.
- Value: ⭐⭐⭐⭐ Reveals core LMM weaknesses, though utility as a benchmark depends on community adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)
- [\[ICLR 2026\] LEXam: Benchmarking Legal Reasoning on 340 Law Exams](lexam_benchmarking_legal_reasoning_on_340_law_exams.md)
- [\[ICLR 2026\] USTBench: Benchmarking and Dissecting Spatiotemporal Reasoning Capabilities of LLMs as Urban Agents](ustbench_benchmarking_and_dissecting_spatiotemporal_reasoning_capabilities_of_ll.md)
- [\[ICLR 2026\] FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning](faithcot-bench_benchmarking_instance-level_faithfulness_of_chain-of-thought_reas.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)

</div>

<!-- RELATED:END -->
