---
title: >-
  [Paper Note] UNICBench: UNIfied Counting Benchmark for MLLM
description: >-
  [CVPR 2026][Multimodal VLM][counting benchmark] This paper introduces UNICBench, the first unified cross-modal (image/text/audio) multi-level counting benchmark, comprising 5,508 + 5,888 + 2,905 = 14,301 QA pairs organized along a three-level capability taxonomy (Pattern/Semantic/Reasoning) × three-level difficulty taxonomy (Easy/Medium/Hard). The benchmark systematically evaluates 45 state-of-the-art MLLMs, revealing that basic counting tasks are near saturation while reasoning-level and hard-difficulty tasks exhibit substantial performance gaps.
tags:
  - CVPR 2026
  - Multimodal VLM
  - counting benchmark
  - multimodal LLM
  - image-text-audio
  - unified evaluation
  - stratified difficulty
date: 2026-05-08
content_hash: ab6fc9718a71e450
---

# UNICBench: UNIfied Counting Benchmark for MLLM

**Conference**: CVPR 2026
**arXiv**: [2603.00595](https://arxiv.org/abs/2603.00595)
**Code**: Public evaluation toolkit
**Area**: Multimodal Benchmarking / MLLM Evaluation
**Keywords**: counting benchmark, multimodal LLM, image-text-audio, unified evaluation, stratified difficulty

## TL;DR

This paper introduces UNICBench, the first unified cross-modal (image/text/audio) multi-level counting benchmark, comprising 5,508 + 5,888 + 2,905 = 14,301 QA pairs organized along a three-level capability taxonomy (Pattern/Semantic/Reasoning) × three-level difficulty taxonomy (Easy/Medium/Hard). The benchmark systematically evaluates 45 state-of-the-art MLLMs, revealing that basic counting tasks are near saturation while reasoning-level and hard-difficulty tasks exhibit substantial performance gaps.

## Background & Motivation

**Background**: Counting is a core cognitive capability of multimodal large models, closely related to numerosity sense—a fundamental cognitive faculty shared by humans and animals. While MLLMs have advanced rapidly on general VQA and reasoning benchmarks, no benchmark has systematically evaluated counting as an isolated capability across modalities.

**Limitations of Prior Work**: (1) Annotation formats for image counting datasets are inconsistent (points/bounding boxes/density maps), making them difficult to directly repurpose for MLLM QA evaluation. (2) Text and audio counting data are extremely scarce—QA datasets for document deduplication counting and audio event counting are virtually nonexistent. (3) Evaluation protocols are inconsistent across works, with different splits, prompts, random seeds, and matching rules rendering results incomparable. (4) Closed-source model APIs incur high costs and are rate-limited, hampering fair cross-model comparison.

**Key Challenge**: Counting ability spans three distinct levels—perceptual localization, semantic filtering, and rule-based reasoning. Existing benchmarks either cover only a single modality or fail to distinguish capability levels, making it impossible to systematically identify MLLM counting bottlenecks.

**Goal**: To establish a counting benchmark that covers three modalities (image/text/audio), employs a unified QA format and evaluation protocol, and enables stratified diagnosis of capability deficiencies.

**Key Insight**: Design a cross-classified taxonomy of three capability levels (Pattern/Semantic/Reasoning) and three difficulty levels (Easy/Medium/Hard), paired with evidence-first ground truth and deterministic numeric parsing.

**Core Idea**: Decompose counting ability into three levels—perceptual counting → semantic filtering → rule-based reasoning—and evaluate uniformly across image/text/audio modalities, using metrics such as MAE and HitRate to stratify the diagnosis of MLLM counting bottlenecks.

## Method

### Overall Architecture

UNICBench comprises QA corpora across three modalities, a unified QA-evidence schema, a standardized evaluation protocol (fixed split/prompt/seed with modality-specific matching rules), and a stratified reporting framework that cross-tabulates results by capability × difficulty × modality.

### Key Designs

1. **Three-Level Capability × Three-Level Difficulty Taxonomy**
   - Pattern (L1): Direct perceptual counting, $y=|E|$, e.g., "How many people are in the image?"
   - Semantic (L2): Attribute filtering / deduplication, $y=|\{e \in E \mid P(e)\}|$, e.g., "How many people are wearing red?"
   - Reasoning (L3): Rule-driven / compositional counting, $y=g(|S_1|,\ldots)$, e.g., "How many folders were modified in 2022?"
   - Difficulty is mapped via objective metrics (density/occlusion/repetition rate) to Easy (1–10) / Medium (11–100) / Hard (>100).
   - **Design Motivation**: The cross-classification enables precise diagnosis—distinguishing "poor perception vs. poor reasoning" and "failure on simple vs. dense scenes."

2. **Evidence-First Ground Truth and Cross-Modal Unified Schema**
   - Each GT entry includes `gt_count` and structured `gt_evidence` (images: instance coordinates; text: character spans; audio: timestamps).
   - Question templates: L1 uses deterministic templates to reduce linguistic variation; L2/L3 use free-form templates with explicit filtering rules.
   - Multi-stage quality control: dual independent annotation + arbitration, with 100% annotation consistency.
   - **Design Motivation**: Evidence traceability ensures GT verifiability; the unified schema makes cross-modal comparisons meaningful.

3. **Standardized Evaluation Protocol**
   - Fixed split/prompt/seed eliminates stochasticity.
   - Modality-specific matching rules (exact numeric match vs. ε-tolerance).
   - Deterministic numeric parsing to extract numbers from natural-language responses.
   - Evaluation metrics: MAE, MSE, SuccessRate, HitRate (@100%/@90%/@80%).

### Loss & Training

UNICBench is an evaluation benchmark and does not involve model training. Metric definitions:
- MAE $= \frac{1}{N}\sum|y_i - \hat{y}_i|$, MSE $= \frac{1}{N}\sum(y_i - \hat{y}_i)^2$
- HitRate@X% = accuracy within an X% error tolerance
- SuccessRate = proportion of responses from which a parseable number is successfully extracted

## Key Experimental Results

### Main Results (Image Modality, Top-10 Models)

| Model | Overall MAE↓ | Easy MAE↓ | Hard MAE↓ | Pattern MAE↓ | Reasoning MAE↓ |
|---|---|---|---|---|---|
| GPT-5-mini | 29.8 | 2.1 | 155.0 | 25.4 | 5.3 |
| o4-mini | 42.9 | 2.2 | 239.1 | 39.1 | 4.1 |
| GPT-4o | 43.2 | 2.4 | 238.4 | 41.7 | 5.4 |
| GPT-o3 | 49.0 | 2.8 | 277.1 | 44.3 | 4.4 |
| GPT-5 | 54.1 | 2.5 | 312.4 | 55.1 | 5.9 |
| Claude-Sonnet-4 | 78.1 | 5.4 | 444.6 | 68.8 | 4.4 |
| Gemini-2.5-Pro | 90.0 | 4.3 | 504.9 | 71.1 | 4.6 |
| Gemini-2.5-Flash | 140.5 | 12.0 | 694.2 | 131.4 | 6.7 |
| GLM-4.1V-9B | 97.9 | 3.0 | 542.2 | 90.0 | 3.1 |
| GPT-4o-mini | 73.3 | 2.3 | 424.6 | 72.7 | 5.3 |

### Cross-Modal / Cross-Difficulty Analysis

| Dimension | Finding |
|---|---|
| Easy vs. Hard | Easy MAE: 2–5; Hard MAE: 100–700; gap exceeds 100× |
| Pattern vs. Reasoning | Image Reasoning MAE is low (3–7) but comprises only 4.6% of samples; high Pattern MAE originates from high-density scenes |
| Text modality | Reasoning accounts for the highest proportion (43.7%); models generally underperform on deduplication and cross-passage aggregation |
| Audio modality | Environmental sound events have low density (1.56/sample); meeting speech has extremely high density (81.51/sample) |
| Long-tail distribution | GT count distribution is heavily right-skewed; model error explodes in high-count regions |

### Key Findings

- Simple counting tasks (L1 + Easy) are near saturation across models, with Easy MAE gaps of only 2–12.
- The Hard partition exhibits large gaps—the best model (GPT-5-mini: 155) outperforms the worst (Gemini-2.5-Flash: 694) by 4.5×.
- Reasoning tasks in the text modality (deduplicated citations, cross-passage statistics) represent the most significant weakness of current MLLMs.
- Open-source models perform surprisingly well on Reasoning (GLM-4.1V MAE: 3.1), but lag considerably on Pattern.

## Highlights & Insights

- The first unified counting benchmark spanning three modalities—treating "counting" as a core cognitive capability evaluated in isolation, filling a notable gap.
- The three-level capability × three-level difficulty cross-classification enables precise diagnosis, pinpointing "which capability level fails at which difficulty."
- Evidence-first GT design ensures every answer is traceable and verifiable.
- Long-tail distribution analysis reveals systematic model failure in high-count scenarios—not random error, but a cognitive blind spot.
- The evaluation of 45 models provides broad coverage, lending statistical credibility to the conclusions.

## Limitations & Future Work

- Audio counting data is relatively sparse (2,069 samples vs. 5,300 for images), limiting the robustness of audio-dimension conclusions.
- High evaluation costs for closed-source APIs (GPT-5-level) restrict reproducibility and extensibility.
- Cross-modal joint counting (e.g., simultaneous visual and audio counting in video) is not addressed.
- Image-modality Reasoning accounts for only 4.6% of samples, yielding a limited sample size for conclusions at that level.
- The impact of enhancement strategies such as few-shot prompting and chain-of-thought on counting performance is not explored.

## Related Work & Insights

- **vs. MMBench/MMMU**: General benchmarks do not systematically evaluate counting; UNICBench fills the gap with deep evaluation of this specific capability.
- **vs. FSC-147/ShanghaiTech**: Traditional counting datasets use density maps and point annotations; UNICBench adopts a unified QA format tailored for MLLMs.
- **vs. DocVQA/ChartQA**: These benchmarks involve counting but do not treat it as a core capability; UNICBench focuses on counting and provides stratified diagnosis.
- The stratified evaluation paradigm (capability × difficulty × modality) is generalizable to benchmark design for other specific capabilities (e.g., spatial reasoning, temporal understanding).
- Systematic failure under long-tail distributions suggests that MLLMs may lack genuine "counting" ability, relying more on pattern matching than true enumeration.

## Rating

- Novelty: ⭐⭐⭐⭐ First unified cross-modal counting benchmark with a well-designed taxonomy
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 45 models with three-dimensional cross-analysis
- Writing Quality: ⭐⭐⭐⭐ Clear taxonomy presentation and rich visualizations
- Value: ⭐⭐⭐⭐ Reveals systematic deficiencies in MLLM counting ability; benchmark has long-term utility

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](../../ICLR2026/multimodal_vlm/bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)
- [\[CVPR 2026\] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)
- [\[CVPR 2026\] Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[CVPR 2026\] VecGlypher: Unified Vector Glyph Generation with Language Models](vecglypher_unified_vector_glyph_generation_with_language_models.md)

<!-- RELATED:END -->
