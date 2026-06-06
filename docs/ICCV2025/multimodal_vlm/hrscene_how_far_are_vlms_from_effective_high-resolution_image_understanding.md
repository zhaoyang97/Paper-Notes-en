---
title: >-
  [Paper Note] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?
description: >-
  [ICCV 2025][Multimodal VLM][High-Resolution Image Understanding] This paper introduces HRScene, a benchmark covering 25 real-world scenarios and 2 diagnostic datasets (resolution 1K–35K). Evaluating 28 VLMs reveals that…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "High-Resolution Image Understanding"
  - "VLM Benchmark"
  - "Vision-Language Models"
  - "Multimodal Evaluation"
  - "Needle-in-a-Haystack"
date: 2026-05-08
content_hash: f2a86cfedc5e5331
---

# HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?

**Conference**: ICCV 2025  
**arXiv**: [2504.18406](https://arxiv.org/abs/2504.18406)  
**Code**: [Project Page](https://yszh8.github.io/hrscene/)  
**Area**: Multimodal VLM  
**Keywords**: High-Resolution Image Understanding, VLM Benchmark, Vision-Language Models, Multimodal Evaluation, Needle-in-a-Haystack

## TL;DR

This paper introduces HRScene, a benchmark covering 25 real-world scenarios and 2 diagnostic datasets (resolution 1K–35K). Evaluating 28 VLMs reveals that current state-of-the-art models achieve an average accuracy of only ~50% on real high-resolution tasks, with significant regional performance divergence and a pronounced lost-in-middle problem.

## Background & Motivation

High-resolution image (HRI) understanding is critical in domains such as pathology, autonomous driving, and document understanding. Although VLMs such as Gemini, Claude, and GPT claim to support high-resolution inputs, a serious evaluation gap persists:

**Lack of benchmarks**: Benchmarks used in mainstream VLM reports (MMMU, VQAv2, AI2D, etc.) have average resolutions below 1K, making them unsuitable for HRI evaluation.

**Narrow scenario coverage**: Existing HRI datasets focus on specific scenarios (e.g., long-range imagery) or specific resolutions (e.g., 8K).

**Insufficient diagnostics**: Existing multimodal NIAH tests primarily address long-context text or low-resolution multi-image settings, lacking diagnostics for regional utilization in HRI.

**Goal**: To construct a **unified, comprehensive, and practical** HRI benchmark that systematically evaluates VLMs' high-resolution understanding capabilities and identifies their core deficiencies.

## Method

### Overall Architecture

HRScene consists of two main components:
- **25 real-world scenario datasets**: Resolution 1K–35K, spanning 8 major categories.
- **2 synthetic diagnostic datasets**: Designed to precisely localize VLM deficiencies.

### Key Designs

1. **Taxonomy**:

    - **8 major categories**: Daily photos, urban planning, scanned documents, artwork, sub-images, remote sensing, medical diagnosis, and research understanding.
    - **25 specific scenarios**: Ranging from microscopy to radio telescopes, covering diverse camera types.
    - **Multiple capability tests**: Counting, temporal/semantic reasoning, holistic judgment, visual retrieval, spatial relationships, and small-object detection.
    - 6 datasets require domain expert knowledge; 19 belong to the general domain.

2. **Data Collection and Re-annotation**:

    - Data collected from 25 existing sources; 8 datasets re-annotated by 10 graduate-level annotators.
    - All images have resolution ≥ 1024×1024.
    - Distractor options constructed for 6 datasets (at least 4 options per sample).
    - Numerical answer options generated automatically via random offset.
    - An additional 750-sample human performance subset collected as an upper bound.

3. **WhiteBackground NIAH Diagnostic**:

    - VQAv2 images (needle) are placed at varying row/column positions in an N×N white grid (haystack).
    - Evaluates performance variation across spatial positions to detect **Regional Divergence**.
    - Grid sizes range from 1×1 to 10×10.

4. **ComplexGrid NIAH Diagnostic**:

    - Visually similar distractors retrieved via image retrieval tools are combined with the needle into a larger grid.
    - Models are required to identify the row and column of the needle.
    - Evaluates VLMs' ability to retrieve the correct image among multiple hard distractors.

### Dataset Scale and Splits

| Statistic | Count |
|-----------|-------|
| Total samples | 7,068 |
| Re-annotated | 2,005 |
| Annotated from scratch | 384 |
| val | 750 (= human-annotated) |
| testmini | 1,000 |
| test | 5,323 |

## Key Experimental Results

### Main Results (Tables)

**Overall Performance on Real-World Datasets**:

| Model | Art | Daily | Medical | Paper | Remote | Research | Sub-Img | Urban | Avg |
|-------|-----|-------|---------|-------|--------|----------|---------|-------|-----|
| Qwen2-VL 7B | 69.46 | 64.20 | 40.40 | 64.62 | 50.60 | 36.69 | 71.42 | 40.17 | 56.65 |
| InternVL2 40B | 74.35 | 62.67 | 38.10 | 70.89 | 44.16 | 43.15 | 74.10 | 44.40 | 58.45 |
| Qwen2-VL 72B | 75.85 | 66.20 | 43.69 | 78.13 | 52.48 | 39.36 | 74.89 | 44.66 | 61.85 |
| Gemini2.0 Flash | 76.46 | 62.27 | 51.94 | 75.12 | 47.59 | 34.85 | 68.62 | 44.54 | 59.82 |
| GPT-4o | 69.13 | 55.90 | 22.63 | 66.80 | 44.05 | 35.38 | 65.13 | 41.72 | 52.91 |
| **Human** | **75.33** | **77.75** | **23.81** | **88.75** | **58.33** | **48.50** | **90.00** | **55.25** | **64.72** |
| **28-model Avg** | 61.54 | 53.18 | 36.64 | 58.17 | 41.75 | 36.08 | 60.60 | 37.84 | **49.68** |

### Ablation Study (Tables)

**WhiteBackground NIAH Diagnostic — Regional Divergence Analysis**:

| Model | 1×1 Perf | 3×3 Perf | 3×3 Region↓ | 5×5 Perf | 5×5 Region↓ | 10×10 Perf | 10×10 Region↓ |
|-------|----------|----------|-------------|----------|-------------|------------|---------------|
| Qwen2-VL 7B | 85.93 | 84.22 | 5.30 | 83.14 | 6.52 | 79.91 | 10.56 |
| Qwen2-VL 72B | 84.13 | 84.51 | 5.62 | 84.04 | 6.62 | 84.56 | **9.61** |
| GPT-4o-mini | 68.66 | 60.69 | 13.77 | 52.53 | 19.59 | 32.94 | 33.65 |
| DeepSeek-VL2 | 72.06 | 49.71 | 15.75 | 34.29 | 23.37 | 23.95 | 23.30 |
| InternVL2 40B | 84.53 | 83.42 | 4.57 | 80.02 | 8.84 | 74.95 | 13.18 |

(Region metric = standard deviation of performance across spatial positions; lower is better.)

### Key Findings

- **Significant overall gap**: The average accuracy of 28 VLMs is only 49.68%; even the strongest model, Qwen2-VL 72B, reaches only 61.85%.
- **Large category disparities**: Medical (36.64%) and Research (36.08%) are the weakest categories, while Paper (58.17%) and Art (61.54%) perform relatively better.
- **Human vs. model**: Human average is 64.72%, but only 23.81% on Medical (requiring expert knowledge); humans reach 90% on Sub-Img while models average only 60%.
- **Regional Divergence**: As grid size increases, most models degrade substantially (e.g., GPT-4o-mini drops from 68.66% at 1×1 to 32.94% at 10×10), whereas Qwen2-VL 72B remains nearly unaffected.
- **Scale is not always decisive**: Qwen2-VL 7B outperforms many larger models (e.g., LLaVA-Next 34B) on most metrics.
- **Lost-in-middle phenomenon**: VLMs recognize images placed at central grid positions less accurately than those at peripheral positions.

## Highlights & Insights

- HRScene is the most comprehensive HRI benchmark to date: 25 scenarios, resolution spanning 4 orders of magnitude, covering both expert and general domains.
- The two diagnostic datasets are elegantly designed, quantitatively exposing two core VLM deficiencies: regional divergence and the lost-in-middle effect.
- Qwen2-VL 72B shows near-zero performance degradation with increasing resolution in the WhiteBackground NIAH (Region metric consistently below 10), suggesting a superior internal resolution processing strategy.
- Human performance on Medical is only 23.81%, yet the model average is 36.64%, indicating that models have already surpassed non-expert humans in certain specialist domains.

## Limitations & Future Work

- The benchmark is predominantly multiple-choice, offering limited evaluation of open-ended responses.
- Diagnostic datasets rely on white backgrounds and visually similar distractors; more naturalistic composite scenes could further challenge models.
- Some datasets have relatively small sample sizes (e.g., Galaxy and Grass), warranting stronger statistical validation.
- High-resolution video understanding (e.g., high-resolution video frame sequences) is not addressed.
- Test set answers are not publicly released and require online platform submission, which may hinder rapid iterative research.

## Related Work & Insights

- HRScene complements MME-Realworld and HR-Bench by providing broader scenario coverage.
- Extending NIAH testing from text/multi-image settings to single high-resolution images is a natural and important generalization.
- Qwen2-VL's tile-based processing strategy warrants further investigation, given its markedly superior resistance to resolution-induced degradation.
- Comparative analysis of VLM high-resolution processing architectures (dual-encoder vs. tiling strategies) offers useful guidance for model design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First large-scale unified HRI benchmark with cleverly designed diagnostic datasets.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 28 models, 27 datasets, human performance comparison, and diagnostic analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear taxonomy and precise articulation of identified issues.
- **Value**: ⭐⭐⭐⭐⭐ Provides a much-needed systematic evaluation tool for high-resolution VLM understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](../../ICLR2026/multimodal_vlm/vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers](falcon_resolving_visual_redundancy_and_fragmentation_in_high.md)
- [\[ICCV 2025\] DASH: Detection and Assessment of Systematic Hallucinations of VLMs](dash_detection_and_assessment_of_systematic_hallucinations_of_vlms.md)
- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](../../ICML2026/multimodal_vlm/cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)

</div>

<!-- RELATED:END -->
