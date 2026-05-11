---
title: >-
  [Paper Note] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit
description: >-
  [AAAI 2026][Multimodal VLM][Vision-language models] This paper presents LLMC+, a comprehensive benchmark and plug-and-play toolkit for vision-language model (VLM) compression, supporting 20+ compression algorithms across 5 representative VLM families. It systematically investigates the independent and joint effects of token-level and model-level compression, revealing three key findings.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Vision-language models
  - model compression
  - token pruning
  - quantization
  - benchmarking
date: 2026-05-08
content_hash: e58d82f6fe8efbc8
---

# LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit

**Conference**: AAAI 2026
**arXiv**: [2508.09981](https://arxiv.org/abs/2508.09981)
**Code**: [GitHub](https://github.com/ModelTC/LightCompress)
**Area**: Multimodal VLM / Model Compression
**Keywords**: Vision-language models, model compression, token pruning, quantization, benchmarking

## TL;DR

This paper presents LLMC+, a comprehensive benchmark and plug-and-play toolkit for vision-language model (VLM) compression, supporting 20+ compression algorithms across 5 representative VLM families. It systematically investigates the independent and joint effects of token-level and model-level compression, revealing three key findings.

## Background & Motivation

**Background**: Large vision-language models (VLMs, e.g., LLaVA, InternVL, Qwen-VL) demonstrate strong multimodal understanding capabilities, yet their computational and memory demands are substantial—excessively long visual token sequences and massive parameter counts are the two primary bottlenecks. A variety of training-free compression methods have recently emerged, including token pruning (reducing the number of visual tokens) and model quantization (reducing parameter precision).

**Limitations of Prior Work**: (1) Existing methods do not decouple techniques into comparable modules, preventing fair comparison between spatial-redundancy and temporal-redundancy approaches; (2) evaluation is limited to simple single-turn tasks and fails to reflect real-world scenarios such as multi-turn dialogue; (3) compression techniques are applied in isolation, leaving the potential of joint compression unexplored.

**Key Challenge**: The absence of a unified evaluation framework leads to "apples-to-oranges" comparisons among compression methods, making reliable method selection guidance infeasible.

**Goal**: Construct a unified VLM compression benchmark that supports fair evaluation and systematic study.

**Key Insight**: Develop a modular toolkit that decouples various compression methods into composable modules.

**Core Idea**: Enable fair comparison and joint optimization of VLM compression methods through a unified benchmark and toolkit.

## Method

### Overall Architecture

LLMC+ consists of: (1) a **unified interface**—all compression methods are unified as composable modules; (2) **comprehensive evaluation**—covering single-turn/multi-turn, visual QA, reasoning, and detailed description tasks; and (3) **joint compression**—exploring combinations of token pruning and model quantization.

### Key Designs

1. **Modular Compression Framework**:

    - Function: Enables different compression methods to be compared and composed under a unified interface.
    - Mechanism: Compression methods are divided into two major categories—token-level compression (visual token pruning/merging) and model-level compression (weight quantization/pruning). Within each category, methods are decoupled into interchangeable modules, such as token importance estimation and token pruning strategy modules.
    - Design Motivation: Prior work employed disparate evaluation settings and datasets, rendering performance figures across papers incomparable. A unified framework eliminates evaluation bias.

2. **Multi-dimensional Evaluation Benchmark**:

    - Function: Comprehensively evaluates capability retention in compressed VLMs.
    - Mechanism: Evaluation covers single-turn visual QA, multi-turn dialogue, detail-sensitive tasks (e.g., OCR, fine-grained recognition), and visual reasoning. Multi-turn dialogue testing is specifically included—a dimension nearly absent from existing benchmarks yet critical for real-world applications.
    - Design Motivation: Single-turn VQA may mask information loss—a model can still guess the correct answer even after losing some visual detail, but such deficiencies are exposed in multi-turn dialogue requiring sustained understanding.

3. **Joint Compression Exploration**:

    - Function: Investigates the feasibility of dual token-level and model-level compression.
    - Mechanism: Token pruning is first applied to reduce sequence length, followed by model quantization to reduce parameter precision. Combinations of different compression levels are tested to identify configurations that minimize performance degradation under extreme compression.
    - Design Motivation: The gains from a single compression dimension are limited (e.g., 4-bit quantization is already near its ceiling); joint compression can simultaneously reduce costs along multiple dimensions.

### Loss & Training

All compression methods are training-free, requiring only a small amount of calibration data. Evaluation follows the standard VLM inference pipeline.

## Key Experimental Results

### Main Results

Covering 5 VLM families and 20+ compression algorithms.

| Finding | Details | Remarks |
|---------|---------|---------|
| Finding 1 | Spatial and temporal redundancy require different technical strategies | Token-level and model-level compression are mutually irreplaceable |
| Finding 2 | Token pruning degrades significantly on multi-turn dialogue and detail-sensitive tasks | An artifact of single-turn evaluation |
| Finding 3 | Joint token + model compression achieves extreme compression with minimal performance loss | A synergistic effect exceeding individual methods |

### Ablation Study

| Compression Method | Compression Ratio | Performance Retention | Remarks |
|-------------------|------------------|-----------------------|---------|
| Token pruning 50% only | Moderate | Good on single-turn / poor on multi-turn | Cumulative information loss |
| 4-bit quantization only | Moderate | Uniform degradation | Task-type independent |
| Token 50% + 4-bit quantization | Extreme | Comparable to single compression | Strong joint effect |

### Key Findings

- The degradation of token pruning methods in multi-turn dialogue is far more severe than anticipated, exposing a blind spot in existing evaluation practices.
- The key to effective joint compression lies in the two techniques targeting distinct redundancy dimensions—token pruning reduces spatial redundancy while quantization reduces precision redundancy, making the two nearly orthogonal.
- The toolkit's value lies in enabling fair comparisons, thereby revealing method strengths and weaknesses that were previously obscured in individual papers.

## Highlights & Insights

- **Systematic benchmark construction** offers long-term value to the VLM compression community—future work can directly build upon this benchmark for fair comparison.
- The inclusion of **multi-turn dialogue evaluation** is a significant contribution, exposing hidden risks associated with token pruning.
- **Findings on joint compression** carry direct practical guidance for deployment.

## Limitations & Future Work

- Five VLM families may not cover all architectural variants.
- The performance ceiling of training-free compression may be lower than that of quantization-aware training approaches.
- Extension to compression of video-understanding VLMs is a natural next step.

## Related Work & Insights

- **vs. LLM compression**: VLMs introduce an additional redundancy dimension—visual tokens—absent in LLMs, requiring compression strategies that account for this distinction.
- **vs. individual compression method papers**: LLMC+ does not propose a new method but rather provides a level playing field for all existing methods.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic benchmark and joint compression findings are novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 VLM families + 20+ algorithms + multi-dimensional evaluation
- Writing Quality: ⭐⭐⭐⭐ Findings are clearly summarized with insightful observations
- Value: ⭐⭐⭐⭐⭐ Infrastructure-level contribution to the VLM compression community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](global_compression_commander_plug-and-play_inference_acceler.md)
- [\[CVPR 2026\] Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness](../../CVPR2026/multimodal_vlm/seeing_clearly_reasoning_confidently_plug-and-play_remedies_for_vision_language_.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[CVPR 2026\] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](../../CVPR2026/multimodal_vlm/prune2drive_a_plug-and-play_framework_for_accelerating_vision-language_models_in.md)
- [\[AAAI 2026\] Towards Long-window Anchoring in Vision-Language Model Distillation](towards_long-window_anchoring_in_vision-language_model_distillation.md)

</div>

<!-- RELATED:END -->
