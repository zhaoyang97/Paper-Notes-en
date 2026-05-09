---
title: >-
  [Paper Note] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps
description: >-
  [CVPR 2026][Multimodal VLM][Visual reasoning] This paper introduces the ReasonMap benchmark, constructed from high-resolution transit maps of 30 cities comprising 1,008 QA pairs, to systematically evaluate fine-grained visual understanding and spatial reasoning capabilities of 16 MLLMs. The work reveals the counter-intuitive phenomenon that base variants of open-source models consistently outperform their reasoning counterparts, and establishes a GRPO-based reinforcement fine-tuning training baseline.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Visual reasoning
  - transit maps
  - MLLM evaluation
  - spatial reasoning
  - reinforcement fine-tuning
date: 2026-05-08
content_hash: 3b94b41bfe476304
---

# ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps

**Conference**: CVPR 2026
**arXiv**: [2505.18675](https://arxiv.org/abs/2505.18675)
**Code**: [Project Page](https://fscdc.github.io/ReasonMap)
**Area**: Multimodal VLM
**Keywords**: Visual reasoning, transit maps, MLLM evaluation, spatial reasoning, reinforcement fine-tuning

## TL;DR

This paper introduces the ReasonMap benchmark, constructed from high-resolution transit maps of 30 cities comprising 1,008 QA pairs, to systematically evaluate fine-grained visual understanding and spatial reasoning capabilities of 16 MLLMs. The work reveals the counter-intuitive phenomenon that base variants of open-source models consistently outperform their reasoning counterparts, and establishes a GRPO-based reinforcement fine-tuning training baseline.

## Background & Motivation

Existing MLLM reasoning benchmarks exhibit notable blind spots:
- **Math/Logic benchmarks** (MathVQA, MMMU, MathVerse): visual understanding plays a limited role
- **Fine-grained visual benchmarks** (V*Bench, VisualPuzzles): require detailed perception but rarely involve spatial planning and reasoning
- **Spatial reasoning benchmarks** (CityBench, MapBench): relatively coarse-grained and often rely on external tools (map APIs) to bypass genuine visual reasoning

**Core Problem**: Benchmarks that simultaneously require fine-grained visual understanding (identifying station names, line colors/numbers) and spatial reasoning (planning transfer routes) remain absent.

Transit maps serve as an ideal test medium — they are information-dense, structured, require precise spatial interpretation, and are closely tied to real-world applications (navigation, urban planning).

## Method

### Overall Architecture

The ReasonMap construction pipeline consists of three stages:
1. **Data collection and preprocessing**: High-resolution transit maps from 30 cities across 13 countries are collected; line and station information is extracted via MLLM + human correction and standardized into JSON (Metro Data).
2. **QA pair construction**: Two stations are randomly selected from each map to generate short-form questions (fixed template) and long-form questions (two templates); reference routes are collected via Google Maps/Amap APIs.
3. **Quality control**: Correctness verification, diversity assurance, and difficulty balancing (map difficulty: easy/medium/hard, 10 maps each; question difficulty stratified by number of transfers).

### Key Designs

1. **Two-tier evaluation framework**:
   - **Correctness evaluation (Accuracy)**: Verifies consistency of departure/arrival stations, line names, and intermediate stops in the answer — all checks must pass for a response to be considered correct.
   - **Quality evaluation (Map Score)**: Assesses route quality even when the answer is not fully correct — 1 point for matching station names, 2 points for matching line names, intermediate stop count comparison or set IoU; the maximum score is capped per question type. Correct answers receive additional bonus points, ensuring correct responses always score higher than incorrect ones.

2. **Difficulty-aware weighting**: Evaluation metrics incorporate difficulty weighting, assigning greater weight to harder samples to prevent models from achieving inflated scores by solving only easy instances.

3. **GRPO reinforcement fine-tuning baseline**: Based on Qwen2.5-VL-3B/7B-Instruct, the paper designs an accuracy reward (binary signal from correctness evaluation) and a format reward (encouraging parseable outputs), with generalization validated under a cross-city setting.

### Loss & Training

- GRPO optimization: AdamW, initial learning rate $1.0 \times 10^{-6}$, KL divergence coefficient $1.0 \times 10^{-3}$
- 8 responses sampled per query, global batch size 16
- Training and test cities are completely non-overlapping (cross-city generalization validation)

## Key Experimental Results

### Main Results

Performance of 16 MLLMs on ReasonMap (weighted accuracy):

| Model | Type | Short Acc | Long Acc | Map Score (S/L) |
|-------|------|-----------|----------|-----------------|
| **OpenAI o3** | Closed-source Reasoning | **63.02%** | **59.11%** | **9.53/17.96** |
| Gemini-2.5-Flash | Closed-source Reasoning | 46.09% | 29.86% | 7.64/9.98 |
| Doubao-415 | Closed-source Reasoning | 43.14% | 46.09% | 7.33/14.67 |
| OpenAI 4o | Closed-source Base | 41.15% | 42.80% | 6.84/13.57 |
| **Qwen2.5-VL-72B** | Open-source Base | **26.65%** | **24.22%** | **5.09/8.80** |
| InternVL3-78B | Open-source Base | 25.35% | 19.62% | 4.80/7.50 |
| QvQ-72B-Preview | Open-source Reasoning | 9.03% | 4.25% | 1.59/1.55 |
| Skywork-R1V | Open-source Reasoning | 6.86% | 3.21% | 2.11/3.11 |

### Ablation Study

GRPO reinforcement fine-tuning effectiveness (cross-city generalization):

| Model | Short Acc | Long Acc | Map Score (S/L) |
|-------|-----------|----------|-----------------|
| Qwen2.5-VL-3B | 8.68% | 7.99% | 2.75/3.70 |
| +RL | **11.46%** (↑2.78) | **10.50%** (↑2.51) | 3.81/6.09 |
| Qwen2.5-VL-7B | 13.28% | 7.12% | 4.01/5.74 |
| +RL | **26.22%** (↑12.94) | **26.04%** (↑18.92) | 5.52/9.52 |

Visual masking experiment (text-only input):
- Most models show significant performance degradation (Qwen2.5-VL-72B: 26.65%→16.41%; Doubao-415: 43.14%→21.53%)
- Smaller models (Qwen2.5-VL-3B) show a slight improvement (8.68%→9.38%), suggesting greater reliance on prior knowledge rather than genuine visual reasoning

### Key Findings

- **Counter-intuitive phenomenon**: Among open-source models, base variants consistently outperform reasoning variants (e.g., Qwen2.5-VL-72B 26.65% vs. QvQ-72B 9.03%), whereas closed-source reasoning variants outperform their base counterparts (o3 63.02% vs. 4o 41.15%).
- **Root cause analysis**: Open-source reasoning models tend to introduce "visual confusion" during repeated self-verification — correctly identifying a route initially but overwriting it with an incorrect answer during self-reflection. Closed-source reasoning models possess stronger visual grounding, enabling self-correction within the reasoning chain.
- **Model scaling laws remain valid**: Larger models within the same family achieve higher accuracy with fewer tokens.
- The 7B model yields the largest gain after reinforcement fine-tuning (+18.92%), with a concurrent reduction in token usage.

## Highlights & Insights

- **Exposing MLLM blind spots**: This work is the first to systematically demonstrate the severe deficiencies of current MLLMs on spatial reasoning tasks that require genuine visual grounding.
- The **base vs. reasoning reversal** phenomenon provides an important clue for understanding the effect of RL fine-tuning on visual reasoning.
- **Refined evaluation framework design**: The two-tier evaluation separating correctness from quality, combined with difficulty-aware weighting, is more informative than simple answer comparison.
- **High-resolution challenge**: The average map resolution of 5839×5449 far exceeds typical VQA benchmarks, testing models' ability to process information-dense visual inputs.

## Limitations & Future Work

- The data scale is relatively limited (1,008 QA pairs / 30 cities); extending to more cities and transportation modes would enhance generalization evaluation.
- Only metro/light rail systems are evaluated, excluding buses, walking, and other multimodal transportation.
- Station name languages in certain cities may affect model OCR performance, though this has not been rigorously quantified.
- Even the strongest closed-source model (o3) achieves only 63% accuracy, indicating high task difficulty but also potentially suggesting ambiguity in some data instances.
- Reinforcement fine-tuning is validated only on 3B/7B models; the benefits for larger models remain unknown.

## Related Work & Insights

- **Comparison with MapBench/CityBench**: These benchmarks are coarser-grained or rely on external APIs; ReasonMap requires purely visual reasoning.
- **Comparison with MathVerse**: MathVerse reinforces visual dependency by generating multiple visual/textual variants; ReasonMap achieves this naturally through information-dense high-resolution maps.
- **RL fine-tuning trend**: The success of GRPO in textual reasoning is extending to multimodal reasoning; ReasonMap provides an effective training and evaluation scenario.
- **Inspiration**: The benchmark design methodology is transferable to domains such as architectural floor plan understanding and circuit diagram reasoning, which similarly require fine-grained visual perception combined with spatial reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Using transit maps as a visual reasoning testbed is a creative choice with a well-designed evaluation framework, though the benchmark construction methodology itself is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Systematic evaluation of 16 models + visual masking ablation + RL training baseline + detailed error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with clearly articulated findings and information-rich tables.
- **Value**: ⭐⭐⭐⭐ — Reveals critical shortcomings of MLLMs in fine-grained visual reasoning and provides the community with a valuable evaluation tool.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)
- [\[CVPR 2026\] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[CVPR 2026\] Zina: Multimodal Fine-grained Hallucination Detection and Editing](zina_multimodal_fine-grained_hallucination_detection_and_editing.md)
- [\[CVPR 2026\] EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)

<!-- RELATED:END -->
