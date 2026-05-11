---
title: >-
  [Paper Note] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps
description: >-
  [CVPR2026][Multimodal VLM][Multimodal reasoning] This paper introduces the ReasonMap benchmark, which constructs 1,008 QA pairs from high-resolution transit maps of 30 cities and proposes a two-level evaluation framework…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Multimodal reasoning"
  - "visual reasoning"
  - "spatial reasoning"
  - "metro maps"
  - "benchmark"
  - "reinforcement fine-tuning"
  - "GRPO"
date: 2026-05-08
content_hash: 99ec5ccfbaf93de1
---

# ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps

**Conference**: CVPR2026
**arXiv**: [2505.18675](https://arxiv.org/abs/2505.18675)
**Code**: [fscdc/ReasonMap](https://fscdc.github.io/ReasonMap)
**Area**: Multimodal VLM
**Keywords**: Multimodal reasoning, visual reasoning, spatial reasoning, metro maps, benchmark, reinforcement fine-tuning, GRPO

## TL;DR

This paper introduces the ReasonMap benchmark, which constructs 1,008 QA pairs from high-resolution transit maps of 30 cities and proposes a two-level evaluation framework (correctness + quality) to systematically assess fine-grained visual reasoning capabilities of 16 MLLMs. A key finding is that among open-source models, base models outperform reasoning models, while the opposite holds for closed-source models.

## Background & Motivation

**Insufficient visual reasoning evaluation for MLLMs**: Existing multimodal reasoning benchmarks (MathVQA, MMMU, MathVerse) primarily assess symbolic/mathematical reasoning with limited emphasis on visual understanding, lacking joint evaluation of fine-grained visual comprehension and spatial reasoning.

**Coarse granularity of existing benchmarks**: VisuLogic and VisualPuzzles focus on fine-grained perception but do not involve spatial planning; CityBench and MapBench address spatial reasoning but lack fine granularity and rely on external tools (map APIs) to complete tasks, thereby circumventing genuine visual reasoning.

**Maps as ideal evaluation vehicles**: Transit maps, as structured and information-dense visual artifacts, inherently demand precise spatial interpretation, making them well-suited for evaluating fine-grained visual reasoning.

**Questionable performance of reasoning models**: While reasoning-oriented MLLMs excel at mathematical and logical tasks, whether they are equally effective on spatial reasoning tasks requiring visual grounding has not been systematically validated.

**Visual dependency vs. linguistic priors**: Prior work has noted that MLLMs may rely on internal knowledge priors rather than genuinely attending to visual inputs, necessitating validation through visual-masking experiments.

**Lack of training baselines**: The absence of RL training baselines for fine-grained visual reasoning scenarios hinders subsequent research comparisons and exploration.

## Method

### Overall Architecture

The ReasonMap construction pipeline consists of three stages:

**Stage 1: Data Collection and Preprocessing**

- High-resolution metro maps from 30 cities (13 countries) are collected from public sources.
- Average resolution is 5,839×5,449, far exceeding existing visual reasoning datasets (typically <1,000×1,000).
- GPT-4o is used to extract line and station names, which are manually corrected and stored in a unified JSON format (Metro Data).
- Special cases (transfer stations, branch line terminals) are annotated separately in a standardized format.

**Stage 2: QA Pair Construction**

- Two stations are randomly selected and questions are generated in both short and long forms using predefined templates.
- Short questions use one fixed template; long questions are randomly drawn from two templates (one asking for the number of intermediate stops, one requiring enumeration of specific stops).
- Reference routes are obtained via the Amap API (Chinese cities) or Google Maps API (other cities).
- Question difficulty is divided by the number of transfers (0=easy, 1=medium, ≥2=hard).
- Map difficulty is divided by the number of lines and transfer stations (easy/medium/hard, 10 maps each).
- Each map has a fixed quota of 20:15:5 (easy:medium:hard), yielding 40 questions per map.

**Stage 3: Quality Control**

- QA pairs are inspected for correctness, diversity, and difficulty balance.
- Erroneous QA pairs are manually corrected or discarded.
- Routes that cannot be visually traced on the map are discarded to ensure consistency with visual content.

### Two-Level Evaluation Framework

- **Accuracy Evaluation**: Verifies correctness of departure/arrival stations → existence of each route segment name → validity of departure/arrival stations in each segment → consistency of transfer stations between adjacent segments; all checks must pass for a response to be deemed correct.
- **Quality Evaluation (Map Score)**:
    - **Short questions**: Compares answer and reference route segment pairs, awarding points for stop1/stop2 match (1 pt), route name match (2 pts), and departure/arrival stations within a segment (1 pt each), capped at 10 pts, with bonus points for correct answers.
    - **Long questions**: Extends short-question scoring with evaluation of the number of intermediate stops (num_via_stop_score, mapping absolute error to a 4-point scale) or specific intermediate stops (via_stop_score, averaging IoU and exact match then truncating to 10 pts).
- **Difficulty weighting**: Higher-difficulty samples are assigned greater weights to more accurately reflect model robustness.

### GRPO Training Baseline

- Reinforcement fine-tuning is applied to Qwen2.5-VL-3B/7B-Instruct using GRPO (Group Relative Policy Optimization).
- **Reward design**: (1) accuracy reward — binary signal based on the correctness evaluation; (2) format reward — encourages parseable output format.
- Training configuration: AdamW, lr=1e-6, KL coefficient 1e-3, 8 responses sampled per query, global batch size 16.
- Cross-city splits (training and test cities are completely disjoint) are adopted to validate generalization.

## Experiments

### Main Results

| Model | Type | Short Weighted Acc | Long Weighted Acc | Map Score (S/L) |
|---|---|---|---|---|
| Qwen2.5-VL-72B | Base | 26.65% | 24.22% | 5.09 / 8.80 |
| InternVL3-78B | Base | 25.35% | 19.62% | 4.80 / 7.50 |
| QvQ-72B-Preview | Reasoning | 9.03% | 4.25% | 1.59 / 1.55 |
| Kimi-VL-A3B-Thinking | Reasoning | 5.47% | 5.47% | 2.44 / 3.17 |
| OpenAI o3 | Reasoning | **63.02%** | **59.11%** | **9.53 / 17.96** |
| OpenAI 4o | Base | 41.15% | 42.80% | 6.84 / 13.57 |
| Gemini-2.5-Flash | Reasoning | 46.09% | 29.86% | 7.64 / 9.98 |

### Ablation Study on RL Training Baseline

| Model | Short Acc Gain | Long Acc Gain | Map Score Gain (S/L) |
|---|---|---|---|
| Qwen2.5-VL-3B + RL | +2.78% | +2.51% | +1.06 / +2.39 |
| Qwen2.5-VL-7B + RL | +12.94% | +18.92% | +1.51 / +3.78 |

### Key Findings

1. **Open-source base > reasoning; closed-source reasoning > base**: Open-source reasoning models introduce visual confusion during their chain-of-thought (initially correct answers are subsequently self-negated), whereas closed-source reasoning models possess stronger visual grounding and can self-correct visual confusion within the reasoning chain.
2. **Scaling laws still hold**: Larger models within the same family achieve higher accuracy with fewer tokens (Qwen2.5-VL-72B short-question Acc of 26.65% vs. 8.68% for the 3B variant).
3. **Visual-masking experiments**: Removing visual input degrades performance for most models, with more pronounced degradation for closed-source models (Doubao-415 short-question Acc drops by 21.61%), indicating effective utilization of visual information; Qwen2.5-VL-3B shows minimal change or even slight improvement, suggesting smaller models rely more heavily on linguistic priors.
4. **RL fine-tuning is consistently effective**: Under the cross-city setting, the 7B model's short-question Acc improves from 13.28% to 26.22% and long-question Acc from 7.12% to 26.04%, with concurrent reduction in token usage.
5. **Error type analysis**: Primary errors include visual confusion (misidentification of similarly colored lines), format errors, hallucinations (repetition of correct answers or generation of irrelevant content), and refusals. Multiple error types can co-occur within a single response.
6. **Large inter-city variation**: Even for maps of comparable difficulty, model performance varies substantially across cities, closely correlated with city prominence and the language of station names.

## Highlights & Insights

- The first high-resolution map benchmark targeting fine-grained visual reasoning, with resolutions far exceeding existing datasets (5,839×5,449 vs. typically <1,000×1,000).
- The two-level evaluation framework (correctness + quality) is elegantly designed; Map Score differentiates models more effectively than simple accuracy.
- Reveals counterintuitive performance discrepancies between open-source/closed-source base/reasoning models, with plausible explanations supported by case analysis.
- A semi-automated and scalable data construction pipeline that facilitates future expansion to additional cities.
- Visual-masking experiments validate the necessity of visual grounding.

## Limitations & Future Work

- Limited data scale (1,008 QA pairs, 30 cities) with restricted city coverage and linguistic diversity.
- Restricted to metro/transit maps; more complex map types (e.g., road networks, floor plans) are not addressed.
- Reference routes depend on Google Maps/Amap APIs, which may introduce coverage biases.
- Evaluation relies on strict format parsing; directly penalizing format errors may underestimate the true reasoning ability of certain models.
- RL training baselines are validated only on Qwen2.5-VL, without coverage of additional architectures.

## Related Work & Insights

- **Multimodal reasoning benchmarks**: MMMU, MathVerse, VisuLogic, VisualPuzzles, VGRP-Bench — focus on mathematical/logical or abstract visual reasoning.
- **Map/spatial reasoning**: CityBench, MapBench, MapEval, GeoNav — address spatial reasoning but with coarse granularity or reliance on external tools.
- **Reasoning-oriented MLLMs**: Kimi-VL-Thinking, QvQ, Skywork-R1V (open-source); OpenAI o3, Gemini-2.5-Flash, Doubao-415 (closed-source).
- **Reinforcement fine-tuning**: Successful applications of GRPO in LLM reasoning are transferred to the multimodal domain.

## Rating

- Novelty: ⭐⭐⭐⭐ — First benchmark focused on fine-grained spatial reasoning over high-resolution maps; novel topic selection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive comparison across 16 models, visual-masking experiments, RL baselines, and error analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rigorous description of the evaluation framework.
- Value: ⭐⭐⭐⭐ — Provides an important benchmark for fine-grained visual reasoning; the open-source/closed-source performance gap is an insightful finding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)
- [\[CVPR 2026\] Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models](thinking_diffusion_penalize_and_guide_visual-grounded_reasoning_in_diffusion_mul.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)
- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)

</div>

<!-- RELATED:END -->
