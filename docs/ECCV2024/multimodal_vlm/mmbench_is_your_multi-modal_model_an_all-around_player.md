---
title: >-
  [Paper Note] MMBench: Is Your Multi-modal Model an All-Around Player?
description: >-
  [ECCV2024][Multimodal VLM][VLM benchmark] Proposes MMBench—a bilingual (English/Chinese) multimodal benchmark comprising 3,217 multiple-choice questions across 20 fine-grained ability dimensions, featuring a CircularEval evaluation strategy and an LLM-based choice extraction mechanism to significantly improve evaluation robustness and fairness.
tags:
  - "ECCV2024"
  - "Multimodal VLM"
  - "VLM benchmark"
  - "multi-modal evaluation"
  - "CircularEval"
  - "choice extraction"
  - "bilingual benchmark"
date: 2026-05-08
content_hash: 92bedf4882f49942
---

# MMBench: Is Your Multi-modal Model an All-Around Player?

**Conference**: ECCV2024  
**arXiv**: [2307.06281](https://arxiv.org/abs/2307.06281)  
**Code**: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)  
**Area**: Multimodal VLM  
**Keywords**: VLM benchmark, multi-modal evaluation, CircularEval, choice extraction, bilingual benchmark

## TL;DR

Proposes MMBench—a bilingual (English/Chinese) multimodal benchmark comprising 3,217 multiple-choice questions across 20 fine-grained ability dimensions, featuring a CircularEval evaluation strategy and an LLM-based choice extraction mechanism to significantly improve evaluation robustness and fairness.

## Background & Motivation

Large vision-language models (VLMs) have advanced rapidly in recent years, but they lack systematic and reliable quantitative evaluation methods:

- **Traditional objective benchmarks** (e.g., VQAv2, COCO Caption) suffer from "false negative" issues—e.g., predicting "bicycle" when the ground truth is "bike" is penalized as incorrect; moreover, they evaluate only a single task, failing to provide **fine-grained ability profiles**.
- **Subjective evaluations** (e.g., OwlEval, LVLM-eHub) rely on human annotation, which is high-cost, biased, non-scalable, and difficult to reproduce.
- The **instruction-following capabilities** of different VLMs vary significantly; many models cannot directly output option labels (A/B/C/D), leading to exact-match evaluations severely underestimating their true capability.

Therefore, a systematically designed, robustly evaluated, and capability-comprehensive objective benchmark is highly demanded.

## Core Problem

1. How to construct a VLM evaluation benchmark that has **comprehensive capability coverage** and **controllable data quality**?
2. How to resolve the difficulty of **option extraction** caused by differences in instruction-following capabilities across various VLMs?
3. How to eliminate bias caused by **random guessing** and **option preference** in multiple-choice question evaluations?

## Method

### 1. Hierarchical Capability Taxonomy

MMBench designs a three-level capability taxonomy:

- **L-1** (2 categories): Perception, Reasoning
- **L-2** (6 categories): Coarse Perception (CP), Fine-grained Perception - Single Instance (FP-S), Fine-grained Perception - Cross Instance (FP-C), Attribute Reasoning (AR), Logic Reasoning (LR), Relation Reasoning (RR)
- **L-3** (20 categories): Covers fine-grained abilities such as object localization, action recognition, spatial relations, and social reasoning.

Each L-3 capability includes at least 125 questions, maintaining a balanced distribution.

### 2. Data Collection and Quality Control

- **Source**: Over 80% of the questions are collected from the Internet, and the remaining ~20% are constructed based on validation sets of public datasets.
- **Text-Only Filtering**: Multiple SOTA LLMs (e.g., GPT-4, Gemini-Pro) are queried using only the text. If more than half answer correctly, the question is removed (indicating it can be answered without the image, hence unsuitable for multimodal evaluation).
- **Error Sample Filtering**: All questions are fed into multiple SOTA VLMs. If all models answer incorrectly, the sample is manually reviewed, and genuinely incorrect samples are removed.
- **Bilingual Version**: Translated into Chinese leveraging GPT-4, preserving proper nouns, and manually verified $\rightarrow$ MMBench-CN.

### 3. LLM-Assisted Option Extraction

To address the issue where free-form text output from VLMs cannot be directly matched to options, a two-step extraction workflow is designed:

- **Step 1** (Heuristic Matching): Attempts to directly extract option labels A/B/C/D from the model's output.
- **Step 2** (LLM Extraction): If Step 1 fails, the question, options, and model output are sent to GPT-4 to determine which option the model's prediction matches best.
- As an option extractor, GPT-4 achieves a **91.5% alignment rate** with human annotators, which is much higher than GPT-3.5-Turbo (around 85%).

### 4. CircularEval Circular Evaluation Strategy

To eliminate biases from random guessing (a 25% baseline for 4 options) and positional preferences in multiple-choice questions:

- Perform $N$ inferences for each question with $N$ options, applying a **circular shift** to the options each time.
- The question is considered correctly answered only if the model is correct in **all $N$** inferences.
- In practice, inference can be early-terminated once the model fails, keeping the computational cost lower than $N$ times.
- Effect: Compared to VanillaEval (single inference), CircularEval generally reduces accuracy by 8–34 percentage points, thereby wider separating the gap between models.

## Key Experimental Results

| Model | Overall | CP | FP-S | FP-C | AR | LR | RR |
|---|---|---|---|---|---|---|---|
| InternLM-XComposer2 | **78.1** | 80.4 | 83.5 | 73.0 | 83.7 | 63.6 | 74.4 |
| Qwen-VL-Max | 75.4 | 74.8 | **87.2** | 67.0 | 85.3 | 54.9 | 70.5 |
| GPT-4v | 74.3 | 77.6 | 73.8 | 71.5 | **85.3** | **63.6** | 68.6 |
| LLaVA-InternLM2-20B | 72.3 | 78.3 | 76.6 | **68.2** | 78.4 | 46.2 | 69.4 |
| Gemini-Pro-V | 70.2 | 70.0 | 78.9 | 65.9 | 82.9 | 46.2 | 65.9 |
| Yi-VL-34B | 68.4 | 72.0 | 78.0 | 54.7 | 81.2 | 38.6 | 68.2 |
| OpenFlamingo v2 | 2.3 | 1.1 | 3.5 | 1.5 | 5.3 | 0.0 | 2.7 |

Key Findings:

- **LLM Backbone is Crucial**: Within the same LLaVA architecture, switching the LLM from Vicuna-7B to InternLM2-20B increases the overall accuracy from 63.4% to 72.3%, with reasoning capabilities showing particularly significant packages.
- **Model Scaling is Effective**: MiniGPT4 improves by 8.3% scaled from 7B to 13B; LLaVA v1.5 improves by 3.5% scaled from 7B to 13B.
- **Potential of Small Models**: MiniCPM-V ($\le$ 3B parameters) still achieves 61.4% under CircularEval.
- **Small Bilingual Gap**: Top models show only a 1–2% gap between MMBench and MMBench-CN, with InternLM-XComposer2 having a gap of less than 1%.
- **Impact of Content Censorship**: GPT-4v refuses to answer 1.8% of the tests (mainly celebrity identification), while Gemini-Pro-V refuses 1.6%.

## Highlights & Insights

1. **Clever Design of CircularEval**: Eliminates positional bias and random guessing through option circular shifting, substantially improving evaluation robustness at acceptable computational cost.
2. **LLM Option Extractor**: Elegantly resolves the discrepancy in instruction-following capabilities across VLMs, achieving a 91.5% alignment rate with humans.
3. **Three-level Capability Taxonomy**: 20 L-3 capability dimensions provide fine-grained diagnostic capacity, directly localizing model weaknesses.
4. **Systematic Quality Control**: Double-filtering mechanism (text-only filtering + consensus-error filtering) secures data quality.
5. **Bilingual Aligned Evaluation**: The English and Chinese versions are perfectly mapped, facilitating a fair comparison of target VLMs' cross-lingual capabilities.

## Limitations & Future Work

- The multiple-choice format itself is inherently limited—it cannot evaluate capabilities like open-ended generation, multi-turn dialogue, or long-text reasoning.
- Quality control relies on SOTA models; it might fail to detect errors when all SOTA models make the same mistake.
- CircularEval is sensitive to the number of options; the difficulty varies significantly between 2-option and 4-option settings.
- Option extraction relies heavily on the GPT-4 API, incurring non-trivial cost and risk of API version drift.
- Although covering 20 dimensions, the current evaluation lacks recent hot capability dimensions such as OCR, chart understanding, and mathematical reasoning.

## Related Work & Insights

| Benchmark | Questions | Capability Dimensions | Evaluation Method | Bilingual | Robustness Strategy |
|---|---|---|---|---|---|
| **MMBench** | 3217 | 20 (Three-level) | Multiple Choice + CircularEval | ✓ | CircularEval + LLM Extraction |
| MME | ~2400 | 14 | Yes/No | ✗ | None |
| OwlEval | 82 | Various | Subjective/Human | ✗ | None |
| SEED-Bench | 19K | 12 | Multiple Choice | ✗ | None |
| VQAv2 | 1.1M | Single | Open-ended | ✗ | Exact Match |

Compared to the simple Yes/No questions of MME, MMBench's multiple-choice questions are closer to real-world reasoning. Compared to SEED-Bench, which has a larger question pool but lacks a robustness strategy, MMBench secures evaluation reliability through CircularEval.

## Insights & Connections

- The "multiple inferences + consistency verification" concept of CircularEval can be generalized to other multiple-choice evaluation scenarios (e.g., coding capability, mathematical reasoning benchmarks).
- LLM-assisted option extraction provides a general paradigm for evaluating open-ended models, bypassing the constraint of requiring models to strictly follow output formats.
- The paper indicates the decisive impact of LLM backbones on VLM performance, inspiring subsequent research to focus more closely on language model selection and alignment.
- The evaluation code is integrated into VLMEvalKit, which has become a standard evaluation tool for subsequent VLM research.

## Rating

- Novelty: ⭐⭐⭐⭐ — CircularEval and the LLM option extractor represent meaningful methodological innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluates 21 VLMs, with multi-dimensional analysis and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich charts, and comprehensive motivation discussions.
- Value: ⭐⭐⭐⭐⭐ — Has become a standard for VLM evaluation, with VLMEvalKit being widely adopted.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)
- [\[ECCV 2024\] m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks](m_ampmaposs_a_benchmark_to_evaluate_tool-use_for_multi-step_multi-modal_tasks.md)
- [\[CVPR 2025\] EgoLM: Multi-Modal Language Model of Egocentric Motions](../../CVPR2025/multimodal_vlm/egolm_multi-modal_language_model_of_egocentric_motions.md)
- [\[CVPR 2026\] ROSE: Rotate Your Large Language Model to See](../../CVPR2026/multimodal_vlm/rose_rotate_your_large_language_model_to_see.md)
- [\[ECCV 2024\] Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)

</div>

<!-- RELATED:END -->
