---
title: >-
  [Paper Note] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models
description: >-
  [ICCV 2025][Multimodal VLM][multi-turn dialogue] This paper proposes MultiVerse, a multi-turn conversation evaluation benchmark comprising 647 dialogues collected from 12 VLM evaluation datasets…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "multi-turn dialogue"
  - "VLM evaluation"
  - "benchmark dataset"
  - "checklist evaluation"
  - "in-context learning"
date: 2026-05-08
content_hash: 0554cdff6f1f1006
---

# MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models

**Conference**: ICCV 2025
**arXiv**: [2510.16641](https://arxiv.org/abs/2510.16641)
**Code**: [MultiVerse](https://passing2961.github.io/multiverse-project-page/)
**Area**: Multimodal VLM
**Keywords**: multi-turn dialogue, VLM evaluation, benchmark dataset, checklist evaluation, in-context learning

## TL;DR

This paper proposes MultiVerse, a multi-turn conversation evaluation benchmark comprising 647 dialogues collected from 12 VLM evaluation datasets, spanning 484 task types and 484 interaction goals. Using a checklist-based evaluation approach, the benchmark reveals that even the strongest model, GPT-4o, achieves only ~50% success rate on complex multi-turn conversations.

## Background & Motivation

VLMs demonstrate strong performance on single-turn benchmarks, yet real-world scenarios typically involve multi-turn interactive dialogues. Existing multi-turn evaluation benchmarks exhibit significant shortcomings:

**Limitations of MMDU**: Predominantly knowledge-oriented images (scenery, animals, art) with fewer than 15 task types, limited query styles (low lexical diversity), and sources restricted to WIT.

**Limitations of ConvBench**: Although covering 219 sub-tasks, it lacks advanced reasoning tasks such as mathematics and programming, and user queries exhibit simple linguistic structures.

**Insufficient breadth and depth**: Neither benchmark adequately covers the diversity of task types and reasoning depth encountered in realistic scenarios.

Core problem: **Do VLMs that excel on single-turn benchmarks also satisfy user needs in more interactive, multi-turn settings?**

## Method

### Dataset Construction Pipeline

MultiVerse is constructed via a five-step pipeline:

**Step 1: Source Image Collection**
49,700 images are collected from 12 VLM evaluation benchmarks (MegaBench, CharXiv, MMMU, MMMU-Pro, NaturalBench, etc.), followed by:
- Deduplication (pHash): removes 29.1% of duplicates
- Quality scoring (GPT-4o, 1–5): removes 64.77% low-quality images
- Category classification: 57 categories with small categories removed
- Weighted sampling: capped at 1K images

**Step 2: Personal Background Generation**
A fictional persona (age, occupation, hobbies) and situational context are created for each dialogue. This step is motivated by real user behavior—people typically ask questions with specific goals in mind. GPT-4o generates persona backgrounds and conversational goals. Experiments show that generating dialogues from images alone yields lower diversity and quality.

**Step 3: Multi-Turn Dialogue Generation**
Based on the persona and goals, GPT-4o generates 4-turn dialogues (8 messages) following four principles:
- Detailed and informative responses
- **Incremental complexity**: subsequent queries become progressively more challenging
- Diverse linguistic styles (avoiding templated questions such as "describe" or "what is")
- Fixed 4-turn length

**Step 4: Human Review**
Dialogues are filtered according to three criteria:
- Naturalness and authenticity (48 removed)
- Correctness (65 removed)
- Blind test (104 removed, ensuring responses require the image)

**647 dialogues** are retained in the final dataset.

**Step 5: Checklist Generation**
An instance-level checklist containing multiple binary questions is generated for each query at every turn, covering 37 key aspects (perceptual accuracy, linguistic clarity, factual correctness, etc.). Checklists are generated using GPT-4o and Claude-3.5-Sonnet and subsequently verified by human annotators.

### Evaluation Metrics

Two sub-metrics are adopted:
- **Checklist Completion Ratio**: the proportion of checklist items satisfied by a response (fraction of "Yes" answers)
- **Quality Assessment**: an integer score from 1–10 (scaled to 10–100)

The final score is the product of the two. Experiments confirm a strong positive correlation between them ($R^2 = 0.44$).

### Dataset Statistics

- 647 dialogues, average 3.91 turns
- 8 high-level interaction goals / 484 sub-interaction goals
- 9 high-level task types / 484 sub-tasks
- 25 image categories / 384 sub-categories
- Average query length: 30.53 tokens; average response length: 221.51 tokens
- 21,995 unique checklist items
- Query/response lexical diversity (MTLD): 112.0 / 118.0

## Key Experimental Results

### Oracle Setting Evaluation Across 18 VLMs

| Model | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Avg. | Slope r |
|------|------|------|------|------|------|------|
| GPT-4o | 48.56 | 50.28 | 50.54 | 49.12 | **49.63** | 0.19 |
| Qwen2.5-VL-7B | 45.13 | 47.60 | 49.31 | 50.58 | 48.15 | 1.81 |
| Qwen2.5-VL-72B | 52.05 | 47.72 | 45.82 | 46.19 | 47.95 | -1.95 |
| Claude-3.5-Sonnet | 46.60 | 47.16 | 48.30 | 45.00 | 46.76 | -0.37 |
| Gemini-2.0-Flash | 42.03 | 49.37 | 51.23 | 48.41 | 47.76 | 2.10 |
| LLaVA-1.5-7B | 9.10 | 26.43 | 29.14 | 31.81 | 24.12 | 7.08 |
| InternVL2.5-1B | 13.93 | 21.36 | 23.51 | 26.08 | 21.22 | 3.86 |

**Key Findings**:
- **Even GPT-4o achieves only 49.63%**, demonstrating that multi-turn interaction remains a substantial challenge.
- Weaker models (LLaVA-1.5-7B) exhibit the largest slope (7.08), indicating that in-context learning from ground-truth dialogue history benefits weaker models most.
- Qwen2.5-VL-72B shows a negative slope (−1.95), possibly due to a stylistic mismatch between its outputs and GPT-4o-generated reference dialogues.

### Oracle vs. Self-Prediction

| Model | Oracle | Self-Prediction | Gap |
|------|------|------|------|
| GPT-4o | 49.63 | - | - |
| Qwen2.5-VL-72B | 47.95 | - | −44.64% (max) |
| Qwen2.5-VL-7B | 48.15 | - | −30.44% |

All models exhibit significant performance degradation under the self-prediction setting, with the largest drop reaching 44.64% (Qwen2.5-VL-72B), underscoring the critical role of accurate dialogue context for model performance.

### Analysis by Interaction Goal

| Model | Verification | Analysis | Exploration | Optimization | Computation | Understanding | Research | Creation |
|------|------|------|------|------|------|------|------|------|
| GPT-4o | 46.80 | **53.67** | 42.70 | 46.36 | 50.54 | **56.41** | 43.50 | 44.51 |
| Qwen2.5-VL-7B | 46.14 | 54.23 | 44.23 | 40.64 | 44.86 | 54.71 | 39.73 | 44.45 |

**Key Findings**:
- Models perform relatively well on "Analysis" and "Understanding" tasks.
- Tasks requiring creative thinking, such as "Optimization" and "Research," yield weaker performance.
- Weaker models struggle particularly on verifiable tasks such as "Verification" and "Computation."

### Model Scaling Effects

Across the InternVL2.5, LLaMA-3.2, and Qwen2.5-VL model families, larger models generally perform better, though the effect varies by task:
- Qwen2.5-VL-72B excels on verifiable tasks (mathematics, programming).
- Qwen2.5-VL-7B outperforms the larger variant on creative tasks.

### Verbosity Bias Analysis

The linear correlation $R^2$ between response length and performance under the GPT-4o evaluator decreases across turns, suggesting that the checklist evaluation approach effectively mitigates verbosity bias.

## Highlights & Insights

1. **Breadth and depth in tandem**: The benchmark covers advanced reasoning tasks including mathematics, programming, and chart interpretation, addressing gaps left by MMDU and ConvBench.
2. **Personal Background → Conversation pipeline**: Driving dialogue generation through fictional persona backgrounds substantially improves linguistic diversity and conversational authenticity.
3. **Discovery of in-context learning effects**: Weaker models benefit most from ground-truth dialogue history, revealing the potential of conversational context as a form of guidance.
4. **Robustness of checklist evaluation**: More reliable than simple scoring, it mitigates verbosity bias and covers 37 evaluation dimensions.

## Limitations & Future Work

1. Reference dialogues are generated by GPT-4o; stylistic bias may disadvantage models whose outputs diverge significantly from GPT-4o's style.
2. The fixed 4-turn structure may not capture degradation patterns in longer dialogues (10+ turns).
3. The scale of 647 dialogues is relatively small, which may limit statistical significance in fine-grained subcategory analyses.
4. Reliance on GPT-4o as the evaluator introduces known biases, such as preference for its own stylistic outputs.

## Related Work & Insights

- **Single-turn VLM benchmarks**: MMBench, MMMU, MathVista, MM-Vet, etc., evaluate perception and reasoning.
- **Multi-turn LLM benchmarks**: MT-Bench-101, Multi-IF, and others have explored this direction in NLP.
- **Multi-turn VLM benchmarks**: MMDU (knowledge-oriented) and ConvBench (three categories: perception, reasoning, and creation).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting](capture_evaluating_spatial_reasoning_in_vision_language_models_via_occluded_obje.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[CVPR 2026\] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](../../CVPR2026/multimodal_vlm/llavashield_multimodal_multiturn_safety.md)
- [\[ICCV 2025\] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)

</div>

<!-- RELATED:END -->
