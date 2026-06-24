---
title: >-
  [Paper Note] MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning
description: >-
  [ICLR 2026][VLM Reasoning][Multimodal Reasoning] This paper proposes the MMR-Life benchmark (2,646 5-way multiple-choice questions based on 19,108 real images, covering 7 reasoning types and 21 tasks) to systematically evaluate the multi-image reasoning capabilities of MLLMs in real-life scenarios. The study finds that the strongest model, GPT-5, achieves only 58.69% accuracy—14% behind human performance—and reveals key insights such as the failure of reasoning enhancement me…
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "Multimodal Reasoning"
  - "Multi-image Reasoning"
  - "Real-life Scenes"
  - "Reasoning Types"
  - "Benchmark Evaluation"
date: 2026-05-08
content_hash: b54ebfdd494fb1de
---

# MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning

**Conference**: ICLR 2026  
**arXiv**: [2603.02024](https://arxiv.org/abs/2603.02024)  
**Code**: [Project Page](https://mmr-life-bench.github.io/)  
**Area**: Multimodal Evaluation Benchmarks  
**Keywords**: Multimodal Reasoning, Multi-image Reasoning, Real-life Scenes, Reasoning Types, Benchmark Evaluation

## TL;DR

This paper proposes the MMR-Life benchmark (2,646 5-way multiple-choice questions based on 19,108 real images, covering 7 reasoning types and 21 tasks) to systematically evaluate the multi-image reasoning capabilities of MLLMs in real-life scenarios. The study finds that the strongest model, GPT-5, achieves only 58.69% accuracy—14% behind human performance—and reveals key insights such as the failure of reasoning enhancement methods on large models and the observation that RL generalization is weaker than BoN (Best-of-N).

## Background & Motivation

1. MLLM reasoning evaluation currently follows two main routes, both of which deviate from everyday reasoning scenarios:
    - **Knowledge-intensive benchmarks** (e.g., MMMU, GPQA): These use expert-level STEM problems, whereas everyday reasoning rarely requires specialized domain knowledge.
    - **Synthetic symbolic benchmarks** (e.g., VisualPuzzles, PuzzleVQA): These use jigsaw puzzles or symbolic patterns, which have a large gap from real visual scenes.

2. Multi-image input is severely insufficient:
    - Most multimodal reasoning benchmarks use single-image inputs (MMMU averages 1.05 images), which does not align with the human perception mode of acquiring information from multi-image sequences.
    - Existing multi-image benchmarks either include non-reasoning tasks or cover limited reasoning types (e.g., spatial reasoning only).

3. **Core Need**: A multimodal reasoning benchmark that comprehensively covers multiple reasoning types, is based on real-life scenarios, and supports multi-image inputs.

## Method

### Overall Architecture

MMR-Life is a multi-image multimodal reasoning benchmark oriented toward real-life scenarios, consisting of 2,646 5-way multiple-choice questions based on 19,108 real images. It covers 7 reasoning types and 21 sub-tasks, with an average of 7.22 images per question. The construction follows a data pipeline: first, real images are collected from multiple sources (public image datasets, open web resources, video frames, and existing benchmarks). Then, tasks are designed according to 7 reasoning types, and candidate questions (approx. 3.2K) are generated via two paths: "automated synthesis + manual annotation." Confusable distractors are constructed for each question. Finally, 2,646 questions are selected after a three-stage filtering process (difficulty, format, and quality). The design deliberately avoids domain-specific knowledge, focusing the difficulty on the core of everyday reasoning: "integrating multi-image information + applying multiple reasoning capabilities."

### Key Designs

**1. Systematic Classification of Seven Reasoning Types: Covering the Full Spectrum of Daily Reasoning**
Existing multi-image benchmarks often cover only a single reasoning type, failing to characterize the mixture of reasoning abilities humans use in real scenes. MMR-Life explicitly divides questions into seven categories with a relatively balanced distribution: Abductive Reasoning (inferring the most plausible explanation from observations, 11.60%), Analogical Reasoning (finding similarities and transferring to new situations, 21.47%), Causal Reasoning (inferring effects from causes, 9.94%), Deductive Reasoning (inferring specifics from general rules, 10.66%), Inductive Reasoning (inducing patterns from specific observations, 16.21%), Spatial Reasoning (understanding object positions and spatial relationships, 9.64%), and Temporal Reasoning (inferring event order and time, 20.48%).

**2. Multi-source Collection and Three-stage Quality Control: Constructing Robust Questions on Real Images**
To ensure scene realism and question reliability, data is aggregated from public image datasets (Kaggle), open web resources (e.g., eBird), video frames, and existing benchmarks. Question generation uses two paths: automated pipelines for tasks that can be reliably synthesized (e.g., temporal ordering) and manual annotation for tasks requiring implicit reasoning (e.g., abduction). Distractor construction is particularly rigorous—image options use heuristic rules to sample confusable candidates, while text options are generated by GPT-5-mini/GPT-4o/Qwen2.5-VL-32B and then manually screened to ensure 4 plausible but incorrect options. The three-stage filtering process removes questions that are too easy, standardizes formats to prevent shortcuts, and eliminates ambiguity or requirements for professional knowledge through manual audit.

**3. Mixed Text and Image Option Formats: Blocking Single-modal Shortcuts**
Answer options are not exclusively text; 1,454 questions (54.95%) use text options, while 1,192 questions (45.05%) use image options. Mixed formats force models to mobilize both language and visual understanding across different questions, ensuring the evaluation reflects cross-modal multi-image reasoning rather than single-modal shortcuts.

### Loss & Training

This paper presents an evaluation benchmark rather than a new model; therefore, no loss function design is involved. During evaluation, a unified zero-shot CoT prompt is applied to all models. Open-source models are run 5 times and averaged to suppress sampling variance.

## Experimental Design

### Evaluation Models

| Category | Representative Models | Count |
|------|---------|------|
| Closed-source + Thinking | GPT-5, Gemini-2.5-Pro, o4-mini, Claude-Sonnet-4 | 6 |
| Closed-source + No Thinking | GPT-4.1, GPT-4o, Claude-3.7-Sonnet, Doubao-1.5-vision | 5 |
| Open-source + Thinking | VL-Rethinker-72B, QVQ-72B, MM-Eureka-32B, MiMo-VL-7B | 6 |
| Open-source + No Thinking | Qwen2.5-VL-7/32/72B, Gemma3-12/27B, InternVL3.5-8B/30B | 7+ |
| Human | 12 students of various backgrounds, 210-question subset | 12 |

### Comparison with Existing Benchmarks

| Benchmark | Scale | Image Type | Reasoning Types | Knowledge Req. | Avg. Images |
|------|------|---------|---------|---------|-----------|
| MME-Reasoning | 1.2K | Symbolic | 3 types | Low | 1 |
| VisualPuzzles | 1.1K | Symbolic | 5 types | Low | 1 |
| MMMU | 11.5K | Mixed | - | High | 1.05 |
| MMRB | 4.8K | Mixed | 3 types | Medium | 6.17 |
| **MMR-Life** | **2.7K** | **Natural** | **7 types** | **Low** | **7.22** |

## Main Results

### Main Results (37 Models)

| Model | Abductive | Analogical | Causal | Deductive | Inductive | Spatial | Temporal | Average |
|------|------|------|------|------|------|------|------|------|
| Human | 79.76 | 57.65 | 75.00 | 70.59 | 63.41 | 79.76 | 79.76 | **72.28** |
| GPT-5 | 53.75 | **78.87** | 41.06 | **80.14** | **78.32** | 17.25 | 41.70 | **58.69** |
| Gemini-2.5-Pro | 54.40 | 73.77 | 36.99 | 79.43 | 73.66 | 25.10 | 35.79 | 56.86 |
| o4-mini | 41.37 | 73.59 | 27.38 | 71.28 | 68.07 | 19.22 | 32.66 | 50.49 |
| Claude-Sonnet-4 | 36.96 | 60.92 | 44.11 | 67.02 | 56.64 | 15.69 | 28.23 | 45.32 |
| GPT-4.1 | 44.30 | 71.30 | 22.43 | 67.38 | 70.16 | 13.73 | 27.31 | 48.15 |
| Qwen2.5-VL-72B | 35.50 | 55.46 | 35.36 | 52.13 | 55.48 | 12.94 | 23.80 | 40.21 |
| VL-Rethinker-72B | 36.48 | 50.88 | 33.08 | 56.03 | 57.58 | 15.69 | 21.59 | 39.68 |
| InternVL3.5-8B | 35.18 | 11.44 | 18.63 | 34.04 | 11.19 | 14.90 | 16.61 | 18.67 |

**Key Findings**:

1. ⭐⭐⭐ **MMR-Life is highly challenging**: GPT-5 achieves only 58.69%, a 14% gap from the human score of 72.28%. Almost all open-source models score below 40%, with some (e.g., InternVL3.5-8B at 18.67%) performing near the random guess level (20%).
2. ⭐⭐⭐ **Significant variance across reasoning types**: All models perform poorly on spatial reasoning (max 25.10% vs. human 79.76%), while some closed-source models outperform humans in analogical and deductive reasoning. Spatial, temporal, and causal reasoning are significant bottlenecks for current MLLMs.
3. ⭐⭐ **Open-source Thinking models show no improvement**: The average score for open-source thinking models is 27.15%, which is lower than the 29.01% of no-thinking models, suggesting that the reasoning patterns of open-source models lack generalization in real scenes.

### Reasoning Paradigm Analysis

| Analysis Dimension | Core Finding |
|---------|---------|
| Thinking length vs. Accuracy | Accuracy shows a log-linear relationship with thinking token count, but some open-source models are in the inefficient zone (high tokens, low accuracy). |
| Is long CoT always effective? | No—Inductive reasoning accuracy actually decreases with CoT, while analogical reasoning benefits significantly. Long CoT is likely suitable only for step-by-step derivation. |
| BoN vs. GRPO | BoN@8 generalizes better than GRPO across all model scales; GRPO even performs below the baseline CoT on large models. |
| Correlations between reasoning types | Analogical and inductive reasoning are highly correlated (Pearson $r=0.97$), while spatial reasoning has low correlation with others ($r=0.40$). |

### Comparison of Reasoning Enhancement Methods

| Model | Method | Abductive | Analogical | Causal | Deductive | Inductive | Spatial | Temporal | Average (Δ) |
|------|------|------|------|------|------|------|------|------|----------|
| Qwen2.5-VL-7B | CoT | 26.06 | 35.74 | 20.53 | 20.92 | 38.93 | 9.41 | 12.18 | 24.68 |
| Qwen2.5-VL-7B | BoN@8 | 27.64 | 44.72 | 22.81 | 25.53 | 48.02 | 13.33 | 13.10 | 29.54 (+4.86) |
| Qwen2.5-VL-72B | CoT | 35.50 | 55.46 | 35.36 | 52.13 | 55.48 | 12.94 | 23.80 | 40.21 |
| Qwen2.5-VL-72B | BoN@8 | 34.20 | 53.35 | 32.70 | 51.77 | 56.88 | 13.73 | 24.72 | 39.80 (-0.41) |
| Qwen2.5-VL-72B | GRPO | 36.48 | 50.88 | 33.08 | 56.03 | 57.58 | 15.69 | 21.59 | 39.68 (-0.53) |

**Key Findings**:

1. ⭐⭐⭐ **Reasoning enhancement fails on large models**: Gains from SC/BoN/GRPO relative to CoT decrease monotonically from 7B to 32B to 72B. On 72B models, BoN and GRPO performed worse than baseline CoT, suggesting diminishing marginal returns for large models that already have a high probability of sampling the correct path.
2. ⭐⭐ **RL generalization is weaker than BoN**: Across all model scales, GRPO generalization is inferior to BoN@8, indicating a risk of RL models overfitting to specific datasets during reasoning training.

### Error Analysis (GPT-5 & Gemini-2.5-Pro)

| Error Type | Ratio | Explanation |
|---------|------|------|
| Reasoning Error | 32% | Includes causal inversion (24%), temporal confusion (42%), or missing key steps (24%). |
| Abstraction Error | 17% | Insufficient capability for abstract associations or generalizations. |
| Knowledge Error | 17% | Failure to invoke correct common sense or world knowledge for reasoning. |
| Perception Error | 12% | Failure in identifying static attributes (color/shape) or dynamic changes (motion). |

## Highlights & Insights

1. ⭐⭐⭐ **Fills the gap in real-life multi-image reasoning**: This is the first benchmark to simultaneously satisfy "real-life images + multi-image input + 7 reasoning types," aligning closely with everyday reasoning scenarios.
2. ⭐⭐⭐ **Reveals critical research findings**: Discoveries such as the scaling failure of reasoning enhancement and the weaker generalization of RL compared to BoN provide important guidance for future research.
3. ⭐⭐ **Strict quality control**: The three-stage filtering (difficulty/format/quality) and manual audit reduce risks of shortcuts and data contamination.
4. ⭐⭐ **Clustering analysis of reasoning types**: Correlation analysis and hierarchical clustering reveal the internal structure of reasoning capabilities (e.g., the shared pattern between analogical and inductive reasoning).
5. ⭐ **Large-scale evaluation**: Covers 37 models, including the latest state-of-the-art closed-source models.

## Limitations & Future Work

1. ⭐⭐ **Relatively limited scale**: With 2,646 questions (some reasoning types have only 250+), the sample size per sub-task is small, which may affect statistical significance.
2. ⭐⭐ **MCQ format only**: The 5-way multiple-choice format includes a 20% guessing baseline and cannot evaluate open-ended reasoning.
3. ⭐ **Blurred boundaries between reasoning types**: Abductive and causal reasoning may overlap in practice, and some questions may involve multiple types.
4. ⭐ **Image diversity**: A high proportion of video frames and surveillance captures may not fully represent daily handheld photography.
5. ⭐ **Lack of training signals**: As an evaluation benchmark, it does not provide a training set to guide model improvement on weak reasoning types.

## Summary

MMR-Life is the first multimodal multi-image reasoning benchmark oriented toward real-life scenes, systematically covering 7 reasoning types and 21 tasks. Large-scale evaluation of 37 MLLMs reveals significant bottlenecks in spatial, temporal, and causal reasoning (GPT-5 achieves 58.69% vs. human 72.28%). Key insights include the failure of reasoning enhancement methods on large models and the insufficient generalization of open-source thinking models. This benchmark provides a foundation for evaluating and improving next-generation multimodal reasoning systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SpaCE-Eval: A Benchmark for Real-World Multi-Modal Reasoning](space-eval_a_benchmark_for_real-world_multi-modal_reasoning.md)
- [\[ICLR 2026\] Spatial Reasoning with Vision-Language Models in Ego-Centric Multi-View Scenes](spatial_reasoning_with_vision-language_models_in_ego-centric_multi-view_scenes.md)
- [\[ICLR 2026\] MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](mmr-v_whats_left_unsaid_a_benchmark_for_multimodal_deep_reasoning_in_videos.md)
- [\[ICLR 2026\] LLMs as Rules Oracles: Exploring Real-World Multimodal Reasoning in Tabletop Strategy Game Environments](llms_as_rules_oracles_exploring_real-world_multimodal_reasoning_in_tabletop_stra.md)
- [\[ICLR 2026\] IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs](iv-bench_a_benchmark_for_image-grounded_video_perception_and_reasoning_in_multim.md)

</div>

<!-- RELATED:END -->
