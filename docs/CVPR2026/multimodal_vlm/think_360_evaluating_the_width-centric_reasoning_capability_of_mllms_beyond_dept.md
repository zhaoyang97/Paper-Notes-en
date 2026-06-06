---
title: >-
  [Paper Note] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal reasoning] This paper presents Think360, a multimodal benchmark focused on *reasoning width*—i.e., a model's capability for multi-path search, multi-constraint pruning, backtracking…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Multimodal reasoning"
  - "reasoning width"
  - "Tree-of-Thought evaluation"
  - "benchmark"
  - "large language models"
date: 2026-05-08
content_hash: a41bed4dfffeecd4
---

# Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth

**Conference**: CVPR 2026
**arXiv**: [2603.22689](https://arxiv.org/abs/2603.22689)  
**Code**: [Think360](https://github.com/Think360-Benchmark/Think360)  
**Area**: Multimodal VLM / LLM Reasoning
**Keywords**: Multimodal reasoning, reasoning width, Tree-of-Thought evaluation, benchmark, large language models

## TL;DR

This paper presents Think360, a multimodal benchmark focused on *reasoning width*—i.e., a model's capability for multi-path search, multi-constraint pruning, backtracking, and trial-and-error exploration. The benchmark comprises 1,200+ high-quality samples and introduces a fine-grained Tree-of-Thought evaluation protocol, revealing significant deficiencies in current MLLMs along the width dimension of reasoning.

## Background & Motivation

1. **Background**: Recent large reasoning models (LRMs) have made remarkable progress in test-time scaling and long-chain reasoning. Existing benchmarks such as MathVista, MathVerse, and OlympiadBench have continuously raised difficulty and task coverage, spanning from K-12 to graduate-level problems and from text-only to multimodal inputs.

2. **Limitations of Prior Work**: Nearly all existing evaluation benchmarks implicitly measure only *reasoning depth*—the ability to derive conclusions step by step along a single reasoning chain. However, humans rarely rely solely on linear reasoning; they more often search across the solution space in multiple directions, branch and backtrack, prune by trial and error, and integrate partial findings into a final answer.

3. **Key Challenge**: Reasoning depth and reasoning width are two orthogonal dimensions. Existing benchmarks conflate the two, making it impossible to distinguish whether a model "reasons deeply" or "searches broadly." The absence of systematic evaluation along the width dimension leads to a one-sided assessment of models' true reasoning capabilities.

4. **Goal**: To construct a multimodal benchmark specifically designed to evaluate reasoning width, including: (a) a systematic definition of the cognitive capability dimensions of reasoning width; (b) an evaluation protocol that simultaneously quantifies depth and width; and (c) a comprehensive assessment of mainstream MLLMs on width-centric reasoning.

5. **Key Insight**: The authors draw an analogy between architectural "width" designs in neural networks (shortcut connections, dropout, pyramidal features, gradient backpropagation) and reasoning strategies (pruning, divide-and-conquer, trial-and-error, backtracking), establishing a correspondence between architectural and reasoning dimensions.

6. **Core Idea**: By constructing Think360—a 1,200+ sample multimodal benchmark focused on width reasoning—and a Tree-of-Thought evaluation protocol, the paper systematically exposes the inadequacy of current MLLMs in exploratory reasoning.

## Method

### Overall Architecture

Think360 is an evaluation benchmark rather than a model. The construction pipeline consists of three stages: (1) multi-source raw data collection → (2) coarse-to-fine quality filtering → (3) annotation and rewriting. Evaluation employs pass@1 accuracy, Tree-of-Thought depth/width scores, and reasoning time/token consumption.

### Key Designs

1. **Formal Definition of Reasoning Width**

    - **Function**: Explicitly distinguishes reasoning width from reasoning depth as two orthogonal dimensions.
    - **Mechanism**: Reasoning depth measures the ability to extend step-by-step along a single reasoning chain; reasoning width focuses on five cognitive capabilities: systematic trial-and-error search, branch-and-bound pruning, divide-and-conquer strategy, hypothesize-and-test, and perceive-and-comprehend. These five capabilities correspond to different "lateral" search strategies, analogous to dropout↔pruning and shortcut connections↔backtracking in neural networks.
    - **Design Motivation**: Existing benchmarks provide virtually no dedicated quantification of width reasoning, leading to models being considered "capable of reasoning" when they can merely traverse a fixed path at length, without any systematic evaluation of multi-path search capability.

2. **Multi-source Data Construction and Quality Filtering**

    - **Function**: Constructs 1,225 high-quality multimodal reasoning problems.
    - **Mechanism**: Data is sourced from four categories—math/logic competition problems, textbook examples, existing benchmarks (MathVision, DynaMath, MME-Reasoning, etc.), and online puzzles/IQ tests. Filtering adopts a two-stage strategy: coarse filtering uses keyword matching (e.g., *maximum/minimum*, *possible ways*) combined with GPT-4o as a judge; fine filtering involves manual secondary quality and diversity checks. Proof-based problems are rewritten to yield verifiable answers, and game problems are reformulated into enumerable QA formats.
    - **Design Motivation**: Width-reasoning problems directly drawn from existing benchmarks constitute an extremely small fraction (e.g., only 2.7% in MathVista, 1.7% in OlympiadBench), necessitating dedicated collection and adaptation. The significant format heterogeneity across sources also requires unification into objectively verifiable forms.

3. **Fine-grained Taxonomy**

    - **Function**: Categorizes problems along multiple axes to support fine-grained analysis.
    - **Mechanism**: Four classification axes are employed—answer type (multiple choice 16.9%, free response 83.1%), difficulty level (five tiers: Easy/Basic/Medium/Hard/Olympiad, approximately normally distributed), cognitive capability (5 non-exclusive categories), and problem type (6 non-exclusive categories). Non-exclusive categorization allows a single problem to be annotated with multiple cognitive capabilities simultaneously.
    - **Design Motivation**: Exclusive categorization fails to capture the fact that width-reasoning problems typically require multiple cognitive capabilities at once. Non-exclusive categorization, visualized through frequency statistics and chord diagrams, reveals co-occurrence patterns among different capabilities.

4. **Tree-of-Thought Evaluation Protocol (ToT-Eval)**

    - **Function**: Goes beyond traditional pass@1 accuracy by quantifying model reasoning processes along both depth and width dimensions.
    - **Mechanism**: The protocol proceeds in two steps—(a) *Tree construction*: given a problem and the model's complete response, GPT-4o extracts key reasoning steps and organizes them into a hierarchical tree, where depth represents sequential reasoning dependencies (parent–child relationships) and width represents parallel alternative explorations (sibling nodes at the same level). (b) *Depth/width scoring*: GPT-4o assesses the correctness of each node (logical validity and factual accuracy). The depth score equals the length of the longest correct reasoning chain; the width score counts the number of valid parallel reasoning branches.
    - **Design Motivation**: Traditional outcome-based evaluation considers only the correctness of the final answer, making it impossible to distinguish whether a model arrived at the answer directly or through thorough exploration and verification. ToT-Eval simultaneously quantifies exploratory breadth and reasoning depth, providing a more precise characterization of width-centric reasoning capability.

### Loss & Training

This paper is a benchmark evaluation study and does not involve model training. For evaluation, temperature is set to 0.7; each problem is repeated 3 times and averaged to reduce variance. All models are configured with their maximum supported output length. The effect of Chain-of-Thought prompting (with vs. without) is also examined.

## Key Experimental Results

### Main Results

The evaluation covers 12 major model families (GPT, Gemini, Claude, Grok, Doubao, QwenVL, InternVL, LLaVA, Llama, GLM-V, MiMo, Kimi), comprising 30+ models in total.

| Model | Overall Accuracy | Reasoning Time (s) | Token Consumption | Trial-and-Error | Branch-and-Bound |
|---|---|---|---|---|---|
| Gemini-2.5-pro | **46.0%** | 160.19 | 17270 | 38.5% | 51.8% |
| o3 | 42.3% | 261.59 | 6326 | 35.5% | 48.0% |
| o4-mini | 42.1% | 84.61 | 6736 | 34.3% | 48.0% |
| Gemini-2.5-flash-thinking | 38.3% | 107.33 | 21273 | 31.1% | 43.4% |
| o1 | 36.8% | 186.81 | 6537 | 29.6% | 40.6% |
| Claude-3.7-Sonnet-Thinking | 35.5% | 295.94 | 13819 | 29.4% | 38.8% |
| MiMo-VL-RL (7B) | 28.3% | 334.21 | 7381 | 24.9% | 27.9% |
| GPT-4o | 16.0% | 13.28 | 309 | 15.3% | 16.8% |
| LLaVA-Onevision (7B) | 8.3% | 36.58 | 648 | 5.8% | 10.0% |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| CoT prompting (GPT-4o) | +0.4% accuracy | CoT prompting yields marginal improvement; reasoning time doubles |
| Perceive-and-Comprehend subset | Above overall average | Models perform relatively well on perceptual comprehension tasks |
| Trial-and-Error subset | Below overall average | Trial-and-error search is a systematic weakness |
| Divide-and-Conquer subset | Below overall average | Divide-and-conquer tasks are similarly difficult |
| Text-only vs. Image+Text | See appendix | Analysis of the impact of multimodal input |

### Key Findings

- **Gemini-2.5-pro ranks first with 46.0% accuracy**. Its average thinking token count is approximately 17,270—roughly 3× that of o3/o4-mini—yet its reasoning time is shorter (160s vs. o3's 262s), indicating higher reasoning efficiency.
- **Best cost-effectiveness: o4-mini**—42.1% accuracy comparable to o3, but reasoning time of only 85s (one-third of o3).
- **All models struggle below 40%**: only 3 models exceed the 40% threshold, indicating that width-centric reasoning remains a formidable challenge for current MLLMs.
- **Divergence between perceive-and-comprehend and trial-and-error**: Models universally score above average on the Perceive-and-Comprehend subset, but significantly below average on the Trial-and-Error and Divide-and-Conquer subsets, suggesting that current MLLMs are more adept at structured perception than exploratory reasoning.
- **Substantial gap for open-source models**: The best open-source model, MiMo-VL-RL (7B), achieves 28.3% accuracy, approximately 18 percentage points behind the leading closed-source models.

## Highlights & Insights

- **Conceptualization of reasoning width**: Explicitly separating reasoning width from depth and establishing an insightful analogy to neural network architectural design (dropout↔pruning, shortcut↔backtracking, pyramidal features↔divide-and-conquer, etc.) yields a clear and thought-provoking conceptual framework.
- **ToT-Eval protocol**: By analyzing the tree structure of reasoning processes rather than just final answers, ToT-Eval quantifies both depth and width dimensions, providing richer diagnostic information than traditional pass@1. This evaluation paradigm is transferable to any scenario requiring assessment of reasoning quality.
- **Rigorous construction pipeline for 1,200+ problems**: Spanning competition problems to logic puzzles, multi-source data undergoes three-stage filtering (keyword matching + LLM-as-Judge + human review), ensuring problem quality and targeted coverage of width reasoning. The approach to adapting proof-based and game problems is particularly instructive.

## Limitations & Future Work

- **Dependence on GPT-4o/GPT-4o-mini**: Both tree construction and node correctness judgment rely on GPT-4o, introducing evaluator bias and incurring substantial evaluation costs.
- **Limited dataset scale**: 1,225 problems is relatively small compared to mainstream reasoning benchmarks (e.g., MathVista with 5,000+), and sample sizes within individual cognitive capability subsets may be insufficient for robust statistical conclusions.
- **No process reward/supervision evaluation**: Although ToT-Eval is proposed, it is not applied to training (e.g., as a process-based reward), leaving its utility for guiding model improvement unvalidated.
- **Scalability**: Automatically generating more high-quality width-reasoning problems at scale—avoiding the bottleneck of manual annotation—is a critical challenge for broader adoption.

## Related Work & Insights

- **vs. MathVista/MathVerse**: These benchmarks cover multimodal mathematical reasoning but contain very few width-reasoning problems (<3%). Think360 focuses exclusively on the width dimension and is thus complementary.
- **vs. CLEVR/GQA**: Early compositional visual reasoning benchmarks emphasize semantic understanding, whereas Think360 targets higher-level search and planning strategies.
- **vs. OlympiadBench**: Competition-level difficulty benchmarks emphasize long-chain reasoning (depth); Think360 focuses on multi-path search (width) at comparable difficulty levels.
- **Insights**: This benchmark exposes a systematic deficiency in current MLLMs—the lack of effective exploration and backtracking capability. This suggests that RL-based training (as in o1/o3) may need to more actively encourage multi-branch search during reasoning, rather than simply extending chain length.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Systematic evaluation of reasoning width as an independent dimension is a novel perspective, though the inherent innovation of benchmark-oriented work is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation of 30+ models with detailed per-dimension and per-difficulty analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Concepts are clearly articulated with apt analogies, though some tables are overly dense and impede readability.
- **Value**: ⭐⭐⭐⭐ — Reveals a blind spot in MLLM reasoning capability and provides meaningful guidance for future model design and training strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)
- [\[CVPR 2026\] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](humanvbench_probing_human_centric_video_understanding_in_mllms_with_automatica.md)
- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[CVPR 2026\] Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)

</div>

<!-- RELATED:END -->
