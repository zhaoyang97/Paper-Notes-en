---
title: >-
  [Paper Note] CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models
description: >-
  [AAAI 2026][Multimodal VLM][Cross-video reasoning] This paper introduces CrossVid, the first comprehensive benchmark for systematically evaluating the Cross-Video Reasoning (CVR) capabilities of multimodal large language models (MLLMs). CrossVid encompasses 10 tasks across 4 dimensions, 5,331 videos, and 9,015 QA pairs. Experiments reveal that the current best-performing model, Gemini-2.5-Pro, achieves only 50.4% accuracy, far below the human performance of 89.2%.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Cross-video reasoning
  - video QA benchmark
  - multi-video understanding
  - multimodal large language model evaluation
  - spatiotemporal reasoning
date: 2026-05-08
content_hash: 0680cbe786f6d0e2
---

# CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.12263](https://arxiv.org/abs/2511.12263)
**Code**: [https://github.com/chuntianli666/CrossVid](https://github.com/chuntianli666/CrossVid)
**Area**: Video Understanding / Multimodal VLM
**Keywords**: Cross-video reasoning, video QA benchmark, multi-video understanding, multimodal large language model evaluation, spatiotemporal reasoning

## TL;DR
This paper introduces CrossVid, the first comprehensive benchmark for systematically evaluating the Cross-Video Reasoning (CVR) capabilities of multimodal large language models (MLLMs). CrossVid encompasses 10 tasks across 4 dimensions, 5,331 videos, and 9,015 QA pairs. Experiments reveal that the current best-performing model, Gemini-2.5-Pro, achieves only 50.4% accuracy, far below the human performance of 89.2%.

## Background & Motivation
- Existing video understanding benchmarks (Video-MME, NExT-QA, ActivityNet-QA, etc.) almost exclusively focus on **single-video analysis**, and are incapable of evaluating a model's ability to reason over multiple videos simultaneously.
- Recent multi-view benchmarks such as All-Angles Bench are limited to multi-perspective videos of the same scene, with extremely narrow task variety and scene coverage (only 90 scenes, 6 tasks, and 2,132 QA pairs).
- Numerous real-world scenarios require **cross-video** comparison, aggregation, and reasoning (e.g., contrasting procedural differences across multiple cooking videos, or tracking targets across views of the same scene). Existing benchmarks offer no coverage of these scenarios.
- Open-source MLLMs have received virtually no targeted training for CVR tasks, leaving their cross-video reasoning capabilities as a black box.

## Core Problem
How can the CVR capabilities of MLLMs be evaluated comprehensively, systematically, and reliably? What are the true capability boundaries of current MLLMs when required to integrate and compare information across multiple videos for reasoning?

## Method

### Overall Architecture
CrossVid is an **evaluation benchmark** rather than a training methodology. Its core design comprises:

1. **Hierarchical Task Taxonomy**: 4 high-level dimensions → 10 specific tasks
    - **Comparative Analysis**: Behavior Understanding (BU), Narrative Comprehension (NC), Cooking Comparison (CC), Procedural Error Analysis (PEA)
    - **Temporal Understanding**: Plot Inference (PI), Functional Step Alignment (FSA), Procedural Step Sequencing (PSS)
    - **Multi-view Reasoning**: Multi-view Spatial Reasoning (MSR), Multi-view Object Counting (MOC)
    - **Free-form QA**: Cooking Comparison QA (CCQA)

2. **Data Scale**: 5,331 videos sourced from 6 public datasets (Animal Kingdom, MovieChat-1K, YouCook2, VisDrone, Charades, Assembly101), yielding 9,015 QA pairs covering 32 categories; each query requires understanding approximately 770 seconds of video content on average.

3. **Question Formats**: Single-choice (SC), multiple-choice (MC), closed-form generation (CG), and open-form generation (OG).

### Key Designs
**Semi-automatic Annotation Pipeline (Four Stages)**:
1. **Frame Description**: Dense frames are extracted from videos; Qwen2.5-VL-72B generates frame-level descriptions, combined with original dataset metadata (plot summaries, scene descriptions, action labels).
2. **QA Generation**: Videos are clustered by original dataset labels to ensure semantic relatedness within each group; DeepSeek-R1 automatically generates QA pairs from frame descriptions with task-specific prompts, requiring the model to analyze cross-video relationships and provide reasoning explanations.
3. **Data Filtering and Refinement**: Ten expert annotators apply a three-step filtering process—removing questions that do not require video understanding → removing questions involving only a single video → excluding subjective or overly complex questions; in the refinement stage, annotators independently re-answer questions to verify answer uniqueness.
4. **Quality Control**: An independent expert panel conducts a final review.

**Anti-shortcut Design**: In the PSS task, temporal re-alignment (each preceding clip advanced by 1–5 seconds, subsequent clips delayed accordingly) eliminates visual shortcuts arising from camera-angle continuity, compelling models to rely on semantic content rather than low-level features.

**Fully Manual Annotation for Multi-view Tasks**: Because MSR and MOC involve fine-grained spatial relationships, automatic generation is not used; both tasks are annotated entirely by hand based on per-frame bounding box annotations from the VisDrone dataset.

### Loss & Training
Not applicable (this paper presents an evaluation benchmark, not a training methodology).

**Evaluation Metric Design**:
- Single-choice: exact-match accuracy
- Multiple-choice: all correct options must be fully matched for a response to be counted as correct
- FSA task: measured by IoU, defined as $\text{IoU} = \frac{\max(0, \min(A_{end}, G_{end}) - \max(A_{start}, G_{start}))}{\max(A_{end}, G_{end}) - \min(A_{start}, G_{start})}$
- PSS task: exact positional sequence match
- CCQA open-form questions: GPT-4.1 two-stage scoring—first evaluating coverage (whether scoring points are mentioned), then evaluating accuracy (whether details match the reference answer)

## Key Experimental Results

**Overall performance of 22 MLLMs** (O.Avg = average accuracy across 10 tasks):

| Model | O.Avg | C.Avg | T.Avg | M.Avg | CCQA |
|---|---|---|---|---|---|
| Human | 89.2 | 88.1 | 88.9 | 93.7 | 85.2 |
| Gemini-2.5-Pro | **50.4** | 54.7 | **56.0** | 28.7 | **59.8** |
| GPT-4.1 | 45.2 | 47.6 | 46.7 | 38.4 | 44.6 |
| Doubao-1.5-VL-pro | 44.3 | **53.8** | 36.1 | 34.7 | 50.1 |
| GPT-4o | 36.8 | 43.1 | 35.5 | 27.4 | 34.2 |
| GLM-4.1V-9B-Thinking | 35.1 | 44.7 | 23.1 | 37.8 | 26.9 |
| Qwen2.5-VL-72B | 34.4 | 42.1 | 29.2 | 23.5 | 41.2 |
| Qwen2.5-VL-32B | 33.7 | 38.3 | 26.5 | 31.7 | 41.2 |
| MiMo-7B | 28.3 | 31.2 | 23.0 | 33.6 | 22.0 |
| InternVL3-8B | 25.6 | 26.1 | 20.3 | **40.7** | 9.7 |

**Particularly Challenging Tasks**:
- **FSA (Functional Step Alignment)**: The best-performing Gemini-2.5-Pro achieves only 13.4%, compared to 85.2% for humans—a gap of 71.8 percentage points.
- **PSS (Procedural Step Sequencing)**: Gemini-2.5-Pro achieves a relatively higher 78.2%, but most open-source models score below 15%.
- **Multi-view Reasoning**: The best open-source model, InternVL3-8B, achieves only 40.7%, compared to 93.7% for humans.

### Ablation Study

**Effect of Frame Count** (Qwen2.5-VL-72B):
- 32 frames → 256 frames: O.Avg increases from 33.8% to 39.5% (+5.7%); CCQA increases from 18.9% to 34.0% (+15.1%).
- However, more frames are not always better: excessive frames may introduce noise—for instance, in plot inference tasks, an abundance of irrelevant shots disrupts causal chain reasoning.

**Effect of Chain-of-Thought (CoT) Prompting**:
- Larger models benefit more from CoT: Qwen2.5-VL-72B +5.1% (34.4→39.5); InternVL3-38B +0.9%.
- Smaller models may even regress: MiniCPM-o 2.6 −1.9% (25.6→23.7).
- CoT yields the most pronounced improvements on temporal understanding and multi-view reasoning tasks.

**Error Type Analysis** (4 categories):
1. **Critical Frame Loss**: Simultaneous input of multiple videos reduces per-video frame count, causing the omission of key information.
2. **Single-Video Comprehension Errors**: Key frames are captured, but understanding of individual videos is insufficient.
3. **Cross-Video Comparison Errors**: Individual videos are correctly understood, but aggregation and comparative reasoning across videos fails (the most fundamental issue).
4. **Format Errors**: Models fail to produce outputs in the required format (e.g., temporal interval format).

## Highlights & Insights
1. **Fills a Critical Gap**: CrossVid is the first systematic CVR benchmark, advancing video understanding from a "single-video, single-question" paradigm to a "multi-video, single-question" paradigm.
2. **Rich Hierarchical Task Design**: 10 tasks across 4 dimensions cover comparative, temporal, spatial, and open-form reasoning types, spanning 32 categories—far surpassing the previous maximum of 6 tasks in All-Angles Bench.
3. **Anti-shortcut Mechanism**: The temporal re-alignment design in the PSS task elegantly prevents models from exploiting visual continuity as a shortcut.
4. **Rigorous Annotation Quality Control**: The semi-automatic pipeline incorporates 10 expert annotators, multi-stage filtering and refinement, and independent quality control.
5. **Comprehensive Experimental Coverage**: 22 models (closed- and open-source, ranging from 7B to 78B+ MoE), accompanied by detailed frame count and CoT ablations and four-category error analysis.
6. **Identifies Core Bottleneck**: The primary weakness of current MLLMs is explicitly identified as "cross-video evidence aggregation and comparison" rather than single-video comprehension.

## Limitations & Future Work
1. **Evaluation Only, No Solutions Proposed**: As a benchmark paper, no training methods, architectural improvements, or data augmentation strategies are proposed to enhance CVR capabilities.
2. **Limited Video Sources**: The 6 public datasets are largely concentrated in specific domains (cooking via YouCook2, film via MovieChat-1K, aerial surveillance via VisDrone), lacking coverage of medical, educational, industrial, and other practical application scenarios.
3. **Open-form Evaluation Relies on GPT-4.1**: CCQA scoring depends entirely on GPT-4.1 as a judge, introducing potential bias from the evaluator itself.
4. **Uniform Frame Sampling Strategy**: All experiments use uniform sampling; more intelligent strategies such as keyframe selection are not explored.
5. **No Guidance on Cross-Video Training Data Construction**: The problem is identified but no investigation is conducted into how to construct cross-video training data to improve CVR capabilities in MLLMs.
6. **Multi-view Tasks Limited to Drone Scenarios**: Both MSR and MOC are entirely sourced from VisDrone, resulting in insufficient scene diversity.
7. **Inflexible Per-Query Frame Allocation**: Frames are distributed uniformly across all videos, despite potentially large differences in information density across videos.

## Related Work & Insights

| Dimension | CrossVid | All-Angles Bench | Video-MME | NExT-QA |
|---|---|---|---|---|
| # Videos | 5,331 | 90 scenes | 900 | 5,440 |
| # QA Pairs | 9,015 | 2,132 | 2,700 | 52,044 |
| # Tasks | 10 | 6 | 12 | 2 |
| Multi-video | ✓ | ✓ | ✗ | ✗ |
| Open-form QA | ✓ | ✗ | ✗ | ✓ |
| Cross-video Reasoning | ✓ | Multi-view only | ✗ | ✗ |

The core distinction of CrossVid lies in extending coverage beyond multi-view (different perspectives of the same scene) to **comparison, temporal, and reasoning tasks across different scenes and different videos**.

The following research directions are suggested:
1. **Construction of cross-video training data** is an important research direction—the poor CVR performance of current MLLMs is primarily attributable to the lack of targeted training.
2. **Keyframe selection** becomes increasingly critical in multi-video scenarios, as frame budgets distributed across multiple videos substantially reduce the per-video frame count.
3. **Explicit thinking mechanisms** (thinking-enabled models) confer clear advantages for complex cross-video reasoning and merit broader adoption in open-source models.
4. Improving **cross-video comparative reasoning** capabilities may require the introduction of cross-video attention or contrastive learning mechanisms at the architectural level.
5. This benchmark can serve as a useful test platform for evaluating video agent systems that must integrate information from multiple videos for decision-making.

## Rating (⭐ 1–5 with reasons)
⭐⭐⭐⭐ (4/5)

**Strengths**:
- The problem is clearly defined and significant; CVR is a genuinely underexplored direction with strong practical demand.
- Benchmark construction is rigorous; the anti-shortcut design and multi-stage quality control reflect high standards.
- Experimental coverage is comprehensive: 22 models, multi-dimensional ablations, and error analysis provide valuable insights.
- The identified bottleneck of "cross-video comparative reasoning" offers clear guidance for future research.

**Weaknesses**:
- As a purely benchmark-oriented work, no solutions are proposed—not even simple baseline improvements are attempted.
- Video sources are concentrated in a small number of domains, and multi-view tasks are limited to drone scenarios.
- The reliability of the CCQA evaluation methodology (GPT-4.1 judge) lacks empirical validation.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)

<!-- RELATED:END -->
