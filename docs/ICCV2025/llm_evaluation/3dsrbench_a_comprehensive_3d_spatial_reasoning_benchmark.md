---
title: >-
  [Paper Note] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark
description: >-
  [ICCV 2025][LLM Evaluation][3D Spatial Reasoning] This paper introduces 3DSRBench, the first comprehensive 3D spatial reasoning benchmark comprising 2…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "3D Spatial Reasoning"
  - "benchmark"
  - "Large Multimodal Models"
  - "visual question answering"
  - "Camera Viewpoint Robustness"
date: 2026-05-08
content_hash: ea4617b6b10832cc
---

# 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark

**Conference**: ICCV 2025
**arXiv**: [2412.07825](https://arxiv.org/abs/2412.07825)  
**Code**: [Project Page](https://3dsrbench.github.io/)  
**Area**: LLM Evaluation
**Keywords**: 3D Spatial Reasoning, benchmark, Large Multimodal Models, visual question answering, Camera Viewpoint Robustness

## TL;DR

This paper introduces 3DSRBench, the first comprehensive 3D spatial reasoning benchmark comprising 2,772 manually annotated VQA pairs across 12 question types. Through balanced data distribution and a novel FlipEval strategy, the benchmark enables robust evaluation. Results reveal that state-of-the-art LMMs—including GPT-4o and Gemini—fall far short of human performance on 3D spatial reasoning (≈52% vs. 95.7%), with substantial performance degradation under uncommon camera viewpoints.

## Background & Motivation

- **3D spatial reasoning is a fundamental capability for intelligent agents**: Downstream tasks such as autonomous navigation, robotic manipulation, and AR/VR all require models to understand the positions, orientations, and relationships of objects in 3D space.
- **Limitations of existing spatial reasoning benchmarks**:
    - Early datasets (VQA, GQA, etc.) focus only on 2D spatial relations (left/right from an observer's perspective), which can be inferred from 2D bounding boxes alone.
    - Synthetic datasets (e.g., CLEVR) exhibit a large domain gap relative to natural images.
    - SpatialRGPT relies on Omni3D 3D annotations, restricting coverage to rigid object categories in indoor/autonomous driving scenes.
    - Rule-generated VQA pairs are prone to shortcuts and biases.
- **The 3D awareness of LMMs remains underexplored**: Despite strong performance on many VQA tasks, the capabilities of models such as GPT-4o on genuine 3D spatial reasoning—requiring understanding of depth, camera extrinsics, and 3D object orientation—have rarely been systematically evaluated.

## Core Problem

1. How to construct a comprehensive and robust 3D spatial reasoning benchmark that covers multiple dimensions of 3D reasoning?
2. How do current state-of-the-art LMMs perform across the various aspects of 3D spatial reasoning (height, position, orientation, multi-object reasoning)?
3. How robust is LMMs' 3D spatial reasoning to changes in camera viewpoint, particularly under uncommon viewpoints?

## Method

### Overall Architecture

3DSRBench consists of three subsets:
- **3DSRBench-real**: 2,100 VQA pairs based on natural images from MS-COCO (5,250 after data augmentation).
- **3DSRBench-synthetic-common**: VQA pairs rendered from HSSD indoor scenes under common viewpoints.
- **3DSRBench-synthetic-uncommon**: VQA pairs from the same scenes under uncommon viewpoints.

A total of 672 synthetic VQA pairs (1,692 after augmentation) are organized into 12 question types across 4 major categories.

### Key Designs

1. **12 Question Types across 4 Categories**:

    - **Height** (1 type): Determining which object is higher in 3D world space (requiring correction for camera pitch angle).
    - **Position** (3 types): Relative object distances, proximity to the camera, and whether one object is directly above/below another.
    - **Orientation** (3 types): Which face of an object is visible to the camera; front-back relations from the object's perspective (not the observer's); left-right relations from the object's perspective.
    - **Multi-Object Reasoning** (5 types): 3D distance comparisons and orientation relationships involving three or more objects, requiring multi-step 3D computation.

2. **FlipEval Strategy**: Horizontally flipped images are paired with their original counterparts to generate complementary VQA pairs. For spatial relations involving left/right, the correct answer is flipped accordingly, eliminating positional biases (e.g., "the driver typically sits on the left") and preventing random guessing. This is used in conjunction with CircularEval, which shuffles option orders and requires all orderings to be answered correctly.

3. **Balanced Data Distribution and Shortcut Prevention**: Yes/no answers are approximately balanced; complementary image pairs (yielding opposite answers to the same question) are collected; trivially answerable questions (e.g., when two objects are at vastly different distances) are excluded; and questions are designed to be unanswerable from 2D information alone.

4. **Open-Vocabulary Entities**: Coverage is not restricted to rigid object categories but extends to humans, animals, and implicit concepts (e.g., logos on vehicles, arrows on billboards), better reflecting real-world scenarios.

### Evaluation Design

- **CircularEval**: Each question is repeated 2–4 times with different option orderings; a response is counted as correct only when all orderings are answered correctly, eliminating guessing and order bias.
- **FlipEval**: Horizontally flipped images generate paired questions to eliminate left-right bias.
- **LLM-Assisted Answer Extraction**: GPT-4 is used to extract option labels from free-form model responses.

## Key Experimental Results

### Main Results (3DSRBench-real)

| Model | Overall | Height | Position | Orientation | Multi-Object |
|-------|---------|--------|----------|-------------|--------------|
| Random | 20.9 | 25.0 | 25.0 | 16.8 | 20.1 |
| **Human** | **95.7** | 92.9 | 96.4 | 97.7 | 94.9 |
| LLaVA-v1.5-7B | 38.1 | 39.1 | 46.9 | 28.7 | 34.7 |
| InternVL2.5-8B | 50.9 | 45.9 | 68.1 | 38.7 | 43.3 |
| QWen2.5-VL-7B | 48.4 | 44.1 | 62.7 | 40.6 | 40.5 |
| SpatialReasoner | **60.3** | 52.5 | **75.2** | **55.2** | **51.8** |
| Claude-3.5V-Sonnet | 48.2 | 53.5 | 63.1 | 31.4 | 41.3 |
| GPT-4o | 44.2 | 53.2 | 59.6 | 21.6 | 39.0 |
| QWenVLMax | 52.0 | 45.1 | 70.7 | 37.7 | 44.8 |
| Gemini-2.0-Flash-think | 51.1 | 53.0 | 67.1 | 35.8 | 43.6 |

**Key Finding**: The best-performing open-source model, SpatialReasoner (60.3%), still lags behind humans by 35.4%. Orientation and multi-object reasoning represent the greatest bottlenecks.

### Viewpoint Robustness (Synthetic Splits)

| Model | Common Viewpoint | Uncommon Viewpoint | Performance Drop |
|-------|-----------------|-------------------|-----------------|
| GPT-4o | 51.2 | 44.3 | -13.5% |
| Gemini-1.5-Pro | 59.9 | 49.5 | **-32.2%** |
| LLaVA-NeXT-8B | 45.5 | 36.8 | -19.1% |
| Cambrian-1-8B | 48.1 | 39.9 | -17.0% |

### Ablation Study

- **Visual Encoder Design**: DINOv2 as a secondary encoder provides the greatest benefit for orientation and multi-object reasoning; MAE and SAM yield notable gains on height questions; the SVA connector further improves performance (37.2% → 37.8%).
- **Language Model Scale**: Performance improves consistently from 0.5B to 72B parameters; however, InternVL2.5 with a 72B LM and 6B vision encoder still trails humans by over 40%, indicating that scaling alone is insufficient.
- **Failure Mode of GPT-4o**: The model lacks explicit 3D representations (e.g., metric depth) and relies solely on visual cues for reasoning, leading to frequent errors.
- **Failure Mode of Gemini Thinking**: Although the model correctly decomposes problems into sub-steps, the absence of explicit 3D representations renders each step's execution unreliable.

## Highlights & Insights

- **First comprehensive 3D spatial reasoning benchmark**: Covers real and synthetic images, 12 question types, and both common and uncommon viewpoints.
- **FlipEval is an elegant design**: Generating paired VQA via horizontal flipping elegantly eliminates left-right bias in 3D spatial relations—a problem overlooked by prior benchmarks.
- **Reveals fundamental limitations of LMMs**: State-of-the-art models fall far short of human performance in 3D spatial reasoning, reflecting the absence of explicit 3D representations and genuine 3D reasoning capabilities in current LMMs.
- **Uncommon viewpoint analysis**: Critically relevant to robotics and embodied AI, exposing the fragility of seemingly strong models under non-canonical viewpoints.
- **Open-vocabulary entities**: Unrestricted by object category, enabling more comprehensive evaluation of 3D understanding.

## Limitations & Future Work

- Only static images are evaluated; 3D spatial reasoning in video is not addressed.
- Questions remain in binary/multiple-choice VQA format; open-ended 3D reasoning is not assessed.
- MS-COCO images are predominantly captured from a human-level perspective; uncommon viewpoint data relies mainly on synthetic images.
- No training data or fine-tuning methodology for 3D spatial reasoning is provided; the benchmark serves purely as a diagnostic tool.
- The 12 question types, while comprehensive, may still omit certain 3D reasoning capabilities (e.g., physical reasoning, spatial planning).

## Related Work & Insights

- vs. **CLEVR/Super-CLEVR**: 3DSRBench uses real natural images rather than synthetic renderings and employs open-vocabulary entities instead of fixed categories.
- vs. **SpatialRGPT**: 3DSRBench uses manually annotated rather than rule-generated VQA pairs, avoiding shortcuts and biases, and covers a broader range of 3D reasoning dimensions (orientation, multi-object reasoning).
- vs. **CVBench**: 3DSRBench focuses more comprehensively on 3D spatial reasoning and introduces FlipEval along with uncommon viewpoint evaluation.

**Insights and Connections**:
- LMMs require explicit 3D representations to perform 3D spatial reasoning reliably—providing a clear direction for research on "3D-aware LMMs."
- Performance degradation under uncommon viewpoints has significant safety implications for embodied AI and robotics.
- 3D spatial reasoning may require novel training paradigms beyond pure scaling, such as incorporating intermediate representations like depth estimation or 3D scene graphs.

## Rating

- Novelty: ⭐⭐⭐⭐ — FlipEval is a clever design; the question taxonomy is comprehensive; uncommon viewpoint analysis is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 20+ models (open- and closed-source) with multi-dimensional analysis (encoder design, model scale, viewpoint robustness, failure cases).
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with detailed question type descriptions and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐⭐ — Fills a gap in 3D spatial reasoning evaluation and provides important guidance for both the VLM community and embodied AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](../../ICLR2026/llm_evaluation/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](../../NeurIPS2025/llm_evaluation/houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](../../NeurIPS2025/llm_evaluation/rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)
- [\[ICCV 2025\] SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)

</div>

<!-- RELATED:END -->
