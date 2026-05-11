---
title: >-
  [Paper Note] From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes
description: >-
  [NeurIPS 2025][3D Vision][3D visual grounding] This paper proposes Anywhere3D-Bench, the first 3D visual grounding benchmark spanning four levels—area, space, object…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D visual grounding"
  - "multi-level grounding"
  - "benchmark"
  - "spatial reasoning"
  - "MLLM"
date: 2026-05-08
content_hash: 6468d611be6ebf29
---

# From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes

**Conference**: NeurIPS 2025
**arXiv**: [2506.04897](https://arxiv.org/abs/2506.04897)
**Code**: [https://anywhere-3d.github.io](https://anywhere-3d.github.io)
**Area**: 3D Vision / Multimodal VLM / Visual Grounding
**Keywords**: 3D visual grounding, multi-level grounding, benchmark, spatial reasoning, MLLM

## TL;DR
This paper proposes Anywhere3D-Bench, the first 3D visual grounding benchmark spanning four levels—area, space, object, and part—revealing that even the strongest models (Gemini-2.5-Pro and o3) achieve only ~30% accuracy on space-level tasks and ~40% on part-level tasks, far below the human performance of 95%.

## Background & Motivation

**Background**: 3D visual grounding has achieved notable progress at the object level; existing benchmarks (ScanRefer, Nr3D, Sr3D) and methods primarily focus on aligning language expressions to objects in a scene.

**Limitations of Prior Work**: (a) Existing benchmarks are restricted to the object level, neglecting spatial regions beyond objects (e.g., "place the lamp next to the armchair" requires reasoning about unoccupied space); (b) grounding at the object-part level (e.g., opening a drawer) and activity-area level (e.g., a collaborative study zone) remains unevaluated; (c) existing benchmarks rarely test models' ability to reason about quantitative sizes and distances.

**Key Challenge**: Humans can naturally perform visual grounding at arbitrary locations in 3D scenes—estimating sizes, spatial relationships, and object parts—yet current models and benchmarks are confined to object-level semantics.

**Goal**: Construct a comprehensive benchmark that evaluates models across four levels of 3D visual grounding (area / space / object / part).

**Key Insight**: Design a hierarchical taxonomy of expression types (activity → space → object → part), with multiple sub-types at each level to probe different capabilities.

**Core Idea**: Systematically expose the fundamental deficiencies of current 3D visual grounding models in spatial reasoning and fine-grained perception through a multi-level, multi-type grounding benchmark.

## Method

### Overall Architecture
Anywhere3D-Bench contains 2,886 expression–3D bounding-box pairs drawn from 276 scenes (ScanNet, MultiScan, 3RScan, ARKitScenes), organized into four levels. Data are created via a pipeline of GPT-4o-generated expressions, manual bounding-box annotation, and iterative verification.

### Key Designs

1. **Four-Level Visual Grounding Taxonomy**:

    - **Area level (189 items)**: Describes indoor activity zones (e.g., "collaborative study area"), requiring reasoning over functional regions composed of multiple objects.
    - **Space level (1,209 items)**: Targets spatial regions in a scene that are not occupied by objects; comprises five sub-types—*Size* (resizing an object), *Distance* (placement at a specified distance), *Situation* (egocentric viewpoint reasoning), *Commonsense* (e.g., "a 40-inch TV"), and *Trajectory* (endpoint of an object placed along a path).
    - **Object level (954 items)**: Similar to prior benchmarks but with a particular emphasis on quantitative size, shape, and distance reasoning.
    - **Part level (534 items)**: Targets specific parts of objects; includes three sub-types—*Motion* (predicting the position of a part after movement), *Relation*, and *Function*.

2. **Adaptive IoU Evaluation Metric**:

    - **Function**: A piecewise IoU formula designed to accommodate the geometric characteristics of different levels.
    - **Mechanism**: The area level uses 2D XY-plane IoU (ignoring height); when any dimension of the ground-truth box is extremely small (< threshold $t$), the metric uses 2D IoU over the remaining two dimensions and checks whether the center distance and predicted size along the thin dimension fall within the threshold; otherwise standard 3D IoU is applied. The primary metric is $\mathrm{Acc}@0.25\mathrm{IoU}$.
    - **Design Motivation**: Space-level grounding may involve planar regions (e.g., placing a book), where one dimension is extremely thin; standard 3D IoU would be unduly harsh in such cases.

3. **Data Generation and Quality Control Pipeline**:

    - **Function**: GPT-4o generation + manual annotation + iterative verification.
    - **Mechanism**: A 3D scene graph (containing object labels, IDs, bounding boxes, descriptions, and relations) is constructed for each scene; GPT-4o is prompted with level- and type-specific instructions to generate expressions. Annotators adjust 3D bounding boxes in an interactive interface (supporting scaling, translation, and distance measurement) and may revise expressions. All annotations undergo a second round of manual verification.
    - **Design Motivation**: Ensure each expression corresponds unambiguously to a unique bounding box.

### Three Categories of Baseline Models

Three categories of models are evaluated: (1) **LLMs** (scene graph as text input): GPT-4.1, o4-mini, Qwen series, DeepSeek; (2) **MLLMs** (text + BEV + video frames): GPT-4.1, o3, Gemini-2.5-Pro, etc.; (3) **3D visual grounding specialist models** (point cloud + video frames): 3D-VisTA, PQ3D, Chat-Scene, Grounded 3D-LLM.

## Key Experimental Results

### Main Results (Acc@0.25 IoU)

| Model | Type | Area↑ | Space↑ | Object↑ | Part↑ | Overall↑ |
|-------|------|-------|--------|---------|-------|----------|
| GPT-4.1 | LLM | 76.19 | 17.28 | 48.00 | 22.94 | 32.34 |
| DeepSeek-R1-671B | LLM (thinking) | 71.96 | 14.61 | 47.76 | 20.92 | 30.49 |
| GPT-4.1 | MLLM | 81.48 | 19.03 | 53.88 | 25.85 | 35.90 |
| o3 | MLLM (thinking) | 87.83 | 31.26 | 60.27 | 38.77 | 45.94 |
| **Gemini-2.5-Pro** | **MLLM (thinking)** | **83.60** | **29.86** | **64.47** | **38.77** | **46.47** |
| Chat-Scene* | 3D specialist | 49.10 | 6.55 | 31.73 | 22.99 | 22.90 |
| **Human** | — | **100** | **92** | **98** | **97** | **95** |

### Breakdown Analysis: Space-Level Sub-types

| Expression Type | Gemini-2.5-Pro | o3 | GPT-4.1 (MLLM) | Notes |
|----------------|---------------|-----|----------------|-------|
| Size | ~50% | ~45% | ~40% | Relatively easy |
| Distance | ~25% | ~28% | ~15% | Requires distance reasoning |
| Situation | ~25% | ~30% | ~15% | Requires first-person viewpoint transformation |
| Commonsense | ~30% | ~32% | ~20% | Requires commonsense reasoning |
| Trajectory | ~15% | ~20% | ~10% | Hardest: requires integrated understanding |

### Key Findings
- **Space level is the largest bottleneck**: The strongest models reach only ~30%, versus 92% for humans—a gap of 60+ percentage points.
- **Thinking models substantially outperform non-thinking models**: Gemini-2.0-Flash thinking vs. non-thinking shows a ~9% gap; Qwen3-32B thinking improves space-level accuracy from 9.6% to 12.57%.
- **Visual input provides limited benefit**: The LLM→MLLM upgrade yields an average gain of 8.19% at the object level but only 3.47% at the space level.
- **3D specialist models perform worst**: Chat-Scene achieves only 6.55% at the space level, far below MLLMs.
- **Error analysis (Gemini-2.5-Pro)**: Space-level errors are mainly attributable to language reasoning and spatial reasoning failures; part-level errors are primarily due to visual perception failures.

## Highlights & Insights
- **Four-level grounding taxonomy**: A comprehensive evaluation framework that goes beyond the object level; space-level grounding constitutes an entirely new evaluation dimension. This hierarchical design can be transferred to embodied intelligence and robotic manipulation evaluation.
- **Exposes MLLM weaknesses in 3D spatial reasoning**: Even o3 with visual thinking remains weak at understanding 3D spatial relationships from 2D images, particularly directional mappings (left/right → spatial axes).
- **Thinking capability is critical for spatial reasoning**: Thinking models achieve more accurate spatial understanding through explicit reasoning steps (e.g., "thickness is the dimension perpendicular to the wall").
- The large human–model gap indicates a fundamental deficiency in current AI's 3D spatial intelligence.

## Limitations & Future Work
- The benchmark scale is relatively small (2,886 items); although the space level has the most entries (1,209), the sub-type distribution is uneven.
- Only indoor scenes are covered; generalization to outdoor scenes is not validated.
- Evaluation relies on a specific IoU threshold; rankings may differ under alternative thresholds.
- Providing 3D input (point clouds) directly to MLLMs is not evaluated.
- Data generation depends on GPT-4o, which may introduce expression biases.
- The benchmark data could serve as training signal for spatial reasoning enhancement in MLLMs.

## Related Work & Insights
- **vs. ScanRefer/Nr3D**: Classic object-level benchmarks; Chat-Scene achieves >50% on ScanRefer but only 31.73% at the object level of this benchmark, indicating that this benchmark imposes higher demands on quantitative reasoning.
- **vs. MMScan**: Includes region-level grounding but lacks space-level and part-level evaluation.
- **vs. SceneFun3D**: Covers part-level grounding but is limited to predefined functional elements (handles, buttons); this benchmark encompasses more open-ended object parts and tests motion reasoning.
- **vs. VSI-Bench/Space3D-Bench**: Video-level spatial reasoning benchmarks; neither involves 3D bounding-box prediction.
- This benchmark can serve as a core test set for evaluating the spatial understanding capabilities of future embodied intelligence systems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First four-level 3D visual grounding benchmark; space-level grounding is an entirely new evaluation dimension.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20+ models (LLM / MLLM / 3D specialist) × four levels × multiple sub-types + human evaluation + error analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, highly informative figures and tables, intuitive qualitative case analysis.
- Value: ⭐⭐⭐⭐⭐ Systematically exposes fundamental deficiencies of current AI in 3D spatial reasoning, providing important guidance for the community.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](../../ICCV2025/3d_vision/articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[NeurIPS 2025\] ROGR: Relightable 3D Objects using Generative Relighting](rogr_relightable_3d_objects_using_generative_relighting.md)
- [\[ICCV 2025\] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models](../../ICCV2025/3d_vision/mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)

</div>

<!-- RELATED:END -->
