---
title: >-
  [Paper Note] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
description: >-
  [ICLR 2026][Multimodal VLM][Spatial Reasoning] Grounded in cognitive psychology, this work introduces OmniSpatial—the first comprehensive spatial reasoning benchmark—systematically covering 4 dimensions (dynamic reasoning, complex spatial logic, spatial interaction, and perspective transformation) across 50 subcategories with 8.4K manually annotated QA pairs. The strongest reasoning model, o3, achieves only 56.33% while humans reach 92.63%, revealing that complex spatial reasoning remains a fundamental bottleneck for VLMs.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Spatial Reasoning
  - VLM Benchmark
  - Cognitive Psychology
  - Dynamic Reasoning
  - Perspective Transformation
date: 2026-05-08
content_hash: e49ff2a067a38e61
---

# OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models

**Conference**: ICLR 2026
**arXiv**: [2506.03135](https://arxiv.org/abs/2506.03135)
**Code**: [Project Page](https://github.com/omnispatial)
**Area**: Multimodal VLM / Benchmarking
**Keywords**: Spatial Reasoning, VLM Benchmark, Cognitive Psychology, Dynamic Reasoning, Perspective Transformation

## TL;DR

Grounded in cognitive psychology, this work introduces OmniSpatial—the first comprehensive spatial reasoning benchmark—systematically covering 4 dimensions (dynamic reasoning, complex spatial logic, spatial interaction, and perspective transformation) across 50 subcategories with 8.4K manually annotated QA pairs. The strongest reasoning model, o3, achieves only 56.33% while humans reach 92.63%, revealing that complex spatial reasoning remains a fundamental bottleneck for VLMs.

## Background & Motivation

**Background**: Spatial reasoning is a core capability of VLMs. Existing benchmarks (SpatialBot-Bench, EmbSpatial, etc.) focus on basic spatial relationships—left/right discrimination, distance estimation, and object counting. State-of-the-art reasoning models (o3, Gemini-2.5-Pro) have already surpassed 90% accuracy on these benchmarks, indicating that basic spatial understanding is approaching saturation.

**Limitations of Prior Work**:
- Basic spatial relations (left/right/front/back/counting) ≠ complex spatial reasoning (rotation/deformation/path planning/viewpoint transformation) → existing benchmarks underestimate the true capability gap.
- Existing benchmarks rely heavily on template-based automatic annotation → insufficient data diversity and challenge, with rigid question phrasing (e.g., "Is A to the left of B?").
- Lack of a systematic taxonomy grounded in cognitive psychology → task designs across benchmarks are fragmented and limited in coverage.

**Key Challenge**: The "high scores" VLMs achieve on existing benchmarks mask their fundamental deficiencies in complex spatial reasoning in real-world scenarios—understanding an AED location in an emergency requires not only identifying "to the right of the door," but also reading a schematic diagram, associating a map with the physical scene, and planning a route.

**Goal**: To construct a "non-saturable" comprehensive spatial reasoning benchmark covering the full spectrum of spatial cognitive abilities from basic to high-order.

**Key Insight**: Drawing from cognitive psychology theories of spatial cognition (Chabris 2006; Meneghetti 2022), the paper decomposes complex spatial reasoning into 4 complementary dimensions, using this framework to design 50 subcategories and ensure theoretical completeness.

**Core Idea**: Redefine the complete boundaries of spatial reasoning evaluation by leveraging cognitive psychology theories of spatial cognition.

## Method

### Overall Architecture: A 4-Dimension × 50-Subcategory Taxonomy

Visual-spatial reasoning is formalized as the mapping $f:(\mathbf{I}_{1:T}, q) \longrightarrow a$, where $\mathbf{I}_{1:T}$ is an RGB observation stream, $q$ is the task query, and $a$ belongs to a verifiable answer/action space. Non-visual priors are excluded to ensure that improvements can be attributed to visual reasoning.

The taxonomy rests on two pillars: (i) cognitive psychology foundations—independent faculties of spatial cognition (visualization, mental rotation, perspective transformation, spatial updating); and (ii) going beyond basic relations—tasks already saturated in existing benchmarks no longer provide discriminative power.

### Key Design 1: Comprehensive Coverage Across Four Cognitive Dimensions

| Dimension | Subcategories | Core Cognitive Ability | Representative Tasks |
|---|:---:|---|---|
| **Dynamic Reasoning** | 11 | Inferring motion and temporal change from visual evidence | Motion trajectory prediction, physical simulation, traffic scene analysis |
| **Complex Spatial Logic** | 15 | High-order reasoning over relations, transformations, and geometric structures | 3D structural inference, mental folding/unfolding, spatial compatibility judgment |
| **Spatial Interaction** | 12 | Task-oriented reasoning under environmental constraints | Path planning, obstacle avoidance, context-aware action selection |
| **Perspective Transformation** | 12 | Ability to adopt alternative viewpoints | Mental rotation, mirror image understanding, multi-agent perspective coordination |

**Design Motivation**: Each dimension corresponds to a distinct cognitive faculty—dynamic reasoning emphasizes motion inference, complex logic captures abstract transformations, spatial interaction focuses on real-time environmental engagement, and perspective transformation reflects cognitive flexibility. Together, they cover the full application spectrum from robotic manipulation to autonomous driving.

### Key Design 2: Multi-Source Data and Rigorous Manual Annotation

Data are drawn from four sources: (1) web images—spanning multiple countries, scenes, and weather conditions, with specific search terms (-ai, -generated) to exclude synthetic content; (2) cognitive test items—public spatial cognition tests emphasizing pure spatial reasoning; (3) driving examination questions—license exam scenarios from at least 3 countries, with frame extraction and annotation from U.S. driving test videos; (4) existing datasets—MME (with depth information) and HOI4D (human-object interaction video frame sequences).

Annotation employs conversational natural phrasing (e.g., "If you are entering the classroom, on which side are the students?") rather than templates ("Is A to the left of B?"), with 6 annotators cross-validated at Krippendorff's $\alpha = 0.84$, indicating high agreement. The final benchmark comprises a 1.5K test set (fully manually annotated) and a 6.9K training set.

### Key Design 3: PointGraph and SpatialCoT Augmentation Strategies

**PointGraph**: An open-vocabulary grounding model (Florence-2) is used to localize multiple objects and extract centroids and bounding boxes, assembled into a JSON-format scene graph → providing VLMs with explicit geometric cues to assist reasoning.

**SpatialCoT**: Inspired by human mental imagery, InstantMesh generates 6 novel viewpoints for each input image, combined into a multi-view mosaic → fed together with the question into the VLM → chain-of-thought reasoning. This provides strong geometric priors, helping resolve occlusion and viewpoint-related reasoning ambiguities.

## Key Experimental Results

### Main Results: Representative Model Performance on OmniSpatial-test (%)

| Model | Avg. | Operations | Motion Analysis | Traffic | Localization | Geography | Strategy | Pattern Recognition | Geometric Reasoning | Egocentric | Allocentric |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 24.98 | - | 24.86 | 26.30 | 25.88 | 23.43 | 27.27 | 21.44 | 24.77 | 22.55 | 24.84 |
| GPT-4o | 47.81 | 65.54 | 57.23 | 56.47 | 52.38 | 54.09 | 26.29 | 25.48 | 75.98 | 39.49 | 39.76 |
| o3 | **56.33** | 71.89 | 66.18 | 61.18 | 68.57 | 65.45 | **40.21** | 29.68 | 77.06 | **48.40** | **48.19** |
| Gemini-2.5-Pro | 55.19 | 67.57 | **71.39** | 62.35 | **75.24** | **64.55** | 43.30 | 34.84 | 74.51 | 38.03 | 37.35 |
| InternVL3-78B | 49.33 | 63.78 | 63.12 | 56.24 | 59.24 | 51.45 | 27.63 | 30.19 | 74.51 | 38.46 | 35.90 |
| SoFar-3B | 45.14 | 56.49 | 51.16 | 54.12 | 53.14 | 52.73 | 31.75 | 22.88 | **71.60** | 36.56 | 41.69 |
| **Human** | **92.63** | **94.62** | **96.07** | **91.38** | **95.11** | **92.15** | **89.02** | **85.90** | **98.53** | **94.30** | **90.26** |

### Saturation Comparison: Existing Benchmarks vs. OmniSpatial

| Model | SpatialBot-Bench | EmbSpatial | OmniSpatial |
|---|:---:|:---:|:---:|
| o3 | >90% | >90% | 56.33% |
| Gemini-2.5-Pro | >90% | >90% | 55.19% |
| Human | ~95% | ~95% | 92.63% |

### Key Findings

- The strongest reasoning model, o3 (56.33%), vs. humans (92.63%) → a gap of 36 percentage points → complex spatial reasoning is far from solved.
- **Strategy (~40%) and Pattern Recognition (~30%)** are the most challenging dimensions → even o3 answers fewer than half correctly.
- Perspective transformation (egocentric/allocentric, ~48%/~48%) proves notably difficult → VLMs lack intrinsic 3D representations and mental rotation capabilities.
- Specialized spatial models (SpatialBot, RoboPoint) show no advantage on OmniSpatial (35–40%) → their "specialized" training sets are too simplistic.
- PointGraph and SpatialCoT improve performance on certain dimensions but with limited gains → the root cause is a deficiency in fundamental spatial cognitive abilities.

## Highlights & Insights

- **"A Warning Against Saturation"**: The paper clearly demonstrates that existing benchmarks have been "solved" by state-of-the-art models → the community needs harder evaluation standards. OmniSpatial elevates assessment from pattern matching to cognitive reasoning.
- **Theoretical Anchor in Cognitive Psychology**: Rather than arbitrarily adding difficult questions, the taxonomy is derived from spatial cognition theory → systematicity and completeness are theoretically guaranteed.
- **Diagnostic Value of 50 Subcategories**: Difficulty varies dramatically across subtasks (geometric reasoning ~75% vs. pattern recognition ~30%) → providing precise guidance for model improvement.
- **Human Performance Upper Bound at 92.63%**: Even humans do not achieve 100% → some tasks (e.g., pattern recognition at 85.90%) are genuinely challenging for humans as well → demonstrating the depth of the benchmark design.

## Limitations & Future Work

- The benchmark primarily relies on static images and a small number of video frames → dynamic spatial reasoning can be further extended to continuous video.
- All 3D reasoning tasks are still conducted on 2D images → truly interactive 3D environments (VR/simulators) are not addressed.
- Manual annotation ensures high quality but incurs large scaling costs → semi-automatic annotation pipelines should be explored for continuous data expansion.
- PointGraph and SpatialCoT show limited effectiveness as augmentation strategies → more fundamental improvements may require introducing 3D spatial priors at the model architecture level.

## Related Work & Insights

- **vs. SpatialBot-Bench/EmbSpatial**: Only 6–8 basic spatial relation categories with template annotation → OmniSpatial offers 50 categories with manual annotation → a comprehensive upgrade in dimensionality and difficulty.
- **vs. VSI-Bench (Yang et al., 2024)**: 8 indoor scene categories with template annotation and 288 samples → OmniSpatial covers indoor/outdoor scenes across multiple countries with 6.5K images.
- **vs. RoboSpatial (Song et al., 2024)**: Template-based automatic annotation at the million-sample scale → large in volume but limited in diversity and difficulty.
- **Insight**: Could OmniSpatial be integrated with embodied AI → enabling models to execute actions based on spatial reasoning in simulators → shifting from "answering questions" to "completing tasks"?

## Rating

⭐⭐⭐⭐⭐ (5/5)

Overall assessment: The first comprehensive spatial reasoning benchmark grounded in cognitive psychology theory, featuring carefully curated manual annotations across 50 categories × 8.4K questions. The substantial gap between o3 (56%) and humans (93%) validates the benchmark's discriminative power and value—establishing a new standard for evaluating spatial cognitive abilities in VLMs.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)

<!-- RELATED:END -->
