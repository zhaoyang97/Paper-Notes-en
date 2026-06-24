---
title: >-
  [Paper Note] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
description: >-
  [ICLR 2026][VLM Reasoning][Spatial Reasoning] OmniSpatial is the first comprehensive spatial reasoning benchmark built on cognitive psychology. It systematically covers 4 dimensions and 50 subcategories (Dynamic Reasoning, Complex Spatial Logic, Spatial Interaction, and Perspective Transformation) with 8.4K human-annotated QA pairs. Results show that the o3 model achieves only 56.33% compared to a human score of 92.63%→ revealing that complex spatial reasoning remains a core…
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "Spatial Reasoning"
  - "VLM Benchmark"
  - "Cognitive Psychology"
  - "Dynamic Reasoning"
  - "Perspective Transformation"
date: 2026-05-08
content_hash: 3224e08dd1a6a4a4
---

# OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models

**Conference**: ICLR 2026  
**arXiv**: [2506.03135](https://arxiv.org/abs/2506.03135)  
**Code**: [Project Page](https://github.com/omnispatial)  
**Area**: Multimodal VLM/Benchmarking  
**Keywords**: Spatial Reasoning, VLM Benchmark, Cognitive Psychology, Dynamic Reasoning, Perspective Transformation

## TL;DR

OmniSpatial is the first comprehensive spatial reasoning benchmark built on cognitive psychology. It systematically covers 4 dimensions and 50 subcategories (Dynamic Reasoning, Complex Spatial Logic, Spatial Interaction, and Perspective Transformation) with 8.4K human-annotated QA pairs. Results show that the o3 model achieves only 56.33% compared to a human score of 92.63%→ revealing that complex spatial reasoning remains a core bottleneck for VLMs.

## Background & Motivation

**Background**: Spatial reasoning is a core capability of VLMs. Existing benchmarks (SpatialBot-Bench, EmbSpatial, etc.) focus on basic spatial relations—distinguishing left/right, estimating distance, and object counting. Recent reasoning models (o3, Gemini-2.5-Pro) have reached >90% accuracy on these benchmarks → basic spatial understanding is nearing saturation.

**Limitations of Prior Work**:
- Basic spatial relations (left/right/front/back/counting) $\neq$ complex spatial reasoning (rotation/deformation/path planning/viewpoint transformation) → existing benchmarks underestimate the real capability gap.
- Existing benchmarks mostly use template-based automatic annotation → insufficient data diversity and challenge, with rigid question phrasing (e.g., "Is A to the left of B?").
- Lack of systematic classification supported by cognitive psychology theory → task designs are fragmented with limited coverage.

**Key Challenge**: High scores on existing benchmarks mask fundamental deficiencies in real-world complex spatial reasoning. For instance, understanding the location of an AED in an emergency requires more than identifying "to the right of the door"; it requires interpreting diagrams, correlating maps with real scenes, and planning routes.

**Goal**: Construct an "unsaturable" comprehensive spatial reasoning benchmark covering the full spectrum of spatial cognitive abilities from basic to high-level.

**Key Insight**: Grounded in cognitive psychology theories (Chabris 2006; Meneghetti 2022), complex spatial reasoning is divided into 4 complementary dimensions → using this framework to design 50 subcategories → ensuring theoretical completeness.

**Core Idea**: Redefine the complete boundaries of "spatial reasoning" evaluation using cognitive psychology.

## Method

### Overall Architecture

OmniSpatial formalizes visual-spatial reasoning as a mapping $f:(\mathbf{I}_{1:T}, q) \longrightarrow a$: given an RGB observation stream $\mathbf{I}_{1:T}$ and a task query $q$, the model outputs $a$ within a verifiable answer/action space. Annotations deliberately exclude questions solvable by linguistic common sense alone to ensure score improvements are attributable to visual reasoning. The benchmark is built around a cognitive-driven taxonomy—dividing spatial cognition into 4 complementary dimensions and 50 subcategories—complemented by a multi-source, human-annotated data pipeline and two inference enhancement strategies to probe bottlenecks.

### Key Designs

**1. Taxonomy of Four Cognitive Dimensions: Redefining "Spatial Reasoning" for Completeness**

Existing benchmarks only cover basic relations like left/right or counting, which state-of-the-art models have already "solved" (>90% accuracy). OmniSpatial avoids saturation not by randomly adding difficult questions, but by deriving dimensions from independent spatial cognitive faculties (visualization, mental rotation, perspective taking, and spatial updating): Dynamic Reasoning (11 subcategories) infers motion and temporal changes from visual evidence; Complex Spatial Logic (15 subcategories) performs high-level reasoning on relations/transformations/geometry; Spatial Interaction (12 subcategories) executes task-oriented reasoning under environmental constraints; Perspective Transformation (12 subcategories) examines the ability to adopt other perspectives. These 50 subcategories ensure theoretical completeness and cover applications from robotics to autonomous driving.

**2. Multi-source Collection and Conversational Human Annotation: Closing Template Loopholes**

Template-based automatic annotation lacks diversity and is easily exploited. OmniSpatial collects images from four sources to expand the distribution: web images (excluding synthetic content via `-ai` tags), public spatial cognition tests, driver's license exams from at least 3 countries (including frames from US driving videos), and existing datasets like MME and HOI4D. All questions are converted into conversational natural language (e.g., "If you are entering the classroom, which side are the students on?"). Every question is cross-validated by 6 annotators (Krippendorff's $\alpha = 0.84$), resulting in a 1.5K human-annotated test set and a 6.9K training set.

**3. PointGraph and SpatialCoT Enhancement: Probing Bottlenecks with Geometric Priors**

VLMs lack intrinsic 3D representations, leading to reasoning failures under occlusion or viewpoint ambiguity. Two strategies were designed as probes: PointGraph uses Florence-2 to ground objects and extract centroids/bounding boxes into a JSON scene graph as explicit geometric cues; SpatialCoT uses InstantMesh to generate 6 new viewpoints of the input image, forming a multi-view grid for chain-of-thought reasoning. Limited improvements from these strategies suggest that the bottleneck is the inherent lack of spatial cognitive grounding in VLMs rather than a lack of geometric labels.

## Key Experimental Results

### Main Results: Representative Model Performance on OmniSpatial-test (%)

| Model | Avg | Manip. | Motion | Traffic | Loc. | Geo. | Strategy | Pattern | Geom. | Ego. | Allo. |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 24.98 | - | 24.86 | 26.30 | 25.88 | 23.43 | 27.27 | 21.44 | 24.77 | 22.55 | 24.84 |
| GPT-4o | 47.81 | 65.54 | 57.23 | 56.47 | 52.38 | 54.09 | 26.29 | 25.48 | 75.98 | 39.49 | 39.76 |
| o3 | **56.33** | 71.89 | 66.18 | 61.18 | 68.57 | 65.45 | **40.21** | 29.68 | 77.06 | **48.40** | **48.19** |
| Gemini-2.5-Pro | 55.19 | 67.57 | **71.39** | 62.35 | **75.24** | **64.55** | 43.30 | 34.84 | 74.51 | 38.03 | 37.35 |
| InternVL3-78B | 49.33 | 63.78 | 63.12 | 56.24 | 59.24 | 51.45 | 27.63 | 30.19 | 74.51 | 38.46 | 35.90 |
| SoFar-3B | 45.14 | 56.49 | 51.16 | 54.12 | 53.14 | 52.73 | 31.75 | 22.88 | **71.60** | 36.56 | 41.69 |
| **Human** | **92.63** | **94.62** | **96.07** | **91.38** | **95.11** | **92.15** | **89.02** | **85.90** | **98.53** | **94.30** | **90.26** |

### Saturation Comparison: Existing Benchmarks vs OmniSpatial

| Model | SpatialBot-Bench | EmbSpatial | OmniSpatial |
|------|:---:|:---:|:---:|
| o3 | >90% | >90% | 56.33% |
| Gemini-2.5-Pro | >90% | >90% | 55.19% |
| Human | ~95% | ~95% | 92.63% |

### Key Findings

- Strongest model o3 (56.33%) vs Human (92.63%) → 36 percentage point gap → complex spatial reasoning is far from solved.
- **Strategy (~40%) and Pattern Recognition (~30%)** are the most difficult dimensions → even o3 is only half correct.
- Perspective Transformation (Egocentric/Allocentric, ~48%) is significantly challenging → VLMs lack internal 3D representations and mental rotation capabilities.
- Domain-specific spatial models (SpatialBot, RoboPoint) show no advantage on OmniSpatial (35-40%) → "specialized" training sets are too simplistic.
- PointGraph and SpatialCoT yield limited gains → confirms that the root cause is the absence of basic spatial cognitive grounding.

## Highlights & Insights

- **"Warning of Saturation"**: Clearly demonstrates that existing benchmarks are "solved," signaling the need for harder benchmarks that shift from "Pattern Matching" to "Cognitive Reasoning."
- **Theoretical Anchoring in Cognitive Psychology**: The benchmark is not just a collection of hard questions; it is theoretically grounded, ensuring systematicity and completeness.
- **Diagnostic Value of 50 Subcategories**: Significant variance in difficulty (Geometry Reasoning ~75% vs. Pattern Recognition ~30%) provides precise directions for model improvement.
- **Human Upper Bound of 92.63%**: Even humans do not reach 100% on some tasks (e.g., Pattern Recognition 85.90%), proving the depth and challenge of the question design.

## Limitations & Future Work

- Primarily based on static images or short video frames → dynamic spatial reasoning could be expanded to continuous video.
- 3D reasoning tasks are conducted on 2D images → true 3D interactive environments (VR/Simulators) are not yet involved.
- Human annotation is high-quality but costly → semi-automated annotation schemes should be explored for scaling.
- PointGraph and SpatialCoT have limited effects → fundamental improvements likely require introducing 3D spatial priors at the architectural level.

## Related Work & Insights

- **vs SpatialBot-Bench/EmbSpatial**: These use only 6-8 basic spatial relations and template annotation → OmniSpatial offers 50 categories and human annotation, upgrading dimensions and difficulty.
- **vs VSI-Bench (Yang et al., 2024)**: Focused on 8 indoor scenes with 288 samples → OmniSpatial covers international indoor/outdoor scenes with 6.5K images.
- **vs RoboSpatial (Song et al., 2024)**: Reaches million-scale with automatic templates → large scale but limited in diversity and difficulty.
- **Insight**: Could OmniSpatial be integrated with Embodied AI → allowing models to perform actions in a simulator after spatial reasoning → shifting from "answering questions" to "executing tasks"?

## Rating

⭐⭐⭐⭐⭐ (5/5)

Overall Assessment: The first comprehensive spatial reasoning benchmark based on cognitive psychology. The massive gap between o3 (56%) and Human (93%) across 50 categories and 8.4K human-annotated samples proves its discriminative power and value—setting a new bar for VLM spatial cognitive evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models](internspatial_a_comprehensive_dataset_for_spatial_reasoning_in_vision-language_m.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)
- [\[ICLR 2026\] SpatialLadder: Building Spatial Reasoning Capabilities for Vision-Language Models via Progressive Training](spatialladder_progressive_training_for_spatial_reasoning_in_vision-language_mode.md)
- [\[ICLR 2026\] Spatial Reasoning with Vision-Language Models in Ego-Centric Multi-View Scenes](spatial_reasoning_with_vision-language_models_in_ego-centric_multi-view_scenes.md)

</div>

<!-- RELATED:END -->
