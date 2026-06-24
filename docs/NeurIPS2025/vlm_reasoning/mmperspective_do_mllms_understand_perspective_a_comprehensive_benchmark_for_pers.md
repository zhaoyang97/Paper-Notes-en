---
title: >-
  [Paper Note] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness
description: >-
  [NeurIPS 2025][VLM Reasoning][perspective understanding] The first benchmark to systematically evaluate the perspective understanding capabilities of multimodal large language models (MLLMs), comprising 10 tasks across 3 dimensions, 2,711 images, and 5,083 question–answer pairs. It reveals significant deficiencies in perspective reasoning and robustness across 43 state-of-the-art models.
tags:
  - "NeurIPS 2025"
  - "VLM Reasoning"
  - "perspective understanding"
  - "multimodal large language models"
  - "benchmark"
  - "spatial reasoning"
  - "geometric perception"
date: 2026-05-08
content_hash: 8749c361bc75647b
---

# MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness

**Conference**: NeurIPS 2025
**arXiv**: [2505.20426](https://arxiv.org/abs/2505.20426)  
**Code**: [GitHub](https://yunlong10.github.io/MMPerspective/)  
**Area**: Multimodal VLM
**Keywords**: perspective understanding, multimodal large language models, benchmark, spatial reasoning, geometric perception

## TL;DR

The first benchmark to systematically evaluate the perspective understanding capabilities of multimodal large language models (MLLMs), comprising 10 tasks across 3 dimensions, 2,711 images, and 5,083 question–answer pairs. It reveals significant deficiencies in perspective reasoning and robustness across 43 state-of-the-art models.

## Background & Motivation

**Background**: Perspective understanding is fundamental to human visual cognition, and perspective projection has been widely used to represent three-dimensional space on two-dimensional planes, from Renaissance paintings to modern camera calibration. Current MLLMs demonstrate strong performance on high-level tasks such as visual question answering and image captioning.

**Limitations of Prior Work**: Existing benchmarks (e.g., MMBench, MVBench) rarely evaluate models' geometric reasoning capabilities, particularly perspective understanding—including vanishing point localization, parallel line convergence reasoning, and spatial relationship judgment.

**Key Challenge**: Although MLLMs exhibit human-like visual perception, whether they have internalized perspective geometric priors remains entirely unknown. Specialized perspective methods rely on precise mathematical models or domain-specific datasets, making generalization to open-ended tasks difficult.

**Goal**: Do MLLMs genuinely understand perspective? Can they localize vanishing points, reason about parallel line convergence, infer three-dimensional spatial relationships, and maintain consistency under viewpoint transformations?

**Key Insight**: Constructing a hierarchical and systematic evaluation framework spanning low-level perception, high-level reasoning, and robustness.

**Core Idea**: Design 10 categories of perspective understanding tasks covering the perception–reasoning–robustness triad to systematically expose the spatial–geometric shortcomings of MLLMs.

## Method

### Overall Architecture

The MMPerspective benchmark consists of three complementary hierarchical dimensions: **Perspective Perception (P'Percep)**, **Perspective Reasoning (P'Reason)**, and **Perspective Robustness (P'Robust)**, encompassing 10 tasks in total. Evaluation difficulty increases progressively from low-level visual recognition to high-level spatial inference and transformation consistency verification.

### Key Designs

1. **Perspective Perception (P'Percep)**: Assesses the model's ability to detect and interpret explicit perspective cues in images.

    - **Vanishing Point Perception (VPP)**: Determines the location of a vanishing point or whether it falls within a specified region.
    - **Critical Line Perception (CLP)**: Identifies the horizon line from a set of candidate lines.
    - **Lens Distortion Perception (LDP)**: Distinguishes regions in an image that are free of curvilinear distortion.
    - **Viewpoint Angle Perception (VAP)**: Infers the viewing direction (upward/downward/horizontal) from visual cues.

2. **Perspective Reasoning (P'Reason)**: Tests the model's ability to integrate multiple spatial cues for geometric reasoning.

    - **Perspective Type Reasoning (PTR)**: Classifies the perspective structure of an image (one-point / two-point / three-point / nonlinear perspective).
    - **Line Relationship Reasoning (LRR)**: Determines whether two lines in 3D space are parallel, perpendicular, or intersecting.
    - **Perspective Transformation Detection (PTS)**: Detects changes in perspective type between paired images.
    - **Vanishing Point Counting (VPC)**: Estimates the number of identifiable vanishing points in a scene.
    - **Out-of-View Reasoning (OVR)**: Infers the quadrant in which a vanishing point lies when it falls outside the image frame.

3. **Perspective Robustness (P'Robust)**: Evaluates model consistency under perspective-preserving image transformations. Original images are augmented via cropping, flipping, occlusion, and similar operations to test whether models produce consistently correct answers. Two metrics are employed:

    - **Binary P'Robust Score**: Requires all transformed versions to be answered correctly, $\text{Binary-Robust}_{\mathcal{M}} = \frac{1}{|\mathcal{S}|}\sum \mathbb{1}[\bigwedge_{I \in V_s} \mathcal{M}(I,q) = a^*]$
    - **Graded P'Robust Score**: Computes the average proportion of correctly answered instances within each transformation group.

### Data Construction Pipeline

- **Data Sources**: Web-crawled architectural and indoor scene images; real-captured fisheye/linear perspective image pairs; the open-source RPVP dataset; Blender-synthesized images with precise vanishing point coordinate ground truth.
- **Annotation**: A hybrid pipeline—PTS samples are manually annotated; LDP samples use randomized combinations with recorded labels; PTR/LRR/VAP/CLP/VPC samples are sourced from web images with manual annotation; VPP combines web and Blender-synthesized images.
- **Quality Control**: Multi-stage review process; subjective tasks are independently annotated by at least two annotators; ambiguous samples are excluded.

## Key Experimental Results

### Main Results

| Model | VPP | CLP | VAP | LDP | PTR | LRR | OVR | PTS | VPC | Overall | Graded Robust |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|---------|---------------|
| InternVL2.5-8B | 38.5 | 17.9 | 53.1 | 75.4 | 40.8 | 48.3 | 34.7 | 24.9 | 67.5 | 44.6 | 38.7 |
| Qwen2.5-VL-7B | 35.3 | 29.3 | 70.4 | 73.7 | 42.4 | 44.4 | 32.1 | 28.6 | 44.7 | 44.5 | 33.2 |
| InternVL2.5-26B | 41.7 | 35.0 | 55.6 | 81.8 | 65.5 | 46.4 | 43.5 | 34.3 | 46.5 | 50.0 | 52.9 |
| Eagle-X4-8B | 39.1 | 17.1 | 46.9 | 47.7 | 65.3 | 37.1 | 18.2 | 32.9 | 68.4 | 41.4 | 60.7 |
| InternVL2.5-2B | 47.4 | 22.8 | 13.0 | 65.3 | 62.2 | 31.8 | 16.6 | 30.0 | 50.0 | 37.7 | 59.1 |

### Ablation Study

| Analysis Dimension | Key Findings |
|-------------------|-------------|
| Model Scale | Larger models generally perform better on reasoning tasks, but robustness does not exhibit a clear positive correlation with scale. |
| Perception vs. Reasoning | Models perform reasonably on surface-level perception tasks but degrade noticeably on reasoning and robustness tasks. |
| Open-source vs. Closed-source | Closed-source models such as GPT-4o lead overall, yet remain far from perfect. |
| CoT Prompting | Chain-of-thought prompting is beneficial for certain tasks. |

### Key Findings

- All 43 state-of-the-art models perform poorly on perspective reasoning and robustness; even GPT-4o exhibits significant limitations.
- Models perform relatively well on surface-level perception tasks (e.g., LDP, VPC), but degrade substantially on compositional reasoning (e.g., OVR, PTS) and robustness consistency.
- Simple geometry-preserving transformations (flipping, cropping) severely disrupt model predictions, indicating a lack of genuine geometric understanding.
- An interesting non-monotonic relationship exists between model architecture/scale and perspective capability.

## Highlights & Insights

- **Pioneer Contribution**: The first MLLM benchmark specifically designed for perspective understanding, filling a critical gap in geometric perception evaluation.
- **Hierarchical Design**: The three-dimensional evaluation framework of perception → reasoning → robustness is highly systematic, with a well-motivated progression in difficulty.
- **Large-Scale Evaluation**: Comprehensive assessment across 43 models provides rich analytical perspectives.
- **Data Diversity**: Combining real-captured, web-crawled, and synthetically rendered data sources ensures evaluation comprehensiveness.
- **Blender Synthesis Innovation**: Claude 3.7 Sonnet combined with Blender-MCP is used to automatically generate synthetic data with precise vanishing point annotations.

## Limitations & Future Work

- The benchmark focuses primarily on multiple-choice format; extension to open-ended or generative evaluation is warranted.
- The domain gap between synthetic data and real-world scenes requires further investigation.
- Robustness evaluation considers only geometry-preserving transformations; a broader range of perturbation types could be introduced.
- Perspective understanding in video or dynamic scenes is not addressed.
- Integrating perspective understanding training data to improve model capability, rather than solely evaluating it, represents a natural next step.

## Related Work & Insights

- Complements 3D spatial understanding benchmarks (e.g., SpatialBench, ScanQA) by focusing on perspective geometry in 2D images.
- Motivates research into incorporating classical computer vision geometric priors into MLLM training pipelines.
- Demonstrates that emergent capabilities do not necessarily imply systematic spatial cognition; fundamental geometric understanding still requires dedicated design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first benchmark dedicated to perspective understanding, with highly innovative problem formulation and task design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large-scale evaluation across 43 models with multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with a logical flow from geometric foundations to task definitions.
- Value: ⭐⭐⭐⭐ — Provides an important evaluation tool and directional guidance for improving spatial geometric capabilities in MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/vlm_reasoning/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[ACL 2026\] Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision](../../ACL2026/vlm_reasoning/do_mllms_understand_pointing_benchmarking_and_enhancing_referential_reasoning_in.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](../../ICCV2025/vlm_reasoning/perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[ICML 2025\] Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas](../../ICML2025/vlm_reasoning/why_is_spatial_reasoning_hard_for_vlms_an_attention_mechanism_perspective_on_foc.md)

</div>

<!-- RELATED:END -->
