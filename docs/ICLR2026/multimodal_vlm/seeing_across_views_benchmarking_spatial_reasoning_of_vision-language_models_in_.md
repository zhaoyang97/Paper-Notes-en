---
title: >-
  [Paper Note] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes
description: >-
  [ICLR2026][Multimodal VLM][multi-view spatial reasoning] This paper proposes MV-RoboBench, the first benchmark integrating multi-view spatial reasoning with robotic manipulation tasks…
tags:
  - "ICLR2026"
  - "Multimodal VLM"
  - "multi-view spatial reasoning"
  - "benchmark"
  - "embodied AI"
  - "VLM evaluation"
  - "robotic manipulation"
date: 2026-05-08
content_hash: dc704f229539c892
---

# Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes

**Conference**: ICLR2026
**arXiv**: [2510.19400](https://arxiv.org/abs/2510.19400)  
**Code**: [Project Page](https://github.com/) (open-sourced)  
**Area**: Multimodal VLM
**Keywords**: multi-view spatial reasoning, benchmark, embodied AI, VLM evaluation, robotic manipulation

## TL;DR
This paper proposes MV-RoboBench, the first benchmark integrating multi-view spatial reasoning with robotic manipulation tasks, systematically evaluating 40+ VLMs (open-source, closed-source, and reasoning-enhanced). The best-performing model, GPT-5, achieves only 56.4% accuracy, far below the human baseline of 91.0%. The study further reveals a positive correlation between spatial and robotic reasoning, and that performance on single-view benchmarks does not reliably transfer to multi-view settings.

## Background & Motivation
- VLMs serve as core components of Embodied AI, providing perceptual and reasoning capabilities for Vision-Language-Action (VLA) models.
- Most VLM evaluations focus on **single-view** settings, leaving multi-view information integration severely underexplored.
- Multi-camera configurations have become standard in robotic platforms, offering complementary viewpoints to mitigate occlusion and depth ambiguity.
- Existing spatial reasoning benchmarks (EmbSpatial-Bench, RoboSpatial, etc.) primarily target single-view reasoning; ERQA and MMSI-Bench include only limited multi-view data.
- All-Angles Bench and Ego3D-Bench employ multi-view inputs but are restricted to photo alignment or navigation perception, lacking manipulation-oriented embodied reasoning.

## Method

### Overall Architecture: MV-RoboBench Benchmark Design
Built upon the AgiWorld and BridgeV2 datasets, MV-RoboBench contains 1,708 manually annotated multiple-choice questions spanning **Spatial Understanding** and **Robotic Execution**, organized into 8 sub-tasks:

- **Spatial Understanding (4 sub-tasks)**:
    - Cross-View Matching: identifying the same object across viewpoints
    - Distance Judgement: estimating relative distances between objects
    - Viewpoint Identification: reasoning about viewpoint transformations
    - 3D Spatial Consistency: maintaining consistent 3D spatial relationships

- **Robotic Execution (4 sub-tasks)**:
    - Action Planning: planning multi-step action sequences
    - Step Execution: verifying correctness of a single next step
    - Trajectory Selection: assessing feasibility of candidate motion paths
    - Affordance Recognition: evaluating object interaction feasibility

### Key Design 1: Multi-Stage Human Quality Control Pipeline
- **Data Collection**: Rule-based filtering + GPT-4.1-assisted screening (for triage only, not QA generation) + human verification.
- **QA Generation**: Task-specific templates combined with trained annotators to construct five-choice QA pairs with plausible yet distinguishable distractors.
- **Iterative Review**: Multiple rounds of annotation, revision, and answer distribution balancing to eliminate bias.

### Key Design 2: CoT-Inspired Augmentation Exploration
Three CoT-style augmentation strategies are systematically investigated:
1. **Text CoT (w text)**: GPT-4.1-generated scene descriptions as supplementary textual context.
2. **Visual CoT (w vggt)**: Novel view synthesis via VGGT to provide additional visual evidence.
3. **Structural CoT (w depth)**: Depth estimation via MoGe-2 to introduce geometric constraints.

### Key Design 3: Dual-Axis Correlation Analysis
- **Internal correlation axis**: Relationship between spatial reasoning and robotic execution performance within multi-view scenes.
- **External transfer axis**: Whether performance on a single-view spatial benchmark (OmniSpatial) reliably predicts multi-view embodied reasoning capability.

## Experiments

### Main Results

| Model Type | Representative Model | Average Accuracy |
|---|---|---|
| Random Baseline | — | 19.7% |
| Closed-Source VLM | GPT-4.1 | 30.9% |
| Open-Source VLM | Qwen2.5-vl-72B | 24.3% |
| Open-Source MoE | Llama-4-Maverick | 26.1% |
| Reasoning Model | GPT-5 | **56.4%** |
| Reasoning Model | Gemini-2.5-pro | 49.5% |
| Human | — | **91.0%** |

### Ablation Study: CoT Augmentation

| Model | Base | w cot | w text | w vggt | w depth |
|---|---|---|---|---|---|
| Qwen2.5-vl-7B | 20.84 | 20.49 | 20.90 | 20.02 | **21.14** |
| Gemma-3-12B | 20.49 | **24.19** | 18.43 | 18.31 | 20.41 |
| GPT-4.1 | 29.87 | 29.84 | 31.66 | 28.02 | **33.12** |

### Key Findings
1. **Reasoning capability is the primary differentiator**: Reasoning-enhanced models (GPT-5, o4-mini) substantially outperform perception-focused models, yet remain far below human performance.
2. **3D Spatial Consistency is the most challenging sub-task**: Most non-reasoning models perform at or below random chance (~19.07%) on this task.
3. **CoT augmentation effects are model-dependent**: Novel view synthesis generally degrades performance; depth priors are effective only for high-capacity models; CoT prompting is most beneficial for mid-scale open-source models.
4. **Spatial and robotic reasoning are positively correlated**: This holds only for models with sufficient cross-view fusion capability.
5. **Single-view to multi-view transfer fails**: Strong performance on OmniSpatial does not reliably predict multi-view embodied reasoning ability.

## Highlights & Insights
- First systematic benchmark for multi-view robotic manipulation spatial reasoning, filling a critical gap in the field.
- Evaluation covers 40+ models across five categories, providing comprehensive experimental coverage.
- Dual-axis analysis reveals an important negative result: single-view spatial capability does not transfer reliably.
- All 1.7K QA items are fully human-curated, with data covering both single-arm and dual-arm manipulation scenarios.

## Limitations & Future Work
- Only 2D images are used as input; the impact of explicit 3D representations (point clouds, meshes) remains unexplored.
- Camera configurations are fixed by the source datasets; the effect of varying camera layouts is not investigated.
- The multiple-choice format precludes evaluation of open-ended spatial reasoning.
- CoT augmentation strategies are relatively basic; more advanced approaches such as active view selection are not explored.

## Related Work & Insights
- **Spatial reasoning benchmarks**: EmbSpatial-Bench, Visual Spatial, RoboSpatial, Spatial-MM, SpatialVLM, and VSI-Bench are all limited to single-view settings.
- **Multi-view benchmarks**: All-Angles Bench (photo alignment) and Ego3D-Bench (navigation perception) do not address robotic manipulation.
- **Robotic scene evaluation**: ShareRobot (single-view), ERQA (partial multi-view but small scale).
- **Geometry-augmented VLMs**: SpatialRGPT, SpatialLLM, and 3D-LLM explore the injection of geometric priors into language models.

## Rating
⭐⭐⭐⭐ (4/5)

A solid benchmark contribution with large-scale and systematic evaluation. The dual-axis correlation analysis provides valuable insights. As a benchmark paper, however, the methodological contribution is limited, and the CoT augmentation exploration remains relatively shallow.

---
title: >-
  [Paper Review] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes
description: >-
  [ICLR2026][Multimodal][Spatial Reasoning] Proposes MV-RoboBench, the first VLM evaluation benchmark targeting multi-view spatial reasoning in robotic scenes, comprising 1.7K manually annotated QA pairs across eight sub-tasks in spatial understanding and robotic execution. Experiments show that current state-of-the-art VLMs fall far short of human performance, and that single-view spatial benchmark performance does not reliably transfer to multi-view robotic scenarios.
tags:
  - ICLR2026
  - multimodal
  - spatial reasoning
  - benchmark
  - multi-view
  - robotic manipulation
  - VLM evaluation
  - embodied AI
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](../../ICML2026/multimodal_vlm/3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)

</div>

<!-- RELATED:END -->
