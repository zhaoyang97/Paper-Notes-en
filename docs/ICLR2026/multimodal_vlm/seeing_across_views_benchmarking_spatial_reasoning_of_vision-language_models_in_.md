---
title: >-
  [Paper Note] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes
description: >-
  [ICLR2026][Multimodal VLM][multi-view spatial reasoning] This paper proposes MV-RoboBench, the first benchmark integrating multi-view spatial reasoning with robotic manipulation evaluation…
tags:
  - "ICLR2026"
  - "Multimodal VLM"
  - "multi-view spatial reasoning"
  - "robotic manipulation"
  - "VLM benchmark"
  - "embodied AI"
  - "MV-RoboBench"
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

# Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes

**Conference**: ICLR2026
**arXiv**: [2510.19400](https://arxiv.org/abs/2510.19400)  
**Code**: [GitHub](https://github.com/) (project page available)  
**Area**: multimodal_vlm
**Keywords**: multi-view spatial reasoning, robotic manipulation, VLM benchmark, embodied AI, MV-RoboBench

## TL;DR
This paper proposes MV-RoboBench, the first benchmark integrating multi-view spatial reasoning with robotic manipulation evaluation, comprising 1.7K manually annotated QA pairs. It reveals a large performance gap between the best current VLM (GPT-5 at 56.4%) and humans (91.0%).

## Background & Motivation
- VLMs serve as the core foundation of Embodied AI and VLA models, playing a critical role in robotic perception, reasoning, and decision-making.
- Most VLM evaluations focus on single-view settings, while multi-camera configurations have become increasingly prevalent in robotic platforms, providing complementary views to mitigate occlusion and depth ambiguity.
- Existing spatial reasoning benchmarks (EmbSpatial-Bench, Visual Spatial, RoboSpatial, etc.) are primarily limited to single-view relational reasoning, lacking the combination of multi-view and robotic manipulation.
- The few available multi-view benchmarks (All-Angles Bench, Ego3D-Bench) address only photo alignment or navigation perception, without manipulation-oriented embodied reasoning.
- **Core gap**: No benchmark systematically evaluates VLMs' spatial reasoning capabilities in multi-view robotic manipulation scenarios.

## Method

### Overall Architecture: MV-RoboBench
Constructed from two real-robot datasets—AgiWorld and BridgeV2—covering single-arm and dual-arm manipulation scenarios, MV-RoboBench contains 1,708 five-choice QA items derived from 980 manipulation episodes.

### Key Design 1: Systematic Two-Category, Eight-Task Evaluation Framework
**Spatial Understanding** — four sub-tasks:
1. **Cross-View Matching**: identifying the same object across viewpoints
2. **Distance Judgement**: estimating relative distances between objects
3. **Viewpoint Identification**: reasoning about viewpoint transformation relationships
4. **3D Spatial Consistency**: maintaining consistent relative object positions in 3D space

**Robotic Execution** — four sub-tasks:
1. **Action Planning**: selecting an appropriate multi-step manipulation sequence
2. **Step Execution**: verifying correctness of the next single-step action
3. **Trajectory Selection**: assessing feasibility of candidate motion paths
4. **Affordance Recognition**: evaluating feasibility of specific object interactions

### Key Design 2: High-Quality Human Construction Pipeline
A three-stage pipeline is adopted:
1. **Data Collection**: rule-based filtering + GPT-4.1-assisted screening (triage only, no QA generation) + human verification.
2. **QA Generation**: task-specific templates combined with trained annotators to construct five-choice QA pairs.
3. **Human-in-the-loop Quality Review**: iterative review, revision, and answer distribution balancing.

### Key Design 3: CoT Augmentation Exploration
Three CoT-style augmentation strategies are explored:
- **Text CoT**: GPT-4.1-generated scene descriptions as supplementary text.
- **Visual CoT**: novel view synthesis via VGGT to provide additional visual evidence.
- **Structural CoT**: depth estimation via MoGe-2 to add geometric constraints.

### Correlation Analysis
Two analytical axes are designed:
- **Internal correlation**: relationship between spatial reasoning and robotic execution within multi-view scenes.
- **External transferability**: whether single-view spatial benchmark performance transfers to multi-view embodied reasoning.

## Experiments

### Main Results: Multi-Model, Multi-Category Evaluation

| Model | Avg. Accuracy | Spatial Understanding | Robotic Execution |
|---|---|---|---|
| Random Choice | 19.71% | ~19% | ~20% |
| GPT-4.1 | 30.90% | 26.8% avg | 32.8% avg |
| GPT-5 (best overall) | **56.41%** | 52.7% avg | 60.4% avg |
| Gemini-2.5-pro | 49.52% | 45.8% avg | 53.2% avg |
| o4-mini | 46.47% | 40.4% avg | 52.5% avg |
| Qwen2.5-vl-72B (best open-source) | 24.29% | 21.9% avg | 26.7% avg |
| InternVL3-78B | 23.25% | 20.9% avg | 25.6% avg |
| Human | **91.04%** | 93.7% avg | 88.2% avg |

### Ablation Study: CoT Augmentation

| Augmentation | Qwen2.5-vl-7B | Gemma-3-12B | GPT-4.1 |
|---|---|---|---|
| None (baseline) | 20.84% | 20.49% | 29.87% |
| + CoT prompting | 20.49 (−0.35) | **24.19 (+3.70)** | 29.84 (−0.03) |
| + Text description | 20.90 (+0.06) | 18.43 (−2.06) | **31.66 (+1.79)** |
| + Novel view synthesis | 20.02 (−0.82) | 18.31 (−2.18) | 28.02 (−1.85) |
| + Depth prior | 21.14 (+0.30) | 20.41 (−0.08) | **33.12 (+3.25)** |

### Key Findings
1. **3D Spatial Consistency is the most challenging sub-task**: Most non-reasoning models perform at or below random chance (~19%); reasoning-enhanced models improve to 49–82%.
2. **Spatial and robotic reasoning are positively correlated**: This holds only when models possess sufficient cross-view fusion capability.
3. **Single-view performance does not reliably transfer**: Models performing well on OmniSpatial may still approach random chance on MV-RoboBench.
4. **Mixed effects of CoT augmentation**: Novel view synthesis tends to degrade performance; depth priors are effective only for high-capacity models.
5. **Reasoning-optimized architectures substantially outperform perception-focused models**: GPT-5 surpasses GPT-4.1 by approximately 25 percentage points.

## Highlights & Insights
- First benchmark to systematically integrate multi-view spatial reasoning with robotic manipulation evaluation, filling a critical gap.
- All 1,708 QA items are manually curated at high quality, covering eight sub-task dimensions with fine evaluation granularity.
- Establishes two important findings: a positive correlation between spatial and robotic reasoning, and unreliable transfer from single-view benchmarks, offering guidance for future research.
- Systematically explores CoT augmentation in multi-view scenarios, finding that naively stacking geometric cues is insufficient.

## Limitations & Future Work
- The benchmark scale is relatively small (1.7K QA items), potentially insufficient to cover the full diversity of manipulation scenarios.
- All tasks use a five-choice MCQ format, precluding evaluation of open-ended spatial reasoning.
- Only two data sources are used (AgiWorld + BridgeV2), limiting scene diversity.
- CoT augmentation exploration is preliminary and does not deeply engage methods such as geometric encoders.
- Dynamic or video-based multi-view reasoning scenarios are not included.

## Related Work & Insights
- Single-view spatial benchmarks: EmbSpatial-Bench, Visual Spatial, RoboSpatial, SpatialVLM, VSI-Bench, OmniSpatial.
- Multi-view benchmarks: All-Angles Bench, Ego3D-Bench, ERQA, MMSI-Bench.
- Robotic evaluation: ShareRobot.
- 3D understanding methods: SpatialRGPT, 3D-LLM, SpatialBot, VLM-3R.
- VLA models: π0, CogAct, OpenVLA.

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Impact | ⭐⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐** |

> As a benchmark contribution, the core value lies in identifying the critical gap of multi-view + robotic manipulation and constructing a high-quality evaluation suite. The evaluation covering 30+ models is comprehensive, and the dual correlation analysis is insightful. However, the paper offers no model-level technical innovation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)

</div>

<!-- RELATED:END -->
