---
title: >-
  [Paper Note] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes
description: >-
  [ICLR2026][VLM Reasoning][multi-view spatial reasoning] The authors introduce MV-RoboBench, the first benchmark integrating multi-view spatial reasoning with robotic manipulation execution. It contains 1.7K human-annotated QA pairs and reveals a massive gap between the strongest current VLMs (GPT-5 at only 56.4%) and humans (91.0%).
tags:
  - "ICLR2026"
  - "VLM Reasoning"
  - "multi-view spatial reasoning"
  - "robotic manipulation"
  - "VLM benchmark"
  - "embodied AI"
  - "MV-RoboBench"
date: 2026-05-08
content_hash: 42b5ed35fc5910d6
---

# Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes

**Conference**: ICLR2026  
**arXiv**: [2510.19400](https://arxiv.org/abs/2510.19400)  
**Code**: [GitHub](https://github.com/) (Project page released)  
**Area**: Multimodal VLM  
**Keywords**: multi-view spatial reasoning, robotic manipulation, VLM benchmark, embodied AI, MV-RoboBench  

## TL;DR
The authors introduce MV-RoboBench, the first benchmark integrating multi-view spatial reasoning with robotic manipulation execution. It contains 1.7K human-annotated QA pairs and reveals a massive gap between the strongest current VLMs (GPT-5 at only 56.4%) and humans (91.0%).

## Background & Motivation
- Vision-Language Models (VLMs) are the core foundation of Embodied AI and Vision-Language-Action (VLA) models, playing a critical role in robotic perception, reasoning, and decision-making.
- Most VLM evaluations focus on single-view settings, but multi-camera configurations are increasingly prevalent on robotic platforms to provide complementary perspectives and mitigate occlusion or depth ambiguity.
- Existing spatial reasoning benchmarks (EmbSpatial-Bench, Visual Spatial, RoboSpatial, etc.) primarily focus on single-view relational reasoning, lacking the integration of multi-view inputs and robotic manipulation.
- A few multi-view benchmarks (All-Angles Bench, Ego3D-Bench) only focus on photo alignment or navigation-related perception, without touching upon manipulation-oriented embodied reasoning.
- **Key Challenge**: There is a lack of a benchmark to systematically evaluate the spatial reasoning capabilities of VLMs in multi-view robotic manipulation scenarios.

## Method

### Overall Architecture
MV-RoboBench is a benchmark designed specifically for spatial reasoning in multi-view robotic manipulation scenes. Built upon two real-world robotic datasets, AgiWorld and BridgeV2, it covers both single-arm and dual-arm operations. Through a three-stage human-in-the-loop pipeline, 1,708 five-choice QA pairs were refined from approximately 980 operation episodes. The tasks are organized along two main axes—"Spatial Understanding" and "Robotic Execution"—comprising eight sub-tasks. By unifying "understanding multi-view scenes" and "judging manipulation rationality" under a single protocol, the benchmark enables a horizontal comparison of over 40 VLMs (open-source, closed-source, and reasoning-enhanced). On top of this, the benchmark includes two additional layers of analysis: first, injecting three types of Chain-of-Thought (CoT) enhancements to detect if failures stem from "missing clues" or "inability to reason"; second, using internal and external correlation axes to test the common assumptions that "accurate spatial perception leads to correct manipulation" and "strong single-view performance transfers to multi-view."

### Key Designs

**1. Eight Sub-tasks: Decomposing Multi-view Embodied Reasoning into Locatable Dimensions**  
Reporting a single total accuracy fails to pinpoint where a model struggles. Therefore, the benchmark splits capabilities into two categories and eight sub-tasks. Spatial Understanding focuses on "merging scenes across views into a consistent 3D mental representation," including Cross-View Matching, Distance Judgement, Viewpoint Identification, and 3D Spatial Consistency. Robotic Execution examines "manipulation decision-making based on this spatial understanding," including Action Planning, Step Execution, Trajectory Selection, and Affordance Recognition. This fine-grained decomposition allows for precise error localization—for instance, showing that nearly all non-reasoning models fail at 3D Spatial Consistency rather than making a generic claim that the "model is weak."

**2. High-Quality Human Construction: Rational Distractors with Unique Answers**  
Quality is ensured via a three-stage pipeline. In the data collection stage, candidate scenes are filtered by rules and then triaged by GPT-4.1 (used only for triaging, not for generating QA content), followed by human verification. In the QA generation stage, trained annotators use task-specific templates to construct five-choice questions, purposefully making the four distractors plausible yet distinguishable from the correct answer. Finally, a human-in-the-loop review iteratively revises questions and balances the answer distribution across options. Restricting the LLM to triage rather than generation prevents self-evaluation bias and closes shortcuts where models could guess based on option priors.

**3. CoT Enhancement Exploration: Testing if External Clues Fix Spatial Shortcomings**  
To answer whether poor performance is due to missing clues or reasoning deficits, three types of CoT enhancements are injected into the input without modifying the models: Textual CoT uses GPT-4.1 to generate scene descriptions as supplementary text; Visual CoT uses VGGsfm for novel view synthesis to provide extra visual evidence; Structural CoT uses MoGe-2 to estimate depth priors for geometric constraints. These supplement language, vision, and geometry respectively. Subsequent ablation studies can then distinguish which information gap hinders multi-view reasoning and how different model capacities absorb external clues.

**4. Dual-Axis Correlation Analysis: Testing Default Assumptions**  
The benchmark also uses two analysis axes to challenge common assumptions. The internal axis measures the correlation between spatial reasoning scores and robotic execution scores in multi-view scenes to test if "accurate perception leads to correct manipulation." The external axis compares model performance on the single-view benchmark OmniSpatial against MV-RoboBench to test if "single-view strength reliably transfers to multi-view." These axes allow the benchmark to provide falsifiable conclusions about capability transferability rather than just a leaderboard.

## Key Experimental Results

### Main Results: Multi-Model Multi-Category Evaluation

| Model | Avg. Accuracy | Spatial Understanding | Robotic Execution |
| :--- | :--- | :--- | :--- |
| Random Choice | 19.71% | ~19% | ~20% |
| GPT-4.1 | 30.90% | 26.8% avg | 32.8% avg |
| GPT-5 (Best) | **56.41%** | 52.7% avg | 60.4% avg |
| Gemini-2.5-pro | 49.52% | 45.8% avg | 53.2% avg |
| o4-mini | 46.47% | 40.4% avg | 52.5% avg |
| Qwen2.5-vl-72B (Best OS) | 24.29% | 21.9% avg | 26.7% avg |
| InternVL3-78B | 23.25% | 20.9% avg | 25.6% avg |
| Human | **91.04%** | 93.7% avg | 88.2% avg |

### Ablation Study: CoT Enhancement

| Method | Qwen2.5-vl-7B | Gemma-3-12B | GPT-4.1 |
| :--- | :--- | :--- | :--- |
| Baseline (No enhancement) | 20.84% | 20.49% | 29.87% |
| + CoT prompting | 20.49 (-0.35) | **24.19 (+3.70)** | 29.84 (-0.03) |
| + Text description | 20.90 (+0.06) | 18.43 (-2.06) | **31.66 (+1.79)** |
| + Novel view synthesis | 20.02 (-0.82) | 18.31 (-2.18) | 28.02 (-1.85) |
| + Depth prior | 21.14 (+0.30) | 20.41 (-0.08) | **33.12 (+3.25)** |

### Key Findings
1.  **3D Spatial Consistency is most challenging**: Most non-reasoning models perform near or below chance (~19%) on this sub-task, while reasoning-enhanced models can improve this to 49-82%.
2.  **Spatial and Robot reasoning are positively correlated**: However, this holds only when models possess sufficient cross-view fusion capabilities.
3.  **Single-view performance does not transfer reliably**: Models that excel on OmniSpatial can still perform near chance on MV-RoboBench.
4.  **Mixed effects of CoT enhancements**: Synthetic novel views tend to degrade performance, and depth priors are only effective for high-capacity models.
5.  **Reasoning-optimized architectures significantly outperform perception models**: GPT-5 shows an approximately 25 percentage point improvement over GPT-4.1.

## Highlights & Insights
- First benchmark to systematically integrate multi-view spatial reasoning and robotic manipulation, filling a vital gap.
- High-quality dataset of 1,708 human-annotated QA pairs across eight dimensional sub-tasks with fine-grained granularity.
- Identified two critical conclusions: the positive correlation between spatial-robotic reasoning and the unreliable transfer from single-view performance, providing guidance for future research.
- Systematically explored the effects of CoT enhancements in multi-view scenarios, finding that simply stacking geometric clues is insufficient.

## Limitations & Future Work
- The benchmark scale is relatively small (1.7K QA), which may not cover the full diversity of manipulation scenarios.
- All tasks utilize a five-choice MCQ format rather than open-ended spatial reasoning.
- Limited diversity in scenes as data is drawn from only two sources (AgiWorld and BridgeV2).
- Preliminary exploration of CoT; lacks integration of deeper methods like geometric encoders.
- Does not include multi-view reasoning in dynamic or video-based scenes.

## Related Work & Insights
- **Single-view spatial benchmarks**: EmbSpatial-Bench, Visual Spatial, RoboSpatial, SpatialVLM, VSI-Bench, OmniSpatial.
- **Multi-view benchmarks**: All-Angles Bench, Ego3D-Bench, ERQA, MMSI-Bench.
- **Robot evaluation**: ShareRobot.
- **3D Understanding methods**: SpatialRGPT, 3D-LLM, SpatialBot, VLM-3R.
- **VLA Models**: $\pi_0$, CogAct, OpenVLA.

## Rating

| Dimension | Rating |
| :--- | :--- |
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐** |

> This is a strong evaluation-focused work. Its core contribution lies in identifying the critical gap between multi-view perception and robotic manipulation and building a high-quality benchmark. The experiment is comprehensive, covering 30+ models, and the internal/external correlation analysis provides genuine insight. However, the methodology leans heavily toward data construction without novel model-side innovations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spatial Reasoning with Vision-Language Models in Ego-Centric Multi-View Scenes](spatial_reasoning_with_vision-language_models_in_ego-centric_multi-view_scenes.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](../../ICML2026/vlm_reasoning/3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models](internspatial_a_comprehensive_dataset_for_spatial_reasoning_in_vision-language_m.md)

</div>

<!-- RELATED:END -->
