---
title: >-
  [Paper Note] Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence
description: >-
  [CVPR 2025][VLM Reasoning][Video reasoning] Proposes the VAEX-Bench benchmark to systematically evaluate the "abstract spatiotemporal reasoning" capability of MLLMs for the first time. Unlike extractive tasks that pull information from single frames, abstract reasoning requires integrating observations across rooms and time to infer global spatial layouts, perform cross-scene counting, etc. The study reveals that all SOTA models (including GPT-5.2 and Gemini-3 Pro) perform si…
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "Video reasoning"
  - "Spatiotemporal reasoning"
  - "Abstract reasoning"
  - "Egocentric video"
  - "Benchmark"
date: 2026-05-08
content_hash: df37a7627367cd26
---

# Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence

**Conference**: CVPR 2025  
**arXiv**: [2603.13091](https://arxiv.org/abs/2603.13091)  
**Code**: Coming soon  
**Area**: Multimodal VLM  
**Keywords**: Video reasoning, Spatiotemporal reasoning, Abstract reasoning, Egocentric video, Benchmark

## TL;DR

Proposes the VAEX-Bench benchmark to systematically evaluate the "abstract spatiotemporal reasoning" capability of MLLMs for the first time. Unlike extractive tasks that pull information from single frames, abstract reasoning requires integrating observations across rooms and time to infer global spatial layouts, perform cross-scene counting, etc. The study reveals that all SOTA models (including GPT-5.2 and Gemini-3 Pro) perform significantly worse than humans on abstract reasoning.

## Background & Motivation

**Background**: Existing video spatiotemporal benchmarks (VSI-Bench, VSTI-Bench) primarily test "extractive reasoning," where answers can be directly extracted from single frames or local spatiotemporal events (e.g., object appearance order, relative directions).

**Limitations of Prior Work**: Extractive reasoning fails to evaluate whether models can construct a globally consistent spatial representation. For instance, can a model reconstruct house floorplans from fragmented egocentric observations? Can it count objects across rooms? Can it comprehend the global directional relationships between rooms?

**Key Challenge**: The "abstract spatiotemporal reasoning" capability required for embodied AI (integrating scattered observations and inferring implicit spatial layouts) remains largely unevaluated.

**Goal**: To construct a controllable benchmark to systematically evaluate the abstract versus extractive spatiotemporal reasoning capabilities of MLLMs.

**Key Insight**: Instead of annotating questions on existing videos (where evidence is fixed and designing abstract reasoning questions is difficult), a "query-conditioned video construction" paradigm is proposed—designing the questions first, and then generating the environments and videos.

**Core Idea**: Extending from extractive reasoning to abstract reasoning via one-to-one task pairs, using controllable synthetic environments to expose the spatiotemporal reasoning bottlenecks of MLLMs.

## Method

### Overall Architecture

VAEX-Bench consists of 10 indoor scenes × 10 tasks (5 extractive + 5 abstract) × 3 questions = 300 queries. The core pipeline consists of: scene specification design → query construction → SketchUp modeling → Enscape rendering → egocentric video recording → human verification.

### Key Designs

1. **One-to-One Task Extension from Extractive to Abstract**:

    - Appearance Order → **Memory-Action**: Extended from "object appearance order" to "what activities can be performed in the third room" (requires long-term memory).
    - Relative Direction → **Map Direction**: Extended from single-viewpoint relative directions to "what direction is room3 relative to room4" (requires global localization modeling).
    - Relative Distance → **Map Scale**: Extended from local distance estimation to estimating distances between rooms given a reference distance (requires global metric reasoning).
    - Route Plan → **Simulation**: Extended from local navigation to "which room is directly opposite to the kitchen" (requires floorplan layout reasoning).
    - Object Counting → **Global Counting**: Extended from single-room counting to global counting across all rooms (requires cross-scene aggregation and de-duplication).

2. **Query-Conditioned Video Construction**:

    - Designing questions first to determine the required evidence, then constructing the scene and trajectory accordingly to ensure the evidence is distributed in space and time under a controlled manner.
    - Key constraints: temporal cue separation (decisive evidence is scattered across different segments of the video) + spatial mapping (cannot be solved using local navigation cues).
    - Each video takes approximately 2-3 weeks to produce.

3. **Controllability of the Synthetic Environment**:

    - Interior scenes modeled via SketchUp and rendered using Enscape.
    - Unified evaluation of 14 MLLMs: 32-frame sampling, temperature 0.7, Accuracy@5 (averaging over 5 generation runs).

## Key Experimental Results

### Main Results

| Model | Abstract Avg | Extractive Avg | Memory | Map Direction | Global Counting |
|------|---------|---------|------|---------|---------|
| Human | 81.7% | 88.0% | 89.3% | 83.3% | 82.7% |
| Gemini-3 Flash | **40.3%** | 50.0% | 60.7% | 34.0% | 31.3% |
| GPT-5.2 | 30.1% | 44.5% | 38.0% | 26.0% | 23.3% |
| Qwen3-VL-32B | 29.9% | 45.5% | 40.0% | 26.0% | 17.3% |
| Qwen3-VL-235B | 26.7% | 49.7% | 43.3% | 16.7% | 13.3% |
| Random | 26.5% | 24.8% | 30.7% | 22.0% | N/A |

### Key Findings

- **Huge Gap between Abstract and Extractive Reasoning**: All models perform significantly worse on abstract tasks than on extractive ones. The best-performing model, Gemini-3 Flash, achieves an average of 40.3% on abstract vs. 50.0% on extractive tasks, compared to human performance of 81.7% vs. 88.0%.
- **Model Ranking Reversal**: Gemini-3 Flash significantly outperforms Gemini-3 Pro on abstract tasks (40.3% vs. 29.7%), whereas Pro performs better on extractive tasks. This indicates that short-range recognition capabilities do not necessarily translate to abstract reasoning.
- **Scaling Up Does Not Imply Abstract Reasoning Improvements**: Qwen3-VL-32B/235B do not show significant improvement over the 8B model on abstract tasks (29.9%/26.7% vs. 24.5%).
- **Global Counting as the Main Bottleneck**: All models perform far below humans on Global Counting (13-31% vs. 82.7%), revealing that models fail in de-duplication and cross-scene aggregation.
- **Performance Drop from MCQ to Free-form Generation**: Models rely heavily on options cues, and their performance deteriorates further when options are removed.
- **Human Performance is Also Poor on Map Scale (60%)**: Distance metric reasoning presents a significant challenge for both humans and models.

## Highlights & Insights

- **Clear and Powerful "Extractive vs. Abstract" Dichotomy**: The one-to-one task comparison intuitively highlights where models fail. Rather than simply changing to a "harder question," this approach tests distinct levels of cognitive processing under equivalent semantic intent.
- **Significant Innovation in the Query-Conditioned Pipeline**: Designing the questions prior to generating the videos ensures a precise and controllable distribution of evidence for each query, yielding much higher quality than annotating existing videos.
- **Evaluating Real Capabilities through Synthetic Data**: Although the scenes are synthetic, the evaluated reasoning abilities (such as global spatial modeling and cross-scene aggregation) represent core requirements for embodied intelligence.

## Limitations & Future Work

- **Small Scale**: Contains only 10 scenes and 300 queries, which limits statistical significance.
- **Synthetic-to-Real Gap**: Post-rendered scenes from SketchUp/Enscape exhibit visual discrepancies compared to real-world environments.
- **Extremely High Production Cost**: Requiring 2-3 weeks per video limits its scalability.
- **Insufficiency of Uniform 32-Frame Sampling**: For longer videos (~37s), sampling 32 frames may miss crucial evidence.
- **Limited to Indoor Scenes**: Outdoor or open-world abstract spatiotemporal reasoning remains unexplored.

## Related Work & Insights

- **vs. VSI-Bench**: While VSI-Bench focuses on 3D visual spatial intelligence, it remains extractive. VAEX-Bench systematically introduces the dimension of abstract reasoning.
- **vs. Text-Based Abstract Reasoning (e.g., HotpotQA)**: Multi-hop reasoning in the textual domain has been widely studied, but abstract reasoning in spatiotemporal video is significantly more challenging due to the need to handle partial observability and spatial reasoning.
- **Insight**: The failure of models on global counting exposes a fundamental issue of "entity persistence under partial observability"—models cannot determine whether chairs seen in room 1 and room 3 represent the same physical entity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Systematically defines and evaluates video "abstract spatiotemporal reasoning" for the first time, featuring an elegant task taxonomy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison of 14 models, including human baselines, though dataset scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definitions, intuitive examples, and a systematic categorization system.
- Value: ⭐⭐⭐⭐⭐ Highly instructional for video understanding research in embodied intelligence, exposing fundamental limitations of current models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](../../NeurIPS2025/vlm_reasoning/video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[CVPR 2025\] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](thinking_in_space_how_multimodal_large_language_models_see_remember_and_recall_s.md)
- [\[CVPR 2025\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)
- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](../../CVPR2026/vlm_reasoning/think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)
- [\[CVPR 2026\] Select Less, Reason More: Prioritizing Evidence Purity for Video Reasoning](../../CVPR2026/vlm_reasoning/select_less_reason_more_prioritizing_evidence_purity_for_video_reasoning.md)

</div>

<!-- RELATED:END -->
