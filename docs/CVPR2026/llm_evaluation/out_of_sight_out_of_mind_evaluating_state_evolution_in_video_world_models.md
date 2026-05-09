---
title: >-
  [Paper Note] Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models
description: >-
  [CVPR 2026][LLM Evaluation][Video World Models] This paper proposes StEvo-Bench, a benchmark comprising 225 tasks across 6 evolution categories, which systematically evaluates whether 9 video world models can decouple state evolution from observation via occlusion or camera-away controls. All models achieve a success rate below 10% under observation interruption, and 5 specialized verifiers are employed to precisely localize failure modes.
tags:
  - CVPR 2026
  - LLM Evaluation
  - Video World Models
  - State Evolution
  - Benchmark
  - Occlusion Control
  - VLM Verifier
date: 2026-05-08
content_hash: cc12cf25e061d74f
---

# Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models

**Conference**: CVPR 2026
**arXiv**: [2603.13215](https://arxiv.org/abs/2603.13215)
**Code**: [Project Page](https://glab-caltech.github.io/STEVOBench/)
**Area**: Video World Model Evaluation
**Keywords**: Video World Models, State Evolution, Benchmark, Occlusion Control, VLM Verifier

## TL;DR

This paper proposes StEvo-Bench, a benchmark comprising 225 tasks across 6 evolution categories, which systematically evaluates whether 9 video world models can decouple state evolution from observation via occlusion or camera-away controls. All models achieve a success rate below 10% under observation interruption, and 5 specialized verifiers are employed to precisely localize failure modes.

## Background & Motivation

**Background**: Video world models (Veo 3, Sora 2 Pro, Genie 3, etc.) simulate the world by generating pixel frames and can already produce visually realistic dynamic scenes. In the real world, physical processes (e.g., water poured into a glass, ice melting) evolve continuously regardless of whether they are observed.

**Limitations of Prior Work**:

1. Physics-correctness benchmarks (VideoPhys, PAIBench) evaluate only fully visible physical processes.
2. Consistency benchmarks (WorldScore, MIND) evaluate only memory of static scenes.
3. No existing benchmark evaluates whether states evolve correctly during unobserved intervals—yet this is the key capability distinguishing a world model from a video generator.

**Key Challenge**: As world models are required to generate larger worlds and support longer interactions, the majority of the generated world is **unobserved** at any given moment. A model that fails to correctly evolve state during unobserved intervals does not deserve to be called a world model.

**Goal**: Design a benchmark that quantitatively evaluates a model's ability to decouple state evolution from observation, and comprehensively diagnose failure modes in current models.

**Key Insight**: Decompose evaluation into two stages—control success (observation control + action control) and evolution success (state progression + physical plausibility + consistency)—and construct 5 specialized binary VLM verifiers for automated assessment.

**Core Idea**: By inserting occlusions or moving the camera away during an ongoing evolution process, test whether video world models can correctly advance physical processes even when they are out of sight.

## Method

### Overall Architecture

StEvo-Bench contains 225 tasks spanning 6 categories of natural evolution processes (continuous processes, kinematics, relational changes, causal changes, state transitions, and expected human/animal behavior). Each task is specified by a triple of (initial image, text prompt, [optional camera trajectory]), where the text prompt encodes both action control (initiating the evolution) and observation control (interrupting observation). Evaluation follows a two-stage, five-verifier pipeline.

### Key Designs

1. **Dual Observation Control Mechanisms Adapted to Model Type**

    - Video generation models (Veo 3, Sora 2 Pro, etc.): text prompts specify in-scene occluders (cardboard/curtain) or light shutoff for a duration, followed by removal.
    - Camera-controllable models (Genie 3, HunyuanWorld, etc.): camera trajectories are specified to move the subject entirely out of frame and then back.
    - **Design Motivation**: Different models expose different control interfaces, yet both mechanisms achieve the core test objective of temporarily hiding the evolving subject from view.

2. **Two-Stage, Five-Verifier Automated Evaluation Protocol**

    - Stage 1 (Control Verification): The observation control verifier checks whether the subject is fully occluded for a sufficient duration; the action control verifier checks whether the evolution process is correctly initiated. Failure on either criterion disqualifies the sample.
    - Stage 2 (Evolution Verification): The state progression verifier (majority-vote ensemble of $n=3$) judges whether state advances during the unobserved interval; the physical plausibility verifier checks for two classes of physical violations (instantaneous violations and causal violations); the consistency verifier applies a structured checklist to assess temporal consistency before and after occlusion.
    - All verifiers are built on Gemini 3.1 Pro, each answering a single narrow yes/no question, with majority voting to suppress hallucinations.
    - **Design Motivation**: Decomposing evaluation into independent specialized verifiers is more reliable than a single comprehensive prompt, and enables precise localization of failure causes.

3. **Verifier Reliability Calibration**

    - On a subset of 180 videos, verifier outputs are compared against annotations from 3 human annotators using accuracy, ROC-AUC, and Model Ranking Agreement (MRA).
    - Verifier–human agreement on the MRA metric **consistently surpasses** human–human agreement across all verifiers.
    - This validates the trustworthiness of the automated evaluation.

### Loss & Training

StEvo-Bench is an evaluation benchmark and does not involve model training.

## Key Experimental Results

### Main Results

| Model | Type | Success Rate (%) | State Progression (%) | Physical Plausibility (%) | Consistency (%) |
|-------|------|-----------------|----------------------|--------------------------|----------------|
| Veo 3 | Video Gen. | 8.7 | 17.4 | 82.6 | 66.5 |
| Sora 2 Pro | Video Gen. | 8.1 | 13.1 | 85.5 | 69.7 |
| WAN 2.2 | Video Gen. | 0.9 | 7.7 | 52.0 | 58.4 |
| HY-Video 1.5 | Video Gen. | 0.9 | 4.1 | 42.1 | 59.1 |
| CogVideoX 1.5 | Video Gen. | 0.5 | 1.4 | 68.5 | 67.1 |
| Genie 3 | Camera Ctrl. | 0.0 | 2.9 | 15.2 | 27.3 |
| HY-WorldPlay | Camera Ctrl. | 0.0 | 0.0 | 72.2 | 88.2 |
| Lingbot | Camera Ctrl. | 0.0 | 3.4 | 40.7 | 76.3 |
| GEN3C | Camera Ctrl. | 0.0 | 0.0 | 30.6 | 82.4 |

### Ablation Study (Effect of Observation Control)

| Condition | State Progression Rate (%) | Task Success Rate (%) |
|-----------|--------------------------|----------------------|
| Full Observation (no occlusion) | 84.6 | 46.2 |
| Observation Control (with occlusion) | 17.4 | 12.4 |

(Averaged over Veo 3 and Sora 2 Pro. Progression rate drops sharply from 84.6% to 17.4% upon introduction of observation control.)

| Verifier Metric | V-H Accuracy | V-H MRA | H-H Accuracy | H-H MRA |
|----------------|-------------|---------|-------------|---------|
| State Progression | 0.795 | 0.829 | 0.747 | 0.807 |
| Physical Plausibility | 0.700 | 0.743 | 0.722 | 0.757 |
| Consistency | 0.860 | 0.905 | 0.911 | 0.919 |
| Observation Control | 0.878 | 0.891 | 0.659 | 0.825 |
| Action Control | 0.903 | 0.949 | 0.885 | 0.868 |

### Key Findings

- **All models achieve <10% success rate**: Neither closed-source frontier models nor open-source models can effectively decouple state evolution from observation.
- **Two dominant failure modes in video generation models**: (1) **Evolution freezing**—the process stalls during occlusion (e.g., a deflating air mattress stops deflating); (2) **Inconsistency**—object appearance changes after occlusion is lifted (e.g., a rectangular sponge becomes round).
- **Strong static bias in camera-controllable models**: All such models exhibit state progression rates below 5%; training data (static 3D scene renderings and indoor/outdoor videos) causes these models to associate camera motion with static scenes.
- **Mutual exclusivity of evolution and camera control**: In the rare cases where dynamic content is generated, models fail to follow the specified camera trajectory.
- **Memory modules provide no remedy**: Memory-augmented architectures such as VMem facilitate appearance memorization but further reinforce static scene bias.
- **Knowledge exists but cannot be leveraged**: A progression rate of 84.6% under full observation demonstrates that models **possess** the relevant physical knowledge, but observation control disrupts their ability to apply it.

## Highlights & Insights

1. **Precise problem formulation**: The capacity of a world model to truly understand the world is operationalized as a quantifiable test of state evolution–observation decoupling.
2. **Specialized verifier design**: Complex multi-dimensional evaluation is decomposed into 5 independent binary questions, with MRA consistently exceeding human–human agreement.
3. **Mechanistic insight**: The drop from 84.6% (full observation) to 17.4% (observation control) indicates that the bottleneck lies not in physical knowledge but in attention architecture—occluding frames carry no state-evolution signal yet are still incorporated into context by bidirectional attention.
4. **Training data bias analysis**: 3D scene renderings and static videos lead camera-controllable models to associate complex motion with static scenes.

## Limitations & Future Work

1. Evaluation relies on a VLM (Gemini 3.1 Pro), whose own limitations may introduce systematic bias.
2. Only the complete occlusion→restoration paradigm is tested; partial or gradual occlusion is not addressed.
3. Although 225 tasks span 6 evolution categories, the number of tasks per category is limited, constraining statistical power.
4. The work is purely evaluative and proposes neither architectural improvements nor constructive solutions.

## Related Work & Insights

- **vs. VideoPhys/WorldScore**: Existing benchmarks focus on physical correctness and consistency but assume full observability throughout. StEvo-Bench introduces a third dimension: correct evolution during unobserved intervals.
- **vs. VMem/WorldMem**: Memory-augmented architectures should theoretically facilitate observation–state decoupling, but experiments on StEvo-Bench show they instead reinforce static scene bias.
- **vs. MIND**: MIND evaluates memory under static settings; StEvo-Bench further requires correct advancement of dynamic processes on top of memory.
- **Implications**: (1) Video world models require selective attention architectures that go beyond all-to-all bidirectional attention; (2) Training data must include more dynamic videos in which objects continue to evolve while occluded; (3) Post-training preference optimization may help eliminate static scene bias.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic evaluation of state evolution–observation decoupling in world models; the problem formulation is highly insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 9 models including frontier systems (Veo 3, Sora 2 Pro); verifier reliability is validated against human annotations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clearly structured; findings are layered, progressing from phenomena to causal analysis to actionable implications.
- **Value**: ⭐⭐⭐⭐⭐ Directly targets a core deficiency of video world models, with important implications for future architectural design and data curation strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Enhancing Out-of-Distribution Detection with Extended Logit Normalization](enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)
- [\[CVPR 2026\] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)
- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)
- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [\[CVPR 2026\] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)

</div>

<!-- RELATED:END -->
