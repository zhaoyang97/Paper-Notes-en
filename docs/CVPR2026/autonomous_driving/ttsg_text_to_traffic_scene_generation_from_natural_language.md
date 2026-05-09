---
title: >-
  [Paper Note] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model
description: >-
  [CVPR 2026][Autonomous Driving][Text-to-traffic scene generation] This paper proposes TTSG, a modular framework that leverages LLMs to convert free-form text descriptions into executable traffic scenarios. Through prompt analysis, road retrieval, agent planning, and a novel plan-aware road ranking algorithm, TTSG generates diverse scenes and achieves a minimum average collision rate of 3.5% on SafeBench.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Text-to-traffic scene generation
  - large language model
  - safety-critical scenarios
  - road ranking
date: 2026-05-08
content_hash: 7ac244c8dab2852c
---

# Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model

**Conference**: CVPR 2026
**arXiv**: [2409.09575](https://arxiv.org/abs/2409.09575)
**Code**: [https://basiclab.github.io/TTSG](https://basiclab.github.io/TTSG)
**Area**: Autonomous Driving / Scene Generation
**Keywords**: Text-to-traffic scene generation, large language model, autonomous driving, safety-critical scenarios, road ranking

## TL;DR

This paper proposes TTSG, a modular framework that leverages LLMs to convert free-form text descriptions into executable traffic scenarios. Through prompt analysis, road retrieval, agent planning, and a novel plan-aware road ranking algorithm, TTSG generates diverse scenes and achieves a minimum average collision rate of 3.5% on SafeBench.

## Background & Motivation

1. **State of the Field**: Traffic scene datasets such as nuScenes and Waymo provide rich driving logs but are constrained by safety and controllability. Simulators like CARLA and MetaDrive support customizable scenes yet rely on random sampling or trajectory replay.
2. **Limitations of Prior Work**: Methods such as LCTGen and ChatScene either require structured inputs and cannot handle free-form text, or demand that users manually specify spawn points and map locations, while neglecting environmental conditions (traffic lights, weather, etc.).
3. **Root Cause**: How to generate spatially valid and semantically coherent traffic layouts directly from unstructured natural language, without relying on predefined routes or spawn points.
4. **Paper Goals**: A training-free modular framework that generates realistic traffic scenes directly from natural language.
5. **Starting Point**: Employing the LLM as a general-purpose planner within a controlled pipeline rather than as an end-to-end generator.
6. **Core Idea**: A plan-aware road ranking algorithm ensures consistency between agent actions and road geometry.

## Method

### Overall Architecture

A five-stage pipeline: (1) **Prompt Analysis** — the LLM parses user input into structured elements; (2) **Road Retrieval** — candidate roads are retrieved from a pre-built graph; (3) **Agent Planning** — the LLM plans multi-agent behaviors; (4) **Road Ranking** — road–plan compatibility is evaluated; (5) **Scene Generation** — the layout is rendered into an executable traffic scenario.

### Key Designs

1. **Road Graph Construction and Agent Set**:

    - **Function**: Encodes road network information to support automatic spawn point selection.
    - **Mechanism**: CARLA maps are converted to OpenDRIVE format; features including traffic lights, static objects, intersections, and lane configurations are parsed and organized as a graph structure. The agent set supports nine agent types.
    - **Design Motivation**: The graph structure enables efficient querying of road connectivity, allowing flexible scene generation without predefined geometry.

2. **Plan-Aware Road Ranking Algorithm**:

    - **Function**: Selects the road most compatible with the agent plan from a set of candidates.
    - **Mechanism**: For each candidate road, the algorithm checks whether it satisfies each agent's conditions (turning permissions, road type, length, etc.) and accumulates scores via an indicator function: $r^* = \arg\max_{r \in R_c} \sum_{a \in A} \mathbf{1}_{\{\text{match}(r,a)\}}$. Ties are broken randomly to preserve diversity.
    - **Design Motivation**: Prior methods select roads randomly, ignoring critical factors such as turning permissions and spawn point adequacy.

3. **Prompt Analysis and Sequential Event Support**:

    - **Function**: Decomposes free-form text into structured components and supports multi-stage scenarios.
    - **Mechanism**: The LLM decomposes the input into required signals, objects, and agent configurations. Sequential events are supported through iterative planning, where the final position of one event serves as the starting point of the next.
    - **Design Motivation**: Replaces chain-of-thought (CoT) approaches to substantially reduce token consumption while maintaining comparable planning quality.

### Loss & Training

This is a training-free framework using GPT-4o as the default LLM. Format validation is applied after each stage, with resubmission triggered upon failure.

## Key Experimental Results

### Main Results

| Scenario | Metric | TTSG | ChatScene (Prev. SOTA) | Gain |
|----------|--------|------|------------------------|------|
| Straight Obstacle | Collision Rate↓ | 0.021 | 0.030 | −0.009 |
| Lane Change | Collision Rate↓ | 0.085 | 0.110 | −0.025 |
| Unprotected Left Turn | Collision Rate↓ | 0.000 | 0.100 | −0.100 |
| Average | Collision Rate↓ | 0.035 | 0.080 | −0.045 |

### Ablation Study

| Configuration | Agent Acc | Road Acc | Notes |
|---------------|-----------|----------|-------|
| w/ analysis + CoT | 0.975 | 0.940 | Best performance but higher token cost |
| w/ analysis (default) | 0.925 | 0.875 | Comparable performance, fewer tokens |
| w/o analysis | 0.833 | 0.775 | Significant degradation |
| w/ road ranking | SA = 0.800 | — | Scene accuracy improved |
| w/o road ranking | SA = 0.560 | — | Marked degradation |

### Key Findings

- The road ranking strategy improves scene accuracy from 56% to 80%.
- Training the driving description model yields a CIDEr improvement of over 30 points.
- The open-source model Gemma3-12B can also effectively support the framework.

## Highlights & Insights

- **Training-Free Design**: The entire pipeline requires no model training and relies solely on the LLM's reasoning capability.
- **Simple yet Effective Ranking**: A straightforward indicator-function matching scheme replaces complex optimization procedures.
- **Cross-LLM Generalization**: The framework performs consistently across multiple open-source and closed-source LLMs.

## Limitations & Future Work

- Scenarios requiring precise timing control (e.g., a pedestrian suddenly stepping out) exhibit lower accuracy.
- The framework is currently limited to the CARLA simulator and has not been extended to other platforms.
- Future work plans to extend the framework to the generation of entirely new traffic objects.

## Related Work & Insights

- **vs. ChatScene**: ChatScene requires manual specification of spawn points, whereas TTSG selects them automatically; TTSG additionally supports environmental conditions and sequential events.
- **vs. LCTGen**: LCTGen requires structured inputs, while TTSG handles free-form text.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The plan-aware road ranking algorithm is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ SafeBench validation, multi-LLM generalization, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rich illustrations.
- **Value**: ⭐⭐⭐⭐ High practical value; directly applicable to autonomous driving system testing.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](nord_a_data-efficient_vision-language-action_model_that_drives_without_reasoning.md)
- [\[CVPR 2026\] Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)
- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[CVPR 2026\] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)

<!-- RELATED:END -->
