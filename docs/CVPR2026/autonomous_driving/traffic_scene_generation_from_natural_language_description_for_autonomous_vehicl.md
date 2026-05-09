---
title: >-
  [Paper Note] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model
description: >-
  [CVPR 2026][Autonomous Driving][Traffic scene generation] This paper proposes TTSG, a training-free modular framework that generates realistic traffic scenes directly from free-form natural language descriptions. It employs LLM-driven prompt analysis, road retrieval, agent planning, and a plan-aware road ranking algorithm, requiring no predefined routes or spawn points, and achieves a minimum average collision rate of 3.5% on SafeBench.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Traffic scene generation
  - natural language-driven
  - large language model
  - autonomous driving simulation
  - CARLA
date: 2026-05-08
content_hash: 5ca62bda7f3443b9
---

# Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model

**Conference**: CVPR 2026
**arXiv**: [2409.09575](https://arxiv.org/abs/2409.09575)
**Code**: [https://basiclab.github.io/TTSG](https://basiclab.github.io/TTSG)
**Area**: Autonomous Driving / Scene Generation
**Keywords**: Traffic scene generation, natural language-driven, large language model, autonomous driving simulation, CARLA

## TL;DR

This paper proposes TTSG, a training-free modular framework that generates realistic traffic scenes directly from free-form natural language descriptions. It employs LLM-driven prompt analysis, road retrieval, agent planning, and a plan-aware road ranking algorithm, requiring no predefined routes or spawn points, and achieves a minimum average collision rate of 3.5% on SafeBench.

## Background & Motivation

Traffic scene datasets such as nuScenes and Waymo provide rich multimodal driving logs for autonomous driving models, yet real-world data collection is constrained by safety limitations and insufficient controllability. Simulation platforms like CARLA and MetaDrive offer safe and scalable experimental environments; however, existing scene generation approaches exhibit notable shortcomings: random sampling lacks targeted control for systematically evaluating specific failure modes and edge cases, while log-replay methods are restricted to the distribution of collected data and struggle to generate novel scenarios.

Recent instruction-driven simulation methods have enhanced controllability but suffer from three core limitations: (1) approaches such as LCTGen and ProSim rely on structured inputs and cannot handle free-form natural language; (2) ChatScene focuses solely on agent planning and still requires users to manually specify spawn points and map locations; (3) all prior work ignores environmental conditions such as traffic signals, static road objects, and weather.

The root cause of the problem is that users wish to describe complex scenes in natural language (e.g., "a fire truck approaches from the left as the ego vehicle turns right"), yet existing systems lack the ability to ground free-form text into spatially valid, semantically coherent layouts—particularly when composing scenes without predefined locations.

The core idea of TTSG is to design a training-free modular pipeline that embeds an LLM within a strictly controlled pipeline for structured and feasible scene decomposition, while employing a plan-aware road ranking algorithm to ensure consistency between agent actions and road geometry.

## Method

### Overall Architecture

TTSG consists of five stages: (1) **Prompt Analysis**—the LLM decomposes the input text into structured scene elements; (2) **Road Candidate Retrieval**—candidate roads are retrieved from a pre-built road graph based on the analysis results; (3) **Agent Planning**—the LLM determines the type, action, and relative position of each agent; (4) **Road Ranking**—the compatibility between candidate roads and agent plans is evaluated; (5) **Scene Generation**—all information is translated into an executable traffic scene via a custom rendering module.

### Key Designs

1. **Road Graph & Agent Set**

    - **Function**: Constructs a structured database of road information to support automatic spawn point selection.
    - **Mechanism**: Converts CARLA's built-in maps to OpenDRIVE format, parses road features (traffic signals, static objects, intersections, lane configurations, etc.), and organizes them into a graph structure where edges represent road connectivity. Agents are categorized by type (regular vehicles, emergency vehicles, pedestrians, etc.), with random instance selection supported within each category.
    - **Design Motivation**: The graph structure enables efficient querying of road attributes and adjacency relationships (e.g., turnability, road connectivity), providing the infrastructure necessary for automated scene generation.

2. **Prompt Analysis & Plan-Aware Road Ranking**

    - **Function**: Transforms free-form text into a structured scene representation and selects the optimal road.
    - **Mechanism**: In the prompt analysis stage, the LLM decomposes the input into explicit components such as traffic signals, objects, and agent configurations, serving as contextual knowledge for subsequent stages. In the road ranking stage, each candidate road is scored for compatibility with agent plans as $r^* = \arg\max_{r \in R_c} \sum_{a \in A} \mathbf{1}_{\{\text{match}(r,a)\}}$, selecting the road that satisfies the greatest number of agent conditions. When multiple roads achieve the same score, one is chosen at random, ensuring the ranking–random strategy balances alignment and diversity.
    - **Design Motivation**: Direct use of chain-of-thought (CoT) reasoning consumes a large number of tokens; the analysis-based strategy reduces token usage from 1,022 to 682 while maintaining comparable quality. Road ranking addresses the issue of scene–description mismatch that arises from random road selection without ranking.

3. **Agent Planning & Sequential Events**

    - **Function**: Generates detailed multi-agent behavior plans and supports multi-phase event composition.
    - **Mechanism**: The LLM assigns each agent one of eight directional orientations, a specific action (straight, stop, etc.), and a relative distance. Relative positional relationships between agents are supported via a "position" attribute, where lower values indicate a more forward position. Sequential events are realized through iterative planning: the full pipeline first generates an initial event, and the terminal positions are then used as starting points for subsequent events, requiring only re-execution of prompt analysis and agent planning.
    - **Design Motivation**: This supports complex multi-agent interaction scenarios (e.g., "two vehicles block the ego vehicle") and temporally continuous multi-phase scenarios (e.g., "blocked by two vehicles after turning left"), which prior work cannot achieve.

### Loss & Training

TTSG is a fully training-free framework and involves no model training. In downstream applications, TTSG-generated scenes serve as training environments when a soft-actor-critic model is used to train an autonomous driving agent. Format validation is applied after each stage to check the correctness of all keys, types, and values; errors trigger automatic resubmission to the LLM for correction.

## Key Experimental Results

### Main Results

| Method | SO CR↓ | LC CR↓ | ULT CR↓ | Avg. CR↓ | Avg. DS↑ |
|--------|--------|--------|---------|---------|---------|
| Learning-to-Collide | 0.120 | 0.510 | 0.000 | 0.210 | 0.822 |
| AdvSim | 0.230 | 0.430 | 0.050 | 0.270 | 0.796 |
| Adversarial-Trajectory | 0.140 | 0.300 | 0.000 | 0.150 | 0.867 |
| ChatScene | 0.030 | 0.110 | 0.100 | 0.080 | 0.905 |
| **TTSG (Ours)** | **0.021** | **0.085** | **0.000** | **0.035** | **0.914** |

### Ablation Study

| Configuration | AA (Avg)↑ | RA (Avg)↑ | Notes |
|---------------|-----------|-----------|-------|
| w/o analysis | 0.833 | 0.775 | Without analysis stage |
| w/ analysis | 0.925 | 0.875 | With analysis stage |
| w/ analysis + CoT | **0.975** | **0.940** | Analysis + CoT hybrid |
| w/o ranking (SA) | 0.560 | — | Without road ranking |
| w/ ranking (SA) | **0.800** | — | With road ranking |

### Key Findings

- **Road ranking contributes the most**: Scene accuracy improves from 0.560 to 0.800 (+43%), demonstrating that agent–road alignment is critical to scene quality.
- **Analysis strategy is efficient and composable**: The analysis stage achieves quality approaching CoT while reducing token usage by 33%, and can be further improved when combined with CoT.
- **Strong cross-LLM generalizability**: The framework operates effectively across models ranging from the lightweight open-source Gemma3-12B to Claude-Sonnet-3.5, with Claude achieving near-perfect planning accuracy.
- **Driving description enhancement**: Fine-tuning on only 20 key scenes raises the CIDEr score for reasoning from 18.4 to 51.9 (+33.5 points).

## Highlights & Insights

- **Training-free end-to-end scene generation** is the primary contribution—no model training is required whatsoever; the framework generates executable CARLA scenes from natural language purely through LLM inference and a structured pipeline. Embedding the LLM within a constrained pipeline effectively mitigates hallucination.
- **The plan-aware ranking strategy** is simple yet highly effective—a straightforward match-counting metric raises scene accuracy from 56% to 80%, indicating that the key challenge lies not in algorithmic complexity but in correct problem decomposition.
- **The sequential event composition mechanism** is transferable to domains requiring multi-phase coherent scenarios, such as robot task planning and game scene design.

## Limitations & Future Work

- The framework relies entirely on the LLM's language comprehension, and lacks robust error recovery mechanisms for ambiguous or contradictory descriptions.
- Agent behavior patterns are relatively simple (7 basic actions) and cannot express continuous complex driving behaviors such as gradual lane changes.
- Validation is limited to CARLA; generalizability to other simulators (e.g., MetaDrive) has not been verified.
- The road graph is statically pre-built and cannot dynamically create new road layouts or traffic infrastructure.

## Related Work & Insights

- **vs. ChatScene**: ChatScene requires manual specification of spawn points and initial positions, whereas TTSG is fully automated; the collision rate is reduced from 0.08 to 0.035.
- **vs. LCTGen**: LCTGen depends on structured input formats, while TTSG supports fully free-form natural language.
- **vs. CTG++**: CTG++ uses an LLM to generate code-level loss functions to guide a diffusion model, which is more complex but less flexible than TTSG's modular design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The plan-aware road ranking and training-free modular pipeline are novel designs, though the core component (LLM-based planning) is not a breakthrough contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation covers SafeBench benchmarking, ablation studies, multi-LLM comparisons, and diversity testing.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich illustrations and clear explanations of each pipeline stage.
- **Value**: ⭐⭐⭐⭐ Offers practical value for autonomous driving scene generation; the training-free nature lowers the barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](nord_a_data-efficient_vision-language-action_model_that_drives_without_reasoning.md)
- [\[CVPR 2026\] Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)
- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[CVPR 2026\] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)

</div>

<!-- RELATED:END -->
