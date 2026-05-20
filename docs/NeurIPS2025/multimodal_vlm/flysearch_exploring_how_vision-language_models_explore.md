---
title: >-
  [Paper Note] FlySearch: Exploring how vision-language models explore
description: >-
  [NeurIPS 2025 (Datasets and Benchmarks Track)][Multimodal VLM][vision-language models] FlySearch introduces a photorealistic 3D outdoor environment built on Unreal Engine 5 to evaluate the exploration capabilities of VLM…
tags:
  - "NeurIPS 2025 (Datasets and Benchmarks Track)"
  - "Multimodal VLM"
  - "vision-language models"
  - "object navigation"
  - "exploration capability"
  - "3D environments"
  - "UAV"
date: 2026-05-08
content_hash: 72d6d031a682e067
---

# FlySearch: Exploring how vision-language models explore

**Conference**: NeurIPS 2025 (Datasets and Benchmarks Track)
**arXiv**: [2506.02896](https://arxiv.org/abs/2506.02896)  
**Code**: Available (environment, scenes, and codebase publicly released)  
**Area**: Multimodal VLM / Embodied Intelligence / Benchmarking
**Keywords**: vision-language models, object navigation, exploration capability, 3D environments, UAV

## TL;DR

FlySearch introduces a photorealistic 3D outdoor environment built on Unreal Engine 5 to evaluate the exploration capabilities of VLMs. Results reveal that state-of-the-art VLMs fail to reliably complete even simple search tasks, and the performance gap relative to humans widens dramatically as task difficulty increases.

## Background & Motivation

Vision-language models (VLMs) have demonstrated strong performance on tasks such as image captioning and VQA, yet their capacity for active exploration in unstructured real-world environments remains largely unknown. Object navigation (ObjectNav) — requiring an agent to locate a specified target and navigate to it within a simulated environment — is a critical capability for evaluating embodied intelligence. However, existing benchmarks suffer from several limitations:

**Environmental constraints**: Most ObjectNav benchmarks (Habitat, AI2-THOR) focus exclusively on indoor scenes.

**System evaluation vs. model evaluation**: Prior work predominantly integrates VLMs as components within complex systems rather than directly assessing VLMs' intrinsic exploration capabilities.

**Lack of zero-shot openness**: Most approaches rely on task-specific object detectors, making them ill-suited for open-world search scenarios.

**Insufficient visual realism**: Some benchmarks employ simplified rendering pipelines.

FlySearch addresses these gaps by placing VLMs in control of an unmanned aerial vehicle (UAV) tasked with searching for targets across large-scale outdoor scenes, thereby systematically evaluating their capacity to formulate and execute exploration strategies.

## Method

### Overall Architecture

The FlySearch system comprises three core components:

1. **Simulator**: A high-fidelity rendering environment based on Unreal Engine 5, supporting dynamic lighting, wind variation, and procedural generation.
2. **Evaluation Controller**: A Python-based module that manages scene generation, VLM communication, and result aggregation.
3. **Scene Generator**: Procedurally generates an unlimited number of test scenarios.

**Evaluation task procedure**:
- The model receives a text prompt describing the target to be found.
- At each step, the model receives a 500×500-pixel top-down RGB image from the UAV (with a coordinate grid overlay).
- The model outputs a relative displacement command in the form `<action>(X, Y, Z)</action>`.
- Upon locating the target, the model outputs `FOUND` to terminate the search.

**Success criterion**: The difference between the agent altitude $h_{agent}$ and the highest point of the target $h_{object}$ must be $\leq 10\text{m}$, with the target within the field of view.

### Key Designs

**Two evaluation environment types**:
- **Forest environment**: Based on the Unreal "Electric Dreams" demo, with procedurally generated vegetation and rocks.
- **Urban environment**: Based on the "City Sample" demo, representing a modern American city of approximately $4 \times 4$ km.

**Three standardized challenge sets**:

| Challenge | Scenes | Step Limit | Description |
|-----------|--------|------------|-------------|
| FS-1 | 400 | 10 | Basic search; target visible within initial field of view |
| FS-Anomaly-1 | 200 | 10 | Search for "out-of-place" targets (e.g., a giraffe in a city) |
| FS-2 | 200 | 20 | Difficult search; target may be occluded, requiring systematic exploration |

**Search targets**:
- Urban: construction sites, crowds, rubbish piles, fires, vehicles
- Forest: campsites, rubbish piles, people (simulating injured hikers), forest fires, buildings
- Anomalies: UFOs, light aircraft, dinosaurs, tanks, giraffes, etc.

### Loss & Training

FlySearch is an evaluation benchmark. For VLM fine-tuning experiments, the authors apply GRPO (Group Relative Policy Optimization) to train Qwen2.5-VL 7B in the forest environment:

- 6,750 unique scenes and 67,500 training samples are generated.
- The reward function combines reasoning quality (at least 100 tokens of reasoning) and action quality (whether the agent moves closer to the target).
- LoRA fine-tuning is applied with the visual encoder frozen.
- Training is conducted on 4 NVIDIA H100 GPUs over several hours.

## Key Experimental Results

### Main Results

| Model | FS-1 Overall (%) | FS-2 Overall (%) |
|-------|-----------------|-----------------|
| Human (untrained) | 67.0 | — |
| Gemini 2.0 Flash | **42.0** | ~7.0 |
| Claude 3.5 Sonnet | ~37.0 | — |
| GPT-4o | ~36.0 | ~4.0 |
| Pixtral-Large 124B | ~30.0 | — |
| Qwen2-VL 72B | ~15.0 | — |
| LLaVA-OneVision 72B | ~12.0 | — |
| Small models (≤11B) | <4.0 | — |
| Qwen2.5-VL 7B (GRPO fine-tuned) | 21.5 (City) | 0.0 |

FS-Anomaly-1 results:

| Model | FS-Anomaly-1 Overall (%) |
|-------|--------------------------|
| Gemini 2.0 Flash | **35.5** |
| GPT-4o | 27.0 |
| Claude 3.5 Sonnet | 27.5 |
| Pixtral-Large | 15.0 |
| Qwen2-VL 72B | 7.5 |
| Small models (≤11B) | <3.5 |

### Ablation Study

| Configuration | Gemini | Pixtral |
|---------------|--------|---------|
| FS-1 baseline (10 steps) | 42.5 | 30.0 |
| FS-1 reduced to 5 steps | ~38.0 (−10%) | ~25.0 (−17%) |
| FS-1 increased to 20 steps | ~40.0 (−6%) | ~25.0 (−17%) |
| FS-1 with compass-based actions | 17.5 (Forest) | 22.0 (Forest) |
| FS-1 without grid overlay | 31.5 (Forest) | 20.0 (Forest) |
| FS-Anomaly with explicit target type | Significant gain | Significant gain |

### Key Findings

1. **VLMs lag far behind humans**: On FS-1, the best-performing VLM (Gemini, 42%) trails humans (67%) by 37 percentage points; on FS-2, this gap widens to 835%.
2. **Small models are nearly incapable**: Open-source models with ≤11B parameters achieve success rates below 4%, primarily due to failure to correctly format outputs.
3. **More steps can be harmful**: Increasing the step budget from 10 to 20 degrades performance for both Gemini and Pixtral, indicating that VLMs struggle to maintain coherent strategies over longer horizons.
4. **Fine-tuning yields limited gains**: GRPO fine-tuning improves Qwen's FS-1 urban performance from 1.5% to 21.5%, yet FS-2 performance remains at 0%.
5. **Anomaly detection is challenging**: VLMs tend to flag visually salient but normal objects (e.g., yellow taxis) as anomalies while overlooking genuinely anomalous ones (e.g., a nearby tank).
6. **Coordinate grid overlay is critical**: Removing the grid overlay causes a substantial performance drop (−26% in the forest environment), demonstrating that VLMs rely heavily on explicit spatial references.

## Highlights & Insights

1. **Precise diagnosis of VLM failure modes**: The paper systematically categorizes failure causes into visual hallucination, contextual misinterpretation, and task planning failure.
2. **High-quality environment design**: Near-photorealistic rendering via Unreal Engine 5 combined with procedural generation enables unlimited scenario scalability.
3. **Meaningful human baseline**: An online user study establishes a well-grounded human reference for comparison.
4. **Strong reproducibility**: Three standardized challenge sets ensure fair and comparable evaluation conditions.
5. **Reveals a fundamental gap**: VLMs lack systematic exploration strategies — humans search along streets, whereas VLMs move randomly.

## Limitations & Future Work

1. **Pure VLM evaluation only**: More complex ObjectNav systems (e.g., SLAM-integrated methods) are not assessed.
2. **Simple prompting strategy**: Only zero-shot prompting is employed; few-shot learning and prompt optimization are unexplored.
3. **Fixed top-down camera perspective**: The camera always points downward, limiting environmental perception modalities.
4. **Collision issues in urban environments**: Models frequently trigger collision avoidance (particularly in FS-2), reducing search efficiency.
5. **No multi-target or cooperative search**: Only single-agent, single-target search is evaluated.

## Related Work & Insights

- **BALROG**: Evaluates VLMs in game environments but lacks outdoor 3D scenes.
- **VisualAgentBench**: A multi-environment VLM benchmark, but not focused on exploration capability.
- **Habitat / AI2-THOR**: Standard ObjectNav environments, but restricted to indoor settings.
- **AirSim**: A UAV simulator built on Unreal Engine, but based on an older engine version.
- **Insight**: "Knowing where to look" and "knowing how to search systematically" represent two fundamentally distinct capabilities; current VLM architectures may have a structural deficiency in the latter.

## Rating

- **Novelty**: ★★★★★ — First large-scale benchmark for evaluating VLM exploration capability in outdoor 3D environments.
- **Technical Depth**: ★★★★☆ — Carefully designed environment and evaluation framework; primary contribution lies in the benchmarking methodology.
- **Experimental Thoroughness**: ★★★★★ — 9 VLMs, 3 challenge sets, human baselines, and multi-dimensional ablation analysis.
- **Practical Value**: ★★★★☆ — Offers direct guidance for understanding and improving spatial reasoning and exploration in VLMs.
- **Overall Recommendation**: ★★★★☆

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Explore How to Inject Beneficial Noise in MLLMs](../../AAAI2026/multimodal_vlm/explore_how_to_inject_beneficial_noise_in_mllms.md)
- [\[NeurIPS 2025\] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)
- [\[NeurIPS 2025\] GoalLadder: Incremental Goal Discovery with Vision-Language Models](goalladder_incremental_goal_discovery_with_vision-language_models.md)
- [\[NeurIPS 2025\] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](roborefer_towards_spatial_referring_with_reasoning_in_vision-language_models_for.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)

</div>

<!-- RELATED:END -->
