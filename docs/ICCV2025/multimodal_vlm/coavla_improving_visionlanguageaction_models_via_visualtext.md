---
title: >-
  [Paper Note] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance
description: >-
  [ICCV 2025][Multimodal VLM][VLA] This paper proposes the Chain-of-Affordance (CoA-VLA) framework, which injects four categories of robot affordances (object, grasp, spatial…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "VLA"
  - "chain-of-affordance"
  - "robotic manipulation"
  - "visual prompting"
  - "affordance reasoning"
date: 2026-05-08
content_hash: eda858958fb74461
---

# CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance

**Conference**: ICCV 2025
**arXiv**: [2412.20451](https://arxiv.org/abs/2412.20451)
**Code**: [https://chain-of-affordance.github.io](https://chain-of-affordance.github.io)
**Area**: Multimodal VLM / Embodied Intelligence / Robotic Manipulation
**Keywords**: VLA, chain-of-affordance, robotic manipulation, visual prompting, affordance reasoning

## TL;DR
This paper proposes the Chain-of-Affordance (CoA-VLA) framework, which injects four categories of robot affordances (object, grasp, spatial, and movement) into the policy network of a VLA model in both textual and visual modalities. The approach achieves an 85.54% success rate on a real-robot multi-task benchmark spanning 7 tasks, outperforming OpenVLA by 30.65%, and demonstrates generalization to unseen object poses and obstacles.

## Background & Motivation
**Background**: VLA models have acquired strong generalization capabilities through large-scale pretraining. However, existing approaches either rely on LLMs/VLMs for high-level planning (external reasoning) or predict actions end-to-end without intermediate reasoning. OpenAI O1 has demonstrated that long-chain reasoning can substantially improve performance on complex problems.

**Limitations of Prior Work**: Current VLA models lack self-driven intermediate reasoning in complex environments, leading to failures in tasks requiring precise grasping, spatial reasoning, and obstacle avoidance. Existing reasoning methods such as ECoT focus on task decomposition but lack a structured understanding of physical interaction.

**Key Challenge**: Executing complex manipulation requires answering a sequence of questions—what to manipulate, how to grasp it, where to place it, and how to move there—yet existing VLAs do not explicitly model these intermediate reasoning steps.

**Goal**: Design a structured affordance reasoning chain that enables VLA models to reason over four task-relevant affordance categories before predicting actions, and inject the resulting representations into the policy network.

**Key Insight**: Starting from the classical concept of robot affordances, this work formalizes them as a chain-of-thought (CoT) reasoning sequence and represents them innovatively in both textual and visual modalities.

**Core Idea**: Construct a reasoning chain from four affordance types (object/grasp/spatial/movement) represented in both textual and visual formats, then inject them into the diffusion policy head of a VLA model to guide action generation.

## Method

### Overall Architecture
Built upon DiffusionVLA (Qwen2-VL + diffusion action head), the model takes robot observation images and task instructions as input. It autoregressively generates an affordance reasoning chain in textual form while simultaneously producing corresponding visual affordance annotations overlaid on the observation image. Both modalities are fused through a co-injection module and fed into the diffusion policy network to generate continuous actions.

### Key Designs

1. **Four Affordance Categories**:

    - **Object affordance**: Identifies the target object and its location in the image (via bounding box), addressing the questions of *what* to manipulate and *where* it is.
    - **Grasp affordance**: Determines the most suitable part of the object for grasping (represented as 2D keypoints), such as the handle of a teapot.
    - **Spatial affordance**: Identifies available space in the environment (e.g., empty regions on a plate) for determining placement locations.
    - **Movement affordance**: Plans collision-free motion trajectories to ensure safe robot movement.
    - The four categories form a sequential chain: identify what to manipulate → how to grasp → where to place → how to move. This chained structure enforces explicit ordering dependencies in the reasoning process.

2. **Visual-Textual Dual-Modality Affordance Representation**:

    - **Textual affordance**: Describes each affordance category in natural language along with coordinate information (e.g., bounding box coordinates, keypoint positions). ChatGPT is used to linguistically diversify affordance descriptions, preventing template-induced bias.
    - **Visual affordance**: Overlays affordance information directly onto the observation image—bounding boxes and grasp points are rendered with high-contrast annotations, while motion trajectories are drawn as low-salience thin lines. This hierarchical visual encoding allows the model to distinguish different affordance types at a glance.
    - **Design Motivation**: Textual affordances provide semantically rich reasoning signals, while visual affordances offer spatially aligned perceptual cues. The two modalities are complementary; ablations show that using either alone is inferior to their combination.

3. **Visual-Textual Co-Injection Module**:

    - **Function**: Unifies and fuses textual and visual affordances before injecting them into the diffusion policy network.
    - **Mechanism**: Textual affordances are projected into a token sequence via the VLM's last-layer embeddings and an MLP; visual affordances are encoded as patch tokens via a pretrained ViT-Small. The two token sets are fused through a 2-layer Transformer block for cross-modal integration, and then injected into the diffusion model via FiLM conditioning layers.
    - **Design Motivation**: FiLM layers dynamically modulate the diffusion process, enabling the policy to generate actions that jointly consider spatial constraints and semantic intent without altering the overall diffusion framework.

4. **Dynamic Affordance Selection Mechanism**:

    - **Function**: Adaptively selects which affordance types are needed based on task progress and robot state, avoiding redundant computation.
    - **Mechanism**: Proprioceptive information (e.g., joint angles) is encoded as a single token prepended to the visual token sequence, allowing the model to learn to select relevant affordances at each timestep. For example, when the gripper is closed and the wrist camera observes the object, the model skips object and grasp affordances and generates only spatial and movement affordances.
    - **Effect**: Inference runs at 6 Hz (vs. 1 Hz without dynamic selection), and removing dynamic selection actually degrades accuracy, indicating that redundant affordances introduce noise rather than useful information.

### Data Generation Pipeline
GPT-4o is used to generate scene descriptions and entity recognition → Grounding DINOv2 + SAM to produce bounding boxes → RoboPoint + GPT-4o to predict and cluster spatial affordance points → CoTracker to track end-effector trajectories for movement affordances. The entire pipeline is automated, substantially reducing manual annotation requirements.

## Key Experimental Results

### Main Results (Real Robot, 7-Task Multi-Task Learning)

| Model | In-Distribution Avg | Visual Generalization Avg |
|------|-------------------|--------------------------|
| Diffusion Policy | 33/77 (42.93%) | 3/63 (4.76%) |
| Octo | 34/77 (44.13%) | 12/63 (19.05%) |
| OpenVLA | 52/77 (54.89%) | 14/63 (22.22%) |
| DiffusionVLA | 59/77 (76.60%) | 28/63 (44.44%) |
| **CoA-VLA** | **64/77 (85.54%)** | **36/63 (57.14%)** |

LIBERO Simulation Benchmark:

| Model | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.6 | 76.5 |
| **CoA-VLA** | **88.0** | **90.4** | **82.0** | **59.0** | **79.8** |

### Ablation Study

| Configuration | LIBERO Avg | Notes |
|------|-----------|------|
| CoA-VLA (Full) | 79.8 | Full model |
| w/o visual affordance | Decreased | Removes visual affordance |
| w/o textual affordance | Decreased more | Textual affordance contributes more |
| w/o dynamic selection | Decreased + 6× slower | Redundant affordances introduce noise |

### Key Findings
- Textual affordances contribute more than visual affordances, as language encodes richer task semantics.
- Removing dynamic selection degrades performance, confirming that redundant affordances act as noise rather than providing information gain.
- CoA-VLA shows a larger advantage in visual generalization scenarios (57.14% vs. DiffusionVLA's 44.44%), demonstrating that affordance reasoning substantially enhances robustness to visual variation.
- Spatial affordances enable the robot to identify empty regions on a plate for accurate object placement; movement affordances enable obstacle avoidance.
- The model generalizes to unseen object poses (e.g., teapot handles facing different directions), but fails when objects are placed completely horizontally.

## Highlights & Insights
- **Affordances as Structured CoT**: Bridging the classical affordance concept with modern chain-of-thought reasoning is a natural yet elegant combination. Affordances inherently answer the question of "how an object can be acted upon," making them well-suited as intermediate reasoning steps before action prediction.
- **Dual-Modality Injection**: Affordances are not only verbalized (textual) but also visualized (overlaid on images), with both modalities injected into the diffusion head via FiLM conditioning. This design is transferable to any diffusion-based VLA.
- **Dynamic Selection Reduces Redundancy**: More information is not always better—redundant affordances introduce noise. Leveraging proprioception for simple selection achieves a 6× speedup without accuracy loss.

## Limitations & Future Work
- The pipeline depends on multiple external models (GPT-4o, Grounding DINO, SAM, CoTracker) for affordance data generation, resulting in considerable complexity.
- Validation is limited to a Franka single-arm 7-task setup and LIBERO, constraining task diversity and scale.
- Dynamic affordance selection relies on simple proprioceptive heuristics; more sophisticated strategies (e.g., attention-based selection) may yield further improvements.
- Grasping fails when objects are placed completely horizontally, indicating that affordance prediction remains challenging under extreme pose configurations.

## Related Work & Insights
- **vs. ECoT**: ECoT reasons via task decomposition, sub-task descriptions, and motion instructions, whereas this work uses four affordance categories that are more closely grounded in the nature of physical interaction.
- **vs. CoT-VLA**: CoT-VLA generates sub-goals to guide an autoregressive VLA; this work generates finer-grained affordance information and injects it into a diffusion head.
- **vs. TraceVLA**: TraceVLA uses visual trajectories as additional input, whereas this work treats visual affordances not only as input but also as intermediate products of the reasoning process.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of four affordance categories, dual-modality representation, and co-injection is relatively novel, though no single idea is entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers real-robot 7-task evaluation, LIBERO simulation, generalization experiments, and ablations—fairly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clear affordance definitions and detailed method descriptions.
- Value: ⭐⭐⭐⭐ Provides a practical direction for introducing structured reasoning into VLAs, with meaningful implications for the embodied intelligence community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](advancing_textual_prompt_learning_with_anchored_attributes.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[ICCV 2025\] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models](instruction-grounded_visual_projectors_for_continual_learning_of_generative_visi.md)

</div>

<!-- RELATED:END -->
