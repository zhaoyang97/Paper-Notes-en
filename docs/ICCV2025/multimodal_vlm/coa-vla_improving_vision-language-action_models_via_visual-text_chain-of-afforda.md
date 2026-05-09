---
title: >-
  [Paper Note] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance
description: >-
  [ICCV 2025][Multimodal VLM][VLA] This paper proposes CoA-VLA, which organizes four categories of robotic affordances (object, grasp, spatial, and motion) into a chain-of-thought reasoning process, and injects them into a diffusion policy network via a visual-textual co-injection module, significantly improving the accuracy and generalization of VLA models in multi-task manipulation.
tags:
  - ICCV 2025
  - Multimodal VLM
  - VLA
  - Chain-of-Affordance
  - robot manipulation
  - reasoning enhancement
  - diffusion policy
date: 2026-05-08
content_hash: b236ee7fc7b97706
---

# CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance

**Conference**: ICCV 2025
**arXiv**: [2412.20451](https://arxiv.org/abs/2412.20451)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: VLA, Chain-of-Affordance, robot manipulation, reasoning enhancement, diffusion policy

## TL;DR

This paper proposes CoA-VLA, which organizes four categories of robotic affordances (object, grasp, spatial, and motion) into a chain-of-thought reasoning process, and injects them into a diffusion policy network via a visual-textual co-injection module, significantly improving the accuracy and generalization of VLA models in multi-task manipulation.

## Background & Motivation

Vision-Language-Action (VLA) models have become the dominant paradigm for robot policy learning. However, existing approaches generally rely on external LLMs/VLMs for high-level planning, leaving the models themselves without autonomous reasoning capabilities. Inspired by OpenAI's O1 model, which improves complex problem-solving through reasoning chains, the authors raise a central question: **Can robot models guide action prediction by reviewing prior observations and generating task-relevant reasoning?**

Key limitations of existing VLAs include:
- End-to-end policy learning lacks intermediate reasoning, making it difficult to handle complex environments.
- Models lack autonomous judgment for locating and interacting with objects under ambiguous instructions (e.g., "pour a drink").
- Generalization is insufficient when facing visual distractors and obstacles.

## Method

### Overall Architecture

CoA-VLA builds upon DiffusionVLA (Qwen2-VL + diffusion policy head). The core idea is to introduce chain-of-affordance reasoning prior to action prediction, injecting the reasoning results into the policy network in both textual and visual formats. The overall pipeline is: observation + instruction → VLM generates affordance reasoning → visual-textual co-injection → diffusion policy generates action.

### Key Designs

1. **Four Affordance Categories (Chain-of-Affordance)**:

    - **Object Affordance $z_{obj}$**: Identifies the target object and its location in the visual field (semantic localization + 2D bounding box), resolving the "what to manipulate" ambiguity under vague instructions.
    - **Grasp Affordance $z_{grasp}$**: Determines the most suitable part of the object for grasping, represented as 2D keypoints, ensuring stable and safe grasps.
    - **Spatial Affordance $z_{spat}$**: Identifies spatial coordinate sets satisfying language-specified relational descriptions (e.g., free placement regions), represented as discrete 2D coordinates of feasible interaction areas.
    - **Motion Affordance $z_{move}$**: Plans collision-free motion trajectories, providing the robot with dynamically adaptive paths.

   The four affordance categories form a sequential dependency chain: first identify "what and where to manipulate" → then determine "how to grasp" → then "where to place" → finally "how to move." The learning objective is an intermediate language output mapping $z: \mathcal{O} \times \mathcal{G} \rightarrow \mathcal{Z}$, and action generation is conditioned as $a \sim p(a|\tau, g, z)$.

2. **Visual-Textual Co-Injection Module**:

    - **Textual Affordance**: Encodes affordance information in natural language (e.g., bounding box coordinates, placement region descriptions), tokenized via the last-layer embeddings of the VLM followed by an MLP. To avoid fixed-template bias, ChatGPT is used to paraphrase descriptions with diverse rewrites.
    - **Visual Affordance**: Overlays pixel-aligned visual annotations (bounding boxes, interaction points, motion trajectories) onto historical observation frames, encoded as patch tokens via a pretrained ViT-Small. Motion trajectories are rendered with thin lines at low saliency, while key interaction points use high-contrast semi-transparent overlays.
    - **Fusion Mechanism**: Textual and visual tokens are processed by two standard Transformer blocks, then injected into the diffusion model via FiLM conditioning layers to dynamically modulate the denoising process. The FiLM layer acts as a bottleneck, extracting only the most salient affordance cues.

3. **Dynamic Affordance Selection**:

    - In practice, not all affordances need to be generated at every step (e.g., object and grasp affordances are unnecessary once an object is grasped).
    - Proprioceptive signals (e.g., joint angles) are converted into a single token concatenated with visual tokens, allowing the model to adaptively select affordances needed at each timestep.
    - After training on the large-scale Droid dataset, the model learns to intelligently select affordances based on state — for instance, automatically skipping object/grasp affordances when the gripper is partially closed and the wrist camera detects the object.

### Data Generation Pipeline

To prevent overfitting to limited affordance diversity, an automated data generation pipeline is designed:
- GPT-4o generates scene descriptions and identifies entities.
- Grounding DINOv2 + SAM jointly generate object bounding boxes (refined via IoU alignment).
- RoboPoint + GPT-4o jointly predict spatial affordance points and perform clustering.
- CoTracker tracks robot gripper motion trajectories.

### Loss & Training

During pretraining, the Droid dataset (39K trajectories after filtering unannotated samples) is used to generate synthetic chain-of-affordance data. Fine-tuning is performed on 692 trajectories from 7 real-world tasks. The learning rate is 2e-5; VLM parameters are frozen with LoRA fine-tuning applied. The fine-tuning stage uses a learning rate of 2e-6 with cosine decay.

## Key Experimental Results

### Main Results: Real Robot (Franka, 7-Task Multi-Task Learning)

| Model | In-Distribution Avg. Success Rate | Visual Generalization Avg. Success Rate |
|------|--------------------------|-------------------------------|
| Diffusion Policy | 33/77 (42.93%) | 3/63 (4.76%) |
| Octo | 34/77 (44.13%) | 12/63 (19.05%) |
| OpenVLA | 52/77 (54.89%) | 14/63 (22.22%) |
| DiffusionVLA | 59/77 (76.60%) | 28/63 (44.44%) |
| **CoA-VLA** | **64/77 (85.54%)** | **36/63 (57.14%)** |

### Simulation Results: LIBERO Benchmark

| Method | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| Diffusion Policy | 78.3% | 92.5% | 68.3% | 50.5% | 72.4% |
| Octo | 78.9% | 85.7% | 84.6% | 51.1% | 75.1% |
| OpenVLA | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |
| **CoA-VLA** | **85.3%** | **93.1%** | **85.8%** | **55.0%** | **79.8%** |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| CoA-VLA (full) | 79.8% (LIBERO avg.) | All affordances + co-injection |
| Text affordance only | Performance drops | Lacks spatially aligned visual cues |
| Visual affordance only | Performance drops | Lacks semantic reasoning capability |
| Without dynamic selection | Slower inference, slight performance drop | Redundant affordances increase compute |

### Key Findings

- In visual generalization tests, CoA-VLA outperforms OpenVLA by 34.92%, demonstrating that affordance reasoning is critical for robustness to environmental variation.
- Spatial affordance enables successful bread placement across all three spatial configurations in the PlaceBread task, whereas OpenVLA and DiffusionVLA each succeed in only one.
- Motion affordance enables the robot to successfully complete all obstacle avoidance scenarios (vase detour, tabletop obstacle navigation).

## Highlights & Insights

- The unification of affordance concepts with chain-of-thought reasoning is elegant; the four affordance categories cover the complete reasoning chain of "what, where, how to grasp, where to place, and how to move" in manipulation tasks.
- The visual-textual dual-modality injection mechanism effectively fuses pixel-level spatial information with semantic reasoning, and the FiLM conditioning layer design is efficient.
- Dynamic affordance selection adaptively prunes reasoning steps based on proprioceptive signals, reducing inference overhead while maintaining performance.

## Limitations & Future Work

- The training data generation pipeline relies on multiple external tools (GPT-4o, Grounding DINO, SAM, CoTracker), making it complex and costly.
- The four affordance categories are manually designed and may not cover all manipulation scenarios.
- Real-robot experiments are conducted only on a Franka single-arm platform, with no evaluation on dexterous hands or bimanual setups.
- The fine-tuning dataset is small (692 trajectories); performance at scale remains to be validated.

## Related Work & Insights

- **ECoT** and **CoT-VLA** also explore reasoning in VLAs, but the former focuses on task decomposition and the latter generates subgoals, whereas CoA-VLA unifies reasoning through the affordance framework.
- **TraceVLA** introduces visual traces to enhance spatiotemporal awareness in VLAs; the visual affordance concept in CoA-VLA is complementary to this approach.
- The automated affordance data generation pipeline is generalizable to other robot learning scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The framework combining four affordance categories with chain-of-thought reasoning is novel, and the visual-textual co-injection module offers meaningful engineering contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both real robots and simulation, with broad coverage across 7 real-world tasks.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed figures and intuitive affordance visualizations.
- Value: ⭐⭐⭐⭐ Presents a viable path for integrating reasoning capabilities into VLAs, with meaningful implications for the design of robotic foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[NeurIPS 2025\] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](../../NeurIPS2025/multimodal_vlm/forcevla_enhancing_vla_models_with_a_force-aware_moe_for_contact-rich_manipulati.md)
- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](advancing_textual_prompt_learning_with_anchored_attributes.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
