---
title: >-
  [Paper Note] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Simulation-based reasoning] SIMPACT proposes a test-time simulation-augmented action planning framework that automatically constructs a physics simulation environment from a single RGB-D image…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Simulation-based reasoning"
  - "vision-language models"
  - "action planning"
  - "physical reasoning"
  - "robotic manipulation"
date: 2026-05-08
content_hash: 695a2ceec6d18682
---

# SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2512.05955](https://arxiv.org/abs/2512.05955)
**Code**: None (coming soon)
**Area**: Multimodal VLM / Robotic Manipulation
**Keywords**: Simulation-based reasoning, vision-language models, action planning, physical reasoning, robotic manipulation

## TL;DR

SIMPACT proposes a test-time simulation-augmented action planning framework that automatically constructs a physics simulation environment from a single RGB-D image, enabling VLMs to propose actions, observe simulation outcomes, and iteratively refine their reasoning—achieving SOTA performance on both rigid and deformable object manipulation tasks without any additional training.

## Background & Motivation

**Background**: Vision-language models (VLMs) such as GPT-4V and Gemini have demonstrated remarkable commonsense reasoning and semantic understanding capabilities, and have been widely explored for robot task planning. However, VLMs are trained on static image-text pairs from the internet, which contain no causal interactions or action-conditioned state transitions.

**Limitations of Prior Work**: (1) VLMs lack a deep understanding of physical dynamics—they do not know "what happens when an object is pushed" or "how different force magnitudes affect outcomes"; (2) existing VLM-based robot methods typically prompt models to directly output action parameters, without any physical verification mechanism; (3) enabling VLMs to "understand" the physical world without training new models remains an open problem.

**Key Challenge**: VLMs possess strong semantic reasoning capabilities but lack understanding of physical dynamics, fundamentally because internet-scale data contains no causal chains of the form "action → consequence."

**Goal**: To augment VLMs with physical reasoning capability at test time—without additional training—enabling them to plan robotic manipulation tasks that require fine-grained physical understanding.

**Key Insight**: The authors observe that physics simulators (e.g., PyBullet, MuJoCo) can provide accurate physical predictions. If a simulator can be embedded as a "world model" within the VLM's reasoning loop at test time, it can compensate for the VLM's lack of physical understanding.

**Core Idea**: Embed a physics simulation loop within VLM inference—the VLM proposes an action → the simulator executes it → the VLM observes the simulation outcome → the VLM iteratively refines its reasoning—realizing "simulation as world model" for physics-augmented inference.

## Method

### Overall Architecture

SIMPACT consists of three stages: (1) **Simulation Construction**: automatically building a physics simulation environment from a single RGB-D image; (2) **Action Sampling & Optimization**: the VLM proposes candidate actions based on a language task description, executes them in simulation, observes the results, and iteratively refines the action parameters; (3) **Real-World Execution**: transferring the optimized action sequence from simulation to a real robot. No additional VLM training is required throughout this process.

### Key Designs

1. **Automatic Simulation Construction**:

    - Function: Automatically creates an interactive physics simulation environment from a single RGB-D image.
    - Mechanism: Given an RGB-D image and a language task description, the pipeline automatically performs the following steps—(a) leverages depth information and segmentation models to identify objects in the scene; (b) generates mesh models for rigid objects and places them in a simulator (e.g., PyBullet); (c) applies particle-based simulation (e.g., DiffSim) for deformable objects such as ropes and clay; (d) prompts the VLM to infer physical parameters (mass, friction coefficients, etc.) for each object. The result is an interactive simulation environment corresponding to the real scene.
    - Design Motivation: Physics simulation requires 3D models and physical parameters; although VLM estimates of these quantities are imprecise, they are sufficient to support reasonable physical predictions. Constructing simulations from a single image greatly reduces the reliance on expensive equipment such as 3D scanners.

2. **VLM-based Action Sampling & Optimization**:

    - Function: Leverages VLMs to propose, evaluate, and refine robot actions.
    - Mechanism: The VLM first proposes a set of candidate actions (including parameters such as push direction, force magnitude, and contact point) based on the task description and scene understanding. Each candidate action is executed in simulation, generating rollout videos or keyframe images. The VLM observes these simulation results, assesses which actions are closer to the goal, and proposes improved candidates accordingly. This "propose → simulate → evaluate → refine" loop iterates until a satisfactory action plan is found.
    - Design Motivation: The VLM's commonsense reasoning enables it to propose reasonable initial action hypotheses, while the simulator provides accurate physical verification. The combination yields action planning that is both semantically guided and physically grounded.

3. **Rigid-Deformable Dual-Mode Simulation**:

    - Function: Supports physics simulation of both rigid and deformable objects.
    - Mechanism: The simulation mode is automatically selected based on object type—rigid objects use mesh-based simulation with collision detection and rigid-body dynamics to model pushing, pulling, and collision interactions; deformable objects (ropes, dough, etc.) use particle-based simulation to model stretching, deformation, and cutting behaviors. The VLM is responsible for identifying object types and inferring the corresponding physical parameters.
    - Design Motivation: Real-world robot tasks frequently involve mixed manipulation of rigid and deformable objects; a single simulation mode cannot cover all scenarios.

### Loss & Training

SIMPACT is a purely inference-time framework with no model training or fine-tuning:

- **No loss function**: VLM weights are frozen; reasoning is performed at test time via in-context learning.
- **Action optimization criterion**: The VLM judges action quality (i.e., proximity to the goal state) based on visual results of simulation rollouts, constituting an implicit optimization in which the VLM's semantic judgment serves as the evaluation function.
- **Iterative strategy**: Typically 3–5 iterations are performed; in each round, $N$ candidate actions are proposed and simulated, the best is selected, and sampling continues within its neighborhood.

## Key Experimental Results

### Main Results

| Task | SIMPACT | RT-2 | Code-as-Policies | VoxPoser | Notes |
|------|---------|------|------------------|----------|-------|
| Rigid-body pushing to target position | Best | Poor | Medium | Medium | Fine-grained force control |
| Object sorting/rearrangement | Best | Fair | Good | Fair | Multi-object planning |
| Rope manipulation | Best | Fails | Fails | Poor | Deformable object |
| Clay shaping | Best | Fails | Fails | Fails | High-difficulty deformation |
| Multi-object collision prediction | Best | Poor | Poor | Fair | Contact dynamics |

### Ablation Study

| Configuration | Avg. Success Rate | Notes |
|---------------|------------------|-------|
| Full SIMPACT | Best | Simulation optimization + iterative refinement |
| w/o Simulation (direct VLM) | Significant drop | Direct VLM action output lacks physical verification |
| w/o Iterative Refinement | Notable drop | Single-round sampling without fine-grained tuning |
| Random Physics Params | Slight drop | Accuracy of physical parameters has some impact |
| Simulation w/ 1 round only | Below multi-round | Iterative refinement yields clear improvement |

### Key Findings

- The physics predictions provided by the simulation loop are the single largest contributor to performance gains—removing simulation causes the VLM to largely fail on tasks requiring fine-grained force control.
- Deformable object manipulation (ropes, clay) represents a blind spot for conventional methods; SIMPACT demonstrates for the first time the feasibility of VLM-based planning on such tasks via particle simulation.
- Even when simulator physical parameters are not fully accurate (being VLM estimates), simulation feedback still substantially outperforms the no-simulation baseline—indicating that "coarse but directionally correct physical prediction" is far superior to "no physical prediction."
- The system exhibits strong robustness to variations in object appearance (different colors and shapes) and the presence of distractor objects.

## Highlights & Insights

- **The elegant "simulation as world model" paradigm**: Rather than modifying the VLM or training a new model, SIMPACT equips the VLM at test time with a physics simulator as a "physical engine in the mind." This paradigm is generalizable to any reasoning task requiring physical understanding.
- **Automatic simulation construction from a single RGB-D image**: This dramatically lowers the barrier to simulation construction and enables rapid deployment in novel scenes. Although simulation fidelity is limited, "having a simulation" is far better than "having none."
- **Unified framework for rigid and deformable manipulation**: The ability to simultaneously handle rigid and deformable object manipulation is achieved for the first time among VLM-based robot planning methods.

## Limitations & Future Work

- Simulation construction relies on depth information and segmentation models, which may be unreliable in outdoor environments or scenes with high depth noise.
- The accuracy of VLM-estimated physical parameters (mass, friction, etc.) is limited, which may degrade performance on tasks that are highly sensitive to physical parameters.
- A sim-to-real gap persists, particularly in the simulation fidelity for deformable objects.
- Constructing the simulation environment and running multiple rollout rounds per inference leads to high inference latency.
- The current method is limited to tabletop-level manipulation; extension to more complex long-horizon tasks (e.g., cooking, assembly) requires further investigation.

## Related Work & Insights

- **vs. Code-as-Policies**: CaP prompts LLMs to directly output robot control code but lacks physical verification. SIMPACT provides the VLM with a physical "sandbox" via simulation to predict action consequences.
- **vs. VoxPoser**: VoxPoser uses VLMs to generate value functions for planning guidance but does not perform explicit physics simulation. SIMPACT's simulation yields more accurate physical predictions.
- **vs. end-to-end methods (RT-2, Octo, etc.)**: These methods require large amounts of robot demonstration data for training, whereas SIMPACT relies solely on a pretrained VLM combined with simulation, requiring no additional training data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of simulation-augmented VLM reasoning is highly novel and elegant, opening a new research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across 5 real-world tasks covering both rigid and deformable objects, with thorough robustness experiments.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and visualizations are rich.
- Value: ⭐⭐⭐⭐⭐ Provides important insights for the VLM-based robotics community; the training-free property is a significant practical advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)
- [\[CVPR 2026\] HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](../../ICCV2025/multimodal_vlm/perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)
- [\[CVPR 2026\] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)

</div>

<!-- RELATED:END -->
