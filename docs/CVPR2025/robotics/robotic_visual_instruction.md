---
title: >-
  [Paper Note] Robotic Visual Instruction
description: >-
  [CVPR 2025][Robotics][Visual Instructions] Proposes Robotic Visual Instruction (RoVI), a visual instruction paradigm centered on hand-drawn arrows and circles to guide robotic manipulation instead of natural language, and designs the VIEW pipeline to translate 2D visual instructions into 3D action sequences, achieving an 87.5% success rate in real-world environments.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Visual Instructions"
  - "Human-Computer Interaction"
  - "Hand-drawn Symbols"
  - "Robotic Manipulation"
  - "VLM"
date: 2026-05-08
content_hash: d445e9a032a8d372
---

# Robotic Visual Instruction

**Conference**: CVPR 2025  
**arXiv**: [2505.00693](https://arxiv.org/abs/2505.00693)  
**Code**: [https://robotic-visual-instruction.github.io/](https://robotic-visual-instruction.github.io/)  
**Area**: Robotics / Human-Computer Interaction  
**Keywords**: Visual Instructions, Human-Computer Interaction, Hand-drawn Symbols, Robotic Manipulation, VLM

## TL;DR

Proposes Robotic Visual Instruction (RoVI), a visual instruction paradigm centered on hand-drawn arrows and circles to guide robotic manipulation instead of natural language, and designs the VIEW pipeline to translate 2D visual instructions into 3D action sequences, achieving an 87.5% success rate in real-world environments.

## Background & Motivation

**Background**: Current human-robot interaction primarily relies on natural language, utilizing LLMs to translate textual instructions into robot actions. Some works also employ image-conditional policies (e.g., goal images, trajectory images) to convey spatial information.

**Limitations of Prior Work**: Natural language is inherently insufficient for describing spatial details (precise locations, orientations, distances), which easily leads to ambiguity and redundancy. For example, instructions like "move the lemon close to the area below the potato" struggle to convey location precisely. Additionally, certain public scenarios (e.g., libraries, hospitals) are not suitable for voice interaction. Meanwhile, goal-image methods require users to provide the final state image after task completion, and trajectory-based methods require users to imagine and draw the complete motion path of the end-effector—both of which are user-unfriendly.

**Key Challenge**: The contradiction between user-friendliness and spatial precision—natural language is convenient but imprecise, while images/trajectories are precise but user-unfriendly.

**Goal**: To design a human-robot interaction approach that simultaneously balances user-friendliness, interpretability, and spatio-temporal precision, and to construct a complete processing workflow from visual instructions to robot actions.

**Key Insight**: In daily life, people naturally convey spatial information (e.g., drawing routes on a map) through hand-drawn arrows and circles; this object-centric symbolic language can naturally encode spatio-temporal information.

**Core Idea**: To replace natural language for robot task definition with 2D hand-drawn symbols (arrows representing motion trajectories and directions, circles representing interaction areas, and colors/numbers representing temporal sequence), and leverage VLMs to comprehend these symbols and translate them into executable 3D action sequences.

## Method

### Overall Architecture

The input of the RoVI system is a hand-drawn visual instruction image overlaid on the initial observation image, and the output is the robot's 3D action sequence. The entire workflow comprises three core components: (1) a VLM responsible for understanding RoVI and generating hierarchical linguistic responses and executable code; (2) a keypoint module that extracts spatial constraints from RoVI symbols; and (3) a keypoint-based low-level policy that executes the specific actions.

### Key Designs

1. **RoVI Visual Instruction Paradigm Design**:

    - **Function**: Defines a concise visual symbolic language to encode spatio-temporal information for robotic manipulation.
    - **Mechanism**: All manipulations are decomposed into three basic motions: moving from A to B (represented by arrows), rotating objects (circle + arrow), and picking/selecting (marked by circles). Arrows are decomposed into the tail (start point $p_0$), shaft (intermediate path points), and head (endpoint $p_n$). Different colors (green $\rightarrow$ blue $\rightarrow$ pink) denote the temporal sequence of multi-step manipulations, and numeric labels are used for dual-arm systems. Additionally, two drawing styles are designed: sketchy style and geometric style.
    - **Design Motivation**: Compressing the temporal sequence of 3D coordinates into a human-understandable 2D visual language to address the spatial ambiguity of natural language. Experiments show that the structured geometric style is more friendly to VLM comprehension than the sketchy style.

2. **VIEW (Visual Instruction Embodied Workflow) Pipeline**:

    - **Function**: Translates 2D hand-drawn visual instructions into executable 3D robot actions.
    - **Mechanism**: The VLM receives the RoVI image and the initial observation image, and generates a hierarchical output through Chain-of-Thought reasoning: coarse-grained task prediction $\rightarrow$ fine-grained planning $\rightarrow$ executable Python functions. Concurrently, the keypoint module uses YOLOv8 to detect keypoints of arrows and circles, providing spatial constraints. Ultimately, the code functions are combined with the keypoint coordinates, mapped to the 3D space via an RGB-D camera, and executed.
    - **Design Motivation**: Compared to end-to-end policies directly outputting SE(3) parameters, linguistic action representation generalizes better across different tasks and environments. Using YOLOv8 to detect RoVI symbols rather than environmental objects makes the system robust to environmental variations and distractors.

3. **Keypoint-based Low-level Policy**:

    - **Function**: Generates and executes the motion of the robot's end-effector based on the sequence of keypoints.
    - **Mechanism**: The 2D keypoints are mapped to 3D coordinates $p'_i \in \mathbb{R}^3$ using RGB-D depth data, and then mapped to a sequence of end-effector poses in the SE(3) space. At each time step, the cost function $\mathcal{L}_i(t) = \alpha_i \delta_{trans}(t) + (1-\alpha_i)\delta_{rot}(t)$ is minimized, where $\alpha_i$ distinguishes translation and rotation operations. The policy switches to the next keypoint once the cost drops below a threshold $\epsilon$.
    - **Design Motivation**: Unifying translation and rotation into a single framework with adaptive switching via $\alpha_i$, enabling the handling of complex multi-step compositional actions.

### Loss & Training

The RoVI Book dataset contains 15K image-text QA pairs, constructed based on the Open-X Embodiment dataset. LoRA is used to fine-tune LLaVA-7B/13B with a learning rate of 2e-4 for 1 epoch. The data covers 64% single-step tasks and 36% multi-step tasks, comprising 5 basic manipulation skills. Data augmentation is applied to RoVI (3-8 variants with different paths, styles, and line widths).

## Key Experimental Results

### Main Results

| Method | Average Success Rate (Real) | Average Success Rate (Sim) |
|------|-------------------|-------------------|
| VoxPoser | 43.8% | - |
| CoPa | 45.0% | - |
| VIEW-GPT4o | 82.5% | - |
| VIEW-LLaVA-13B (RoVI Book) | **87.5%** | - |
| RT-1-X | - | 20% |
| Octo-goal-image | - | 13.3% |
| Octo-language | - | 3% |
| VIEW* | - | **76.6%** |

### Ablation Study

| Configuration | Task Planning Accuracy | Description |
|------|---------------|------|
| GPT-4o (Zero-shot) | 81% | Strongest commercial model |
| Gemini-1.5 Pro | 68% | Weaker simulation performance |
| Claude 3.5 Sonnet | 70% | Accuracy drops in multi-step tasks |
| LLaVA-13B (RoVI Book) | 38% | Low planning accuracy but high execution success rate |
| Small Models (<13B) | 0% | Completely fails to understand RoVI instructions |
| Sketchy Style | 74% | - |
| Geometric Style | **80%** | Structured style is more friendly for VLM understanding |

### Key Findings

- While LLaVA-13B is significantly lower in task planning accuracy (38%) than GPT-4o (81%), it performs comparably or even better at the action execution level (87.5% vs 82.5%). This is because the executable functions map away action and sequence errors, remaining unaffected by perception errors.
- VIEW significantly outperforms language-instruction methods in cluttered environments and trajectory-following tasks because the keypoint module provides pixel-level precise spatial constraints.
- All models with fewer than 13B parameters completely fail to understand RoVI, indicating that comprehending such visual symbols requires sufficient model capacity.

## Highlights & Insights

- **Object-Centric Symbolic Design**: Utilizing only four basic elements—arrows, circles, colors, and numbers—to encode complex multi-step manipulations, the design is extremely concise. This design philosophy can be transferred to other scenarios requiring precise spatial representation (e.g., surgical robot instructions).
- **Code Generation via VLM Comprehension of RoVI**: Prompting the VLM to output Python code functions instead of direct action parameters provides excellent debuggability and interpretability.
- **Keypoint Module Detecting RoVI Symbols Instead of Environmental Objects**: This cleverly bypasses the difficulties of open-vocabulary object detection in cluttered environments, making the system robust to environmental changes and distractors.

## Limitations & Future Work

- Currently, RoVI still requires drawing with a stylus on a tablet/PC, leaving room for improvement in interaction convenience; future work could consider more natural input modalities like AR or gestures.
- The 2D-to-3D mapping relies on the accuracy of depth cameras, which might fail in scenarios with severe occlusion or inaccurate depth.
- The color-coding scheme limits the maximum supported steps and assumes a darker background to ensure symbol visibility.
- Support for dual-arm collaborative manipulation is still preliminary; complex collaborative tasks may require richer symbolic semantics.

## Related Work & Insights

- **vs VoxPoser/CoPa**: These methods still rely on natural language input combined with object detection, underperforming in cluttered environments and tasks requiring precise spatial alignment. RoVI avoids the spatial ambiguity of language through pixel-level visual instructions.
- **vs Goal-Image / Trajectory Policies**: Goal-image methods are user-unfriendly (requiring knowledge of the final state), and trajectory methods make it difficult for users to envision the entire motion process. RoVI strikes a compromise—conveying key spatial information using simple symbols.
- **vs RT-1-X/Octo**: End-to-end VLA models are severely limited when generalizing to new tasks, whereas VIEW's modular design (VLM + keypoints + low-level policy) offers better generalization capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ Proposes a brand-new visual symbolic instruction paradigm. The concept is concise and elegant, though the symbolic design itself is not overly complex.
- Experimental Thoroughness: ⭐⭐⭐⭐ 11 tasks covering both real and sim environments with multi-perspective ablations and comparisons, though the scale is not very large.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive illustrations, and the teaser figure is clear at a glance.
- Value: ⭐⭐⭐⭐ Opens up a new direction for visual symbolic interaction, but the convenience of practical deployment and user acceptance remain to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation](mitigating_the_human-robot_domain_discrepancy_in_visual_pre-training_for_robotic.md)
- [\[ACL 2026\] GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning](../../ACL2026/robotics/govig_goal-conditioned_visual_navigation_instruction_generation_via_multimodal_r.md)
- [\[CVPR 2025\] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)
- [\[CVPR 2025\] Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation](overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)
- [\[CVPR 2025\] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](roboground_robotic_manipulation_with_grounded_vision-language_priors.md)

</div>

<!-- RELATED:END -->
