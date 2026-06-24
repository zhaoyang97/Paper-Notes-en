---
title: >-
  [Paper Note] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics
description: >-
  [CVPR 2025][Robotics][VLA] SaPaVe proposes an end-to-end active manipulation framework. By decoupling the action space of camera movement and manipulation actions, it adopts a bottom-up, two-stage training strategy (learning semantic camera control first, followed by joint optimization) to train active perception priors on a 200K semantic camera movement dataset. Coupled with a 3D geometry-aware module to enhance execution robustness under viewpoint changes…
tags:
  - "CVPR 2025"
  - "Robotics"
  - "VLA"
  - "active perception"
  - "active manipulation"
  - "camera control"
  - "humanoid robots"
date: 2026-05-08
content_hash: 87c1521652ae72de
---

# SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics

**Conference**: CVPR 2025  
**arXiv**: [2603.12193](https://arxiv.org/abs/2603.12193)  
**Code**: [https://lmzpai.github.io/SaPaVe](https://lmzpai.github.io/SaPaVe)  
**Area**: Robotics  
**Keywords**: VLA, active perception, active manipulation, camera control, humanoid robots

## TL;DR
SaPaVe proposes an end-to-end active manipulation framework. By decoupling the action space of camera movement and manipulation actions, it adopts a bottom-up, two-stage training strategy (learning semantic camera control first, followed by joint optimization) to train active perception priors on a 200K semantic camera movement dataset. Coupled with a 3D geometry-aware module to enhance execution robustness under viewpoint changes, it achieves 31.25% and 40% higher success rates than GR00T-N1 and $\pi_0$, respectively, in real-world tasks.

## Background & Motivation
**Background**: VLA models ($\pi_0$, GR00T-N1) trained and deployed under fixed, near-optimal viewpoints have achieved good manipulation capabilities.

**Limitations of Prior Work**: (1) The real world contains occlusions and out-of-view objects, which cannot be covered by fixed viewpoints; (2) directly incorporating camera movement into the VLA action space disrupts existing fixed-viewpoint manipulation priors and requires a large amount of expensive active manipulation data; (3) VLA models lack 3D geometric understanding, leading to unstable execution under viewpoint changes.

**Key Challenge**: Active manipulation requires two complementary capabilities—semantic active perception (selecting appropriate viewpoints) and active viewpoint execution (manipulating even under non-optimal viewpoints)—but existing methods either fail to support semantic-input active perception or cannot handle manipulation under viewpoint changes.

**Goal**: How to achieve both semantic-driven active perception and viewpoint-robust execution in VLAs in a data-efficient manner?

**Key Insight**: Camera movement is embodiment-agnostic (independent of the robot embodiment) and can be learned separately using large-scale image-language-camera movement data, while manipulation actions are embodiment-specific and require joint optimization.

**Core Idea**: Decoupled action space (camera vs. manipulation) + bottom-up training (learning active perception first, then active manipulation) + 3D geometry injection.

## Method

### Overall Architecture
Input RGB image $I_t$ + language instruction $L$ + optional 3D information $G_t$ → VLM backbone → decoupled action heads outputting camera movement $A_{\text{head}}$ (pitch/yaw) and manipulation actions $A_{\text{other}}$ (26-DoF joint angle increments).

### Key Designs

1. **Decoupled Action Heads & Camera Adapter**:

    - **Function**: Separates camera control and manipulation actions into two decoders and uses a LoRA adapter to learn camera movement.
    - **Mechanism**: The Camera Adapter is a LoRA module on the VLM, specifically learning semantic camera control priors. Two independent action decoders respectively predict $A_{\text{head}} \in \mathbb{R}^2$ (pitch/yaw) and $A_{\text{other}} \in \mathbb{R}^{26}$ (dual-arm + dual-hand joints).
    - **Design Motivation**: Decoupling avoids interference between camera movement and manipulation actions inside a unified action space; the Camera Adapter keeps the original VLM weights intact.

2. **Universal Spatial Knowledge Injection**:

    - **Function**: Injecting 3D geometric information (depth maps, camera intrinsic/extrinsic parameters, etc.) into the action generation process.
    - **Mechanism**: Uses the encoder of a pre-trained 3D geometric model to encode geometric information into spatial tokens, which are element-wise added to the VLM output tokens to guide action prediction during the action denoising process.
    - **Design Motivation**: Provides 3D spatial understanding for active manipulation under viewpoint changes, without requiring retraining or architecture modifications.

3. **Two-Stage Bottom-Up Training**:

    - Stage 1 (Active Perception Alignment): Train the Camera Adapter + Camera Decoder using ActiveViewPose-200K, with MSE loss supervising the camera movement. Learn "where to look in what scenes".
    - Stage 2 (Active Manipulation Fine-tuning): Freeze the Camera Adapter and train the two Action Decoders using mixed data (ActiveViewPose-200K + manipulation data) to jointly optimize both camera and manipulation.

### Datasets & Benchmark
- **ActiveViewPose-200K**: 200K image-language-camera movement pairs, with 4K finely annotated 3D assets + 500 scenes, generated via a semi-automated pipeline.
- **ActiveManip-Bench**: The first active manipulation simulation benchmark, featuring 12 tasks $\times$ 100 objects $\times$ 20 scenes.

## Key Experimental Results

### Main Results (ActiveManip-Bench Simulation)

| Method | Unoccluded | Occluded | Out-of-View | Average |
|------|-----------|----------|-------------|---------|
| GR00T-N1 | 50.0 | 24.2 | 5.0 | 17.2 |
| $\pi_0$ | 31.7 | 17.5 | 8.3 | 14.2 |
| **SaPaVe** | **83.3** | **76.7** | **70.0** | **75.2** |

### Real-World Results

| Method | Occluded PnP | OoV PnP | Occluded Arti | OoV Arti | Avg |
|------|-------------|---------|---------------|----------|-----|
| GR00T-N1 | 70 | 45 | 55 | 40 | 52.5 |
| $\pi_0$ | 55 | 35 | 45 | 30 | 41.25 |
| **SaPaVe** | **90** | **85** | **85** | **80** | **85.0** |

### Ablation Study

| Ablation | Avg Success Rate |
|------|-----------------|
| w/o Stage 1 | 53.75% |
| w/o Stage 2 | 66.25% |
| w/o Decoupled Head | 71.25% |
| w/o Camera Adapter (full finetune) | 73.75% |
| w/o Spatial Knowledge | 71.25% |
| **Full SaPaVe** | **85.0%** |

### Key Findings
- Active viewpoint is superior to the combination of fixed viewpoint + wrist camera—the success rate of fixed-viewpoint on Out-of-View tasks is <20%, while active viewpoint is >70%.
- Both stages of training are essential: removing Stage 1 cuts the Out-of-View success rate in half.
- Decoupled > Unified: Unified action space performs ~14% worse than decoupled.
- LoRA adapter > Full-parameter fine-tuning: Full fine-tuning disrupts VLM semantic understanding.
- The 2B-parameter SaPaVe outperforms Gemini 2.5 Pro by 16% on semantic active perception.

## Highlights & Insights
- **Insight that "camera movement is embodiment-agnostic"**: This key observation highlights that "where to look" does not depend on the robot embodiment itself. Therefore, it can be learned separately using large-scale image datasets and transferred to any robot.
- **Bottom-up training strategy**: Building perception priors first and then learning manipulation on top of them is much more data-efficient than end-to-end joint training.
- **The first active manipulation benchmark (ActiveManip-Bench)**: Fills the evaluation gap, covering critical scenarios such as occlusions and out-of-view objects.

## Limitations & Future Work
- Only validated on the Unitree G1 humanoid robot; generalization to other robotic platforms (e.g., robotic arms) remains to be tested.
- Camera movement is restricted to 2 DoF (pitch/yaw), without considering translation and full 6 DoF motion.
- ActiveViewPose-200K contains synthetic data, and the sim-to-real gap may affect real-world active perception quality.
- Not compared with Next-Best-View (NBV) methods or multi-view fusion methods.

## Related Work & Insights
- **vs. GR00T-N1**: Fixed-viewpoint VLA, where direct fine-tuning with camera movement yields poor results. SaPaVe's decoupled + two-stage strategy outperforms it by 31.25% in the real world.
- **vs. $\pi_0$**: Displays the same fixed-viewpoint limitations. SaPaVe outperforms it by 40%.
- **vs. VQA-based active perception**: Uses discrete candidate viewpoint selection, unable to achieve continuous camera control. SaPaVe directly outputs continuous camera movements.

## Rating
- Novelty: ⭐⭐⭐⭐ The decoupled + bottom-up training design is clever, presenting a systematic contribution to active manipulation frameworks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very comprehensive experiments across simulation, real-world, ablation studies, generalization, and comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and reasonable system design.
- Value: ⭐⭐⭐⭐⭐ The first framework to systematically address active manipulation in VLAs, with datasets and a benchmark that hold long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ActiveVLA: Injecting Active Perception into Vision-Language-Action Models for Precise 3D Robotic Manipulation](../../CVPR2026/robotics/activevla_injecting_active_perception_into_vision-language-action_models_for_pre.md)
- [\[CVPR 2025\] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)
- [\[CVPR 2025\] MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)
- [\[CVPR 2025\] Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation](overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](../../NeurIPS2025/robotics/real-world_reinforcement_learning_of_active_perception_behaviors.md)

</div>

<!-- RELATED:END -->
