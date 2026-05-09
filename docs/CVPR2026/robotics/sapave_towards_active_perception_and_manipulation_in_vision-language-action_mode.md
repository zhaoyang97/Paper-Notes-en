---
title: >-
  [Paper Note] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics
description: >-
  [CVPR 2026][Robotics][Active Perception] SaPaVe is an end-to-end framework that decouples camera motion from manipulation actions via a two-stage bottom-up learning strategy, enabling semantics-driven active perception and viewpoint-invariant manipulation execution. It surpasses GR00T N1 and π₀ by 31.25% and 40%, respectively, on real-world tasks.
tags:
  - CVPR 2026
  - Robotics
  - Active Perception
  - Active Manipulation
  - Vision-Language-Action
  - Semantic Camera Control
  - Decoupled Action Space
  - 3D Spatial Awareness
date: 2026-05-08
content_hash: 7dd05e1f806e5331
---

# SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics

**Conference**: CVPR 2026
**arXiv**: [2603.12193](https://arxiv.org/abs/2603.12193)
**Code**: [Project Page](https://lmzpai.github.io/SaPaVe)
**Area**: Robotics
**Keywords**: Active Perception, Active Manipulation, Vision-Language-Action, Semantic Camera Control, Decoupled Action Space, 3D Spatial Awareness

## TL;DR

SaPaVe is an end-to-end framework that decouples camera motion from manipulation actions via a two-stage bottom-up learning strategy, enabling semantics-driven active perception and viewpoint-invariant manipulation execution. It surpasses GR00T N1 and π₀ by 31.25% and 40%, respectively, on real-world tasks.

## Background & Motivation

1. **Core challenge of active manipulation**: Robots must simultaneously possess semantic active perception (strategically adjusting viewpoints to acquire task-critical information) and active viewpoint execution (robustly completing operations under dynamic viewpoints)—two complementary capabilities that existing methods struggle to unify.
2. **Limitations of VLM discretization**: VLM-based methods model active perception as a VQA task, selecting the optimal viewpoint from discrete candidates, and are therefore incapable of continuous, fine-grained camera control.
3. **Fragility of fixed-viewpoint VLAs**: Existing end-to-end VLA models (e.g., π₀, GR00T N1) are primarily trained under fixed near-optimal viewpoints, making them highly sensitive to viewpoint changes and ill-suited for active manipulation scenarios.
4. **High cost of data acquisition**: Real-world data containing both head camera motion and manipulation action annotations is extremely scarce and expensive to collect; directly fine-tuning VLAs in a unified action space leads to conflicts and suboptimal performance.
5. **Lack of 3D geometric awareness**: Current VLA models do not sufficiently exploit 3D geometric priors, resulting in high sensitivity to camera perturbations and an inability to reason effectively under viewpoint changes.
6. **Absence of evaluation benchmarks**: Existing simulation benchmarks (e.g., RLBench, CALVIN) are restricted to fixed viewpoints, and no standardized benchmark exists for evaluating active manipulation capabilities.

## Method

### Overall Architecture

SaPaVe is built on a VLM backbone that receives RGB images and language instructions, and outputs camera motion and manipulation actions through a **decoupled action space**. The core design philosophy is that camera motion is embodiment-agnostic and easier to learn than manipulation actions; accordingly, a **bottom-up two-stage training strategy** is adopted.

### Key Designs

1. **Decoupled Action Heads**: Two independent decoders are designed to handle camera motion (2-DoF pitch/yaw) and manipulation actions (26-DoF joint position deltas) separately, avoiding the learning conflicts caused by a unified action space.

2. **Camera Adapter**: LoRA adapters are added on top of the VLM to learn semantic camera motion priors without modifying the original VLM weights, thereby preserving general semantic understanding capabilities.

3. **Universal Spatial Knowledge Injection (USKI)**: A pretrained 3D geometric encoder (inherited from MAPAnything) supports arbitrary types of 3D geometric inputs (depth maps, camera intrinsics/extrinsics, etc.). The encoded spatial tokens are element-wise added to the VLM output tokens and injected into the decoupled action heads to guide action decoding, enhancing spatial robustness under viewpoint changes.

4. **ActiveViewPose-200K Dataset**: A dataset of 200K image–language–camera-motion triplets, efficiently constructed via a semi-automatic pipeline using 4K high-quality semantically annotated assets, 500 diverse scenes, heuristic algorithms, and GPT-4o-generated instructions.

5. **ActiveManip-Bench**: Built on NVIDIA Isaac Sim, this benchmark features the G1 humanoid robot, 12 semantic active manipulation tasks, 100 objects, and 20 scenes—the first simulation benchmark for evaluating active manipulation.

### Loss & Training

- **Stage 1 – Semantic Active Perception Alignment**: ActiveViewPose-200K is used to train only the Camera Adapter and camera decoder. The loss is the MSE between predicted and ground-truth camera motion: $\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{MSE}}(A_{\text{head}}, A_{\text{head}}^*)$
- **Stage 2 – Active Manipulation Fine-tuning**: The Camera Adapter is frozen; ActiveViewPose-200K and robot manipulation data are mixed to train the decoupled action heads: $\mathcal{L}_{\text{stage2}} = \lambda_{\text{head}} \mathcal{L}_{\text{head}} + \lambda_{\text{other}} \mathcal{L}_{\text{other}}$

## Key Experimental Results

### Semantic Active Perception Evaluation (ActiveViewPose-200K Test Set)

| Method | Val | Test1 | Test2 | Avg |
|--------|-----|-------|-------|-----|
| Qwen2.5-VL-72B | 63.9 | 65.1 | 58.0 | 62.3 |
| Multi-SpatialMLLM | 72.8 | 74.3 | 63.6 | 70.2 |
| Gemini-2.5-Pro | 73.3 | 76.5 | 68.2 | 72.7 |
| **SaPaVe (Stage 1)** | **85.5** | **89.1** | **78.3** | **84.3** |

With only 2B parameters, SaPaVe surpasses Gemini-2.5-Pro by 11.6%, demonstrating that semantic active perception is not an emergent capability of general large models and requires dedicated data training.

### Real-World Active Manipulation (vs. Existing VLAs)

| Method | Occluded Pick-and-Place | Out-of-View Pick-and-Place | Occluded Articulated Manipulation | Out-of-View Articulated Manipulation | Avg |
|--------|------------------------|---------------------------|----------------------------------|--------------------------------------|-----|
| π₀ | 55 | 45 | 45 | 35 | 45.00 |
| GR00T-N1 | 60 | 55 | 50 | 50 | 53.75 |
| **SaPaVe** | **90** | **85** | **85** | **80** | **85.00** |

### Ablation Study

| Ablation | Occluded Pick-and-Place | Out-of-View Pick-and-Place | Occluded Articulated | Out-of-View Articulated | Avg |
|----------|------------------------|---------------------------|----------------------|-------------------------|-----|
| w/o Stage 1 | 65 | 55 | 50 | 45 | 53.75 |
| w/o Stage 2 | 75 | 60 | 70 | 60 | 66.25 |
| w/o Decoupled Action Heads | 80 | 70 | 70 | 65 | 71.25 |
| w/o Camera Adapter | 80 | 75 | 70 | 70 | 73.75 |
| w/o USKI | 75 | 75 | 65 | 60 | 68.75 |
| **Full Model** | **90** | **85** | **85** | **80** | **85.00** |

Each component contributes significantly. Stage 1 has the greatest impact on out-of-view tasks (success rate roughly halved when removed), and USKI is critical for basic manipulation robustness.

## Highlights & Insights

- **Elegant decoupling**: Separating embodiment-agnostic camera motion from embodiment-specific manipulation actions reduces data requirements and avoids learning conflicts.
- **Data efficiency**: The bottom-up strategy reuses large-scale, easily collectible camera motion data, requiring only limited robot data to generalize.
- **Complete ecosystem**: The simultaneous release of a dataset (ActiveViewPose-200K) and an evaluation benchmark (ActiveManip-Bench) fills a critical gap in active manipulation assessment.
- **Strong experimental validity**: Simulation, real-world, generalization, and ablation experiments collectively provide clear attribution of each component's contribution.
- **Substantial real-world gains**: Large improvements over π₀ (+40%) and GR00T N1 (+31.25%) in real-world settings demonstrate the necessity of active perception.

## Limitations & Future Work

- Currently limited to 2-DoF head camera motion (pitch/yaw), without extension to full 6-DoF viewpoint control.
- Manipulation is evaluated on the Unitree G1 humanoid robot (26-DoF); transferability to other robot morphologies is not sufficiently validated.
- Camera motion in ActiveViewPose-200K is generated by heuristic algorithms, which may deviate from human-intuitive viewpoint adjustment strategies.
- Sim-to-real transfer details are insufficiently discussed.
- The combination of active camera and wrist camera yields limited gains (Tab. 2), suggesting that multi-view fusion strategies warrant further optimization.

## Related Work & Insights

- **vs. Next-Best-View methods**: NBV methods are not end-to-end and lack semantic inputs; SaPaVe achieves semantics-driven end-to-end active perception.
- **vs. VQA-based methods**: VQA-based methods (e.g., Look Further) select from discrete candidate viewpoints, whereas SaPaVe directly predicts camera motion in continuous space.
- **vs. π₀ / GR00T N1**: Directly extending the action space of these VLAs to include camera motion performs poorly due to unified-space conflicts and the absence of perceptual priors; SaPaVe substantially surpasses them through decoupling and the two-stage strategy.
- **vs. teleoperation methods (e.g., Open-TeleVision)**: These methods rely on costly real-world data collection, whereas SaPaVe's camera motion training data can be generated at scale with low cost.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first end-to-end active manipulation framework; the decoupling and bottom-up strategy are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Simulation, real-world, generalization, and ablation experiments, with dataset and benchmark released simultaneously.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and well-motivated problem statement, though some details require consulting the appendix.
- **Value**: ⭐⭐⭐⭐⭐ — Fills an important gap in active manipulation research and represents a significant contribution to the robotics community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](mergevla_crossskill_model_merging_toward_a_general.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)

</div>

<!-- RELATED:END -->
