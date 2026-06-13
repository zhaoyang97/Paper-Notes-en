---
title: >-
  [Paper Note] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics
description: >-
  [CVPR 2026][Robotics][Active Perception] SaPaVe proposes an end-to-end active manipulation framework that decouples camera actions from manipulation actions via a bottom-up training strategy: it first learns active perce…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "Active Perception"
  - "VLA Models"
  - "Decoupled Action Space"
  - "3D Spatial Injection"
  - "Humanoid Robots"
date: 2026-05-08
content_hash: 3f342f94e108ab51
---

# SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics

**Conference**: CVPR 2026
**arXiv**: [2603.12193](https://arxiv.org/abs/2603.12193)  
**Code**: [https://lmzpai.github.io/SaPaVe](https://lmzpai.github.io/SaPaVe)  
**Area**: Multimodal VLM / Robotics
**Keywords**: Active Perception, VLA Models, Decoupled Action Space, 3D Spatial Injection, Humanoid Robots

## TL;DR
SaPaVe proposes an end-to-end active manipulation framework that decouples camera actions from manipulation actions via a bottom-up training strategy: it first learns active perception priors from 200K semantic camera-control pairs, then jointly optimizes for active manipulation, surpassing π₀ and GR00T N1 by up to 31.25% in real-world success rate.

## Background & Motivation

1. **Background**: Active perception and manipulation are core capabilities for robots interacting with complex scenes. Existing VLMs (e.g., Qwen2.5-VL, Gemini 2.5 Pro) have improved semantic understanding, while VLA models (e.g., π₀, GR00T N1) aim to bridge vision–language–action end-to-end.
2. **Limitations of Prior Work**:
    - VLMs treat active perception as a VQA task (selecting the best viewpoint from discrete candidates), precluding continuous fine-grained camera control.
    - VLA models are typically trained and evaluated under fixed, optimal head-camera viewpoints, making them sensitive to viewpoint changes and unable to actively adjust perspective.
    - Naively extending the VLA action space with camera actions causes conflicts and requires large amounts of expensive real-world active-perception-plus-manipulation data.
3. **Key Challenge**: Active manipulation requires tight coupling of *semantic active perception* (adjusting viewpoint based on task strategy to acquire critical information) and *active-viewpoint execution* (robust manipulation under dynamic viewpoints), yet data scarcity and action-space conflicts make it difficult for existing methods to achieve both.
4. **Goal**: Enable robots to simultaneously learn semantically driven active viewpoint adjustment and robust manipulation under viewpoint changes, in a data-efficient manner.
5. **Key Insight**: The key insight is that camera motion is embodiment-agnostic and can therefore be learned independently before joint optimization, enabling an efficient bottom-up training pipeline.
6. **Core Idea**: Decouple camera actions from manipulation actions; first establish active-perception priors from large-scale semantic camera-motion data, then jointly optimize for data-efficient active manipulation.

## Method

### Overall Architecture
SaPaVe builds on a VLA model backbone. Given RGB images and task instructions, it outputs two decoupled action streams: head-camera actions $A_{head}$ (pitch/yaw adjustments) and manipulation actions $A_{other}$ (26-DoF joint position deltas for the Unitree G1 dual-arm dual-hand platform). Action chunking is used to predict action sequences of temporal horizon $k$, ensuring temporal consistency and smooth execution.

### Key Designs

1. **Decoupled Action Heads & Camera Adapter**

    - **Function**: Enable the model to learn camera control and manipulation actions separately without degrading the VLM's original semantic capabilities.
    - **Mechanism**: A camera adapter applies LoRA on top of the VLM to learn semantic active-perception priors without modifying the original VLM weights. Two independent denoising decoders constitute the decoupled action heads, outputting camera actions and manipulation actions respectively. This lightweight decoupled design allows the model to learn both action types accurately while avoiding interference from a unified action space.
    - **Design Motivation**: Adding camera motion directly to the existing VLA action space disrupts priors learned from large-scale fixed-viewpoint manipulation data. Since camera motion is inherently embodiment-agnostic, learning it via an independent adapter is more efficient and preserves manipulation capability. Experiments confirm that full fine-tuning of the VLM for camera motion is inferior to using a lightweight adapter (Tab. 5), as the adapter retains high-level semantic information.

2. **Universal Spatial Knowledge Injection**

    - **Function**: Enhance the model's 3D spatial awareness and robustness to dynamic viewpoint changes.
    - **Mechanism**: A Universal Spatial Encoder inherited from a strong feed-forward 3D geometry model accepts arbitrary 3D geometric inputs (depth maps, camera intrinsics/extrinsics, etc.) without retraining or architectural modification. Encoded spatial tokens are element-wise added to VLM output tokens, and the fused tokens are injected into the action denoising process of the decoupled action heads.
    - **Design Motivation**: VLA models lack 3D geometric priors and cannot maintain consistent spatial understanding under active viewpoint changes. Directly injecting diverse 3D information fundamentally improves robustness to viewpoint variation. Ablations show that removing this module causes a 15% drop in success rate even on simple occlusion-grasping tasks (Tab. 5).

3. **Two-Stage Bottom-Up Training Strategy**

    - **Function**: Build active perception and active manipulation capabilities layer by layer in a data-efficient manner.
    - **Mechanism**:
     - **Stage 1 (Semantic Active Perception Alignment)**: Trains only the camera adapter and camera action decoder using the ActiveViewPose-200K dataset, with MSE loss $\mathcal{L}_{stage1} = \mathcal{L}_{MSE}(A_{head,t}, A_{head,t}^*)$. This stage establishes strong semantically driven viewpoint adjustment priors.
     - **Stage 2 (Active Manipulation Fine-tuning)**: Freezes the camera adapter and trains the decoupled action heads on mixed data (ActiveViewPose-200K + robot manipulation data), with $\mathcal{L}_{stage2} = \lambda_{head}\mathcal{L}_{head} + \lambda_{other}\mathcal{L}_{other}$.
    - **Design Motivation**: Joint training from scratch requires large amounts of scarce active-manipulation data. By first building perception priors on abundantly available viewpoint-only data and then fine-tuning with a small amount of manipulation data, the approach achieves data-efficient transfer learning.

### Loss & Training
- Stage 1: MSE loss supervising camera action prediction only.
- Stage 2: Weighted MSE loss supervising both camera and manipulation actions; the camera adapter is frozen to protect the priors learned in Stage 1.
- Action chunking ensures temporal smoothness.

## Key Experimental Results

### Main Results: Semantic Active Perception Evaluation

| Method | Val | Test1 | Test2 | Avg |
|--------|-----|-------|-------|-----|
| Qwen2.5-VL-72B | 63.9 | 65.1 | 58.0 | 62.3 |
| Multi-SpatialMLLM | 72.8 | 74.3 | 63.6 | 70.2 |
| Gemini-2.5-Pro | 73.3 | 76.5 | 68.2 | 72.7 |
| **SaPaVe (2B)** | **85.5** | **89.1** | **78.3** | **84.3** |

Real-world active manipulation (success rate %):

| Method | Occluded Pick-Place | Out-of-View Pick-Place | Occluded Articulated Manip. | Out-of-View Articulated Manip. | Avg |
|--------|--------------------|-----------------------|-----------------------------|-------------------------------|-----|
| π₀ | 55 | 45 | 45 | 35 | 45.00 |
| GR00T-N1 | 60 | 55 | 50 | 50 | 53.75 |
| **SaPaVe** | **90** | **85** | **85** | **80** | **85.00** |

### Ablation Study

| Configuration | Occluded Pick-Place | Out-of-View Pick-Place | Occluded Manip. | Out-of-View Manip. | Avg |
|---------------|--------------------|-----------------------|-----------------|-------------------|-----|
| Full Model | 90 | 85 | 85 | 80 | 85.00 |
| w/o Stage 1 | 65 | 55 | 50 | 45 | 53.75 |
| w/o Stage 2 | 75 | 60 | 70 | 60 | 66.25 |
| w/o Decoupled Action Heads | 80 | 70 | 70 | 65 | 71.25 |
| w/o Camera Adapter | 80 | 75 | 70 | 70 | 73.75 |
| w/o Spatial Knowledge Injection | 75 | 75 | 65 | 60 | 68.75 |

### Key Findings
- Stage 1 contributes the most: removing it causes a 31.25% average drop (85→53.75), with out-of-view tasks nearly halved, confirming that active-perception priors are central.
- Removing Universal Spatial Knowledge Injection causes a 16.25% drop; even simple occlusion-grasping tasks fall by 15%, underscoring the importance of 3D information for viewpoint robustness.
- SaPaVe with only 2B parameters outperforms Qwen2.5-VL (72B) and Gemini 2.5 Pro on semantic active perception, demonstrating that this capability does not emerge from general-purpose VLMs and requires dedicated training.
- A fixed-camera plus wrist-camera setup still falls far short of an active camera, especially on out-of-view tasks (gap > 40%), showing that "more viewpoints" does not substitute for "actively controlled viewpoints."

## Highlights & Insights
- The insight that **camera motion is embodiment-agnostic** is the most fundamental contribution of the paper. It directly motivates the decoupled, bottom-up training strategy, which is both elegant and effective, and can transfer to other robot learning settings requiring separation of general capabilities from embodiment-specific ones.
- The **ActiveViewPose-200K** construction pipeline (4K high-quality assets + heuristic action generation + GPT-4o instruction generation + human refinement) is efficient and reproducible, providing the community with a benchmark that fills an existing gap.
- The absolute real-world success rate improvements over π₀ (+40%) and GR00T N1 (+31.25%) demonstrate that active manipulation capability cannot be achieved by simply adding action dimensions.

## Limitations & Future Work
- Validation is limited to the Unitree G1 humanoid; transferability to other robot morphologies (e.g., single-arm, mobile-base platforms) remains unverified.
- Current camera actions are only 2-DoF (pitch/yaw), without considering translational or more complex viewpoint adjustments.
- ActiveViewPose-200K is constructed semi-automatically from static scenes; real-world dynamic occlusion changes may require additional data.
- Online learning mechanisms could be explored to allow robots to continuously improve active-perception strategies during execution.

## Related Work & Insights
- **vs. π₀ [6]**: π₀ is a strong general VLA but lacks active perception; naively fine-tuning it with camera actions yields only 45% success rate, whereas SaPaVe's decoupled strategy reaches 85%.
- **vs. GR00T-N1 [5]**: Similarly lacks active perception priors; despite being designed specifically for humanoids, it is surpassed by SaPaVe by 31.25% on active manipulation.
- **vs. NBV methods [7,54]**: Traditional Next-Best-View methods are non-end-to-end and lack semantic inputs; SaPaVe integrates semantic understanding and continuous camera control end-to-end.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The decoupled bottom-up strategy and ActiveManip-Bench are both pioneering contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Full coverage of simulation, real-world, ablation, and generalization experiments with well-chosen baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Clear paper structure and thorough experimental analysis.
- **Value**: ⭐⭐⭐⭐⭐ Fills the gap for VLA models in active manipulation; the dataset and benchmark are of high value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)
- [\[ICML 2026\] ManiSoft: Towards Vision-Language Manipulation for Soft Continuum Robotics](../../ICML2026/robotics/manisoft_towards_vision-language_manipulation_for_soft_continuum_robotics.md)

</div>

<!-- RELATED:END -->
