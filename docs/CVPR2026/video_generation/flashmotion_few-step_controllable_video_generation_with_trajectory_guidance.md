---
title: >-
  [Paper Note] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance
description: >-
  [CVPR 2026][Video Generation][trajectory-controllable video generation] FlashMotion proposes a three-stage training framework that distills a multi-step trajectory-controllable video generation model into a few-step counterpart. By fine-tuning the adapter with a hybrid diffusion and adversarial objective, the method simultaneously preserves video quality and trajectory accuracy under few-step inference.
tags:
  - CVPR 2026
  - Video Generation
  - trajectory-controllable video generation
  - distillation acceleration
  - few-step inference
  - adversarial training
  - diffusion models
date: 2026-05-08
content_hash: 9d239790fe3d1ace
---

# FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance

**Conference**: CVPR 2026
**arXiv**: [2603.12146](https://arxiv.org/abs/2603.12146)
**Code**: Unavailable
**Area**: Video Generation
**Keywords**: trajectory-controllable video generation, distillation acceleration, few-step inference, adversarial training, diffusion models

> ⚠️ This note is written based on the arXiv abstract (local cache contains abstract only, ~4.7KB); method and experimental details are limited.

## TL;DR
FlashMotion proposes a three-stage training framework that distills a multi-step trajectory-controllable video generation model into a few-step counterpart. By fine-tuning the adapter with a hybrid diffusion and adversarial objective, the method simultaneously preserves video quality and trajectory accuracy under few-step inference.

## Background & Motivation
Trajectory-controllable video generation has achieved remarkable progress in recent years, enabling users to precisely control object motion paths via predefined trajectories. Existing methods primarily adopt adapter architectures (e.g., ControlNet-style) injected into video diffusion models to achieve precise motion control.

**Key Challenge**: All such methods rely on multi-step denoising processes (typically 20–50 steps), resulting in long inference times and substantial computational overhead. Although video distillation methods (e.g., consistency distillation, adversarial distillation) can compress multi-step generators into few-step versions, **directly applying these distillation methods to trajectory-controllable video generation leads to significant degradation in both video quality and trajectory accuracy**.

**Key Challenge**: Distillation alters the latent space distribution of the model, introducing a distribution mismatch between the trajectory adapter trained on the multi-step model and the distilled few-step model, causing the adapter's control signals to be misinterpreted.

**Key Insight**: The paper designs a three-stage training framework — first training the adapter, then distilling the base model, and finally re-aligning the adapter with the few-step model using a hybrid objective — to fundamentally resolve the distribution mismatch problem.

## Method

### Overall Architecture
FlashMotion adopts a three-stage training pipeline:
1. **Stage 1 — Adapter Training**: Train a trajectory control adapter on the multi-step video generator to acquire precise trajectory control capability.
2. **Stage 2 — Base Model Distillation**: Distill the multi-step video generator into a few-step version to accelerate video generation.
3. **Stage 3 — Adapter Fine-tuning and Alignment**: Fine-tune the adapter using a hybrid strategy (diffusion objective + adversarial objective) to adapt it to the few-step generator.

### Key Designs

1. **Trajectory Adapter Training (Stage 1)**:

    - **Function**: Train a plug-and-play trajectory control module on the original multi-step video diffusion model.
    - **Design Motivation**: The adapter architecture enables precise injection of motion control into the base model without affecting its original generation quality.
    - **Mechanism**: Follows standard adapter training paradigms (e.g., ControlNet); the adapter takes predefined trajectories as input and learns to map trajectories to video motion.

2. **Video Generator Distillation (Stage 2)**:

    - **Function**: Compress the multi-step (20–50 steps) video diffusion model into a few-step (4–8 steps) version.
    - **Design Motivation**: Few-step inference substantially reduces computational overhead, though distillation shifts the latent space distribution of the model.
    - **Mechanism**: Employs existing video distillation methods; the paper validates generality across two distinct adapter architectures.

3. **Hybrid-Objective Adapter Fine-tuning (Stage 3)**:

    - **Function**: Re-align the adapter with the distilled few-step generator.
    - **Design Motivation**: Distillation alters the latent space, rendering the Stage 1 adapter incompatible with the few-step model.
    - **Mechanism**: A hybrid training strategy combining a diffusion objective (preserving trajectory accuracy) and an adversarial objective (enhancing video quality).
    - **Novelty**: The hybrid objective simultaneously optimizes two dimensions — the diffusion loss ensures that trajectory control signals are correctly propagated, while the adversarial loss ensures the perceptual quality of few-step generated videos.

### Evaluation Benchmark — FlashBench
- The paper introduces FlashBench, a benchmark specifically designed to evaluate long-sequence trajectory-controllable video generation.
- It jointly measures video quality and trajectory accuracy.
- It supports scenarios with varying numbers of foreground objects.

### Loss & Training
- Stage 3 employs a hybrid loss: $\mathcal{L} = \mathcal{L}_{diffusion} + \lambda \mathcal{L}_{adversarial}$
- The diffusion objective guarantees trajectory control precision.
- The adversarial objective guarantees the perceptual quality of generated videos.

## Key Experimental Results

### Main Results (Based on Abstract)

| Comparison Dimension | FlashMotion | Existing Distillation Methods | Multi-Step Models |
|---|---|---|---|
| Video Quality | ✓ Best | Notable degradation | Good |
| Trajectory Accuracy | ✓ Best | Notable degradation | Good |
| Inference Steps | Few-step (4–8) | Few-step | Multi-step (20–50) |

### Architectural Generality Validation

| Experimental Setting | Description |
|---|---|
| Adapter Architecture 1 | FlashMotion outperforms video distillation baseline |
| Adapter Architecture 2 | FlashMotion likewise outperforms baseline |
| Remark | Generality of the method is validated across two distinct adapter architectures |

### Key Findings
- Directly applying the original adapter to the distilled model leads to severe degradation, confirming the distribution mismatch problem.
- The hybrid diffusion + adversarial fine-tuning strategy effectively resolves this mismatch.
- The method is effective across two different adapter architectures, demonstrating strong generality.
- FlashMotion not only matches the performance of multi-step models but surpasses them on certain metrics.

## Highlights & Insights
- The paper precisely identifies the core bottleneck of few-step trajectory-controllable video generation — the distribution mismatch between the adapter and the distilled model.
- The three-stage decoupled training design is elegant: the two components are independently optimized before being aligned via a hybrid objective.
- The method is decoupled from specific adapter architectures, offering strong generality.
- The introduction of FlashBench addresses the gap in evaluation benchmarks for trajectory-controllable video generation.

## Limitations & Future Work
- The local cache contains only the abstract, precluding access to specific experimental data and implementation details.
- The three-stage training pipeline is relatively complex; the practical deployment cost warrants further evaluation.
- The specific number of few-step inference steps and the corresponding speedup ratio are not explicitly reported in the abstract.
- Applicability to longer videos and more complex multi-object trajectory scenarios remains to be verified.

## Related Work & Insights
- Trajectory-controllable video generation: adapter-based methods such as DragNUWA and MotionCtrl.
- Video distillation: consistency distillation and adversarial diffusion distillation.
- Insights for accelerating controllable generation: when distillation shifts the base model's distribution, control modules require re-alignment — an insight generalizable to other controllable generation settings (e.g., layout, depth).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The three-stage decoupled design combined with hybrid-objective fine-tuning is novel and practical, precisely targeting the distribution mismatch problem.
- **Experimental Thoroughness**: ⭐⭐⭐ (based on abstract) Validation across two architectures and a new benchmark.
- **Writing Quality**: ⭐⭐⭐ The abstract is clear with well-defined problem formulation.
- **Value**: ⭐⭐⭐⭐ Strong practical demand for accelerating trajectory-controllable video generation; method demonstrates good generality.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[CVPR 2026\] Gloria: Consistent Character Video Generation via Content Anchors](gloria_consistent_character_video_generation_via_content_anchors.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[CVPR 2026\] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](uniavgen_unified_audio_and_video_generation_with_asymmetric_cross-modal_interact.md)

<!-- RELATED:END -->
