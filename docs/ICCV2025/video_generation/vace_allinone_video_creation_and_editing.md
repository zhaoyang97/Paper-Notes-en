---
title: >-
  [Paper Note] VACE: All-in-One Video Creation and Editing
description: >-
  [ICCV 2025][Video Generation][Unified Framework] VACE is proposed as a unified framework for video creation and editing. It introduces a Video Condition Unit (VCU) that consolidates text, image, video, and mask inputs into a unified conditional representation. Combined with a Context Adapter that injects task concepts into a DiT model, VACE is the first single video DiT to simultaneously support reference-guided generation, video editing, mask-based editing, and their arbitrary combinations.
tags:
  - ICCV 2025
  - Video Generation
  - Unified Framework
  - DiT
  - Video Condition Unit
  - Controllable Generation
  - Video Editing
date: 2026-05-08
content_hash: 99b26ce7a842a0a6
---

# VACE: All-in-One Video Creation and Editing

**Conference**: ICCV 2025
**arXiv**: [2503.07598](https://arxiv.org/abs/2503.07598)
**Code**: [Project](https://ali-vilab.github.io/VACE-Page/)
**Area**: Video Generation / Video Editing
**Keywords**: Unified Framework, DiT, Video Condition Unit, Controllable Generation, Video Editing

## TL;DR

VACE is proposed as a unified framework for video creation and editing. It introduces a Video Condition Unit (VCU) that consolidates text, image, video, and mask inputs into a unified conditional representation. Combined with a Context Adapter that injects task concepts into a DiT model, VACE is the first single video DiT to simultaneously support reference-guided generation, video editing, mask-based editing, and their arbitrary combinations.

## Background & Motivation

- **Background**: The image domain already has unified generation-editing frameworks such as ACE and OmniGen, but the video domain—due to its higher spatiotemporal consistency requirements—still relies predominantly on single-task, single-model approaches.
- **Limitations of Prior Work**: The large variety of video tasks (reference-guided generation, style transfer, inpainting, etc.) makes deploying separate models per task costly; chaining multiple tasks (e.g., long-video editing pipelines) is difficult to realize.
- **Key Challenge**: Unifying diverse video task input modalities while preserving spatiotemporal consistency.
- **Goal**: Build a single model that covers as many video generation and editing tasks as possible.
- **Key Insight**: Design a unified input interface (VCU) + a concept decoupling strategy + a plug-and-play Context Adapter.
- **Core Idea**: Represent all video task conditions as a triplet (text, frame sequence, mask sequence), and achieve multi-task unification via concept decoupling and adapter injection.

## Method

### Overall Architecture

Built upon a pretrained T2V DiT, the model takes a VCU triplet $[T; F; M]$ as input. $F$ and $M$ are decomposed via concept decoupling into reactive frames (to be modified) and inactive frames (to be preserved). Both are VAE-encoded and fused with noisy video tokens, with training performed either by full fine-tuning or via the Context Adapter.

### Key Designs

**Design 1: Video Condition Unit (VCU)**
- **Function**: Unifies four fundamental task types—T2V, R2V, V2V, and MV2V—and their combinations into a $(T, F, M)$ triplet.
- **Mechanism**: $T$ denotes text; $F$ is a frame sequence (reference images, control signals, edited video, or blank frames); $M$ is a binary mask sequence (1 = to be generated, 0 = to be preserved). Different tasks are represented by different compositions of $F$ and $M$.
- **Design Motivation**: Eliminates the need for task-specific input interfaces and enables a high degree of combinatorial flexibility.

**Design 2: Concept Decoupling Strategy**
- **Function**: Decomposes $F$ via $M$ into reactive frames ($F_c = F \times M$) and inactive frames ($F_k = F \times (1-M)$).
- **Mechanism**: Explicitly separates pixels to be edited from pixels to be preserved, enabling the model to distinguish between editing and reference roles. Each subset is encoded independently by the VAE.
- **Design Motivation**: Natural video frames and control signals (e.g., depth maps, pose sequences) have different distributions; mixing them hinders convergence.

**Design 3: Context Adapter Tuning**
- **Function**: An optional lightweight training strategy that freezes the DiT backbone and trains only the Context Embedder and Context Blocks.
- **Mechanism**: A subset of Transformer Blocks is copied from the DiT to form Context Blocks, which process context tokens and add them back to the main branch—analogous to Res-Tuning.
- **Design Motivation**: Avoids full fine-tuning, achieves faster convergence, and remains plug-and-play with respect to the base model.

### Loss & Training

Standard diffusion denoising loss is used. Training is conducted under either full fine-tuning or Context Adapter tuning, with random conditional dropout applied across different conditions to enable multi-task training.

## Key Experimental Results

### Main Results

**VACE-Benchmark: Automatic Evaluation Across 12 Tasks (Partial, Normalized Average Score)**

| Task Type | Baseline | VACE (Ours) |
|-----------|----------|-------------|
| I2V | CogVideoX-I2V: 73.66 | **74.38** |
| Inpaint | ProPainter: 70.15 | Competitive |
| Depth Control | Task-specific model | Comparable |
| Pose Control | Task-specific model | Comparable |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Without concept decoupling | Confusion between editing and reference tasks |
| Full fine-tuning vs. Adapter | Full fine-tuning slightly better; Adapter converges faster |
| Without random condition dropout | Degraded task generalization |

### Key Findings

1. The unified model achieves performance comparable to task-specific models across all sub-tasks, validating the feasibility of the unified framework.
2. Task combinations (e.g., reference-guided generation + inpainting) are natively supported by VACE but are not achievable with task-specific models.
3. In human evaluations, VACE significantly outperforms most baselines on the temporal consistency dimension.

## Highlights & Insights

1. The VCU design is remarkably elegant, unifying the maximum number of task types with minimal formalization.
2. Concept decoupling is the critical enabler—explicitly separating editing and reference information substantially improves convergence and output quality.
3. VACE is the first all-in-one generation-and-editing model in the video domain, representing pioneering work in this direction.

## Limitations & Future Work

1. Temporal consistency in long-video scenarios remains a room for improvement.
2. Extending the range of supported control signals (e.g., depth, pose) requires additional training data.
3. The scale of the user study is limited.

## Related Work & Insights

- ACE and OmniGen achieve unified generation and editing in the image domain; VACE extends this paradigm to video.
- Key insight: The bottleneck for modality unification lies not in complex architectural design but in the elegance of the input interface.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★★ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★☆ |
| Value | ★★★★★ |

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)
- [\[ICCV 2025\] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](dive_taming_dino_for_subject-driven_video_editing.md)
- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)

<!-- RELATED:END -->
