---
title: >-
  [Paper Note] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation
description: >-
  [CVPR 2026][Video Generation][Human video generation] PoseGen achieves dual condition injection (token-level appearance + channel-level pose) via in-context LoRA finetuning…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Human video generation"
  - "pose control"
  - "LoRA finetuning"
  - "long video generation"
  - "diffusion models"
date: 2026-05-08
content_hash: 8d228be58c3b8e9e
---

# PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation

**Conference**: CVPR 2026 Findings  
**arXiv**: [2508.05091](https://arxiv.org/abs/2508.05091)  
**Code**: [https://github.com/Jessie459/PoseGen](https://github.com/Jessie459/PoseGen)  
**Area**: Video Generation
**Keywords**: Human video generation, pose control, LoRA finetuning, long video generation, diffusion models

## TL;DR
PoseGen achieves dual condition injection (token-level appearance + channel-level pose) via in-context LoRA finetuning, and proposes a segmented interleaved generation strategy (KV sharing + pose-aware frame interpolation) to generate high-fidelity long human videos using only 33 hours of training data.

## Background & Motivation
1. **Background**: Controllable video generation based on diffusion models has made significant progress, yet serious challenges remain in identity preservation, motion accuracy, and video duration.
2. **Limitations of Prior Work**: (i) Identity drift: character appearance degrades over time; (ii) Imprecise motion: accurate motion control often introduces visual artifacts; (iii) Duration constraints: most methods are limited to short clips under 10 seconds, with long-form generation causing severe cumulative errors.
3. **Key Challenge**: Existing methods either require large-scale private datasets (>10K hours) or rely on complex architectural designs (e.g., dedicated pose encoders), making it difficult to balance efficiency, data requirements, and generation quality.
4. **Goal**: Design an efficient, low-data framework for long human video generation while maintaining identity consistency and motion accuracy.
5. **Key Insight**: Leverage the parameter efficiency of LoRA to achieve dual condition injection with minimal architectural modification; design a long video generation strategy that requires no architectural changes.
6. **Core Idea**: A dual conditioning mechanism that injects appearance along the token dimension and pose along the channel dimension, combined with KV-sharing-based segmented interleaved generation for long video synthesis.

## Method

### Overall Architecture
Built upon the pretrained video diffusion model Wan2.1, the framework employs two LoRA modules with distinct roles: the first optimizes the generation of non-overlapping segments, while the second focuses on stitching adjacent segments to achieve temporally coherent long videos.

### Key Designs

1. **In-Context LoRA Dual Conditioning Mechanism**:
    - **Function**: Simultaneously achieves appearance identity preservation and pose-driven motion control.
    - **Mechanism**: (1) **Motion control**: Skeleton pose maps and hand surface normals are used as control signals, concatenated with the noisy video along the channel dimension in latent space. Hand normals provide rich geometric cues to handle complex scenarios such as hand occlusion. (2) **Reference injection**: Reference images are encoded into VAE latent representations and concatenated with the noisy latents along the token dimension; image and video tokens are processed through shared DiT block parameters, with LoRA applied to self-attention, cross-attention, and feed-forward layers.
    - **Design Motivation**: Channel-level pose injection eliminates the need for an additional heavyweight pose encoder; token-level appearance injection leverages the cross-token interaction capability of DiT self-attention, avoiding dedicated identity encoding modules.

2. **Segmented Interleaved Generation Strategy**:
    - **Function**: Overcomes video duration limitations to generate temporally coherent long videos.
    - **Mechanism**: The process proceeds in two steps — (1) multiple non-overlapping short segments are first generated, with background consistency maintained by caching and reusing Key-Value (KV) pairs from the self-attention layers of the source segment; (2) a second LoRA module then stitches adjacent segments into a continuous video via pose-aware frame interpolation. Binary masks are introduced to specify which frames require synthesis.
    - **Design Motivation**: Directly generating long videos leads to severe cumulative errors; overlap-based fusion methods are prone to boundary inconsistencies. KV sharing provides an implicit background consistency constraint.

3. **Hand Surface Normal Assistance**:
    - **Function**: Improves generation quality in hand regions.
    - **Mechanism**: Hand normals are estimated using a surface normal prediction model combined with a body part segmentation model, serving as an auxiliary control signal beyond skeleton pose.
    - **Design Motivation**: Hands are the most quality-degraded region in video generation due to high-frequency textures and rapid motion. Normals provide richer geometric information than skeletons and are more robust than mesh estimation in scenarios involving hand occlusion.

### Loss & Training
Standard diffusion model denoising loss, with only LoRA parameters trained. The model is trained on a 33-hour video dataset, far less than comparable methods (>10K hours).

## Key Experimental Results

### Main Results

| Dataset / Metric | Ours | DreamActor-M1 | VACE | Notes |
|------------------|------|---------------|------|-------|
| Identity fidelity | Best | 2nd | — | Face similarity metric |
| Pose accuracy | Best | — | 2nd | Pose error metric |
| Temporal consistency | Best | — | — | Long-video FVD |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full PoseGen | Best | Complete model |
| w/o hand normals | Degraded hand quality | Normals are critical for hand generation |
| w/o KV sharing | Background inconsistency | KV caching maintains background coherence |
| w/o segmented interleaving | Poor long-video quality | Interleaving strategy is essential for long videos |

### Key Findings
- Only 33 hours of data suffices to outperform methods trained on >10K hours, demonstrating the high parameter efficiency of in-context LoRA.
- The introduction of hand surface normals significantly improves hand generation quality, particularly in complex scenarios such as interlocked fingers.
- The KV sharing mechanism effectively maintains background consistency across non-overlapping segments, serving as a critical component for long video generation.

## Highlights & Insights
- **Extremely low data requirement** (33 hours vs. >10K hours) is the most prominent advantage, substantially lowering the barrier to practical deployment.
- The **dual-dimension condition injection** design is elegantly simple: channel-level for motion control and token-level for appearance preservation, each exploiting the most natural dimension.
- The segmented interleaving strategy is transferable to other scenarios requiring long video generation.

## Limitations & Future Work
- The method relies on pretrained pose estimation and surface normal prediction models, whose accuracy directly affects final output quality.
- Subtle discontinuities may still appear at transition regions between non-overlapping segments.
- Only single-person scenarios are supported; multi-person interaction scenes remain unexplored.

## Related Work & Insights
- **vs. DreamActor-M1**: Employs complex attention mechanisms for identity feature injection; this work achieves superior results with simpler in-context concatenation.
- **vs. AnimateDiff**: Requires a dedicated motion module; this work achieves motion control directly via channel-wise concatenation.
- **vs. MAGI-1 / SkyReels-V2**: These methods require training from scratch, whereas this work achieves comparable capabilities by finetuning only LoRA parameters.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of dual-dimension condition injection and interleaved generation is novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive quantitative and qualitative evaluation, though publicly available benchmarks are relatively limited.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear with well-motivated design choices.
- **Value**: ⭐⭐⭐⭐ An efficient human video generation solution with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[CVPR 2026\] PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation](pam_a_pose-appearance-motion_engine_for_sim-to-real_hoi_video_generation.md)
- [\[CVPR 2026\] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)
- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](../../ICLR2026/video_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)

</div>

<!-- RELATED:END -->
