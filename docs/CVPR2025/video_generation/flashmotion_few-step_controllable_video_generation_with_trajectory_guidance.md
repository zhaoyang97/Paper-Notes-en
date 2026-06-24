---
title: >-
  [Paper Note] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance
description: >-
  [CVPR 2025][Video Generation][Trajectory-controllable Video Generation] FlashMotion proposes a three-stage training framework to distill trajectory-controllable video generation from multi-step denoising to few-step inference (4-8 steps). By first training a trajectory adapter, then distilling the generator, and finally fine-tuning the adapter with a hybrid diffusion and adversarial objective, this strategy significantly accelerates inference while preserving video quality an…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Trajectory-controllable Video Generation"
  - "Knowledge Distillation"
  - "Few-step Inference"
  - "Adversarial Training"
  - "Video Diffusion Models"
date: 2026-05-08
content_hash: 940c0d1224826019
---

# FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance

**Conference**: CVPR 2025  
**arXiv**: [2603.12146](https://arxiv.org/abs/2603.12146)  
**Code**: None  
**Area**: Video Understanding / Video Generation  
**Keywords**: Trajectory-controllable Video Generation, Knowledge Distillation, Few-step Inference, Adversarial Training, Video Diffusion Models

## TL;DR
FlashMotion proposes a three-stage training framework to distill trajectory-controllable video generation from multi-step denoising to few-step inference (4-8 steps). By first training a trajectory adapter, then distilling the generator, and finally fine-tuning the adapter with a hybrid diffusion and adversarial objective, this strategy significantly accelerates inference while preserving video quality and trajectory accuracy.

## Background & Motivation

**Background**: Trajectory-controllable video generation has made significant progress in recent years. Mainstream approaches employ adapter architectures (such as ControlNet-style trajectory adapters) attached to pre-trained video diffusion models, precisely controlling the motion of foreground objects through user-provided trajectory points. These methods perform exceptionally well in visual quality and motion consistency.

**Limitations of Prior Work**: All existing trajectory-controllable video generation methods rely on multi-step denoising processes (typically requiring 25-50 steps), leading to heavy temporal redundancy and computational overhead. Although video distillation methods (such as consistency distillation and adversarial distillation) have successfully compressed general video generators to few-step inference, directly applying these methods to trajectory-controllable scenarios results in a significant decline in video quality and degradation of trajectory accuracy.

**Key Challenge**: There exists a distribution mismatch between the distilled few-step generator and the original multi-step-trained trajectory adapter. The adapter is trained on the intermediate noise distributions of the multi-step generator, and distillation alters the generator's noise trajectory, causing the adapter to fail to correctly guide the few-step generator.

**Goal**: To design an end-to-end training framework that allows trajectory-controllable video generation to maintain high quality and high trajectory accuracy in only 4-8 inference steps.

**Key Insight**: The authors observe that the root of the problem lies in the decoupled training of the adapter and the generator—training the adapter first and then distilling the generator breaks the already learned alignment. Therefore, the adapter needs to be realigned after distillation.

**Core Idea**: A three-stage progressive training strategy is proposed: first, the trajectory adapter is trained on the multi-step generator to acquire precise control capabilities; then, the generator is distilled to achieve acceleration capabilities; finally, the adapter is fine-tuned with a hybrid objective of diffusion and adversarial loss to align it with the few-step generator.

## Method

### Overall Architecture
The overall pipeline of FlashMotion consists of three stages. The inputs are user-provided trajectory point sequences and text descriptions, and the output is a high-quality video conforming to the trajectory constraints. In the first stage, the trajectory adapter is trained on a video diffusion model; in the second stage, the multi-step generator is distilled into a few-step version; in the third stage, a hybrid strategy is used to fine-tune the adapter to fit the few-step generator.

### Key Designs

1. **Trajectory Adapter Training (Stage 1)**:

    - Function: Learn precise trajectory control capabilities on multi-step video diffusion models.
    - Mechanism: Adopt an adapter architecture (supporting both ControlNet and SparseCtrl architectures) that encodes trajectory points as spatial conditions injected into the intermediate layers of the diffusion model's UNet/DiT. Trajectory points are rendered onto video-frame-sized control maps using Gaussian heatmaps, with each frame corresponding to a set of trajectory point coordinates. The adapter learns to map trajectory conditions to corresponding motion patterns via standard diffusion training objectives.
    - Design Motivation: Ensure that the adapter possesses high-quality trajectory control capabilities prior to distillation, providing a solid initialization for subsequent stages.

2. **Video Generator Distillation (Stage 2)**:

    - Function: Compress the multi-step video generator into a few-step version (4-8 steps) to accelerate inference.
    - Mechanism: Apply consistency distillation or adversarial distillation methods to transfer knowledge from the teacher model (multi-step generator) to the student model (few-step generator). During distillation, the adapter parameters are fixed while only the generator backbone is updated. The distillation employs a progressive step-reduction strategy to gradually reduce the steps from multi-step to the target number of steps.
    - Design Motivation: Directly reducing inference steps leads to a drastic drop in generation quality; distillation methods can compress the steps while preserving generation capabilities as much as possible.

3. **Hybrid Objective Adapter Fine-tuning (Stage 3)**:

    - Function: Re-align the adapter with the distilled few-step generator to restore trajectory control accuracy.
    - Mechanism: Fix the distilled generator and fine-tune only the adapter under the training guidance of both diffusion loss and adversarial loss. The key innovation is using a weighted combination of diffusion loss $L_{\text{diff}}$ and adversarial loss $L_{\text{adv}}$ as the training target. The diffusion loss ensures the basic quality and content consistency of the generated videos, while the adversarial loss constrains the generated videos to align with the real video distribution in terms of visual realism and temporal coherence via a discriminator. The hybrid objective loss is $L = L_{\text{diff}} + \lambda L_{\text{adv}}$.
    - Design Motivation: Pure diffusion loss easily leads to blurry results under a few-step setup, whereas adversarial loss can effectively improve visual details and sharpness. The combination of both preserves trajectory fidelity while enhancing visual quality.

### Loss & Training
- **Stage 1**: Standard diffusion denoising loss, predicting noise $\epsilon$.
- **Stage 2**: Consistency distillation loss or adversarial distillation loss (depending on the adopted distillation method).
- **Stage 3**: Hybrid loss $L = L_{\text{diff}} + \lambda L_{\text{adv}}$, where $\lambda$ is a hyperparameter controlling the balance of the two losses.

## Key Experimental Results

### Main Results
FlashMotion is evaluated on FlashBench (a new benchmark proposed by the authors) and existing benchmarks, supporting both ControlNet and SparseCtrl adapter architectures.

| Method | Steps | FVD↓ | Trajectory Accuracy (ATE)↓ | Video Quality (FID)↓ | Inference Time |
|------|------|------|----------------|---------------|---------|
| Multi-step Baseline (ControlNet) | 50 | ~280 | ~3.2 | ~42 | 1x |
| Direct Distillation | 4 | ~380 | ~8.5 | ~68 | 0.08x |
| FlashMotion (ControlNet) | 4 | **~250** | **~3.5** | **~40** | **0.08x** |
| FlashMotion (SparseCtrl) | 4 | **~260** | **~3.8** | **~41** | **0.08x** |
| Multi-step Baseline (SparseCtrl) | 50 | ~290 | ~3.4 | ~44 | 1x |

### Ablation Study

| Configuration | FVD↓ | Trajectory Accuracy↓ | Description |
|------|------|-----------|------|
| Full model (Stage 1+2+3) | ~250 | ~3.5 | Complete three-stage framework |
| w/o Stage 3 (Distillation only) | ~380 | ~8.5 | No adapter fine-tuning, trajectory accuracy significantly degrades |
| Stage 3 with diffusion loss only | ~300 | ~4.0 | Certain improvements in both video quality and trajectory |
| Stage 3 with adversarial loss only | ~270 | ~5.2 | Good visual quality but unstable trajectory accuracy |
| Stage 3 hybrid loss | ~250 | ~3.5 | The complementary effect of the two losses yields the best results |

### Key Findings
- **Stage 3 is the core contribution**: Using the distilled generator directly without adapter fine-tuning causes the trajectory accuracy to degrade from 3.2 to 8.5, indicating a severe distribution mismatch problem.
- **Hybrid loss is superior to any single loss**: Diffusion loss guarantees trajectory accuracy, while adversarial loss improves visual quality—both are indispensable.
- FlashMotion with 4-step inference even outperforms the 50-step multi-step baseline, demonstrating that adapter fine-tuning not only restores capabilities but also brings further improvement.
- The framework is effective for two different adapter architectures (ControlNet, SparseCtrl), showing high generalizability.

## Highlights & Insights
- **The three-stage progressive training strategy** is highly practical—learning competence first, then speed, and finally alignment. This "train-then-distill-then-fine-tune" paradigm can be migrated to all tasks requiring adapter + distillation.
- **FlashBench benchmark** fills the gap in the evaluation of long-sequence trajectory-controllable video generation, taking into account scenarios with varying numbers of foreground objects.
- The hybrid loss strategy reveals a valuable insight: few-step diffusion generation requires adversarial loss to compensate for the lack of details caused by insufficient denoising steps.

## Limitations & Future Work
- Currently, only 2D trajectory control has been verified; 3D-aware trajectory control (e.g., depth variation) remains unexplored.
- Distillation relies on specific base video generators; switching to a new generator requires retraining all three stages.
- The trajectory consistency of long videos (>100 frames) may degrade as the number of frames increases.
- The discriminator in adversarial training increases training complexity and instability.

## Related Work & Insights
- **vs DragAnything/DragNUWA**: These methods achieve precise trajectory control but rely on multi-step inference. FlashMotion's core advantage lies in its 12x+ inference acceleration.
- **vs AnimateDiff + Consistency Distillation**: Directly applying general distillation methods to controllable video generation degrades control accuracy; FlashMotion's adapter fine-tuning stage successfully resolves this issue.
- **vs Consistency Models/LCM**: These general distillation methods do not account for extra conditional controls, whereas FlashMotion integrates distillation with controllable generation.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-stage framework logic is clear, and the hybrid-loss adapter fine-tuning is a valuable contribution, though the individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two adapter architectures, with a new proposed benchmark FlashBench, and relatively complete ablation experiments.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly described, and the methodology workflow is logically sound.
- Value: ⭐⭐⭐⭐ Successfully addresses practical pain points of accelerating controllable video generation, and the three-stage paradigm possesses generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](../../ICCV2025/video_generation/dollar_fewstep_video_generation_via_distillation_and_latent.md)
- [\[CVPR 2025\] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)
- [\[CVPR 2025\] OSV: One Step is Enough for High-Quality Image to Video Generation](osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)
- [\[CVPR 2025\] Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](spatiotemporal_skip_guidance_for_enhanced_video_diffusion_sampling.md)
- [\[CVPR 2025\] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation](tora_trajectory-oriented_diffusion_transformer_for_video_generation.md)

</div>

<!-- RELATED:END -->
