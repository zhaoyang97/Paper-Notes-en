---
title: >-
  [Paper Note] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance
description: >-
  [CVPR 2026][Video Generation][Trajectory-controllable video generation] FlashMotion is proposed as the first three-stage training framework for few-step (4-step) trajectory-controllable video generation. By sequentially…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Trajectory-controllable video generation"
  - "few-step distillation"
  - "adversarial training"
  - "diffusion discriminator"
  - "video acceleration"
date: 2026-05-08
content_hash: 079dee125f734cf1
---

# FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance

**Conference**: CVPR 2026
**arXiv**: [2603.12146](https://arxiv.org/abs/2603.12146)
**Code**: [https://github.com/quanhaol/FlashMotion](https://github.com/quanhaol/FlashMotion)
**Area**: Video Understanding / Video Generation
**Keywords**: Trajectory-controllable video generation, few-step distillation, adversarial training, diffusion discriminator, video acceleration

## TL;DR

FlashMotion is proposed as the first three-stage training framework for few-step (4-step) trajectory-controllable video generation. By sequentially training a trajectory adapter, distilling a fast generator, and fine-tuning the adapter via a hybrid adversarial and diffusion loss, the method surpasses existing multi-step approaches in both visual quality and trajectory accuracy under 4-step inference, achieving a 47× speedup.

## Background & Motivation

**Background**: Diffusion model–driven video generation has achieved remarkable progress, particularly in trajectory-controllable video generation, where users specify the motion trajectories of foreground objects (via bounding boxes or segmentation masks) and the model generates videos that follow the prescribed paths. Methods such as MagicMotion, Tora, and LeviTor realize precise motion control by attaching trajectory adapters to pretrained video generation backbones.

**Limitations of Prior Work**: All existing trajectory-controllable methods rely on multi-step denoising inference (50+ steps), resulting in approximately 1,160 seconds (>19 minutes) to generate a 121-frame video. Although video distillation methods (e.g., DMD, LCM, CausVid) can compress general video generation models into few-step variants, directly applying these distillation techniques to trajectory-controllable generation leads to significant degradation in both visual quality and trajectory accuracy.

**Key Challenge**: A multi-step adapter (SlowAdapter) is trained along the progressive denoising trajectory of a multi-step generator (SlowGenerator), where trajectory conditioning guides noise refinement iteratively. The denoising trajectory of a fast generator (FastGenerator) is fundamentally different—completing the entire generation in only 4 steps. Consequently, the SlowAdapter is inherently incompatible with the FastGenerator; naive combination produces color shifts, blurriness, and loss of trajectory control.

**Goal**: Design a training framework that enables the trajectory adapter to function correctly on the fast generator, simultaneously guaranteeing visual quality and trajectory accuracy within 4-step inference.

**Key Insight**: The authors observe that fine-tuning the adapter with a standard diffusion loss restores trajectory accuracy but introduces severe blurriness (since pixel-level supervision cannot ensure distributional consistency), whereas introducing adversarial training eliminates blurriness but degrades trajectory accuracy. Consequently, both losses must be used concurrently with dynamic balancing.

**Core Idea**: A three-stage training pipeline—first train a multi-step adapter, then distill the fast generator, and finally fine-tune the adapter for the fast generator using a hybrid diffusion and adversarial loss strategy.

## Method

### Overall Architecture

The FlashMotion training pipeline consists of three stages. **Stage 1** trains a SlowAdapter on the multi-step video generation backbone (Wan2.2-TI2V-5B) to learn trajectory control. **Stage 2** distills the multi-step generator into a 4-step FastGenerator via DMD. **Stage 3** fine-tunes the SlowAdapter into a FastAdapter compatible with the FastGenerator using a hybrid diffusion and adversarial strategy. At inference time, the FastGenerator combined with the FastAdapter requires only 4 denoising steps to produce high-quality videos with accurate trajectory following.

### Key Designs

1. **Trajectory Adapter Architecture**:

    - **Function**: Injects user-specified motion trajectories (bbox/mask) into the video generation process.
    - **Mechanism**: Two adapter architectures are designed—ControlNet and a lightweight ResNet. The number of adapter blocks matches that of the base DiT. A 3D VAE encoder encodes trajectory maps into the latent space as $Z_{trajectory} \in R^{T/4 \times H/16 \times W/16 \times 48}$; the output of each adapter block is added to the corresponding DiT block via a zero-initialized convolutional layer for trajectory guidance. Training adopts a coarse-to-fine progressive strategy—first training 4.6K steps with segmentation masks, then fine-tuning 5.4K steps with bounding boxes.
    - **Design Motivation**: ControlNet has a larger parameter count (10.28B) but yields more precise trajectory control; ResNet has fewer parameters (5.02B) but faster inference. The two architectures validate the generality of the FlashMotion framework.

2. **Diffusion Discriminator**:

    - **Function**: Distinguishes real from generated videos during adversarial training to eliminate blurring artifacts caused by reliance on the diffusion loss alone.
    - **Mechanism**: The discriminator backbone is cloned from Wan2.2-TI2V-5B with frozen weights; only a newly added attention-based classifier is trained. The classifier receives intermediate DiT features and processes a learnable query token through three attention layers: **semantic self-attention** (integrating first-frame image and text information), **trajectory cross-attention** (attending to trajectory tokens), and **video cross-attention** (attending to video tokens), ultimately producing a real/fake logit.
    - **Design Motivation**: The pure diffusion loss $\mathcal{L}_{diffusion} = \|G_\theta(x_t, t) - x_0^{real}\|^2$ provides only pixel-level supervision and cannot guarantee alignment between the generated and real distributions, leading to blurriness. The discriminator imposes distribution-level constraints to eliminate blur, while the three-layer attention design enables simultaneous awareness of semantic, trajectory, and video information.

3. **Dynamic Diffusion Loss Scale**:

    - **Function**: Balances the gradient magnitudes of the diffusion loss and the adversarial loss.
    - **Mechanism**: The total loss is $\mathcal{L} = \mathcal{L}_{\mathcal{G}} + \lambda \mathcal{L}_{diffusion}$, where $\lambda = \frac{1}{4 \times 10^{-3} \times step + 0.1}$. In early training, $\lambda$ is large so the diffusion loss dominates and ensures trajectory alignment; as training progresses, $\lambda$ decreases gradually so the adversarial loss takes over to improve visual quality.
    - **Design Motivation**: Experiments show that the gradient of the diffusion loss greatly exceeds that of the adversarial loss in early training; equal weighting still leads to blurriness. Dynamic scaling allows each loss to fulfill its role at the appropriate training stage—"get it right first, then get it sharp."

### Loss & Training

- **Stage 1** (SlowAdapter): Standard diffusion denoising loss; 16 GPUs × 10K steps.
- **Stage 2** (FastGenerator): DMD distribution matching loss $\nabla\mathcal{L}_{DMD} = \mathbb{E}[-(s_{real} - s_{fake})\frac{dG_\theta}{d\theta}]$; 16 GPUs × 5.5K steps.
- **Stage 3** (FastAdapter): Hybrid loss $\mathcal{L} = \mathcal{L}_{\mathcal{G}} + \lambda\mathcal{L}_{diffusion}$; discriminator and adapter are updated alternately (1:5 update ratio); only **4 GPUs × 1K steps** are required—extremely lightweight.

## Key Experimental Results

### Main Results (FlashBench)

| Method | Steps | FID↓ | FVD↓ | M IoU↑ | B IoU↑ | Denoising Time (s) |
|--------|------|------|----------|------|------|------|
| MagicMotion | 50 | 20.03 | 138.83 | 68.10 | 73.68 | 1158.63 |
| Wan+ResNet | 50 | 19.03 | 139.61 | 52.19 | 57.76 | 333.00 |
| DMD (ResNet) | 4 | 24.38 | 228.33 | 43.24 | 52.61 | 11.72 |
| LCM (ResNet) | 4 | 26.79 | 462.09 | 55.31 | 60.80 | 11.72 |
| **FlashMotion (ResNet)** | **4** | **15.81** | **108.96** | **63.96** | **70.01** | **11.72** |
| **FlashMotion (ControlNet)** | **4** | **14.35** | **96.08** | **69.15** | **75.38** | **24.44** |

### Ablation Study (FlashBench, ResNet)

| Configuration | FID↓ | FVD↓ | M IoU↑ | B IoU↑ |
|------|---------|------|------|------|
| SlowAdapter applied directly | 22.75 | 168.46 | 49.79 | 56.62 |
| w/o diffusion loss | 18.87 | 161.07 | 52.04 | 58.04 |
| w/o GAN loss | 22.74 | 206.75 | 65.82 | 70.60 |
| w/o dynamic scaling | 26.32 | 210.93 | 65.54 | 69.77 |
| **FlashMotion (full)** | **15.81** | **108.96** | **63.96** | **70.01** |

### Key Findings

- **FlashMotion at 4 steps surpasses MagicMotion at 50 steps**: The ControlNet variant achieves FID 14.35 vs. 20.03 and FVD 96.08 vs. 138.83, while delivering a **47× speedup** (24 s vs. 1158 s).
- **The GAN loss is critical for eliminating blurriness**: Removing the GAN loss causes FVD to surge from 108.96 to 206.75 (~90% degradation), while trajectory accuracy actually improves slightly, indicating that the GAN loss primarily governs visual quality.
- **The diffusion loss is critical for trajectory accuracy**: Removing the diffusion loss reduces M IoU from 63.96 to 52.04, causing generated objects to deviate substantially from their intended trajectories.
- **Dynamic scaling is indispensable**: Fixing $\lambda=1$ raises FID from 15.81 to 26.32 (increased blurriness), confirming that the excessive gradient magnitude of the diffusion loss in early training must be suppressed.
- The ControlNet adapter outperforms the ResNet adapter across all trajectory accuracy and visual quality metrics, but incurs approximately twice the inference time.

## Highlights & Insights

- **Elegant decomposition via three stages**: Decoupling "fast + controllable" into learning control first, learning speed second, and aligning the two last keeps each stage's objective clear. Stage 3 requires only 1K training steps, demonstrating the strong prior provided by the SlowAdapter.
- **Three-layer attention design in the diffusion discriminator**: The separate injection of semantic, trajectory, and video information enables the discriminator to perceive multi-dimensional signals simultaneously, making it more suitable for conditional generation than a simple CNN discriminator. This design is transferable to the acceleration of other conditional video generation tasks.
- **Practical value of dynamic loss scaling**: A simple yet effective solution to the gradient imbalance between the diffusion and adversarial losses, expressed compactly as a single formula, yet with a substantial impact on final performance.

## Limitations & Future Work

- Although FlashBench supports long-video evaluation, it currently relies on an extension of MagicData with limited data diversity.
- Stage 2 distillation depends on a specific DMD method; whether alternative distillation approaches (e.g., CausVid) could yield further improvements remains unexplored.
- The current framework supports only bbox/mask trajectory conditions; more flexible trajectory representations (e.g., point trajectories, semantic descriptions) await future investigation.
- The ControlNet variant, despite superior performance, still has 10.28B parameters, posing challenges for on-device deployment.
- DMD and GAN training under the ControlNet configuration directly causes out-of-memory errors, limiting broader comparisons of distillation methods.

## Related Work & Insights

- **vs. MagicMotion**: MagicMotion achieves precise trajectory control on CogVideoX but requires 50-step inference; FlashMotion achieves 4-step inference on Wan2.2 with superior results.
- **vs. APT/APT2**: The APT family of one-step adversarial distillation is effective for general video generation but does not support trajectory conditioning; FlashMotion's diffusion discriminator design is inspired by APT but incorporates trajectory awareness.
- **vs. Tora/LeviTor**: Tora and LeviTor are built on CogVideoX and SVD, respectively; both underperform FlashMotion in quality and require more inference steps.

## Rating

- Novelty: ⭐⭐⭐⭐ First framework for few-step trajectory-controllable video generation; the three-stage decomposition is well-motivated and the discriminator design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks, two adapter architectures, comprehensive ablations including analysis grouped by object count.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed method description, and informative figures.
- Value: ⭐⭐⭐⭐⭐ 47× speedup with superior quality directly advances the practical deployment of controllable video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](../../ICCV2025/video_generation/dollar_fewstep_video_generation_via_distillation_and_latent.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[CVPR 2026\] SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution](swift_sliding_window_reconstruction_for_few-shot_training-free_generated_video_a.md)
- [\[CVPR 2026\] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)

</div>

<!-- RELATED:END -->
