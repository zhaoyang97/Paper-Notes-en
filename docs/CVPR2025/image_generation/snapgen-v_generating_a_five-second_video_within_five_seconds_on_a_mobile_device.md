---
title: >-
  [Paper Note] SnapGen-V: Generating a Five-Second Video within Five Seconds on a Mobile Device
description: >-
  [CVPR 2025][Image Generation][Mobile Deployment] SnapGen-V proposes a comprehensive acceleration framework for mobile video diffusion models. By pruning an efficient spatial backbone, determining temporal layer design via a joint latency-memory architecture search, and utilizing specialized adversarial fine-tuning to reduce denoising steps to 4, the model finishes generating a 5-second video in under 5 seconds on an iPhone 16 with 0.6B parameters. This represents the first wo…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Mobile Deployment"
  - "Video Diffusion Models"
  - "Architecture Search"
  - "Adversarial Distillation"
  - "Model Acceleration"
date: 2026-05-08
content_hash: e43e2016298563d9
---

# SnapGen-V: Generating a Five-Second Video within Five Seconds on a Mobile Device

**Conference**: CVPR 2025  
**arXiv**: [2412.10494](https://arxiv.org/abs/2412.10494)  
**Code**: [https://snap-research.github.io/snapgen-v/](https://snap-research.github.io/snapgen-v/)  
**Area**: Image Generation / Video Generation  
**Keywords**: Mobile Deployment, Video Diffusion Models, Architecture Search, Adversarial Distillation, Model Acceleration

## TL;DR

SnapGen-V proposes a comprehensive acceleration framework for mobile video diffusion models. By pruning an efficient spatial backbone, determining temporal layer design via a joint latency-memory architecture search, and utilizing specialized adversarial fine-tuning to reduce denoising steps to 4, the model finishes generating a 5-second video in under 5 seconds on an iPhone 16 with 0.6B parameters. This represents the first work to achieve real-time text-to-video generation on mobile devices.

## Background & Motivation

**Background**: Video diffusion models (e.g., CogVideoX, Sora) have achieved impressive progress in generation quality, but they possess a massive number of parameters (billions) and suffer from extremely slow generation speeds (requiring minutes on an A100 GPU), confining their deployment to cloud servers.

**Limitations of Prior Work**: (1) Video models require substantially more parameters than image models to model motion, and the token count arising from spatiotemporal resolution far exceeds that of images; (2) All existing video diffusion models (including the smallest AnimateDiff and SVD) trigger Out-Of-Memory (OOM) errors on mobile devices; (3) Acceleration methods for image diffusion models cannot be directly transferred to video domains because the design space and computational characteristics of temporal modeling layers are entirely different.

**Key Challenge**: Content creators require instantaneous video generation on mobile devices, but the computational and memory requirements of video diffusion models far exceed the capabilities of mobile chips.

**Goal**: To design the first text-to-video diffusion model capable of running in real-time on mobile devices.

**Key Insight**: Systematically compressing the model across three dimensions: spatial architecture (pruning the image backbone), temporal architecture (searching for the optimal combination of temporal layers), and sampling steps (adversarial distillation to 4 steps).

**Core Idea**: Starting from a pre-trained image model, pruning is first applied to obtain an efficient spatial backbone. Then, an evolutionary search under mobile latency and memory constraints is performed over 6 temporal layer candidates (Conv1D/3D, SelfAttn1D/3D, CrossAttn1D/3D) to determine the optimal temporal design. Finally, joint image-video adversarial fine-tuning is leveraged to achieve 4-step generation.

## Method

### Overall Architecture

A three-stage framework: (1) Pruning SD v1.5 to obtain an efficient UNet image backbone with $2.5\times$ compression; (2) Performing evolutionary architecture search over temporal layer types, positions, and counts to find the Pareto-optimal mobile spatiotemporal UNet; (3) Utilizing video-specific adversarial fine-tuning to distill the searched model from 25 steps to 4 steps, while simultaneously eliminating Classifier-Free Guidance (CFG).

### Key Designs

1. **Joint Latency-Memory Architecture Search for Mobile Devices**:

    - **Function**: Searching for the optimal combination among 6 temporal layer candidates to satisfy mobile OOM and latency constraints.
    - **Mechanism**: First, a lookup table is constructed to record the latency and memory footprint of each temporal layer type under different spatiotemporal resolutions. After excluding OOM states, the spatial backbone is frozen, and search is restricted to the type (Conv/Attn, 1D/3D), position (which resolution level), and quantity of temporal layers. Candidate architectures are trained for 20K steps using pre-computed video latents and then evaluated on VBench. Evolutionary search is used to obtain the latency-quality Pareto-optimal solution.
    - **Design Motivation**: The computational characteristics of different temporal layers vary dramatically across distinct resolutions: 3D attention is cost-effective at low resolutions but infeasible at high resolutions, whereas 1D attention complexity grows linearly rather than quadratically with spatial resolution. A hybrid combination is therefore necessary.

2. **Joint Image-Video Adversarial Fine-Tuning**:

    - **Function**: Reducing the number of denoising steps from 25 to 4 while preserving video quality.
    - **Mechanism**: The generator is initialized with the pre-trained text-to-video model. The discriminator utilizes a frozen UNet encoder as its backbone, with spatiotemporal discriminator heads (spatial ResBlock + temporal self-attention) appended after each downsampling block; only the parameters of these heads are updated. A key innovation is the support of joint image-video training: the discriminator heads process both image and video data simultaneously, where image data helps maintain texture quality and video data ensures temporal consistency. The framework is trained using Rectified Flow on 4 fixed timesteps.
    - **Design Motivation**: Compact models (0.6B) exhibit less trajectory redundancy than large models, making direct application of existing distillation methods ineffective. The distribution-level supervision provided by adversarial loss is more suitable for few-step generation than point-wise losses.

3. **VAE Decoder Compression**:

    - **Function**: Eliminating the VAE decoding bottleneck.
    - **Mechanism**: With a frozen encoder, temporal and spatial decoders are pruned separately. The original temporal decoder latency is reduced from 23,100 ms to 210 ms, and the spatial decoder from 4,100 ms to 330 ms, achieving a $50\times$ speedup with negligible quality degradation.
    - **Design Motivation**: VAE decoding on mobile devices also represents a significant bottleneck.

### Loss & Training

Three-step training: (1) Image backbone pruning utilizes knowledge distillation; (2) After temporal layer search, joint image-video training is conducted for 100K steps using Flow Matching loss; (3) Adversarial fine-tuning employs the generator's Rectified Flow loss and the discriminator's hinge loss, trained with a fixed set of 4 inference steps. Timesteps are sampled from a logit-normal distribution.

## Key Experimental Results

### Main Results

| Model | Params | Steps | A100 (s) | iPhone (s) | VBench ↑ |
|------|-------|------|---------|-----------|---------|
| CogVideoX-2B | 1.6B | 50 | 54.09 | ✗ | 80.91 |
| AnimateDiff-V2 | 1.2B | 25 | 9.04 | ✗ | 80.27 |
| AnimateDiffLCM | 1.2B | 4 | 1.77 | ✗ | 79.42 |
| OpenSora-v1.2 | 1.2B | 30 | 31.00 | ✗ | 79.76 |
| SnapGen-V | **0.6B** | **4** | **0.47** | **4.12** | **81.14** |

### Ablation Study

| Component | VBench ↑ | iPhone Latency |
|------|---------|-----------|
| Without Temporal Search (All 1D Attn) | 79.5 | 5.8s |
| Searched Architecture | 80.8 | 4.5s |
| + Adversarial Fine-Tuning (4 steps) | 81.14 | 4.12s |

### Key Findings

- SnapGen-V is the **only video diffusion model capable of running on mobile devices**, whereas all other models (including AnimateDiff with the fewest parameters) trigger OOM errors.
- On an A100, SnapGen-V takes only **0.47 seconds** (4 steps), which is 115x faster than CogVideoX while achieving a higher VBench score.
- Architecture search reveals that high-resolution stages should utilize 1D temporal attention or Convolution, whereas low-resolution stages can leverage the more expressive 3D attention; this hybrid scheme is difficult to discover via manual design.
- Joint image-video adversarial fine-tuning improves VBench by 0.5+ compared to video-only fine-tuning, demonstrating that image data is critical for maintaining texture quality.

## Highlights & Insights

- **"The first mobile-end video diffusion model"** is a milestone, proving the viability of lightweight video generation.
- **The architecture search paradigm is highly instructive**—the efficiency profiles of different temporal layers vary widely across multi-resolution stages, making manually designed combinations sub-optimal.
- **The discriminator design is highly clever**—it reuses the pre-trained UNet encoder as a frozen feature extractor, training only the lightweight heads.

## Limitations & Future Work

- Based on the UNet architecture; the feasibility of deploying DiT on mobile platforms is left unexplored (the quadratic token complexity of DiT makes deployment particularly challenging).
- Currently supports only one-off generation of complete videos, without supporting streaming or continuous real-time generation.
- The generation quality of the 0.6B model in complex scenarios (multi-object interactions, fine facial details) may lag behind that of larger models.
- The latency of 4.12 seconds on an iPhone 16 Pro Max is close to real-time but still offers room for optimization.

## Related Work & Insights

- **vs. CogVideoX / OpenSora**: These large models pursue maximum visual fidelity but can only run on GPU servers; SnapGen-V proves that a reasonable quality-efficiency trade-off is achievable on mobile devices.
- **vs. AnimateDiff**: Although AnimateDiff is relatively small, its 1.2B parameters combined with 25 steps still prevent mobile execution; SnapGen-V achieves a qualitative leap through systematic compression.
- **Inspirations for Image Diffusion**: The discriminator design incorporating joint image-video training in adversarial fine-tuning can be generalized to other multimodal distillation scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ First mobile video diffusion model, featuring a highly systematic three-stage acceleration framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive quantitative comparison via VBench, with authentic and reliable mobile latency data.
- Writing Quality: ⭐⭐⭐⭐ Framework diagrams are clear, and logic across different stages is coherent.
- Value: ⭐⭐⭐⭐⭐ Pioneering work with significant value for democratizing the deployment of video generation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents](../../ICCV2025/image_generation/reducio_generating_1k_video_within_16_seconds_using_extremely_compressed_motion_.md)
- [\[CVPR 2025\] MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices](mobileportrait_real-time_one-shot_neural_head_avatars_on_mobile_devices.md)
- [\[CVPR 2025\] ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions](showhowto_generating_scene-conditioned_step-by-step_visual_instructions.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)

</div>

<!-- RELATED:END -->
