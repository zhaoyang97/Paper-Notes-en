---
title: >-
  [Paper Note] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context
description: >-
  [CVPR 2026][Video Generation][scene-consistent video generation] This paper proposes the Geometry-as-Context (GaC) framework, which replaces the non-differentiable operators (3D reconstruction + rendering) in reconstruction-based scene video generation with a unified autoregressive video generation model. By embedding geometric information (depth maps) as interleaved context into the generation sequence, GaC enables end-to-end training and mitigates accumulated errors.
tags:
  - CVPR 2026
  - Video Generation
  - scene-consistent video generation
  - geometry context
  - autoregressive generation
  - camera control
  - 3D reconstruction
date: 2026-05-08
content_hash: 81a0d4ec9beedf8d
---

# Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context

**Conference**: CVPR 2026
**arXiv**: [2602.21929](https://arxiv.org/abs/2602.21929)
**Code**: None
**Area**: Video Generation
**Keywords**: scene-consistent video generation, geometry context, autoregressive generation, camera control, 3D reconstruction

## TL;DR

This paper proposes the Geometry-as-Context (GaC) framework, which replaces the non-differentiable operators (3D reconstruction + rendering) in reconstruction-based scene video generation with a unified autoregressive video generation model. By embedding geometric information (depth maps) as interleaved context into the generation sequence, GaC enables end-to-end training and mitigates accumulated errors.

## Background & Motivation

Scene-consistent video generation aims to explore 3D scenes along camera trajectories while maintaining high 3D consistency. Existing methods fall into two categories:
- **Video-based methods** (CameraCtrl, VMem, etc.): rely solely on video models to maintain consistency; memory retrieval struggles with complex scenes and large camera motions.
- **Reconstruction-based methods** (SceneScape, ViewCrafter, GEN3C, etc.): iteratively execute "geometry estimation → 3D reconstruction → rendering → inpainting," but suffer from two fundamental issues:
  1. **Non-differentiable operators**: back-projection and rendering operations in inverse rendering are non-differentiable, blocking gradient propagation.
  2. **Non-end-to-end training**: geometry prediction and image inpainting rely on separate models, so accumulated errors cannot be mitigated through learning.

Unlike accumulated errors in long-range video generation that can be alleviated via autoregressive training, accumulated errors in reconstruction-based methods are difficult to eliminate due to non-differentiable operations and model separation. This constitutes the core problem addressed in this paper.

## Method

### Overall Architecture

GaC "flattens" the iterative pipeline of reconstruction-based methods into a single autoregressive video generation framework: a unified DiT model handles geometry estimation, viewpoint transformation simulation, and image inpainting simultaneously. The input sequence interleaves RGB frames and geometry frames: $\{I_i, \text{<Geometry>}, G_i, \text{<Image>}, I_{i+1}, \cdots\}$, where text tokens instruct the model whether to generate geometry or RGB next.

### Key Designs

1. **Geometry as Context (Variant #1)**: Simplifies the original four-step iteration (geometry estimation → back-projection → rendering → inpainting) to: $\{G_i, I_{i+1}\} = \varrho(\{I_i, G_i\}, P_{i+1})$. The model first estimates the geometry $G_i$ of the current frame, then generates the next RGB frame based on $G_i$ and the target pose $P_{i+1}$. Incorporating geometry context: (a) shortens sequence length for improved efficiency; (b) endows the model with 3D awareness to enhance scene consistency; (c) the large modality gap between RGB and geometry helps the model distinguish between tasks.

2. **Camera Gated Attention (CGA)**: Enhances the model's utilization of camera pose. The Plücker ray-encoded camera pose $r_i$ is added to the self-attention query, and a gating matrix is generated to modulate the attention output:

   - $\{Q_{res}, Gate\} = \text{Linear}_2(Q + r_i)$
   - $O = \text{SDPA}(Q + Q_{res}, K, V)$
   - $O = \text{Linear}_3(O * \sigma(Gate))$

   This design enables the model to distinguish the different roles of camera pose in geometry prediction vs. novel view synthesis.

3. **Geometry Dropout**: During training, geometry context in the interleaved sequence is randomly dropped at rate $r$; dropped frames degrade to pure image-to-image generation (Variant #3). Benefits: (a) reduces sequence length to improve training efficiency; (b) allows inference to produce RGB outputs without geometry prediction; (c) the model maintains scene consistency with or without geometry context. Training time is halved from 24 s/step to 11 s/step, and inference time is halved from 4.6 s/img to 2.2 s/img, with negligible performance degradation.

### Loss & Training

- Base model: Bagel-7B (supporting text-image interleaved modeling)
- Training data: RealEstate10K (66,033 video clips)
- 8-frame sequence training; the first 1–4 frames serve as context views, the remaining as target views
- Every 4 consecutive views are tiled into a grid frame to enhance consistency (resolution $640 \times 352$)
- Images encoded with FLUX-VAE
- Trained on 8× H100 GPUs for 40,000 steps (~2 days)
- Context-as-memory strategy used at inference to select context views; no classifier-free guidance

## Key Experimental Results

### Main Results

| Dataset | Metric | GaC (Ours) | Voyager | GEN3C | ViewCrafter |
|---------|--------|------------|---------|-------|-------------|
| RE10K | PSNR↑ | **19.01** | 18.70 | 18.12 | 16.72 |
| RE10K | SSIM↑ | **0.656** | 0.616 | 0.624 | 0.585 |
| RE10K | LPIPS↓ | **0.354** | 0.395 | 0.402 | 0.417 |
| RE10K | FID↓ | **55.76** | 65.12 | 66.20 | 80.47 |
| RE10K | $R_{err}$↓ | **0.024** | 0.035 | 0.027 | 0.022 |
| RE10K | $T_{err}$↓ | **0.270** | 0.596 | 0.344 | 0.327 |
| T&T | PSNR↑ | **15.77** | 15.24 | 15.32 | 12.59 |
| RE10K (round-trip) | PSNR↑ | **16.34** | 15.80 | 15.28 | 15.77 |
| RE10K (round-trip) | FID↓ | **64.31** | 79.81 | 80.03 | 72.14 |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | $T_{err}$↓ | Note |
|---------------|-------|-------|--------|------|-----------|------|
| None (Variant #3) | 16.34 | 0.551 | 0.412 | 89.03 | 0.351 | No geometry context |
| Warped img (V#2) | 18.33 | 0.671 | 0.383 | 59.12 | 0.299 | Rendered image as context |
| Geometry (V#1) | **19.01** | 0.656 | **0.354** | **55.76** | **0.270** | Geometry as context |
| w/o CGA | 18.57 | 0.581 | 0.461 | 68.42 | 0.469 | CGA removed |
| w/ CGA | **19.01** | **0.656** | **0.354** | **55.76** | **0.270** | Full method |
| w/o Geo Dropout | 19.23 | 0.660 | 0.342 | 57.18 | 0.248 | No dropout (marginally better but 2× slower) |
| w/ Geo Dropout | 19.01 | 0.656 | 0.354 | 55.76 | 0.270 | ~2× speedup |

### Key Findings

- Geometry as context vs. no context: PSNR improves by 2.67 and FID decreases by 33.27, demonstrating the critical role of explicit 3D information.
- CGA reduces translation error $T_{err}$ from 0.469 to 0.270 (a 42% reduction), substantially improving camera control precision.
- Geometry Dropout achieves ~2× speedup in both training and inference with negligible performance loss.
- Depth maps vs. point maps as geometry: performance is comparable, but depth maps are slightly superior (smaller modality gap to natural images, easier for the VAE to encode).
- In round-trip trajectory evaluation, GaC faithfully recovers objects upon return (e.g., a disappeared monitor), demonstrating long-range 3D memory capability.

## Highlights & Insights

- **Elegant unified framework**: Flattening the iterative reconstruction pipeline into a single autoregressive DiT model fundamentally resolves the issues of non-differentiable operations and non-end-to-end training.
- **Geometry Dropout achieves dual benefits**: It reduces computational cost while enabling the model to flexibly choose whether to output geometry information at inference time.
- **CGA is an elegant design**: Query modulation combined with gated output allows a single model to distinguish the role of camera pose across different sub-tasks.
- **Round-trip trajectory robustness**: GaC demonstrates strong scene memory and consistency on forward-and-return trajectories.

## Limitations & Future Work

- All methods exhibit significant performance degradation on round-trip trajectories; long-range context memory strategies require further improvement.
- Training exclusively on RealEstate10K limits generalization to more diverse scenes (outdoor, in-the-wild), necessitating more varied data.
- The resolution of $640 \times 352$ is relatively low; high-resolution scene generation remains unexplored.
- FID on Tanks-and-Temples under round-trip trajectories is inferior to Voyager, indicating room for improvement in large-motion scenarios.
- The base model Bagel-7B is large, and inference cost remains non-trivial (2.2 s/img).

## Related Work & Insights

- **ViewCrafter**: An iterative method combining point clouds and video diffusion; the unified framework proposed in this paper is more elegant and incurs smaller accumulated errors.
- **GEN3C/Voyager**: Introduce point clouds/3DGS as 3D representations but remain constrained by non-differentiable rendering.
- **ReCamMaster**: A camera control method based on frame-dimension concatenation; GaC inherits this idea while incorporating geometry context.
- **Insights**: The paradigm of "internalizing non-differentiable operations as capabilities of the generative model" is generalizable to a broader range of 3D vision tasks; text-guided multi-task scheduling (geometry vs. RGB generation) constitutes an effective design paradigm for interleaved multimodal models.

## Rating

- Novelty: ⭐⭐⭐⭐ Flattening the iterative reconstruction pipeline into autoregressive generation is an elegant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, round-trip trajectories, and thorough ablations, though training data is limited in diversity.
- Writing Quality: ⭐⭐⭐⭐ Motivation is thoroughly analyzed, variant analysis is clear, and algorithmic descriptions are complete.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for scene video generation; the end-to-end philosophy has broad applicability.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors](orbital_video_3d_foundation_priors.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](../../ICLR2026/video_generation/geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[CVPR 2026\] Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)
- [\[ICCV 2025\] Long Context Tuning for Video Generation](../../ICCV2025/video_generation/long_context_tuning_for_video_generation.md)

</div>

<!-- RELATED:END -->
