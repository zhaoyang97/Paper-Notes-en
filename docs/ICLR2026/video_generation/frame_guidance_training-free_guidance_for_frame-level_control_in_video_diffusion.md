---
title: >-
  [Paper Note] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models
description: >-
  [ICLR 2026][Video Generation][Training-free guidance] This paper proposes Frame Guidance, a training-free frame-level guidance method that enables controllable video generation tasks — including keyframe guidance…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "Training-free guidance"
  - "video diffusion models"
  - "frame-level control"
  - "keyframe generation"
  - "stylized video"
date: 2026-05-08
content_hash: 93143072be64a6ab
---

# Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2506.07177](https://arxiv.org/abs/2506.07177)  
**Code**: [https://frame-guidance-video.github.io/](https://frame-guidance-video.github.io/)  
**Area**: Diffusion Models / Video Generation
**Keywords**: Training-free guidance, video diffusion models, frame-level control, keyframe generation, stylized video

## TL;DR

This paper proposes Frame Guidance, a training-free frame-level guidance method that enables controllable video generation tasks — including keyframe guidance, stylization, and looping video — without modifying the model, via two core components: latent slicing (reducing memory by 60×) and Video Latent Optimization (VLO).

## Background & Motivation

**Growing demand for controllable video generation**: As the quality of video diffusion models improves, users increasingly require fine-grained control over generated content.

**High cost of training-based methods**: Existing approaches typically require fine-tuning large-scale VDMs; as model sizes grow (e.g., Wan 14B), fine-tuning becomes increasingly impractical.

**Limited generality of training-free methods**: Existing training-free methods (e.g., CamTrol, MotionClone) are designed for specific tasks and lack a unified framework.

**Memory bottleneck of Video CausalVAE**: The causal dependency in CausalVAE requires decoding the full sequence to reconstruct a single frame, causing gradient computation to exceed 650 GB of memory.

**Inapplicability of existing guidance strategies to video**: The time-travel trick from image guidance washes out guidance signals in early denoising steps for video.

**Dual objective gap**: No prior method simultaneously satisfies both "model-agnostic + training-free" and "generalizable across multiple tasks."

## Method

### Overall Architecture

Frame Guidance applies gradient-based guidance to selected frames during the inference of a pretrained VDM, achieving efficient and controllable generation through three core components: **Latent Slicing** (efficient decoding), **VLO** (phased optimization strategy), and **task-adaptive loss design**.

### Key Design 1: Latent Slicing

- **Discovery of temporal locality in CausalVAE**: Experiments show that, despite being designed as causal, perturbations in CausalVAE in practice only affect a small number of adjacent latents (temporal locality).
- **Slice decoding**: When reconstructing frame $i$, only a window of 3 latents is decoded rather than the entire sequence.
- **Spatial downsampling**: Latents are spatially downsampled by $2\times$ before decoding for loss computation.
- **Effect**: Memory is reduced by up to **60×**, enabling guidance of large models such as Wan-14B on a single GPU.

### Key Design 2: Video Latent Optimization (VLO)

- **Core insight**: The global layout of video frames is determined within the first few denoising steps; early-stage guidance is most critical for temporal consistency.
- **Phased strategy**:
    - **Early phase** ($t > t_E$): Deterministic update $z_t \leftarrow z_t - \eta \nabla_{z_t} \mathcal{L}_e$, preserving the guidance signal.
    - **Middle phase** ($t_E \geq t > t_L$): Stochastic update (with re-noising) to correct accumulated errors.
    - **Late phase** ($t \leq t_L$): No guidance; free refinement of details.

### Critical Role of Gradient Propagation

- Guidance gradients must propagate through the denoising network $v_\theta$ to influence temporal consistency across the entire video.
- Guidance is applied only to sliced latents, but gradients propagate through the network to all frames.
- Shortcut-based updates (bypassing the network) affect only the guided frames, leading to temporal discontinuities.

### Multi-Task Loss Design

| Task | Loss Function |
|------|--------------|
| Keyframe guidance | $\mathcal{L}_e = \sum_{i \in \mathcal{I}} \|x_*^i - x_{0\vert t}^i\|_2^2$ |
| Stylization | $\mathcal{L}_e = -\sum_{i \in \mathcal{I}} \cos(\Psi(x_{\text{style}}), \Psi(x_{0\vert t}^i))$ |
| Looping video | $\mathcal{L}_e = \|x_{0\vert t}^1 - x_{0\vert t}^L\|_2^2$ |
| General conditioning (depth/edge) | $\mathcal{L}_e = \sum_{i \in \mathcal{I}} \|\Psi(x_*^i) - \Psi(x_{0\vert t}^i)\|_2^2$ |

## Key Experimental Results

### Keyframe Guidance (DAVIS Dataset)

| Method | Training | FID ↓ | FVD ↓ |
|--------|----------|-------|-------|
| CogX-I2V | ✓ | 60.36 | 890.1 |
| TRF (training-free) | ✓ | 62.07 | 923.1 |
| **Ours (CogX, I+F)** | ✓ | 57.62 | 613.4 |
| **Ours (CogX, I+M+F)** | ✓ | 55.60 | 577.1 |
| SVD-Interp (fine-tuned) | ✗ | 63.89 | 800.3 |
| CogX-Interp (fine-tuned) | ✗ | 46.59 | 506.0 |

### Pexels Dataset

| Method | FID ↓ | FVD ↓ |
|--------|-------|-------|
| CogX-I2V | 74.98 | 1122.6 |
| **Ours (Wan-14B, I+M+F)** | 71.63 | 904.8 |
| **Ours (CogX, I+M+F)** | 68.97 | 989.3 |

**Key Findings**: The training-free Frame Guidance surpasses the training-based SVD-Interp on most metrics, and falls only slightly short of the specially fine-tuned CogX-Interp.

## Highlights & Insights

1. **Discovery of temporal locality in CausalVAE**: Despite its causal design, the VAE exhibits temporal locality in practice — a finding that enables the 60× memory reduction.
2. **Phased strategy in VLO**: Unlike the uniform time-travel trick for images, VLO employs deterministic and stochastic phases tailored to the temporal characteristics of video.
3. **Model agnosticism**: Demonstrated effectiveness across three distinct VDMs: CogVideoX, LTX-Video, and Wan-14B.
4. **High flexibility**: Supports arbitrary keyframe positions, diverse conditioning signals, and multiple tasks without per-task training.
5. **Sparse guidance suffices**: Guiding only a small number of frames is sufficient to control the entire video through gradient propagation in the network.

## Limitations & Future Work

1. Inference is slow (constrained to no more than 4× the base model runtime), and guidance steps and step sizes require manual tuning.
2. Keyframe guidance achieves visual similarity rather than pixel-level exact matching.
3. Stylization depends on the quality of specific style encoders such as CSD.
4. For highly dynamic scenes (e.g., fast motion, scene transitions), layout determination in early steps may be insufficient.
5. Guidance conditioned on additional modalities such as audio and text remains unexplored.

## Related Work & Insights

- **Universal Guidance (Bansal et al., 2024)**: The foundational training-free guidance method for images; this paper extends it to video.
- **TRF (Feng et al., 2024)**: Training-free keyframe interpolation for SVD, but lacks generality; Frame Guidance supports a broader range of tasks through frame-level loss design.
- **CogX-Interp**: A fine-tuning-based keyframe interpolation method with higher precision but requiring training.
- Insight: The temporal locality of CausalVAE may be leveraged by other training-free methods, such as video editing and inpainting.

## Rating

- Novelty: ⭐⭐⭐⭐ — Both Latent Slicing and VLO are clever designs tailored to the video setting.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple models, tasks, and datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, in-depth analysis, and excellent figures.
- Value: ⭐⭐⭐⭐⭐ — Highly practical in the era of large models; a significant milestone for training-free methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](../../CVPR2026/video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[ICML 2026\] EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance](../../ICML2026/video_generation/epic_efficient_video_camera_control_learning_with_precise_anchor-video_guidance.md)
- [\[ICML 2026\] Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](../../ICML2026/video_generation/enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)
- [\[ICLR 2026\] Target-Aware Video Diffusion Models](target-aware_video_diffusion_models.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)

</div>

<!-- RELATED:END -->
