---
title: >-
  [Paper Note] Versatile Transition Generation with Image-to-Video Diffusion
description: >-
  [ICCV 2025][Video Generation][video transition generation] This paper proposes VTG, a unified video transition generation framework built upon an image-to-video diffusion model. VTG achieves smooth, high-fidelity transitions across four task categories — object morphing, motion prediction, concept blending, and scene transition — via interpolation-based initialization (noise SLERP + LoRA interpolation + text SLERP), bidirectional motion fine-tuning, and DINOv2 representation alignment regularization.
tags:
  - ICCV 2025
  - Video Generation
  - video transition generation
  - image morphing
  - bidirectional motion prediction
  - LoRA interpolation
  - representation alignment regularization
date: 2026-05-08
content_hash: 4d7fd997528e6fce
---

# Versatile Transition Generation with Image-to-Video Diffusion

**Conference**: ICCV 2025
**arXiv**: [2508.01698](https://arxiv.org/abs/2508.01698)
**Code**: [Project Page](https://mwxely.github.io/projects/yang2025vtg/index)
**Area**: Video Generation
**Keywords**: video transition generation, image morphing, bidirectional motion prediction, LoRA interpolation, representation alignment regularization

## TL;DR
This paper proposes VTG, a unified video transition generation framework built upon an image-to-video diffusion model. VTG achieves smooth, high-fidelity transitions across four task categories — object morphing, motion prediction, concept blending, and scene transition — via interpolation-based initialization (noise SLERP + LoRA interpolation + text SLERP), bidirectional motion fine-tuning, and DINOv2 representation alignment regularization.

## Background & Motivation

1. **State of the Field**: Transition video generation encompasses object morphing (DiffMorpher), video frame interpolation (RIFE, etc.), and scene transition (SEINE), yet each method targets a specific task with no unified framework.
2. **Limitations of Prior Work**: (1) Image morphing methods (e.g., DiffMorpher) produce discontinuous static images rather than temporally coherent frames; (2) video frame interpolation yields unnatural transitions under large content discrepancies; (3) existing frameworks address either morphing/motion prediction or scene transitions, but not both.
3. **Root Cause**: High-quality transitions must simultaneously satisfy four criteria: semantic similarity, input fidelity, inter-frame smoothness, and text alignment. Random latent initialization in I2V diffusion models causes inter-frame "flickering," and support for only forward motion prediction creates asymmetry between forward and backward inputs.
4. **Paper Goals**: Can a general-purpose transition generator be designed to handle object morphing, concept blending, motion prediction, and scene transitions in a unified manner?
5. **Starting Point**: Three complementary designs are introduced on top of an I2V diffusion model: interpolation-based initialization (handling large content discrepancies), bidirectional motion (eliminating directional asymmetry), and representation alignment (enhancing fidelity).
6. **Core Idea**: Unify four transition task categories via spherical interpolation of noise + LoRA fusion + text SLERP, while bidirectional motion fine-tuning eliminates directional bias.

## Method

### Overall Architecture
VTG is built upon the DynamiCrafter pre-trained I2V diffusion model. Given a start frame $x^1$, an end frame $x^N$, and corresponding text prompts, VTG comprises an inference stage and a training stage. During inference, DDIM inversion is applied to obtain latent noise at both endpoints, which are then SLERP-interpolated. During training, only the value/output matrices of temporal attention layers and an MLP projector are fine-tuned on 150 high-quality videos.

### Key Designs

1. **Interpolation-based Initialization**:
    - Function: Mitigates abrupt changes caused by random noise, preserves object identity, and handles large content discrepancies.
    - Mechanism: Triple interpolation — (1) **Noise SLERP**: DDIM inversion is applied to both endpoints to obtain $z_{t1}$ and $z_{tN}$; intermediate frame noise is correlated via spherical linear interpolation $z_{tn} = \frac{\sin((1-\lambda)\phi)}{\sin\phi}z_{t1} + \frac{\sin(\lambda\phi)}{\sin\phi}z_{tN}$, injected only in early denoising steps. (2) **LoRA Interpolation**: Separate LoRAs $\Delta\theta_1, \Delta\theta_N$ are trained for each endpoint (only 200 steps), then linearly interpolated as $\Delta\theta = (1-\lambda_{LoRA})\Delta\theta_1 + \lambda_{LoRA}\Delta\theta_N$ to fuse semantics. (3) **Frame-aware Text SLERP**: Text embeddings $c_1, c_N$ from both endpoints are interpolated via SLERP as $c_{\lambda} = \text{SLERP}(c_1, c_N, \lambda_{text})$ to achieve per-frame text-conditioned transitions.
    - Design Motivation: Linear interpolation in Gaussian latent space produces unlikely norms; SLERP preserves Euclidean norms and enables in-distribution sampling. LoRA captures high-level semantics absent in image diffusion models. Text SLERP addresses the inability of a single caption to describe the blended meaning of intermediate frames.

2. **Bidirectional Motion Prediction**:
    - Function: Eliminates quality asymmetry caused by the ordering of forward and backward inputs in I2V diffusion models.
    - Mechanism: The temporal self-attention map is rotated by 180° to invert attention relationships, while the temporal dimension of the noise latent is simultaneously reversed. A forward U-Net and a backward U-Net predict forward and backward motion noise, respectively. The backward prediction is re-reversed and fused with the forward prediction: $\epsilon_t = (1-\lambda_{BMP})\epsilon_{t,i} + \lambda_{BMP}\epsilon'_{t,N-i}$ (with $\lambda_{BMP}=0.5$). Only the value and output matrices of temporal attention layers are fine-tuned. Loss: $\mathcal{L}_{BMP} = \|\text{flip}(\epsilon_t) - \epsilon_{\theta_{w,o}}(z_{t'}, c, t, A'_{i,j})\|_2^2$.
    - Design Motivation: I2V models are biased toward similarity with the start frame (conditional image leakage) and are pre-trained only for forward motion. Bidirectional fusion ensures a consistent motion trajectory.

3. **Representation Alignment Regularization**:
    - Function: Enhances the fidelity of generated transition frames and reduces blurriness.
    - Mechanism: Intermediate diffusion latents are patchified per frame and aligned to DINOv2 features via a trainable MLP projector. Cosine similarity is computed per patch: $\mathcal{L}_{RAR} = -\sum_{n=1}^{N}\mathbb{E}[\frac{1}{P}\sum_{p=1}^{P}\text{sim}(y_*^{[p]}, y_\phi(h_t)^{[p]})]$. The DINOv2 encoder and MLP are discarded at inference time.
    - Design Motivation: Diffusion latents inherently lack high-frequency semantics, whereas DINOv2 features contain rich self-supervised semantic information. Distilling DINOv2 features into the diffusion process during training incurs zero overhead at inference.

### Loss & Training
Lightweight fine-tuning on only 150 high-quality videos. BMP fine-tunes temporal attention V/O matrices; RAR trains the MLP projector. AdamW optimizer, learning rate 1e-5, ~20K iterations on 4 A100 GPUs. LoRA training requires only 200 steps (~85 seconds) per input pair. DDIM sampling with 50 steps.

## Key Experimental Results

### Main Results

| Method | MorphBench FID↓ | MorphBench PPL↓ | TC-Bench TCR↑ | Smoothness↑ |
|--------|------|------|----------|------|
| DiffMorpher | 70.49 | 18.19 | 41.82 | — |
| SEINE | 82.03 | 47.72 | — | — |
| DynamiCrafter | 87.32 | 42.09 | — | — |
| TVG | 86.92 | 35.18 | — | — |
| **VTG (Ours)** | **67.39** | **22.80** | **Best** | **Best** |

### Ablation Study

| Configuration | FID↓ | PPL↓ | Note |
|------|------|------|------|
| Full VTG | Best | Best | Complete model |
| w/o Noise SLERP | ↑ | ↑ | Random abrupt changes in intermediate frames |
| w/o LoRA Interpolation | ↑ | — | Insufficient semantic fusion |
| w/o Text SLERP | ↑ | — | No per-frame text conditioning |
| w/o BMP | ↑ | ↑ | Forward/backward directional asymmetry |
| w/o RAR | ↑ | — | Loss of fine-grained detail |

### Key Findings
- VTG significantly outperforms DiffMorpher on object morphing (FID 67.39 vs. 70.49), as DiffMorpher lacks temporal modeling.
- In concept blending, VTG generates semantically plausible intermediate states (e.g., a truck with lion coloring and proportions), whereas baselines produce abrupt transitions.
- A bidirectional motion weight of $\lambda_{BMP}=0.5$ effectively eliminates directional bias.
- RAR yields the most pronounced gains in high-frequency texture scenarios (bicycle spokes, fabric patterns).

## Highlights & Insights
- A unified definition and framework for four transition task categories: object morphing, concept blending, motion prediction, and scene transition.
- The triple interpolation strategy (noise + LoRA + text) forms a logically complementary hierarchy: structural consistency at the noise level, semantic fusion at the LoRA level, and conditional guidance at the text level.
- Construction of the TransitBench benchmark: 200 start–end frame pairs, providing the first standardized evaluation for concept blending and scene transition.
- Zero inference overhead for RAR: DINOv2 is used solely for training-time regularization.

## Limitations & Future Work
- LoRA training requires ~85 seconds per input pair, which becomes costly for batch generation.
- Only 150 training videos limit motion diversity.
- The framework is based on DynamiCrafter (UNet architecture) and may be transferable to more recent DiT architectures.
- TransitBench is relatively small (200 pairs) and could be substantially expanded.

## Related Work & Insights
- **vs. DiffMorpher**: Image diffusion-based morphing lacks temporal modeling, producing discontinuous frame sequences.
- **vs. SEINE**: Uses randomly masked conditioning layers for scene transitions but performs poorly on concept blending.
- **vs. Generative Inbetweening**: Fuses forward and backward noise but overlooks identity preservation and large content discrepancies.

## Rating
- Novelty: ⭐⭐⭐⭐ Unified framework for four transition tasks + triple interpolation strategy
- Experimental Thoroughness: ⭐⭐⭐⭐ Benchmarks for all four task categories + new TransitBench
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and logically coherent method components
- Value: ⭐⭐⭐⭐ Unified paradigm for transition generation with practical value for video editing and filmmaking

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)
- [\[ICCV 2025\] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)

<!-- RELATED:END -->
