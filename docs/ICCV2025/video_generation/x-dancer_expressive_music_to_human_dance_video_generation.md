---
title: >-
  [Paper Note] X-Dancer: Expressive Music to Human Dance Video Generation
description: >-
  [ICCV 2025][Video Generation][music-driven dance generation] X-Dancer proposes a unified Transformer–diffusion framework that takes a single static image and a music sequence as input, autoregressively generates 2D whole-body dance pose token sequences synchronized with musical beats via a Transformer, and then synthesizes high-fidelity dance videos from these tokens using a diffusion model, surpassing existing methods in diversity, expressiveness, and video quality.
tags:
  - ICCV 2025
  - Video Generation
  - music-driven dance generation
  - human image animation
  - autoregressive Transformer
  - diffusion models
  - 2D pose modeling
date: 2026-05-08
content_hash: 11564fd9ba8b6242
---

# X-Dancer: Expressive Music to Human Dance Video Generation

**Conference**: ICCV 2025
**arXiv**: [2502.17414](https://arxiv.org/abs/2502.17414)
**Code**: [https://zeyuan-chen.com/X-Dancer/](https://zeyuan-chen.com/X-Dancer/) (project page)
**Area**: Diffusion Models / Video Generation
**Keywords**: music-driven dance generation, human image animation, autoregressive Transformer, diffusion models, 2D pose modeling

## TL;DR

X-Dancer proposes a unified Transformer–diffusion framework that takes a single static image and a music sequence as input, autoregressively generates 2D whole-body dance pose token sequences synchronized with musical beats via a Transformer, and then synthesizes high-fidelity dance videos from these tokens using a diffusion model, surpassing existing methods in diversity, expressiveness, and video quality.

## Background & Motivation

**Background**: Music-driven dance generation is an active research area. Existing approaches fall into two main categories — 3D skeleton-based motion generation (e.g., Bailando, EDGE) and end-to-end diffusion-based video synthesis (e.g., Hallo).

**Limitations of Prior Work**:
   - 3D methods are constrained by dataset availability (primarily relying on the AIST++ multi-view dataset), resulting in limited dance diversity; furthermore, they only model body poses without head or hand movements.
   - Recovering 3D poses from 2D monocular video is error-prone.
   - End-to-end diffusion methods (e.g., Hallo) struggle to capture long-range motion and audio context, and have been primarily validated on talking-head animation, with uncertain generalization to whole-body motion.

**Key Challenge**: High-quality, diverse dance data is abundant but predominantly consists of 2D monocular videos, whereas existing methods either rely on scarce 3D data or fail to effectively model long-range music–motion correspondences.

**Goal**: To leverage widely available 2D monocular dance videos for generating expressive, music-synchronized long-range dance animations.

**Key Insight**: Rather than 3D skeleton modeling, the paper models dance motion in 2D space, tokenizes each body part separately using VQ-VAE, and autoregressively generates motion sequences via a GPT-style Transformer.

**Core Idea**: Tokenize 2D whole-body poses per body part with keypoint confidence awareness, generate music-synchronized motion sequences via a cross-conditional autoregressive Transformer, and guide a diffusion model with an implicit motion decoder to synthesize the final video.

## Method

X-Dancer operates in two stages: (1) Transformer-based music-to-dance motion generation; (2) diffusion-based video synthesis.

### Overall Architecture

Given a reference portrait $I_R$ and a music sequence $M_t$, the model outputs a dance video sequence $I_{M_t}$ synchronized with musical beats and consistent in appearance with the reference image. A VQ-VAE first encodes 2D whole-body poses into discrete tokens; a GPT model then autoregressively predicts the token sequence; finally, a diffusion model generates video frames conditioned on the token sequence and the reference image.

### Key Designs

1. **Compositional Confidence-Aware Pose Tokenization**:

    - Function: Encodes each frame's 2D whole-body pose (60 keypoints + confidence scores) by partitioning it into 5 body parts (upper body, lower body, left hand, right hand, head), each encoded and quantized independently.
    - Mechanism: Each part uses a dedicated 1D convolutional encoder $E^j$ and a codebook $Z_q^j$ with 512 entries, representing each part with 6 tokens at a codebook embedding dimension of 6D. After quantization, the latent codes of all parts are concatenated and fed into a shared decoder to reconstruct the full-body pose.
    - Training loss: $\mathcal{L}_{VQ} = \sum_{j=1}^{B} \|\hat{p}^j - p^j\|_2 + \beta \sum_{j=1}^{B} \|sg[z_e^j(p)] - z_q^j(p)\|_2$, with codebook updated via EMA.
    - Design Motivation: A single VQ-VAE struggles to capture high-frequency details such as finger movements and head tilts. Part-wise modeling allows motions of different frequencies to be represented independently, broadening expressive coverage. Confidence information enables the model to handle motion blur and occlusion.

2. **Cross-Conditional Autoregressive Motion Modeling**:

    - Function: A GPT model predicts the next-frame token distribution for each body part while maintaining synchronization with music features.
    - Mechanism: Music conditioning is injected in two ways — (1) Jukebox and Librosa music embeddings are merged into a global start token providing style and genre information; (2) per-frame music embeddings are concatenated with motion tokens to ensure frame-level synchronization. The model learns the joint distribution $\phi(C_{1:T}^{1:B}|F_g) = \prod_t \prod_j \prod_k \phi(c_{k,t}^j | ...)$.
    - Cross-part cross-conditioning: A hierarchical dependency is established in the order "upper/lower body → head → hands," where each token is conditioned on all parts at all previous timesteps and all already-generated parts at the current timestep.
    - Long-range consistency: Eight frames are uniformly sampled from the previous segment as cross-segment motion context.
    - Design Motivation: Independently modeling body parts may produce asynchronous motions (e.g., upper and lower body moving in inconsistent directions). Cross-conditioning leverages inter-part mutual information to maintain whole-body coordination.

3. **Implicit Motion Token Decoder**:

    - Function: Implicitly converts 1D pose tokens into 2D spatial guidance injected into the diffusion UNet.
    - Mechanism: Starting from a learnable 2D feature map, AdaIN is applied to inject the token sequence (including confidence) within a 16-frame window, progressively upsampled to match UNet feature resolutions at each scale. This module is jointly trained with the temporal modules.
    - Comparison with explicit skeleton rendering: The conventional approach decodes tokens into keypoint coordinates and renders skeleton images, introducing a non-differentiable step that impedes end-to-end training and discards confidence information. The implicit approach preserves temporal context and handles jittery poses produced by the Transformer more robustly.
    - Design Motivation: End-to-end differentiable training enables joint optimization of motion decoding and appearance reference, and the resulting pose guidance adapts to varying body shapes.

### Loss & Training

Three-stage training (8× A100 GPUs):
- Stage 1: VQ-VAE, 40k steps, batch=2048, lr=$2\times10^{-4}$
- Stage 2: Autoregressive Transformer (GPT-2 initialization), 300k steps, batch=24, lr=$1\times10^{-4}$, 64-frame sequences, 2224-token context window
- Stage 3: Diffusion model (SD1.5 + ReferenceNet), trained first on 2 frames then extended to 16-frame temporal modeling, 90k steps, lr=$1\times10^{-5}$

## Key Experimental Results

### Main Results

**Motion Generation Quality Comparison (Tab. 1)**:

| Method | FVD↓ (AIST/In-House) | DIV↑ (AIST/In-House) | BAS↑ (AIST/In-House) |
|------|------|------|------|
| Ground Truth | 509.58/129.75 | 34.10/29.67 | 0.24/0.22 |
| Hallo | 548.81/249.12 | 28.66/28.98 | 0.16/0.20 |
| Bailando | 621.22/534.02 | 22.34/24.05 | 0.19/0.19 |
| EDGE | 639.46/303.36 | 24.87/27.29 | **0.26**/0.24 |
| **X-Dancer** | **531.52/238.22** | 25.61/28.08 | 0.23/0.21 |

**Video Synthesis Quality Comparison (Tab. 2)**:

| Method | FVD↓ | FID-VID↓ | ID-SIM↑ |
|------|------|---------|---------|
| Hallo | 609.08 | 76.99 | 0.4870 |
| EDGE+PG | 613.81 | 93.73 | 0.3034 |
| Our motion+PG | 735.05 | 72.71 | 0.4894 |
| **X-Dancer** | **507.06** | **61.94** | **0.5317** |

### Ablation Study

**Transformer Design Ablation (Tab. 3)**:

| Configuration | FVD↓ | DIV↑ | BAS↑ |
|------|------|------|------|
| w/o global music context | 265.73 | 27.04 | 0.2142 |
| w/o global motion context | 247.54 | 26.42 | 0.2154 |
| Subset + GPT-medium | 402.63 | 24.40 | 0.2112 |
| Subset + GPT-small | 332.93 | 24.58 | 0.2046 |
| **X-Dancer (Full)** | **238.22** | **28.08** | **0.2182** |

**VQ-VAE and Video Diffusion Ablation (Tab. 4)**:

| Configuration | Pose L1 (Full/Head/Hands) | Video PSNR↑/LPIPS↓/FVD↓ |
|------|---|---|
| Single-Part VQVAE | 0.83/0.64/0.52 | - |
| **Multi-Part VQVAE** | **0.50/0.40/0.42** | - |
| GT Pose + PG | - | 19.465/0.197/294.52 |
| Multi-Part + PG | - | 18.836/0.207/384.16 |
| **Multi-Part + MD** | - | **19.148/0.207/295.87** |

### Key Findings

- **Multi-Part VQ-VAE substantially reduces pose reconstruction error**: Full-body L1 drops from 0.83 to 0.50, with notable improvements on head and hands. Qualitative results show that the single-part variant fails to control subtle motions such as hip sways, leg raises, and head tilts.
- **Both global music and motion context contribute**: Removing either leads to higher FVD and lower BAS.
- **Scalability with model and data size**: Scaling from GPT-small to GPT-medium and from 10k to 100k training samples consistently improves all metrics.
- **Implicit Motion Decoder outperforms explicit Pose Guider**: Achieves better identity preservation (ID-SIM: 0.5317 vs. 0.4894) and temporal smoothness.

## Highlights & Insights

- **2D pose modeling as an alternative to 3D skeleton modeling** — a pragmatic and clever choice. 2D pose detection is more reliable than 3D pose estimation, and 2D dance video data is far more abundant than 3D data, substantially improving model diversity and scalability.
- **Confidence-aware tokenization** — encoding keypoint confidence into tokens allows the model to learn the temporal distribution of motion blur and occlusion, a feature rarely seen in prior motion generation methods.
- **AdaIN-based implicit motion token injection** — bypasses the non-differentiable skeleton rendering step, enabling end-to-end training; the approach is transferable to other tasks requiring discrete tokens to be converted into spatial guidance.

## Limitations & Future Work

- Training is limited to everyday dance videos without professional dancer data, leaving room for improvement in motion precision and music alignment.
- The model may produce rendering artifacts for out-of-distribution portraits (e.g., cartoon-style images).
- The three stages are trained separately due to GPU memory constraints, precluding true end-to-end joint training.
- Multi-person scenarios and complex interactions are not supported.

## Related Work & Insights

- **vs. Bailando/EDGE**: These methods generate motion in 3D space and are constrained by the limited diversity of the AIST++ dataset; X-Dancer operates in 2D space with far superior data scalability.
- **vs. Hallo**: Hallo generates video end-to-end but struggles with large-amplitude joint movements and exhibits poor temporal consistency (high DIV scores reflect jitter noise rather than genuine diversity); X-Dancer decouples motion generation from video synthesis.
- **vs. DabFusion**: DabFusion explores music-conditioned image-to-video generation, but X-Dancer achieves finer control through an explicit motion token intermediate representation.

## Rating

- Novelty: ⭐⭐⭐⭐ — 2D whole-body part-wise tokenization with confidence awareness is a novel contribution, though the overall Transformer + diffusion framework is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Includes detailed quantitative comparisons, ablations, and scalability analysis, but lacks user study results.
- Writing Quality: ⭐⭐⭐⭐ — Method description is clear with well-structured logical progression.
- Value: ⭐⭐⭐⭐ — Represents the first music-driven human image animation framework with practical application value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[ICCV 2025\] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)
- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](../../CVPR2026/video_generation/posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[ICLR 2026\] MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling](../../ICLR2026/video_generation/mosa_motion-coherent_human_video_generation_via_structure-appearance_decoupling.md)

<!-- RELATED:END -->
