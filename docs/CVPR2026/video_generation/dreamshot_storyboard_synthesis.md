---
title: >-
  [Paper Note] DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior
description: >-
  [CVPR 2026][Video Generation][Storyboard Generation] This paper proposes DreamShot, which leverages the spatiotemporal prior of video diffusion models to generate multi-shot storyboards with consistent characters and coh…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Storyboard Generation"
  - "Video Diffusion Model"
  - "Character Consistency"
  - "Multi-Character Reference"
  - "Attention Constraint"
date: 2026-05-08
content_hash: 65bd2e4a63373424
---

# DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior

**Conference**: CVPR 2026
**arXiv**: [2604.17195](https://arxiv.org/abs/2604.17195)
**Code**: [https://ll3rd.github.io/DreamShot/](https://ll3rd.github.io/DreamShot/)
**Area**: Video Generation
**Keywords**: Storyboard Generation, Video Diffusion Model, Character Consistency, Multi-Character Reference, Attention Constraint

## TL;DR

This paper proposes DreamShot, which leverages the spatiotemporal prior of video diffusion models to generate multi-shot storyboards with consistent characters and coherent scenes. A Role-Attention Consistency Loss (RACL) is introduced to address multi-character confusion, and three unified generation modes are supported: text-to-shot, reference-to-shot, and shot-to-shot.

## Background & Motivation

**Background**: Storyboard generation aims to produce coherent sequences of key shots for cinematic narratives. Existing approaches fall into two categories: image diffusion-based methods (e.g., StoryDiffusion, AnyStory, StoryMaker) that maintain character consistency via IP-Adapter or ControlNet; and video model-based methods (e.g., StoryAnchors) that exploit temporal consistency but support only text or previous-frame conditioning.

**Limitations of Prior Work**: Image-based models are inherently biased toward diversity rather than temporal stability, resulting in poor cross-shot character consistency and severe character confusion in multi-character scenarios (erroneous blending of facial and clothing features across characters). Video models offer better consistency but incur high computational cost from dense frame generation and lack fine-grained personalization control.

**Key Challenge**: Image models offer flexibility but lack consistency; video models offer consistency but lack efficiency — a fundamental trade-off.

**Goal**: Combine the spatiotemporal consistency prior of video models with the efficiency and controllability of image-level generation to achieve high-quality personalized storyboard synthesis.

**Key Insight**: When a video VAE (e.g., Wan-VAE) compresses consecutive frames into a latent space, it preserves the causal temporal structure. By repeating each storyboard shot for $T$ frames before encoding, independent static shots can be transformed into a coherent temporal latent sequence.

**Core Idea**: Within a video diffusion (DiT) framework, character reference images are treated as temporally preceding anchor tokens and storyboard shots as subsequent temporal segments. The 3D RoPE positional encoding naturally propagates character identity information along the temporal axis, while RACL constrains cross-character attention to prevent confusion.

## Method

### Overall Architecture

DreamShot is built upon a video diffusion model (Video-VAE + DiT). The inputs consist of $K$ character reference images and text scripts for $S$ shots. Each reference image is individually encoded into a latent vector, and each storyboard shot is repeated $T$ frames before VAE encoding. Reference tokens and shot tokens are concatenated and fed into the DiT, where self-attention is computed jointly over all tokens and cross-attention aligns each shot with its corresponding text. Three modes are supported: Reference-to-Shot (generation from reference images), Text-to-Shot (pure text-driven generation), and Shot-to-Shot (continuation from existing shots).

### Key Designs

1. **Shot Temporal Alignment via Video VAE**:

    - Function: Transforms independent storyboard shots into a coherent temporal sequence processable by the video VAE.
    - Mechanism: Each shot (except the first) is repeated $T$ frames and encoded to obtain $z_{shot} \in \mathbb{R}^{s \times d \times h \times w}$. Reference images are encoded as $z_{ref}$, then concatenated as $z_t = [z_{ref}, z_{shot}]$, with 3D RoPE encoding temporal and spatial positions.
    - Design Motivation: Placing reference images at the front of the sequence and arranging shots in narrative order allows the DiT to naturally propagate character identity forward along the temporal axis. This simple yet critical design endows the DiT with cross-shot consistency propagation that image-based models inherently lack.

2. **Role-Attention Consistency Loss (RACL)**:

    - Function: Prevents cross-character feature confusion in multi-character scenarios.
    - Mechanism: Character masks for reference images are obtained via saliency detection, and character masks for storyboard shots are obtained via grounding segmentation; one-to-one correspondences are established using ArcFace and a VLM. In the DiT's self-attention, the attention map $A_{r_k-s_k}$ between reference character $r_k$ and storyboard character $s_k$ is computed, and the corresponding masks serve as supervision to constrain attention to focus on matching character regions.
    - Design Motivation: The root cause of character confusion in existing methods is the erroneous blending of different characters' features during attention computation. RACL explicitly constrains each character to attend only to its corresponding region, eliminating confusion at the training level.

3. **Mixed-Mode Training and Generation**:

    - Function: Unified support for diverse storyboard generation scenarios.
    - Mechanism: In Reference-to-Shot mode, noise is added only to shot tokens (reference images remain clean); in Text-to-Shot mode, all shot tokens are noised; in Shot-to-Shot mode, preceding shots serve as clean conditions to guide subsequent generation. The model is trained with a Flow Matching objective.
    - Design Motivation: Real-world storyboard production encompasses both creation from scratch and continuation from existing content; a unified framework avoids training separate models for different scenarios.

### Loss & Training

The primary loss is the Flow Matching objective $\mathcal{L}_{diff}$, with RACL serving as an auxiliary loss to constrain character attention consistency. The dataset is constructed from temporally coherent shot sequences extracted from real and synthetic videos, each annotated with representative reference frames and shot-level descriptions.

## Key Experimental Results

### Main Results

The paper emphasizes qualitative and quantitative comparisons against image-based methods, demonstrating advantages in character consistency, scene coherence, and generation efficiency. DreamShot avoids character confusion in multi-character scenarios, whereas image-based methods such as StoryDiffusion and AnyStory frequently exhibit character feature misalignment.

| Comparison Dimension | DreamShot | Image-Based Methods |
|---|---|---|
| Character Consistency | Strong (stable identity across shots) | Weak (frequent character confusion) |
| Scene Coherence | Strong (guaranteed by video prior) | Weak (inter-shot inconsistency) |
| Multi-Character Support | Good (RACL constraint) | Poor (feature entanglement) |
| Generation Efficiency | High (keyframes, not dense frames) | Moderate |

### Ablation Study

| Configuration | Character Consistency Metric | Note |
|---|---|---|
| Full model | Best | RACL + video prior |
| w/o RACL | Degraded | Confusion in multi-character scenes |
| Image model backbone | Significantly degraded | Lacks temporal consistency |

### Key Findings

- The contribution of video diffusion prior to cross-shot consistency is decisive and cannot be substituted by simply "upgrading" image-based models.
- RACL is particularly effective in multi-character ($\geq 2$) scenarios, with limited gain in single-character settings.
- The quality of Shot-to-Shot continuation is highly dependent on the quality of preceding shots.

## Highlights & Insights

- The idea of "using a video model to generate keyframes rather than dense frames" is elegant — it retains the consistency advantages of the video prior while avoiding the computational waste of numerous redundant frames.
- RACL directly targets the root cause of multi-character confusion (feature entanglement at the attention level) and constrains the attention distribution via explicit mask supervision — a clear and effective design.
- Placing reference images at the front of the token sequence and leveraging 3D RoPE temporal encoding to propagate identity information represents a creative exploitation of the semantic structure of video model positional encoding.

## Limitations & Future Work

- Quality depends on the pretrained video model (e.g., Wan2.1) and is bounded by the capabilities of the base model.
- RACL requires character mask detection and one-to-one matching, which may fail under heavy occlusion or when characters have similar appearances.
- Current evaluation relies primarily on qualitative comparisons, lacking a standardized storyboard generation benchmark.
- Future work could extend to interactive editing (modifying specific shots while preserving the rest).

## Related Work & Insights

- **vs StoryDiffusion/StoryAdapter**: These cross-frame attention consistency methods based on image models are fundamentally constrained by the frame-independence of image models; this paper resolves the consistency issue at a fundamental level through video priors.
- **vs StoryAnchors**: Also adopts a video paradigm but supports only text/previous-frame conditioning and lacks multi-character reference control.

## Rating

- Novelty: ⭐⭐⭐⭐ Video prior-driven storyboard generation is a new direction; RACL design is elegant.
- Experimental Thoroughness: ⭐⭐⭐ Primarily qualitative; lacks standardized quantitative comparison.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear; framework description is complete.
- Value: ⭐⭐⭐⭐ Opens a new paradigm for storyboard generation with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[CVPR 2026\] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](moviedrive_multimodal_multiview_video_diffusion.md)
- [\[ICLR 2026\] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](../../ICLR2026/video_generation/javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)
- [\[CVPR 2026\] NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing](nova_sparse_control_dense_synthesis_for_pair-free_video_editing.md)
- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](../../ICCV2025/video_generation/fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)

</div>

<!-- RELATED:END -->
