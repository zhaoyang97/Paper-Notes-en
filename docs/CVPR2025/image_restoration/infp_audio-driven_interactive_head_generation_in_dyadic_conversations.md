---
title: >-
  [Paper Note] INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations
description: >-
  [CVPR 2025][Image Restoration][Interactive Head Generation] INFP proposes a unified, audio-driven interactive head generation framework. By utilizing dual-track audio (agent + conversational partner), the framework naturally drives the agent to switch between speaking and listening states without manual role assignment or explicit role switching, while introducing the large-scale DyConv dataset to support research in this field.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Interactive Head Generation"
  - "Dyadic Conversation"
  - "Audio-Driven"
  - "Motion Latent Space"
  - "Diffusion Transformer"
date: 2026-05-08
content_hash: 56a4ab5bd7dae074
---

# INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations

**Conference**: CVPR 2025  
**arXiv**: [2412.04037](https://arxiv.org/abs/2412.04037)  
**Code**: [https://grisoon.github.io/INFP/](https://grisoon.github.io/INFP/) (Project Page)  
**Area**: Image Restoration  
**Keywords**: Interactive Head Generation, Dyadic Conversation, Audio-Driven, Motion Latent Space, Diffusion Transformer

## TL;DR

INFP proposes a unified, audio-driven interactive head generation framework. By utilizing dual-track audio (agent + conversational partner), the framework naturally drives the agent to switch between speaking and listening states without manual role assignment or explicit role switching, while introducing the large-scale DyConv dataset to support research in this field.

## Background & Motivation

**Background**: Audio-driven head generation is a core technology for building conversational agents. Currently, it is divided into two independent directions: talking-head generation (focusing on lip synchronization) and listening-head generation (focusing on non-verbal feedback such as nodding and facial expression changes).

**Limitations of Prior Work**: Existing methods only focus on one-sided communication, either only speaking or only listening. A few works exploring dyadic conversations (such as ViCo-X and DIM) split the model into a Speaker Generator and a Listener Generator, which requires manual role assignment and explicit switching. This design leads to unnatural state transitions and fails to cover real-world scenarios such as both parties speaking simultaneously.

**Key Challenge**: Dividing "speaking" and "listening" into two separate models is an oversimplification—role switching in real dyadic conversations is frequent and gradual, with no explicit transition points. Manually segmenting the audio into speaking and listening clips and processing them separately inevitably leads to disconnected transitions.

**Goal**: How to design a unified model that automatically drives the agent to naturally transition between speaking, listening, and interactive states based on the content of dual-track conversational audio.

**Key Insight**: The authors observe that in dyadic conversations, the agent's state is fully determined by the dual-track audio—when the agent's own audio is active, it should exhibit a speaking state, and when the other party's audio is active, it should exhibit a listening state. Therefore, the audio signal intensity can be used to implicitly control state transitions, rather than using explicit decision-making.

**Core Idea**: Use dual-track audio to simultaneously retrieve features from verbal and non-verbal motion memory banks, automatically enabling continuous transitions between speaking and listening states based on signal strength.

## Method

### Overall Architecture

INFP is divided into two phases: (1) Motion-Based Head Imitation: learning communicative facial behaviors from real conversational videos and encoding them into a low-dimensional motion latent space, using the motion latent codes to animate static portrait images; (2) Audio-Guided Motion Generation: learning the mapping from dual-track conversational audio to motion latent codes, and achieving audio-driven interactive head generation through diffusion denoising.

### Key Designs

1. **Motion Encoding and Head Imitation**:

    - **Function**: Establish a decoupled motion latent space to isolate facial motions from appearance.
    - **Mechanism**: The motion encoder $E_m$ encodes facial images into low-dimensional 1D motion latent codes. To achieve decoupling, a hybrid facial representation is designed: (a) masking most of the pixels in the facial image, retaining only the eye and lip regions (the most expressive parts) to prevent irrelevant information like hair and background from interfering; (b) using a facial estimation model to obtain facial vertices, and projecting outline points onto the masked image to provide orientation information. Then, a motion flow estimation model $F$ predicts optical flow from the source and driving motion codes, and the final video is synthesized via a face decoder after warping the feature volume.
    - **Design Motivation**: Implicit representations are used instead of explicit 3DMM coefficients because 3DMM has limited expressiveness and its expression coefficients are entangled with face shapes.

2. **Interactive Motion Guider**:

    - **Function**: Adaptively extract interactive motion features from dual-track audio.
    - **Mechanism**: Design two learnable memory banks—$M_v$ (verbal motion bank, 64 learnable 512-dimensional embeddings) and $M_{nv}$ (non-verbal motion bank). The agent's audio $A_{self}$ serves as a Query to retrieve verbal motion features from $M_v$ via cross-attention; the other party's audio $A_{other}$ serves as a Query to retrieve non-verbal motion features from $M_{nv}$. When the agent is speaking, the $A_{self}$ signal is strong, and verbal motion features dominate; when the other party is speaking, the $A_{other}$ signal is strong, and non-verbal motion features dominate. The two are fused into interactive motion features $f_m$ via element-wise addition + MLP. A motion style vector $s_m$ is also introduced to edit the memory bank embeddings through the style modulation layer of StyleGAN2, injecting global emotion/attitude information.
    - **Design Motivation**: The signal strength of dual-track audio naturally corresponds to the role state, completely avoiding any explicit role-determination logic. The memory banks store typical motion patterns, and cross-attention flexibly combines them based on the audio content.

3. **Conditional Diffusion Transformer**:

    - **Function**: Map interactive motion features to the pre-trained motion latent space.
    - **Mechanism**: A lightweight Transformer with only 4 blocks, where each block consists of self-attention $\rightarrow$ motion attention $\rightarrow$ temporal attention. In motion attention, the latent features serve as Queries, and the interactive motion features $f_m$ serve as Key-Values to perform cross-attention. Temporal attention uses the motion latent codes of the last 10 frames of the previous window as a condition to ensure smooth transitions between adjacent windows. A DDIM sampler is used with 20 denoising steps.
    - **Design Motivation**: The lightweight 4-block design supports real-time interaction. The temporal layer references the approach of AnimateDiff to ensure temporal coherence.

### Loss & Training

Phase 1 utilizes standard image reconstruction and perceptual losses. Phase 2 utilizes the diffusion denoising loss (predicting noise) and an AdamW optimizer (lr=1e-4, wd=1e-2, bs=32). Training strategy: the style vector is zeroed out with a probability of 0.3, and the motion features and previous latent codes are dropped with a probability of 0.5 to achieve classifier-free guidance. During the warm-up phase, the model is first trained on one-sided conversational clips to initialize the memory banks, and then the entire training is completed using multi-turn conversational data.

## Key Experimental Results

### Main Results

| Method | SSIM↑ | PSNR↑ | FID↓ | SyncScore↑ | SID↑ | Var↑ |
|------|-------|-------|------|------------|------|------|
| DIM | 0.651 | 20.42 | 34.36 | 4.778 | 0.766 | 0.825 |
| Ours | **0.834** | **31.56** | **15.73** | **7.188** | **2.613** | **2.386** |
| GT | 1.000 | - | 0.000 | 7.261 | 2.891 | 2.435 |

Ours outperforms DIM by a large margin across all metrics. SyncScore (7.188 vs 4.778) is close to GT (7.261), and the SID and Var metrics being close to GT indicate excellent motion diversity.

### Ablation Study

| Configuration | SSIM | FID | SyncScore | SID |
|------|------|-----|-----------|-----|
| Ours (Full) | 0.834 | 15.73 | 7.188 | 2.613 |
| w/o Motion Memory | 0.830 | 18.33 | 6.103 | 2.153 |
| w/o Style Modulation | 0.831 | 16.03 | 7.062 | 2.551 |
| w/ Intact Image (no masking) | 0.802 | 16.99 | 6.812 | 2.470 |
| w/ Landmarks Map (replacing hybrid rep.) | 0.821 | 16.33 | 6.833 | 2.601 |

### Key Findings

- Motion Memory contributes the most (removing it causes SyncScore to drop by 1.085 and FID to increase by 2.6), demonstrating the critical role of the memory bank in extracting interaction information.
- The hybrid facial representation (masking + outline points) outperforms the intact image (SSIM 0.834 vs 0.802) and pure landmarks (0.821), illustrating that removing irrelevant information is crucial for decoupling motion encoding.
- On the ViCo listening-head generation benchmark, Ours surpasses the SOTA in both FD (18.63 vs DIM 23.88) and SID (4.78 vs 3.71).
- In the user study (20-person MOS score), Ours leads DIM significantly in terms of naturalness (4.38 vs 2.71) and motion diversity (4.49 vs 2.14).

## Highlights & Insights

- **Implicitly switching roles using the signal strength of dual-track audio is a highly natural design**: When the agent speaks, the strong $A_{self}$ signal naturally drives speaking motion, requiring no explicit role-determination logic. This is much more concise and robust than the traditional pipeline of first deciding who is speaking and then dispatching the task to different models.
- **The concept of utilizing memory banks as a motion pattern library is transferable**: The design of storing verbal/non-verbal motion prototypes in learnable embeddings and retrieving/combining them through audio cross-attention can be adapted to other tasks, such as gesture generation and full-body motion synthesis.
- **The contribution of the DyConv dataset is significant**: A dyadic conversation dataset with 200+ hours, high facial resolution ($>400 \times 400$), isolated audio, and speaker detection fills a major gap in resources for this field.

## Limitations & Future Work

- Only the head region is generated, without involving richer non-verbal behaviors such as hand gestures and upper body movements.
- The DyConv dataset primarily originates from online videos with face-to-face conversational scenarios; its generalization capability to other scenarios like phone calls or group chats remains unverified.
- Although the 4-block diffusion transformer is lightweight, specific data regarding its real-time inference performance (FPS) is not explicitly provided in the paper.
- The accuracy of audio segment separation directly affects the quality of the agent's and the partner's audio, making robustness in noisy environments worthy of further study.

## Related Work & Insights

- **vs DIM**: DIM splits the model into Speaker/Listener Generators, requiring manual role assignment and post-pretraining fine-tuning. Ours is a unified model that significantly outperforms it across all metrics.
- **vs ViCo-X**: ViCo-X designs an explicit Role Switcher to bridge two generators, leading to unnatural state transitions. Ours utilizes implicit switching, which is much smoother.
- **vs Wav2Lip/VASA-1**: These talking-head methods only handle the speaking state and cannot address listening or interactive scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Proposes a new paradigm for interactive head generation, with novel designs for implicit role switching and motion memory banks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates across three scenarios (interactive, listening, and speaking), but lacks comparison with other more recent methods (some of which are not open-source).
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and detailed method description, though some implementation details require reviewing the supplementary materials.
- Value: ⭐⭐⭐⭐ Offers significant reference value for building more natural conversational AI agents, and the DyConv dataset is also a valuable contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](../../CVPR2026/image_restoration/diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [\[CVPR 2025\] One-Step Event-Driven High-Speed Autofocus](one-step_event-driven_high-speed_autofocus.md)
- [\[CVPR 2025\] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks](vision-language_gradient_descent-driven_all-in-one_deep_unfolding_networks.md)
- [\[NeurIPS 2025\] Audio Super-Resolution with Latent Bridge Models](../../NeurIPS2025/image_restoration/audio_super-resolution_with_latent_bridge_models.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](../../ICCV2025/image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)

</div>

<!-- RELATED:END -->
