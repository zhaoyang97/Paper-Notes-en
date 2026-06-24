---
title: >-
  [Paper Note] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation
description: >-
  [CVPR 2025][Video Generation][Talking Head Animation] This work proposes Teller, the first real-time streaming audio-driven portrait animation framework based on an autoregressive Transformer. By leveraging RVQ to discretize facial motion into tokens and combining it with an efficient temporal module to refine body details, Teller achieves 25 FPS real-time generation speed (requiring only 0.92s to generate a 1s video, compared to 20.93s with Hallo) while delivering animation…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Talking Head Animation"
  - "Autoregressive Generation"
  - "Real-Time Streaming"
  - "Motion Discretization"
  - "Temporal Refinement"
date: 2026-05-08
content_hash: f85039c7d3a95e7f
---

# Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation

**Conference**: CVPR 2025  
**arXiv**: [2503.18429](https://arxiv.org/abs/2503.18429)  
**Code**: [https://teller-avatar.github.io/](https://teller-avatar.github.io/)  
**Area**: Video Generation  
**Keywords**: Talking Head Animation, Autoregressive Generation, Real-Time Streaming, Motion Discretization, Temporal Refinement

## TL;DR
This work proposes Teller, the first real-time streaming audio-driven portrait animation framework based on an autoregressive Transformer. By leveraging RVQ to discretize facial motion into tokens and combining it with an efficient temporal module to refine body details, Teller achieves 25 FPS real-time generation speed (requiring only 0.92s to generate a 1s video, compared to 20.93s with Hallo) while delivering animation quality comparable to diffusion models.

## Background & Motivation
1. **Background**: Audio-driven portrait animation (talking head) has made significant progress in recent years. Diffusion model-based methods (e.g., Hallo, EMO, LOOPY) produce high-quality animations, but their extremely slow inference speed (~20s per second of video) fails to meet real-time requirements.
2. **Limitations of Prior Work**: (a) Diffusion models require multi-step iterative inference, needing multiple forward passes to generate a single frame; (b) GAN-based methods (e.g., SadTalker, LivePortrait), while fast, lack expressive motion, and natural movements of body accessories (earrings, necklaces) and neck muscles are often neglected.
3. **Key Challenge**: High-quality animation requires capturing rich facial and body motion details, but the computational budget is limited (requiring 25+ FPS for real-time applications). How can we achieve the high quality of diffusion models and the fast generation speed of autoregressive models simultaneously?
4. **Goal**: To design the first real-time, streaming, and high-quality audio-driven portrait animation framework.
5. **Key Insight**: Discretize facial motion latents into token sequences, and utilize the highly efficient next-token prediction capability of autoregressive Transformers to map audio to motion in real time.
6. **Core Idea**: A two-stage framework: Facial Motion Latent Generation (FMLG, consisting of RVQ + AR Transformer) to generate facial motion tokens, followed by an Efficient Temporal Module (ETM) to refine body details.

## Method

### Overall Architecture
Teller consists of two stages: **Stage 1 (FMLG)**: LivePortrait extracts implicit keypoint motion latents $m \in \mathbb{R}^{25 \times 3}$ (comprising 21 keypoints, head pose, and expression deformation). RVQ encodes these continuous latents into discrete tokens. An AR Transformer then takes Whisper-encoded audio embeddings as input and generates motion token sequences via next-token prediction. **Stage 2 (ETM)**: A 3D U-Net combined with temporal self-attention performs single-step refinement, enhancing the physical consistency of neck muscles and accessories like earrings.

### Key Designs

1. **Facial Motion Latent Generation (FMLG)**

    - **Function**: Maps continuous facial motion to discrete tokens to achieve highly efficient real-time generation from audio to motion.
    - **Mechanism**: Every 4 frames of motion latents ($4 \times 25 \times 3$) are compressed into 32 discrete tokens. The training objective of RVQ consists of reconstruction loss $\mathcal{L}_{recon}$ and commitment loss $\mathcal{L}_{commit}$. The AR Transformer is built upon the Qwen1.5-4B architecture, processing inputs in 200ms audio chunks (corresponding to Whisper’s $10 \times 512$ embedding and 32 motion tokens). A novel dual-token prediction head is proposed to simultaneously predict a pair of tokens at each step, doubling the inference speed. The learning of the two heads is balanced using a regularization term $\|\mathcal{L}_{head0} - \mathcal{L}_{head1}\|_2^2$.
    - **Design Motivation**: Discrete tokens make autoregressive next-token prediction feasible, avoiding the multi-step iterative process of diffusion models.

2. **Efficient Temporal Module (ETM)**

    - **Function**: Refines natural movements of body accessories and muscles in a single step.
    - **Mechanism**: A VAE encoder extracts video frame features $x \in \mathbb{R}^{b \times t \times h \times w \times c}$, which are reshaped to $(b \times h \times w) \times t \times c$ to perform self-attention along the temporal dimension. Temporal dependencies are then fused with spatial features via residual connections. Facial keypoints detected by MediaPipe define the bounding boxes for regions like the neck and ears. A region-specific masked reconstruction loss $\mathcal{L}_{ETM}$ is applied to focus on the physical consistency of accessory movements. Only a **single-step** forward pass is required (unlike diffusion models), maintaining real-time performance.
    - **Design Motivation**: LivePortrait is based on implicit keypoint driving, which inherently lacks modeling of movements in non-facial regions (e.g., earrings and necklaces).

3. **Streaming Inference Design**

    - **Function**: Achieves end-to-end real-time streaming animation.
    - **Mechanism**: Audio is chunked into 200ms segments. Whisper encoding takes 7ms, the AR Transformer generates 32 tokens (about 6ms per 16 tokens), and motion decoding takes 10ms. Stage 2's VAE encoding/decoding takes 25ms, and the ETM takes 21ms. The total time for a single block is approximately 180ms, which is less than the 200ms audio duration, ensuring real-time performance. After generating 4 frames, interpolation is applied to reach 5 frames to achieve 25 FPS.
    - **Design Motivation**: The 200ms chunking represents a natural constraint of Whisper and aligns with the human perception threshold for audio-video synchronization.

### Loss & Training
- RVQ Loss: $\mathcal{L}_{vq} = \mathcal{L}_{recon} + \mathcal{L}_{commit}$
- AR Loss: $\mathcal{L}_{ar} = \sum[\mathcal{L}_{head0} + \mathcal{L}_{head1} + \|\mathcal{L}_{head0} - \mathcal{L}_{head1}\|_2^2]$
- ETM Loss: Reconstruction loss with region-specific masking $\mathcal{L}_{ETM}$
- Pre-training Data: AV Speech (662h) + VFHQ (2h), SFT Data (32h)

## Key Experimental Results

### Main Results (HDTF Dataset)

| Method | FID↓ | FVD↓ | Sync-C↑ | Sync-D↓ | 1s Generation Time |
|------|------|------|---------|---------|-----------|
| SadTalker | 22.18 | 233.67 | 7.326 | 7.848 | 18.89s |
| EchoMimic | 23.05 | 290.19 | 6.664 | 8.839 | 31.10s |
| AniPortrait | 28.16 | 235.10 | 4.547 | 10.657 | 29.36s |
| Hallo | 20.64 | 174.19 | 7.497 | 7.741 | 20.93s |
| **Teller** | 21.35 | **173.46** | **7.696** | **7.536** | **0.92s** |
| Real Video | - | - | 8.094 | 6.976 | - |

### Ablation Study

| Configuration | FVD↓ | Sync-C↑ | Description |
|------|------|---------|------|
| Full Teller | 173.46 | 7.696 | Full model |
| w/o ETM | ~190 | ~7.5 | Rigid accessory movement |
| w/o Dual-head prediction | ~185 | ~7.6 | Inference speed drops by ~40% |
| Single token prediction | - | - | Halves speed but achieves comparable quality |

### Key Findings
- Teller’s inference speed is **22.7 times** faster than Hallo (0.92s vs 20.93s) while achieving better FVD and Sync-C/D metrics.
- Quantitative and qualitative improvements in neck muscle and accessory movements brought by the ETM are prominent, demonstrating a significant advantage in human evaluations.
- Compressing 4 frames into 32 tokens represents the optimal trade-off between frame count and redundancy.
- On the RAVDESS emotional dataset, Teller performs particularly well on "angry" and "disgusted" expressions.

## Highlights & Insights
- **First autoregressive real-time talking head framework**, fracturing the stereotype that "high-quality equals slow diffusion models". It demonstrates the capability of AR transformers for audio-driven animation.
- **Dual-token prediction heads** are elegant and efficient—predicting two tokens per step doubles the inference speed, while the regularization term ensures balanced learning between the two heads.
- **ETM module** resolves the long-ignored physical movements of accessories in implicit keypoint-driven methods, while maintaining real-time processing via single-step refinement.

## Limitations & Future Work
- Based on LivePortrait's implicit keypoint representation, it inherits the limitation of handling extreme side profile angles.
- The 200ms chunking introduces a fixed latency, which may be insufficient for ultra-low latency scenarios.
- The region mask of ETM relies on MediaPipe keypoint detection, which may fail under facial occlusions.
- Only upper-body synthesis is supported; full-body animation is a future research direction.

## Related Work & Insights
- **vs Hallo/EMO**: Diffusion models deliver high generation quality but require 20+ seconds to generate one second of video, making them incapable of real-time operation. Teller replaces diffusion with AR to achieve a 22x speedup.
- **vs SadTalker**: Also a non-diffusion method, but SadTalker uses FaceVid2Vid for synthesis and lacks accessory movement modeling. Teller's ETM fills this gap.
- **vs VASA-1**: VASA-1 also uses diffusion for motion latent generation but is not real-time. Teller's RVQ + AR scheme enables real-time generation.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of AR + RVQ in talking head generation; ETM design is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-dataset comparisons, human evaluation, and real-time performance analysis.
- Writing Quality: ⭐⭐⭐ Clear framework but the writing style is slightly unpolished (with minor spelling errors in draft).
- Value: ⭐⭐⭐⭐⭐ A milestone for real-time streaming talking heads with high industrial applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Wav2Sem: Plug-and-Play Audio Semantic Decoupling for 3D Speech-Driven Facial Animation](wav2sem_plug-and-play_audio_semantic_decoupling_for_3d_speech-driven_facial_anim.md)
- [\[CVPR 2025\] HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation](hunyuanportrait_implicit_condition_control_for_enhanced_portrait_animation.md)
- [\[ICLR 2026\] Real-Time Motion-Controllable Autoregressive Video Diffusion](../../ICLR2026/video_generation/real-time_motion-controllable_autoregressive_video_diffusion.md)
- [\[CVPR 2025\] Parallelized Autoregressive Visual Generation](parallelized_autoregressive_visual_generation.md)
- [\[CVPR 2026\] PersonaLive! Expressive Portrait Image Animation for Live Streaming](../../CVPR2026/video_generation/personalive_expressive_portrait_image_animation_for_live_streaming.md)

</div>

<!-- RELATED:END -->
