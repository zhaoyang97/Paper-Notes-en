---
title: >-
  [Paper Note] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars
description: >-
  [CVPR 2026][Image Generation][real-time digital human] This paper proposes a two-stage autoregressive adaptation framework (autoregressive distillation + adversarial refinement) that converts a bidirectional human video diffusion model into a real-time streaming generator. Reference Sink, RAPR positional re-encoding, and a consistency-aware discriminator are introduced to ensure long-video stability, realizing the first full-body real-time digital human that supports both speaking and listening interactions.
tags:
  - CVPR 2026
  - Image Generation
  - real-time digital human
  - streaming video generation
  - autoregressive distillation
  - speaking-listening interaction
  - diffusion models
date: 2026-05-08
content_hash: e2c2046eac651976
---

# StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars

**Conference**: CVPR 2026
**arXiv**: [2512.22065](https://arxiv.org/abs/2512.22065)
**Code**: [https://streamavatar.github.io](https://streamavatar.github.io)
**Area**: Image Generation
**Keywords**: real-time digital human, streaming video generation, autoregressive distillation, speaking-listening interaction, diffusion models

## TL;DR
This paper proposes a two-stage autoregressive adaptation framework (autoregressive distillation + adversarial refinement) that converts a bidirectional human video diffusion model into a real-time streaming generator. Reference Sink, RAPR positional re-encoding, and a consistency-aware discriminator are introduced to ensure long-video stability, realizing the first full-body real-time digital human that supports both speaking and listening interactions.

## Background & Motivation

1. **State of the Field**: Diffusion models have achieved remarkable success in audio-driven talking avatar generation, enabling high-quality talking videos from a single image. Representative works include Hallo3, OmniAvatar, and HunyuanVideo-Avatar.

2. **Limitations of Prior Work**: Three major challenges hinder practical deployment:
    - **Real-time streaming generation**: The iterative denoising (25–50 steps) and long-context bidirectional attention in diffusion models are computationally prohibitive, and bidirectional attention is inherently incompatible with streaming. Existing methods require 7–74 minutes to generate a 5-second video.
    - **Long-term stability**: Streaming interaction demands continuous generation of long videos, but autoregressive approaches tend to accumulate errors, leading to identity drift and quality degradation.
    - **Speaking–listening interaction**: Existing methods model only the speaking behavior and ignore the listening state. In conversational scenarios, neglecting the listening state renders interactions unnatural. The few methods that do model listening are limited to the head-and-shoulder region, lacking gesture expressiveness and full-body representation.

3. **Root Cause**: High quality requires powerful bidirectional diffusion models, whereas real-time streaming demands lightweight causal models. The tension between quality and speed is the central conflict.

4. **Paper Goals**: To efficiently convert a high-fidelity but non-causal human video diffusion model into a real-time, streaming, interaction-capable generator.

5. **Starting Point**: A strong bidirectional teacher model supporting speaking–listening interaction is trained first, then compressed into a 3-step causal autoregressive student model via two-stage distillation and adversarial refinement. Dedicated attention mechanisms and positional encoding improvements are proposed to address long-video stability.

6. **Core Idea**: Autoregressive distillation compresses the denoising process from 40+ steps to 3 steps; Reference Sink and RAPR address identity drift; together they enable generation of a 5-second 720p video in 20 seconds.

## Method

### Overall Architecture
Built upon Wan2.2-TI2V-5B as the backbone (30 DiT blocks), the framework first trains a bidirectional teacher model supporting speaking–listening interaction, then converts it into a real-time streaming student model in two stages:
- **Stage 1**: Autoregressive distillation — bidirectional attention is converted to block-wise causal attention, and Score Identity Distillation reduces denoising steps from 40+ to 3.
- **Stage 2**: Adversarial refinement — a consistency-aware discriminator is used in adversarial training to repair quality degradation caused by distillation.

### Key Designs

1. **Autoregressive Distillation (Stage 1)**:

    - **Function**: Converts the bidirectional diffusion model into a real-time causal autoregressive generator.
    - **Mechanism**: The generation window is divided into a reference frame chunk (1 frame) and generation chunks (each containing $C=3$ frames). Causal attention is applied across chunks while bidirectional attention is retained within each chunk. A rolling KV-cache stores context over a finite window. Distillation proceeds in two steps: (a) ODE initialization — the teacher model generates videos with recorded denoising trajectories, and the student is trained to predict $\{x_t^0\}$ from $\{x_t^n\}$; (b) Score Identity Distillation — a student-forcing scheme is adopted where the student predicts the next chunk conditioned on its own previous outputs, mitigating the train–test mismatch. It is found that skipping the KV-cache update step (conditioning on noisy $\{x_t^1\}$ rather than clean $\{x_t^0\}$) incurs negligible quality loss while saving one forward pass.
    - **Design Motivation**: Directly applying the bidirectional model for streaming is infeasible (requiring the full sequence), and 40+ denoising steps are too slow. Block-causal attention preserves local bidirectional dynamics while enabling autoregressive generation.

2. **Reference Sink + RAPR (Positional Encoding Improvement)**:

    - **Function**: Addresses identity drift and quality degradation in long video generation.
    - **Mechanism**: **Reference Sink**: The KV pairs of the reference frame are permanently retained in the rolling KV-cache, ensuring the model can always attend to the original identity. The KV pairs of the first generated chunk are additionally retained to further improve consistency. **RAPR (Reference-Anchored Positional Re-encoding)**: Resolves two problems with standard RoPE — (1) train–test mismatch (the model sees only short sequences during training but encounters out-of-distribution large position indices at inference), and (2) the inherent long-range attention decay of RoPE reducing attention to reference frames. RAPR stores un-encoded keys, and when generating the current frame $x_t$, computes a capped distance $\min(t, D)$ (where $D < T$) to the reference frame as the RoPE index, simultaneously adjusting the relative positions of all cached keys, then applies RoPE uniformly. This (a) bounds the maximum distance to prevent attention decay, and (b) keeps both training and inference within a bounded position space, eliminating the OOD problem.
    - **Design Motivation**: Without Reference Sink, the model loses identity information as frames are evicted from the cache. Without RAPR, even with Sink, RoPE's decay characteristics and OOD position indices still cause instability in long videos. A key elegance of RAPR is that short videos can simulate the positional offsets of long-video inference during training, without requiring actual long video generation.

3. **Consistency-Aware Discriminator (Stage 2 Adversarial Refinement)**:

    - **Function**: Repairs quality degradation after distillation (blurriness, hand/teeth artifacts) and enhances temporal consistency.
    - **Mechanism**: The discriminator is initialized from the pretrained teacher backbone, with $N_Q=3$ Q-Former modules inserted at intermediate layers to extract deep features. Two output branches: (a) **Local authenticity branch** — a linear projection over per-frame features produces frame-wise logits evaluating single-frame quality; (b) **Global consistency branch** — cross-attention between reference frame features and all subsequent frame features outputs a single logit penalizing deviation from the reference identity. Relativistic adversarial loss with R1/R2 gradient penalties is used. Crucially, the adversarial stage trains on real video data to directly push the generated distribution toward the real distribution.
    - **Design Motivation**: Distillation inevitably degrades quality. A conventional discriminator focuses only on per-frame authenticity and cannot resolve inter-frame consistency issues. The global consistency branch explicitly constrains all frames to remain identity-consistent with the reference frame.

4. **Speaking–Listening Interaction Model**:

    - **Function**: Enables digital humans to speak and listen naturally.
    - **Mechanism**: An **Audio Mask** distinguishes speaking and listening phases — obtained via TalkNet (joint audio-visual detection), which is more accurate than audio-separation-based approaches. The audio mask is applied after Wav2Vec 2.0 feature extraction (rather than before), avoiding waveform modification that would cause feature distribution shift. Two audio attention modules are added within each DiT block: Talk Audio Attention injects speaking audio to drive expressions and gestures, while Listen Audio Attention injects listening audio to drive natural reactive movements. The text prompt is fixed as "a person is speaking and listening."
    - **Design Motivation**: Audio separation modifies the waveform, causing Wav2Vec-extracted features to deviate from the pretrained distribution. Ablation experiments (Pre-Mask vs. Ours) confirm that post-Wav2Vec masking outperforms pre-masking on all metrics.

### Loss & Training
- Teacher model: fine-tuned from Wan2.2-TI2V-5B for 20,000 steps, batch size 32, lr 5e-6.
- Student Stage 1: ODE initialization for 5,000 steps (bs 8, lr 2e-6) + SiD distillation for 6,000 steps (bs 16, lr 3e-6).
- Student Stage 2: Adversarial refinement for 1,400 steps (bs 32, lr 5e-6).
- Training data: ~200 hours of 720p video (SpeakerVid-5M + self-collected), with speaking/listening samples balanced according to TalkNet-detected listening ratios.
- At inference, the DiT and VAE decoder are pipelined across two H800 GPUs, with a latency of 1.2 seconds.

## Key Experimental Results

### Main Results (Talking Video Generation)

| Method | FID ↓ | FVD ↓ | IQA ↑ | Sync-C ↑ | HKV (Gesture) | HA ↑ | Steps | Resolution | Time (5s) |
|--------|-------|-------|-------|----------|--------------|------|-------|------------|-----------|
| StableAvatar | 75.20 | 603.54 | 4.66 | 4.24 | 42.92 | 0.909 | 40 | 480p | 12 min |
| OmniAvatar | 87.24 | 851.93 | 4.45 | 7.60 | 8.64 | 0.974 | 25 | 480p | 36 min |
| HY-Avatar | 76.49 | 557.46 | 4.67 | 6.71 | 54.31 | 0.947 | 50 | 720p | 74 min |
| EchoMimicV3 | 78.65 | 724.29 | 4.66 | 3.10 | 25.53 | 0.969 | 25 | 480p | 7 min |
| **Ours** | **74.21** | **707.34** | **4.68** | **7.06** | 48.35 | **0.974** | **3** | **720p** | **20 s** |

### Ablation Study (Incremental Component Addition)

| Configuration | FID ↓ | IQA ↑ | Sync-C ↑ | HA ↑ |
|---------------|-------|-------|----------|------|
| Baseline (Self Forcing) | 96.58 | 4.29 | 7.04 | 0.948 |
| + Reference Sink | 88.75 | 4.55 | 7.03 | 0.950 |
| + RAPR | 81.63 | 4.64 | 7.06 | 0.956 |
| + GAN w/o consistency discriminator | 79.68 | 4.65 | 7.05 | 0.947 |
| **Full (Ours)** | **74.21** | **4.68** | **7.06** | **0.974** |

### Interaction Capability (Listening-Phase Motion Dynamics)

| Method | LBKV (Body) | LHKV (Hand) | LFKV (Face) |
|--------|-------------|-------------|-------------|
| Baseline (silent audio) | 6.05 | 4.53 | 2.39 |
| **Ours** | **15.88** | **16.24** | **7.11** |

### Key Findings
- The speed improvement is dramatic: 3 steps vs. 25–50 steps; generating a 5-second video takes only 20 seconds vs. 7 minutes for the fastest baseline (21× speedup), at a higher resolution of 720p.
- Reference Sink is critical for identity preservation (FID drops from 96.58 to 88.75); RAPR further improves long-video stability (FID to 81.63).
- The global branch of the consistency-aware discriminator is essential — removing it causes significant HA degradation compared to a standard discriminator on long-video data.
- Listening-state motion richness (LHKV) is 3.6× that of the baseline, demonstrating that the model has learned natural listening reactions.

## Highlights & Insights
- RAPR is an elegant positional encoding solution — by capping the maximum distance and dynamically re-encoding all cached keys, it simulates long-video inference positional offsets using short videos during training, without requiring actual long video generation. This idea is broadly applicable to other RoPE-based models requiring long-sequence inference.
- The finding that "skipping the KV-cache update step during training" is practically useful — conditioning the next chunk on noisy outputs rather than clean ones does not affect quality but saves one forward pass, suggesting that autoregressive generation is robust to mild noise.
- Applying the audio mask after Wav2Vec rather than before is a subtle but important design choice — preserving the original waveform yields substantially higher-quality Wav2Vec features, an insight relevant to all works employing pretrained audio features.

## Limitations & Future Work
- Limited temporal context may cause inconsistent content in regions that remain occluded for extended periods.
- Distillation inevitably constrains the range of motion.
- Text input handling is simplistic (fixed prompt), lacking fine-grained semantic control.
- VAE decoding accounts for more than half of total latency and is the primary bottleneck for further latency reduction.
- Only single-person interaction is currently supported; multi-person conversational scenarios are worth exploring.

## Related Work & Insights
- **vs. CausVid/Self-Forcing**: StreamAvatar augments the autoregressive distillation framework with Reference Sink, RAPR, and a consistency-aware discriminator, specifically addressing identity stability in digital human scenarios.
- **vs. Hallo3/EchoMimicV3**: These methods achieve reasonable quality but are slow (7–32 minutes per 5 seconds) and suffer from hand artifacts and identity drift in long sequences. StreamAvatar achieves better quality while being 21× faster.
- **vs. INFP/ARIG**: These methods support speaking–listening interaction but are limited to the head-and-shoulder region. StreamAvatar is the first real-time model supporting full-body speaking–listening interaction.

## Rating
- Novelty: ⭐⭐⭐⭐ — The two-stage framework is well-motivated; RAPR is a novel positional encoding improvement; the full-body speaking–listening interactive model is the first of its kind.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive comparisons, detailed ablations, user studies, and real-time performance analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with precise technical descriptions.
- Value: ⭐⭐⭐⭐⭐ — Real-time interactive digital humans address a pressing practical need; 20 seconds per 5-second clip makes real-world deployment feasible.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation](../../ICCV2025/image_generation/streamdiffusion_a_pipeline-level_solution_for_real-time_interactive_generation.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](dcw_snr_t_bias_diffusion.md)
- [\[CVPR 2026\] Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)

<!-- RELATED:END -->
