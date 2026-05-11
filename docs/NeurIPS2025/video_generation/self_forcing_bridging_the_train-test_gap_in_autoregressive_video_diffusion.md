---
title: >-
  [Paper Note] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
description: >-
  [NeurIPS 2025][Video Generation][Autoregressive video generation] This paper proposes the Self Forcing training paradigm, which eliminates the exposure bias caused by train-inference distribution mismatch in Teacher Forc…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Autoregressive video generation"
  - "exposure bias"
  - "distribution matching"
  - "real-time video generation"
  - "KV cache"
date: 2026-05-08
content_hash: 44da4649d2d60b86
---

# Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion

**Conference**: NeurIPS 2025
**arXiv**: [2506.08009](https://arxiv.org/abs/2506.08009)
**Code**: [https://github.com/self-forcing](https://github.com/self-forcing) (Project page: [https://self-forcing.github.io/](https://self-forcing.github.io/))
**Area**: Video Generation / Autoregressive Diffusion Models
**Keywords**: Autoregressive video generation, exposure bias, distribution matching, real-time video generation, KV cache

## TL;DR

This paper proposes the Self Forcing training paradigm, which eliminates the exposure bias caused by train-inference distribution mismatch in Teacher Forcing and Diffusion Forcing by performing autoregressive self-rollout during training and applying a holistic video-level distribution matching loss (DMD/SiD/GAN). Built on Wan2.1-1.3B, it achieves real-time streaming video generation at 17 FPS on a single GPU while matching or surpassing the quality of bidirectional diffusion models that are orders of magnitude slower.

## Background & Motivation

- **Background**: Video diffusion models have advanced substantially in recent years, but dominant approaches (e.g., Wan2.1, Sora) employ bidirectional attention to jointly denoise all frames, imposing two fundamental limitations: (1) future frames can influence past frames, violating causal structure; and (2) the entire video must be generated at once, precluding real-time streaming applications.

Autoregressive (AR) models generate video frame-by-frame and are naturally suited for real-time interactive scenarios (game simulation, robotics, live streaming, etc.), but face the core challenge of **Exposure Bias**. Specifically, both dominant training paradigms exhibit a train-inference distribution mismatch:

**Teacher Forcing (TF)**: During training, ground-truth frames are used as context to denoise the next frame, whereas at inference the model must condition on its own previously generated, imperfect frames — the model never observes its own erroneous outputs during training, causing errors to accumulate.

**Diffusion Forcing (DF)**: Context frames are corrupted with independent noise during training, which covers the "clean context + noisy current frame" inference scenario, yet the training outputs themselves still do not belong to the true distribution visited during inference.

- **Limitations of Prior Work**: CausVid also attempts distribution matching via DMD, but since it uses DF outputs during training (rather than the model's actual inference distribution), it effectively matches the wrong distribution.

- **Key Challenge**: The fundamental mismatch between what the model sees during training and what it encounters at inference.

- **Goal**: Eliminate exposure bias in autoregressive video diffusion by ensuring training and inference follow identical processes.

- **Key Insight**: The core insight is inspired by GANs — **a GAN generator undergoes exactly the same process during training and inference, inherently avoiding exposure bias.** Applied to autoregressive video diffusion, the model must also "consume its own outputs" during training.

## Method

### Overall Architecture

Self Forcing operates as a post-training procedure. During training, the model generates video frame-by-frame through autoregressive self-rollout: each frame is denoised conditioned on previously generated frames (not ground-truth frames), with causal dependencies maintained via KV cache. After generating a complete video, a holistic distribution matching loss aligns the generated video distribution to the real video distribution. This perfectly mirrors the inference process, fundamentally eliminating exposure bias.

### Key Designs

1. **Few-Step Diffusion Model + Stochastic Gradient Truncation**: Naïvely backpropagating through the full autoregressive chain of multi-step diffusion is computationally prohibitive. Self Forcing therefore uses a 4-step diffusion model to approximate the conditional distribution of each frame, restricts gradient backpropagation to the last denoising step of each frame (with step $s \in [1,T]$ sampled randomly), and truncates gradients across frames — KV cache gradients are not propagated to preceding frames. This renders training computationally feasible while preserving effectiveness.

2. **Holistic Distribution Matching Loss**: The framework supports three distribution matching objectives:

    - **DMD (Distribution Matching Distillation)**: Minimizes reverse KL divergence by leveraging the difference between real and fake score networks to guide gradient updates.
    - **SiD (Score Identity Distillation)**: Distribution matching based on Fisher divergence.
    - **GAN (R3GAN)**: Relativistic paired GAN loss with R1/R2 regularization.

   A critical distinction from conventional distillation: the objective is not to accelerate sampling but to **eliminate exposure bias** through distribution matching. All three losses achieve comparably strong results.

3. **Rolling KV Cache Mechanism**: A key innovation enabling arbitrary-length video generation. A fixed-size KV cache of $L$ frames is maintained; when a new frame is generated and the cache is full, the oldest entry is evicted, yielding $O(TL)$ complexity. Prior methods (e.g., CausVid/MAGI-1) require recomputing KV caches for overlapping frames during sliding-window inference, incurring $O(L^2 + TL)$ complexity. However, naïve Rolling KV Cache produces severe flickering — because the image latent statistics of the first frame are distinctive and always visible to the model during training, but not during rolling. The solution is to restrict the attention window during training so that the model cannot attend to the first chunk when denoising the last chunk, simulating the rolling scenario.

### Loss & Training

DMD training uses Wan2.1-14B as the real score network and 1.3B as the fake score network. GAN training employs a large batch size of 768 to maintain stability. All three Self Forcing variants converge in approximately 1.5–3 hours on 64 H100 GPUs. DMD/SiD training is entirely data-free, requiring no real video training data — only the score functions of a pretrained diffusion model.

## Key Experimental Results

### Main Results

| Model | Type | Params | Resolution | Throughput (FPS)↑ | Latency (s)↓ | VBench Total↑ | Quality↑ | Semantic↑ |
|-------|------|--------|------------|-------------------|--------------|--------------|----------|-----------|
| Wan2.1 | Diffusion | 1.3B | 832×480 | 0.78 | 103 | 84.26 | 85.30 | 80.09 |
| CausVid | Chunk AR | 1.3B | 832×480 | 17.0 | 0.69 | 81.20 | 84.05 | 69.80 |
| SkyReels-V2 | Chunk AR | 1.3B | 960×540 | 0.49 | 112 | 82.67 | 84.70 | 74.53 |
| **Self Forcing (chunk)** | Chunk AR | 1.3B | 832×480 | **17.0** | 0.69 | **84.31** | **85.07** | **81.28** |
| **Self Forcing (frame)** | Frame AR | 1.3B | 832×480 | 8.9 | **0.45** | 84.26 | 85.25 | 80.30 |

Self Forcing surpasses Wan2.1 on VBench — a bidirectional model that is 150× slower — and substantially outperforms CausVid.

### Ablation Study

| Training Paradigm | Distribution Matching | VBench Total↑ (chunk) | VBench Total↑ (frame) | Notes |
|-------------------|-----------------------|----------------------|----------------------|-------|
| Diffusion Forcing | None | 82.95 | 77.24 | DF degrades severely in frame-level AR |
| Teacher Forcing | None | 83.58 | 80.34 | TF relatively stable but still lags |
| DF + DMD | DMD | 82.76 | 80.56 | ≈ CausVid reproduction; matches wrong distribution |
| TF + DMD | DMD | 82.32 | 78.12 | TF outputs + DMD also insufficient |
| **Self Forcing** | DMD | **84.31** | **84.26** | Frame-level mode shows almost no degradation! |
| **Self Forcing** | SiD | 84.07 | 83.54 | SiD also performs well |
| **Self Forcing** | GAN | 83.88 | 83.27 | GAN equally effective |

### Key Findings

- **Exposure bias has a dramatic effect in frame-level AR**: DF drops sharply from 82.95 (chunk) to 77.24 (frame), whereas Self Forcing maintains 84.31→84.26 with almost no degradation — the strongest evidence for Self Forcing's efficacy.
- CausVid exhibits progressively accumulating oversaturation artifacts over time; Self Forcing resolves this entirely.
- Rolling KV Cache improves throughput for 10-second videos from 4.6 FPS (with KV recomputation) to 16.1 FPS.
- Self Forcing is surprisingly training-efficient: per-iteration wall-clock time is comparable to TF/DF, yet achieves superior quality at equal training time, owing to full attention enabling more efficient FlashAttention-3 kernels.

## Highlights & Insights

- **Paradigm-level innovation**: Elevates "parallel pretraining + sequential post-training" to a new paradigm, analogous to RLHF in LLMs but applied to video generation.
- **Deep integration of three generative paradigms — AR, diffusion, and GAN**: AR and diffusion models provide chain factorization and latent variable factorization respectively, while the distribution matching idea from GANs drives training — three paradigms complement each other.
- **Data-free training via DMD/SiD**: No real video training data is required; the score functions of a pretrained diffusion model suffice to convert a bidirectional model into a high-quality autoregressive model.
- **Elegant Rolling KV Cache design**: A single training modification — "cannot attend to the first chunk" — resolves the distribution shift problem.

## Limitations & Future Work

- Video quality still degrades when generating sequences substantially longer than the training context length.
- The gradient truncation strategy limits the model's ability to learn long-range dependencies.
- Validation is currently limited to the 1.3B model; the effectiveness of Self Forcing at larger scales (e.g., 14B) remains to be verified.
- Combination with sampling acceleration techniques such as timestep distillation has not been explored.
- Ethical risks of real-time video generation (deepfakes) necessitate corresponding detection and watermarking technologies.

## Related Work & Insights

- CausVid is the most direct comparison baseline — Self Forcing precisely identifies its fundamental flaw of "matching the wrong distribution."
- Scheduled Sampling and SeqGAN from the RNN era address analogous exposure bias problems — Self Forcing upgrades this idea to video diffusion models.
- GANs inherently have no exposure bias — the core source of inspiration.
- DeepSeek's RLHF/GRPO advances the "parallel pretraining + sequential post-training" paradigm in LLMs — Self Forcing represents the first corresponding attempt in the video domain.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[CVPR 2026\] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](../../CVPR2026/video_generation/infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[ICLR 2026\] TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](../../ICLR2026/video_generation/ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)

</div>

<!-- RELATED:END -->
