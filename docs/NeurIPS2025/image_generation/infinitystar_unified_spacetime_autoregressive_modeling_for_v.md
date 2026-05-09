---
title: >-
  [Paper Note] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation
description: >-
  [Image Generation] This paper proposes InfinityStar, the first purely discrete autoregressive model capable of generating industrial-grade 720p video. Through spacetime pyramid modeling, it unifies T2I/T2V/I2V/interactive long video generation, achieving a VBench score of 83.74 that surpasses HunyuanVideo, with inference speeds 10–32× faster than diffusion models.
tags:
  - Image Generation
date: 2026-05-08
content_hash: 6841b67546af92ff
---

# InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation

## Basic Information
- **arXiv**: 2511.04675
- **Conference**: NeurIPS 2025 **Oral**
- **Authors**: Jinlai Liu, Jian Han, Bin Yan, Hui Wu, et al.
- **Institution**: ByteDance
- **Code**: https://github.com/FoundationVision/InfinityStar

## TL;DR
This paper proposes InfinityStar, the first purely discrete autoregressive model capable of generating industrial-grade 720p video. Through spacetime pyramid modeling, it unifies T2I/T2V/I2V/interactive long video generation, achieving a VBench score of 83.74 that surpasses HunyuanVideo, with inference speeds 10–32× faster than diffusion models.

## Background & Motivation
The two dominant paradigms for video generation each have critical shortcomings:
- **Diffusion models** (Sora, HunyuanVideo, Wan): High quality but slow inference (dozens of denoising steps), and difficult to naturally extend to video extrapolation.
- **Autoregressive models** (Emu3, Nova): Support streaming generation, but next-token prediction requires thousands of inference steps and trails diffusion models in quality.

VAR (Visual AutoRegressive) and Infinity have demonstrated that next-scale prediction (coarse-to-fine) can match diffusion models in image generation at significantly higher speeds. The core goal of InfinityStar is to extend this paradigm to video generation.

## Core Problem
How to extend spatial next-scale prediction to the spacetime dimension to achieve high-quality, efficient, and unified visual generation?

## Method

### 1. Spacetime Pyramid Modeling
Core design: decompose video into an **Image Pyramid + Clip Pyramids**.
- The first frame serves as $c_1$ ($T=1$), encoding static appearance.
- Subsequent clips share the same duration $T > 1$, encoding dynamic motion.
- Within each clip, multi-scale pyramids ($K$ scales) are applied along the spatial dimension, while the temporal dimension remains unchanged.

The autoregressive likelihood is:
$$p(r_1^1, \ldots, r_K^N) = \prod_{c=1}^N \prod_{k=1}^K p(r_k^c | r_1^1, \ldots, r_{k-1}^c, \psi(t))$$

This design: (1) decouples appearance and motion; (2) allows direct inheritance of T2I model knowledge; (3) naturally supports I2V and video extrapolation.

### 2. Visual Tokenizer Innovations
**Knowledge Inheritance from Continuous VAE**:
- Reuses the architecture and weights of the Wan 2.1 VAE, inserting a parameter-free quantizer (BSQ) between the encoder and decoder.
- Introduces no codebook; instead applies binary spherical quantization directly.
- Achieves video reconstruction with zero fine-tuning, with significant quality improvement after fine-tuning.
- Converges several times faster than training from scratch.

**Stochastic Quantizer Depth (SQD)**:
- Problem: in multi-scale quantization, information is heavily biased toward the final few scales, leaving early scales with little useful representation.
- Solution: during training, randomly drop the last $N$ scales with probability $p$, forcing the model to encode more information in early scales.
- Effect: substantial improvement in early-scale reconstruction quality; VBench +0.21.

### 3. Spacetime Autoregressive Transformer
**Semantic Scale Repetition (SSR)**:
- Early scales determine overall layout and motion direction ("semantic scales").
- The first $K_s=12$ scales are repeated $N=3$ times, allowing the model to iteratively refine semantic representations.
- Computational overhead is nearly negligible (early-scale tokens account for a minimal fraction of total tokens).
- VBench gain: 75.72 → 81.28 (+5.56, a substantial improvement).

**Spacetime Sparse Attention (SSA)**:
- Within each clip, attention is restricted to inputs from preceding scales only (not all historical tokens).
- Across clips, attention is limited to the largest scale of the immediately preceding clip (rather than the full history).
- 1.5× faster than full attention at 192p, and significantly more efficient at 480p (from OOM to feasible).
- Achieves better performance than full attention (81.28 vs. 80.77) by reducing exposure bias and error accumulation.

**Spacetime RoPE**: decomposed into four components: scale, time, height, and width.

### 4. Long Interactive Video Generation (InfinityStar-Interact)
- Sliding window approach: long videos are decomposed into 10s chunks with 5s overlap.
- **Semantic-Detail Conditions**:
    - Detail features: full-resolution features from the last $K$ frames of the preceding clip.
    - Semantic features: semantically compressed features obtained by spatially downsampling the preceding clip.
    - Condition tokens are compressed from 33.6K to 5.8K.

## Key Experimental Results

### T2I Generation

| Model | Params | GenEval Overall | DPG Overall |
|---|---|---|---|
| FLUX-dev | 12B | 0.67 | 84.0 |
| Infinity | 2B | 0.73† | 83.46 |
| **InfinityStar-T2I** | **8B** | **0.79†** | **86.55** |

### T2V Generation (VBench)

| Model | Type | VBench Overall |
|---|---|---|
| HunyuanVideo | Diffusion (13B) | 83.24 |
| Wan 2.1 | Diffusion (14B) | 84.70 |
| Emu3 | AR (8B) | 80.96 |
| Nova | AR (0.6B) | 80.12 |
| **InfinityStar** | **AR (8B)** | **83.74** |

### Inference Latency (5s 720p)

| Model | Latency | Speedup |
|---|---|---|
| Wan 2.1 (14B) | 1864s | 1× |
| Nova (0.6B) | 354s | 5× |
| **InfinityStar (8B)** | **58s** | **32×** |

### Ablation Study (192p)

| Configuration | VBench Total |
|---|---|
| Full model | 81.28 |
| w/o SSR | 75.72 (−5.56) |
| w/o Spacetime Pyramid | 80.30 (−0.98) |
| w/o SQD | 81.07 (−0.21) |
| Full Attention | 80.77 (−0.51) |

## Highlights & Insights
1. **Milestone work (Oral)**: the first discrete AR model to produce industrial-grade 720p video.
2. **Overwhelming speed advantage**: 32× faster than Wan 2.1; generates a 5s 720p video in 58s on a single GPU.
3. **Unified framework**: a single model covers T2I, T2V, I2V, video extrapolation, and interactive long video generation.
4. **Knowledge inheritance strategy**: the weight transfer from a continuous VAE to a discrete tokenizer is elegant and highly practical.
5. **SSR contributes enormously**: repeating only the early scales yields a +5.56 leap in VBench score.

## Limitations & Future Work
1. A trade-off between image quality and motion fidelity exists in high-motion scenes.
2. Model scale and training compute remain below those of top-tier diffusion models due to resource constraints.
3. The inference pipeline is not yet fully optimized, leaving room for further acceleration.
4. Long interactive generation is susceptible to accumulated errors, leading to quality degradation over multiple rounds.

## Related Work & Insights
- **vs. Emu3**: Emu3 uses next-token prediction (token-by-token); InfinityStar uses next-scale prediction (scale-by-scale), requiring orders of magnitude fewer inference steps.
- **vs. Nova**: Nova adopts spatial set-by-set and temporal frame-by-frame prediction; InfinityStar adopts spatial multi-scale and temporal clip-by-clip prediction.
- **vs. HunyuanVideo/Wan**: diffusion models require dozens of denoising steps; InfinityStar performs one forward pass per scale, with total steps of $K \times N$.
- **vs. Infinity (T2I)**: InfinityStar extends Infinity to video by introducing spacetime pyramids and several video-specific optimizations.
- **vs. SANA-Sprint**: both focus on efficient AR-style generation, but SANA-Sprint targets image generation (continuous tokens + diffusion hybrid), while InfinityStar targets video generation (purely discrete).

### Broader Implications
- **AR vs. Diffusion competition**: InfinityStar demonstrates that discrete AR models can match or surpass diffusion models in video quality while maintaining a 10–30× speed advantage, potentially signaling a paradigm shift in video generation.
- **Generalizability of the knowledge inheritance paradigm**: the weight transfer strategy from continuous VAE to discrete tokenizer is applicable to other VQ-based systems.
- **Connection to FramePack**: InfinityStar-Interact draws inspiration from FramePack's condition compression strategy, further validating the effectiveness of dual-path "semantic + detail" conditioning.

## Rating
- **Novelty**: ★★★★★ — Multiple innovations including spacetime pyramid, knowledge inheritance, SQD, and SSR.
- **Technical Depth**: ★★★★★ — Full-stack optimization spanning tokenizer, transformer, and training strategy.
- **Experimental Thoroughness**: ★★★★★ — Comprehensive coverage of T2I/T2V/I2V/extrapolation/interaction/human evaluation/ablation.
- **Writing Quality**: ★★★★★ — Oral paper with clear structure and deep insight.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](../../ICCV2025/image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](../../ICCV2025/image_generation/rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](../../ICCV2025/image_generation/matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](../../ICCV2025/image_generation/lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] Rethinking Layered Graphic Design Generation with a Top-Down Approach](../../ICCV2025/image_generation/rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)

</div>

<!-- RELATED:END -->
