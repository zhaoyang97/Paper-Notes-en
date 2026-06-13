---
title: >-
  [Paper Note] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks
description: >-
  [ICCV 2025][Video Generation][diffusion transformer] This paper proposes Video Interface Networks (VINs), an abstraction module analogous to "fast thinking…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "diffusion transformer"
  - "parallel inference"
  - "temporal consistency"
  - "long video"
date: 2026-05-08
content_hash: 0ab937dd07c1add6
---

# Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks

**Conference**: ICCV 2025
**arXiv**: [2503.17539](https://arxiv.org/abs/2503.17539)  
**Code**: None  
**Area**: Video Generation
**Keywords**: video generation, diffusion transformer, parallel inference, temporal consistency, long video

## TL;DR

This paper proposes Video Interface Networks (VINs), an abstraction module analogous to "fast thinking," which encodes long videos into fixed-size global tokens at each diffusion step to guide a DiT in generating multiple video chunks in parallel, enabling efficient and temporally consistent long video generation.

## Background & Motivation

- **Diffusion Transformers (DiTs)** can generate high-quality short videos, but extending them to long videos faces a quadratic complexity bottleneck.
- Full-attention approaches on long videos lead to motion stagnation and repetition.
- **Autoregressive approaches** (chunk-by-chunk generation) suffer from catastrophic forgetting, subject inconsistency, and temporal incoherence.
- Existing parallel methods (e.g., FreeNoise, FreeLong) rely on handcrafted templates (noise rescheduling, frequency-band filtering) as consistency priors, capturing only shallow visual features and lacking deep semantic abstraction.
- The work is inspired by Kahneman's dual-process theory of human cognition: System 1 (fast intuition) + System 2 (slow reasoning). DiTs operate solely as System 2 and lack the global abstraction capability of System 1.

## Method

### Overall Architecture

At each diffusion timestep: (1) the VIN encodes global semantics from the noisy input into fixed-size global tokens; (2) the DiT uses these global tokens to denoise each video chunk in parallel; (3) overlapping regions maintain consistency via token fusion. The VIN and DiT are jointly trained end-to-end.

### Key Designs

1. **Video Interface Network (VIN)**: The VIN consists of three components:

    - **Global Tokens**: Fixed-size learnable embeddings $Z_{init} \in \mathbb{R}^{N_{global} \times d}$ (512 tokens, dimension 4096), independent of the input.
    - **VIN Encoder**: Samples keyframes from the input video at $T_s = 1.0$ second intervals and encodes video information into the global tokens via cross-attention (global tokens as queries, video tokens as key-values).
    - **VIN Processor**: Four self-attention blocks (32 heads) that iteratively refine the global tokens while incorporating text prompt embeddings.
    - Core advantage: the size of global tokens is fixed and does not grow with video length, decoupling computation from input size and enabling scaling to arbitrarily long videos.

2. **End-to-End Joint Training Objective**: The noise distribution is factored as a product of conditional distributions over each chunk: $P_\theta(\epsilon_t|X_t,t,Z_t) = \prod_i P_\theta(\epsilon_t^i | X_t^i, t, Z_t)$. The loss function is $\mathcal{L}_{\alpha,\theta} = \mathbb{E}[\sum_i \|\epsilon_\theta([X_t^i, Z_t], t) - \epsilon_t^i\|^2]$. Each chunk additionally receives a local context of the last $F_{local}=8$ frames from the preceding chunk (with stop-gradient to prevent inter-chunk gradient interference).

3. **Token Fusion at Inference**: Overlapping regions between adjacent chunks are merged via weighted averaging: $\hat{\epsilon}_t^{fused}[k] = \frac{(\mathcal{F}_{local} - \mathcal{W}(k))\hat{\epsilon}_t^i[k] + \mathcal{W}(k)\hat{\epsilon}_t^{i+1}[k]}{\mathcal{F}_{local}}$, where $\mathcal{W}(k)$ denotes the relative temporal position. An **early fusion** strategy ($t > t_\alpha = 20$) is adopted, as fusion in the early stages of the sampling chain yields the best results.

### Loss & Training

- Training data: 840,000 annotated videos, mixed at 64/128/256 frames (20/40/80 latent frames).
- Chunk size $F_{chunk}=20$ latent frames, local context $F_{local}=8$ latent frames, 512 global tokens.
- Inference: 50-step reverse diffusion, extended $F_{local}=12$, early fusion cutoff $t_\alpha=20$.
- Backbone: a pretrained latent video DiT based on a modified Open-Sora, with a 3D VAE encoding 16 frames into 5 latent frames, resolution 192×320, 16 FPS.

## Key Experimental Results

### Main Results

**VBench Long Evaluation (higher is better; Dynamic Degree requires balance)**:

| Method | Subject Consistency | Background Consistency | Characteristic |
|---|---|---|---|
| Full Attention | Degrades with length | High but dynamic degree drops sharply | Motion stagnation |
| AutoRegressive | Lower than VIN | Lower than VIN | Catastrophic forgetting |
| StreamingT2V | Lowest | Lowest | Insufficient memory module |
| FreeNoise | Medium | Medium | Shallow prior |
| Spectral Blending | Medium | Medium | Limited frequency-domain filtering |
| **VIN (Ours)** | **Highest** | **Highest** | Preserves dynamics |

**Optical Flow Analysis (MAWE↓)**:

| Method | 64 frames | 128 frames | 256 frames | 512 frames |
|---|---|---|---|---|
| AutoRegressive | ~2.5 | ~3.0 | ~3.5 | ~4.5 |
| FreeNoise | ~2.0 | ~2.5 | ~3.0 | ~4.0 |
| Full Attention | ~1.5 | ~2.0 | ~2.5 | ~3.5 |
| **VIN** | **~1.0** | **~1.1** | **~1.5** | **<2.0** |

### Ablation Study

| Configuration | MAWE↓ | Scene Cuts↓ |
|---|---|---|
| Full Model | **1.09** | **0.21** |
| w/o Global Tokens | 1.69 | 0.33 |
| w/o fusion | 1.13 | 1.00 |
| Mid fusion | 1.11 | 0.33 |
| Late fusion | 1.22 | 0.74 |
| Local 8 / 10 frames | 1.51 / 1.17 | 0.24 / 0.22 |
| Keyframe 0.5s / 0.2s | 1.14 / 1.21 | 0.34 / 0.29 |

### Key Findings

- **Global tokens are the core component**: removing them raises MAWE from 1.09 to 1.69, the most severe degradation observed.
- **Early fusion is the most effective**: consistent with the intuition that diffusion models form object structure in the early stages of sampling.
- **Dense keyframe sampling is not beneficial**: $T_s = 0.2s$ actually underperforms $T_s = 1.0s$, indicating that the VIN's semantic encoding possesses redundancy-suppression capability.
- VIN reduces FLOPs by **25–40%** and achieves **40–75%** speedup compared to full attention, with only a marginal increase in memory.
- In user studies, VIN is preferred by human evaluators for both overall appearance and temporal consistency (loss rate < 30%).
- VIN attention heads exhibit semantic specialization: different heads focus on human bodies, architectural structures, objects, etc.

## Highlights & Insights

- **An elegant dual-system analogy**: VIN as System 1 for global abstraction and DiT as System 2 for local refinement, analogous to a painter's cognitive workflow.
- **End-to-end training**: compared to template-based methods (FreeNoise/FreeLong), the consistency prior is learned rather than manually designed, yielding a more principled formulation.
- **Dynamically computed representations**: global tokens are recomputed at every timestep rather than serving as static anchors, enabling graceful degradation.
- **Stop-gradient design**: gradients are not propagated across shared chunk boundaries, preventing inter-chunk interference.

## Limitations & Future Work

- The VIN learns representations solely through the generation objective, without supervision from downstream tasks (e.g., segmentation, depth estimation).
- Modalities beyond the original patch input (e.g., depth, 3D information) remain unexplored.
- Resolution is limited to 192×320; practical applications require scaling to higher resolutions.
- The token fusion mechanism is relatively straightforward, and more effective fusion strategies may exist.

## Related Work & Insights

- Inspired by Recurrent Interface Networks (RINs), the method decouples semantic encoding from per-pixel denoising.
- Compared to the long-term memory module in StreamingT2V, VIN's global tokens are dynamic and cover the entire video.
- Unlike the shallow priors in FreeNoise/Spectral Blending, VIN learns deep semantic representations.
- The global tokens exhibit potential as general-purpose feature representations that could be integrated with video editing and video understanding tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ A dual-system paradigm for parallel video generation; the combination of global tokens and DiT is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation covering VBench, optical flow, scene cuts, user studies, and ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rich figures, and an intuitive dual-system analogy.
- **Value**: ⭐⭐⭐⭐⭐ Provides a scalable new paradigm for long video generation with strong practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[NeurIPS 2025\] MagCache: Fast Video Generation with Magnitude-Aware Cache](../../NeurIPS2025/video_generation/magcache_fast_video_generation_with_magnitudeaware_cache.md)
- [\[CVPR 2026\] FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](../../CVPR2026/video_generation/fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)
- [\[ICLR 2026\] SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion](../../ICLR2026/video_generation/sigmark_scalable_in-generation_watermark_with_blind_extraction_for_video_diffusi.md)
- [\[CVPR 2026\] OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens](../../CVPR2026/video_generation/omnilottie_generating_vector_animations_via_parameterized_lottie_tokens.md)

</div>

<!-- RELATED:END -->
