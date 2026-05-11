---
title: >-
  [Paper Note] StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation
description: >-
  [ICCV 2025][Image Generation][real-time generation] StreamDiffusion proposes a pipeline-level real-time diffusion framework that achieves up to 91 fps on a single RTX 4090—59.6× faster than Diffusers AutoPipeline—through Stream Batch (batched denoising steps), R-CFG (residual classifier-free guidance), and SSF (stochastic similarity filtering).
tags:
  - ICCV 2025
  - Image Generation
  - real-time generation
  - streaming diffusion
  - Stream Batch
  - residual CFG
  - stochastic similarity filtering
  - pipeline optimization
date: 2026-05-08
content_hash: e25774bc5c0547a2
---

# StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation

**Conference**: ICCV 2025
**arXiv**: [2312.12491](https://arxiv.org/abs/2312.12491)
**Code**: [GitHub](https://github.com/cumulo-autumn/StreamDiffusion)
**Area**: Diffusion Models · Image Generation
**Keywords**: real-time generation, streaming diffusion, Stream Batch, residual CFG, stochastic similarity filtering, pipeline optimization

## TL;DR

StreamDiffusion proposes a pipeline-level real-time diffusion framework that achieves up to 91 fps on a single RTX 4090—59.6× faster than Diffusers AutoPipeline—through Stream Batch (batched denoising steps), R-CFG (residual classifier-free guidance), and SSF (stochastic similarity filtering).

## Background & Motivation

Diffusion models have demonstrated strong performance in image and video generation, yet their throughput falls far short of the demands of **real-time interactive scenarios** such as augmented/virtual reality, live streaming, and game rendering.

Existing acceleration methods focus primarily on **reducing denoising steps** (e.g., LCM, consistency models) or **quantization**, all of which are model-level optimizations. The authors approach the problem from a **pipeline level** and identify the following issues:

**Serial denoising inefficiency**: The conventional approach waits for one image to be fully denoised before processing the next.

**Redundant CFG computation**: Each step requires UNet forward passes for both conditional and unconditional branches, wasting half the compute.

**Repeated computation in static scenes**: The GPU continues running even when inputs remain unchanged, wasting energy.

## Method

### 1. Stream Batch: Batched Denoising Steps

Core Idea: **Rather than waiting for a single image to finish denoising, a new input is accepted after each denoising step.** Denoising steps belonging to different images are interleaved into a batch for parallel processing.

For $n$ denoising steps, Stream Batch assembles a batch of size $n$ by interleaving steps across frames, completing the corresponding denoising step for all frames in a single UNet forward pass. An image encoded at timestep $t$ finishes generation at $t+n$.

**Temporal consistency enhancement**: Stream Batch naturally supports the use of **future frame information**, improving temporal consistency via cross-frame attention:

$$\text{Attn}(Q_{t,i}, K_{\text{Batch}}, V_{\text{Batch}}) = \text{Softmax}\left(\frac{Q_{t,i} \cdot K_{\text{Batch}}^T}{\sqrt{d}}\right) V_{\text{Batch}}$$

### 2. Residual Classifier-Free Guidance (R-CFG)

Standard CFG requires $2n$ UNet evaluations (conditional + unconditional). R-CFG replaces the unconditional prediction with a **virtual residual noise** derived from the original input image:

$$\epsilon_{\tau_i, \bar{c}'} = \frac{x_{\tau_i} - \sqrt{\alpha_{\tau_i}} x_0}{\sqrt{\beta_{\tau_i}}}$$

The final R-CFG formulation is:

$$\epsilon_{\tau_i, \text{cfg}} = \delta \epsilon_{\tau_i, \bar{c}'} + \gamma(\epsilon_{\tau_i, c} - \delta \epsilon_{\tau_i, \bar{c}'})$$

Two variants are proposed:
- **Self-Negative R-CFG**: 0 unconditional UNet evaluations (only $n$ total).
- **Onetime-Negative R-CFG**: 1 unconditional evaluation ($n+1$ total).

### 3. Stochastic Similarity Filtering (SSF)

The cosine similarity between the current frame $I_t$ and a reference frame $I_{\text{ref}}$ is computed, and computation is skipped with a probability defined as:

$$P(\text{skip} | I_t, I_{\text{ref}}) = \max\left\{0, \frac{S_C(I_t, I_{\text{ref}}) - \eta}{1 - \eta}\right\}$$

Probabilistic sampling rather than a hard threshold avoids video stuttering and produces smoother visual output.

## Key Experimental Results

### Throughput Comparison

| Denoising Steps | StreamDiffusion (ms) | StreamDiffusion w/o TRT (ms) | AutoPipeline (ms) |
|---------|---------------------|----------------------------|--------------------|
| 1 | 10.65 (**59.6×**) | 21.34 (29.7×) | 634.40 |
| 2 | 16.74 (39.3×) | 30.61 (21.3×) | 652.66 |
| 4 | 26.93 (25.8×) | 48.15 (14.4×) | 695.20 |
| 10 | 62.00 (13.0×) | 96.94 (8.3×) | 803.23 |

At one denoising step, approximately 91 fps is achieved; at ten steps, 16 fps is still maintained—significantly outperforming the baseline.

### R-CFG Acceleration

| Denoising Steps | Self-Negative R-CFG | Onetime-Negative R-CFG | CFG |
|---------|--------------------|-----------------------|-----|
| 1 | 11.04 (1.52×) | 16.55 (1.01×) | 16.74 |
| 5 | 31.47 (**2.05×**) | 36.04 (1.79×) | 64.64 |

At 5 denoising steps, Self-Negative R-CFG is 2.05× faster than standard CFG.

### Energy Consumption

| GPU | Power w/o SSF (W) | Power w/ SSF (W) | Savings |
|-----|---------------|----------------|---------|
| RTX 3060 | 85.96 | 35.91 | 2.39× |
| RTX 4090 | 238.68 | 119.77 | 1.99× |

### Image Quality

StreamDiffusion outperforms the LCM baseline on FID (26.79 vs. 29.69) while maintaining comparable CLIP scores (24.99 vs. 24.95), demonstrating that acceleration does not sacrifice generation quality.

## Highlights & Insights

1. **Pipeline-level optimization** is orthogonal to model-level optimization and is compatible with any accelerated model (LCM, TurboSD).
2. **Generality of Stream Batch**: The approach generalizes to continuous generation tasks such as video, audio, and robotic action sequences.
3. **R-CFG analytically replaces** expensive unconditional UNet computations with near-zero overhead.
4. Probabilistic SSF produces smoother video streams than hard-threshold alternatives.

## Limitations & Future Work

- R-CFG is primarily designed for SDEdit-style image-to-image generation; its applicability to pure text-to-image tasks is limited.
- The latency introduced by Stream Batch scales linearly with the number of denoising steps ($n$ frames of delay).
- The SSF threshold $\eta$ requires manual tuning depending on the application scenario.
- The current framework is optimized primarily for single-GPU settings; gains from multi-GPU parallelism are limited.

## Related Work & Insights

- Efficient diffusion models: step-reduction methods such as DPM++, LCM, and InstaFlow.
- Model acceleration: quantization and TensorRT-based inference acceleration.
- Parallel sampling: ParaDiGMS (targets latency reduction, orthogonal to StreamDiffusion's focus on throughput).

## Rating

| Dimension | Score (1–5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| **Overall** | **4.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](../../CVPR2026/image_generation/streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)
- [\[ICCV 2025\] SDMatte: Grafting Diffusion Models for Interactive Matting](sdmatte_grafting_diffusion_models_for_interactive_matting.md)
- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[ICCV 2025\] GameFactory: Creating New Games with Generative Interactive Videos](gamefactory_creating_new_games_with_generative_interactive_videos.md)
- [\[NeurIPS 2025\] Real-Time Execution of Action Chunking Flow Policies](../../NeurIPS2025/image_generation/real-time_execution_of_action_chunking_flow_policies.md)

</div>

<!-- RELATED:END -->
