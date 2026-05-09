---
title: >-
  [Paper Note] MotionStream: Real-Time Video Generation with Interactive Motion Controls
description: >-
  [ICLR 2026][Video Generation][streaming video generation] MotionStream is proposed as the first real-time streaming video generation system with motion control. It first trains a bidirectional motion-control teacher with a lightweight track head on Wan DiT, then distills it into a causal student via Self Forcing + DMD. Attention sink and rolling KV cache are introduced to achieve full train-inference distribution matching, enabling infinite-length generation at constant speed — reaching 17 FPS / 29 FPS (+ Tiny VAE) at 480P on a single H100 GPU.
tags:
  - ICLR 2026
  - Video Generation
  - streaming video generation
  - motion control
  - causal distillation
  - attention sink
  - distribution matching distillation
  - real-time interaction
date: 2026-05-08
content_hash: 87352ade420860c1
---

# MotionStream: Real-Time Video Generation with Interactive Motion Controls

**Conference**: ICLR 2026
**arXiv**: [2511.01266](https://arxiv.org/abs/2511.01266)
**Code**: None
**Area**: Video Generation
**Keywords**: streaming video generation, motion control, causal distillation, attention sink, distribution matching distillation, real-time interaction

## TL;DR
MotionStream is proposed as the first real-time streaming video generation system with motion control. It first trains a bidirectional motion-control teacher with a lightweight track head on Wan DiT, then distills it into a causal student via Self Forcing + DMD. Attention sink and rolling KV cache are introduced to achieve full train-inference distribution matching, enabling infinite-length generation at constant speed — reaching 17 FPS / 29 FPS (+ Tiny VAE) at 480P on a single H100 GPU.

## Background & Motivation

**Background**: Motion-controlled video generation (e.g., Motion Prompting) can produce high-quality trajectory-tracking videos, but inference is extremely slow (12 minutes for a 5-second video), non-causal (requiring complete control signals upfront), and limited to fixed-length outputs.

**Limitations of Prior Work**:
- Bidirectional attention in diffusion models requires all future trajectories before generation can begin, precluding real-time interaction.
- Causal distillation methods such as CausVid suffer severe out-of-distribution drift beyond the training length (>81 frames) — manifesting as color shift and quality degradation.
- ControlNet-style architectures double the FLOPs, further slowing inference.
- Unbounded RoPE position growth in sliding-window attention leads to high latency variance and throughput instability.

**Key Challenge**: Interactive creative workflows demand "real-time + causal + infinite-length" generation, which is fundamentally at odds with the "slow + bidirectional + fixed-length" paradigm of diffusion models.

**Goal**: Transform motion-controlled video generation from a "render-and-wait" mode to a "real-time creation" mode, where users see results instantly as they draw trajectories.

**Key Insight**: Simultaneously address three dimensions — (1) lightweight teacher architecture to reduce baseline overhead; (2) joint guidance embedding distillation to eliminate multi-NFE costs; (3) attention sink plus training-time inference distribution simulation to eliminate long-video drift.

**Core Idea**: Realize real-time infinite streaming video generation with motion control through a pipeline of "efficient teacher → causal distillation → attention-sink extrapolation training."

## Method

### Overall Architecture

A two-stage pipeline: **Stage 1** trains a bidirectional motion-control teacher by adding a lightweight track head to Wan DiT; **Stage 2** obtains a causal student via causal adaptation and Self Forcing-style DMD distillation, with attention sink and rolling KV cache used during training to simulate the inference-time distribution.

### Key Designs

1. **Lightweight Track Head with Sinusoidal Trajectory Encoding**:
    - **Function**: Efficiently encode 2D trajectories as motion conditions, avoiding the FLOPs doubling of ControlNet.
    - **Mechanism**: Each trajectory is assigned a unique $d$-dimensional sinusoidal positional encoding $\phi_n$, placed at the corresponding spatial location in the input: $c_m[t, \lfloor y_t^n/s \rfloor, \lfloor x_t^n/s \rfloor] = v[t,n] \cdot \phi_n$. After 4× temporal compression and a $1\times1\times1$ convolution, the result is channel-concatenated with the video latent, modifying only the patchify layer input channels of the DiT.
    - **Design Motivation**: 40× faster than RGB-VAE encoding (24.8 ms vs. 1053 ms) with better trajectory tracking (EPE: 6.54 vs. 8.57) — sinusoidal encoding provides richer identity signals than RGB.

2. **Joint Text-Motion Guidance Embedding Distillation**:
    - **Function**: "Bake" the teacher's 3× NFE joint guidance cost into the student's single NFE.
    - **Mechanism**: The teacher applies joint guidance $\hat{v} = v_{\text{base}} + w_t(v(c_t,c_m) - v(\emptyset,c_m)) + w_m(v(c_t,c_m) - v(c_t,\emptyset))$, with $w_t=3.0, w_m=1.5$. During distillation, this joint-guided output is defined as $s_{\text{real}}$ for DMD, while $s_{\text{fake}}$ uses no CFG (only $f_\psi(c_t,c_m)$), enabling the student to reproduce the teacher's joint guidance quality in a single forward pass.
    - **Design Motivation**: Pure motion guidance produces rigid 2D translational motion; text guidance supplies natural secondary motion (e.g., a rainbow appearing in the background as an elephant moves). The two are complementary, and distillation incurs no additional inference cost.

3. **Extrapolation Training with Attention Sink and Rolling KV Cache**:
    - **Function**: Achieve constant-speed inference and drift-free generation for infinite-length video.
    - **Mechanism**: A fixed-size KV cache is maintained, consisting of $S$ sink chunks (initial frames) and $W$ local window chunks. As new tokens are generated, the window scrolls to maintain constant size. The key innovation is that **the same attention sink and rolling KV cache are used during self-rollout at training time**, with RoPE positions assigned according to cache positions rather than absolute time, completely eliminating the train-test distribution gap. At inference, latency and throughput remain constant regardless of video length.
    - **Design Motivation**: Attention analysis (Figure 3) reveals that many heads persistently attend to initial-frame tokens — analogous to the findings of StreamingLLM. Retaining initial frames as a global anchor prevents color and content drift. The optimal configuration c3s1w1 (chunk=3, sink=1, window=1) shows that a larger window actually degrades quality, as attending to long-past history causes errors to accumulate in the context.

### Loss & Training

Teacher training uses a flow matching loss $\mathcal{L}_{\text{FM}} = \mathbb{E}_{z_0,z_1,t}[w_t \| v_\theta(z_{t'},t',c_t,c_m) - (z_1-z_0) \|^2]$, in two stages (OpenVid-1M for 4,800 steps → synthetic fine-tuning for 800 steps). Causal adaptation uses 4,000 ODE trajectories generated by the teacher for regression over 2,000 steps. Self Forcing DMD distillation uses a generator-to-critic update ratio of 1:5, with gradients truncated to a randomly sampled single denoising step, converging in only ~400 steps. Total training: ~3 days on 32× A100 (teacher) + 20 hours (distillation).

## Key Experimental Results

### Motion Transfer — Reconstruction Quality

| Method | Backbone | FPS | PSNR↑ | LPIPS↓ | EPE↓ |
|------|----------|-----|-------|--------|------|
| Go-With-The-Flow | CogVideoX-5B | 0.60 | 15.62 | 0.490 | 41.99 |
| Diffusion-As-Shader | CogVideoX-5B | 0.29 | 15.80 | 0.483 | 40.23 |
| ATI | Wan 2.1-14B | 0.23 | 15.33 | 0.473 | 17.41 |
| **MotionStream Teacher** | Wan 2.1-1.3B | 0.79 | **16.61** | **0.427** | **5.35** |
| **MotionStream Causal** | Wan 2.1-1.3B | **16.7** | 16.20 | 0.443 | 7.80 |

### Novel View Synthesis (LLFF Dataset)

| Method | Resolution | FPS | PSNR↑ | LPIPS↓ |
|------|--------|-----|-------|--------|
| DepthSplat | 576P | 1.40 | 13.9 | 0.30 |
| ViewCrafter | 576P | 0.26 | 14.0 | 0.30 |
| SEVA | 576P | 0.20 | 14.1 | 0.29 |
| **MotionStream Teacher** | 480P | 0.79 | **16.0** | **0.21** |
| **MotionStream Causal** | 480P | **16.7** | 15.7 | 0.23 |

### Ablation Study — Attention Configuration

| Configuration | LPIPS↓ | EPE↓ | Latency Variance | Throughput |
|------|--------|------|---------|--------|
| c3s1w1 (standard) | **0.464** | **25.34** | 0.70±0.01 | 16.92±0.80 |
| c3s0w1 (no sink) | 0.501 | 26.64 | 0.68±0.005 | 17.43±0.88 |
| c1s1w1 (chunk=1) | 0.597 | 76.21 | 0.30±0.01 | 13.26±1.36 |
| Sliding window | 0.480 | 28.09 | 0.80±**0.08** | 14.96±**1.42** |

### Key Findings
- MotionStream Causal is 20–70× faster than all baselines while achieving state-of-the-art motion tracking metrics on DAVIS/Sora.
- Zero-shot novel view synthesis (3D camera control) surpasses dedicated 3D methods (DepthSplat/ViewCrafter/SEVA): PSNR +1.6, LPIPS −0.07.
- The attention sink is critical: removing the sink chunk degrades LPIPS from 0.464 to 0.501, with visible color drift in long video generation (Figure A3).
- Counterintuitively, a larger attention window degrades quality — attending to long-past history allows errors to accumulate in the context.
- The sliding window approach exhibits latency variance of ±0.08 s (vs. ±0.01 s for c3s1w1) due to unbounded RoPE positions causing computational instability.
- Tiny VAE improves FPS from 16.7 to 29.5 and reduces latency from 0.69 s to 0.39 s with negligible quality loss (PSNR: 16.67 → 16.68).

## Highlights & Insights
- **Paradigm shift from "render-and-wait" to "real-time creation"**: A two-orders-of-magnitude speedup (minutes → sub-second) brings motion-controlled video generation to the speed threshold required for interactive creative workflows for the first time.
- **Cross-domain transfer of attention sink**: The "initial tokens attract disproportionate attention" phenomenon observed in StreamingLLM is successfully transferred to video diffusion models — initial frames serve as anchors to prevent content/color drift during infinite-length generation.
- **Simulating inference distribution at training time**: The key distinction from methods such as TalkingMachines — using the same rolling KV cache and attention sink during self-rollout as at inference time eliminates train-test mismatch, which is the fundamental guarantee of long-video stability.
- **Complementarity of joint guidance**: Pure trajectory guidance → rigid 2D translation; pure text guidance → poor trajectory adherence; joint guidance with $w_t=3.0, w_m=1.5$ → natural motion + precise tracking.

## Limitations & Future Work
- The fixed attention sink anchors to initial frames, making the approach unsuitable for applications requiring complete scene transitions (e.g., open-world game exploration), which would necessitate dynamic anchor refresh.
- Extremely fast or physically implausible trajectories lead to temporal inconsistencies or appearance distortion.
- Wan 2.1 (1.3B) better preserves source structure than Wan 2.2 (5B) — a larger backbone does not necessarily yield more robustness.
- Trajectory disappearance: when users release control, the model cannot distinguish between occlusion and "unspecified" (both represented as zero values); mid-frame masking only partially alleviates this.

## Related Work & Insights
- **vs. Motion Prompting**: Both use 2D trajectories for control, but Motion Prompting is an offline bidirectional diffusion approach (12 min/5 s), whereas MotionStream is real-time causal streaming (29 FPS).
- **vs. Self Forcing (Huang et al.)**: Self Forcing introduces the causal distillation framework but uses an unbounded sliding window, leading to latency variance and long-video drift; MotionStream addresses both issues via attention sink and extrapolation training.
- **vs. TalkingMachines**: Also employs attention sink, but synchronized denoising with causal masking does not fully simulate autoregressive inference; temporal discontinuities between sink frames and subsequent frames also impair teacher scoring accuracy.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First real-time streaming video generation with motion control; multiple system-level innovations working in concert.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of motion transfer, camera control, user drag interaction, multi-resolution settings, and ablations.
- **Writing Quality**: ⭐⭐⭐⭐ System design is presented with clear hierarchical structure; ablation experiments are well-designed, particularly the attention configuration analysis.
- **Value**: ⭐⭐⭐⭐⭐ Significant contributions to both the engineering implementation and academic understanding of interactive video creation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[CVPR 2026\] StreamDiT: Real-Time Streaming Text-to-Video Generation](../../CVPR2026/video_generation/streamdit_real-time_streaming_text-to-video_generation.md)
- [\[ICLR 2026\] TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)
- [\[CVPR 2026\] U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](../../CVPR2026/video_generation/u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)
- [\[ICLR 2026\] MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling](mosa_motion-coherent_human_video_generation_via_structure-appearance_decoupling.md)

<!-- RELATED:END -->
