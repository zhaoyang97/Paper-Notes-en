---
title: >-
  [Paper Note] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout
description: >-
  [CVPR 2026][Video Generation][Autoregressive video generation] This paper proposes ∞-RoPE, a training-free inference-time framework comprising three components — Block-Relativistic RoPE, KV Flush…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Autoregressive video generation"
  - "positional encoding"
  - "infinite-length video"
  - "action control"
  - "inference-time method"
date: 2026-05-08
content_hash: 5b863b11e6d9bc6e
---

# Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout

**Conference**: CVPR 2026
**arXiv**: [2511.20649](https://arxiv.org/abs/2511.20649)
**Code**: [Project Page](https://infinity-rope.github.io)
**Area**: Video Generation / Diffusion Models
**Keywords**: Autoregressive video generation, positional encoding, infinite-length video, action control, inference-time method

## TL;DR

This paper proposes ∞-RoPE, a training-free inference-time framework comprising three components — Block-Relativistic RoPE, KV Flush, and RoPE Cut — that extends an autoregressive video diffusion model trained solely on 5-second clips to support infinite-length generation, fine-grained action control, and cinematic scene transitions.

## Background & Motivation

Current autoregressive video diffusion models face three core bottlenecks:

**Limited temporal horizon**: 3D-RoPE positional encoding constrains generation to a fixed window of 1024 frames, beyond which attention quality degrades sharply.

**Sluggish action responsiveness**: During long-sequence rollout, prompt changes cannot take effect immediately, as stale semantics in the KV cache continue to influence generation.

**Lack of scene-jump capability**: Single-stream generation cannot support cinematic discontinuous scene transitions.

**Key Insight**: A model trained under the Self-Forcing paradigm on only 5-second clips already possesses the capacity for highly dynamic infinite-length generation — the bottleneck lies not in model capacity but in the absolute indexing mechanism of positional encoding. The authors propose to overcome this through relativistic positional encoding reparameterization and KV cache management, requiring no additional training.

## Method

### Overall Architecture

∞-RoPE builds upon the Wan2.1-T2V-1.3B model distilled via Self-Forcing (a 4-step causal generator), and introduces three interconnected components at inference time:
- **Block-Relativistic RoPE**: Relativistic temporal positional encoding to break the fixed-frame limit.
- **KV Flush**: KV cache reset mechanism for instant prompt responsiveness.
- **RoPE Cut**: Controlled discontinuous jumps in temporal coordinates for multi-shot scene switching.

### Key Designs

1. **Block-Relativistic RoPE (Core)**

   Autoregressive generation proceeds in blocks of 3 frames: $\mathbf{B}_f = \{f-2, f-1, f\}$. In conventional absolute RoPE, positions $i \gg f_{\text{limit}}$ fall outside the training distribution, causing failure. Block-Relativistic RoPE defines temporal coordinates as a **moving local reference frame**:

   $$\tilde{\mathbf{B}}_i = \begin{cases} \mathbf{B}_i, & \text{if } i \leq f_0 \\ \mathbf{B}_{f_0} = \{f_0-2, f_0-1, f_0\}, & \text{otherwise} \end{cases}$$

   When a new block is generated, its RoPE indices are always rotated within the model's maximum frame range $f_{\text{limit}}$, while the temporal phases of earlier blocks are counter-rotated to preserve relative temporal geometry. **Design Motivation**: Analogous to *semanticization* in cognitive neuroscience, where remote memories lose precise temporal tags but retain semantic content — the temporal coordinates of the earliest cached frames collapse to a shared minimum index $\mathbf{B}_{\bar{1}} = \{1,1,1\}$.

2. **KV Flush (Action Control)**

   Upon a prompt change, all KV cache entries are cleared, retaining only two anchors: a **global sink frame** (stabilizing attention normalization) and the **last generated frame** (maintaining local temporal continuity). New actions are conditioned directly on these minimal anchors, achieving zero-latency prompt response. Compared to no-cache (abrupt changes), full-cache (semantic lag), and KV re-cache (high latency), KV Flush achieves superior efficiency and controllability.

3. **RoPE Cut (Scene Switching)**

   Cinematic multi-shot transitions are achieved by introducing controlled discontinuous jumps in the temporal RoPE coordinates. For the current block $\mathbf{B}_f = \{f-2, f-1, f\}$, the coordinates are remapped as:

   $$\mathbf{B}_{f \to f+\Delta} = \{f-2, f+\Delta-1, f+\Delta\}$$

   Post-jump frames are treated as past context, and generation restarts from a new origin in temporal position space. Since no absolute positions exist in the relativistic formulation, the coordinate frame shifts automatically at each cut, preserving identity consistency even across large temporal or semantic jumps.

### Loss & Training

∞-RoPE is a **purely inference-time method** and involves no additional training. The underlying Self-Forcing model is trained under the Rectified Flow formulation: $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$, with the inverse process solved via an ODE parameterized by a neural velocity field $v_\theta$. Experiments fix the KV cache size to 6, onset index $f_0=21$, CFG scale 3.0, and timestep shift 5.0.

## Key Experimental Results

### Main Results

VBench evaluation on 5-second and 60-second video generation (60-second results shown):

| Model | Background Consistency | Dynamic Degree | Subject Consistency | Overall |
|-------|----------------------|---------------|-------------------|---------|
| NOVA | 0.8806 | 0.12 | 0.7750 | 0.6901 |
| SkyReels-V2 | 0.8995 | 0.44 | 0.8499 | 0.7768 |
| CausVid | 0.8985 | 0.52 | 0.8675 | 0.7940 |
| Self-Forcing | 0.8784 | 0.32 | 0.8360 | 0.7715 |
| Rolling-Forcing | 0.9447 | 0.36 | 0.9409 | 0.8146 |
| **∞-RoPE** | **0.9490** | **0.52** | **0.9444** | **0.8298** |

Ultra-long video generation at 120 seconds and 240 seconds (240-second results shown):

| Model | Background Consistency | Dynamic Degree | Subject Consistency | Overall |
|-------|----------------------|---------------|-------------------|---------|
| Rolling-Forcing | 0.9248 | 0.40 | 0.9080 | 0.8017 |
| **∞-RoPE** | **0.9361** | **0.64** | **0.9256** | **0.8309** |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| Block-Relativistic RoPE on vs. off | Self-Forcing alone cannot sustain dynamic long video | 5s-trained model + BRRoPE suffices for high-quality 30s+ generation |
| KV cache size sweep | Overall / Aesthetic / Dynamic vary with cache size | Cache size 6 achieves the best balance across all durations |
| KV Flush vs. no-cache / full-cache / re-cache | Instant semantic response + smooth motion continuity | KV Flush consistently outperforms alternatives in efficiency and controllability |

### Key Findings

- ∞-RoPE achieves the highest (or tied highest) Overall score across all durations (5s / 60s / 120s / 240s).
- Key advantages lie in **Subject Consistency** and **Background Consistency**, with the margin growing larger in ultra-long video settings.
- Dynamic Degree reaches **0.64** at 240s, far surpassing other methods (most in the range 0.24–0.40), demonstrating that long-horizon generation does not degrade into near-static output.

## Highlights & Insights

- **Cognitive science-inspired design**: Collapsing the temporal coordinates of distant frames into a form of "semantic memory" draws an analogy to the semanticization process in human long-term memory.
- **Interpretable attention maps**: Attention map visualizations clearly illustrate the distinct structural signatures of BRRoPE (diagonal band + sink column), KV Flush (severed intermediate history), and RoPE Cut (split into two independent diagonal blocks).
- **Zero training overhead**: As a purely inference-time method, ∞-RoPE can be seamlessly plugged into any Self-Forcing variant.

## Limitations & Future Work

- Performance is upper-bounded by the quality of the underlying Self-Forcing distilled model.
- Semantic coherence across scene cuts relies on global information retained in the sink frame, which may be insufficient for complex scenes.
- Validation is limited to a 1.3B parameter model; behavior at the 14B scale remains unexplored.

## Related Work & Insights

- **Self-Forcing / Self-Forcing++**: Provide the autoregressive rollout training paradigm upon which ∞-RoPE achieves inference-time improvements.
- **Rolling Forcing**: The primary competing approach using a progressive noise window, but still constrained by the RoPE range.
- **FLEX**: A subsequent work introducing frequency-aware RoPE modulation, complementary to the present approach.

## Rating

- **Novelty**: ★★★★☆ — The relativistic reparameterization of positional encoding is elegant, and the cognitive science analogy is insightful.
- **Technical Depth**: ★★★★☆ — The three components are well-designed, mutually reinforcing, and accompanied by thorough mechanistic analysis.
- **Experimental Thoroughness**: ★★★★☆ — Comprehensive VBench evaluation across multiple durations, though a user study is absent.
- **Practicality**: ★★★★★ — Training-free and plug-and-play, with strong potential for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](../../NeurIPS2025/video_generation/self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_fewstep_controllable_video_generation.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [\[CVPR 2026\] From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning](from_static_to_dynamic_exploring_self-supervised_image-to-video_representation_t.md)

</div>

<!-- RELATED:END -->
