---
title: >-
  [Paper Note] TTOM: Test-Time Optimization and Memorization for Compositional Video Generation
description: >-
  [ICLR 2026][Video Generation][Test-time optimization] This paper proposes TTOM, a framework that aligns attention maps of video generation models with LLM-generated spatiotemporal layouts by optimizing newly introduced parameters at inference time, while a parameter memorization mechanism stores historical optimization contexts for reuse. TTOM achieves relative improvements of 34% (CogVideoX) and 14% (Wan2.1) on T2V-CompBench.
tags:
  - ICLR 2026
  - Video Generation
  - Test-time optimization
  - compositional video generation
  - parameter memorization
  - spatiotemporal layout
  - attention alignment
date: 2026-05-08
content_hash: bc7f5bff0c682dcf
---

# TTOM: Test-Time Optimization and Memorization for Compositional Video Generation

**Conference**: ICLR 2026
**arXiv**: [2510.07940](https://arxiv.org/abs/2510.07940)
**Code**: [https://ttom-t2v.github.io/](https://ttom-t2v.github.io/)
**Area**: Video Generation / Compositional Reasoning
**Keywords**: Test-time optimization, compositional video generation, parameter memorization, spatiotemporal layout, attention alignment

## TL;DR
This paper proposes TTOM, a framework that aligns attention maps of video generation models with LLM-generated spatiotemporal layouts by optimizing newly introduced parameters at inference time, while a parameter memorization mechanism stores historical optimization contexts for reuse. TTOM achieves relative improvements of 34% (CogVideoX) and 14% (Wan2.1) on T2V-CompBench.

## Background & Motivation

**Background**: Text-to-video (T2V) models perform well on single-object scenarios but suffer from severe misalignment in compositional scenes involving multiple objects, attributes, motions, and spatial relationships. Existing methods leverage LLMs to generate spatiotemporal layouts and guide generation by modifying latent variables or attention maps.

**Limitations of Prior Work**: (a) Direct manipulation of latent variables or attention maps disrupts feature distributions, causing flickering and collapse; (b) each sample is processed independently without leveraging historical context; (c) interventions optimized for one sample do not generalize to others.

**Key Challenge**: Fine-grained control over compositional layouts is required, yet such control must not corrupt the feature distribution of the pre-trained model.

**Goal**: Achieve model-agnostic compositional layout alignment at test time while reusing historical optimization results.

**Key Insight**: Rather than modifying latent variables, the approach inserts and optimizes new parameters to align attention with the target layout, then stores the optimized parameters in memory for future reuse.

**Core Idea**: Optimize parameters instead of latent variables to achieve layout alignment, and accumulate cross-sample knowledge via parameter memorization.

## Method

### Overall Architecture
Streaming setting: users submit prompts sequentially. (1) An LLM generates a spatiotemporal layout (bounding box sequences per object). (2) The memory is queried for a matching entry — if found, parameters are loaded (with optional continued optimization); otherwise, new parameters are initialized. (3) Test-time optimization (TTO) aligns attention maps with the layout. (4) Optimized parameters are stored in memory.

### Key Designs

1. **Attention–Layout Correlation Probing**:

    - Function: Identifies which DiT layers have attention maps most correlated with the final video layout.
    - Mechanism: Generate a video → segment with GroundingDINO + SAM2 → compute mIoU between per-layer attention maps and segmentation masks. Results reveal large variance in correlation across layers.
    - Design Motivation: Only layers with high correlation are targeted for optimization, avoiding unnecessary interference.

2. **Test-Time Optimization (TTO)**:

    - Function: Inserts new parameters and optimizes them to align attention with the layout.
    - Mechanism: Lightweight parameters $\phi$ are inserted into the video foundation model (VFM) and optimized via a JSD alignment loss $L_{align} = \frac{1}{N}\sum_i JSD(\bar{A}_i \| \bar{B}_i)$ that aligns attention maps with Gaussian-smoothed layout masks.
    - Design Motivation: Optimizing $\phi$ rather than latent variables $z_t$ avoids distribution collapse.

3. **Parameter Memorization**:

    - Function: Stores historical optimization contexts for future reuse.
    - Mechanism: Memory $\mathcal{M} = \{g(C): \phi^*_C\}$, where keys are text embeddings of abstracted scene descriptions. Supports insert/read/update/delete operations; LFU eviction is applied when capacity is exceeded.
    - Design Motivation: Parameters from similar scenes can be loaded directly to skip optimization (efficiency), or used as strong initializations (quality).

### Loss & Training

The method is unsupervised — only $L_{align}$ (JSD) is used to optimize the newly inserted parameters at test time. LLM-generated layouts include a validation step to ensure consistency. When a memory match is found, optimization can be skipped entirely for direct inference.

## Key Experimental Results

### Main Results

T2V-CompBench (7 compositional video generation categories):

| Model | Avg. Score | Motion | Numeracy | Spatial |
|-------|-----------|--------|----------|---------|
| CogVideoX-5B | baseline | low | low | low |
| CogVideoX + TTOM | **+34%** | significant gain | significant gain | significant gain |
| Wan2.1-14B | baseline | medium | medium | medium |
| Wan2.1 + TTOM | **+14%** | gain | gain | gain |

Consistent improvements are also observed on VBench.

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| Optimize latents vs. optimize parameters | Parameter optimization yields higher quality without collapse |
| With memory vs. without memory | Memory significantly improves both efficiency and quality |
| Layer selection | Optimizing only high-correlation layers achieves the best results |
| Skip optimization on memory hit | Large efficiency gains with only marginal quality degradation |
| Transferability | Parameters optimized for one scene transfer effectively to similar scenes |

### Key Findings
- TTOM decouples compositional world knowledge — optimized parameters exhibit strong transferability and generalization.
- Parameter memorization enables progressive improvement in streaming inference, as accumulated compositional patterns can be reused by new scenes.
- The approach is model-agnostic, demonstrating effectiveness on both CogVideoX and Wan2.1 with distinct architectures.
- JSD loss is more stable than direct $L_2$ loss.

## Highlights & Insights
- **Optimizing parameters rather than latents**: Avoids feature distribution corruption caused by direct intervention, offering a more principled alternative to latent guidance.
- **"Better with use" property of parameter memory**: Transforms test-time optimization from a one-shot cost into cumulative knowledge accumulation, conceptually analogous to human experiential learning.
- **Forward-looking streaming setting**: Frames video generation as a continuous service rather than isolated requests, better reflecting real-world deployment scenarios.
- **Attention–layout correlation probing**: Provides the first systematic quantification of the correspondence between per-layer DiT attention maps and final video layouts, offering independent analytical value.

## Limitations & Future Work
- TTO requires additional optimization steps, resulting in slower cold-start inference.
- Inaccurate LLM-generated spatiotemporal layouts propagate errors to the generated output.
- Scene abstraction for memory keys (e.g., "<object A> drifts above <object B>") may be overly coarse.
- Validation is limited to T2V; extension to image generation or 3D scenes remains unexplored.
- Memory capacity management and the LFU eviction strategy may be suboptimal.

## Related Work & Insights
- **vs. LLM-grounded Diffusion (Lian et al., 2023b)**: Prior work optimizes latent variables, leading to quality degradation. TTOM optimizes newly inserted parameters to avoid this issue.
- **vs. TTT layers (Sun et al., 2024)**: TTT memory operates within a sample (across frames), whereas TTOM memory operates across samples.
- **vs. Attend-and-Excite**: The latter controls attention at the image level. TTOM extends attention control to spatiotemporal attention in video.
- **Implications for video generation**: The paradigm of parameter-level control combined with cross-sample memorization is broadly applicable to other generative control settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of TTO and parameter memorization is highly novel; the streaming setting is forward-looking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, multiple VFMs, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is built up progressively; the framework design is elegant.
- Value: ⭐⭐⭐⭐⭐ Constitutes a substantial advance in compositional video generation; the parameter memorization paradigm has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Training-free Motion Factorization for Compositional Video Generation](../../CVPR2026/video_generation/training-free_motion_factorization_for_compositional_video_generation.md)
- [\[ICLR 2026\] MotionStream: Real-Time Video Generation with Interactive Motion Controls](motionstream_real-time_video_generation_with_interactive_motion_controls.md)
- [\[ICLR 2026\] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)
- [\[AAAI 2026\] DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation](../../AAAI2026/video_generation/dreamrunner_fine-grained_compositional_story-to-video_genera.md)
- [\[CVPR 2026\] StreamDiT: Real-Time Streaming Text-to-Video Generation](../../CVPR2026/video_generation/streamdit_real-time_streaming_text-to-video_generation.md)

</div>

<!-- RELATED:END -->
