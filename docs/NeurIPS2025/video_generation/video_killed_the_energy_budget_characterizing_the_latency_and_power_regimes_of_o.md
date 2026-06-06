---
title: >-
  [Paper Note] Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models
description: >-
  [NeurIPS 2025][Video Generation][Text-to-video generation] This paper presents a systematic analysis of latency and energy consumption for open-source text-to-video (T2V) models. It establishes a FLOP-based analytical mo…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Text-to-video generation"
  - "energy consumption analysis"
  - "latency benchmarking"
  - "diffusion models"
  - "sustainable AI"
date: 2026-05-08
content_hash: d8a594038105f08c
---

# Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.19222](https://arxiv.org/abs/2509.19222)  
**Code**: [GitHub](https://github.com/anonymized)  
**Area**: Video Generation
**Keywords**: Text-to-video generation, energy consumption analysis, latency benchmarking, diffusion models, sustainable AI

## TL;DR

This paper presents a systematic analysis of latency and energy consumption for open-source text-to-video (T2V) models. It establishes a FLOP-based analytical model to predict scaling laws for WAN2.1 — quadratic scaling along spatial/temporal dimensions and linear scaling with respect to denoising steps — and provides a cross-model energy benchmark across 7 T2V models.

## Background & Motivation

1. **Background**: T2V generation has advanced rapidly, from closed-source systems such as Sora and Veo to open-source models including WAN2.1 and CogVideoX. These models are transitioning from research prototypes to production use cases, increasingly deployed in creative tools and video synthesis APIs.

2. **Limitations of Prior Work**: Most T2V evaluations focus exclusively on perceptual quality metrics (FID, motion smoothness, etc.) while neglecting latency and energy efficiency. Generating a few seconds of coherent video requires dozens of denoising steps, high resolution, and hundreds of frames, incurring substantial energy costs. Prior work is limited to a single model, Open-Sora, evaluated only on 2-second 240p videos.

3. **Key Challenge**: While T2V model quality continues to improve, the computational cost and environmental impact remain poorly understood — making it impossible to reason about principled quality-sustainability trade-offs.

4. **Goal**: To systematically quantify how latency and energy consumption of T2V models scale with key parameters: resolution, frame count, and number of denoising steps.

5. **Key Insight**: T2V inference is modeled as a compute-bound process. The paper decomposes per-operator FLOPs, derives closed-form scaling laws, and validates them empirically.

6. **Core Idea**: T2V inference is dominated by DiT self-attention — latency and energy grow quadratically with spatial/temporal dimensions and linearly with denoising steps, with cross-model energy differences reaching up to 3000×.

## Method

### Overall Architecture

The paper adopts a two-stage methodology:
1. **Theoretical Analysis**: Using WAN2.1-T2V-1.3B as a reference, inference FLOPs are decomposed and scaling laws are derived analytically.
2. **Empirical Validation**: Micro-benchmarks validate the scaling predictions, followed by cross-model comparisons across 7 T2V models.

### Key Designs

**1. Per-Operator FLOP Decomposition**

- **Function**: Precisely predict the computational cost of T2V inference.
- **Mechanism**: WAN2.1 inference is decomposed into five components:
    - **One-time**: Text encoder (T5), VAE decoder
    - **Per-step** (×$g \cdot S$): DiT self-attention ($N(8\ell d^2 + 4\ell^2 d)$), cross-attention ($N(4\ell d^2 + 4md^2 + 4\ell md)$), MLP ($N(4f\ell d^2)$), timestep MLP

  The DiT token length is $\ell = (1+T/4) \cdot H/16 \cdot W/16$, which grows linearly with resolution and frame count.
- **Design Motivation**: Understanding the FLOP composition of each operator enables cost prediction across configurations, thereby guiding efficient deployment.

**2. Compute-Bound Latency Model**

- **Function**: Predict actual latency and energy consumption from theoretical FLOPs.
- **Mechanism**: Profiling on H100 confirms that dominant operations (self-attention, MLP) are compute-bound rather than memory-bound. An empirical hardware utilization coefficient is defined as $\mu = F_{total}/(D_{measured} \cdot \Theta_{peak})$, estimated via regression as $\mu \approx 0.456$ ($R^2 = 0.998$). Latency and energy are then estimated as $D_{total} \approx F_{total}/(\mu \cdot \Theta_{peak})$ and $E_{total} \approx P_{max} \cdot D_{total}$.
- **Design Motivation**: The compute-bound assumption yields a linear proportionality between latency and FLOPs, greatly simplifying cost prediction.

**3. Scaling Law Derivation**

- **Function**: Characterize three distinct scaling regimes for T2V inference.
- **Mechanism**:
    - **Spatial dimensions $(H,W)$**: $\ell \propto HW$ → self-attention $\propto \ell^2 \propto (HW)^2$ → **quadratic scaling**
    - **Temporal dimension $T$**: $\ell \propto T$ → self-attention $\propto T^2$ → **quadratic scaling**
    - **Denoising steps $S$**: identical operations per step → **linear scaling**
    - Auxiliary components (text encoder, timestep MLP) contribute negligibly
- **Design Motivation**: Identifying which dimension is most computationally expensive is fundamental to optimizing deployment.

### Loss & Training

(This is a benchmarking study; no training is involved. Hardware: NVIDIA H100 SXM 80 GB. Each configuration is measured over 2 warmup runs and 5 timed runs. Energy consumption is measured via CodeCarbon.)

## Key Experimental Results

### Main Results

**Scaling Law Validation (WAN2.1-T2V-1.3B)**

| Scaling Dimension | Theoretical Prediction | Validation | Mean Energy Error | Mean Latency Error |
|---|---|---|---|---|
| Spatial resolution | Quadratic | ✓ Quadratic | 11.6% | 14.0% |
| Frame count | Quadratic | ✓ Quadratic | 6.6% | 10.5% |
| Denoising steps | Linear | ✓ Perfect linear | **1.9%** | **1.9%** |

**Cross-Model Energy Benchmark (Default Settings, Single Video Generation)**

| Model | Latency (s) | GPU Energy (Wh) | Resolution | Frames | Steps |
|---|---|---|---|---|---|
| AnimateDiff | **0.68** | **0.115** | 512² | 16 | 4 |
| LTX-Video | 9.7 | 3.16 | 512×704 | 121 | 40 |
| CogVideoX-2b | 50.6 | 8.3 | 480×720 | 49 | 50 |
| CogVideoX-5b | 124 | 21.6 | 480×720 | 49 | 50 |
| Mochi-1-preview | 263 | 44.7 | 480×848 | 84 | 64 |
| WAN2.1-1.3B | 410 | 78.8 | 720×1280 | 81 | 50 |
| WAN2.1-14B | **1875** | **359.7** | 720×1280 | 81 | 50 |

### Ablation Study

| Component | FLOP Share | Role |
|---|---|---|
| DiT self-attention | ~60–70% | **Dominant** |
| DiT MLP | ~20–25% | Secondary |
| DiT cross-attention | ~5–10% | Minor |
| VAE decoder | <5% | Negligible |
| Text encoder | One-time | Negligible |

### Key Findings

- **Up to 3000× cross-model gap**: AnimateDiff (0.14 Wh) vs. WAN2.1-14B (415 Wh)
- Primary drivers of energy and latency: model size > resolution > frame count > denoising steps
- GPU accounts for **80–90%** of total energy; CPU and RAM contributions are negligible
- Generating a single 5-second 720p video with WAN2.1-14B consumes 360 Wh of GPU energy — roughly equivalent to running a refrigerator for one hour
- Denoising steps represent the most cost-effective optimization dimension (linear scaling); step reduction strategies should be prioritized

## Highlights & Insights

- The first systematic energy benchmark for T2V models, filling a critical gap in sustainability evaluation for this domain
- The high agreement between the FLOP analytical model and measured values ($R^2 = 0.998$) validates the compute-bound assumption
- A clear optimization priority ordering is established: reducing steps (linear) is far more cost-efficient than reducing resolution or frame count (quadratic, but with greater quality impact)
- Cross-model comparisons reveal substantial efficiency gaps, providing practical guidance for model selection

## Limitations & Future Work

- Perceptual quality is not evaluated, precluding the construction of a quality-energy Pareto frontier
- Experiments are conducted only on a single H100 GPU; multi-GPU parallelism and alternative hardware remain unstudied
- The impact of inference optimization techniques (e.g., diffusion caching, quantization, distillation) on energy consumption is not analyzed
- The utilization coefficient $\mu$ is empirically calibrated as a fixed scalar and may vary across models
- Coverage is limited to UNet/DiT-based architectures; other paradigms such as autoregressive video generation are not analyzed

## Related Work & Insights

- This work extends the tradition of ML energy consumption research established by Strubell et al. (2019) and Luccioni et al. (2024)
- Li et al. (2024)'s preliminary study on Open-Sora serves as a direct predecessor; the present paper substantially expands the coverage of models and configurations
- The findings suggest that as T2V models are deployed at scale, energy efficiency should be established as a standard dimension of model evaluation

## Rating

- **Novelty**: ⭐⭐⭐ The methodology (FLOP analysis + benchmarking) is not novel in itself, but the systematic scope and coverage are unprecedented in this domain
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Theory-experiment correspondence is thorough, with three-dimensional scaling validation and cross-model comparison across 7 models
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with detailed theoretical derivations and highly informative tables and figures
- **Value**: ⭐⭐⭐⭐ Provides important reference data and optimization guidelines for sustainable T2V deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[NeurIPS 2025\] DisMo: Disentangled Motion Representations for Open-World Motion Transfer](dismo_disentangled_motion_representations_for_openworld_moti.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](../../ICCV2025/video_generation/efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[NeurIPS 2025\] Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)

</div>

<!-- RELATED:END -->
