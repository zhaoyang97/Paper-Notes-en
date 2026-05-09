---
title: >-
  [Paper Note] Cinematic Audio Source Separation Using Visual Cues
description: >-
  [CVPR 2026][Image Generation][Cinematic audio source separation] This paper proposes the first audio-visual cinematic audio source separation (AV-CASS) framework, which leverages visual cues from dual video streams (face and scene) to perform generative three-way audio separation (dialogue/effects/music) via conditional flow matching, training solely on synthetic data while generalizing to real films.
tags:
  - CVPR 2026
  - Image Generation
  - Cinematic audio source separation
  - audio-visual learning
  - conditional flow matching
  - synthetic training data
  - multi-source separation
date: 2026-05-08
content_hash: 8d75e3db254b2848
---

# Cinematic Audio Source Separation Using Visual Cues

**Conference**: CVPR 2026
**arXiv**: [2603.26113](https://arxiv.org/abs/2603.26113)
**Code**: [Project Page](https://cass-flowmatching.github.io)
**Area**: Image Generation (Audio-Visual Multimodal)
**Keywords**: Cinematic audio source separation, audio-visual learning, conditional flow matching, synthetic training data, multi-source separation

## TL;DR

This paper proposes the first audio-visual cinematic audio source separation (AV-CASS) framework, which leverages visual cues from dual video streams (face and scene) to perform generative three-way audio separation (dialogue/effects/music) via conditional flow matching, training solely on synthetic data while generalizing to real films.

## Background & Motivation

**Background**: Cinematic audio source separation (CASS) was formalized as a three-way separation problem (dialogue/effects/music) with the introduction of the DnR dataset. Methods such as BandIt have advanced audio-only performance, yet all existing approaches overlook the multimodal nature of cinema.

**Limitations of Prior Work**: (a) All CASS methods are purely audio-based, ignoring visual cues (lip motion corresponding to speech, scene actions corresponding to sound effects); (b) no dataset exists that simultaneously provides source-separated audio tracks and temporally aligned video; (c) predictive separation models are prone to spectral hole artifacts.

**Key Challenge**: Visual information clearly benefits audio separation, yet obtaining isolated tracks from real films is practically infeasible.

**Goal**: To train an effective AV-CASS model using independently obtainable in-the-wild audio-visual data, without access to real isolated tracks.

**Key Insight**: A synthetic training data pipeline (face video → speech, scene video → sound effects, music only) combined with a generative flow matching separation model.

**Core Idea**: Training with dual video streams (face + scene) from independent sources; at inference, both streams are extracted from a single real film, enabling zero-shot generalization.

## Method

### Overall Architecture

A Vision Extractor (face encoder + scene encoder → fused visual condition $\mathbf{c}^V$) combined with a Flow Matching generative model (from noise to three-way spectrograms, conditioned on the mixture audio $\mathbf{s}^A$ and $\mathbf{c}^V$).

### Key Designs

1. **Synthetic Training Data Pipeline**:

   - Dialogue (DX): LRS3 dataset (lip-sync video + speech), 152K clips
   - Sound Effects (FX): VGGSound (everyday event video + audio), filtered via SMAD to remove clips containing speech/music, ~62K
   - Music (MX): FMA (music only), filtered to ~49K
   - Mixture: $\mathbf{a}^A = \mathbf{a}^{DX} + \mathbf{a}^{FX} + \mathbf{a}^{MX}$

   Design Motivation: Real cinematic source separation data is unavailable, but single-source audio-visual data is abundant. Synthetic mixing preserves complete ground truth and remains fully controllable.

2. **Dual-Stream Visual Encoder and Fusion**: A face encoder (AVDiffuSS) extracts lip-sync features and a scene encoder (CAVP) extracts temporally-semantically aligned features. Both are frozen and projected, then concatenated along the time axis: $\mathbf{c}^V \in \mathbb{R}^{(T_f+T_s) \times C'}$, injected into a U-Net via cross-attention.

   Design Motivation: The face stream serves as a speech cue and the scene stream as a sound-effect cue, providing complementary coverage of the two visually grounded sources in CASS.

3. **Conditional Flow Matching for Multi-Source Separation**:

   $$\mathcal{L} = \mathbb{E}_{t, \pi_1, \pi_0} \|\mathbf{u}_\theta(\mathbf{x}_t, t | \mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0)\|_2^2$$

   Logit-normal timestep sampling; three-way spectrograms are concatenated along the channel dimension.

   Design Motivation: Flow matching requires fewer inference steps than diffusion and produces more natural audio than masking-based methods.

### Loss & Training

- Audio denoising warm-up → zero-initialized convolution for progressive visual conditioning (ControlNet-style)
- Adam optimizer, LR 1e-4, 600K steps, 4× RTX 4090; 128-step inference

## Key Experimental Results

### Main Results

| Method | Real Film MOS↑ | AVDnR FAD↓ | AVDnR PESQ↑ | AVDnR WPR↓ |
|---|---|---|---|---|
| MRX | 2.55 | 3.47 | 1.89 | 14.91 |
| BandIt | 3.78 | 2.15 | 2.15 | 4.65 |
| DAVIS-Flow (AV) | — | 5.94 | 1.96 | 12.14 |
| **AV-CASS** | **4.13** | **0.84** | **2.26** | **1.84** |

### Ablation Study

| Configuration | FAD↓ | WPR↓ | Note |
|---|---|---|---|
| Audio-only | 1.63 | 2.01 | Pure audio baseline |
| AV-CASS | **0.84** | **1.84** | Visual conditioning improves FAD by 48% |
| DAVIS-Flow | 5.94 | 12.14 | General AV separation unsuitable for CASS |

### Key Findings

1. Visual cues reduce FAD from 1.63 to 0.84 (48% improvement) and WPR from 2.01 to 1.84.
2. Qualitative analysis: birdsong is misattributed to music by the audio-only model, whereas AV-CASS correctly assigns it to sound effects via the bird visible in the scene.
3. Synthetic training generalizes successfully to real films (MOS 4.13/5).
4. CASS ≠ general AV separation: DAVIS-Flow achieves low WPR on FX but performs poorly on DX and MX.

## Highlights & Insights

- The train-inference paradigm shift is elegant: training uses dual video streams from independent sources, while inference extracts face and scene streams from a single film without architectural modification.
- Flow matching demonstrates strong perceptual quality in audio separation.
- The WPR metric is a novel contribution—measuring cross-track leakage without requiring ground-truth references.
- Audio warm-up followed by progressive visual injection prevents premature over-reliance on visual cues.

## Limitations & Future Work

- Music lacks visual correspondence, so visual cues provide limited gain for MX separation.
- 128-step inference is relatively slow; distillation-based acceleration warrants exploration.
- The model operates at 16 kHz mono; validation at production-level 48 kHz multi-channel audio remains future work.

## Related Work & Insights

- The synthetic training → real-world generalization paradigm offers inspiration for other multimodal tasks lacking paired data.
- Applications of conditional flow matching in audio generation are expanding rapidly.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First AV-CASS framework + synthetic pipeline + dual-stream design
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Real-film MOS + full metrics on synthetic test set + public benchmarks
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, well-motivated contributions
- Value: ⭐⭐⭐⭐⭐ Opens the AV-CASS research direction with direct application to cinematic post-production

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[AAAI 2026\] MACS: Multi-source Audio-to-Image Generation with Contextual Significance and Semantic Alignment](../../AAAI2026/image_generation/macs_multi-source_audio-to-image_generation_with_contextual_significance_and_sem.md)
- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](../../NeurIPS2025/image_generation/a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[CVPR 2026\] Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models](probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)
- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](depthvar_depth_adaptive_var.md)

<!-- RELATED:END -->
