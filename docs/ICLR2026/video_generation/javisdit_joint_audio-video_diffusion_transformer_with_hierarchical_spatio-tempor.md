---
title: >-
  [Paper Note] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization
description: >-
  [ICLR 2026][Video Generation][Joint Audio-Video Generation] This paper proposes JavisDiT, a joint audio-video generation model built on the DiT architecture. It achieves fine-grained spatio-temporal audio-video alignment via a Hierarchical Spatio-Temporal Synchronization Prior Estimator (HiST-Sypo). The work also introduces a new benchmark, JavisBench (10K complex-scene samples), and a new evaluation metric, JavisScore.
tags:
  - ICLR 2026
  - Video Generation
  - Joint Audio-Video Generation
  - DiT
  - Spatio-Temporal Synchronization
  - Contrastive Learning
  - Benchmark Dataset
date: 2026-05-08
content_hash: 3f04eba6300ecc2f
---

# JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization

**Conference**: ICLR 2026
**arXiv**: [2503.23377](https://arxiv.org/abs/2503.23377)
**Code**: [https://javisverse.github.io/JavisDiT-page/](https://javisverse.github.io/JavisDiT-page/)
**Area**: Diffusion Models / Video Generation
**Keywords**: Joint Audio-Video Generation, DiT, Spatio-Temporal Synchronization, Contrastive Learning, Benchmark Dataset

## TL;DR

This paper proposes JavisDiT, a joint audio-video generation model built on the DiT architecture. It achieves fine-grained spatio-temporal audio-video alignment via a Hierarchical Spatio-Temporal Synchronization Prior Estimator (HiST-Sypo). The work also introduces a new benchmark, JavisBench (10K complex-scene samples), and a new evaluation metric, JavisScore.

## Background & Motivation

**Background**: Audio and video are naturally coupled in real-world scenarios, making joint audio-video generation (JAVG) valuable for film production and short-video creation.

**Limitations of Prior Work**: Cascaded asynchronous approaches—generating audio first and then synthesizing video, or vice versa—accumulate noise; end-to-end methods are more promising. Existing DiT backbones such as AV-DiT and MM-LDM rely on image-based DiTs and struggle to model fine-grained spatio-temporal relationships. Current synchronization strategies achieve only coarse temporal alignment (parameter sharing) or semantic alignment (embedding alignment), lacking fine-grained spatial synchronization. Existing benchmarks such as AIST++ and Landscape contain overly simple scenes that fail to capture complex, multi-event real-world scenarios. The AV-Align metric relies on optical flow and audio onset detection, which are unreliable in complex scenes.

**Key Challenge**: Prior JAVG methods lack a mechanism to jointly model the *spatial* and *temporal* dimensions of audio-video synchronization at a fine granularity.

**Goal**: To develop an end-to-end JAVG framework with hierarchical spatio-temporal prior estimation, alongside a more challenging benchmark and a more robust evaluation metric.

## Method

### Overall Architecture

JavisDiT comprises a video branch and an audio branch that share AV-DiT blocks. Each branch sequentially passes through: **ST-SelfAttn** → **coarse-grained CrossAttn** (T5 semantics) → **fine-grained ST-CrossAttn** (spatio-temporal prior) → **bidirectional CrossAttn** (cross-modal fusion).

### Key Design 1: Hierarchical Spatio-Temporal Synchronization Prior Estimator (HiST-Sypo)

**Coarse-grained prior**: The semantic embeddings from the T5 encoder are directly reused to describe overall sound events.

**Fine-grained prior estimation**:
- 77 hidden states from the ImageBind text encoder serve as input.
- $N_s = 32$ spatial tokens and $N_t = 32$ temporal tokens serve as queries.
- A 4-layer Transformer encoder-decoder $\mathcal{P}$ extracts spatio-temporal priors.
- The outputs parameterize a Gaussian distribution; stochastic spatio-temporal priors are sampled as $(p_s, p_t) \leftarrow \mathcal{P}_\phi(s; \epsilon)$.
- The estimator is trained via **contrastive learning** using negative samples (asynchronous audio-video pairs) and a dedicated loss function.

### Key Design 2: Multi-Modal Bidirectional Cross-Attention (MM-BiCrossAttn)

- An attention matrix $A$ is computed from video queries $q_v$ and audio keys $k_a$.
- $A \times v_a$ → audio-to-video attention.
- $A^T \times v_v$ → video-to-audio attention.
- The bidirectional information flow enables deep cross-modal fusion.

### Three-Stage Training Strategy

1. **Audio pre-training** (0.8M audio-text pairs): The audio branch is initialized with weights from OpenSora's video branch.
2. **ST-Prior training** (0.6M synchronized audio-video triplets): The HiST-Sypo estimator is trained.
3. **JAVG training** (0.6M samples): The self-attention and ST-Prior modules are frozen; only ST-CrossAttn and Bi-CrossAttn are trained.

### Loss & Training

- Diffusion denoising loss (Flow Matching or DDPM).
- ST-Prior estimator: contrastive learning loss (synchronized positive samples vs. asynchronous negative samples).
- Dynamic temporal masking supports multiple conditional generation tasks.

## Key Experimental Results

### Main Results on JavisBench

| Method | FVD ↓ | FAD ↓ | TV-IB ↑ | AV-IB ↑ | JavisScore ↑ |
|--------|-------|-------|---------|---------|-------------|
| TempoToken (T2A→A2V) | 539.8 | - | 0.084 | - | - |
| MM-Diffusion (JAVG) | - | - | - | - | - |
| **JavisDiT** | **Best** | **Best** | **Best** | **Best** | **Best** |

### JavisBench Dataset Characteristics

| Dimension | # Categories | Description |
|-----------|-------------|-------------|
| Event scene | Multiple | Nature, industrial, indoor, etc. |
| Spatial composition | 2 | Single / multiple sounding subjects |
| Temporal composition | 3 | Single event / sequential / concurrent |
| Total samples | 10,140 | 75% multi-event; 57% concurrent events |

### Ablation Study

JavisDiT also significantly outperforms MM-Diffusion and cascaded methods on traditional benchmarks (AIST++ and Landscape) across FVD, KVD, and FAD metrics.

## Highlights & Insights

1. **Fine-grained spatio-temporal alignment**: The model aligns not only *when* a sound occurs but also *where in the frame* it occurs—a spatial dimension overlooked by prior work.
2. **Stochastic prior sampling**: The same text prompt can correspond to different spatio-temporal prior distributions, explicitly modeling uncertainty in the location and timing of events.
3. **Challenging nature of JavisBench**: With 75% of samples containing multiple events and 57% containing concurrent events, the benchmark far exceeds the complexity of existing datasets.
4. **Robustness of JavisScore**: The metric computes ImageBind synchronization scores over sliding windows and selects the least-synchronized 40% of frames, making it more reliable than AV-Align.
5. **Modular design**: Single-modality self-attention blocks are frozen, and only cross-modal modules are trained, yielding parameter efficiency.

## Limitations & Future Work

1. Video generation resolution is relatively low (240P/24fps), lagging behind state-of-the-art video generation models.
2. The model relies on OpenSora pre-trained weights; feasibility of training from scratch has not been verified.
3. ImageBind's joint audio-visual embedding space may lack sufficient granularity in extreme scenarios.
4. The choice of $N_s = 32, N_t = 32$ for the HiST-Sypo estimator has not been thoroughly ablated.
5. Controllability of generated audio (e.g., specific instrument timbre) is not discussed.
6. Although JavisBench contains 10K samples, extension to more diverse linguistic and cultural contexts is needed.

## Related Work & Insights

- **MM-Diffusion (Ruan et al.)**: The first end-to-end JAVG model, using simple parameter sharing for alignment.
- **SyncFlow (Liu et al.)**: Employs STDiT3 blocks but lacks bidirectional information exchange.
- **Seeing-Hearing (Xing et al.)**: Relies on simple embedding alignment without fine-grained spatial information.
- **OpenSora (Zheng et al.)**: The source of pre-trained video branch weights; provides the dynamic temporal masking technique.
- **Insight**: Audio-video synchronization is fundamentally a conditional consistency problem; the paradigm of hierarchical prior estimation combined with contrastive learning is generalizable to other multi-modal alignment scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The fine-grained spatio-temporal prior estimation in HiST-Sypo is genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — New benchmark + new metric + multi-method comparison, though some baselines are not open-sourced.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with clear figures, though some details are deferred to the appendix.
- Value: ⭐⭐⭐⭐ — JAVG is an important yet immature direction; this paper advances standardization in the field.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)
- [\[ICLR 2026\] QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification](quantsparse_comprehensively_compressing_video_diffusion_transformer_with_model_q.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](../../CVPR2026/video_generation/cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](../../CVPR2026/video_generation/generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[CVPR 2026\] DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior](../../CVPR2026/video_generation/dreamshot_storyboard_synthesis.md)

<!-- RELATED:END -->
