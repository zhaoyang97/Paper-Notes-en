---
title: >-
  [Paper Note] Few-shot Acoustic Synthesis with Multimodal Flow Matching
description: >-
  [CVPR2026][Image Generation][flow matching] This paper proposes FLAC, the first flow matching-based few-shot room impulse response (RIR) generation framework, capable of synthesizing spatially consistent acoustic responses in unseen scenes from a single recording. It further introduces AGREE, a joint embedding for geometry–acoustic consistency evaluation.
tags:
  - CVPR2026
  - Image Generation
  - flow matching
  - room impulse response
  - few-shot acoustic synthesis
  - diffusion transformer
  - multimodal conditioning
  - joint embedding
date: 2026-05-08
content_hash: 76e9d9fe42cc2d35
---

# Few-shot Acoustic Synthesis with Multimodal Flow Matching

**Conference**: CVPR2026  
**arXiv**: [2603.19176](https://arxiv.org/abs/2603.19176)  
**Code**: [Project Page](https://amandinebtto.github.io/FLAC/)  
**Area**: Image Generation (Audio Generation / Acoustic Synthesis)  
**Keywords**: flow matching, room impulse response, few-shot acoustic synthesis, diffusion transformer, multimodal conditioning, joint embedding

## TL;DR

This paper proposes FLAC, the first flow matching-based few-shot room impulse response (RIR) generation framework, capable of synthesizing spatially consistent acoustic responses in unseen scenes from a single recording. It further introduces AGREE, a joint embedding for geometry–acoustic consistency evaluation.

## Background & Motivation

**Importance of room acoustic modeling**: Immersive virtual environments require spatial consistency between sound and space. Room impulse responses (RIRs) characterize sound propagation between a source and a receiver and are essential for spatial audio rendering.

**Limitations of neural acoustic fields**: Existing neural acoustic field methods (e.g., NeRAF, AV-GS) achieve spatially continuous rendering within a single scene but require dense recordings and per-scene training, preventing generalization to new environments.

**Insufficiency of few-shot methods**: Few-shot approaches such as FewShotRIR, MAGIC, and xRIR require 8–20 reference recordings and produce deterministic predictions, ignoring the inherent uncertainty of acoustic responses under sparse observations.

**Drawbacks of deterministic modeling**: With limited scene information, a single source–receiver configuration may correspond to multiple plausible RIRs (e.g., carpet versus hardwood floors yield significantly different acoustics); deterministic methods cannot capture this ambiguity.

**Potential of flow matching for audio generation**: Flow matching, as an efficient alternative to diffusion models, has demonstrated strong performance in text-to-speech and music generation, yet has not been applied to explicit RIR synthesis.

**Lack of geometry-consistency evaluation**: Conventional acoustic metrics (T60, C50, EDT) measure perceptual quality only and lack the means to assess the geometric consistency of generated RIRs with the scene.

## Method

### Overall Architecture

FLAC is a conditional latent generative model comprising three core modules:

1. **VAE encoder**: Compresses RIR waveforms into a compact latent representation $\mathbf{z}_0$ with a bottleneck dimension of 32.
2. **Multimodal conditioner**: Fuses acoustic (reference RIRs), spatial (source position), and geometric (panoramic depth map) conditioning.
3. **Diffusion Transformer (DiT)**: Trained with a flow matching objective to generate RIR latent representations from noise.

Training employs rectified flow matching with linear interpolation between data and noise: $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\boldsymbol{\epsilon}$, with the model predicting the velocity field $\mathbf{v}_t = \boldsymbol{\epsilon} - \mathbf{z}_0$. At inference, the ODE is solved in reverse from Gaussian noise to produce RIRs.

### Key Designs

- **Timestep sampling strategy**: Sampled from $\alpha \sim \mathcal{N}(-1.2, 4)$ and mapped through sigmoid, concentrating on intermediate noise levels ($t \approx 0.7$–$0.8$) to improve training efficiency.
- **Multimodal conditioning injection**:
    - **Acoustic conditioning**: $K$ reference RIRs encoded by ResNet-18 into 512-dimensional embeddings.
    - **Spatial conditioning**: Source position coordinates encoded via sinusoidal positional encoding followed by a linear projection.
    - **Geometric conditioning**: Panoramic depth maps converted to 3D coordinates via equirectangular projection, reflection maps computed, and encoded by a fine-tuned DINOv3 ViT-S/16.
- **DiT architecture**: 12-layer Transformer with 8-head attention and hidden dimension 256. Target pose and timestep are injected via AdaLN; multimodal context is fused via cross-attention. RoPE positional encoding is used.
- **Classifier-free guidance**: Conditions are randomly dropped during training; guidance weight $\omega$ controls conditioning strength at inference.
- **AGREE joint embedding**: A CLIP-style dual encoder aligning RIRs and scene geometry in a shared latent space, enabling zero-shot cross-modal retrieval.

### Loss & Training

- **Flow matching loss**: $\mathcal{L}_{\text{RFM}} = \mathbb{E}[\|u(\mathbf{z}_t, t, \boldsymbol{\tau}) - \mathbf{v}_t\|^2]$
- **VAE training loss**: Multi-resolution STFT loss $\mathcal{L}_{\text{MR}}$ (spectral convergence + energy decay) + adversarial hinge loss $\mathcal{L}_{\text{adv}}$ + feature matching loss $\mathcal{L}_{\text{feat}}$ (Encodec multi-scale STFT discriminator) + KL divergence $\mathcal{L}_{\text{KL}}$
- **AGREE contrastive loss**: Maximizes similarity for matched pairs and minimizes similarity for unmatched pairs.

## Key Experimental Results

### Datasets & Setup

- **AcousticRooms (AR)**: 260 rooms with 300k+ RIRs at 22,050 Hz, simulated via the wave equation; 243 seen / 17 unseen rooms.
- **Hearing-Anything-Anywhere (HAA)**: 4 real rooms for sim-to-real transfer evaluation.
- Training conducted on a single H100 GPU with AdamW optimizer, learning rate $5 \times 10^{-5}$, batch size 64, BF16 precision.

### Main Results

**8-shot generation on unseen scenes (AcousticRooms)**:

| Method | K | T60 (%) ↓ | C50 (dB) ↓ | EDT (ms) ↓ | R@5 (%) ↑ |
|--------|---|-----------|------------|------------|-----------|
| xRIR | 8 | 9.98 | 1.354 | 49.40 | 2.00 |
| **FLAC** | **8** | **8.60** | **0.970** | **37.13** | **19.38** |
| xRIR | 1 | 14.47 | 1.961 | 74.45 | 1.36 |
| **FLAC** | **1** | **9.95** | **1.046** | **40.04** | **18.92** |

**Sim-to-real transfer (HAA)**:

| Method | K | T60 (%) ↓ | C50 (dB) ↓ | EDT (ms) ↓ |
|--------|---|-----------|------------|------------|
| Diff-RIR† | 12 | 3.74 | 2.067 | 88.09 |
| **FLAC** | **8** | **3.10** | **2.167** | **84.52** |
| **FLAC** | **1** | **3.45** | **2.170** | **90.02** |

### Ablation Study

- **Conditioning modality ablation**: Geometry-only conditioning yields better C50 and EDT (early reflections determined by nearby surfaces); acoustic-only conditioning yields better T60 (global reverberation is difficult to infer from local geometry); combining both achieves the best performance.
- **Geometric encoder**: Fine-tuned DINOv3 ViT-S/16 outperforms both training from scratch and frozen variants, as well as xRIR's ViT.
- **DiT conditioning strategy**: AdaLN + Cross-Attention significantly outperforms In-Context and pure Cross-Attention approaches.
- **Acoustic encoder**: A frozen VAE encoder generalizes slightly better across rooms than ResNet-18, at higher computational cost.

### Key Findings

- FLAC with 1-shot surpasses all 8-shot baselines; 93.01% of participants (n=46) in subjective listening tests preferred FLAC.
- Uncertainty analysis: Low-frequency bands exhibit higher sample variance and longer durations, consistent with room acoustics theory — low-frequency responses are dominated by sparse boundary modes, while high frequencies stabilize above the Schröder frequency.
- Intra-condition diversity ratio is 4.5% (1.03 vs. 22.96), indicating that the model introduces meaningful stochasticity while maintaining contextual consistency.
- The deterministic variant (fixed noise) shows significantly degraded performance (+6% T60, +10% C50, −40% R@5), confirming that stochasticity is essential for few-shot acoustic synthesis.

## Highlights & Insights

- **Pioneering contribution**: First application of flow matching to explicit RIR synthesis, framing few-shot acoustic synthesis as a probabilistic generative problem.
- **Exceptional data efficiency**: 1-shot performance surpasses the previous 8-shot SOTA, reducing required recordings by 8×.
- **AGREE evaluation framework**: Proposes a CLIP-style acoustic–geometry joint embedding that fills the gap in geometry-consistency evaluation and enables zero-shot cross-modal retrieval.
- **Physically grounded uncertainty modeling**: Higher uncertainty at low frequencies and convergence at high frequencies align with room acoustics Schröder frequency theory.
- **Practical applicability**: Trained on a single H100; inference requires only one step to obtain high-quality results; the few-shot method adapts to new scenes in minutes.

## Limitations & Future Work

- **Inaccurate area classification**: This paper genuinely belongs to audio/acoustic synthesis; its classification under image generation is inappropriate.
- **Limited real-scene generalization**: Geometric annotations in the HAA dataset are simplified (e.g., tables modeled as planes), and the VAE is not fine-tuned on real recordings, limiting sim-to-real transfer.
- **Single sample rate constraint**: The current model supports only 22,050 Hz; high-fidelity applications require higher sample rates.
- **Elevated FDG metric**: The generated distribution still diverges from the real distribution in AGREE space, particularly on real data.
- **Scarcity of real data**: The absence of large-scale, diverse real audio-visual datasets constrains VAE and overall model performance in real-world scenarios.
- **Monaural limitation**: Only monaural omnidirectional RIRs are handled; binaural and multichannel scenarios are not addressed.

## Related Work & Insights

- **Neural acoustic fields**: Per-scene training methods such as NeRAF and AV-GS achieve spatially continuous rendering but are non-generalizable.
- **Few-shot acoustic synthesis**: FewShotRIR (20 samples) → MAGIC (semantic augmentation) → xRIR (8 samples + depth maps); all are deterministic methods.
- **Audio diffusion and flow matching**: Diffusion models have succeeded in speech/music generation; flow matching improves efficiency; this paper is the first to introduce it to RIR synthesis.
- **Joint embedding models**: CLIP → audio-visual/audio-text embeddings, but standard audio embeddings are unsuitable for RIRs; AGREE is the first to align RIRs with scene geometry.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First application of flow matching to RIR synthesis; novel probabilistic modeling perspective; pioneering AGREE evaluation framework)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two datasets, multiple baselines, comprehensive ablations, uncertainty analysis, subjective listening tests, cross-modal retrieval validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures, adequate physical intuition; notation is somewhat dense in places)
- Value: ⭐⭐⭐⭐ (Opens a new direction for few-shot acoustic synthesis with exceptional practical data efficiency, though the field is relatively niche)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [\[CVPR 2026\] MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching](mpdit_multi-patch_global-to-local_transformer_architecture_for_efficient_flow_ma.md)
- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](../../ICLR2026/image_generation/flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)

</div>

<!-- RELATED:END -->
