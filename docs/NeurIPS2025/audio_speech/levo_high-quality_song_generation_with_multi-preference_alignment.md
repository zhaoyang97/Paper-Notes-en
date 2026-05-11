---
title: >-
  [Paper Note] LeVo: High-Quality Song Generation with Multi-Preference Alignment
description: >-
  [NeurIPS 2025][Audio & Speech][song generation] This paper proposes LeVo, a song generation framework that employs a language model to jointly model mixed tokens and dual-track tokens…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "song generation"
  - "language model"
  - "multi-preference alignment"
  - "DPO"
  - "music codec"
date: 2026-05-08
content_hash: 49dd22b5daa6b878
---

# LeVo: High-Quality Song Generation with Multi-Preference Alignment

**Conference**: NeurIPS 2025
**arXiv**: [2506.07520](https://arxiv.org/abs/2506.07520)
**Code**: [GitHub](https://github.com/tencent-ailab/songgeneration)
**Area**: Audio & Speech Generation
**Keywords**: song generation, language model, multi-preference alignment, DPO, music codec

## TL;DR

This paper proposes LeVo, a song generation framework that employs a language model to jointly model mixed tokens and dual-track tokens, thereby reconciling vocal-accompaniment harmony with audio quality. It further introduces a DPO-based multi-preference alignment method to enhance musicality and instruction-following capability.

## Background & Motivation

**Background**: Advances in LLMs and audio language models have accelerated progress in lyrics-to-song generation. Jukebox pioneered the paradigm of predicting mixed tokens with language models; YuE introduced dual-track (vocal + accompaniment) token prediction; SongGen explored interleaved prediction patterns. Industrial systems (Suno, Mureka, Udio) have demonstrated strong results but remain technically closed.

**Limitations of Prior Work**:
   - Mixed-token methods have limited vocabularies and cannot fully capture complex vocal-accompaniment combinations, constraining audio quality.
   - Dual-track token methods achieve higher audio quality but predict each track independently, leading to vocal-accompaniment disharmony.
   - Interleaved prediction patterns drastically increase sequence length, limiting scalability and long-form song generation.
   - Inconsistent data quality and unreliable music annotations constrain training effectiveness.

**Key Challenge**: Mixed tokens ensure harmony but limit audio quality; dual-track tokens offer better audio quality but suffer from poor harmony. Scarcity of high-quality data further restricts musicality and instruction-following capability.

**Goal**: Simultaneously optimize song generation in terms of audio quality, musicality, instruction-following capability, and vocal-accompaniment harmony.

**Key Insight**: Parallel modeling of mixed tokens and dual-track tokens, combined with modular progressive training and multi-preference DPO alignment.

**Core Idea**: Mixed tokens govern global harmony while dual-track tokens refine audio quality; the two operate in parallel without interfering with each other.

## Method

### Overall Architecture

LeVo = **LeLM** (language model) + **Music Codec**

LeLM consists of a large language model (predicting mixed tokens) and an AR decoder (predicting dual-track tokens). The Music Codec extends MuCodec: an encoder extracts tokens, and a decoder (diffusion Transformer + VAE decoder) reconstructs high-fidelity audio from tokens.

### Key Designs

1. **LeLM Two-Level Architecture**:

    - **Language Model** (decoder-only Transformer): focuses on next-token prediction of mixed tokens, capturing high-level structural information such as melody, rhythm, and beat, ensuring vocal-accompaniment harmony:
    $p(\mathbf{S}_m | \mathbf{C}; \boldsymbol{\theta}) = \prod_{t=0}^T p(\mathbf{S}_{m,t} | \mathbf{S}_{m,<t}, \mathbf{C}; \boldsymbol{\theta})$
    - **AR Decoder** (a shallower decoder-only Transformer): conditioned on the language model's hidden states, it predicts vocal and accompaniment tokens in parallel. A delay pattern is introduced so that when predicting step $t$, the dual-track tokens can attend to the language model's future $k$ steps:
    $p(\mathbf{S}_v, \mathbf{S}_a | \mathbf{C}; \boldsymbol{\theta}) = \prod_{t=0}^{T-k} p(\mathbf{S}_{v,t}, \mathbf{S}_{a,t} | \mathbf{S}_{v,<t}, \mathbf{S}_{a,<t}, \mathbf{S}_{m,<t+k}, \mathbf{C}; \boldsymbol{\theta})$

2. **Music Codec Design**:

    - Encoder = MuEncoder (extracts music-relevant representations) + RVQ (quantizes into tokens)
    - Decoder = diffusion Transformer (reconstructs VAE features from token embeddings) + VAE decoder (directly generates audio)
    - Two token strategies: mixed tokens (processing the full song) and dual-track tokens (separating vocals and accompaniment before encoding each independently)

3. **DPO-Based Multi-Preference Alignment**:

    - **Data Construction**: An LLM generates 20,000 lyrics; each is paired with a random audio prompt and text description, and multiple samples are generated per entry.
    - **Strategy 1 — Lyrics Alignment Preference**: Phoneme error rate (PER) is computed via ASR; pairs with a PER gap exceeding 40 are used as preference pairs.
    - **Strategy 2 — Prompt Consistency Preference**: The MuQ-MuLan model computes similarity scores; threshold-based filtering selects preference pairs.
    - **Strategy 3 — Musicality Preference**: A three-stage pipeline — crowdsourced ranking → reward model training → large-scale filtering — yields approximately 60,000 preference pairs.
    - **Fusion Method — Deep Network Interpolation (DNI)**: Three sets of DPO-fine-tuned parameters are obtained separately on each preference dataset and linearly interpolated into the final model, supporting controllable coefficient adjustment.

### Loss & Training

**Three-Stage Training Paradigm**:

- **Stage 1 — Pre-training**: The language model is trained to align conditioning inputs with mixed tokens; the AR decoder is frozen; audio prompts and text descriptions are each randomly dropped with 50% probability. This stage establishes generation diversity and vocal-accompaniment harmony.
- **Stage 2 — Modular Progressive Training**: The AR decoder is trained to model dual-track tokens; all Stage 1 modules are frozen. This improves audio quality and musicality without disturbing pre-trained knowledge.
- **Stage 3 — Multi-Preference Alignment**: The full LeLM is fine-tuned using DPO loss on multi-dimensional preference data, substantially enhancing musicality and instruction-following capability.

Model scale: LeLM ~2B parameters, MuEncoder 300M parameters, diffusion model ~700M parameters, VAE 150M parameters. Training data: 2 million songs (~110,000 hours).

## Key Experimental Results

### Main Results

Objective metric comparison (open-source and closed-source systems):

| Model | FAD ↓ | MuQ-T ↑ | MuQ-A ↑ | PER ↓ | CE ↑ | CU ↑ | PQ ↑ |
|------|-------|---------|---------|-------|------|------|------|
| Suno-V4.5 | 2.59 | **0.34** | 0.84 | 21.6 | 7.65 | 7.86 | 8.35 |
| Mureka-O1 | **2.50** | 0.33 | **0.87** | **7.2** | 7.71 | 7.83 | 8.44 |
| YuE | 2.65 | 0.27 | 0.74 | 36.4 | 7.13 | 7.39 | 7.77 |
| SongGen* | 2.68 | 0.25 | 0.80 | 27.5 | 7.63 | 7.79 | 8.37 |
| **LeVo** | 2.68 | **0.34** | 0.83 | **7.2** | **7.78** | **7.90** | **8.46** |

Subjective MOS (1–5 scale):

| Model | OVL | MEL | HAM | SSC | AQ | LYC |
|------|-----|-----|-----|-----|-----|-----|
| Suno-V4.5 | **3.59** | **4.10** | **3.93** | **4.19** | **4.00** | 3.17 |
| Mureka-O1 | 3.42 | 3.88 | 3.89 | 4.14 | 3.87 | 3.32 |
| **LeVo** | 3.42 | 3.93 | 3.90 | 4.09 | 3.96 | **3.38** |
| SongGen* | 2.91 | 3.43 | 3.44 | 3.66 | 3.69 | 2.84 |

### Ablation Study

Comparison of DPO multi-preference alignment strategies:

| Method | FAD ↓ | MuQ-T ↑ | PER ↓ | CE ↑ | PQ ↑ |
|------|-------|---------|-------|------|------|
| w/o DPO | **2.60** | 0.31 | 10.6 | 7.70 | 8.39 |
| Strategy 1 only (lyrics alignment) | 2.85 | 0.30 | **6.5** | 7.72 | 8.42 |
| Strategy 2 only (prompt consistency) | 2.89 | **0.34** | 10.3 | 7.75 | 8.43 |
| Strategy 3 only (musicality) | 2.63 | 0.32 | 11.2 | **7.78** | 8.45 |
| Joint training | 2.75 | 0.33 | 7.5 | 7.76 | 8.43 |
| **LeVo (interpolation)** | 2.68 | **0.34** | 7.2 | **7.78** | **8.46** |

### Key Findings

- Removing Stage 2 (modular progressive training) or the AR decoder both lead to performance degradation, validating the necessity of preventing interference between mixed and dual-track tokens.
- The interpolation fusion method outperforms naive joint training; each strategy contributes distinctly, and the approach supports smooth preference transitions.
- LeVo surpasses Suno-V4.5 on lyrics alignment (LYC) by +0.21 MOS.
- LeVo achieves comprehensive superiority among open-source models and competitive performance compared to closed-source industrial systems.

## Highlights & Insights

- **Mixed + dual-track parallel modeling paradigm**: Elegantly resolves the tension between harmony and audio quality without increasing sequence length.
- **Modular progressive training strategy**: Freezing pre-trained modules before training new ones effectively prevents catastrophic forgetting and token interference.
- **First application of multi-preference DPO to song generation**: Three preference data construction strategies each exhibit unique designs.
- **DNI parameter interpolation**: Not only combines multiple preferences but also provides controllable preference weight adjustment.
- **Industrial-grade benchmarking**: The first academic open-source method to achieve performance comparable to industrial systems such as Suno.

## Limitations & Future Work

- Audio quality remains constrained by inconsistent training data quality and the information bottleneck of discrete tokens.
- Song structure modeling (verse/chorus, etc.) remains inferior to Suno and Mureka.
- The precision of pseudo-label annotations (text descriptions generated by Qwen2-Audio) is limited, capping the upper bound of instruction-following performance.
- Training data are not released due to copyright restrictions, limiting reproducibility.
- Style transfer and end-to-end generation capabilities may be exploited for deepfake applications.

## Related Work & Insights

- **Jukebox** (Dhariwal et al., 2020): Pioneered the mixed-token LM paradigm for song generation.
- **YuE** (Yuan et al., 2025): Introduced the dual-track token strategy.
- **SongGen** (Liu et al., 2025): Explored interleaved prediction patterns.
- **MusicRL** (Cideron et al., 2024): Applied RLHF to music generation.
- **Tango2** (Majumder et al., 2024): Pioneering work on DPO for audio generation.
- The adaptation of Deep Network Interpolation (DNI) to the music domain represents a technical direction worthy of further attention.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The mixed + dual-track parallel modeling and multi-preference DPO design demonstrate significant originality.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 comparison systems, complete subjective and objective evaluation, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with thorough technical descriptions.
- Value: ⭐⭐⭐⭐⭐ A milestone contribution to open-source song generation that substantially narrows the gap between academic research and industrial systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)
- [\[NeurIPS 2025\] Segment-Factorized Full-Song Generation on Symbolic Piano Music](segment-factorized_full-song_generation_on_symbolic_piano_music.md)
- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)

</div>

<!-- RELATED:END -->
