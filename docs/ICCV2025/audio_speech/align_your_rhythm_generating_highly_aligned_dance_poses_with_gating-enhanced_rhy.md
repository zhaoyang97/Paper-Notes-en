---
title: >-
  [Paper Note] Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation
description: >-
  [ICCV 2025][Audio & Speech][music-driven dance generation] This paper proposes Danceba, a framework comprising three core modules — Phase-based Rhythm Extraction (PRE), Temporal Gated Causal Attention (TGCA)…
tags:
  - "ICCV 2025"
  - "Audio & Speech"
  - "music-driven dance generation"
  - "rhythm alignment"
  - "gated attention"
  - "Mamba"
  - "human motion generation"
date: 2026-05-08
content_hash: 29a48c0d865c5244
---

# Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation

**Conference**: ICCV 2025
**arXiv**: [2503.17340](https://arxiv.org/abs/2503.17340)  
**Code**: [https://danceba.github.io/](https://danceba.github.io/)  
**Area**: Human Understanding / Music-Driven Dance Generation
**Keywords**: music-driven dance generation, rhythm alignment, gated attention, Mamba, human motion generation

## TL;DR
This paper proposes Danceba, a framework comprising three core modules — Phase-based Rhythm Extraction (PRE), Temporal Gated Causal Attention (TGCA), and Parallel Mamba Motion Modeling (PMMM) — to achieve music-driven dance generation with high rhythm alignment and diversity, attaining a 48.68% improvement in FIDk and a 12% improvement in BAS on the AIST++ dataset.

## Background & Motivation

Music-driven dance generation aims to automatically synthesize 3D dance motions synchronized with music, with broad applications in gaming, VR, and film production. The central challenge for existing methods is generating dances that accurately follow musical rhythm while remaining natural and diverse.

Representative methods such as Bailando introduce cross-conditional causal attention to align music and dance poses, but this mechanism primarily captures local correspondences and fails to sufficiently leverage global rhythmic features in music. Subsequent approaches like Enhancing-Bailando employ the MERT pre-trained audio encoder, which focuses on semantic information rather than explicit rhythmic features. Beat-It explicitly incorporates beat synchronization, but its rigid control constraints and strong regularization substantially limit the diversity and expressiveness of generated dances.

The starting point of this paper is that existing methods either overlook the explicit rhythmic structure in music or enforce beat alignment at the cost of diversity. The core idea is to construct gating-enhanced rhythm-aware feature representations that achieve precise rhythm alignment without sacrificing motion naturalness or diversity.

## Method

### Overall Architecture
Danceba follows the VQ-VAE + autoregressive framework of Bailando: a pre-trained Pose VQ-VAE encodes upper- and lower-body dance into discrete codebook sequences, and an autoregressive model predicts the next pose code. The key improvement replaces the original attention with a cascade of three new modules: PRE → TGCA → PMMM → TGCA.

### Key Designs

1. **Phase-based Rhythm Extraction (PRE)**:

    - Function: Explicitly extracts rhythmic features from music input, decoupled from semantic features.
    - Mechanism: Applies Short-Time Fourier Transform (STFT) to the input music feature $\mathbf{m}$ and extracts the phase angle $\boldsymbol{\varphi} = \text{Angle}(\text{STFT}(\mathbf{m}))$. Phase angles naturally encode temporal offsets and periodic structure — low-frequency phase variations correspond to macro-level rhythmic patterns, while high-frequency variations capture fine-grained rhythmic details. After center-cropping for temporal alignment, the features are transformed into embedding-space representations $\mathbf{X}_\varphi$ via Linear + BN + ReLU, then added to the original input features.
    - Design Motivation: Raw music features are "rhythm-poor," whereas STFT phase features are "rhythm-rich." STFT is preferred over standard FT because the non-stationary nature of music requires a sliding window to capture sudden beat transitions.

2. **Temporal Gated Causal Attention (TGCA)**:

    - Function: Strengthens the guidance of global rhythmic features on dance generation, addressing the lack of historical token control signals for the next token in cross-conditional causal attention.
    - Mechanism: Introduces a gating mechanism on top of cross-conditional causal attention $C^3\text{Attention}$:
    $\mathbf{X}_{attn} = C^3\text{Attention}(\mathbf{X}) \odot \text{SiLU}(\text{Linear}(\mathbf{X}))$
      The gating signal is combined with the attention output via element-wise multiplication.
    - Design Motivation: Visualization analysis reveals that in the original causal attention, predicted tokens lack explicit control signals from historical tokens, leading to error accumulation and unnatural motions. The gating mechanism enables historical tokens to better guide next-token prediction and establishes stable global beat attention.

3. **Parallel Mamba Motion Modeling (PMMM)**:

    - Function: Models upper-body and lower-body motion sequences separately to capture their respective unique motion dynamics.
    - Mechanism: Three parallel Mamba branches process the TGCA-enhanced features of music, upper body, and lower body respectively. Each branch consists of Mamba(RMSNorm(·)) + residual connection + GateMlp(RMSNorm(·)) + residual connection. GateMlp selectively retains key motion features via a gating mechanism.
    - Design Motivation: Upper- and lower-body motion patterns and velocities differ substantially, necessitating separate modeling to maintain coherence and diversity. Mamba has demonstrated superior sequence modeling capability over Transformers in human motion generation and is better suited to long-sequence generation.

### Loss & Training
- Cross-entropy loss supervises upper- and lower-body pose code predictions at each time step:
  $$\mathcal{L}_{CE} = \frac{1}{T'} \sum_{t=0}^{T'-1} \sum_{h=u,l} \text{CrossEntropy}(a_t^h, p_{t+1}^h)$$
- During inference, dance is autoregressively predicted from initial pose codes and the full music sequence, then decoded by the Pose VQ-VAE decoder.

## Key Experimental Results

### Main Results: AIST++ Dataset

| Method | FIDk↓ | FIDg↓ | Divk↑ | Divg↑ | BAS↑ |
|------|-------|-------|-------|-------|------|
| Ground Truth | 17.10 | 10.60 | 8.19 | 7.45 | 0.2374 |
| Bailando | 28.16 | 9.62 | 7.83 | 6.34 | 0.2332 |
| DiffDance | 24.09 | 20.68 | 6.02 | 2.89 | 0.2418 |
| Bailando++ | 22.74 | 11.58 | 7.94 | 6.34 | 0.2263 |
| E3D2 | 26.25 | 8.94 | 7.96 | 6.49 | 0.2232 |
| Lodge | 37.09 | 18.79 | 5.58 | 4.85 | 0.2423 |
| **Danceba (Best)** | **11.67** | 11.90 | **8.52** | **7.55** | **0.2714** |

### Ablation Study

| Configuration | FID↓ | Div↑ | BAS↑ |
|------|------|------|------|
| Ground Truth | 13.85 | 7.82 | 0.2374 |
| w/o TGCA | 33.22 | 6.74 | – |
| w/o PMMM | 25.24 | 5.78 | – |
| w/o PRE | 20.62 | 6.53 | 0.2474 |
| **Danceba** | **14.53** | **7.67** | **0.2779** |
| Danceba-Single | FIDk=82.50 | – | – |
| Danceba-Parallel | FIDk=15.48 | – | – |

### Key Findings
- FIDk improves by 48.68% (11.67 vs. 22.74), BAS improves by 12% (0.2714 vs. 0.2423), and both Divk and Divg exceed Ground Truth.
- Removing the PRE module degrades FID by 41.9% and BAS by 11%, underscoring the importance of phase-based rhythm extraction.
- Parallel Mamba vs. single Mamba: FIDk drops from 82.50 to 15.48 (a gain of 67), demonstrating the necessity of modeling upper and lower body separately.
- Attention heatmap visualizations of TGCA confirm that the gating mechanism effectively establishes control signals from historical tokens to predicted tokens.

## Highlights & Insights
- The use of STFT phase information as rhythmic features is a concise and effective approach — a generally applicable technique, as phase naturally encodes temporal offsets and periodic information.
- Combining SiLU gating with causal attention is a low-cost strategy for global context enhancement that is transferable to other autoregressive generation tasks.
- The parallel Mamba design suggests a broader principle: for generation targets with clear structural divisions (e.g., upper and lower body), separate modeling followed by fusion outperforms unified modeling.

## Limitations & Future Work
- Music feature encoding relies on simple linear layers without leveraging pre-trained audio models (e.g., Jukebox, MERT, CLAP), potentially missing subtle musical structures.
- The Pose VQ-VAE from Bailando is retained as-is; its limited encoding space may constrain the representation of fine-grained motion details.
- Evaluation is conducted only on AIST++, leaving cross-dataset generalization unverified.
- Style-conditioned inputs (e.g., dance genre or style control) are not considered, limiting flexibility.
- Random sampling during training introduces variance, requiring averaging over 5 independent training runs for stable results.

## Related Work & Insights
- **vs. Bailando**: Danceba inherits Bailando's VQ-VAE + autoregressive framework but systematically improves it across three dimensions — feature extraction, attention mechanism, and motion modeling — via the PRE-TGCA-PMMM modules.
- **vs. Beat-It**: Beat-It enforces synchronization with an explicit beat predictor at the cost of diversity; Danceba achieves implicit alignment through rhythm-aware features, substantially improving BAS without sacrificing Div.
- **vs. EDGE**: EDGE uses a conditional diffusion model for dance generation and underperforms Danceba on both FIDk and Div, suggesting that the autoregressive + VQ-VAE framework augmented with rhythm enhancement remains competitive.
- **vs. Motion Mamba variants**: Danceba validates the effectiveness of Mamba for dance generation, and the parallel structure significantly outperforms a single Mamba, confirming the value of the proposed design.

## Rating
- Novelty: ⭐⭐⭐⭐ — Phase-based rhythm extraction via STFT and gated causal attention are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations (including 5-run averages), though limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, heatmap visualizations are intuitive, and inter-module relationships are well explained.
- Value: ⭐⭐⭐⭐ — Represents a significant contribution to music-driven dance generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](../../NeurIPS2025/audio_speech/megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] Generating Physically Sound Designs from Text and a Set of Physical Constraints](../../NeurIPS2025/audio_speech/generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[NeurIPS 2025\] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization](../../NeurIPS2025/audio_speech/unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)

</div>

<!-- RELATED:END -->
