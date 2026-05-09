---
title: >-
  [Paper Note] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders
description: >-
  [NeurIPS 2025 (AI for Music Workshop)][Audio & Speech][Perceptual Alignment] This paper demonstrates that applying noise augmentation to latent variables during autoencoder training, combined with a perceptual loss, induces a "perceptual hierarchy" in the encoding space — the most perceptually salient musical features (e.g., pitch) are encoded in the coarsest latent structures, while secondary features (e.g., timbral details) are encoded in finer structures. This alignment improves music surprisal estimation under latent diffusion decoding and enhances EEG brain response prediction.
tags:
  - NeurIPS 2025 (AI for Music Workshop)
  - "Audio & Speech"
  - Perceptual Alignment
  - Noise-Augmented Autoencoder
  - Latent Diffusion
  - Music Surprisal
  - EEG Prediction
date: 2026-05-08
content_hash: 35476edba62a56cf
---

# Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders

**Conference**: NeurIPS 2025 (AI for Music Workshop)
**arXiv**: [2511.05350](https://arxiv.org/abs/2511.05350)
**Code**: [https://github.com/CPJKU/pa-audioic](https://github.com/CPJKU/pa-audioic)
**Area**: Audio / Music Representation Learning
**Keywords**: Perceptual Alignment, Noise-Augmented Autoencoder, Latent Diffusion, Music Surprisal, EEG Prediction

## TL;DR
This paper demonstrates that applying noise augmentation to latent variables during autoencoder training, combined with a perceptual loss, induces a "perceptual hierarchy" in the encoding space — the most perceptually salient musical features (e.g., pitch) are encoded in the coarsest latent structures, while secondary features (e.g., timbral details) are encoded in finer structures. This alignment improves music surprisal estimation under latent diffusion decoding and enhances EEG brain response prediction.

## Background & Motivation

**Background**: The information content (IC) / surprisal of music can be estimated via the negative log-likelihood of autoregressive models, and its correlation with human perception has been validated in behavioral and neuroscientific studies. Recent approaches compute IC using autoregressive diffusion models in the latent space of audio autoencoders.

**Limitations of Prior Work**: Diffusion models decode structures of different granularities at different noise levels — coarse structures are reconstructed before fine ones. However, standard autoencoder training does not guarantee that perceptually important information (e.g., pitch) is encoded in coarse structures, resulting in suboptimal IC estimation at intermediate noise levels.

**Key Challenge**: The spectral SNR characteristics of the diffusion process dictate a "coarse-before-fine" decoding order; yet if perceptually important features are distributed across all granularities, this property cannot be exploited to optimize surprisal estimation.

**Goal**: To explicitly align the latent space learned by the autoencoder with the perceptual hierarchy — placing the most perceptually important information in the coarsest structures.

**Key Insight**: During training, latent variables are corrupted with noise of varying intensity and the model is required to reconstruct the original input; combined with a perceptual loss constraint, the encoder is forced to place information most critical to perceptual loss within coarse structures (those preserved under high SNR conditions).

**Core Idea**: Noise-augmented training + perceptual loss = automatically pushing perceptually salient information into the coarse-grained structures of the latent space.

## Method

### Overall Architecture
Two-stage latent diffusion: (1) an audio autoencoder (Music2Latent/CAE) encodes audio into compressed latent variables and decodes them for reconstruction; (2) an autoregressive rectified flow model performs next-step prediction in the latent space. This work introduces noise-augmented training in stage one, and leverages the aligned latent space in stage two to improve IC estimation.

### Key Designs

1. **Noise-Augmented Latent Training**:

    - Function: Corrupt latent variables with noise during autoencoder training and require reconstruction from the noisy inputs.
    - Mechanism: $z' = (1-t)z + t \cdot n(\gamma)$, where $n(\gamma) \sim \gamma \cdot \mathcal{N}(0,I)$ and $t \sim \mathcal{U}(0,1)$. The decoder must reconstruct the original audio from the noisy $z'$.
    - Design Motivation: Due to spectral SNR characteristics, only coarse structures survive under high noise. The perceptual loss compels the encoder to place the most perceptually important information in coarse structures, as this is the only way to satisfy the perceptual loss under high noise conditions.

2. **Variance Fixing + Biased Sampling**:

    - Function: Prevent the encoder from "cheating" by inflating variance to circumvent the noise.
    - Mechanism: LayerNorm is used to fix the variance of $z$ equal to the noise distribution variance, $\gamma^2=1$. The variable $t$ is sampled from a logit-normal distribution (rather than a uniform distribution) to control the SNR distribution.
    - Design Motivation: In the prior method (yang2025detok), the encoder can increase the variance of $z$ to raise the SNR, which effectively weakens the impact of noise augmentation.

3. **Improved Surprisal Estimation**:

    - Function: Estimate music surprisal using a diffusion model in the aligned latent space.
    - Mechanism: An autoregressive rectified flow model is trained to predict the next step in the latent space; IC/NLL is computed via the instantaneous change-of-variables formula. IC is computed at different noise levels and evaluated via correlation with IDyOM (a symbolic music expectation model) and EEG prediction accuracy.
    - Design Motivation: After alignment, intermediate noise levels should yield better surprisal estimates for perceptually relevant features such as pitch.

### Loss & Training
- Base loss: Perceptually weighted complex spectrogram difference (consistency training)
- Fine-tuning from a pretrained Music2Latent checkpoint
- Consistency step size fixed to the final value of the pretrained model

## Key Experimental Results

### Main Results: Reconstruction Quality at Different SNR Levels

| Training | SNR | ViSQOL (V)↑ | SI-SDR (SI)↑ | FAD-VGG↓ | FAD-CLAP↓ |
|---------|-----|-------------|-------------|----------|----------|
| Aligned (E,D) | ∞ | 3.73 | -5.18 | 1.53 | 0.05 |
| Aligned (E,D) | 4.0 | 3.48 | -9.05 | 2.46 | 0.08 |
| Aligned (E,D) | 1.0 | 3.19 | -15.73 | 3.64 | 0.17 |
| Unaligned | 4.0 | 2.94 | -11.44 | 6.63 | 0.42 |
| Unaligned | 1.0 | 2.53 | -18.82 | 11.15 | 0.84 |

### Ablation Study: Surprisal Estimation

| Method | IDyOM Correlation↑ | EEG Prediction↑ |
|------|-------------|----------|
| Unaligned baseline | Lower | Lower |
| Aligned (Ours) | Significant improvement | Significant improvement |
| Aligned + optimal noise level | Best | Best |

### Key Findings
- **Aligned models substantially outperform unaligned models in perceptual quality at low SNR**: ViSQOL at SNR=4 is 3.48 vs. 2.94, confirming that alignment pushes perceptual information into coarse structures.
- **Surprisal estimation at intermediate noise levels is optimal after alignment**: This validates the core hypothesis — once perceptual features are aligned with coarse structures, the "coarse-first" decoding property of diffusion models is fully exploited.
- **Variance fixing is critical**: Without it, the encoder circumvents the noise augmentation by inflating its variance.
- **Aligned models achieve better FAD at low SNR**: This suggests that the decoder generates more plausible content when information is missing.

## Highlights & Insights
- The concept of a **"perceptual hierarchy"** is insightful: different granularities of latent structure correspond to different levels of perceptual importance. This idea generalizes to other modalities (e.g., global structure vs. textural details in vision).
- The **implicit alignment mechanism via noise augmentation + perceptual loss** is elegant and concise — no explicit design is needed to assign information to specific levels; the spectral properties of noise automatically perform the hierarchical assignment.
- The work offers broadly applicable insights for latent diffusion model design: the structure of an autoencoder's latent space should match the decoding order of the diffusion process.

## Limitations & Future Work
- Validation is limited to monophonic symbolic music and vocal audio; performance on complex polyphonic music remains unknown.
- Evaluation metrics (IDyOM correlation, EEG) are restricted to the pitch dimension; perceptual alignment for rhythm and harmony has not been verified.
- As a workshop paper, the study is small in scale, and large-scale validation is lacking.
- The choice of perceptual loss may affect the resulting hierarchy — whether different perceptual metrics yield different hierarchical structures warrants further exploration.
- Fixing variance via LayerNorm may constrain the encoder's expressive capacity, necessitating a balance between alignment quality and reconstruction quality.

## Related Work & Insights
- **vs. Music2Latent (Pasini, 2024)**: This work builds upon Music2Latent by introducing noise-augmented training.
- **vs. yang2025detok**: This work improves upon the variance fixing strategy and sampling distribution proposed therein.
- **vs. bjare2025diffusionsurprisal**: The proposed alignment improves performance within the diffusion surprisal framework introduced in that work.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical analysis of how noise-augmented training induces a perceptual hierarchy is substantive.
- Experimental Thoroughness: ⭐⭐⭐ A workshop paper, but validated across reconstruction quality, surprisal estimation, and EEG prediction.
- Writing Quality: ⭐⭐⭐⭐ Theoretical motivation is clearly and rigorously derived.
- Value: ⭐⭐⭐⭐ Offers broadly applicable insights for latent diffusion model design.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Aligning Generative Music AI with Human Preferences: Methods and Challenges](../../AAAI2026/audio_speech/aligning_generative_music_ai_with_human_preferences_methods_and_challenges.md)
- [\[NeurIPS 2025\] Segment-Factorized Full-Song Generation on Symbolic Piano Music](segment-factorized_full-song_generation_on_symbolic_piano_music.md)
- [\[ICLR 2026\] Scaling Speech Tokenizers with Diffusion Autoencoders](../../ICLR2026/audio_speech/scaling_speech_tokenizers_with_diffusion_autoencoders.md)
- [\[NeurIPS 2025\] Ethics Statements in AI Music Papers: The Effective and the Ineffective](ethics_statements_in_ai_music_papers_the_effective_and_the_ineffective.md)
- [\[NeurIPS 2025\] BNMusic: Blending Environmental Noises into Personalized Music](bnmusic_blending_environmental_noises_into_personalized_music.md)

<!-- RELATED:END -->
