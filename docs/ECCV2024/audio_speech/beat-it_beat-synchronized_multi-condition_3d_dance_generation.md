---
title: >-
  [Paper Note] Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation
description: >-
  [ECCV 2024][Audio & Speech][Dance Generation] The Beat-It framework is proposed to achieve beat-synchronized and keyframe-controllable 3D dance generation by decoupling beat conditions from music and designing a hierarchical multi-condition fusion mechanism, significantly outperforming existing methods on AIST++.
tags:
  - "ECCV 2024"
  - "Audio & Speech"
  - "Dance Generation"
  - "Beat Synchronization"
  - "Multi-Condition Diffusion"
  - "Keyframe Control"
  - "3D Motion"
date: 2026-05-08
content_hash: bce5f7cb30e57d1b
---

# Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation

**Conference**: ECCV 2024  
**arXiv**: [2407.07554](https://arxiv.org/abs/2407.07554)  
**Code**: [Project Page](https://zikaihuangscut.github.io/Beat-It/)  
**Area**: Audio & Speech  
**Keywords**: Dance Generation, Beat Synchronization, Multi-Condition Diffusion, Keyframe Control, 3D Motion

## TL;DR

The Beat-It framework is proposed to achieve beat-synchronized and keyframe-controllable 3D dance generation by decoupling beat conditions from music and designing a hierarchical multi-condition fusion mechanism, significantly outperforming existing methods on AIST++.

## Background & Motivation

*   **Core requirements of choreography**: Real dance creation requires precise synchronization between movements and music beats, while being able to assign specific key poses to designated beats. However, traditional manual choreography is time-consuming and labor-intensive.
*   **Poor beat alignment of existing methods**: Existing music-driven dance generation methods (e.g., EDGE, Bailando, FACT) primarily learn a direct mapping from music to dance without explicit beat modeling, leading to out-of-sync movements with music beats.
*   **Weak keyframe control capability**: Existing methods supporting keyframes (e.g., EDGE) introduce keyframe constraints through simple temporal/spatial blending. However, due to the sparsity of keyframes, dense music features tend to suppress keyframe conditions, resulting in insufficient controllability.
*   **Difficulty in multi-condition fusion**: Keyframe conditions are extremely sparse while music/beat conditions are dense. Simple concatenation introduces a large amount of padding noise, causing condition conflicts and seriously affecting the generation quality.
*   **Lack of beat controllability**: Currently, no method can simultaneously achieve beat-specified and keyframe-guided dance generation, failing to meet the actual choreography requirement of "placing specified key poses at specific beats."
*   **Inadequate beat representation**: Directly using binary masks (0/1) to represent beats is overly sparse, making it difficult for the model to effectively leverage and easy to be ignored as noise.

## Method

### Overall Architecture

Beat-It is based on a diffusion model and accepts three input conditions: music condition $\mathcal{C}$, sparse keyframe condition $\mathcal{X}^{ref}$, and beat condition $\mathcal{B}$. These three inputs are processed by their respective encoders to obtain embeddings, which are then passed to a hierarchical multi-condition fusion module to generate synthesized conditional features. Finally, a conditional diffusion denoising module generates the dance motion sequence. The diffusion module adopts a Transformer architecture similar to EDGE, directly predicting the clean sample $\mathbf{x}_0$.

### Key Design 1: Nearest Beat Distance Representation

Traditional binary beat masks are abandoned, and the beat condition is represented as a vector $b$, where each component $b^i$ represents the distance from the current frame to the nearest beat frame. This representation not only alleviates the sparsity issue but also provides the model with local temporal context information, helping to more accurately capture rhythmic features. Beat embeddings are obtained through an independent embedding layer and a Transformer encoder $\mathcal{E}_b$ obtained.

### Key Design 2: Hierarchical Multi-Condition Fusion Mechanism

The fusion of conditions with different levels of sparsity is performed in two stages:

- **First Stage (Sparse-to-Dense Fusion)**: Sparse keyframe conditions are injected into dense music and beat conditions through Beat-Aware Dilated Cross-Attention. The core technique is **Beat-Aware Mask Dilation**—keyframes closer to the beat frames obtain larger dilation steps $n = \lceil s \cdot e^{-2b^i/d^i} \rceil$, allowing keyframes near beats to influence more surrounding frames for beat-aware condition propagation.
- **Second Stage (Dense-to-Dense Fusion)**: The keyframe-enhanced music embeddings and beat embeddings are concatenated and fed into a Transformer fusion module to produce the final multi-condition features.

In practice, 6 sparse-dense fusion blocks (dilation steps $s$ = 4, 8, 12, 16, 20, 24) and 2 dense-dense fusion blocks are used.

### Key Design 3: Beat Alignment Loss

A beat distance estimator is pre-trained to provide explicit supervision on the beats of generated motions during training:

$$\mathcal{L}_{beat} = \sum_{i=1}^{L} w_s^i \cdot w_b^i \cdot \text{MSE}(b^i, \hat{b}^i)$$

where $w_s^i$ is an adaptive weight that increases the penalty in regions with poor beat alignment; $w_b^i = e^{-2b^i/d^i}$ applies stronger supervision to frames closer to the beats. This ensures that the physical motion beats of the generated dance align precisely with the given beat conditions.

## Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{simple} + \lambda_{kin}\mathcal{L}_{kin} + \lambda_{beat}\mathcal{L}_{beat}$, where:

- $\mathcal{L}_{simple}$: Base diffusion reconstruction loss.
- $\mathcal{L}_{kin}$: Kinematic loss (joint positions + velocity + foot contact consistency + acceleration), with weight $\lambda_{kin}=1$.
- $\mathcal{L}_{beat}$: Beat alignment loss, with weight $\lambda_{beat}=0.5$.

During training, 1%-30% of the ground truth (GT) frames are randomly sampled as keyframe conditions, and the GT motion beats are used as beat conditions. Cosine noise scheduling is adopted with 1000 diffusion steps, using the Adam optimizer ($lr=2e-4$), a batch size of 64, and 4 RTX3090 GPUs.

## Key Experimental Results

The dataset is AIST++ (1408 dance sequences, 10 street dance genres), cropped into 5-second segments (30fps) with 2.5-second overlap.

### Table 1: Quantitative Comparison with Existing Methods on AIST++

| Method | PFC ↓ | BAS ↑ | Div_k → | Div_m → | KPD ↓ | BAP ↑ |
|------|-------|-------|---------|---------|-------|-------|
| Ground Truth | 1.338 | 0.384 | 9.773 | 7.212 | - | - |
| FACT | 2.698 | 0.202 | 9.704 | 7.342 | - | - |
| Bailando | 1.578 | 0.215 | 9.622 | 7.175 | - | - |
| EDGE (keyframes) | 1.084 | 0.235 | 9.743 | 7.274 | 0.859 | - |
| **Beat-It (Ours)** | **0.966** | **0.661** | **9.660** | **7.248** | **0.306** | **0.793** |

Beat-It improves BAS by 0.426 and KPD by 0.553 compared to EDGE, significantly leading in both beat synchronization and keyframe controllability. PFC also achieves the best performance, with kinematic quality outperforming all other methods (even better than GT).

### Table 2: Ablation Study

| Variant | PFC ↓ | BAS ↑ | KPD ↓ | BAP ↑ |
|------|-------|-------|-------|-------|
| w/o HF (without hierarchical fusion) | 25.626 | 0.322 | 0.477 | 0.323 |
| w/o BD (without beat-aware dilation) | 1.632 | 0.358 | 0.389 | 0.371 |
| w/o $\mathcal{L}_{beat}$ | 1.342 | 0.397 | 0.343 | 0.411 |
| **Full Model** | **0.966** | **0.661** | **0.306** | **0.793** |

All three core components are indispensable. Removing hierarchical fusion causes PFC to surge to 25.6, representing quality collapse; removing the beat alignment loss reduces BAP from 0.793 to 0.411.

User study (18 participants evaluating 20 dance sequences): Beat-It achieves a win rate of 92.2% against FACT, 78.8% against Bailando, and 60.3% / 86.9% against EDGE in terms of quality and controllability, respectively.

## Highlights & Insights

- **First to achieve beat-controllable + keyframe-guided dance generation**, filling a gap in this direction.
- **Nearest beat distance representation** is clean and effective, converting sparse binary beat signals into active continuous representations containing temporal contexts.
- **Hierarchical multi-condition fusion** cleverly resolves conflicts between conditions with different levels of sparsity, where beat-aware dilation is a key design highlight.
- **Beat alignment loss** provides explicit supervision via a pre-trained estimator, significantly boosting BAS (0.235 to 0.661).
- Supports arbitrary beat specification and flexible keyframe allocation, showing high value for practical choreography applications.

## Limitations & Future Work

- Validated only on the AIST++ dataset, which is dominated by street dance; its generalization to other dance styles (such as ballet and folk dance) remains unknown.
- The beat distance estimator requires additional pre-training, increasing the complexity of the pipeline.
- The paper does not discuss the long-term consistency and transition naturalness of the generated dances.
- Direct comparison with non-open-source methods like DanceFormer is lacking.
- The automatic keyframe selection strategy is not discussed in depth; random sampling of 10% during testing might not be the optimal strategy.

## Related Work & Insights

- **Single-condition dance generation**: FACT (Full-Attention Cross-Modal Transformer), Bailando (VQ-VAE + Motion GPT), DanceFormer (two-stage deterministic framework).
- **Multi-condition dance generation**: EDGE (diffusion model + temporal/spatial hybrid keyframe control), LDA (style label assistance), TM2D (music + text bimodal).
- **Multi-condition diffusion generation**: ControlNet, T2I-Adapter, Uni-ControlNet, and other works in the image domain; Beat-It brings multi-condition fusion concepts to dance generation.
- **Keyframe control**: Yang et al. (normalizing flows + temporal embedding), DiffKFC (dilated attention, on which Beat-It builds by adding beat awareness).

## Rating

- Novelty: ⭐⭐⭐⭐ — The combined design of beat decoupling, hierarchical fusion, and beat-aware dilation is novel, addressing beat-controllable dance generation for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quantitative evaluation, ablations, and user study are comprehensive, but limited to a single dataset and lacking comparisons with some methods.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, detailed method descriptions, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Represents a substantial advancement in the controllability of dance generation, with the beat distance representation and hierarchical fusion ideas extensible to other sequential generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](../../NeurIPS2025/audio_speech/megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)
- [\[ICML 2026\] Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer](../../ICML2026/audio_speech/towards_streaming_synchronized_spatial_audio_generation_via_autoregressive_diffu.md)
- [\[CVPR 2025\] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition](../../CVPR2025/audio_speech/synchronized_video-to-audio_generation_via_mel_quantization-continuum_decomposit.md)
- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[CVPR 2025\] Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model](../../CVPR2025/audio_speech/enhancing_dance-to-music_generation_via_negative_conditioning_latent_diffusion_m.md)

</div>

<!-- RELATED:END -->
