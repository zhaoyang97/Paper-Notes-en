---
title: >-
  [Paper Note] MACS: Multi-source Audio-to-Image Generation with Contextual Significance and Semantic Alignment
description: >-
  [AAAI 2026][Image Generation][Audio-to-image generation] This paper proposes MACS, the first two-stage framework that explicitly separates multi-source audio prior to image generation. The framework combines weakly super…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Audio-to-image generation"
  - "multi-source audio"
  - "sound source separation"
  - "cross-modal alignment"
  - "diffusion models"
date: 2026-05-08
content_hash: 9893b5d82958a1d8
---

# MACS: Multi-source Audio-to-Image Generation with Contextual Significance and Semantic Alignment

**Conference**: AAAI 2026
**arXiv**: [2503.10287](https://arxiv.org/abs/2503.10287)
**Code**: [https://github.com/alxzzhou/MACS](https://github.com/alxzzhou/MACS)
**Area**: Image Generation
**Keywords**: Audio-to-image generation, multi-source audio, sound source separation, cross-modal alignment, diffusion models

## TL;DR

This paper proposes MACS, the first two-stage framework that explicitly separates multi-source audio prior to image generation. The framework combines weakly supervised sound source separation, CLAP-space semantic alignment (via ranking loss and contrastive loss), and decoupled cross-attention diffusion generation, achieving comprehensive state-of-the-art performance on multi-source, mixed-source, and single-source audio-to-image generation tasks.

## Background & Motivation

Audio-to-image generation is a cross-modal task that transforms auditory signals into visual representations. Leveraging advances in diffusion models and multimodal learning, the field has made significant progress in recent years. However, existing methods suffer from fundamental limitations:

**Focus solely on single-source audio**: Existing methods (AudioToken, Sound2Scene, TempoTokens) all assume single-source audio input.

**Natural soundscapes are multi-source mixtures**: Real-world audio typically comprises multiple overlapping sources (e.g., guitar and speech). Directly mapping mixed audio to images leads to semantic incoherence.

**Absence of multi-source benchmarks**: No dedicated evaluation datasets exist for multi-source audio-to-image generation.

The core philosophy is **"separation before generation"**: rather than generating images directly from complex mixed audio, the approach first decomposes the mixture into individual sources, then synthesizes an image from the combined semantics of each source.

This strategy faces three main challenges:
- How to robustly separate overlapping sound sources?
- How to preserve semantic consistency and relative importance of each separated source?
- How to effectively integrate multiple audio conditions within a diffusion model?

## Method

### Overall Architecture

MACS is a two-stage framework:

- **Stage 1 (Multi-Source Separation, MSS)**: Decomposes mixed audio into multiple sub-audio signals via reconstruction loss training, with contrastive and ranking losses in CLAP space for semantic alignment and contextual significance modeling.
- **Stage 2 (Diffusion Generation)**: Maps multiple audio embeddings to a single generated image using a decoupled cross-attention module.

### Key Designs

#### 1. **Multi-Source Separation (MSS)**

**Problem formulation**: Given mixed audio $\mathbf{m}$, a UNet $\mathcal{U}_\theta$ predicts $M$ binary masks on the spectrogram. Waveforms are recovered by applying masks to the magnitude spectrum followed by iSTFT:

$$\mathcal{G}_\theta(\mathbf{m}) = \mathcal{T}^{-1}\left(|\mathcal{T}(\mathbf{m})| \odot \mathcal{U}_\theta(|\mathcal{T}(\mathbf{m})|), \phi(\mathcal{T}(\mathbf{m}))\right)$$

**Unsupervised MoM (Mixture of Mixtures) training**:

Inspired by MixIT, two mixed audio signals $\bm{m}_1, \bm{m}_2$ are further combined into $\bm{m} = \bm{m}_1 + \bm{m}_2$. After separating $M$ signals, an optimal binary assignment is searched to minimize the reconstruction loss:

$$\mathcal{L}_{Rec} = \min_{(\Lambda_1, \Lambda_2) \in \Lambda} \left[\mathcal{L}_{SISDR}(\bm{m}_1, \sum_{i \in \Lambda_1}\bm{s}_i) + \mathcal{L}_{SISDR}(\bm{m}_2, \sum_{i \in \Lambda_2}\bm{s}_i)\right]$$

where the SI-SDR loss is:

$$\mathcal{L}_{SISDR}(\bm{m}_j, \hat{\bm{s}}_j) = -10\log_{10}\frac{\|\alpha\bm{m}_j\|_2^2}{\|\alpha\bm{m}_j - \hat{\bm{s}}_j\|_2^2}$$

Key advantage: Completely **unconditional** — the UNet requires no auxiliary input, making it more suitable for real-world audio than traditional supervised methods.

#### 2. **Audio-Text Semantic Alignment**

A pretrained CLAP model projects separated audio and text labels into a shared embedding space, pursuing two alignment objectives:

**Ranking Loss** — modeling contextual significance:

$$\mathcal{L}_{Rank} = 1 - r_s(\mathbf{S}, \text{Sorted}(\mathbf{S}))$$

where $r_s$ is the Spearman rank correlation coefficient and $\mathbf{S}$ is the cosine similarity vector between separated audio embeddings and the original mixed audio embedding. The intuition is that separated audio more similar to the original mixture should be ranked higher, as it encodes more semantically significant information. Differentiable sorting is used to ensure training feasibility.

**Contrastive Loss** — achieving semantic matching:

Audio embeddings are first soft-assigned to text embeddings:

$$\mathcal{E}'^T = \text{Softmax}\left(\frac{\langle\mathcal{E}^A\rangle\langle\mathcal{E}^T\rangle^\top}{\tau}\right)\mathcal{E}^T$$

A standard contrastive loss then pulls matched audio-text pairs together and pushes unmatched pairs apart:

$$\mathcal{L}_{CL} = -\frac{1}{2M}\sum_{i=1}^{M}\log\frac{\exp(W_{ii})}{\sum_j \exp(W_{ij})} - \frac{1}{2M}\sum_{i=1}^{M}\log\frac{\exp(W_{ii})}{\sum_j \exp(W_{ji})}$$

**Stage 1 overall training objective**:

$$\mathcal{L}_1 = \lambda\mathcal{L}_{Rec} + \mu\mathcal{L}_{CL} + \gamma\mathcal{L}_{Rank}$$

#### 3. **Multi-Source Audio Diffusion Generation**

A **Decoupled Cross-Attention** module is adopted to integrate multiple audio embeddings:

- $M$ audio embeddings are augmented with trainable positional encodings and projected via MLP: $\mathcal{E}'^A = \text{MLP}(\mathcal{E}^A + \mathcal{E}^{Pos})$
- Audio and text cross-attention are computed independently and summed: $\mathbf{H}' = \mathbf{H}_A + \mathbf{H}_T$
- The audio branch uses newly initialized $\mathbf{W}_k, \mathbf{W}_v$ while sharing $\mathbf{W}_q$

### Loss & Training

**Stage 2 training**: Only $\mathbf{W}_k, \mathbf{W}_v$, positional encodings, and the MLP are updated; the base diffusion model is frozen:

$$\mathcal{L}_2 = \mathbb{E}_{\mathbf{z},\epsilon,t}\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, c)\|_2^2$$

Training configuration: AdamW ($\beta_1=0.9, \beta_2=0.999$), batch size 16, single RTX 4090D GPU. Stage 1 is pretrained on FSD50K.

## Key Experimental Results

### Main Results

**Multi-source audio (LLP-multi)**:

| Method | FID↓ | CLIP-FID↓ | KID↓ | AIS↑ | AIS-z↑ | IIS↑ | IIS-z↑ |
|--------|------|-----------|------|------|--------|------|--------|
| AudioToken | 143.62 | 52.21 | 0.0431 | 0.0591 | 0.6201 | 0.4914 | 0.6799 |
| Sound2Scene | 105.14 | 33.79 | 0.0240 | 0.0711 | 0.8176 | 0.5545 | 0.7877 |
| CoDi | 116.67 | 44.96 | 0.0283 | 0.0747 | 1.1068 | 0.5179 | 1.4429 |
| **MACS** | **87.09** | **20.47** | **0.0157** | 0.0754 | **1.3038** | **0.6269** | **1.7231** |

**Single-source audio (Landscape)**:

| Method | FID↓ | CLIP-FID↓ | KID↓ | AIS↑ | IIS↑ | IIS-z↑ |
|--------|------|-----------|------|------|------|--------|
| CoDi | 158.31 | 39.97 | 0.0180 | 0.1094 | 0.6961 | 1.0942 |
| ImageBind | 207.93 | 41.49 | 0.0304 | 0.1189 | 0.6673 | 0.7681 |
| **MACS** | **147.23** | **26.91** | **0.0098** | 0.1015 | **0.7422** | **1.4805** |

MACS achieves the best performance on **17 out of 21** evaluation metrics, comprehensively surpassing prior SOTA.

### Ablation Study

**Separation model substitution (LLP-multi)**:

| Separation Model | FID↓ | CLIP-FID↓ | KID↓ | AIS↑ | IIS↑ |
|-----------------|------|-----------|------|------|------|
| MixIT (waveform-level) | 98.73 | 28.42 | 0.0201 | 0.0688 | 0.5782 |
| **MACS (spectrogram-level)** | **87.09** | **20.47** | **0.0157** | **0.0754** | **0.6269** |

**Adaptability of MSS**: Connecting MACS Stage 1 output to AudioToken (AudioToken w/MSS) outperforms the original AudioToken across all datasets, demonstrating the generalizability of the MSS module.

**Effect of pretraining**: Pretraining on FSD50K substantially improves separation quality (measured by the standard deviation of cosine similarity between separated audio and text labels). Pretraining alone is sufficient for image generation quality; fine-tuning is not strictly necessary.

### Key Findings

1. **"Separation before generation" is effective**: Separated audio embeddings produce more localized, semantically aligned attention maps.
2. **Ranking loss identifies salient sources**: Higher-ranked embeddings encode more semantically important information; the top 3 embeddings already capture most of the semantics.
3. **Audio is interpolable**: Linear interpolation between two audio embeddings produces smooth semantic transitions (e.g., dog barking → car engine).
4. **Separation improves single-source quality**: Even with single-source input, the separation process reduces noise and improves quality.
5. **Grad-CAM visualizations** demonstrate clear correspondence between separated audio and image regions.

## Highlights & Insights

- **Strong novelty**: The first work to explicitly perform multi-source audio separation prior to image generation.
- **Conceptually clear design**: The "separation before generation" paradigm is simple yet effective, with targeted solutions for each of the three identified challenges.
- **Clever exploitation of CLAP space**: Knowledge from pretrained large models is transferred as semantic supervision for audio separation.
- **Innovation in ranking loss**: Spearman rank correlation is used to model contextual significance, resolving the permutation ambiguity inherent in separation outputs.
- **Strong generalizability**: The MSS module is plug-and-play compatible with other audio-to-image generation methods.

## Limitations & Future Work

- Dataset scale is limited (LLP-multi contains only 6,595 frames); validation at larger scale remains to be conducted.
- The default setting of $M=6$ separation channels leaves the robustness to varying numbers of sources insufficiently analyzed.
- Only Stable Diffusion v1.5 is used as the backbone; stronger diffusion models (e.g., SDXL, SD3) are not explored.
- The quality of the pretrained CLAP model imposes an upper bound on semantic alignment performance.
- Separation quality may be limited for highly overlapping sources with similar semantics (e.g., multiple speakers).

## Related Work & Insights

This work integrates three research directions:
- **Sound source separation**: From supervised → unsupervised (MixIT) → weakly supervised (CLAP-guided, as in this work).
- **Multimodal contrastive pretraining**: The progression from CLIP → AudioCLIP → CLAP.
- **Audio-conditioned image generation**: From GANs (Wav2Pix) → diffusion models (AudioToken, ImageBind, CoDi).

Key insight: In cross-modal generation tasks, **explicit signal decomposition combined with semantic alignment** is more effective than end-to-end direct mapping.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First framework for multi-source audio separation + image generation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three dataset types + 5-fold cross-validation + extensive ablations and visualizations)
- Writing Quality: ⭐⭐⭐⭐ (Detailed method description with complete formalization)
- Value: ⭐⭐⭐⭐ (Opens a new direction in multi-source audio-to-image generation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cinematic Audio Source Separation Using Visual Cues](../../CVPR2026/image_generation/cinematic_audio_source_separation_using_visual_cues.md)
- [\[AAAI 2026\] Multi-Metric Preference Alignment for Generative Speech Restoration](multi-metric_preference_alignment_for_generative_speech_restoration.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[AAAI 2026\] Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback](playmate2_training-free_multi-character_audio-driven_animation_via_diffusion_tra.md)
- [\[AAAI 2026\] FreeInpaint: Tuning-free Prompt Alignment and Visual Rationality Enhancement in Image Inpainting](freeinpaint_tuning-free_prompt_alignment_and_visual_rationality_enhancement_in_i.md)

</div>

<!-- RELATED:END -->
