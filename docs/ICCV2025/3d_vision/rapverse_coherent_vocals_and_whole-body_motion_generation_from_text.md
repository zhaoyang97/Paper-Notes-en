---
title: >-
  [Paper Note] RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text
description: >-
  [3D Vision] This work constructs the large-scale rap dataset RapVerse and proposes a unified autoregressive transformer framework that, for the first time, simultaneously generates coherent singing vocals and whole-body 3D motion from lyric text.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: a4c9e3715210f87d
---

# RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2405.20336](https://arxiv.org/abs/2405.20336)
- **Code**: [Project Page](https://jiabenchen.github.io/RapVerse/)
- **Area**: 3D Vision
- **Keywords**: Text-to-Motion, Singing Voice Synthesis, Whole-Body Motion Generation, Multimodal Generation, Autoregressive Models

## TL;DR

This work constructs the large-scale rap dataset RapVerse and proposes a unified autoregressive transformer framework that, for the first time, simultaneously generates coherent singing vocals and whole-body 3D motion from lyric text.

## Background & Motivation

Multimodal content generation has achieved remarkable progress in individual modalities (text-to-music, text-to-speech, text-to-motion, audio-to-motion), yet these efforts operate in isolation, handling each modality independently. However:

**Intrinsic coupling of voice and motion**: Psychological evidence indicates that human voice and motion generation are highly correlated and coupled. A unified generation system enables richer emotional expression, where one modality can guide and assist the other.

**Cascading error accumulation**: Pipeline approaches of text→speech + speech→motion accumulate errors at each stage. For instance, errors in speech synthesis propagate to inaccurate facial expressions and lip synchronization. Moreover, multiple independently trained and inferred models incur substantial computational overhead.

**Absence of datasets**: No dataset simultaneously contains lyrics, singing vocals, and 3D whole-body motion annotations. Existing singing voice datasets are predominantly Chinese or small in scale; motion datasets contain only textual descriptions without audio; speech-motion datasets focus on speech rather than singing.

## Method

### RapVerse Dataset

The dataset is divided into two subsets:

**Rap-Vocal Subset** (108.44 hours): High-quality English rap vocals from 32 singers, without background music. Pipeline: crawling via Spotdl/Spotipy → vocal separation via Spleeter → loudness normalization → lyric cleaning and timestamp alignment → segmentation into 10–20 second clips.

**Rap-Motion Subset** (26.8 hours): Rap performance videos with SMPL-X 3D whole-body mesh annotations, singing vocals, and lyrics. Pipeline: YouTube crawling → YOLO human detection + RAFT motion magnitude filtering → SMPL-X parameter extraction via the Motion-X pipeline. Motion representation $\mathcal{M} = \{\mathcal{M}_f, \mathcal{M}_b, \mathcal{M}_h, \zeta, \epsilon\}$, corresponding to jaw pose, body pose, hand pose, facial expression, and global translation, respectively.

### Motion VQ-VAE Tokenizer

A compositional VQ-VAE is adopted, constructing three independent tokenizers for the face, body, and hands, which encode continuous motion sequences into discrete tokens. Quantized codebook lookup:

$$Q(z^{\mathcal{M}}; \mathcal{C}^{\mathcal{M}}) = \arg\min_k \|z^{\mathcal{M}} - c_k^{\mathcal{M}}\|_2$$

The design of three independent tokenizers avoids insufficient expressiveness of a single VQ-VAE for whole-body dynamics, particularly for facial expressions—facial movements dominate in singing performances, and a single VQ-VAE would compromise facial detail.

### Vocal2unit Audio Tokenizer

Built upon a self-supervised speech resynthesis framework, comprising three encoders and a vocoder:

- **Semantic Encoder**: Pretrained HuBERT + K-means clustering to generate discrete semantic units $z^{\mathcal{S}} = \{z_i^{\mathcal{S}}\}_{i=1}^{L_s}$
- **F0 Encoder**: YAAPT algorithm extracts fundamental frequency → VQ-VAE encodes into discrete pitch tokens $z^{\mathcal{P}}$
- **Singer Encoder**: Extracts a 256-dimensional singer embedding $z^{\mathcal{I}}$ from mel-spectrograms, conditioned solely on singer identity
- **Vocoder**: A modified HiFi-GAN that reconstructs waveforms from semantic tokens, pitch tokens, and singer embeddings

### Autoregressive Unified Generation

Lyrics are converted to text tokens via T5-Tokenizer, and all three modalities are unified in token space. A large Text-Motion-Audio foundation model (decoder-only temporal transformer) performs next-token prediction:

$$-\sum_{i=1}^{\mathcal{N}^{\mathcal{X}}} \log p_\delta(t_i^{\mathcal{X}}; h^{\mathcal{L}})$$

**Multimodal Token Organization**: Tokens are arranged in an interleaved fashion. Motion tokens: $\mathcal{T}^{\mathcal{M}} = \{t_1^{\mathcal{M}_f}, t_1^{\mathcal{M}_b}, t_1^{\mathcal{M}_h}, t_2^{\mathcal{M}_f}, \ldots\}$; audio tokens: $\mathcal{T}^{\mathcal{V}} = \{t_1^{\mathcal{V}_h}, t_1^{\mathcal{V}_p}, t_2^{\mathcal{V}_h}, \ldots\}$.

Vocal tokens are placed before motion tokens because: (1) vocals are directly conditioned on lyrics; (2) motion tokens (e.g., lip movements) are conditioned on the previously generated vocals.

## Key Experimental Results

### Main Results — Motion Generation

| Method | FID↓ | DIV↑ | BC↑ | MSE↓ | LVD↓ |
|--------|------|------|-----|------|------|
| T2M-GPT (Text→Motion) | 23.45 | 11.75 | — | — | — |
| MLD (Text→Motion) | 26.34 | 12.15 | — | — | — |
| Talkshow (Audio→Motion) | 18.23 | 13.14 | 0.482 | 2.05 | 9.20 |
| EMAGE (Audio→Motion) | 21.18 | 12.65 | 0.488 | 1.96 | 8.45 |
| MotionCraft (Audio→Motion) | 17.75 | 13.76 | 0.482 | 2.06 | 9.23 |
| Cascaded System | 23.42 | 12.87 | 0.479 | 2.09 | 9.38 |
| **Ours (Text→Audio+Motion)** | **17.58** | **14.08** | **0.485** | **2.03** | **7.23** |

### Main Results — Singing Voice Generation

| Method | MOS↑ |
|--------|------|
| Ground Truth | 4.45 ± 0.06 |
| DiffSinger | 3.72 ± 0.12 |
| **Ours** | **3.64 ± 0.15** |
| FastSpeech2 | 3.41 ± 0.18 |

### Ablation Study

| Configuration | FID↓ | DIV↑ | LVD↓ |
|---------------|------|------|------|
| Pretrained LLM | 48.25 | 12.65 | 12.15 |
| Single Motion Token | 19.15 | 12.75 | 10.12 |
| **Full Method** | **17.58** | **14.08** | **7.23** |

### Key Findings

1. **Unified generation outperforms cascading**: Joint generation surpasses the DiffSinger + Talkshow cascaded system on FID, DIV, and BC, while incurring lower computational overhead.
2. **Vocal quality is competitive with specialized systems**: Despite the added complexity of simultaneously generating two modalities, the vocal quality MOS remains close to the dedicated system DiffSinger.
3. **Pretrained LLMs are ill-suited for multimodal tokens**: Models pretrained on language tokens transfer poorly to singing and motion tokens (FID 48.25 vs. 17.58).
4. **Compositional VQ-VAE is critical**: A single VQ-VAE degrades LVD from 7.23 to 10.12, with a severe drop in facial expression quality.

## Highlights & Insights

1. **Pioneering task definition**: This is the first work to simultaneously generate singing vocals and whole-body 3D motion from lyric text, with substantial practical application value.
2. **Carefully constructed dataset**: RapVerse is the first English dataset containing lyrics, singing vocals, and whole-body motion simultaneously (108 hours of vocals + 27 hours of motion).
3. **Elegant modality unification**: Interleaved token arrangement combined with a unified autoregressive model covers all modalities in a simple yet effective manner.
4. **Compositional tokenization strategy**: Independent VQ-VAEs for face, body, and hands balance the representational demands of different body parts.

## Limitations & Future Work

1. The strong stylistic specificity of rap limits verified generalization to other music genres (e.g., folk, classical).
2. 3D motion is derived from monocular video estimation (Motion-X pipeline), yielding lower accuracy than motion capture data.
3. Only English lyric input is supported.

## Related Work & Insights

- **Text-to-Singing**: DiffSinger, FastSpeech2, StyleSinger
- **Text-to-Motion**: T2M-GPT, MLD, MotionGPT, HumanTomato
- **Audio-to-Motion**: TalkSHOW, EMAGE, MotionCraft
- **VQ-VAE Discretization**: Motion quantization, HuBERT semantic encoding

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Novel task definition + first dataset + unified framework
- **Technical Depth**: ⭐⭐⭐⭐ — Multimodal tokenization and unified modeling are well designed
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dimensional metrics + user study + complete ablations
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with thorough dataset description

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](../../CVPR2025/3d_vision/dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)

</div>

<!-- RELATED:END -->
