---
title: >-
  [Paper Note] LeVo: High-Quality Song Generation with Multi-Preference Alignment
description: >-
  [Audio & Speech] LeVo proposes a language-model-based song generation framework that simultaneously optimizes vocal–accompaniment harmony and audio quality by predicting mixed tokens and dual-track tokens in parallel…
tags:
  - "Audio & Speech"
date: 2026-05-08
content_hash: 240e556fb4d906e0
---

# LeVo: High-Quality Song Generation with Multi-Preference Alignment

## Basic Information

| Item | Content |
|------|------|
| Title | LeVo: High-Quality Song Generation with Multi-Preference Alignment |
| Authors | Shun Lei, Yaoxun Xu, Zhiwei Lin, Huaicheng Zhang, Wei Tan, Hangting Chen, Jianwei Yu, Yixuan Zhang, Chenyu Yang, Haina Zhu, Shuai Wang, Zhiyong Wu, Dong Yu |
| Affiliations | Tsinghua Shenzhen International Graduate School, Tencent AI Lab, Wuhan University, Shanghai Jiao Tong University, Nanjing University, et al. |
| Conference | NeurIPS 2025 |
| arXiv | [2506.07520](https://arxiv.org/abs/2506.07520) |
| Code | [GitHub](https://github.com/tencent-ailab/songgeneration) |

## TL;DR

LeVo proposes a language-model-based song generation framework that simultaneously optimizes vocal–accompaniment harmony and audio quality by predicting mixed tokens and dual-track tokens in parallel, and introduces a DPO-based multi-preference alignment method to enhance musicality and instruction-following ability. LeVo comprehensively outperforms all academic baselines and approaches the performance of industrial systems.

## Background & Motivation

Song generation is one of the most challenging tasks in AIGC, requiring simultaneous generation of high-quality vocal and accompaniment tracks, seamless fusion of the two, and maintenance of both musicality and instruction-following capability. Existing methods face the following core difficulties:

1. **Limitations of mixed-token methods**: Methods such as Jukebox and SongCreator treat the mixed audio of vocals and accompaniment as a single prediction target. The limited vocabulary fails to capture the complex combinations of vocals and accompaniment, resulting in low audio quality.
2. **Difficulties with dual-track token methods**: Methods such as YuE and SongGen generate vocal and accompaniment tokens separately. While audio quality improves, independent prediction struggles to maintain vocal–accompaniment harmony; interleaved prediction patterns substantially increase sequence length, limiting scalability.
3. **Data quality issues**: Available song datasets vary widely in quality, musical annotations are unreliable, models lack prior knowledge of musicality, and accurately following instructions such as lyrics and prompt words remains difficult.

## Method

### Overall Architecture

LeVo consists of two main components: **LeLM** (Language Model) and **Music Codec**.

### LeLM: Parallel Modeling of Mixed Tokens and Dual-Track Tokens

The core innovation of LeLM lies in simultaneously modeling two types of tokens:
- **Mixed Tokens**: Encode the mixed audio of vocals and accompaniment, capturing high-level structural information such as melody, rhythm, and tempo to ensure vocal–accompaniment harmony.
- **Dual-Track Tokens**: Encode vocals and accompaniment separately, capturing finer acoustic details to improve audio quality.

Architecturally, LeLM comprises:
1. **Language Model**: A decoder-only Transformer that performs next-token prediction on mixed tokens.
2. **AR Decoder**: A decoder-only Transformer with substantially fewer parameters than the main language model, which predicts dual-track tokens in parallel conditioned on the hidden states of the language model. A delay pattern is introduced so that dual-track token prediction can leverage mixed-token information from $k$ future steps, providing richer context.

### Music Codec

A 48 kHz music codec built on MuCodec:
- **Encoder**: MuEncoder + RVQ, discretizing audio into tokens.
- **Decoder**: Diffusion Transformer + VAE decoder, reconstructing high-fidelity audio from token embeddings significantly faster than mel-spectrogram-based methods.

### DPO-Based Multi-Preference Alignment

Three preference data construction strategies are proposed to address the multi-dimensional requirements of music generation:

1. **Lyric Alignment Preference (Strategy 1)**: Phoneme error rates are computed via ASR; sample pairs with error count differences greater than 40 are selected.
2. **Prompt Consistency Preference (Strategy 2)**: The MuQ-MuLan model computes similarity scores, and threshold-based filtering is applied to select winning and losing pairs.
3. **Musicality Preference (Strategy 3)**: A three-stage pipeline—crowdsourced human ranking → reward model training → large-scale filtering—ultimately yielding approximately 60,000 preference pairs.

An interpolation-based multi-preference alignment approach (inspired by DNI) is adopted: three sets of parameters are obtained by fine-tuning separately on each of the three preference datasets, and the final model is produced by linear interpolation, achieving a balanced improvement across all dimensions.

### Three-Stage Training Paradigm

1. **Pre-training**: The language model learns mixed-token prediction on large-scale music data; the AR decoder is frozen.
2. **Modular Extension Training**: The first-stage modules are frozen while the AR decoder is trained to learn dual-track tokens, avoiding interference with previously acquired knowledge.
3. **Multi-Preference Alignment**: The entire LeLM is fine-tuned using the DPO loss.

## Key Experimental Results

### Experimental Setup

- Training data: 2 million songs (approximately 110,000 hours)
- LeLM parameters: approximately 2B; diffusion model: approximately 700M; VAE: 150M
- Comparison systems: industrial systems (Suno V4.5, Mureka-O1, Haimian) + academic systems (YuE, DiffRhythm, ACE-Step, SongGen)

### Objective Evaluation Results

| Model | FAD ↓ | MuQ-T ↑ | MuQ-A ↑ | PER ↓ | CE ↑ | CU ↑ | PC ↑ | PQ ↑ |
|------|-------|---------|---------|-------|------|------|------|------|
| Suno-V4.5 | 2.59 | **0.34** | 0.84 | 21.6 | 7.65 | 7.86 | 5.94 | 8.35 |
| Mureka-O1 | **2.50** | 0.33 | **0.87** | **7.2** | 7.71 | 7.83 | **6.39** | 8.44 |
| YuE | 2.65 | 0.27 | 0.74 | 36.4 | 7.13 | 7.39 | 5.90 | 7.77 |
| DiffRhythm | 4.86 | 0.26 | 0.51 | 12.3 | 6.65 | 7.32 | 5.71 | 7.77 |
| ACE-Step | 2.69 | 0.28 | - | 37.1 | 7.37 | 7.52 | 6.26 | 7.85 |
| SongGen* | 2.68 | 0.25 | 0.80 | 27.5 | 7.63 | 7.79 | 5.94 | 8.37 |
| **LeVo** | 2.68 | **0.34** | 0.83 | **7.2** | **7.78** | **7.90** | 6.03 | **8.46** |

LeVo achieves the best or tied-best performance on five metrics—MuQ-T, PER, CE, CU, and PQ—with instruction-following ability and musicality perception significantly superior to all academic methods.

### Subjective Evaluation Results (MOS)

| Model | OVL ↑ | MEL ↑ | HAM ↑ | SSC ↑ | AQ ↑ | LYC ↑ |
|------|-------|-------|-------|-------|------|-------|
| Suno-V4.5 | **3.59** | **4.10** | **3.93** | **4.19** | **4.00** | 3.17 |
| Mureka-O1 | 3.42 | 3.88 | 3.89 | 4.14 | 3.87 | 3.32 |
| YuE | 2.45 | 3.04 | 2.94 | 3.53 | 3.08 | 2.41 |
| SongGen* | 2.91 | 3.43 | 3.44 | 3.66 | 3.69 | 2.84 |
| **LeVo** | 3.42 | 3.93 | 3.90 | 4.09 | 3.96 | **3.38** |

LeVo comprehensively surpasses all academic methods, with overall quality approaching Suno-V4.5, and outperforms Suno by 0.21 points on the lyric alignment (LYC) dimension.

## Highlights & Insights

1. **Parallel dual-type token modeling**: Mixed tokens are leveraged to maintain harmony while dual-track tokens improve audio quality; the AR decoder with delay pattern enables interference-free parallel prediction of both token types, avoiding the sequence-length explosion associated with interleaved patterns.
2. **Modular extension training**: The stage-wise freeze-and-train strategy effectively prevents mutual interference between knowledge acquired at different stages, yielding a clean and effective design.
3. **Multi-preference DPO alignment**: This work represents the first application of multi-preference DPO to song generation. Three targeted preference strategies cover lyric alignment, prompt consistency, and musicality; the interpolation-based fusion outperforms naive mixture training.
4. **New benchmark for academic methods**: LeVo comprehensively surpasses all existing open-source methods on both objective and subjective metrics, with several indicators approaching or even exceeding closed-source industrial systems.

## Limitations & Future Work

1. **Audio quality remains constrained by discrete tokens and training data quality**: A gap with state-of-the-art industrial models (e.g., Suno) persists.
2. **Reliance on pseudo-labels for annotation**: Text descriptions are generated by Qwen2-Audio, limiting their diversity and richness; error accumulation in lyrics extraction and structure recognition pipelines degrades instruction-following precision.
3. **Insufficient song structure modeling**: LeVo still lags behind Suno and Mureka-O1 on the SSC (song structure clarity) dimension.
4. **Ethical risks**: The system's style transfer and end-to-end generation capabilities may be misused for deepfake audio or misinformation creation.

## Related Work & Insights

- **Music generation**: MusicGen, MusicLM, AudioLDM 2, and others achieve end-to-end generation via language models or diffusion models; MeLoDy combines language models with diffusion models to achieve state-of-the-art performance.
- **Song generation**: Jukebox establishes the paradigm of predicting discrete music codes with language models; YuE and SongGen explore dual-track token strategies; DiffRhythm adopts a diffusion-based approach; industrial systems (Suno, Mureka, Udio) demonstrate strong capabilities but have not disclosed technical details.
- **RL/preference alignment in music**: BATON integrates a reward model into the diffusion loss; MusicRL fine-tunes MusicLM via RLHF; Tango2 constructs preference datasets semi-automatically for DPO training. LeVo is the first to achieve multi-dimensional preference alignment for song generation.

## Rating

| Dimension | Score (1–10) | Notes |
|------|-------------|------|
| Novelty | 8 | The combination of parallel dual-type token modeling and multi-preference DPO alignment is significantly novel |
| Technical Depth | 8 | Three-stage training, AR decoder delay pattern, and interpolation-based fusion are all well-grounded designs |
| Experimental Thoroughness | 9 | Dual objective and subjective evaluation, 7 comparison systems, and comprehensive ablation studies |
| Writing Quality | 8 | Clear structure, well-motivated problem statement, thorough experimental analysis |
| Value | 7 | Code is open-sourced, but the 2B-parameter model and 110,000-hour data requirement set a high barrier |
| Overall | 8 | A strong contribution to song generation that systematically addresses multiple core challenges with convincing experiments |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[NeurIPS 2025\] Segment-Factorized Full-Song Generation on Symbolic Piano Music](segment-factorized_full-song_generation_on_symbolic_piano_music.md)
- [\[NeurIPS 2025\] A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)

</div>

<!-- RELATED:END -->
