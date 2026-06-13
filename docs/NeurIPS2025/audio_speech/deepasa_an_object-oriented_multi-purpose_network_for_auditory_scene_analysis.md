---
title: >-
  [Paper Note] DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis
description: >-
  [NeurIPS 2025][Audio & Speech][auditory scene analysis] This paper proposes DeepASA, an object-oriented multi-task unified architecture that simultaneously performs multi-channel source separation (MIMO), dereverberation…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "auditory scene analysis"
  - "source separation"
  - "sound event detection"
  - "direction-of-arrival estimation"
  - "multi-task learning"
date: 2026-05-08
content_hash: 98f04de37d2ade5f
---

# DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis

**Conference**: NeurIPS 2025
**arXiv**: [2509.17247](https://arxiv.org/abs/2509.17247)  
**Code**: [HuggingFace Demo](https://huggingface.co/spaces/donghoney22/DeepASA)  
**Area**: Audio & Speech
**Keywords**: auditory scene analysis, source separation, sound event detection, direction-of-arrival estimation, multi-task learning

## TL;DR

This paper proposes DeepASA, an object-oriented multi-task unified architecture that simultaneously performs multi-channel source separation (MIMO), dereverberation, sound event detection (SED), audio classification, and direction-of-arrival estimation (DoAE) within a single model via object-oriented processing and a chain-of-inference mechanism, achieving state-of-the-art performance on multiple spatial audio benchmarks.

## Background & Motivation

The human auditory system decomposes complex acoustic scenes into independent perceptual streams (auditory streams) by integrating multiple cues such as pitch, temporal structure, and spatial location. Existing deep learning approaches, however, are typically designed for a single task (e.g., separation only or detection only) and lack the capacity for cross-task and cross-cue relational reasoning. Single-task models tend to fail when key auditory cues are absent or degraded.

Recent work has shown that combining multiple auditory cues substantially improves performance: target sound extraction (TSE) outperforms universal sound separation (USS) by exploiting category, activation, and spatial cues; similarly, the SELD task benefits from jointly modeling SED and DoAE through complementary information. These observations motivate the construction of a general ASA model that separates object-level auditory streams at an early stage and completes multiple downstream tasks by estimating complementary relationships among cues.

## Core Problem

1. **Parameter-association ambiguity**: In conventional track-wise processing, information from different sources may be aligned on the same track, making the pairing between SED and DoA outputs ambiguous.
2. **Cascading failure from early separation**: Separating source objects prematurely at the feature level may cause downstream ASA tasks to fail in cascade if the initial separation quality is poor.
3. **Cross-task inconsistency**: Parameters estimated independently by each sub-task (e.g., activity timing, direction of arrival) may exhibit temporal misalignment with one another.

## Method

### Overall Architecture

DeepASA takes a multi-channel audio mixture $\mathbf{x} \in \mathbb{R}^{M \times N}$ ($M$ microphones, $N$ time-domain samples) and models it as the sum of $J$ reverberant foreground sources and background noise. The architecture comprises three main components:

1. **Audio Encoder**: Extracts base-level features.
2. **Object Separator**: Decomposes features into $J+1$ object features ($J$ foreground sources + 1 noise object).
3. **Sub-decoders**: Estimate various auditory parameters from each object feature.

### Object-Oriented Processing (OOP)

The core idea is to separate each sound source into an independent object representation at the feature level. The key advantage of OOP is that the ordering of objects remains consistent across all sub-decoders: the $j$-th object in the audio decoder, SED decoder, and DoA decoder always corresponds to the same sound source. This eliminates the need for manual pairing between different auditory parameters and removes the requirement for permutation-invariant training across tasks.

### Dynamic STFT

A time-varying learnable window function is proposed, using per-frame predicted Gaussian window parameters $\mu_t$ (center position) and $\sigma_t$ (width). A large $\sigma_t$ causes the window to approach a rectangular shape, enhancing spectral resolution, while a small $\sigma_t$ narrows the window to enhance temporal resolution. These parameters are predicted from the waveform via 1D convolution, enabling adaptive time-frequency focusing. During training, the window parameters are first frozen until the model converges, after which joint training is conducted.

### Feature Aggregation and Object Separation

Feature aggregation is based on an improved DeFT-Mamba model that captures temporal, spectral, and inter-channel relationships using Mamba and Transformer layers. For efficiency, only a Mamba-FFN is used in the T-Hybrid Mamba block and a conventional FFN in the F-Hybrid Mamba block, with the unfolding operation removed. The aggregated features are split into $J+1$ object features via 2D convolutional kernels.

### Sub-decoder Design

- **MIMO Audio Decoder**: Estimates the direct-path signal $\mathbf{s}_j$ and reverberant signal $\mathbf{h}_j$ for each foreground source (supporting dereverberation), as well as the background noise signal. The MIMO design preserves spatial information to assist DoAE.
- **SED Decoder**: Combines a pretrained ATST with a dual-branch CRNN (T-CRNN for temporal relations and F-CRNN for frequency relations), predicting class probabilities $(1 \times C)$, activity curves $(T' \times 1)$, and SED maps $(T' \times C)$.
- **DoA Decoder**: A CRNN structure that outputs a stream of DoA vectors in Cartesian coordinates, supporting moving-source trajectory prediction.

### Chain-of-Inference (CoI)

To address the misalignment of auditory parameters in initial estimates, CoI comprises two steps:

1. **Temporal Coherence Matching (TCM)**: Bidirectional cross-attention is used to assess temporal consistency between SED and DoA outputs. One branch treats SED as query and DoA as key/value; the other is reversed. The outputs of the two attention branches are fused to generate a coherence cue.
2. **Feature Fusion (FF)**: The fused cue generates $\beta$ and $\gamma$ via FiLM layers to modulate the output of the feature aggregation module, injecting cross-task information. The refined object features are then passed to a second set of sub-decoders.

Training strategy: Net 1 is trained first; the first three DeFT-Mamba blocks of Net 1 are then frozen while Net 2 is trained.

## Key Experimental Results

### Ablation Study (ASA2 Dataset)

| Configuration | SI-SDRi (dB) | SELD ↓ | Parameters |
|---|---|---|---|
| DeFT-Mamba-MISO baseline | 10.4 | — | 3.6M |
| + SED/DoA decoders | 10.4 | 0.317 | 7.2M |
| + ATST + T&F-CRNN | 10.3 | 0.266 | 8.1M (+96.8M) |
| + Noise decoder | 11.0 | 0.241 | same |
| + Direct/reverberant decoder | 10.8 | 0.237 | same |
| + Dynamic STFT | 11.0 | 0.230 | 8.2M (+96.8M) |
| + Chain-of-Inference | **11.2** | **0.206** | 12.1M (+96.8M) |

### MC-FUSS Dataset (USS Task)

When trained from scratch, DeepASA achieves a total SI-SDRi of 17.5 dB; with ASA2 pretraining and fine-tuning, it reaches **18.5 dB**, surpassing all existing methods including DeFT-Mamba (16.4 dB) and SpatialNet (15.8 dB). The advantage is particularly pronounced in 4-source scenarios (17.6 dB vs. 13.8 dB).

### STARSS23 Dataset (SELD Task)

DeepASA + CoI achieves a SELD score of **0.253**, outperforming the DCASE 2023 challenge winner NERC-SLIP (0.260, with ensemble) without requiring model ensembling. The localization error is only 9.8°, far better than competing methods at 12.8°–20.5°.

## Highlights & Insights

1. **Unified framework for multi-task learning**: DeepASA is the first single model to simultaneously achieve MIMO separation, dereverberation, SED, audio classification, and DoAE, with each task mutually benefiting the others.
2. **OOP eliminates parameter-association ambiguity**: Object-level feature separation naturally preserves permutation consistency across decoders, avoiding the pairing problem inherent in traditional track-wise processing.
3. **CoI simulates human auditory reasoning**: When one type of auditory cue is unreliable, complementary cues are used for compensation and refinement. Ablation studies confirm that SED and DoA branches each enhance their respective tasks.
4. **Dynamic STFT**: The time-varying learnable window function enables adaptive time-frequency resolution trade-offs, outperforming fixed windows and time-invariant learnable windows.
5. **Noise decoder yields significant gains**: Explicitly estimating background noise improves separation performance by 0.7 dB SI-SDRi while simultaneously delivering substantial improvements in SED.

## Limitations & Future Work

- The pretrained ATST model is large (96.8M parameters), constituting the vast majority of total parameter count; lighter alternatives for classification feature extraction warrant exploration.
- Training data reverberation times are limited to 0.2–0.6 seconds; performance may degrade in long-reverberation environments.
- Background noise SNR ranges from 6–30 dB; performance is expected to decrease in extremely low-SNR scenarios (< 0 dB).
- CoI currently fuses only SED and DoA cues; future work could explore incorporating audio separation outputs as a third cue.
- Potential privacy risks: the model's ability to separate individual speakers and analyze their directions could be exploited for eavesdropping.

## Related Work & Insights

| Method | Task Scope | MC-FUSS SI-SDRi | STARSS23 SELD | Characteristics |
|---|---|---|---|---|
| DeFT-Mamba | USS only | 16.4 dB | — | Separation backbone of DeepASA |
| SpatialNet | USS only | 15.8 dB | — | Exploits spatial features |
| NERC-SLIP | SELD only | — | 0.260 | Category-aware separation + ensemble |
| CST-former2 | SELD only | — | 0.301 | Conformer-based |
| **DeepASA** | **USS+SED+DoAE** | **18.5 dB** | **0.253** | **Unified framework, no ensembling** |

DeepASA surpasses specialized models on each individual task and is the only unified model to simultaneously cover USS, SED, and DoAE.

The object-oriented processing (OOP) paradigm can be generalized to visual scene analysis, such as multi-object tracking and attribute estimation in video. The multi-cue complementary refinement mechanism of Chain-of-Inference resembles cross-modal attention in multimodal learning, offering inspiration for audio-visual joint analysis. The time-varying window idea of Dynamic STFT is applicable to other audio tasks such as music information retrieval. The explicit noise estimation strategy of the noise decoder has general value for improving model robustness in low-SNR scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The designs of OOP and CoI are innovative, and Dynamic STFT represents an interesting contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, detailed ablation studies, and module-by-module contribution verification.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and rich figures and tables, though the dense notation requires careful reading.
- Value: ⭐⭐⭐⭐ — Provides a unified paradigm for spatial audio analysis with important reference value for multi-task audio modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings](a_multitask_benchmark_for_abusive_language_detection_in_lowr.md)
- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](../../ICLR2026/audio_speech/flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)
- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](../../AAAI2026/audio_speech/psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)
- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](levo_high-quality_song_generation_with_multi-preference_alignment.md)

</div>

<!-- RELATED:END -->
