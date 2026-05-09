---
title: >-
  [Paper Note] TellWhisper: Tell Whisper Who Speaks When
description: >-
  [ACL 2026][Audio & Speech][Multi-speaker speech recognition] This paper proposes TellWhisper, which jointly encodes speaker identity and temporal information into the speech encoder's self-attention via a time-speaker-aware rotary position encoding (TS-RoPE), coupled with a hyperbolic speaker diarization model (Hyper-SD), to achieve joint modeling of "who speaks what when" and attain state-of-the-art performance on multi-speaker ASR tasks.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Multi-speaker speech recognition
  - speaker diarization
  - rotary position encoding
  - hyperbolic space classification
  - Whisper
date: 2026-05-08
content_hash: f0e1ec8d0aa53a2f
---

# TellWhisper: Tell Whisper Who Speaks When

**Conference**: ACL 2026
**arXiv**: [2601.03712](https://arxiv.org/abs/2601.03712)
**Code**: [Project Page](https://walker-hyf.github.io/TellWhisper)
**Area**: Audio & Speech
**Keywords**: Multi-speaker speech recognition, speaker diarization, rotary position encoding, hyperbolic space classification, Whisper

## TL;DR

This paper proposes TellWhisper, which jointly encodes speaker identity and temporal information into the speech encoder's self-attention via a time-speaker-aware rotary position encoding (TS-RoPE), coupled with a hyperbolic speaker diarization model (Hyper-SD), to achieve joint modeling of "who speaks what when" and attain state-of-the-art performance on multi-speaker ASR tasks.

## Background & Motivation

**Background**: Multi-speaker automatic speech recognition (MASR) aims to predict "who speaks what when" from multi-party conversational audio. Conventional approaches align speaker diarization (SD) and single-speaker ASR via timestamps, but struggle under overlapping speech and rapid speaker turns.

**Limitations of Prior Work**: Even recent methods attempting to unify SD and ASR fundamentally treat temporal and speaker modeling separately. Three representative strategies each exhibit distinct limitations: (1) masking non-target regions with SD labels before encoding, which causes blank inputs that induce hallucinations; (2) separating target-speaker audio, which requires additional speaker prompts and fails in overlapping regions; (3) linearly blending encoder outputs weighted by speaker posteriors after encoding, entangling semantics with speaker cues.

**Key Challenge**: Separate modeling of temporal structure and speaker identity is inherently brittle under rapid speaker changes and overlapping speech — time and speaker are coupled and should be jointly modeled rather than concatenated post hoc.

**Goal**: Jointly model temporal and speaker information naturally within the speech encoder through positional encoding, enabling the self-attention mechanism to simultaneously attend to "when" and "who."

**Key Insight**: Inspired by multi-dimensional RoPE for cross-axis encoding in vision and multimodal settings, the paper extends RoPE from encoding time alone to simultaneously encoding time and speaker activity.

**Core Idea**: Design TS-RoPE to partition Query/Key channels into a temporal subspace and a speaker subspace, realizing joint time-speaker modeling in self-attention through region-specific rotation angles.

## Method

### Overall Architecture

TellWhisper is built upon Whisper large-v3-turbo. After the input multi-speaker audio is encoded by convolutional layers, Hyper-SD first estimates frame-level speaker activities. TS-RoPE then constructs multi-dimensional positional encodings from temporal indices and speaker activity information, which are injected into the encoder self-attention. A structured content predictor finally generates speaker labels, timestamps, and transcriptions autoregressively.

### Key Designs

1. **TS-RoPE (Time-Speaker Rotary Position Encoding)**:

    - **Function**: Jointly encodes temporal and speaker information in the Query/Key of self-attention.
    - **Mechanism**: The channel dimension $D$ of each frame's features is partitioned into groups of 16 dimensions, with each group's 8 rotation pairs assigned alternately to the temporal and four speaker subspaces: $[\psi_{time}, \psi_{spk_1}, \psi_{time}, \psi_{spk_2}, \psi_{time}, \psi_{spk_3}, \psi_{time}, \psi_{spk_4}]$. The temporal position uses the frame index directly, $\psi_{time}(f_t) = t$; the speaker position is composed of the cumulative speaker turn count and the current activity probability, $\psi_{spk_s}(f_t) = \mathcal{C}_{t,s} + \pi_{t,s}$. An additional phase bias $\psi'_{spk_s}(f_t) = \psi_{spk_s}(f_t) + (1 - \pi_{t,s})$ is applied to the speaker subspace of the Query to encourage attention to focus on active speakers.
    - **Design Motivation**: Via rotation angle differences, consecutive frames of the same speaker obtain similar rotation angles (small angle difference → high attention weight), while frames from different speakers or at speaker transitions exhibit large angle differences, thereby modeling intra-speaker continuity and inter-speaker switching.

2. **Hyper-SD (Hyperbolic Speaker Diarization)**:

    - **Function**: Estimates reliable frame-level speaker activity probabilities.
    - **Mechanism**: Weighted aggregation of multi-layer WavLM features is followed by Conformer encoding of contextual information. Euclidean features are mapped to the Poincaré ball model, where learnable hyperbolic prototypes $\mathbf{p}_n$ are assigned to each speaker combination class (totaling $2^4 = 16$ classes, covering silence, single-speaker, and overlapping combinations). Class probabilities are computed from the hyperbolic distance $d_{t,n} = d_{\mathbb{B}_c}(\mathbf{v}'_t, \mathbf{p}_n)$ between frame embeddings and prototypes, and frame-level activity per speaker is obtained by marginalization: $\pi_{t,s} = \sum_n b_{s,n} \sigma(-d_{t,n})$.
    - **Design Motivation**: Hyperbolic space exhibits exponentially growing volume, so small feature perturbations produce large distance changes, significantly improving the separability of acoustically similar speakers and stabilizing speaker posterior estimation.

3. **Structured Content Predictor**:

    - **Function**: Converts encoder outputs into structured "speaker + timestamp + text" sequences.
    - **Mechanism**: Temporally continuous speech from the same speaker is treated as an independent segment, represented as the token sequence $\langle spk_s \rangle, \langle t_{start} \rangle, \langle text \rangle, \langle t_{end} \rangle$, with all segments concatenated in chronological order. An autoregressive framework is used for next-token prediction training, generating tokens one by one until EOS during decoding.
    - **Design Motivation**: The unified prediction format eliminates the alignment problem between SD and ASR outputs in traditional pipelines.

### Loss & Training

A two-stage fine-tuning strategy is adopted: pre-fine-tuning on single-speaker speech (LibriSpeech) to learn single-speaker structured prediction, followed by fine-tuning on multi-speaker conversational speech. Hyper-SD is trained with NLLLoss; the hyperbolic classifier is optimized with RiemannianAdam; remaining components use AdamW. WavLM uses a smaller learning rate, while other modules use a larger one.

## Key Experimental Results

### Main Results

| Dataset | Metric | TellWhisper | Dicow (Prev. SOTA) | Gain |
|--------|------|-------------|----------------|------|
| AMI | CP-WER↓ | 32.53 | 33.57 | -1.04 |
| NotSoFar | CP-WER↓ | 34.48 | 35.22 | -0.74 |
| LibriCSS | CP-WER↓ | 9.88 | 10.62 | -0.74 |
| AMI | TCP-WER↓ | 33.47 | 34.02 | -0.55 |
| NotSoFar | TCP-WER↓ | 34.51 | 35.64 | -1.13 |
| LibriCSS | TCP-WER↓ | 11.06 | 11.33 | -0.27 |

### Ablation Study

| Configuration | AMI CP-WER | AMI TCP-WER | Note |
|------|-----------|-------------|------|
| Full TellWhisper | 32.53 | 33.47 | All components enabled |
| w/o Query phase bias | 35.02 | 35.26 | CP-WER +2.49 |
| w/o speaker turn count | 36.22 | 36.68 | CP-WER +3.69 |
| w/o speaker activity | 36.84 | 36.89 | Largest degradation |

### Key Findings

- Hyper-SD surpasses Pyannote3 and Diarizen on all 6 SD datasets, confirming that hyperbolic space classification outperforms Euclidean linear classification.
- The most significant DER improvement is observed on AliMeeting (13.03→10.76), demonstrating the particular effectiveness of hyperbolic speaker separation in real meeting scenarios.
- Ablation experiments confirm that the three components of TS-RoPE (activity probability, turn count, Query bias) each contribute incrementally, with the speaker activity signal being the most critical.
- TellWhisper's advantages are more pronounced on real meeting data (AMI, NotSoFar) than on simulated data (Libri2Mix), as simulated data features overlap from time zero without speaker turns, limiting the gains from TS-RoPE.

## Highlights & Insights

- The TS-RoPE design is elegant — it injects coupled time-speaker information into the model without modifying the main architecture, relying solely on channel partitioning and angle modulation of rotary position encoding.
- Applying hyperbolic space for speaker activity estimation is insightful — the exponential volume growth of negatively curved space amplifies distances between acoustically similar speakers.
- The design intuition behind the Query-side phase bias is clear: inactive speakers receive larger biases, causing attention to favor active speakers.
- Visualization shows that the 16 class prototypes are uniformly distributed in hyperbolic space without hierarchical structure, which is well-suited to frame-level classification requirements.

## Limitations & Future Work

- The current TS-RoPE design supports 1–4 speakers; extension to a larger number of speakers requires further investigation.
- Hyper-SD performs hyperbolic classification only after feature extraction, with the encoder and classifier residing in separate embedding spaces; end-to-end hyperbolic learning could yield further improvements.
- Experiments are conducted primarily on English datasets; cross-lingual generalization remains to be verified.
- The limited advantage on Libri2Mix suggests that TS-RoPE provides marginal gains in scenarios with extreme overlap but no speaker turns.

## Related Work & Insights

- **vs. Dicow (Polok et al.)**: Dicow filters non-target regions with speaker masks before encoding, which may induce hallucinations; TellWhisper integrates speaker information within the encoder via positional encoding in a more seamless manner.
- **vs. SortFormer (Park et al.)**: SortFormer weights encoder outputs with speaker sinusoidal kernels after encoding, causing linear mixing to entangle semantics with speaker cues; TS-RoPE achieves decoupled joint modeling through rotation angles.
- **vs. Multi-dimensional RoPE (vision)**: Visual RoPE encodes spatial axes such as width and height; TellWhisper innovatively introduces speaker activity as a new dimension in RoPE.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ TS-RoPE extends RoPE to joint time-speaker encoding with a novel and elegant formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 MASR datasets and 6 SD datasets with multiple baselines and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear with complete mathematical derivations.
- **Value**: ⭐⭐⭐⭐⭐ Significantly advances multi-speaker speech understanding; the TS-RoPE idea is extensible to other multi-dimensional sequence modeling tasks.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms](when_misinformation_speaks_and_converses_rethinking_fact-checking_in_audio_platf.md)
- [\[AAAI 2026\] Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation](../../AAAI2026/audio_speech/listen_like_a_teacher_mitigating_whisper_hallucinations_using_adaptive_layer_att.md)
- [\[ICLR 2026\] Knowing When to Quit: Probabilistic Early Exits for Speech Separation](../../ICLR2026/audio_speech/knowing_when_to_quit_probabilistic_early_exits_for_speech_separation.md)
- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)
- [\[ICLR 2026\] When and Where to Reset Matters for Long-Term Test-Time Adaptation](../../ICLR2026/audio_speech/when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)

<!-- RELATED:END -->
