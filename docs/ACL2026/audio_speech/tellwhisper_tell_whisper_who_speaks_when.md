---
title: >-
  [Paper Note] TellWhisper: Tell Whisper Who Speaks When
description: >-
  [ACL 2026][Audio & Speech][Multi-speaker ASR] Ours proposes TellWhisper, which achieves joint modeling of "who spoke what at when" by designing a Time-Speaker aware Rotary Position Embedding (TS-RoPE) to unify speaker id…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Multi-speaker ASR"
  - "Speaker Diarization"
  - "Rotary Position Embedding"
  - "Hyperbolic Space Classification"
  - "Whisper"
date: 2026-05-08
content_hash: cb317d895794c892
---

# TellWhisper: Tell Whisper Who Speaks When

**Conference**: ACL 2026  
**arXiv**: [2601.03712](https://arxiv.org/abs/2601.03712)  
**Code**: [Project Page](https://walker-hyf.github.io/TellWhisper)  
**Area**: Audio & Speech  
**Keywords**: Multi-speaker ASR, Speaker Diarization, Rotary Position Embedding, Hyperbolic Space Classification, Whisper

## TL;DR

Ours proposes TellWhisper, which achieves joint modeling of "who spoke what at when" by designing a Time-Speaker aware Rotary Position Embedding (TS-RoPE) to unify speaker identity and temporal information within the self-attention of the speech encoder. Combined with a Hyperbolic space Speaker Diarization model (Hyper-SD), it achieves state-of-the-art performance in multi-speaker ASR tasks.

## Background & Motivation

**Background**: Multi-speaker Automatic Speech Recognition (MASR) aims to predict "who spoke what at when" from multi-party conversational speech. Traditional solutions fuse Speaker Diarization (SD) and single-speaker ASR through timestamp alignment, but face difficulties in alignment during overlapping speech and rapid speaker transitions.

**Limitations of Prior Work**: Even recent methods attempting to unify SD and ASR essentially treat temporal and speaker modeling separately. This is reflected in the limitations of three strategies: (1) masking non-target regions with SD labels before encoding, which leads to hallucinations caused by blank inputs; (2) attempting to isolate target speaker speech, which requires additional speaker prompts and fails in overlapping regions; (3) linear mixing via speaker posterior weighting after the encoder output, which entangles semantic and speaker cues.

**Key Challenge**: Separate modeling of temporal structure and speaker identity is inherently fragile in scenarios with rapid speaker switches and overlapping speech—time and speaker are coupled and should be modeled jointly rather than concatenated post-hoc.

**Goal**: To naturally and jointly model time and speaker information within the speech encoder via position embeddings, allowing the self-attention mechanism to simultaneously focus on "when" and "who."

**Key Insight**: Inspired by the cross-axis encoding of multi-dimensional RoPE in vision and multi-modal fields, RoPE is extended from encoding only time to encoding both time and speaker activity states.

**Core Idea**: Design TS-RoPE to partition Query/Key channels into time and speaker subspaces, achieving joint temporal-speaker modeling in self-attention via region-specific rotation angles.

## Method

### Overall Architecture

TellWhisper is built upon Whisper large-v3-turbo. Input multi-speaker speech is encoded by convolutional layers, after which Hyper-SD estimates frame-level speaker activity. Then, TS-RoPE constructs multi-dimensional position embeddings using time indices and speaker activity information, which are injected into the encoder's self-attention. Finally, a structured content predictor outputs speaker labels, timestamps, and transcribed text in an auto-regressive manner.

### Key Designs

1.  **TS-RoPE (Time-Speaker Rotary Position Embedding)**:
    - **Function**: Simultaneously encodes time and speaker information within Query/Key of self-attention.
    - **Mechanism**: The channel dimension $D$ of each frame feature is partitioned into groups of 16 dimensions. Within each group, 8 rotation pairs are assigned alternately to time and 4 speaker subspaces: $[\psi_{time}, \psi_{spk_1}, \psi_{time}, \psi_{spk_2}, \psi_{time}, \psi_{spk_3}, \psi_{time}, \psi_{spk_4}]$. Temporal positions use frame indices $\psi_{time}(f_t) = t$; speaker positions consist of cumulative speaker turn counts and current activity probabilities $\psi_{spk_s}(f_t) = \mathcal{C}_{t,s} + \pi_{t,s}$. Additionally, an extra phase bias is applied to the Query speaker subspace $\psi'_{spk_s}(f_t) = \psi_{spk_s}(f_t) + (1 - \pi_{t,s})$ to encourage attention to focus on active speakers.
    - **Design Motivation**: By leveraging rotation angle differences, continuous speech frames from the same speaker receive similar rotation angles (small angle difference $\rightarrow$ high attention weight), while angles differ significantly between different speakers or at speaker transitions, thereby modeling intra-speaker continuity and inter-speaker switching.

2.  **Hyper-SD (Hyperbolic Speaker Diarization)**:
    - **Function**: Estimates reliable frame-level speaker activity probabilities.
    - **Mechanism**: Utilizes WavLM multi-layer feature weighted aggregation followed by a Conformer for context encoding. Euclidean features are mapped to a Poincaré ball model, where learnable hyperbolic prototypes $\mathbf{p}_n$ are assigned to each of the speaker combination classes (total $2^4 = 16$ classes, including silence, single speaker, overlaps, etc.). Class probabilities are calculated via the hyperbolic distance from frame embeddings to prototypes $d_{t,n} = d_{\mathbb{B}_c}(\mathbf{v}'_t, \mathbf{p}_n)$, and then marginalized to obtain frame-level activity $\pi_{t,s} = \sum_n b_{s,n} \sigma(-d_{t,n})$ for each speaker.
    - **Design Motivation**: Hyperbolic space exhibits exponential volume growth, where small feature shifts can produce large distance changes, significantly improving separability between speakers with similar timbres and stabilizing speaker posterior estimation.

3.  **Structured Content Predictor**:
    - **Function**: Transmutes encoder outputs into structured "speaker + timestamp + text" sequences.
    - **Mechanism**: Continuous speech from the same speaker is treated as an independent segment, represented as a token sequence $\langle spk_s \rangle, \langle t_{start} \rangle, \langle text \rangle, \langle t_{end} \rangle$, with all segments concatenated chronologically. An auto-regressive framework is used for next-token prediction, generating token-by-token until EOS.
    - **Design Motivation**: A unified prediction format avoids alignment problems between SD and ASR outputs in traditional pipelines.

### Loss & Training

A two-stage fine-tuning strategy is adopted: first, pre-fine-tuning on single-speaker speech (LibriSpeech) to learn single-speaker structured prediction, followed by fine-tuning on multi-speaker conversational speech. Hyper-SD is trained using NLLLoss, the hyperbolic classifier is optimized with RiemannianAdam, and other components use AdamW. WavLM uses a smaller learning rate compared to other modules.

## Key Experimental Results

### Main Results

| Dataset | Metric | TellWhisper | Dicow (Prev. SOTA) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| AMI | CP-WER↓ | 32.53 | 33.57 | -1.04 |
| NotSoFar | CP-WER↓ | 34.48 | 35.22 | -0.74 |
| LibriCSS | CP-WER↓ | 9.88 | 10.62 | -0.74 |
| AMI | TCP-WER↓ | 33.47 | 34.02 | -0.55 |
| NotSoFar | TCP-WER↓ | 34.51 | 35.64 | -1.13 |
| LibriCSS | TCP-WER↓ | 11.06 | 11.33 | -0.27 |

### Ablation Study

| Configuration | AMI CP-WER | AMI TCP-WER | Description |
| :--- | :--- | :--- | :--- |
| Full TellWhisper | 32.53 | 33.47 | All components enabled |
| w/o Query Phase Bias | 35.02 | 35.26 | CP-WER +2.49 |
| w/o Speaker Turn Count | 36.22 | 36.68 | CP-WER +3.69 |
| w/o Speaker Activity | 36.84 | 36.89 | Maximum degradation |

### Key Findings

- Hyper-SD surpasses Pyannote3 and Diarizen across all 6 SD datasets, confirming that hyperbolic space classification is superior to Euclidean linear classification.
- The DER improvement is most significant on AliMeeting (13.03 $\rightarrow$ 10.76), indicating that hyperbolic space's speaker separation capability is particularly prominent in real meeting scenarios.
- Ablation experiments demonstrate that the three components of TS-RoPE (activity probability, turn count, Query bias) contribute layer by layer, with the speaker activity signal being the most critical.
- The advantages of TellWhisper are more evident in real meeting scenarios (AMI, NotSoFar) than in simulated data (Libri2Mix), as overlaps in simulated data often start from time zero without speaker switches, limiting the utility of TS-RoPE.

## Highlights & Insights

- The design of TS-RoPE is elegant—it injects coupled temporal-speaker information via channel partitioning and angle modulation of RoPE without changing the main model architecture.
- Using hyperbolic space for speaker activity estimation is clever—it exploits the exponential volume growth of negative curvature space to amplify distances between speakers with similar timbres.
- The design of the additional phase bias on the Query side is intuitive: inactive speakers receive a larger bias $\rightarrow$ attention shifts preferentially toward active speakers.
- Visualizations show that the 16 class prototypes are distributed uniformly in hyperbolic space without hierarchical structure, matching the requirements of frame-level classification.

## Limitations & Future Work

- The current TS-RoPE design supports 1-4 speakers; extending it to more speakers requires further research.
- Hyper-SD only performs hyperbolic classification after feature extraction; since the encoder and classifier reside in different embedding spaces, end-to-end hyperbolic learning may provide further improvements.
- Experiments were primarily conducted on English datasets; cross-lingual generalization needs verification.
- The advantage on Libri2Mix is not significant, suggesting limited gains for TS-RoPE in scenarios with extreme overlap but no speaker switches.

## Related Work & Insights

- **vs Dicow (Polok et al.)**: Dicow filters via speaker masks before encoding, potentially triggering hallucinations; TellWhisper fuses speaker information more seamlessly via position embeddings inside the encoder.
- **vs SortFormer (Park et al.)**: SortFormer applies speaker sinusoidal kernel weighting after the encoder output, where linear mixing entangles semantics and speakers; TS-RoPE achieves decoupled joint modeling via rotation angles.
- **vs Dimensional RoPE (Vision)**: Vision RoPE encodes spatial axes such as width/height; TellWhisper innovatively introduces speaker activity as a new dimension in RoPE.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ TS-RoPE extends RoPE to joint temporal-speaker encoding, offering a novel and elegant implementation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 MASR datasets and 6 SD datasets, with multiple baseline comparisons and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear method descriptions and complete formula derivations.
- Value: ⭐⭐⭐⭐⭐ Significantly advances multi-speaker speech understanding; the TS-RoPE concept is extendable to other multi-dimensional sequence modeling tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms](when_misinformation_speaks_and_converses_rethinking_fact-checking_in_audio_platf.md)
- [\[ICLR 2026\] Knowing When to Quit: Probabilistic Early Exits for Speech Separation](../../ICLR2026/audio_speech/knowing_when_to_quit_probabilistic_early_exits_for_speech_separation.md)
- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)
- [\[ICLR 2026\] When and Where to Reset Matters for Long-Term Test-Time Adaptation](../../ICLR2026/audio_speech/when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)
- [\[ICLR 2026\] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment](../../ICLR2026/audio_speech/when_style_breaks_safety_defending_llms_against_superficial_style_alignment.md)

</div>

<!-- RELATED:END -->
