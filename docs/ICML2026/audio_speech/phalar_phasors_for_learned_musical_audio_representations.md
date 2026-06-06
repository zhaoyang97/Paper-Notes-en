---
title: >-
  [Paper Note] PhaLar: Phasors for Learned Musical Audio Representations
description: >-
  [ICML 2026][Audio & Speech][Phase Equivariance] PhaLar projects audio features onto the complex plane and utilizes phase equivariance—corely encoding temporal alignment as phase rotations via FFT—to achieve a 70% relativ…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Phase Equivariance"
  - "Complex-valued Neural Networks"
  - "Audio Coherence"
  - "Stem Retrieval"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: fb53e73aed8ecd3e
---

# PhaLar: Phasors for Learned Musical Audio Representations

**Conference**: ICML 2026  
**arXiv**: [2605.03929](https://arxiv.org/abs/2605.03929)  
**Code**: To be confirmed  
**Area**: Audio / Music Information Retrieval  
**Keywords**: Phase Equivariance, Complex-valued Neural Networks, Audio Coherence, Stem Retrieval, Contrastive Learning

## TL;DR
PhaLar projects audio features onto the complex plane and utilizes phase equivariance—corely encoding temporal alignment as phase rotations via FFT—to achieve a 70% relative improvement over SOTA in music stem retrieval tasks, requiring only 44% of the competitor's parameters and a 7× training speedup. The shift from "phase invariance" to "phase equivariance" represents a fundamental change in architectural philosophy.

## Background & Motivation

**Background**: Existing audio representation learning inherits the computer vision paradigm—treating spectrograms as 2D images processed by CNN / ViT, employing Global Average Pooling (GAP) to achieve translation invariance. Foundation models like CLAP and CDPAM perform well in semantic similarity tasks.

**Limitations of Prior Work**: Translation invariance is beneficial for semantic classification (e.g., identifying the presence of a "guitar") but detrimental to structural coherence tasks. Music **stem retrieval** (matching a missing part, such as drums + bass, that is coherent in time and harmonics) requires precise temporal alignment. GAP discards temporal order, causing completely misaligned rhythms to be mapped to the same latent representation—even if two segments contain the same instruments, they may be entirely incoherent.

**Key Challenge**: Semantic similarity and structural coherence are built on fundamentally different geometric foundations. Foundation models are designed to be "structure-blind"—useful for identifying a "rock song" but useless for evaluating whether two audio segments are temporally aligned.

**Goal**: Design a representation learning framework that **explicitly preserves temporal alignment information** instead of discarding it, while maintaining parameter efficiency and training speed.

**Key Insight**: The Fourier Shift Theorem—**temporal shifts in the time domain correspond to phase rotations in the frequency domain**. By leveraging this physical property to encode temporal relationships as phase angles, coherence can be evaluated on the complex plane.

**Core Idea**: Transition from real-valued magnitude features to complex-valued phasor representations. A learned spectral pooling layer applies FFT to map the temporal dimension to the frequency domain, followed by a phase-equivariant complex-valued neural network to process these features, ensuring phase information is naturally preserved rather than destroyed.

## Method

### Overall Architecture
Three stages: (1) **Harmonic Backbone**—a lightweight CNN extracts pitch-aware features from CQT spectrograms; (2) **Time → Frequency Projection**—a learned spectral pooling layer performs FFT along the temporal dimension to encode temporal positions as phase angles; (3) **Phase-Equivariant Head**—a complex-valued neural network evaluates the alignment of two audio segments. The input is a CQT spectrogram of uncompressed temporal span, and the output is a 512-dimensional complex-valued embedding used to calculate coherence scores.

### Key Designs

1.  **Harmonic Backbone**:
    *   **Function**: Extract pitch-aware features from CQT spectrograms while maintaining computational efficiency.
    *   **Mechanism**: A 10-layer axial residual CNN where each layer contains three types of convolutions: $3 \times 1$ in the frequency direction, $1 \times 3$ in the temporal direction, and $1 \times 1$ point-wise, decoupling harmonic and temporal processing. The logarithmic spacing of CQT ensures that pitch shifts are pure linear translations, allowing kernels to identify the same interval (e.g., a "major third") across all keys without learning key-specific variations. Strided convolutions at every even layer compress time, with a total compression of 32×.
    *   **Design Motivation**: Enforce pitch equivariance bias via log-spaced CQT to learn universal musical patterns rather than key-specific variants; reduce computational costs.

2.  **Learned Spectral Pooling**:
    *   **Function**: Replace GAP to convert temporal information preserved in the sequence into phase rotations on the complex plane.
    *   **Mechanism**: First, flatten the channel and frequency dimensions of the backbone output $X \in \mathbb{R}^{B \times H \times F \times T'}$ into $\bar{X} \in \mathbb{R}^{B \times (HF) \times T'}$. Project onto a learned basis $W_{\text{proj}} \in \mathbb{R}^{(HF) \times D}$ to obtain $Z_{\text{time}} = \bar{X} W_{\text{proj}} \in \mathbb{R}^{B \times T' \times D}$ (projecting across all frequencies ensures simultaneous encoding of interval structure and absolute frequency position). Apply RFFT along the temporal axis to get $S = \text{rfft}(Z_{\text{time}}) \in \mathbb{C}^{B \times C \times D}$ ($C = \lfloor T'/2 \rfloor + 1$), then truncate/pad to a fixed $C = 8$, resulting in a fixed-size embedding of $D \times C = 640$ complex values.
    *   **Design Motivation**: Via the Fourier Shift Theorem, the temporal alignment problem is transformed into a geometric relationship on the complex plane; the phase $\angle S_{c, d}$ explicitly encodes temporal shifts—interpretable as a learned modulation spectrum.

3.  **Complex Projection Head**:
    *   **Function**: Perform phase-equivariant processing in the complex domain to calculate structural coherence scores between two samples.
    *   **Mechanism**: Implements two complex linear layers (with intermediate complex RMSNorm + modReLU activation) satisfying phase equivariance $f(x \cdot e^{i\theta}) = f(x) \cdot e^{i\theta}$. The final projection yields a 512-dimensional complex vector $z \in \mathbb{C}^{512}$, and coherence is calculated using a parameterized Hermitian inner product $s(z_x, z_y) = \Re(z_x^H W z_y)$ (where $W \in \mathbb{C}^{D \times D}$ is a learnable complex weight matrix). During inference, symmetry is ensured via $s_{\text{comm}} = \frac{s(z_x, z_y) + s(z_y, z_x)}{2}$.
    *   **Design Motivation**: Complex weights allow the model to apply learned phase rotations to "align" stems, addressing micro-temporal deviations (e.g., "laid-back" feel); taking the real part projects complex alignment into a scalar score while maintaining sensitivity to phase; omitting saturating nonlinearities allows high-energy transients to contribute more to the final score.

## Key Experimental Results

### Main Results (Music Stem Retrieval Accuracy)

| Dataset | K | PhaLar (2.3M) | COCOLA (5.2M) | Gain |
|--------|---|-------------|-------------|---------|
| MoisesDB | 8 | 86.79% | 75.81% | +14.3% |
| MoisesDB | 16 | 81.49% | 64.44% | +26.4% |
| MoisesDB | 64 | **70.87%** | 41.84% | **+69.2%** |
| Slakh2100 | 8 | 87.69% | 79.33% | +10.5% |
| Slakh2100 | 16 | 83.28% | 71.58% | +16.3% |
| Slakh2100 | 64 | 72.37% | 55.84% | +29.5% |
| ChocoChorales | 64 | 98.61% | 89.34% | +10.3% |

PhaLar establishes a new SOTA across all datasets with only 44% of the parameters of COCOLA. **MoisesDB K=64** (the hardest setting) shows a +69% improvement—the advantage of phase encoding is most pronounced at high K values (where distractors have similar tonality).

### Ablation Study (MoisesDB K=64)

| Configuration | Accuracy ↑ | Drop |
|------|--------|------|
| PhaLar Full | 70.87% | — |
| w/o Spectral Pooling (GAP + Real MLP) | 51.97% | -18.9 |
| w/o Phase Equivariance (Magnitude only + Real MLP) | 60.59% | -10.3 |
| w/o Phase Equivariance (Complex Cosine Sim) | 61.93% | -8.94 |
| w/o Indefinite $W$ (PSD Constraint) | 67.85% | -3.02 |
| w/o Strict Pitch Equivariance (Mel Spectrogram) | 69.21% | -1.66 |

### Key Findings
*   **Phase Information is Essential**: Using only magnitude leads to a 10.3% drop, proving that relative phase angles are critical for detecting musical coherence.
*   **Weight Matrix Indefiniteness is Beneficial**: Compared to PSD constraints, the drop is 3%—an indefinite metric space allows the model to capture destructive interference (negative similarity scores indicating anti-phase alignment).
*   **CQT Outperforms Mel**: The strict log-spacing of CQT provides a stronger pitch equivariance bias compared to Mel's approximate spacing.
*   **Highest Correlation with Human Coherence Judgments**: In 880 ratings from 22 participants, PhaLar achieved Pearson 0.387 and Spearman 0.414, significantly higher than all baselines (CLAP was near-random).
*   **Zero-shot Beat Tracking**: Similarities calculated from metronomes of different BPMs yielded GTZAN F1 = 0.627—confirming the model linearizes "alignment" into geometric primitives (phase rotations).
*   **Parameter Efficiency**: Training completed in 50 GPU hours (vs 340 hours for COCOLA), a 7× speedup.

## Highlights & Insights
*   **Paradigm Shift from Phase Invariance to Phase Equivariance**: A fundamental change in architectural philosophy. While traditional models pursue translation invariance for semantic robustness, PhaLar does the opposite—explicitly preserving temporal structure. This insight extends to fields such as radar and MRI time series.
*   **Elegant Mapping from Real Magnitudes to Complex Phasors**: The Fourier Shift Theorem transforms temporal alignment into a geometric rotation problem; standard FFT automatically handles phase information—a rare successful application of complex-valued neural networks in discriminative (non-generative) tasks.
*   **Interpretability of Phase Equivariance**: The model learns two types of features in the complex plane—"rotating" features that rotate fully around the origin to capture periodic rhythmic structures, and "magnitude-only" features that oscillate within a restricted range representing time-invariant attributes like tonality or mood.

## Limitations & Future Work
*   Reliance on the periodicity assumption of RFFT; performance degrades for non-periodic tempo changes (rubato / ritardando).
*   Limited performance on sustained arpeggiated pads or instruments lacking clear periodicity.
*   Sensitivity to heavy compression or lossy audio formats (which destroy fine-grained magnitude information in the input spectrum).
*   Training data biased toward Western pop music; the geometric concept of "coherence" might not align with cultural contexts where micro-temporal deviations are stylistic features.

## Related Work & Insights
*   **vs COCOLA** (Ciranni 2025): Both target musical coherence, but COCOLA still uses GAP, discarding temporal information; PhaLar achieves deeper structural awareness and higher parameter efficiency through the pitch equivariance of the CQT backbone and the phase equivariance of the complex head.
*   **vs Foundation Models CLAP / CDPAM**: Optimized for semantic classification, these are completely blind to the task; even equipping MERT (95M) with PhaLar's spectral pooling + CVNN head only reached 46% (vs PhaLar's 71%)—end-to-end architectural consistency is more critical than improving the aggregation layer.
*   **vs FAD / ViSQOL**: Traditional feature-based metrics measure marginal distribution similarity, ignoring conditional requirements (whether a stem fits a specific mix); PhaLar as a reference-based perceptual metric avoids these limitations.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Fundamental paradigm shift from phase invariance to phase equivariance; clever combination of FFT physical properties and complex-valued learning; rare innovation for CVNNs in discriminative audio tasks.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three multi-source datasets + comparison with semantic/perceptual baselines + human listening tests + zero-shot beat tracking + chord linear probes + comprehensive ablations.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain, well-articulated motivation, accurate charts; specific sections dedicated to the interpretability of learned features; honest discussion of limitations.
*   Value: ⭐⭐⭐⭐⭐ 69% relative improvement over SOTA; the phase-equivariant framework offers inspiration for domains involving complex-valued signals like radar, medical imaging, and time series.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MSU-Bench: Musical Score Understanding Benchmark](../../ACL2026/audio_speech/musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[NeurIPS 2025\] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](../../NeurIPS2025/audio_speech/perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)
- [\[ICML 2026\] Two-Dimensional Quantization for Geometry-Aware Audio Coding](two-dimensional_quantization_for_geometry-aware_audio_coding.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](../../ACL2026/audio_speech/fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)

</div>

<!-- RELATED:END -->
