---
title: >-
  [Paper Note] WhAM: Towards A Translative Model of Sperm Whale Vocalization
description: >-
  [NeurIPS 2025][Audio & Speech][Sperm whale acoustics] This paper proposes WhAM (Whale Acoustics Model), the first Transformer-based generative model for sperm whale codas, achieving acoustic translation…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "Sperm whale acoustics"
  - "generative audio model"
  - "acoustic translation"
  - "Masked Acoustic Token Model"
  - "cross-domain style transfer"
date: 2026-05-08
content_hash: f09ed0373ffb7fa9
---

# WhAM: Towards A Translative Model of Sperm Whale Vocalization

**Conference**: NeurIPS 2025
**arXiv**: [2512.02206](https://arxiv.org/abs/2512.02206)  
**Code**: [GitHub](https://github.com/Project-CETI/wham)  
**Area**: Audio/Speech (Bioacoustics)
**Keywords**: Sperm whale acoustics, generative audio model, acoustic translation, Masked Acoustic Token Model, cross-domain style transfer

## TL;DR

This paper proposes WhAM (Whale Acoustics Model), the first Transformer-based generative model for sperm whale codas, achieving acoustic translation, synthetic generation, and downstream classification through fine-tuning of VampNet.

## Background & Motivation

**Background**: Sperm whales communicate through short click sequences called codas, whose rhythmic, temporal, and spectral characteristics serve as markers of clan dialect and social identity. Recent machine learning approaches have made progress in coda detection and classification (Bermant et al., 2019), GAN-based generation (Beguš et al., 2023), and temporal analysis (Sharma et al., 2024b).

**Limitations of Prior Work**: (a) Existing GAN-based generative models cannot be conditioned on audio prompts; (b) temporal methods focus solely on inter-click intervals (ICI), neglecting spectral features in raw audio (e.g., ornament patterns); (c) classification and generation are trained as separate models with no unified framework; (d) cross-acoustic-domain translation has not been attempted.

**Key Challenge**: Sperm whale coda datasets are extremely scarce (approximately 10,000 recordings, totaling ~6 hours), whereas modern audio generative models typically require massive training data.

**Goal**: To construct a unified generative model that simultaneously enables coda synthesis, acoustic translation, and feature classification under severely data-limited conditions.

**Key Insight**: VampNet (a Masked Acoustic Token Model pretrained on large-scale music data) is adapted to the sperm whale domain via two-stage fine-tuning (domain adaptation → species-specific fine-tuning).

**Core Idea**: Transfer a pretrained music generation Transformer to the bioacoustics domain via LoRA fine-tuning, yielding the first acoustic translation model capable of converting arbitrary audio into sperm whale coda style.

## Method

### Overall Architecture

WhAM is built on the VampNet architecture and consists of three components:
- **Acoustic Tokenizer $T$**: Encodes $N_{\text{sec}}$ seconds of audio (at $N_{\text{sam}}$ Hz) into a discrete token sequence of length $\ell$: $T: \mathbb{R}^{N_{\text{sec}} \times N_{\text{sam}}} \to \Sigma^{\ell}$
- **Masked Acoustic Token Model (MATM) $M$**: A bidirectional Transformer performing a cloze task: $M: (\Sigma \cup \{[\texttt{MASK}]\})^{\ell} \to \Sigma^{\ell}$
- **Detokenizer $T^{-1}$**: Reconstructs audio from the token sequence: $T^{-1}: \Sigma^{\ell} \to \mathbb{R}^{N_{\text{sec}} \times N_{\text{sam}}}$

Generation employs iterative parallel decoding to progressively unmask tokens.

### Key Designs

#### Two-Stage Fine-Tuning Strategy
- **Stage 1 — Domain Adaptation**: Fine-tuned for 500k iterations on FSD (7h45m animal audio) + AudioSet (~5h animal audio) + WMMS (4h8m marine mammals) + BirdSet (110h subset)
- **Stage 2 — Species-Specific Fine-Tuning**: Further fine-tuned for 500k iterations on DSWP (2,507 codas, 1h26m) + CETI (7,653 codas, 4h33m)
- Both stages employ LoRA (Low Rank Adaptation) for parameter-efficient fine-tuning

#### Acoustic Translation Mechanism
Input audio → Tokenizer → Partial masking → Iterative MATM decoding → Detokenizer → Coda-style output audio. The masking strategy can be flexibly designed, e.g., preserving beat positions to maintain rhythmic structure.

### Loss & Training

- Training objective: Standard masked token prediction loss (cross-entropy)
- Fine-tuning method: LoRA, updating only low-rank adaptation matrices
- Training completes on a single GPU in 5 days

## Key Experimental Results

### Main Results — Acoustic Translation Quality (FAD)

Fréchet Audio Distance (FAD) is computed using BirdNET embeddings and normalized to $[0,1]$:

| Audio Source | FAD Before Translation | FAD After Translation | FAD Indistinguishable |
|---|---|---|---|
| Natural codas (baseline) | — | 0.21 | Baseline threshold |
| 4 marine mammal species | High | < 0.21 | ✅ |
| Digital beeps | ~Same as coda | ~Same as coda | ✅ |
| 12 marine mammal species | High | Significantly reduced | Partial ✅ |

Key finding: Translations of 5 entirely distinct audio sources are FAD-indistinguishable from natural codas after processing by WhAM.

### Expert Perceptual Evaluation

| Task | Accuracy | Fleiss' $\kappa$ |
|---|---|---|
| Auditory-only 2AFC (Task 1) | 81% | 0.41 |
| Mixed classification (Task 2) | — | 0.44 |
| Spectrogram-assisted 2AFC (Task 3) | 83% | 0.41 |

- Natural codas are misclassified as synthetic at a rate of 36%
- Walrus-to-coda translations are identified by experts with only 75% accuracy

### Downstream Classification

| Task | WhAM | AVES | BirdNET | CLAP | Random |
|---|---|---|---|---|---|
| Coda detection | 91.3±0.2 | 92.8±0.1 | 93.0±1.0 | 96.8±1.4 | 60.9 |
| Rhythm classification | 87.4±1.6 | 90.4±1.6 | 88.6±0.2 | 92.4±2.4 | 66.3 |
| Clan classification | 70.5±5.6 | 92.0±0.7 | 93.2±0.1 | 85.5±1.4 | 42.5 |
| Ornament classification | 85.2±2.5 | 91.8±2.9 | 85.9±4.6 | 84.3±0.9 | 66.3 |

Despite being trained primarily for generation, WhAM substantially outperforms random baselines on all downstream classification tasks.

### Key Findings

1. WhAM, trained purely for generation, learns meaningful bioacoustic representations that support coda detection, rhythm, and ornament classification.
2. Strong results are achieved with only 5 days of single-GPU training on data orders of magnitude smaller than typical audio models.
3. Experts cannot reliably distinguish synthetic codas from real ones (accuracy only 81%).

## Highlights & Insights

- **Pioneering scope**: The first unified model for sperm whale coda acoustic translation, generation, and classification, and the first work to conduct expert perceptual evaluation of synthetic codas.
- **Effective transfer learning**: Music pretraining → animal acoustic domain adaptation → species-specific fine-tuning proves effective even under extreme data scarcity.
- **Cross-domain translation**: The ability to translate arbitrary audio (walrus calls, digital beeps, etc.) into coda style demonstrates the model's deep understanding of coda acoustic properties.
- **Generation as learning**: Pure generative training yields useful discriminative representations, consistent with the philosophy of self-supervised learning.

## Limitations & Future Work

1. **Codec bottleneck**: Only the MATM is fine-tuned while the codec remains frozen, potentially limiting faithful representation of ornament patterns in the 3.7–5.7 kHz band.
2. **Click fidelity**: Experts note that synthesized clicks exhibit overly abrupt onsets/decays and unnatural background noise.
3. **Data quality**: Training data contains echolocation sequences (non-communicative) mixed with codas.
4. **Semantic gap**: The current system achieves acoustic-level translation; semantic translation remains a distant goal.

## Related Work & Insights

- **VampNet** (García et al., 2023): The foundational architecture of this work.
- **AVES** (Hagiwara, 2023): A self-supervised bioacoustic encoder serving as the upper-bound reference for classification.
- **Sharma et al. (2024b)**: Temporal Transformer analysis of coda communication.
- **Implication**: The proposed framework is generalizable to the study of other animal communication systems.

## Rating

⭐⭐⭐⭐ (4/5)
Highly innovative with outstanding interdisciplinary value, though notable gaps remain in acoustic fidelity and semantic understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[AAAI 2026\] DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization](../../AAAI2026/audio_speech/deformtrace_a_deformable_state_space_model_with_relay_tokens_for_temporal_forger.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](../../AAAI2026/audio_speech/let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)

</div>

<!-- RELATED:END -->
