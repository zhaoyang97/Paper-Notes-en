---
title: >-
  [Paper Note] Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models
description: >-
  [ACL 2025][LLM (Other)][discrete speech tokens] This paper reveals the phenomenon of Discrete Representation Inconsistency (DRI) in neural audio codecs (such as EnCodec)—where the same audio segment is encoded into different token sequences depending on the presence of context. It proposes two constraint methods: slice consistency and perturbation consistency, improving representation consistency by 21-36%, which leads to a 3.72% reduction in Word Error Rate (WER) and a 5.68%…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "discrete speech tokens"
  - "neural audio codecs"
  - "representation inconsistency"
  - "VALL-E"
  - "speech generation"
date: 2026-05-08
content_hash: 4cd2da70ca36a0f6
---

# Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models

**Conference**: ACL 2025  
**arXiv**: [2409.19283](https://arxiv.org/abs/2409.19283)  
**Code**: [https://consistencyinneuralcodec.github.io](https://consistencyinneuralcodec.github.io)  
**Area**: LLM/NLP  
**Keywords**: discrete speech tokens, neural audio codecs, representation inconsistency, VALL-E, speech generation

## TL;DR
This paper reveals the phenomenon of Discrete Representation Inconsistency (DRI) in neural audio codecs (such as EnCodec)—where the same audio segment is encoded into different token sequences depending on the presence of context. It proposes two constraint methods: slice consistency and perturbation consistency, improving representation consistency by 21-36%, which leads to a 3.72% reduction in Word Error Rate (WER) and a 5.68% improvement in speaker similarity in VALL-E speech generation.

## Background & Motivation

**Background**: Large language model-based speech generation (e.g., VALL-E, SpeechGPT) adopts neural audio codecs (such as EnCodec, SpeechTokenizer) to quantize continuous audio signals into discrete token sequences, and then generates speech by predicting tokens auto-regressively. This paradigm has demonstrated great potential in zero-shot speech synthesis.

**Limitations of Prior Work**: Although autoregressive modeling improves the naturalness and zero-shot capabilities of speech, the Word Error Rate (WER) of synthesized speech remains high, often suffering from omissions and repetitions. The core issue lies in the fact that the same audio segment is encoded into different discrete token sequences within different contexts, causing a many-to-one mapping problem.

**Key Challenge**: Text tokenizers are context-free—the same text is always encoded into the same token regardless of whether context is present. However, encoders in audio tokenizers utilize convolutional layers to incorporate context information for a higher compression rate and reconstruction quality, at the cost of breaking the determinism of the token sequences. This trade-off is a fundamental dilemma of audio discretization.

**Goal**: To quantitatively analyze the severity of the DRI phenomenon and to improve the consistency of discrete tokens without sacrificing audio reconstruction quality.

**Key Insight**: The authors observe that simply reducing the convolution kernel size, while improving consistency, severely degrades reconstruction quality. Consequently, they shift to adding consistency constraints during training, allowing the model to learn to produce more consistent representations while retaining its receptive field.

**Core Idea**: Mitigate the DRI problem from two complementary perspectives through two regularization methods: slice consistency (requiring consistent encoding between a segment and the full audio) and perturbation consistency (requiring consistent encoding before and after subtle phase perturbations).

## Method

### Overall Architecture
Building upon the standard RVQ-GAN audio codec training framework, consistency constraint losses are introduced. During training, the input audio undergoes both slicing and phase perturbation, forcing their latent representations to be as close as possible. At inference time, the usage of the codec remains unchanged, but the encoder's sensitivity to context is reduced due to the consistency constraints applied during training.

### Key Designs

1. **Slice-Consistency Method**:

    - **Function**: Eliminate the influence of context information on encoding results.
    - **Mechanism**: Randomly crop a segment from the full audio. Input both the full audio and the cropped segment into the encoder to obtain two sets of latent representations $Z$ and $Z^{slice}$. Constrain the representations at corresponding positions between $Z^{slice}$ and $Z$ to be consistent using MSE loss: $\mathcal{L}_{slice} = \frac{1}{T}\sum_{t=1}^{T}\text{MSE}(Z^{slice}[t], Z[t])$.
    - **Design Motivation**: Audio clips without context lack surrounding information, and their encoding differences compared to the full audio directly reflect context-induced inconsistency. Constraining the two to align forces the encoder to decrease its dependency on context.

2. **Perturbation-Consistency Method**:

    - **Function**: Improve the encoder's robustness to minor changes imperceptible to the human audio system.
    - **Mechanism**: Apply a slight phase perturbation (which does not affect perception) to the original audio, input the perturbed audio into the encoder to obtain $Z^{perception}$, and use MSE to constrain its consistency with the original representation $Z$.
    - **Design Motivation**: Phase changes are imperceptible to human ears but can lead codecs to generate completely different token sequences. Enhancing the encoder's invariance to such harmless changes via perturbation consistency resolves this.

3. **Joint Implementation**:

    - **Function**: Efficiently satisfy both consistency constraints simultaneously.
    - **Mechanism**: Directly align $Z^{slice}$ from slice consistency with $Z^{perception}$ from perturbation consistency: $\mathcal{L}_{consistency} = \frac{1}{T}\sum_{t=1}^{T}\text{MSE}(Z^{slice}[t], Z^{perception}[t])$. This allows a single forward pass to simultaneously constrain both types of consistency, boosting training efficiency.
    - **Design Motivation**: Since $Z^{slice}$ lacks context information and $Z^{perception}$ contains micro-perturbations, aligning them simultaneously addresses both context dependence and perturbation sensitivity.

### Loss & Training
The total loss is: $\mathcal{L} = \mathcal{L}_{rec} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{fm}\mathcal{L}_{fm} + \lambda_{rvq}\mathcal{L}_{rvq} + \lambda_{con}\mathcal{L}_{consistency}$, where the weight of consistency constraint is $\lambda_{con}=10.0$. Training is conducted based on the RVQ-GAN framework for 350K steps with a batch size of 384 and an audio crop length of 1.28 seconds.

## Key Experimental Results

### Main Results

| Model | Bandwidth | Consistency (All Layers)↑ | Consistency (First 3 Layers)↑ | ViSQOL↑ | PESQ↑ |
|------|------|-------------|--------------|---------|-------|
| EnCodec | 4.5kbps | 47.43% | 61.49% | 4.25 | 2.41 |
| SpeechTokenizer | 4.0kbps | 14.70% | 26.91% | 4.36 | 2.62 |
| DAC | 4.0kbps | 39.14% | 48.43% | 4.44 | 2.68 |
| FunCodec | 4.0kbps | 6.86% | 16.39% | 4.47 | 3.26 |
| **Ours** | **4.0kbps** | **71.03%** | **88.82%** | **4.45** | **3.25** |

| Neural Codec Speech Model | WER↓ | SIM↑ | UTMOS↑ | MOS↑ | SMOS↑ |
|-------------------|------|------|--------|------|-------|
| VALL-E (w/o consistency) | 4.73 | 76.95% | 4.10 | 3.73 | 3.50 |
| **VALL-E (Ours, 960h)** | **1.84** | **83.71%** | **4.31** | **3.97** | **3.73** |
| **VALL-E (Ours, 44Kh)** | **1.37** | **84.14%** | **4.30** | **4.02** | **3.95** |
| Ground Truth | 1.37 | / | 4.15 | 4.43 | 4.23 |

### Ablation Study

| Configuration | Consistency (All Layers) | WER↓ | SIM↑ |
|------|------------|------|------|
| Slice 20% + Phase Perturb | 76.75% | 1.84 | 83.71% |
| Phase Perturb Only (No Slice) | 7.03% | 2.24 | 77.09% |
| Slice 20% Only (No Perturb) | 75.91% | 2.36 | 81.84% |
| No Consistency Constraints | 6.94% | 4.73 | 76.95% |
| Slice 40% + Phase Perturb | 64.74% | 1.90 | 82.81% |
| Slice 60% + Phase Perturb | 31.79% | 3.02 | 82.41% |

### Key Findings
- Slice consistency is the primary contributor, improving consistency from 6.94% to 75.91% when used alone; perturbation consistency plays an auxiliary role.
- A shorter slice ratio (20%) works best, because short segments contain almost no context information and can more effectively guide the encoder to learn context-independent representations.
- The improvement in consistency primarily affects deeper codebooks—the consistency of shallow codebooks is inherently high (storing semantic info), while deep codebooks store sensitive acoustic details.
- The method remains effective when the data scale is scaled up from 960 hours to 44,000 hours, with WER decreasing from 1.84 to 1.37.

## Highlights & Insights
- The discovery and quantitative analysis of the DRI phenomenon are pioneering—explicitly pointing out the fundamental difference that "text tokenizers are context-free, while audio tokenizers are not," providing a new perspective of understanding for the speech LLM field.
- The method is highly generalizable—consistency constraints can be integrated into any RVQ-based audio codec training pipeline without changing the model architecture, resulting in minimal implementation overhead.
- The finding of "semantic info in shallow layers, acoustic info in deep layers" can be transferred to hierarchical usage strategies of speech tokens, such as performing autoregressive prediction only on shallow tokens and non-autoregressive prediction on deep tokens in TTS.

## Limitations & Future Work
- Currently, evaluation has only been performed on English data; the DRI phenomenon might be more severe in multilingual scenarios.
- Phase perturbation is the only perturbation method utilized; other perturbations imperceptible to the human audio system (such as minor volume changes or slight speed jitter) have not been explored.
- Consistency constraints slightly increase the training time (requiring additional forward passes), but the authors did not report the specific training overhead.
- Future work could explore utilizing consistency information to guide token decoding during inference, such as resampling at positions with low consistency.

## Related Work & Insights
- **vs EnCodec (Défossez et al., 2022)**: EnCodec is the most popular audio tokenizer, but its all-layer consistency is only 47.43%. Our method directly boosts it to over 71%.
- **vs SpeechTokenizer (Zhang et al., 2023)**: SpeechTokenizer attempts to disentangle semantic and acoustic information, but its consistency remains low (14.70%), indicating that disentanglement does not fundamentally solve the DRI problem.
- **vs LLM-Codec (Yang et al., 2024)**: LLM-Codec noticed the inconsistency but did not propose a solution; ours fills this gap.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery and systematic analysis of the DRI phenomenon are highly original, and the method is simple yet precise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 baseline codecs, multiple data scales, subjective and objective evaluations, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, the analytical logic is rigorous, and the charts and tables are highly informative.
- Value: ⭐⭐⭐⭐⭐ Provides an important foundational contribution to the speech LLM field, with a method that is simple, practical, and easy to scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)
- [\[ACL 2025\] Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [\[ACL 2025\] Information Locality as an Inductive Bias for Neural Language Models](information_locality_as_an_inductive_bias_for_neural_language_models.md)
- [\[ACL 2025\] Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models](attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)
- [\[ACL 2025\] Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models](locateandfocus_enhancing_terminology_translation_in_speech.md)

</div>

<!-- RELATED:END -->
