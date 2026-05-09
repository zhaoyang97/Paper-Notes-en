---
title: >-
  [Paper Note] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification
description: >-
  [NeurIPS 2025 Workshop (Generative and Protective AI for Content Creation)][AI Safety][deepfake detection] This paper proposes a two-stage pipeline grounded in the premise that the most harmful deepfakes are those of the highest quality. A discriminator first filters out low-quality forgeries to reduce noise; a singer identification model trained exclusively on genuine recordings then performs voiceprint matching. The pipeline consistently outperforms baselines across multiple datasets.
tags:
  - NeurIPS 2025 Workshop (Generative and Protective AI for Content Creation)
  - AI Safety
  - deepfake detection
  - singer identification
  - voice forgery
  - two-stage pipeline
  - audio forensics
date: 2026-05-08
content_hash: 3120c5ee7bd1e2de
---

# Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification

**Conference**: NeurIPS 2025 Workshop (Generative and Protective AI for Content Creation)
**arXiv**: [2510.17474](https://arxiv.org/abs/2510.17474)
**Code**: None
**Area**: AI Security, Audio Deepfake Detection
**Keywords**: deepfake detection, singer identification, voice forgery, two-stage pipeline, audio forensics

## TL;DR

This paper proposes a two-stage pipeline grounded in the premise that the most harmful deepfakes are those of the highest quality. A discriminator first filters out low-quality forgeries to reduce noise; a singer identification model trained exclusively on genuine recordings then performs voiceprint matching. The pipeline consistently outperforms baselines across multiple datasets.

## Background & Motivation

Advances in singing voice cloning have made it possible to generate deepfake singing voices that are nearly indistinguishable from authentic recordings, posing serious threats to artists' portrait rights and content authenticity. Existing research follows two largely separate directions:

- **Deepfake detection**: determining whether audio is synthesized (real/fake binary classification)
- **Singer identification**: verifying singer identity

However, the intersection problem of **identifying singers within deepfakes** has received little attention. The core challenge is that singer voices in low-quality deepfakes are inherently unrecognizable, degrading identification model performance. This paper argues that **harm correlates positively with deepfake quality**—high-quality deepfakes constitute the real threat, while low-quality ones are easier to detect and dismiss.

## Method

### Overall Architecture

A two-stage pipeline (Figure 1):

1. **Stage 1 — Discriminator $\mathcal{D}$**: Filters out low-quality deepfakes. Given an input recording, it outputs a real/fake binary classification. Low-quality deepfakes are readily detected due to their conspicuous artifacts.
2. **Stage 2 — Singer Identification $\mathcal{S}$**: For audio deemed "real" or "high-quality deepfake" by the discriminator, voiceprint embeddings are extracted and matched against a reference database via cosine distance to identify singer identity.

### Key Designs

1. **Discriminator $\mathcal{D}$ (LCNN)**: A Light Convolutional Neural Network deliberately kept simple—by design, it can only detect low-quality deepfakes and is "fooled" by high-quality ones. Trained on the CTRSVDD dataset with mel-spectrogram inputs (512 FFT bins, 80 mel bins), using BCE loss with random oversampling for class balancing.

2. **Singer Identification Model $\mathcal{S}$ (ECAPA-TDNN)**: An ECAPA-TDNN architecture adapted from speaker verification, **trained exclusively on genuine recordings** (no paired deepfake data required), serving as a multi-class classifier. Training data comprises 134,826 commercial recordings from 2,000 singers. Data augmentation includes random background music, noise injection, and pitch shifting.

3. **Source Separation Preprocessing**: Separated versions of all datasets are created using BS-RoFormer for vocal/accompaniment separation, followed by energy-based VAD to remove non-vocal segments, focusing training on vocal-containing samples.

### Inference Strategy

- Raw recordings (without source separation) are used at inference time; 5 non-overlapping 10-second windows are extracted per song.
- Discriminator $\mathcal{D}$: predictions across windows are averaged.
- Identification model $\mathcal{S}$: embeddings from the final fully connected layer are averaged, and cosine distance is used for matching.

## Key Experimental Results

### Singer Identification Model Comparison (Without Discriminator, EER↓ / AUC↑)

| Model | Private EER | Artist20 EER | CTRSVDD EER | WildSVDD EER | Avg. EER | Avg. AUC |
|-------|-------------|--------------|-------------|--------------|----------|----------|
| ECAPA-TDNN | **4.31** | **15.56** | **30.34** | **19.24** | **17.36** | **88.32** |
| SSL | 16.13 | 25.30 | 36.34 | 32.92 | 27.67 | 78.65 |
| ResNet-TDNN | 8.70 | 23.05 | 31.46 | 21.38 | 21.15 | 85.56 |

ECAPA-TDNN achieves consistently superior performance across all datasets. ResNet-TDNN (speech-pretrained) approaches ECAPA-TDNN only on the vocal-only CTRSVDD dataset, with a notable gap on datasets containing accompaniment.

### Two-Stage Pipeline Performance (ECAPA-TDNN)

| Dataset | Pipeline | EER (%) ↓ | AUC (%) ↑ |
|---------|----------|-----------|-----------|
| CTRSVDD | $\mathcal{S}$ only | 30.34 | 76.11 |
| CTRSVDD | $\mathcal{D} \circ \mathcal{S}$ | **16.82** | **88.90** |
| WildSVDD | $\mathcal{S}$ only | 19.24 | 87.41 |
| WildSVDD | $\mathcal{D} \circ \mathcal{S}$ | **15.55** | **91.55** |
| **Average** | $\mathcal{S}$ only | 24.79 | 81.76 |
| **Average** | $\mathcal{D} \circ \mathcal{S}$ | **16.19** | **90.23** |

Incorporating the discriminator reduces average EER from 24.79% to 16.19% and raises AUC from 81.76% to 90.23%, representing a substantial improvement.

### Per-Algorithm Identification Performance (CTRSVDD, ECAPA-TDNN)

| Algorithm | EER (%) | AUC (%) | Quality Assessment |
|-----------|---------|---------|-------------------|
| A02 | 8.83 | 96.94 | High quality |
| REAL | 10.73 | 95.48 | Genuine |
| A04 | 11.68 | 95.57 | High quality |
| A01 | 13.88 | 93.51 | High quality |
| A07 | 36.02 | 68.67 | Low quality |
| A08 | 36.05 | 69.17 | Low quality |
| A10 | 33.98 | 71.12 | Low quality |

Singer identification on high-quality deepfakes (A01–A05) approaches the performance on genuine recordings (EER < 15%), whereas low-quality deepfakes (A07–A10, A13) yield EER > 30%, as their voiceprints bear little resemblance to the target singer.

### Key Findings

- **Deepfake quality strongly correlates with identification difficulty**: voiceprint distortion in low-quality deepfakes degrades model performance, yet these low-quality forgeries pose less harm in practice.
- **The discriminator's false positive rate is very low**: the probability of misclassifying genuine recordings as deepfakes is minimal across all datasets, ensuring authentic recordings are not erroneously flagged.
- **WildSVDD exhibits a higher false negative rate than CTRSVDD**: presumably because WildSVDD contains more high-quality deepfakes that evade the simple discriminator—precisely the intended behavior of the pipeline design.
- **Advantage of training on genuine recordings only**: this avoids the difficulty of acquiring paired deepfake data and facilitates scalability.

## Highlights & Insights

- The core insight that "not all deepfakes are equal"—linking deepfake quality to harm—motivates a pragmatic, risk-tiered defense strategy.
- The two-stage pipeline is elegantly simple: the discriminator is intentionally kept weak, leaving high-quality deepfakes to be handled by the identification model.
- Training the singer identification model on genuine recordings alone makes the approach operationally feasible and highly scalable.
- The per-algorithm performance breakdown provides a valuable diagnostic perspective.

## Limitations & Future Work

- As a workshop paper, the scale is limited: only 6 artists (WildSVDD) and a restricted set of deepfake generation algorithms are covered.
- A systematic quantitative measure of perceived deepfake quality is absent; the quality–harm correlation is primarily supported by empirical observation rather than rigorous quantitative evaluation.
- Both the discriminator and identification model use fixed architectures and training strategies without extensive architecture search.
- The private dataset (134K recordings) is not publicly reproducible.
- End-to-end training or joint optimization of the two stages is not explored.

## Related Work & Insights

- The work extends the spirit of audio fingerprinting (e.g., Shazam) for recording rights protection to the domain of "voiceprint portrait rights."
- The successful transfer of ECAPA-TDNN from speaker verification to singer identification demonstrates the cross-domain generalization capacity of voiceprint representations.
- Future work could incorporate perceptual deepfake quality metrics (e.g., MOS) to more rigorously validate the quality–harm hypothesis.

## Rating

- **Novelty**: ⭐⭐⭐ — The problem framing is insightful, though the technical solution is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐ — Multi-dataset evaluation is provided, but scope and depth are constrained by the workshop format.
- **Writing Quality**: ⭐⭐⭐⭐ — The exposition is clear and the motivation is well-articulated.
- **Value**: ⭐⭐⭐⭐ — The work has practical relevance for music copyright protection, and the two-stage paradigm is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](../../AAAI2026/ai_safety/detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)
- [\[CVPR 2026\] Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](../../CVPR2026/ai_safety/tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[ACL 2026\] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection](../../ACL2026/ai_safety/xlsr-mambo_scaling_the_hybrid_mamba-attention_backbone_for_audio_deepfake_detect.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](../../ICCV2025/ai_safety/fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)

</div>

<!-- RELATED:END -->
