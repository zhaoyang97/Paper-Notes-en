---
title: >-
  [Paper Note] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time
description: >-
  [NeurIPS 2025][Audio & Speech][audio-visual robustness] This paper proposes AVRobustBench, the first benchmark that systematically evaluates the test-time robustness of audio-visual models under **co-occurring correlated dual-modality corruptions**, comprising 4 datasets × 75 corruption types, and introduces AV2C, a TTA method based on low-entropy sample selection.
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - audio-visual robustness
  - distribution shift
  - test-time adaptation
  - multimodal benchmark
  - co-occurring corruptions
date: 2026-05-08
content_hash: 23dfe037b76b49cb
---

# AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time

**Conference**: NeurIPS 2025
**arXiv**: [2506.00358](https://arxiv.org/abs/2506.00358)
**Code**: Available (mentioned in the paper)
**Area**: Audio/Speech (Multimodal Robustness)
**Keywords**: audio-visual robustness, distribution shift, test-time adaptation, multimodal benchmark, co-occurring corruptions

## TL;DR

This paper proposes AVRobustBench, the first benchmark that systematically evaluates the test-time robustness of audio-visual models under **co-occurring correlated dual-modality corruptions**, comprising 4 datasets × 75 corruption types, and introduces AV2C, a TTA method based on low-entropy sample selection.

## Background & Motivation

**State of the Field**: Audio-visual models such as UAVM, CAV-MAE, and ImageBind have achieved remarkable progress, yet their robustness to test-time distribution shifts remains largely understudied. Existing robustness benchmarks focus primarily on single modalities (e.g., ImageNet-C for images) or apply uncorrelated perturbations across modalities.

**Limitations of Prior Work**: (a) Existing benchmarks corrupt only a single modality, or apply corruptions to each modality independently; (b) In real-world scenarios (e.g., autonomous driving in rain), corruptions affect audio and video simultaneously and in a correlated manner; (c) No systematic evaluation of state-of-the-art audio-visual models under co-occurring corruptions exists.

**Root Cause**: Audio-visual models perform well on clean data, but their robustness when both modalities are simultaneously corrupted remains entirely unknown.

**Paper Goals**: To construct a comprehensive audio-visual robustness benchmark that systematically evaluates state-of-the-art supervised/self-supervised models and TTA methods.

**Starting Point**: Designing 15 real-world-inspired audio-visual co-occurring correlated corruptions (5 severity levels) spanning three categories: digital, environmental, and adversarial.

**Core Idea**: Real-world distribution shifts affect audio and video simultaneously and in a correlated fashion — existing models and TTA methods fail severely under such conditions.

## Method

### Overall Architecture

AVRobustBench comprises:
- **4 benchmark datasets**: AudioSet-2C (16,742 samples, 527 classes), VGGSound-2C (14,046 samples, 309 classes), Kinetics-2C (3,111 samples, 32 classes), EpicKitchens-2C (205 samples, 97+300 classes)
- **75 corruptions**: 15 types × 5 severity levels
- **Evaluation**: 6 supervised models + 3 self-supervised models + 6 TTA methods

### Key Designs

#### Corruption Taxonomy (15 types, applied synchronously to audio and video)

| Category | Video Corruption | Audio Corruption |
|----------|-----------------|-----------------|
| **Digital** | Gaussian/Impulse/Shot/Speckle noise + JPEG compression | Corresponding noise (SNR-controlled) + DCT quantization |
| **Environmental** | Snow/Frost/Spatter/Wind (motion blur)/Rain/Underwater | Snow/frost/droplet/wind/rain/underwater sounds |
| **Adversarial** | Concert (brightness shift)/Smoke (haze)/Crowd (shadow occlusion)/Interference (random rotation) | Music noise/alarm/crowd noise/random silence |

Key property: all corruptions are applied **simultaneously** to both audio and video, and are **correlated** (e.g., rain simultaneously introduces raindrop visuals and rain audio).

#### Evaluation Metrics
- **Accuracy/mAP**: Classification accuracy under corruption
- **Absolute robustness** $\alpha_{i,s} = 1 - \frac{\delta A}{100}$
- **Relative robustness** $\rho_{i,s} = 1 - \frac{\delta A}{A_{cl}}$, where $\delta A = A_{cl} - A_{i,s}$

#### AV2C — Proposed TTA Method
- Adapts QKV attention weights (analogous to READ)
- Minimizes weighted Shannon entropy, assigning higher weights to low-entropy (reliable) samples
- Selects diverse samples based on similarity between current predictions and historical exponential moving averages

### Loss & Training

- Evaluation uses frozen pretrained models (standard robustness benchmark protocol)
- TTA experiments: batch=16, single forward+backward pass
- AV2C: online adaptation of QKV weights in the CAV-MAE joint encoder

## Key Experimental Results

### Main Results — Test-Time Robustness (Severity=5)

| Model | VGGSound-2C mAcc | Drop | $\rho$ | Kinetics-2C mAcc | Drop |
|-------|-----------------|------|--------|-----------------|------|
| UAVM | 27.41 | -38.39 | 0.42 | 48.06 | -30.06 |
| CAV-MAE | 35.54 | -29.96 | 0.54 | 58.15 | -29.95 |
| EquiAV | 33.78 | -28.12 | 0.55 | **63.73** | -22.29 |
| AudioCLIP | 11.14 | -15.64 | 0.41 | 23.57 | -27.44 |
| ImageBind | 10.25 | -17.93 | 0.36 | 26.82 | -25.64 |
| Wav2CLIP | 4.99 | -19.33 | 0.21 | 17.25 | -35.40 |

All models exhibit **significant performance degradation** at severity level 5.

### TTA Results (VGGSound-2C, Severity=5)

| TTA Method | Mean Accuracy | vs. Source |
|-----------|--------------|-----------|
| Source (CAV-MAE) | 35.54 | — |
| TENT | 19.09 | -16.45 |
| SAR | 26.07 | -9.47 |
| EATA | 40.60* | +5.06 |
| READ | 35.28 | -0.26 |
| SuMi | 32.10 | -3.44 |
| **AV2C (ours)** | **40.60** | **+5.06** |

Note: EATA and AV2C achieve the best results on VGGSound-2C. TENT degrades severely.

### Ablation Study — Effect of Corruption Severity

- The $\rho$ of all models decreases monotonically as severity increases
- Exception: the Interference corruption has relatively little impact on robustness (some models remain recognizable even under heavy frame rotation and audio silencing)
- Digital corruptions (e.g., Gaussian noise) cause the largest performance drops

### Key Findings

1. **Supervised models**: EquiAV > CAV-MAE > UAVM; equivariant feature learning yields better robustness
2. **Self-supervised models**: ImageBind's zero-shot generalization fails under corruption, with $\rho$ as low as 0.21 (Wav2CLIP on VGGSound)
3. **TTA methods broadly fail**: LayerNorm updates in TENT/RPL/SAR lead to overfitting under dual-modality corruption
4. **Modality bias in READ**: Under dual-modality corruption, cross-attention exhibits progressively increasing modality bias (visual→audio weight increases from 13.09 at $t=0$ to 20.04 at $t=100$)
5. **Prompt engineering is ineffective**: Switching ImageBind to noise-aware prompts yields only marginal improvement

## Highlights & Insights

- **First co-occurring correlated corruption benchmark**: Fills a critical gap in audio-visual robustness evaluation
- **Comprehensive failure analysis**: Systematically exposes the vulnerability of supervised/self-supervised models and TTA methods under dual-modality corruption
- **Discovery of modality bias**: The shift in attention weights during adaptation in READ constitutes an interesting and previously unreported failure mode
- **Simplicity of AV2C**: Low-entropy sample selection combined with QKV adaptation is conceptually straightforward yet effective

## Limitations & Future Work

1. AV2C achieves significant improvement only on VGGSound-2C; gains on Kinetics-2C are marginal
2. The corruption suite can be further expanded (e.g., network latency, codec distortions)
3. Large-scale foundation models (e.g., VideoLLaMA) are not evaluated
4. Only online TTA is considered; offline or few-shot adaptation strategies are not explored

## Related Work & Insights

- **ImageNet-C** (Hendrycks & Dietterich, 2019): Pioneering work on single-modality visual robustness benchmarking
- **READ** (2024): The first audio-visual TTA method, yet shown to fail under dual-modality corruption
- **TENT** (Wang et al., 2020): Foundational TTA method based on entropy minimization
- **Insight**: The robustness of multimodal models cannot be simply inferred from single-modality results — co-occurring corruptions present fundamentally new challenges

## Rating

⭐⭐⭐⭐ (4/5)
The benchmark construction is solid and comprehensive, with broad experimental coverage; however, the proposed AV2C method offers limited improvement, and the work is more diagnostic than prescriptive.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)
- [\[NeurIPS 2025\] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition](mome_mixture_of_matryoshka_experts_for_audio-visual_speech_recognition.md)
- [\[NeurIPS 2025\] Instance-Specific Test-Time Training for Speech Editing in the Wild](instance-specific_test-time_training_for_speech_editing_in_the_wild.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)
- [\[ICLR 2026\] When and Where to Reset Matters for Long-Term Test-Time Adaptation](../../ICLR2026/audio_speech/when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)

<!-- RELATED:END -->
