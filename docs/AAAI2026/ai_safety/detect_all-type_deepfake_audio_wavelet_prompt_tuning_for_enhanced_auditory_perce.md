---
title: >-
  [Paper Note] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception
description: >-
  [AAAI 2026][AI Safety][Deepfake Audio Detection] This work establishes the first all-type (speech/sound/singing/music) audio deepfake detection benchmark and proposes Wavelet Prompt Tuning (WPT) to enhance the full-frequency perception of SSL features using Discrete Wavelet Transform. Without adding training parameters, WPT outperforms full fine-tuning, achieving an average EER of only 3.58% after co-training.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Deepfake Audio Detection"
  - "Wavelet Prompt Tuning"
  - "Self-Supervised Learning"
  - "Cross-Type Detection"
  - "Frequency Domain Analysis"
date: 2026-05-08
content_hash: 122eb27f42c568bb
---

# Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception

**Conference**: AAAI 2026  
**arXiv**: [2504.06753](https://arxiv.org/abs/2504.06753)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Deepfake Audio Detection, Wavelet Prompt Tuning, Self-Supervised Learning, Cross-Type Detection, Frequency Domain Analysis

## TL;DR

This work establishes the first all-type (speech/sound/singing/music) audio deepfake detection benchmark and proposes Wavelet Prompt Tuning (WPT) to enhance the full-frequency perception of SSL features using Discrete Wavelet Transform. Without adding training parameters, WPT outperforms full fine-tuning, achieving an average EER of only 3.58% after co-training.

## Background & Motivation

The rapid development of audio generation technology has made it easy to synthesize any type of audio—including fake speech, sound, singing, and music, posing severe threats to media, cybersecurity, and political communication.

**Limitations of Prior Work**:

1. Current detection models perform well on single-type detection but show **extremely poor cross-type generalization**—models trained on speech perform close to random on sound/music.
2. No prior work has systematically studied the **all-type ADD task**, which involves simultaneously detecting four categories of deepfake audio: speech, sound, singing, and music.
3. SSL-based fine-tuning methods are effective but suffer from **huge parameter sizes and high sensitivity to hyperparameters**.
4. The primary difference in human perception of different audio types lies in the **frequency domain distribution**, but mainstream SSL models are designed for speech recognition and lack the ability to capture full-frequency domain information.

**Design Motivation**: There should exist some **type-invariant frequency-domain features** in deepfake characteristics across different audio types. Capturing such features would enable all-type detection.

## Method

### Overall Architecture

The overall architecture is **SSL-AASIST**, composed of an SSL front-end for feature extraction and an AASIST back-end for classification. Two parameter-efficient training paradigms are proposed:

1. **PT-SSL-AASIST (Prompt Tuning)**: Inserts learnable Prompt Tokens before each Transformer layer while freezing other SSL parameters.
2. **WPT-SSL-AASIST (Wavelet Prompt Tuning)**: Applies Discrete Wavelet Transform (DWT) to a portion of the Prompt Tokens to obtain tokens of different frequency bands, bolstering full-frequency domain perception.

### Key Designs

**PT-SSL-AASIST Design**:

- Zero-pads or truncates the input audio $X$ to a fixed length $L$, and obtains the initial embedding $E_0$ through the CNN feature extractor of the frozen SSL front-end.
- Initializes learnable Prompt Tokens $P_k$ for each Transformer layer (using Xavier uniform initialization).
- Layer-wise computation: $[Z_i, E_i] = L_i([P_i, E_{i-1}])$
- The Prompt output $Z_i$ from the previous layer is discarded and replaced with a new $P_i$.
- The final output $I = [Z_{24}, E_{24}]$ is fed into the AASIST back-end for time-frequency graph attention classification.

**WPT-SSL-AASIST Design**:

- Replaces some Prompt Tokens with wavelet Prompt Tokens, applying a Haar wavelet DWT.
- Haar wavelet low-pass filter $L = [1,1]/\sqrt{2}$, high-pass filter $H = [1,-1]/\sqrt{2}$.
- Calculates DWT on the initial wavelet Token $T_k$ to obtain four sub-bands:
    - **LL** (low-frequency component), **LH** (vertical high-frequency), **HL** (horizontal high-frequency), and **HH** (diagonal high-frequency).
    - Each component is of size $w/2 \times d/2$, reshaped to $w/4 \times d$.
    - Concatenates the four components to form the wavelet Prompt $W_k$.
- The layer-wise computation becomes: $[Z_i, E_i] = L_i([W_i, P_i, E_{i-1}])$
- Key Finding: Each Token corresponds to a specific frequency component, naturally aligning with the four frequency bands when $WPT=4$.
- **Parameter Efficiency**: PT/WPT only learns Prompt Tokens (approx. 0.69M parameters), which reduces parameters by **458 times** compared to full fine-tuning (FT, 315.89M).

### Loss & Training

- Employs **weighted cross-entropy (WCE) loss** to train the binary classifier (bonafide/spoof).
- Downsamples all audio to 16kHz, truncated/zero-padded to 64600 samples (approx. 4s).
- FT learning rate is 1e-6 with a batch size of 14; FR/PT/WPT learning rate is 5e-4 with a batch size of 32.
- Single-type training runs for 50 epochs (LR halved every 10 steps); co-training runs for 20 epochs (LR halved every 4 steps).
- SSL feature dimension is (201, 1024).

## Key Experimental Results

### Main Results

**Dataset Scale**:

| Type | Source | Train Set | Dev Set | Eval Set |
|---|---|---|---|---|
| Speech | 19LA | 25,380 | 24,844 | 71,237 |
| Sound | Codecfake-A3 | 69,378 | 9,911 | 19,823 |
| Singing | CtrSVDD | 84,404 | 43,625 | 92,769 |
| Music | FakeMusicCaps | 20,861 | 6,058 | 6,122 |
| All | Joint | 199,023 | 84,438 | 189,951 |

**Co-training Results (EER%)**:

| Detection Model | Speech | Sound | Singing | Music | Average |
|---|---|---|---|---|---|
| Spec-Resnet | 29.37 | 23.37 | 37.17 | 42.75 | 33.17 |
| AASIST | 3.78 | 0.86 | 20.01 | 11.70 | 9.09 |
| FR-XLSR-AASIST | 3.02 | 5.45 | 10.86 | 22.67 | 10.50 |
| FT-XLSR-AASIST | 1.77 | 0.49 | 8.93 | 8.71 | 4.98 |
| PT-XLSR-AASIST | 2.00 | 1.11 | 14.54 | 9.29 | 6.74 |
| **WPT-XLSR-AASIST** | **0.72** | **1.29** | **7.47** | **4.83** | **3.58** |

WPT achieves an average EER of 3.58% with only 0.69M parameters (1/458 of FT), outperforming FT (4.98%) which requires 315.89M parameters.

**Cross-type Generalization of Single-type Training**:

| Trained Type | Best Model | In-domain EER | Average EER | Cross-type Finding |
|---|---|---|---|---|
| Speech | FR-XLSR-AASIST | 1.28% | 32.58% | Speech generalizes reasonably well to singing |
| Sound | AASIST | 0.43% | 22.71% | Correlation exists between sound and music |
| Singing | FR-XLSR-AASIST | 9.45% | 23.16% | Singing generalizes well to speech |
| Music | FR-MERT-AASIST | 7.62% | 28.63% | MERT is strongest on music |

### Ablation Study

**Ablation on Prompt Token Count** (XLSR-AASIST trained on speech):

| Token Count | Params | Speech EER | Average EER |
|---|---|---|---|
| 2 | 0.50M | 0.75% | 30.94% |
| 10 | 0.69M | **0.22%** | **30.79%** |
| 100 | 2.90M | 3.01% | 31.28% |
| 200 | 5.36M | 4.99% | 33.36% |

Excessive Prompt Tokens dilute the information density of audio tokens, with 10 being the optimal count.

**Comparison of Training Paradigms** (XLSR-AASIST trained on speech):

| Model | Params | Speech EER | Average EER |
|---|---|---|---|
| FR-XLSR-AASIST | 0.45M | 1.28% | 32.58% |
| FT-XLSR-AASIST | 315.89M | 0.38% | 27.68% |
| PT-XLSR-AASIST | 0.69M | 0.22% | 30.79% |
| WPT-XLSR-AASIST | 0.69M | 0.15% | 26.86% |

### Key Findings

- **Shared features exist between speech and singing**, yielding the best cross-type generalization; a correlation also exists between sound and music.
- WPT learns **type-invariant deepfake detection prompts**: In t-SNE visualizations, WPT's genuine/spoof samples do not cluster by audio type.
- Attention map analysis: WPT focuses on the 4th token (corresponding to the **HH band—diagonal high frequency**), which remains consistent across all audio types.
- FR/PT/WPT converge significantly faster than FT with fewer fluctuations.

## Highlights & Insights

1. **Pioneering All-Type ADD Benchmark**: Fills a gap in the field by covering four categories of deepfake detection: speech, sound, singing, and music.
2. **Ingenious Wavelet Prompt Design**: Integrates DWT into the Prompt Token initialization, acquiring full-frequency domain perception without adding training parameters.
3. **458x Gain in Parameter Efficiency**: WPT requires only 0.69M parameters, outperforming the full fine-tuning (FT) method requiring 315.89M.
4. **Discovery of Type Invariance in the HH Band**: Attention maps clearly demonstrate that WPT focuses on the HH Token, providing new insights for frequency-domain deepfake detection.
5. **Rigorous and Systematic Evaluation**: The step-by-step progress from single-type to cross-type and joint training is extremely clear.

## Limitations & Future Work

1. The benchmark datasets only represent relatively clean environments, without considering more complex, real-world scenarios like noise or partial spoofing.
2. All audios are truncated to roughly 4 seconds; applicability to long audios is unverified.
3. WPT only uses the Haar wavelet (the simplest wavelet transform); more complex wavelet families might yield better performance.
4. Detection EER for music remains significantly higher than that for speech, indicating room for improvement in unified cross-type detection.
5. The robustness of detection under adversarial attack scenarios is not investigated.

## Related Work & Insights

- **XLSR-AASIST** (Tak et al. 2022): The classic SSL+AASIST framework, used as the baseline.
- **Visual Prompt Tuning** (Jia et al. 2022): A prompt tuning method in the visual domain, inspiring PT-SSL.
- **CtrSVDD** (Zhang et al. 2024): The first singing voice deepfake detection challenge.
- **FakeMusicCaps** (Comanducci et al. 2024): The first music deepfake detection dataset.
- **Key Insight**: Frequency-domain information is a vital bridge for cross-modal/cross-type detection. DWT sub-band decomposition provides a natural and efficient way to inject frequency-domain features.

## Rating

- Novelty: 5/5 - Both the all-type ADD benchmark and the WPT method are pioneered by this work.
- Technical Depth: 4/5 - The combination of wavelet transform and prompt tuning is ingeniously designed.
- Experimental Thoroughness: 5/5 - Covers single-type, cross-type, and co-training, plus multiple ablations and visualization analyses.
- Writing Quality: 4/5 - Well-structured with abundant charts.
- Overall: 4.5/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification](../../NeurIPS2025/ai_safety/not_all_deepfakes_are_created_equal_triaging_audio_forgeries_for_robust_deepfake.md)
- [\[CVPR 2026\] X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection](../../CVPR2026/ai_safety/x-avdt_audio-visual_cross-attention_for_robust_deepfake_detection.md)
- [\[ICML 2026\] One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](../../ICML2026/ai_safety/one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](../../ICCV2025/ai_safety/fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)

</div>

<!-- RELATED:END -->
