---
title: >-
  [Paper Note] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception
description: >-
  [AAAI 2026][AI Safety][deepfake audio detection] This paper establishes the first all-type (speech/sound/singing/music) audio deepfake detection benchmark and proposes Wavelet Prompt Tuning (WPT), which enhances full-band frequency perception of SSL features via discrete wavelet transform. Without increasing trainable parameters, WPT surpasses full fine-tuning and achieves an average EER of only 3.58% under co-training.
tags:
  - AAAI 2026
  - AI Safety
  - deepfake audio detection
  - wavelet prompt tuning
  - self-supervised learning
  - cross-type detection
  - frequency domain analysis
date: 2026-05-08
content_hash: 12136cfbb362872b
---

# Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception

**Conference**: AAAI 2026
**arXiv**: [2504.06753](https://arxiv.org/abs/2504.06753)
**Code**: None
**Area**: AI Security
**Keywords**: deepfake audio detection, wavelet prompt tuning, self-supervised learning, cross-type detection, frequency domain analysis

## TL;DR

This paper establishes the first all-type (speech/sound/singing/music) audio deepfake detection benchmark and proposes Wavelet Prompt Tuning (WPT), which enhances full-band frequency perception of SSL features via discrete wavelet transform. Without increasing trainable parameters, WPT surpasses full fine-tuning and achieves an average EER of only 3.58% under co-training.

## Background & Motivation

The rapid development of audio generation technology has made it trivial to synthesize any type of audio—including forged speech, environmental sounds, singing voices, and music—posing threats to media integrity, cybersecurity, and political communication.

**Existing Problems**:

1. Current detection models perform well on single-type detection but exhibit **extremely poor cross-type generalization**—models trained on speech perform near-randomly on sound/music.
2. No prior work has systematically studied the **all-type ADD task**, i.e., simultaneously detecting deepfakes across speech, sound, singing, and music.
3. SSL-based fine-tuning methods are effective but suffer from **enormous parameter counts and sensitivity to hyperparameters**.
4. Human perception of different audio types primarily differs in **frequency distribution**, yet mainstream SSL models are designed for speech recognition and lack full-band frequency capture capability.

**Key Motivation**: Deepfake artifacts across different audio types likely share some **type-invariant frequency-domain characteristics**—capturing such characteristics would enable all-type detection.

## Method

### Overall Architecture

The overall architecture follows **SSL-AASIST**: an SSL frontend for feature extraction combined with an AASIST backend for classification. Two parameter-efficient training paradigms are proposed:

1. **PT-SSL-AASIST (Prompt Tuning)**: Learnable Prompt Tokens are inserted before each Transformer layer, while all other SSL parameters are frozen.
2. **WPT-SSL-AASIST (Wavelet Prompt Tuning)**: A subset of Prompt Tokens undergoes discrete wavelet transform (DWT) to obtain tokens at different frequency subbands, enhancing full-band frequency perception.

### Key Designs

**PT-SSL-AASIST Design**:

- Input audio $X$ is zero-padded or truncated to a fixed length $L$; initial embeddings $E_0$ are extracted via the frozen SSL CNN feature extractor.
- Learnable Prompt Tokens $P_k$ (Xavier uniform initialization) are initialized for each Transformer layer.
- Per-layer computation: $[Z_i, E_i] = L_i([P_i, E_{i-1}])$
- The Prompt output $Z_i$ from the previous layer is discarded and replaced by a fresh $P_i$.
- The final output $I = [Z_{24}, E_{24}]$ is fed into the AASIST backend for spectro-temporal graph attention classification.

**WPT-SSL-AASIST Design**:

- A subset of Prompt Tokens is replaced by wavelet Prompt Tokens via DWT using the Haar wavelet.
- Haar wavelet lowpass filter $L = [1,1]/\sqrt{2}$, highpass filter $H = [1,-1]/\sqrt{2}$.
- DWT is applied to the initial wavelet token $T_k$, yielding four subbands:
    - **LL** (low-frequency), **LH** (vertical high-frequency), **HL** (horizontal high-frequency), **HH** (diagonal high-frequency)
    - Each component is of size $w/2 \times d/2$ and reshaped to $w/4 \times d$.
    - The four components are concatenated to form the wavelet Prompt $W_k$.
- Per-layer computation becomes: $[Z_i, E_i] = L_i([W_i, P_i, E_{i-1}])$
- Key finding: each token corresponds to a specific frequency component; with WPT=4, the four subbands are naturally aligned.

**Parameter Efficiency**: PT/WPT only optimize Prompt Tokens (~0.69M parameters), representing a **458×** reduction compared to full fine-tuning FT (315.89M).

### Loss & Training

- **Weighted Cross-Entropy (WCE) loss** for binary classification (genuine/spoof).
- All audio downsampled to 16 kHz, truncated/padded to 64,600 samples (~4 s).
- FT: learning rate $1\times10^{-6}$, batch size 14; FR/PT/WPT: learning rate $5\times10^{-4}$, batch size 32.
- Single-type training: 50 epochs (LR halved every 10 steps); joint training: 20 epochs (LR halved every 4 steps).
- SSL feature dimension: $(201, 1024)$.

## Key Experimental Results

### Main Results

**Dataset Statistics**:

| Type | Source | Train | Dev | Test |
|---|---|---|---|---|
| Speech | 19LA | 25,380 | 24,844 | 71,237 |
| Sound | Codecfake-A3 | 69,378 | 9,911 | 19,823 |
| Singing | CtrSVDD | 84,404 | 43,625 | 92,769 |
| Music | FakeMusicCaps | 20,861 | 6,058 | 6,122 |
| All | Joint | 199,023 | 84,438 | 189,951 |

**Joint Training Results (EER%)**:

| Model | Speech | Sound | Singing | Music | Avg |
|---|---|---|---|---|---|
| Spec-Resnet | 29.37 | 23.37 | 37.17 | 42.75 | 33.17 |
| AASIST | 3.78 | 0.86 | 20.01 | 11.70 | 9.09 |
| FR-XLSR-AASIST | 3.02 | 5.45 | 10.86 | 22.67 | 10.50 |
| FT-XLSR-AASIST | 1.77 | 0.49 | 8.93 | 8.71 | 4.98 |
| PT-XLSR-AASIST | 2.00 | 1.11 | 14.54 | 9.29 | 6.74 |
| **WPT-XLSR-AASIST** | **0.72** | **1.29** | **7.47** | **4.83** | **3.58** |

WPT achieves 3.58% average EER with only 0.69M parameters (1/458 of FT), outperforming FT (4.98%) which uses 315.89M parameters.

**Cross-Type Generalization under Single-Type Training**:

| Training Type | Best Model | In-Domain EER | Avg EER | Cross-Type Finding |
|---|---|---|---|---|
| Speech | FR-XLSR-AASIST | 1.28% | 32.58% | Speech generalizes relatively well to singing |
| Sound | AASIST | 0.43% | 22.71% | Sound shows correlation with music |
| Singing | FR-XLSR-AASIST | 9.45% | 23.16% | Singing generalizes well to speech |
| Music | FR-MERT-AASIST | 7.62% | 28.63% | MERT is strongest for music |

### Ablation Study

**Prompt Token Count Ablation** (speech-trained XLSR-AASIST):

| # Tokens | Parameters | Speech EER | Avg EER |
|---|---|---|---|
| 2 | 0.50M | 0.75% | 30.94% |
| 10 | 0.69M | **0.22%** | **30.79%** |
| 100 | 2.90M | 3.01% | 31.28% |
| 200 | 5.36M | 4.99% | 33.36% |

Excessive Prompt Tokens dilute the information density of audio tokens; 10 tokens is optimal.

**Training Paradigm Comparison** (speech-trained XLSR-AASIST):

| Model | Parameters | Speech EER | Avg EER |
|---|---|---|---|
| FR-XLSR-AASIST | 0.45M | 1.28% | 32.58% |
| FT-XLSR-AASIST | 315.89M | 0.38% | 27.68% |
| PT-XLSR-AASIST | 0.69M | 0.22% | 30.79% |
| WPT-XLSR-AASIST | 0.69M | 0.15% | 26.86% |

### Key Findings

- **Shared features exist between speech and singing**, yielding the best cross-type generalization; sound and music also exhibit correlation.
- WPT learns **type-invariant deepfake detection prompts**: t-SNE visualizations show that genuine/spoof samples under WPT do not cluster by type.
- Attention map analysis reveals that WPT focuses on the 4th token (corresponding to the **HH subband—diagonal high-frequency**), which is consistent across all audio types.
- FR/PT/WPT converge significantly faster than FT with less variance.

## Highlights & Insights

1. **First all-type ADD benchmark**: covering deepfake detection across speech, sound, singing, and music, filling a critical gap in the field.
2. **Elegant wavelet Prompt design**: DWT is embedded into Prompt Token initialization, achieving full-band frequency perception without adding trainable parameters.
3. **458× parameter efficiency gain**: WPT requires only 0.69M parameters yet surpasses the 315.89M FT method.
4. **Discovery of type-invariant HH subband**: attention maps clearly show WPT's focus on the HH token, providing new insights for frequency-domain deepfake detection.
5. Systematic experimental design: the progressive analysis from single-type to cross-type to joint training is thorough and well-structured.

## Limitations & Future Work

1. Benchmark datasets are recorded in relatively clean conditions; noisy environments, partial forgery, and other complex real-world scenarios are not considered.
2. All audio is truncated to ~4 seconds; applicability to longer audio has not been validated.
3. WPT uses only the Haar wavelet (the simplest wavelet transform); more sophisticated wavelet families may yield further improvements.
4. EER for music detection remains significantly higher than for speech; unified cross-type detection still has room for improvement.
5. Detection robustness under adversarial attack scenarios is not explored.

## Related Work & Insights

- **XLSR-AASIST** (Tak et al. 2022): the canonical SSL+AASIST framework serving as the baseline for this work.
- **Visual Prompt Tuning** (Jia et al. 2022): Prompt Tuning in the vision domain, which inspired PT-SSL.
- **CtrSVDD** (Zhang et al. 2024): the first singing voice deepfake detection challenge.
- **FakeMusicCaps** (Comanducci et al. 2024): the first music deepfake detection dataset.
- Insight: frequency-domain information serves as a key bridge for cross-modal/cross-type detection; DWT subband decomposition provides a natural and efficient means of injecting frequency-domain features.

## Rating

- Novelty: 5/5 — both the all-type ADD benchmark and the WPT method are pioneering contributions.
- Technical Depth: 4/5 — the integration of wavelet transform with Prompt Tuning is elegantly designed.
- Experimental Thoroughness: 5/5 — single-type/cross-type/joint training + extensive ablations + visualization analysis.
- Writing Quality: 4/5 — well-structured with rich figures and tables.
- Overall: 4.5/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification](../../NeurIPS2025/ai_safety/not_all_deepfakes_are_created_equal_triaging_audio_forgeries_for_robust_deepfake.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](../../ICCV2025/ai_safety/fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[ACL 2026\] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection](../../ACL2026/ai_safety/xlsr-mambo_scaling_the_hybrid_mamba-attention_backbone_for_audio_deepfake_detect.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)
- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)

</div>

<!-- RELATED:END -->
