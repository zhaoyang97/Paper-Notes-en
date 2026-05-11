---
title: >-
  [Paper Note] GOMPSNR: Reflourish the Signal-to-Noise Ratio Metric for Audio Generation Tasks
description: >-
  [AAAI2026][Audio & Speech][signal-to-noise ratio] This paper reconstructs the SNR metric by introducing omnidirectional phase derivatives to replace instantaneous phase…
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "signal-to-noise ratio"
  - "phase derivatives"
  - "audio quality metric"
  - "loss function"
  - "neural vocoder"
date: 2026-05-08
content_hash: 04b19f1890c78f8b
---

# GOMPSNR: Reflourish the Signal-to-Noise Ratio Metric for Audio Generation Tasks

**Conference**: AAAI2026
**arXiv**: [2601.13758](https://arxiv.org/abs/2601.13758)
**Code**: [lingling-dai/GOMPSNR](https://github.com/lingling-dai/GOMPSNR)
**Area**: Audio & Speech
**Keywords**: signal-to-noise ratio, phase derivatives, audio quality metric, loss function, neural vocoder

## TL;DR
This paper reconstructs the SNR metric by introducing omnidirectional phase derivatives to replace instantaneous phase, proposes GOMPSNR as a more reliable audio quality evaluation metric, and derives a family of novel loss functions that significantly improve neural vocoder performance.

## Background & Motivation

### Limitations of Prior Work

**Background**: Signal-to-noise ratio (SNR) has long served as a fundamental objective metric for evaluating quality in audio generation tasks. However, a growing body of research demonstrates that SNR and its variants (segSNR, SI-SNR, etc.) exhibit low correlation with perceptual metrics such as PESQ and UTMOS, leading to their gradual marginalization. Meanwhile, MCD and M-STFT, which share similar mathematical forms, remain mainstream metrics. This contradiction motivates the authors to pose two questions:

1. **Why does SNR fail?** By expanding SNR from the time domain to the time-frequency domain, the authors find that SNR implicitly couples amplitude and phase measurements. Residuals in the magnitude spectrum exhibit clear structure, whereas residuals in the phase spectrum present uninformative noise patterns, indicating that conventional instantaneous phase (IP) distance measurement is inherently unreliable.
2. **How can SNR be fixed?** Phase derivatives—instantaneous frequency (IF) and group delay (GD)—exhibit clearer structure than instantaneous phase and can be used as substitutes for IP in computing phase distances.

### **Goal**:
- Inaccurate phase distance measurement in SNR is the key factor causing its inconsistency with human auditory perception.
- The phase spectrum is unreliable for direct distance computation due to its wrapping property (range restricted to $[-\pi, \pi)$) and high sensitivity to waveform shifts.
- The sign of the correlation term $C$ in the SNR formula flips near $\theta - \hat{\theta} \approx \pm\pi/2$, causing numerical oscillations and making SNR overly sensitive to phase errors.

## Method

### 1. Omnidirectional Phase Derivatives
A fixed-parameter $3 \times 3$ convolution kernel $\mathcal{K} \in \mathbb{R}^{9 \times 3 \times 3}$ is applied to extract omnidirectional phase derivatives from 8 neighboring directions in the time-frequency map, along with the instantaneous phase itself:

$$\nabla\theta = \theta \circledast \mathcal{K}$$

An anti-wrapping function $f_{AW}(x) = |x - 2\pi \cdot \text{round}(x / 2\pi)|$ is employed to address the phase wrapping problem.

### 2. GOMPSNR Metric
SNR expanded in the time-frequency domain takes the form:

$$SNR = 10\log_{10} \frac{\sum_{k,l} |Y|^2}{\sum_{k,l}(|Y|^2 + |\hat{Y}|^2 + C)}$$

where the correlation term is $C = -2|Y||\hat{Y}|\cos(\theta - \hat{\theta})$. The improvement proceeds in two steps:

- **OMPSNR**: Replaces IP with omnidirectional phase derivatives, $C = -\frac{2}{9}|Y||\hat{Y}|\sum_i \cos(\nabla_i\theta - \nabla_i\hat{\theta})$
- **GOMPSNR**: Further replaces $\cos$ with a linearly mapped anti-wrapping function, ensuring $C$ remains non-positive and eliminating numerical oscillations caused by sign flips: $C = \frac{2}{9}|Y||\hat{Y}|\sum_i(\frac{1}{\pi}f_{AW}(\nabla_i\theta - \nabla_i\hat{\theta}) - 1)$

### 3. Novel Loss Function Family
Grounded in the same phase derivative insight, three categories of new loss functions are proposed:

- **WOP Loss (Amplitude-Weighted Omnidirectional Phase Loss)**: Weights the OP loss by the magnitude spectrum, granting greater attention to high-energy regions.
- **OmniRI Loss**: Replaces IP in the conventional RI loss with omnidirectional phase derivatives, decoupling the joint optimization of phase and amplitude.
- **CORI Loss (Coupled OmniRI)**: Couples amplitude distance and phase derivative distance in a product form for simultaneous optimization of both components.

### 4. Optimal Loss Function Combination
A search over three dimensions—amplitude loss (Log/Lin), phase loss (WOP), and joint optimization loss (CORI)—identifies the optimal combination, with Lin + WOP + CORI(L1) being the recommended configuration.

## Key Experimental Results

**Metric Validation**: PCC and SRCC are computed on LibriTTS using the official pretrained Vocos:
- SNR correlates with perceptual metrics at no more than 0.1, rendering it nearly ineffective.
- GOMPSNR demonstrates strong correlation with PESQ, UTMOS, VQScore, NISQA, and DistillMOS.

**Vocos Loss Function Ablation (LJSpeech)**:
- Baseline: PESQ 3.749, UTMOS 4.128, GOMPSNR 4.299
- +WOP: PESQ 3.928 (+0.18), GOMPSNR 5.232 (+0.93)
- +WOP+CORI(L1): PESQ 4.001, MCD 2.238, GOMPSNR 5.674

**Cross-Vocoder Validation (LJSpeech, Lin+WOP+CORI vs. baseline)**:
- Vocos: PESQ 3.749→4.035, GOMPSNR 4.299→5.749
- APNet2: PESQ 3.643→3.901, GOMPSNR 4.961→5.533
- RNDVoc: PESQ 4.033→4.121, GOMPSNR 5.655→5.822

**Neural Audio Codec**: Both WavTokenizer and Vocos codec achieve consistent improvements across all bandwidths, with more pronounced gains at low bandwidths (high compression ratios).

## Highlights & Insights
- **Rigorous Problem Analysis**: Mathematical derivations and visualizations clearly pinpoint the root cause of SNR failure in phase distance measurement, with tight logical argumentation.
- **Dual Advancement in Metric and Loss Function**: A single core insight simultaneously improves both the evaluation metric and the training loss, yielding a methodologically unified and practical framework.
- **Extensive Experimental Coverage**: 4 vocoders (Vocos, APNet, APNet2, RNDVoc) × 2 datasets (LJSpeech, LibriTTS) + Neural Audio Codec provide thorough validation.
- **Plug-and-Play**: The proposed loss functions require no architectural modifications and can directly replace the original losses, making them engineering-friendly.

## Limitations & Future Work
- Experiments are limited to vocoders and audio codecs; effectiveness on upstream tasks such as speech enhancement and speech separation remains unverified.
- GOMPSNR requires a reference signal (intrusive metric) and cannot be applied in non-reference scenarios.
- The omnidirectional phase derivatives rely on a fixed $3 \times 3$ convolution kernel; larger receptive fields or learnable kernels have not been explored.
- The loss function combination search still depends on manual enumeration, lacking an automated search strategy.
- No direct comparison is made with recent non-intrusive perceptual metrics such as DNSMOS or SpeechLMScore.

## Related Work & Insights

| Method | Type | Phase Handling | Correlation with Perceptual Metrics |
|--------|------|----------------|-------------------------------------|
| SNR/SI-SNR | Metric | Implicit (instantaneous phase) | Extremely low (PCC/SRCC < 0.1) |
| OP Loss | Loss function | Omnidirectional phase derivatives | — |
| **GOMPSNR** | **Metric + Loss** | **Omnidirectional phase derivatives + anti-wrapping + linear mapping** | **Significantly improved** |
| M-STFT | Metric | Magnitude spectrum distance (phase ignored) | Moderate |
| PESQ/UTMOS | Perceptual metrics | Based on auditory models | Reference standard |

GOMPSNR is closely related to the OP representation used in the concurrent work RNDVoc (IJCAI 2025) and can be regarded as its generalization at the metric level.

## Related Work & Insights
- The unreliability of phase distance measurement may also affect other tasks relying on time-frequency representations (e.g., music generation, sound event detection), warranting transfer validation.
- The idea of "weighting phase loss by amplitude" in WOP loss can be generalized to joint optimization scenarios involving other multi-component signals.
- The design philosophy of GOMPSNR—identifying the mathematical root cause of a traditional metric's failure and applying targeted corrections—offers a transferable blueprint for metric improvement in other domains.

## Rating
- **Novelty**: 7/10 — The core contribution lies in incorporating omnidirectional phase derivatives into SNR reconstruction; the idea is straightforward yet effective.
- **Experimental Thoroughness**: 9/10 — Systematic validation across multiple vocoders, datasets, and metrics.
- **Writing Quality**: 8/10 — Mathematical derivations are clear and problem motivation is well articulated.
- **Value**: 8/10 — Provides a drop-in replacement for SNR and plug-and-play loss functions with practical value for the audio generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](../../CVPR2026/audio_speech/echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)
- [\[NeurIPS 2025\] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](../../NeurIPS2025/audio_speech/perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)

</div>

<!-- RELATED:END -->
