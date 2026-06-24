---
title: >-
  [Paper Note] GOMPSNR: Reflourish the Signal-to-Noise Ratio Metric for Audio Generation Tasks
description: >-
  [AAAI2026][Audio & Speech][signal-to-noise ratio] By replacing the instantaneous phase with omnidirectional phase derivatives to reconstruct the SNR metric, GOMPSNR is proposed as a more reliable audio quality evaluation metric, deriving a new family of loss functions that significantly improve neural vocoder performance.
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "signal-to-noise ratio"
  - "phase derivatives"
  - "audio quality metric"
  - "loss function"
  - "neural vocoder"
date: 2026-05-08
content_hash: 9ecbda34b317c26a
---

# GOMPSNR: Reflourish the Signal-to-Noise Ratio Metric for Audio Generation Tasks

**Conference**: AAAI2026  
**arXiv**: [2601.13758](https://arxiv.org/abs/2601.13758)  
**Code**: [lingling-dai/GOMPSNR](https://github.com/lingling-dai/GOMPSNR)  
**Area**: Audio and Speech  
**Keywords**: signal-to-noise ratio, phase derivatives, audio quality metric, loss function, neural vocoder  

## TL;DR
By replacing the instantaneous phase with omnidirectional phase derivatives to reconstruct the SNR metric, GOMPSNR is proposed as a more reliable audio quality evaluation metric, deriving a new family of loss functions that significantly improve neural vocoder performance.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Signal-to-Noise Ratio (SNR) has long served as a fundamental objective metric for quality evaluation in audio generation tasks. However, an increasing number of studies have shown that SNR and its variants (e.g., segSNR, SI-SNR) exhibit extremely low correlation with perceptual metrics such as PESQ and UTMOS, leading to their gradual marginalization. Meanwhile, MCD and M-STFT, which share similar mathematical formulations, remain mainstream metrics. This contradiction motivates the authors to investigate two core questions:

1. **Why does SNR fail?** By expanding SNR from the time domain to the time-frequency domain, the authors find that SNR implicitly couples the measurements of amplitude and phase. While the residuals of the amplitude spectra exhibit a clear structure, the residuals of the phase spectra show an uninformative noise-like pattern, revealing that traditional instantaneous phase (IP) distance metrics are inherently unreliable.
2. **How to fix SNR?** Phase derivatives (instantaneous frequency IF and group delay GD) exhibit much clearer structures than the instantaneous phase and can thus be used to replace IP for computing phase distances.

### Goal

**Goal**: - The inaccurate measurement of phase distance in SNR is the key factor causing its inconsistency with human perception.
- Direct calculation of phase distance is unreliable due to the phase wrapping property (values restricted to $[-\pi, \pi)$) and high sensitivity to waveform shifts.
- The sign of the correlation term $C$ in the SNR formula flips around $\theta - \hat{\theta} \approx \pm\pi/2$, causing numerical oscillation and making SNR excessively sensitive to phase errors.

## Method

### 1. Omnidirectional Phase Derivatives
Using nine $3 \times 3$ convolutional kernels with fixed parameters, $\mathcal{K} \in \mathbb{R}^{9 \times 3 \times 3}$, omnidirectional phase derivatives are extracted from eight adjacent directions on the time-frequency spectrogram plus the instantaneous phase itself:

$$\nabla\theta = \theta \circledast \mathcal{K}$$

This is combined with the anti-wrapping function $f_{AW}(x) = |x - 2\pi \cdot \text{round}(x / 2\pi)|$ to resolve the phase wrapping issue.

### 2. GOMPSNR Metric
SNR is expanded in the time-frequency domain as:

$$SNR = 10\log_{10} \frac{\sum_{k,l} |Y|^2}{\sum_{k,l}(|Y|^2 + |\hat{Y}|^2 + C)}$$

where the correlation term $C = -2|Y||\hat{Y}|\cos(\theta - \hat{\theta})$. The improvement process consists of two steps:

- **OMPSNR**: Replaces IP with omnidirectional phase derivatives, yielding $C = -\frac{2}{9}|Y||\hat{Y}|\sum_i \cos(\nabla_i\theta - \nabla_i\hat{\theta})$.
- **GOMPSNR**: Further replaces the $\cos$ term with a linear mapping of the anti-wrapping function, ensuring $C$ remains non-positive and eliminating numerical oscillations from sign flipping: $C = \frac{2}{9}|Y||\hat{Y}|\sum_i(\frac{1}{\pi}f_{AW}(\nabla_i\theta - \nabla_i\hat{\theta}) - 1)$.

### 3. New Family of Loss Functions
Based on the same phase derivative concept, three types of new loss functions are proposed:

- **WOP Loss (Amplitude-Weighted Omnidirectional Phase Loss)**: Weights the OP loss with the amplitude spectrum so that high-energy regions receive more focus.
- **OmniRI Loss**: Replaces the IP in the traditional RI loss with omnidirectional phase derivatives, decoupling the joint optimization of phase and amplitude.
- **CORI Loss (Coupled OmniRI)**: Couples the amplitude distance and phase derivative distance in a multiplicative manner to optimize both simultaneously.

### 4. Optimal Loss Function Combination
By searching for the optimal combination across three dimensions—amplitude loss (Log/Lin), phase loss (WOP), and coupled optimization loss (CORI)—the combination of Lin + WOP + CORI(L1) is ultimately recommended.

## Key Experimental Results

**Metric Validation**: PCC and SRCC are calculated using the official pre-trained Vocos on LibriTTS:
- The correlation of SNR with perceptual metrics is under 0.1, proving to be practically ineffective.
- GOMPSNR exhibits a strong correlation with PESQ, UTMOS, VQScore, NISQA, and DistillMOS.

**Ablation Study of Loss Functions on Vocos (LJSpeech)**:
- Original Configuration: PESQ 3.749, UTMOS 4.128, GOMPSNR 4.299
- +WOP: PESQ 3.928 (+0.18), GOMPSNR 5.232 (+0.93)
- +WOP+CORI(L1): PESQ 4.001, MCD 2.238, GOMPSNR 5.674

**Cross-Vocoder Validation (LJSpeech, Lin+WOP+CORI Combination vs. Original)**:
- Vocos: PESQ 3.749 -> 4.035, GOMPSNR 4.299 -> 5.749
- APNet2: PESQ 3.643 -> 3.901, GOMPSNR 4.961 -> 5.533
- RNDVoc: PESQ 4.033 -> 4.121, GOMPSNR 5.655 -> 5.822

**Neural Audio Codec**: Both WavTokenizer and Vocos codec achieve improvements across all bandwidths, with more significant gains observed at lower bandwidths (higher compression ratios).

## Highlights & Insights
- **In-depth Problem Analysis**: The root cause of SNR failure is mathematically derived and visualized to reside within the phase distance metric, presenting a highly rigorous logical argument.
- **Dual-track Progress on Metrics and Loss Functions**: Improves both evaluation metrics and training losses from the same core insight, providing a unified and practical methodology.
- **Extensive Experimental Coverage**: Comprehensive validation across 4 vocoders (Vocos, APNet, APNet2, RNDVoc) $\times$ 2 datasets (LJSpeech, LibriTTS) plus Neural Audio Codec.
- **Plug-and-Play**: The proposed loss function requires no modification of the model architecture and can replace the original losses directly, making it highly engineering-friendly.

## Limitations & Future Work
- Experiments are limited to vocoders and audio codecs, without validation on upstream tasks such as speech enhancement and speech separation.
- GOMPSNR relies on reference signals (intrusive metric) and cannot be applied in reference-free scenarios.
- The omnidirectional phase derivatives utilize fixed $3 \times 3$ convolutional kernels, leaving the effects of larger receptive fields or learnable kernels unexplored.
- The search for loss function combinations still relies on manual enumeration, lacking an automated search strategy.
- Direct comparisons with recent non-intrusive perceptual metrics (e.g., DNSMOS, SpeechLMScore) are absent.

## Related Work & Insights

| Method | Type | Phase Processing | Correlation with Perceptual Metrics |
|------|------|---------|----------------|
| SNR/SI-SNR | Metric | Implicit (Instantaneous Phase) | Extremely Low (PCC/SRCC < 0.1) |
| OP Loss | Loss Function | Omnidirectional Phase Derivatives | — |
| **GOMPSNR** | **Metric + Loss** | **Omnidirectional Phase Derivatives + Anti-wrapping + Linear Mapping** | **Significant Gain** |
| M-STFT | Metric | Amplitude Spectrum Distance (Ignoring Phase) | Moderate |
| PESQ/UTMOS | Perceptual Metric | Based on Auditory Models | As Reference Standard |

Highly related to the OP representation used in concurrent work RNDVoc (IJCAI 2025), GOMPSNR can be viewed as its generalization at the metric level.

## Related Work & Insights
- The unreliability of phase distance metrics may also affect other tasks dependent on time-frequency representation (e.g., music generation, sound event detection), which is worth migrating and validating.
- The concept of "weighting phase loss with amplitude" in WOP loss can be generalized to joint optimization scenarios of other multi-component signals.
- The design philosophy of GOMPSNR (identifying the mathematical root cause of traditional metric failure and applying targeted corrections) can serve as a reference for metric improvements in other domains.

## Rating
- Novelty: 7/10 — The core contribution lies in introducing omnidirectional phase derivatives to SNR reconstruction; the concept is simple yet effective.
- Experimental Thoroughness: 9/10 — Systematic evaluation across multiple vocoders, datasets, and metrics.
- Writing Quality: 8/10 — Clear mathematical derivation and well-explained motivation.
- Value: 8/10 — Provides a new metric that can directly replace SNR and plug-and-play loss functions, offering practical value to the audio generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](../../ICML2026/audio_speech/mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[ICML 2026\] Focus Then Listen: An Empirical Study of Plug-and-Play Audio Enhancer for Noise-Robust Large Audio Language Models](../../ICML2026/audio_speech/focus_then_listen_an_empirical_study_of_plug-and-play_audio_enhancer_for_noise-r.md)
- [\[ICLR 2026\] StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs](../../ICLR2026/audio_speech/stabletoken_a_noise-robust_semantic_speech_tokenizer_for_resilient_speechllms.md)
- [\[NeurIPS 2025\] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](../../NeurIPS2025/audio_speech/perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)
- [\[ICLR 2026\] AudioX: A Unified Framework for Anything-to-Audio Generation](../../ICLR2026/audio_speech/audiox_a_unified_framework_for_anything-to-audio_generation.md)

</div>

<!-- RELATED:END -->
