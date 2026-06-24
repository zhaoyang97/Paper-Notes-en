---
title: >-
  [Paper Note] TIDMAD: Time Series Dataset for Discovering Dark Matter with AI Denoising
description: >-
  [NeurIPS 2025 Spotlight][Image Generation][Dark matter detection] This work introduces TIDMAD — the first ultra-long time series denoising benchmark dataset for dark matter searches — comprising training/validation/science data from the ABRACADABRA experiment, a denoising score metric, and a complete analysis pipeline, enabling AI algorithms to directly produce physics-community-standard dark matter search results.
tags:
  - "NeurIPS 2025 Spotlight"
  - "Image Generation"
  - "Dark matter detection"
  - "time series denoising"
  - "ABRACADABRA"
  - "benchmark dataset"
  - "signal recovery"
date: 2026-05-08
content_hash: f40fa93a4d0a80de
---

# TIDMAD: Time Series Dataset for Discovering Dark Matter with AI Denoising

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2406.04378](https://arxiv.org/abs/2406.04378)  
**Code**: [GitHub](https://github.com/jessicafry/TIDMAD)  
**Area**: Dataset / Time Series Denoising
**Keywords**: Dark matter detection, time series denoising, ABRACADABRA, benchmark dataset, signal recovery

## TL;DR

This work introduces TIDMAD — the first ultra-long time series denoising benchmark dataset for dark matter searches — comprising training/validation/science data from the ABRACADABRA experiment, a denoising score metric, and a complete analysis pipeline, enabling AI algorithms to directly produce physics-community-standard dark matter search results.

## Background & Motivation

Dark matter accounts for approximately 85% of the total mass of the universe, yet has never been directly detected in terrestrial laboratories. The axion is among the most prominent dark matter candidate particles, and the ABRACADABRA (ABRA-10cm) experiment searches for axion signals using a superconducting toroidal magnet and SQUID sensors.

The core challenges are as follows:

**Extremely weak signals**: Dark matter signals manifest as sinusoidal oscillation patterns in time series but are overwhelmed by various noise sources.

**Extremely wide frequency range**: Signal frequencies span seven orders of magnitude (1.1 kHz to 4.9 MHz).

**Massive data volume**: Sampling rate reaches 10 MS/s (ten million samples per second).

**Non-Gaussian noise**: Detector noise arises from the superposition of multiple independent noise sources and cannot be described by simple models.

Conventional denoising methods (e.g., Fourier averaging) suffer from signal loss in the high-frequency regime. Machine learning denoising techniques have the potential to substantially improve the sensitivity of dark matter experiments — halving the noise level is equivalent to increasing data acquisition time by a factor of 16. However, a standardized physics experimental dataset for use by the ML community has previously been lacking. The release of TIDMAD fills this gap and bridges the divide between AI and particle physics.

## Method

### Overall Architecture

TIDMAD does not propose a new algorithm; rather, it constructs a complete ecosystem of dataset + benchmark + analysis framework:

- **Training set**: Ultra-long time series with injected signals, providing paired SQUID noise sequences (CH1) and ground-truth injected signals (CH2).
- **Validation set**: Independently collected data with injected signals, used to compute the denoising score (Benchmark 1).
- **Science dataset**: 24 hours of pure experimental data without injected signals, used to produce dark matter limits (Benchmark 2).

### Key Designs

1. **Signal injection mechanism**

   Simulated dark matter signals are injected into the detector hardware via a calibration loop. The injected signals follow the theoretical axion form:

   $\boldsymbol{J}_{eff} = g_{a\gamma\gamma}\sqrt{2\rho_{DM}}\boldsymbol{B}_0 \cos(m_a t)$

   A total of 309 distinct frequencies (1.1 kHz to 4.9 MHz) are injected, simulating 309 axion masses, with a uniform amplitude of 50 mV. The training objective is to recover the injected sinusoidal signal from the noisy SQUID sequence.

2. **Denoising Score (Benchmark 1)**

   Designed based on an improved signal-to-noise ratio, the computation procedure is as follows:
    - Divide the time series into 1-second segments and apply FFT to convert to power spectral density (PSD).
    - Locate the signal frequency $\nu_0$ in the PSD of the injected signal.
    - Compute the ratio of signal-region PSD to noise-region PSD to obtain the SNR.
    - Normalize and compute a weighted sum to obtain $\Lambda$.
    - Apply a logarithmic transformation: $\text{Denoising Score} = \log_{5.27}\Lambda$

   A denoising score of 1 corresponds to the raw undenoised data; higher scores indicate better denoising performance. It has been verified that this score exhibits a linear relationship with noise amplitude.

3. **Dark Matter Limit (Benchmark 2)**

   The denoised science data is fed into an automated analysis pipeline that employs a frequentist log-likelihood ratio test statistic. The limit-setting procedure is repeated over 11.1 million independent mass points, ultimately generating an exclusion plot conforming to physics community standards. ML developers need only execute the `brazilband.py` script to automatically produce dark matter search results.

### Loss & Training

Training strategies for the 8 baseline denoising algorithms:
- Conventional methods (moving average, Savitzky-Golay filter, Fourier averaging) are applied directly without training.
- Deep learning models (FC Net, PU Net, Transformer, WaveNet, RNN Seq2Seq) are trained on the training set.
    - Due to memory constraints imposed by ultra-long sequences, sequences are divided into segments (segment length $2\times10^4$ to $4\times10^4$).
    - PU Net / Transformer / RNN Seq2Seq reformulate denoising as a 256-class classification task and are trained with Focal Loss.
    - All models except WaveNet employ frequency splitting (multiple specialized versions handling different frequency ranges).

## Key Experimental Results

### Main Results

| Algorithm | Type | Segment Length | Freq. Splitting | Fine Score | Coarse Score |
|-----------|------|---------------|-----------------|-----------|-------------|
| No denoising | — | — | — | 1.00 | 1.10 |
| Fourier averaging | Conventional | $1\times10^8$ | — | 0.24 | 0.26 |
| Moving average | Conventional | $1\times10^6$ | — | 0.86 | 0.95 |
| SG filter | Conventional | $1\times10^6$ | — | 0.95 | 1.04 |
| FC Net | DL | $4\times10^4$ | Yes | **6.43** | **6.55** |
| WaveNet | DL | $4\times10^4$ | No | 4.99 | 5.16 |
| Transformer | DL | $2\times10^4$ | Yes | 3.95 | 4.18 |
| PU Net | DL | $4\times10^4$ | Yes | 3.69 | 3.84 |
| RNN Seq2Seq | DL | $4\times10^4$ | Yes | 3.38 | 3.79 |

### Dark Matter Limit Comparison

| Configuration | Sensitivity | Notes |
|--------------|------------|-------|
| ABRA-TIDMAD Raw | Weak | Undenoised, 24h data, coverage smaller than Run 3 |
| ABRA-TIDMAD Denoised (FC Net) | Strong | Limit improves by 1–2 orders of magnitude after AI denoising |
| ABRA-10cm Run 3 | Strongest | 3 months of data, current world-leading result |
| Gap after denoising | Near parity | Using only 1% of the data volume, approaches Run 3 level; surpasses it at low masses |

### Key Findings

1. **All conventional methods degrade the denoising score**, because temporal averaging erases high-frequency signals; all deep learning methods substantially improve the score.
2. **FC Net unexpectedly achieves the best performance** (score 6.43); the simple autoencoder architecture proves most effective for this task.
3. Using only 1% of the data volume (24h vs. 3 months), the AI-denoised dark matter limit approaches the Run 3 level.
4. Coarse Score (10× downsampling) is highly consistent with Fine Score and can be used for rapid evaluation.
5. The denoising score design guarantees a linear relationship with noise level, making it a reliable evaluation metric.

## Highlights & Insights

- **Interdisciplinary bridge**: For the first time, the ML community can directly advance dark matter searches without needing to understand the underlying physics.
- **Dual benchmark design**: The Denoising Score enables rapid model iteration, while the Dark Matter Limit directly connects to physical significance.
- **Data scale and authenticity**: Over 800 Gigasamples of real detector data — not simulated or synthetic.
- The finding that a simple model (FC Net) outperforms more complex models warrants reflection — likely because the task is fundamentally narrow-band signal extraction.

## Limitations & Future Work

- Due to hardware changes (pickup loop replacing pickup tube) and data acquisition time constraints, the baselines do not surpass the Run 3 results.
- The current framework assumes a null result (no dark matter candidate); discovery analysis code has not yet been implemented.
- The dataset targets a specific frequency range and detector; generalization to other dark matter experiments remains to be validated.
- The frequency splitting strategy for deep learning models increases engineering complexity.

## Related Work & Insights

- Time series denoising techniques are broadly applicable to gravitational wave detection, pulsar timing, seismic signal extraction, and related domains.
- Signal recovery tasks under physical constraints may require architectures fundamentally different from general-purpose denoisers.
- Processing ultra-long sequences remains a technical bottleneck; long-context modeling in the era of foundation models may offer a breakthrough.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First ML benchmark targeting dark matter searches; highly significant for interdisciplinary research.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Eight baseline methods, dual benchmark evaluation, closed-loop physical validation.
- **Writing Quality**: ⭐⭐⭐⭐ Physical background and ML methods are presented in a manner accessible to ML readers.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a standardized pathway for AI to directly drive fundamental scientific discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [\[ICCV 2025\] Learning to See in the Extremely Dark](../../ICCV2025/image_generation/learning_to_see_in_the_extremely_dark.md)
- [\[NeurIPS 2025\] CaMiT: A Time-Aware Car Model Dataset for Classification and Generation](camit_a_time-aware_car_model_dataset_for_classification_and_generation.md)
- [\[ICML 2025\] LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation](../../ICML2025/image_generation/lscd_lomb-scargle_conditioned_diffusion_for_time_series_imputation.md)
- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](../../ICLR2026/image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)

</div>

<!-- RELATED:END -->
