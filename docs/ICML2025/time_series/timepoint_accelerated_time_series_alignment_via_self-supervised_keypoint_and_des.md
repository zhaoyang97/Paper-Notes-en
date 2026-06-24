---
title: >-
  [Paper Note] TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning
description: >-
  [ICML 2025][Time Series][Time Series Alignment] Proposes TimePoint—a self-supervised method inspired by 2D keypoint detection but rewritten for 1D signals. It learns sparse representations of time series by detecting keypoints and extracting descriptors, applying DTW to sparse keypoints instead of the full signal. This significantly accelerates alignment while frequently improving alignment accuracy.
tags:
  - "ICML 2025"
  - "Time Series"
  - "Time Series Alignment"
  - "DTW Acceleration"
  - "Keypoint Detection"
  - "Self-Supervised Learning"
  - "Diffeomorphism"
  - "Wavelet Convolution"
date: 2026-05-08
content_hash: 1bf3f015285fb3a2
---

# TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning

**Conference**: ICML 2025  
**arXiv**: [2505.23475](https://arxiv.org/abs/2505.23475)  
**Code**: [https://github.com/BGU-CS-VIL/TimePoint](https://github.com/BGU-CS-VIL/TimePoint)  
**Area**: Time Series  
**Keywords**: Time Series Alignment, DTW Acceleration, Keypoint Detection, Self-Supervised Learning, Diffeomorphism, Wavelet Convolution

## TL;DR
Proposes TimePoint—a self-supervised method inspired by 2D keypoint detection but rewritten for 1D signals. It learns sparse representations of time series by detecting keypoints and extracting descriptors, applying DTW to sparse keypoints instead of the full signal. This significantly accelerates alignment while frequently improving alignment accuracy.

## Background & Motivation

**Core Problem**: Time series alignment is a fundamental operation in time series analysis. The standard method, Dynamic Time Warping (DTW), has a time complexity of $O(N^2)$, making it heavily unscalable to long sequences.

**The Double Dilemma of DTW**:
   - **Poor Scalability**: $O(N^2)$ complexity makes computation prohibitively expensive for long sequences.
   - **Noise Sensitivity**: Performs point-by-point matching on raw signals, allowing noise to directly interfere with alignment.

**Limitations of Prior Work**: Although FastDTW claims acceleration, empirical evidence shows it is often slower and yields poorer accuracy than standard DTW.

**Inspiration from 2D Vision**: Methods like SuperPoint in the image matching domain achieve efficient matching via sparse keypoints + descriptors. Can this idea be transferred to 1D time series?

**Key Insight**: Compress the time series into a sparse keypoint representation and run DTW on these keypoints, resolving both speed and robustness issues simultaneously.

## Method

### Overall Architecture
1. **Synthetic Data Generation**: Uses 1D diffeomorphism to generate training pairs with ground-truth alignments.
2. **Keypoint + Descriptor Network**: Learns to detect keypoints in time series and extract descriptors.
3. **Sparse DTW**: Runs DTW on the descriptors of the sparse keypoints to achieve accelerated alignment.

### Key Designs

1. **SynthAlign Synthetic Data Engine**:

    - Leverages 1D diffeomorphism transformations (CPAB, Continuous Piecewise-Affine-Based) to generate non-linear temporal warping of time series.
    - Diffeomorphism guarantees the transformation is invertible and smooth, naturally simulating real-world temporal warping.
    - Supports multiple signal modes: sine wave combinations (60%), square waves (15%), sawtooth waves (5%), and RBF (20%).
    - Automatically obtains the keypoint correspondences before and after warping to serve as supervision signals.

2. **TimePoint Network Architecture**:

    - Input: $(N, 1, L)$ time series.
    - Encoder dimensions: $[128, 128, 256, 256]$.
    - Descriptor dimension: 256.
    - Two encoder options:
        - **Dense Convolution**: Standard 1D CNN.
        - **Wavelet Convolution (WTConv)**: Multi-scale convolution using wavelet transform, providing a larger receptive field.
    - Output: Keypoint probability map $(N, L)$ + descriptors $(N, 256, L)$.

3. **Keypoint Extraction and Matching**:

    - Selects Top-K keypoints (e.g., 10% of the sequence length) from the probability map.
    - Applies Non-Maximum Suppression (NMS, window=5) to remove redundancy.
    - Extracts descriptor vectors at the keypoint locations.
    - Compares keypoint descriptors using DTW, replacing point-by-point DTW on the raw signals.

4. **Self-Supervised Loss Function**:

$$\mathcal{L} = \mathcal{L}_{det} + \lambda \mathcal{L}_{desc}$$

- Keypoint detection loss $\mathcal{L}_{det}$: Encourages detecting keypoints at locations with significant signal variations (e.g., extrema, sudden changes).
- Descriptor loss $\mathcal{L}_{desc}$: Ensures descriptors of corresponding keypoints are similar, while non-corresponding ones are dissimilar (contrastive learning).
- Key innovation: All supervision signals are derived from the diffeomorphism transformation of synthetic data, requiring zero manual annotation.

### Loss & Training
- Stage 1: Pre-training exclusively on SynthAlign synthetic data.
- Stage 2 (Optional): Fine-tuning on real datasets such as UCR.
- Pre-trained weights: `synth_only.pth` (synthetic only), `synth_and_ucr.pth` (synthetic + UCR fine-tuned).

## Key Experimental Results

### Main Results: UCR Dataset Alignment Accuracy (Alignment Error ↓)

| Method | Mean AE | Relative DTW Gain | Speedup |
|------|---------|-------------|--------|
| DTW (Standard) | 0.142 | — | 1× |
| FastDTW | 0.168 | -18.3% (Worse) | 0.8× |
| Soft-DTW | 0.138 | +2.8% | 0.3× (Slower) |
| DTAN | 0.129 | +9.2% | 2.1× |
| **TimePoint (synth)** | **0.118** | **+16.9%** | **8.5×** |
| **TimePoint (synth+UCR)** | **0.105** | **+26.1%** | **8.5×** |

### Speed Comparison (Sequence Length 512)

| Method | Time (ms) | Speedup |
|------|----------|--------|
| DTW | 48.2 | 1× |
| FastDTW | 52.1 | 0.9× |
| DTAN | 23.4 | 2.1× |
| **TimePoint (10% kp)** | **5.7** | **8.5×** |
| **TimePoint (5% kp)** | **3.2** | **15.1×** |

### Ablation Study

| Configuration | Mean AE | Description |
|------|---------|------|
| Dense Encoder | 0.125 | Standard CNN |
| **WTConv Encoder** | **0.118** | Wavelet convolution, larger receptive field |
| No NMS | 0.131 | Redundant keypoints |
| K=5% | 0.122 | Too few keypoints |
| K=10% | 0.118 | Optimal ratio |
| K=20% | 0.120 | Too many keypoints |
| Synthetic only training | 0.118 | Good generalizability achieved |
| + UCR fine-tuning | 0.105 | Fine-tuning further improves results |

### Key Findings
- TimePoint generalizes well to real-world data even when trained solely on synthetic data, indicating that diffeomorphism-generated synthetic warps are sufficiently realistic.
- WTConv (Wavelet Convolution) is more suitable for time series keypoint detection than standard CNNs, with the large receptive field being critical.
- A 10% keypoint ratio offers the optimal trade-off between speed and accuracy.
- FastDTW actually performs worse than standard DTW, validating the findings of Wu & Keogh (2020).

## Highlights & Insights
- **A Successful Case of Cross-Domain Transfer**: 2D keypoint detection $\rightarrow$ 1D time series, which is not a naive transfer but a redesign tailored for 1D signals.
- **The Power of Synthetic Data**: Diffeomorphism provides a physically plausible temporal warping model, enabling model generalization solely from synthetic training.
- **The Revival of DTW**: It does not replace DTW, but rather enables DTW to work faster and better in a sparse space.
- **Significant Speedup**: An 8-15× acceleration, making long sequence alignment highly practical.

## Limitations & Future Work
- The keypoint ratio K% requires manual configuration, and optimal values may vary across different datasets.
- Currently supports only single-channel (1D) time series, leaving multivariate expansion for future research.
- For smooth sequences lacking prominent keypoint features, keypoint detection effectiveness might degrade.
- Diffeomorphism assumes smooth transformations and cannot simulate abrupt temporal warping.

## Related Work & Insights
- **SuperPoint (DeTone et al., 2018)**: The direct inspiration for TimePoint, though the 2D $\rightarrow$ 1D adaptation is not a mere dimension reduction.
- **DTAN (Weber et al., 2019)**: Previous work from the same authors, using diffeomorphism for time alignment networks.
- **Soft-DTW (Cuturi & Blondel, 2017)**: Differentiable DTW, which, however, does not address the scalability issue.
- **Insight**: Combining sparse representations with traditional algorithms is more flexible and interpretable than purely replacing traditional algorithms with end-to-end models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant transfer of keypoint detection to time series alignment with novel insights.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on numerous UCR datasets with a dual focus on both speed and accuracy.
- Writing Quality: ⭐⭐⭐⭐ Clearly described methodology with a natural motivational progression.
- Value: ⭐⭐⭐⭐ DTW acceleration addresses a practical pain point with an elegant solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](../../NeurIPS2025/time_series/towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](../../NeurIPS2025/time_series/universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[ICML 2026\] Self-Supervised Dynamical System Representations for Physiological Time-Series](../../ICML2026/time_series/self-supervised_dynamical_system_representations_for_physiological_time-series.md)
- [\[ECCV 2024\] OmniSat: Self-Supervised Modality Fusion for Earth Observation](../../ECCV2024/time_series/omnisat_self-supervised_modality_fusion_for_earth_observation.md)
- [\[ICML 2026\] HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series](../../ICML2026/time_series/hepa_a_self-supervised_horizon-conditioned_event_predictive_architecture_for_tim.md)

</div>

<!-- RELATED:END -->
