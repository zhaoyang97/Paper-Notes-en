---
title: >-
  [Paper Note] Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos
description: >-
  [AAAI 2026][Video Understanding][Time-Series Foundation Models] This paper proposes a pipeline for extracting time-series data from real-world videos via optical flow, constructs the REAL-V-TSFM dataset (6,130 sequences), and reveals significant zero-shot generalization gaps in current time-series foundation models (TSFMs) such as Chronos and TimesFM when confronted with real physical dynamics.
tags:
  - AAAI 2026
  - Video Understanding
  - Time-Series Foundation Models
  - Zero-Shot Generalization
  - Optical Flow
  - Video Data
  - Benchmark
date: 2026-05-08
content_hash: e73e725339b95319
---

# Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos

**Conference**: AAAI 2026
**arXiv**: [2509.26347](https://arxiv.org/abs/2509.26347)
**Code**: [github.com/DobricLilujun/benchmarking_nature_tsfm](https://github.com/DobricLilujun/benchmarking_nature_tsfm)
**Area**: Video Understanding
**Keywords**: Time-Series Foundation Models, Zero-Shot Generalization, Optical Flow, Video Data, Benchmark

## TL;DR

This paper proposes a pipeline for extracting time-series data from real-world videos via optical flow, constructs the REAL-V-TSFM dataset (6,130 sequences), and reveals significant zero-shot generalization gaps in current time-series foundation models (TSFMs) such as Chronos and TimesFM when confronted with real physical dynamics.

## Background & Motivation

### Core Problem

Time-Series Foundation Models (TSFMs) aim to learn universal temporal patterns through large-scale pretraining — analogous to BERT/GPT in NLP — enabling zero-shot cross-domain forecasting. However, unlike NLP models whose generalization has been validated by a vast community of users and researchers, the generalization capability of TSFMs remains insufficiently verified due to **limited dataset diversity** and a **smaller user base**.

### Limitations of Prior Work

**Over-reliance on synthetic data augmentation in training**: Chronos, for instance, uses KernelSynth and TSMixup to generate synthetic training data, and it remains questionable whether such synthetic distributions can cover real-world temporal dynamics.

**Insufficient diversity in evaluation datasets**: Existing benchmarks are predominantly drawn from traditional domains such as finance, energy, and transportation. The M4 dataset, for example, contains only 5% stationary sequences, limiting distributional diversity.

**Sensor and stock price data are already well-studied**: There is a lack of time-series data from entirely new sources to test the true generalization ability of these models.

### Core Motivation

Video is one of the richest sources of time-series data in the modern world, yet it has rarely been used to construct time-series benchmarks. Videos contain rich physical temporal dynamics — human swings, animal locomotion, object trajectories, etc. By extracting pixel-level trajectories from video via optical flow, one can obtain time series that reflect genuine physical motion patterns, providing a novel perspective for evaluating the generalization capability of TSFMs.

**Core Research Question**: How generalizable are current TSFMs? Can they forecast data extracted from everyday real-world events?

## Method

### Overall Architecture

A complete pipeline for extracting time series from video is proposed, comprising six steps:

1. **Video Selection**: Long-sequence videos are selected from the LaSOT dataset to ensure clear subjects.
2. **Frame Extraction**: Images are extracted frame by frame.
3. **Foreground Detection**: MOG2 (Mixture of Gaussians) is applied to separate foreground from background.
4. **Corner Detection**: The Shi-Tomasi algorithm detects corner points on foreground subjects.
5. **Optical Flow Tracking + Consistency Check**: Pyramidal Lucas-Kanade optical flow tracks corner points; forward-backward consistency checking filters unreliable trajectories.
6. **Post-processing**: Linear interpolation normalizes sequence lengths; the 5 trajectories with the lowest mutual correlation are retained; $x$ and $y$ coordinates are treated as independent time series.

### Key Designs

#### 1. **Forward-Backward Consistency Check**: Filtering Unreliable Optical Flow Tracking

**Function**: For each tracked point, forward tracking is followed by backward tracking to verify whether the point returns to its original position.

**Core Formula**:

$$e_{fb}(\mathbf{p}_0) = \|\mathbf{p}_0 - (f_{backward} \circ f_{forward})(\mathbf{p}_0)\|_2$$

A track is considered valid if $e_{fb}(\mathbf{p}_0) < \epsilon$.

**Design Motivation**: Optical flow tracking is prone to numerous errors such as target loss and cross-frame misidentification. Forward-backward consistency checking provides a simple yet effective quality control mechanism. A relatively lenient threshold ($FB\_ERR\_THRESH=50.0$) is adopted to retain more tracking points and extend sequence length.

#### 2. **Diversity Preservation Strategy**: Selecting Trajectories with Lowest Mutual Correlation

**Function**: Cross-correlations among all trajectories in a given video are computed, and the 5 trajectories with the lowest mutual correlation are retained.

**Design Motivation**: Subjects typically map to only 1–2 corner points, while background corner points (introduced by camera motion) exhibit different motion patterns from subject points. Retaining trajectories with the lowest correlation maximizes informational diversity while suppressing noise.

#### 3. **Characteristics of the REAL-V-TSFM Dataset**

- **Scale**: 6,130 time series across 609 distinct objects.
- **Length**: Average of 2,043 time steps; coefficient of variation of 0.516.
- **Category Diversity**: Covers a wide range of objects including aircraft, boats, and cats.
- **Stationarity**: 44% of sequences are stationary (vs. only 5% in M4).
- **Information Entropy**: 3.88 bits (vs. 4.17 bits for M4).
- **PCA Distribution**: More uniform than M4, indicating coverage of a broader range of temporal patterns.

### Evaluation Setup

- **Windowing**: All sequences are uniformly segmented into 500-timestep windows (450 steps as context, 50 steps as prediction target).
- **Sliding Window**: Long sequences are segmented with a stride of 500 steps.
- **Short Sequences**: Interpolated to 500 steps via linear interpolation.

**Evaluation Metrics**: MAPE, sMAPE, Agg. Relative WQL (Weighted Quantile Loss), Agg. Relative MASE.

## Key Experimental Results

### Main Results

| Model | Dataset | MAPE↓ | sMAPE↓ | Agg. Rel. WQL↓ | Agg. Rel. MASE↓ |
|------|--------|-------|--------|---------------|-----------------|
| chronos-bolt-base | REAL-V-TSFM | 7.32±17.06 | 6.57±9.93 | 0.93±0.90 | 0.67±0.63 |
| chronos-bolt-base | M4-Weekly | 5.72±3.83 | 5.70±3.83 | 0.79±0.87 | 0.50±0.49 |
| chronos-bolt-base | M4-Daily | 4.93±3.82 | 5.03±3.22 | 1.00±0.85 | 0.63±0.59 |
| chronos-t5-large | REAL-V-TSFM | 9.32±18.46 | 8.40±9.69 | **5.45±31.23** | **5.58±34.82** |
| chronos-t5-large | M4-Daily | 7.11±7.02 | 7.18±4.65 | 1.56±1.26 | 0.98±0.88 |
| timesfm-2.0-500m | REAL-V-TSFM | 6.97±16.63 | 6.24±9.17 | 0.91±1.02 | 0.64±0.65 |
| timesfm-2.0-500m | M4-Daily | **1.9±1.08** | **2.03±2.55** | **0.39±0.22** | **0.23±0.25** |
| LinearRegression | REAL-V-TSFM | 15.52±28.44 | 14.28±20.21 | 1.00 | 1.00 |

### Ablation Study (Effect of Model Scale)

| Model | REAL-V-TSFM MAPE↓ | REAL-V-TSFM WQL↓ | Parameters |
|------|-------------------|------------------|--------|
| chronos-bolt-tiny | 7.43±17.40 | 0.92±0.88 | ~7M |
| chronos-bolt-mini | 7.37±17.22 | 0.92±0.90 | ~21M |
| chronos-bolt-small | 7.50±18.80 | 0.92±0.88 | ~48M |
| chronos-bolt-base | 7.32±17.06 | 0.93±0.90 | ~205M |
| chronos-t5-tiny | 10.40±20.13 | 5.40±30.04 | ~8M |
| chronos-t5-large | 9.32±18.46 | 5.45±31.23 | ~709M |

### Key Findings

1. **REAL-V-TSFM is genuinely more challenging**: Nearly all models rank worst or second-worst on this dataset.
2. **chronos-t5-large exhibits severely degraded distributional capture**: Its Agg. Relative WQL reaches 5.45 on REAL-V-TSFM versus approximately 1.0 on other datasets, indicating the model completely fails to capture predictive distributions reflecting real physical motion.
3. **timesfm-2.0 demonstrates relatively better generalization**: Its decoder-only architecture yields more stable performance on this dataset, though the substantial advantage observed on M4 does not transfer to REAL-V-TSFM.
4. **Scaling laws are not evident in TSFMs**: From 7M to 709M parameters, improvement in MAPE is marginal, with tiny models occasionally matching or outperforming large ones.
5. **The bolt variant significantly outperforms the t5 variant**: Architectural improvements prove more effective than parameter scaling.

## Highlights & Insights

1. **A novel data acquisition paradigm**: Extracting time series from video is a simple yet insightful idea. Given that video constitutes one of the richest data sources on the internet, this pipeline has the potential to substantially scale the diversity of temporal data.
2. **Revealing a critical weakness of TSFMs**: Synthetic data augmentation strategies (KernelSynth, TSMixup) generate distributions that fail to cover real physical dynamics — an important warning for the broader TSFM community.
3. **Questioning the scaling law**: In TSFMs, increasing model scale does not yield significant gains in generalization, contrasting with prevailing assumptions in the LLM community.
4. **Dataset distribution analysis**: PCA projections comparing REAL-V-TSFM and M4 visually demonstrate the distributional limitations of existing benchmarks.

## Limitations & Future Work

1. **Limited dataset scale**: 6,130 sequences is modest compared to commonly used benchmarks (M4 contains over 100,000 sequences).
2. **Exclusive use of Lucas-Kanade optical flow**: More advanced methods such as RAFT could potentially yield higher-quality trajectories.
3. **Only zero-shot forecasting is evaluated**: Few-shot fine-tuning scenarios are not explored (acknowledged in the paper).
4. **Noise in optical flow extraction is unavoidable**: Forward-backward consistency checking reduces errors but cannot eliminate them entirely.
5. **Limited task scope**: Only forecasting is evaluated; other temporal tasks such as anomaly detection and classification are not considered.
6. **Questionable classification under "3D Vision"**: This paper more appropriately belongs to the domain of time-series analysis and foundation model evaluation.

## Related Work & Insights

- Chronos is currently the most influential TSFM; its evaluation across 42 datasets appears comprehensive, yet this paper's experiments reveal significant remaining generalization gaps.
- The GIFT-EVAL benchmark provides a standardized evaluation framework upon which this paper's assessment is built.
- The LaSOT dataset serves as the video source, providing rich long-sequence object tracking videos.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Evaluating TSFMs using time series extracted from video is a novel angle, though the technical contribution (the optical flow extraction pipeline) is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐ — Three primary models and one baseline are compared, but the range of foundation models and evaluation tasks is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear and analysis is substantive.
- **Value**: ⭐⭐⭐⭐ — Provides an important warning about generalization blind spots for the TSFM community; the open-sourced dataset and pipeline have meaningful dissemination value.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models](listening_between_the_frames_bridging_temporal_gaps_in_large_audio-language_mode.md)
- [\[ICLR 2026\] Online Time Series Prediction Using Feature Adjustment](../../ICLR2026/video_understanding/online_time_series_prediction_using_feature_adjustment.md)
- [\[CVPR 2026\] Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](../../CVPR2026/video_understanding/realworld_point_tracking_with_verifierguided_pseud.md)
- [\[AAAI 2026\] UVLM: Benchmarking Video Language Model for Underwater World Understanding](uvlm_benchmarking_video_language_model_for_underwater_world_understanding.md)
- [\[ICLR 2026\] Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding](../../ICLR2026/video_understanding/lets_split_up_zero-shot_classifier_edits_for_fine-grained_video_understanding.md)

<!-- RELATED:END -->
