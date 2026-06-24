---
title: >-
  [Paper Note] VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters
description: >-
  [ICML2025][Time Series][Time Series Forecasting] By reconstructing time series as images, VisionTS leverages ImageNet-pretrained MAE (Masked Autoencoders) for time series forecasting in a **zero-shot** setting, matching or even outperforming specialized time series foundation models without any training on time series data.
tags:
  - "ICML2025"
  - "Time Series"
  - "Time Series Forecasting"
  - "Vision Foundation Models"
  - "Masked Autoencoder"
  - "Zero-Shot Forecasting"
  - "Cross-Modal Transfer"
date: 2026-05-08
content_hash: 3105665cb36e00b2
---

# VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters

**Conference**: ICML2025  
**arXiv**: [2408.17253](https://arxiv.org/abs/2408.17253)  
**Code**: [Keytoyze/VisionTS](https://github.com/Keytoyze/VisionTS)  
**Area**: Time Series Forecasting / Cross-Modal Transfer  
**Keywords**: Time Series Forecasting, Vision Foundation Models, Masked Autoencoder, Zero-Shot Forecasting, Cross-Modal Transfer

## TL;DR

By reconstructing time series as images, VisionTS leverages ImageNet-pretrained MAE (Masked Autoencoders) for time series forecasting in a **zero-shot** setting, matching or even outperforming specialized time series foundation models without any training on time series data.

## Background & Motivation

Foundation models for time series forecasting (TSF) currently follow two main pipelines:

**Textual Pipeline**: Repurposing LLMs (e.g., GPT4TS, TimeLLM). However, the modality gap between language and time series is large, and the effectiveness of such transfer remains questionable.

**Time Series Pipeline**: Training models from scratch on large-scale time series datasets (e.g., Moirai, TimesFM). However, time series data is highly heterogeneous across lengths, frequencies, and domains, making dataset curation challenging.

This paper proposes a **third pipeline**: leveraging pre-trained vision models for TSF. The core observation is that images and time series share several key characteristics:

- **Continuity**: Both are continuous signals, unlike discrete text.
- **Homogeny**: Both are observations of real physical systems.
- **Information Density**: Both contain highly redundant information, unlike language with high semantic density.
- **Feature Similarity**: Pixel rows in ImageNet images naturally exhibit time series characteristics such as trends, seasonality, and stationarity.

## Method

VisionTS reformulates time series forecasting as an image reconstruction task using MAE. The core workflow is: **Segmentation → Normalization → Rendering → Alignment → Reconstruction → Prediction**.

### 1. Segmentation

Given a univariate input $X \in \mathbb{R}^L$, it is segmented into $\lfloor L/P \rfloor$ subseries of length $P$ based on the period $P$, and stacked into a two-dimensional matrix:

$$\boldsymbol{I}_{\text{raw}} \in \mathbb{R}^{P \times \lfloor L/P \rfloor}$$

The period $P$ can be determined via FFT or prior knowledge of the sampling frequency. When the series has no obvious periodicity, $P=1$ is directly set.

### 2. Normalization

Instance normalization is applied to $\boldsymbol{I}_{\text{raw}}$, and the standard deviation is scaled to a hyperparameter $r$ (default 0.4):

$$\boldsymbol{I}_{\text{norm}} = r \cdot \frac{\boldsymbol{I}_{\text{raw}} - \text{Mean}(\boldsymbol{I}_{\text{raw}})}{\text{Std}(\boldsymbol{I}_{\text{raw}})}$$

The design of $r < 1$ scales down the amplitude because the pixel value range of MAE during pre-training is bounded, preventing out-of-bounds reconstructed values.

### 3. Rendering

The normalized matrix is rendered into a grayscale image $\boldsymbol{I}_{\text{grey}} \in \mathbb{R}^{P \times \lfloor L/P \rfloor \times 3}$ by copying the single channel across three channels. Experiments show that additionally learning a color transformation yields no significant gain.

### 4. Alignment

The pre-trained MAE operates on images of size $224 \times 224$, divided into $N \times N$ patches (each of size $S \times S$). Let the number of visible patch columns be $n$, and the number of masked columns be $N - n$:

$$n = \left\lfloor c \cdot N \cdot \frac{L}{L + H} \right\rfloor$$

where $c \in [0,1]$ (default 0.4) controls the width of the visible region. Controlling the visible area ratio keeps the masking rate close to the 75% used in MAE pre-training. Bilinear interpolation is used to resize the grayscale image to $(N \cdot S, n \cdot S)$.

### 5. Reconstruction & Prediction

After MAE reconstructs the complete image, reverse operations are performed: resize back to the original segments $\rightarrow$ average across the three channels $\rightarrow$ inverse-normalize $\rightarrow$ flatten $\rightarrow$ extract the prediction window.

### Multivariate Processing

A Channel Independence strategy is adopted, where each variable is predicted independently without modeling inter-variable interactions.

## Key Experimental Results

### Long-Term Forecasting (8 datasets, averaged over {96,192,336,720} steps)

| Method | Pre-training Data | Setting | Avg MSE | Avg MAE |
|------|-----------|------|---------|---------|
| **VisionTS** | **Images** | **Zero-shot** | **0.309** | **0.345** |
| Moirai-S | Time Series | Zero-shot | 0.327 | 0.357 |
| Moirai-B | Time Series | Zero-shot | 0.310 | 0.344 |
| Moirai-L | Time Series | Zero-shot | 0.329 | 0.350 |
| TimeLLM | Text | Few-shot | 0.336 | 0.368 |
| GPT4TS | Text | Few-shot | 0.360 | 0.378 |
| PatchTST | None | Few-shot | 0.378 | 0.389 |

**Key Findings**: Zero-shot VisionTS achieves the best MSE on 7 out of 8 datasets, and its overall average MSE outperforms all TS-based and Text-based foundation models.

### Monash Benchmark (29 Datasets)

VisionTS outperforms most zero-shot foundation models on the aggregated metrics of the Monash benchmark, covering a wide variety of frequencies and domains.

### GIFT-Eval Benchmark (23 Datasets)

VisionTS also achieves competitive zero-shot performance compared to dedicated time series foundation models on the GIFT-Eval leaderboard.

### Fine-Tuning (1 epoch)

With only 1 epoch of fine-tuning, VisionTS reaches SOTA performance on most long-term forecasting benchmarks.

## Highlights & Insights

1. **Cross-Modal Free Lunch**: Pure vision models can yield high-quality forecasts without any time series adaptation, revealing deep similarities between images and time series.
2. **Minimalist Design**: The overall method introduces zero trainable parameters in the zero-shot setting, relying entirely on the pre-trained weights of MAE.
3. **Innovative Prompting Paradigm**: Reformulating TSF as masked image reconstruction in MAE, which is analogous to prompt tuning in NLP.
4. **Representation Visualization**: Under t-SNE visualization of the MAE encoder, representation overlaps are observed between time series data and ImageNet images, suggesting that images can act as a "bridge" across different time series domains.
5. **Large-Scale Evaluation**: Testing across 60+ datasets (8 long-term + 29 Monash + 23 GIFT-Eval), making it one of the largest-scale evaluations among TSF foundation models at the time.

## Limitations & Future Work

1. **Multivariate Limitations**: Uses only a channel-independent strategy without modeling cross-channel interactions, constrained by the number of image channels (only 3).
2. **Sensitivity to Hyperparameters**: The normalization scale $r$ and visible region ratio $c$ significantly affect performance, currently relying on empirical selection.
3. **Periodicity Assumption**: Segmentation relies heavily on the quality of estimating the period $P$, which may not be robust for non-periodic series.
4. **Resolution Limitations**: MAE uses a fixed 224×224 input size, which poses information compression loss for ultra-long sequences or high-resolution requirements.
5. **Grayscale Limitation**: Renderings are limited to grayscale, which does not fully exploit the capacity of MAE to model color images.

## Related Work & Insights

- **TimesNet** (Wu et al., 2023): Similarly transforms 1D time series to 2D structures, but requires training from scratch.
- **Moirai** (Woo et al., 2024): A time series-based foundation model, which serves as the primary zero-shot baseline for VisionTS.
- **MAE** (He et al., 2022): The backbone network for VisionTS, leveraging its image reconstruction capabilities.
- **SparseTSF** (Lin et al., 2024): A sparse forecasting method that also utilizes periodic segmentation.
- **Insight**: Pre-trained vision models may pack transferability value for more non-vision tasks, which makes it worth exploring cross-modality directions such as image $\rightarrow$ audio or image $\rightarrow$ sensor.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to systematically demonstrate that pure vision models can perform zero-shot time series forecasting, offering a highly innovative research perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation on 60+ datasets, including ablation studies and cross-modal representation analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, abundant figures and tables, though some mathematical notations are dense.
- Value: ⭐⭐⭐⭐⭐ — Opens up a new paradigm of "vision $\rightarrow$ time series" cross-modal foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction](imts_is_worth_time_times_channel_patches_visual_masked_autoencoders_for_irregula.md)
- [\[ACL 2025\] Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models](../../ACL2025/time_series/revisiting_llms_as_zero-shot_time_series_forecasters_small_noise_can_break_large.md)
- [\[NeurIPS 2025\] Rotary Masked Autoencoders are Versatile Learners](../../NeurIPS2025/time_series/rotary_masked_autoencoders_are_versatile_learners.md)
- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](../../NeurIPS2025/time_series/tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)
- [\[ICLR 2026\] Zero-shot Forecasting by Simulation Alone](../../ICLR2026/time_series/zero-shot_forecasting_by_simulation_alone.md)

</div>

<!-- RELATED:END -->
