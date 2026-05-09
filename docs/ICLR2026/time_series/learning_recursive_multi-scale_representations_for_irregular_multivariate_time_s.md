---
title: >-
  [Paper Note] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting
description: >-
  [ICLR 2026][Time Series][Irregular time series] This paper proposes ReIMTS, a plug-and-play framework that preserves the original sampling patterns of irregular multivariate time series (IMTS) via time-period-based recursive partitioning (rather than resampling), combined with an irregularity-aware representation fusion mechanism for multi-scale modeling. ReIMTS achieves an average improvement of 27.1% across six IMTS backbones.
tags:
  - ICLR 2026
  - Time Series
  - Irregular time series
  - multi-scale modeling
  - recursive partitioning
  - sampling pattern preservation
  - forecasting
date: 2026-05-08
content_hash: caa3856b70976f7f
---

# Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting

**Conference**: ICLR 2026
**arXiv**: [2602.21498](https://arxiv.org/abs/2602.21498)
**Code**: [Available](https://github.com/Ladbaby/PyOmniTS)
**Area**: Time Series
**Keywords**: Irregular time series, multi-scale modeling, recursive partitioning, sampling pattern preservation, forecasting

## TL;DR

This paper proposes ReIMTS, a plug-and-play framework that preserves the original sampling patterns of irregular multivariate time series (IMTS) via time-period-based recursive partitioning (rather than resampling), combined with an irregularity-aware representation fusion mechanism for multi-scale modeling. ReIMTS achieves an average improvement of 27.1% across six IMTS backbones.

## Background & Motivation

Irregular multivariate time series (IMTS) are prevalent in domains such as healthcare and meteorology, where observation intervals are non-uniform and different variables may be observed at misaligned timestamps. The sampling pattern itself carries important information — for instance, in ICU settings, a transition from dense to sparse monitoring reflects a patient's improvement from critical to stable condition.

Core limitations of existing multi-scale methods:

- **Regular time series methods** (Scaleformer, TimeMixer, Pathformer) assume uniform sampling and are not applicable to IMTS.
- **IMTS multi-scale methods** (Warpformer, Hi-Patch, HD-TTS) rely on **resampling** to obtain coarser-grained sequences, which destroys original sampling patterns — e.g., the dense-to-sparse sampling pattern of Bilirubin in PhysioNet'12 is disrupted after downsampling.
- Sampling pattern information (e.g., emergency-to-routine monitoring transitions) is clinically critical and should not be discarded.

## Method

### Overall Architecture

ReIMTS is a plug-and-play multi-scale framework compatible with most encoder-decoder IMTS models. The core idea is to recursively partition samples into sub-samples of shorter time periods at each scale level based on time periods, while keeping the original timestamps of all observations unchanged.

The framework consists of three components: (1) a recursive partitioning module, (2) a backbone encoder at each scale level, and (3) an irregularity-aware representation fusion module.

### Key Designs

**1. Time-Period-Based Recursive Partitioning**

At scale level $n$, samples are partitioned into $P^n = T^1/T^n$ sub-samples according to time period $T^n$. The key distinction is:

- Partitioning is based on **time periods** (e.g., 12 hours, 24 hours), not the number of observations.
- Original timestamps are fully preserved; zero-padding is used for alignment.
- This avoids the issue in observation-count-based partitioning where sub-samples correspond to different real-world time spans.

For example, in PhysioNet'12: level 1 covers the full 48 hours; level 2 partitions by 24-hour periods (2 sub-samples); level 3 partitions by 12-hour periods (4 sub-samples), forming a global-to-local hierarchy.

**2. Multi-Scale Representation Learning**

Each scale level $n$ employs an independent backbone encoder $\mathcal{F}^n_{\text{enc}}$:

$$\mathbf{E}^n = \mathcal{F}^n_{\text{enc}}(\mathbf{S}^n)$$

Latent representations are categorized into three types: temporal representations $\mathbf{E}^n_{\text{time}} \in \mathbb{R}^{P^n \times L^n \times D}$, variable representations $\mathbf{E}^n_{\text{var}} \in \mathbb{R}^{P^n \times V \times D}$, and observation representations $\mathbf{E}^n_{\text{obs}} \in \mathbb{R}^{P^n \times L^n \times V \times D}$.

The upper-level global representation $\mathbf{H}^n$ is transformed via splitting (for temporal/observation representations) or replication (for variable representations) to match the shape of the lower-level local representation $\mathbf{E}^{n+1}$.

**3. Irregularity-Aware Representation Fusion (IARF)**

At the lower scale $n+1$, a binary mask $\mathbf{M}^{n+1}$ distinguishes actual observations from padded values:

$$\mathbf{H}^n_{\text{IMTS}} = \begin{cases} \mathbf{H}^n \cdot \mathbf{M}^{n+1}, & \text{temporal/observation representations} \\ \mathbf{H}^n, & \text{variable representations} \end{cases}$$

A lightweight scoring layer computes fusion weights $\alpha = \text{ReLU}(\text{FF}(\mathbf{H}^n_{\text{IMTS}}))$, and the local and global representations are fused as:

$$\mathbf{G}^{n+1} = \mathbf{E}^{n+1} + \alpha \mathbf{H}^n_{\text{IMTS}}$$

Irregularity information in variable representations is already encoded by the IMTS backbone, whereas temporal/observation representations may still contain padded values and therefore require masking.

### Loss & Training

At the lowest scale level $N$, the decoder concatenates representations from all levels and decodes:

$$\hat{\mathbf{Z}} = \mathcal{F}_{\text{dec}}(\text{Concat}(\{\mathbf{G}^n\}_{n=1}^N))$$

Training uses MSE loss, computed only over prediction queries within the forecast window:

$$\mathcal{L} = \frac{1}{Y_Q} \sum_{j=1}^{Y_Q} (\hat{z_j} - z_j)^2$$

Models are trained for up to 300 epochs with an early stopping patience of 10.

## Key Experimental Results

### Main Results

Evaluation is conducted on 5 IMTS datasets (MIMIC-III/IV, PhysioNet'12, Human Activity, USHCN) and 26 baseline methods.

| Backbone | Original MSE(×10⁻¹) | +ReIMTS MSE(×10⁻¹) | Avg. Gain |
|---------|-----------------|-------------------|---------|
| PrimeNet | 9.04/6.25/7.93/26.84/4.57 | 4.76/3.58/3.01/0.82/1.71 | **↑62.3%** |
| mTAN | 8.51/5.09/3.75/0.89/5.65 | 6.37/4.04/3.51/0.89/1.70 | ↑24.3% |
| TimeCHEAT | 4.41/2.50/3.27/0.68/1.73 | 4.40/2.02/2.90/0.52/1.62 | ↑12.1% |
| GRU-D | 4.75/5.97/3.25/1.76/2.42 | 4.67/3.91/3.25/0.51/1.89 | ↑25.8% |
| GraFITi | 4.08/2.39/2.85/0.43/1.71 | 4.07/1.79/2.83/0.42/1.66 | ↑6.3% |

Comparison with other multi-scale IMTS methods (using GraFITi as backbone):

| Method | MIMIC-III | MIMIC-IV | PhysioNet'12 | Human Activity | USHCN |
|-----|----------|---------|-------------|---------------|-------|
| Warpformer | 4.09 | 2.42 | 2.88 | 0.54 | 1.77 |
| HD-TTS | 4.17 | 2.36 | 2.83 | 0.50 | 1.66 |
| Hi-Patch | 4.35 | 2.36 | 3.11 | 0.48 | 2.34 |
| **ReIMTS** | **4.07** | **1.79** | **2.83** | **0.42** | **1.66** |

### Ablation Study

| Variant | MIMIC-III | MIMIC-IV | PhysioNet'12 | Human Activity | USHCN |
|-----|----------|---------|-------------|---------------|-------|
| ReIMTS (full) | **4.07** | **1.79** | **2.83** | **0.42** | **1.66** |
| rp sample (no partitioning) | 4.99 | 1.92 | 2.83 | 0.45 | 1.69 |
| rp split (obs-count partitioning) | 5.02 | 2.36 | 3.20 | 0.61 | 2.31 |
| rp IARF (fusion→addition) | 4.20 | 1.84 | 2.79 | 0.47 | 1.89 |
| w/o IARF (no fusion) | 4.77 | 2.07 | 3.06 | 0.54 | 1.69 |

### Key Findings

- Time-period-based partitioning (ReIMTS) substantially outperforms observation-count-based partitioning (rp split), with a gap of up to 0.65 on USHCN.
- Older models (mTAN, GRU-D) augmented with ReIMTS can surpass more recent models.
- Efficiency analysis: ReIMTS with the GraFITi backbone achieves the fastest training speed and lowest GPU memory usage among Warpformer, HD-TTS, and Hi-Patch.

## Highlights & Insights

1. **Sampling-pattern-preserving multi-scale design**: Recursive partitioning by time period — rather than resampling — is both simple and effective.
2. **Plug-and-play compatibility**: Applicable to most encoder-decoder IMTS models, offering strong generality.
3. **Revitalizing older methods**: PrimeNet gains 62.3% and GRU-D gains 25.8%, demonstrating that multi-scale augmentation addresses a critical missing component.
4. **Efficiency advantage**: Combining ReIMTS with a lightweight backbone (e.g., GraFITi) simultaneously achieves state-of-the-art accuracy and efficiency.

## Limitations & Future Work

- The combination of ODE-based models with ReIMTS lacks theoretical justification.
- The noise-based latent representations of diffusion models are not directly compatible with ReIMTS's fusion mechanism.
- Time period lengths must be manually specified (dataset-specific settings are provided in the appendix); adaptive selection remains an open research direction.
- Only forecasting tasks are evaluated; other downstream tasks such as classification remain unexplored.

## Related Work & Insights

- Relationship to tPatchGNN and PrimeNet: these can be viewed as single-scale special cases of ReIMTS.
- Regular time series multi-scale methods such as Scaleformer destroy sampling pattern information through resampling.
- Insight: For other tasks involving irregular data (e.g., event sequences, point processes), multi-scale methods that preserve original temporal information may similarly prove beneficial.

## Rating

- Novelty: ⭐⭐⭐⭐ (Time-period-based partitioning is a concise and effective idea; the IARF fusion mechanism is well-motivated.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 datasets, 26 baselines, 6 backbones, complete ablation and efficiency analysis.)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, intuitive illustrations, thorough comparison with existing methods.)
- Value: ⭐⭐⭐⭐ (Strong practical utility due to plug-and-play design; open-sourced as part of PyOmniTS.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[NeurIPS 2025\] Learning Time-Scale Invariant Population-Level Neural Representations](../../NeurIPS2025/time_series/learning_time-scale_invariant_population-level_neural_representations.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](../../AAAI2026/time_series/freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](../../NeurIPS2025/time_series/multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
