---
title: >-
  [Paper Note] Harmonic Dataset Distillation for Time Series Forecasting
description: >-
  [AAAI 2026][Time Series][Dataset Distillation] This paper proposes HDT (Harmonic Dataset Distillation for Time Series Forecasting), which decomposes time series into sinusoidal bases via FFT and aligns the core periodic…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Dataset Distillation"
  - "Time Series Forecasting"
  - "Frequency-Domain Optimization"
  - "Harmonic Matching"
  - "Cross-Architecture Generalization"
date: 2026-05-08
content_hash: efc8f38a139efda1
---

# Harmonic Dataset Distillation for Time Series Forecasting

**Conference**: AAAI 2026
**arXiv**: [2603.03760](https://arxiv.org/abs/2603.03760)  
**Code**: None  
**Area**: Time Series Forecasting / Dataset Distillation
**Keywords**: Dataset Distillation, Time Series Forecasting, Frequency-Domain Optimization, Harmonic Matching, Cross-Architecture Generalization

## TL;DR

This paper proposes HDT (Harmonic Dataset Distillation for Time Series Forecasting), which decomposes time series into sinusoidal bases via FFT and aligns the core periodic structure of synthetic and real data through Harmonic Matching in the frequency domain, achieving strong cross-architecture generalization and favorable scalability for time series dataset distillation.

## Background & Motivation

Time series forecasting faces severe challenges in **data storage and computational cost**: industrial sensors and biomedical monitors generate terabytes of data daily, and the emergence of large foundation models such as TimesFM and Moirai further amplifies the computational burden.

**Dataset Distillation (DD)**—synthesizing a small yet informative dataset such that models trained on it approximate the performance of training on the full data—is a promising solution. However, directly applying image DD methods to time series forecasting suffers from two fundamental limitations:

### Limitations of Window-Based Methods (Figure 1a)

Existing methods segment time series into fixed-size windows (e.g., 96-step input + 96-step output), treating each window as an independent sample for distillation. This "local-to-local" matching ignores the global structure of time series:

**Limited Scalability (L1)**: Increasing the synthetic data length $M$ merely extends existing local patterns without capturing broader global structure, leading to diminishing returns.

**Architecture Overfitting (L2)**: Local optimization entirely neglects the global dependencies that constitute the full sequence, causing distilled data to overfit to the inductive biases of specific backbone models, resulting in poor cross-architecture generalization.

### Core Insight

The essence of time series lies in their **global periodic structure**. Decomposing a series into sinusoidal bases via FFT yields basis functions that exert global influence over the entire sequence. Performing distillation in the frequency domain ensures that every update modifies the overall structure of the synthetic series without disrupting temporal dependencies.

## Method

### Overall Architecture

The HDT distillation pipeline (Figure 2):

1. Transform both the real data $\mathcal{X}$ and synthetic data $\mathcal{S}$ to the frequency domain via FFT.
2. Select the top-$k$ frequency components with the largest amplitudes as **Harmonics**.
3. Align the harmonic distributions of the two via the **Harmonic Matching loss** $\mathcal{L}_{\text{harm}}$.
4. Ensure consistent training behavior via the **Gradient Matching loss** $\mathcal{L}_{\text{grad}}$.
5. Optimize harmonic coefficients in the frequency domain, and recover the distilled data via iFFT.

### Key Designs

#### 1. **Harmonic Matching**

To achieve precise alignment of frequency components, subsequences of length $M$ (equal to the synthetic data length) are first sampled from the real data, and FFT is applied to both:

$$\mathcal{F_X} = \text{FFT}(\mathcal{X}_{\text{sub}}), \quad \mathcal{F_S} = \text{FFT}(\mathcal{S})$$

The top-$k$ frequency components with the largest amplitudes are selected as harmonics $\mathcal{H}$:

$$\mathcal{H} = \text{arg top-}k_{i \in [0, \lfloor M/2 \rfloor]}(|\mathcal{F_X}[i]|)$$

Only the harmonic components are retained and all others are zeroed out, yielding $\tilde{\mathcal{F_X}}$ and $\tilde{\mathcal{F_S}}$.

The **harmonic loss** minimizes the $L_p$ distance between their amplitudes:

$$\mathcal{L}_{\text{harm}} = \||\tilde{\mathcal{F_X}}| - |\tilde{\mathcal{F_S}}|\|_p$$

This loss acts as a regularizer that forces the periodic structure of the synthetic data to align with that of the real data. Since harmonics are **intrinsic, model-agnostic properties** of the data, this avoids overfitting to any particular backbone model.

#### 2. **Theoretical Guarantee (Theorem 1)**

The authors provide a rigorous theoretical proof that **minimizing the harmonic loss guarantees that the synthetic data preserves the global temporal dependency structure of the real data**.

The core theorem is grounded in the relationship between the power spectral density (PSD) and the autocorrelation function (ACF) via the Wiener–Khintchine theorem:

$$\max_{|k| \leq K} |r_{\mathcal{S}}(k) - r_{\mathcal{X}}(k)| \leq C \cdot \varepsilon$$

where $\varepsilon$ is the frequency-domain approximation error. This implies that the better the harmonic alignment in the frequency domain, the closer the autocorrelation structure of the synthetic data is to that of the real data.

#### 3. **Global Update Mechanism and Scalability**

Because each harmonic is a sinusoidal basis function with global influence over the entire sequence, every update in the frequency domain modifies the overall structure of the synthetic series.

Increasing the synthetic data length $M$ → a wider representable range of periods → ability to capture longer-period global structure → meaningful and continuous performance improvement with $M$ (resolving L1).

#### 4. **Gradient Matching**

In addition to frequency-domain harmonic matching, gradient matching is employed as the distillation loss. The time-domain signals are first reconstructed from harmonics via iFFT:

$$\mathcal{X_H} = \text{iFFT}(\tilde{\mathcal{F_X}}), \quad \mathcal{S_H} = \text{iFFT}(\tilde{\mathcal{F_S}})$$

Multi-step gradients of the model on the real and synthetic data are then matched:

$$\mathcal{L}_{\text{grad}} = \frac{\|\mathcal{T}_j(\theta, \mathcal{S_H}) - \mathcal{T}_i(\theta, \mathcal{X_H})\|_2^2}{\|\theta - \mathcal{T}_i(\theta, \mathcal{X_H})\|_2^2}$$

### Loss & Training

The final optimization objective is:

$$\underset{\mathcal{F_S}}{\text{argmin}} \; \mathcal{L}_{\text{grad}} + \lambda \mathcal{L}_{\text{harm}}$$

- $\lambda$ balances the two losses.
- The optimization variable is the frequency-domain coefficient $\mathcal{F_S}$ (rather than time-domain data points).
- After convergence, the final distilled data $\mathcal{S}$ is recovered via iFFT.
- DLinear is used as the distillation backbone model.

## Key Experimental Results

### Main Results ($M=384$, MSE)

**DLinear Backbone** (L = same-architecture evaluation, T = iTransformer evaluation, C = xPatch evaluation):

| Method | ETTh1 L/T/C | ETTh2 L/T/C | ETTm2 L/T/C | Electricity L/T/C |
|--------|-------------|-------------|-------------|-------------------|
| Random | 0.945/0.757/0.664 | 1.860/0.406/0.359 | 1.504/0.256/0.234 | 0.400/0.327/0.351 |
| MTT | 0.521/0.640/0.587 | 0.661/0.387/0.346 | 0.702/0.257/0.248 | 0.342/0.412/0.489 |
| CondTSF | 0.510/0.494/0.492 | 0.392/0.336/0.325 | 0.223/0.209/0.204 | 0.231/0.241/0.238 |
| **HDT** | **0.430/0.421/0.409** | **0.359/0.331/0.311** | **0.211/0.205/0.201** | **0.208/0.239/0.232** |
| Full Data | 0.386/0.389/0.384 | 0.326/0.314/0.296 | 0.186/0.185/0.177 | 0.195/0.152/0.175 |

Key finding: Prior methods exhibit sharp performance degradation under cross-architecture evaluation (sometimes worse than Random), while HDT remains stable across all settings.

### Ablation Study

| Method | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Electricity | Traffic |
|--------|-------|-------|-------|-------|-------------|---------|
| Base (window gradient matching) | 0.583 | 0.465 | 0.905 | 0.402 | 0.414 | 0.934 |
| Base + Decomposition (frequency-domain gradient matching) | 0.545 | 0.420 | 0.814 | 0.325 | 0.376 | 0.902 |
| HDT (full method) | **0.420** | **0.334** | **0.386** | **0.206** | **0.226** | **0.760** |

Frequency-domain operations yield substantial improvements, and harmonic matching further provides a significant additional gain.

### Efficiency and Large-Scale Experiments

| Experiment | Result |
|------------|--------|
| Training speedup (iTransformer, Electricity) | Full data: 1650s → Distilled data: 1.98s (**834× speedup**) |
| Training speedup (iTransformer, Traffic) | Full data: 4266s → Distilled data: 2.32s (**1839× speedup**) |
| Large-scale CA dataset (length 201K, 8600 variates) | HDT: 44.25 MSE vs. CondTSF: 197.95 vs. Full: 46.63 |
| Moirai-Large fine-tuning (311M parameters) | Few-shot + HDT: MSE 1.417, only 2.5% behind full fine-tuning, 80× faster |

### Key Findings

1. **Cross-architecture generalization is HDT's greatest advantage**: Prior methods suffer severe performance degradation when the distillation backbone and evaluation model differ, whereas HDT maintains a minimal MSE gap.
2. **Scalability**: HDT continues to improve as $M$ increases, while other methods saturate beyond a certain size.
3. **Negligible distillation overhead**: The $O(M\log M)$ complexity of FFT is negligible relative to the gradient computation of the backbone model.
4. **Foundation model fine-tuning**: Distilled data can be used for few-shot fine-tuning of large foundation models, achieving near-full fine-tuning performance at a fraction of the training cost.

## Highlights & Insights

1. **Paradigm shift from local to global**: Migrating distillation from time-domain local windows to frequency-domain global harmonics is an elegant and theoretically grounded design.
2. **Unification of theory and practice**: Theorem 1 rigorously proves, via the PSD–ACF relationship, that harmonic matching preserves temporal dependencies.
3. **Fundamental resolution of cross-architecture generalization**: Since harmonics are intrinsic properties of the data rather than artifacts of any specific model, model-agnosticism is guaranteed by construction.
4. **Significant practical value**: 834×–1839× training speedups, effectiveness on large-scale datasets, and applicability to foundation model fine-tuning each carry immediate industrial relevance.

## Limitations & Future Work

1. **Selection of synthetic data size $M$**: Although performance improves continuously with $M$, the optimal value must be determined empirically.
2. **Hyperparameter selection for the number of harmonics $k$**: The choice of $k$ in top-$k$ affects results, yet no adaptive selection strategy is provided.
3. **Primarily univariate treatment**: Although experiments include multivariate datasets, frequency-domain decomposition is performed independently per channel, without exploiting cross-channel structure.
4. **Choice of distillation backbone**: Experiments primarily use DLinear as the distillation backbone; the effect of alternative backbones is not thoroughly explored.
5. Extending harmonic matching to other time series tasks (e.g., classification, anomaly detection) warrants future investigation.

## Related Work & Insights

- **DC** (Zhao & Bilen, 2021): Pioneering work on gradient matching.
- **MTT** (Cazenavette et al., 2022): Trajectory matching approach.
- **CondTSF** (Ding et al., 2024): The first DD method specifically targeting TSF, though still window-based.
- **Wiener–Khintchine Theorem**: The duality between PSD and ACF, which serves as the mathematical foundation for the theoretical guarantee of harmonic matching.
- Insight: When data exhibit strong global structure (e.g., periodicity in time series), operating in a transform domain (frequency domain) rather than the original domain (time domain) may be substantially more effective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Frequency-domain distillation combined with harmonic matching constitutes a novel paradigm with solid theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 backbones × 6 datasets + cross-architecture evaluation + large-scale experiments + foundation model fine-tuning.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear, theoretical derivations are rigorous, and experimental design is systematic.
- Value: ⭐⭐⭐⭐⭐ — Addresses two fundamental problems in TSF dataset distillation with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](../../NeurIPS2025/time_series/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](../../ICLR2026/time_series/fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[AAAI 2026\] Optimal Look-back Horizon for Time Series Forecasting in Federated Learning](optimal_look-back_horizon_for_time_series_forecasting_in_federated_learning.md)

</div>

<!-- RELATED:END -->
