---
title: >-
  [Paper Note] LightGTS: A Lightweight General Time Series Forecasting Model
description: >-
  [ICML2025][Model Compression][Time Series Foundation Models] This paper proposes LightGTS, which leverages the inherent scale-invariant periodic inductive bias of time series. Through two core techniques, Periodical Tokenization and Periodical Parallel Decoding, it achieves SOTA performance in both zero-shot and full-shot settings across 9 benchmark datasets using fewer than 5 million parameters, representing a 10-100x reduction in size compared to existing time series founda…
tags:
  - "ICML2025"
  - "Model Compression"
  - "Time Series Foundation Models"
  - "Periodical Tokenization"
  - "Lightweight Forecasting"
  - "Cross-Scale Generalization"
date: 2026-05-08
content_hash: b4ba40782386b402
---

# LightGTS: A Lightweight General Time Series Forecasting Model

**Conference**: ICML2025  
**arXiv**: [2506.06005](https://arxiv.org/abs/2506.06005)  
**Code**: [decisionintelligence/LightGTS](https://github.com/decisionintelligence/LightGTS)  
**Area**: Time Series  
**Keywords**: Time Series Foundation Models, Periodical Tokenization, Lightweight Forecasting, Cross-Scale Generalization

## TL;DR

This paper proposes LightGTS, which leverages the inherent scale-invariant periodic inductive bias of time series. Through two core techniques, Periodical Tokenization and Periodical Parallel Decoding, it achieves SOTA performance in both zero-shot and full-shot settings across 9 benchmark datasets using fewer than 5 million parameters, representing a 10-100x reduction in size compared to existing time series foundation models.

## Background & Motivation

**Core Problem**: Existing Time Series Foundation Models (TSFMs) rely on large-scale multi-source pre-training and massive model parameters to achieve generalization ability, which incurs significant computational overhead and hinders deployment in resource-constrained scenarios.

**Key Observation**: Time series data possess two unique attributes: **scale** (sampling rate, e.g., every 15 minutes or hourly) and **intrinsic period** (the time interval at which patterns repeat, e.g., daily cycle). At different scales, the same intrinsic period corresponds to different **cycle lengths**. For instance, the daily cycle length for ETTh2 (hourly sampling) is 24, while for ETTm2 (15-minute sampling) it is 96.

**Limitations of Prior Work**:

- **Fixed Tokenization**: Each token contains a fixed number of data points, leading to inconsistent information density across different scales and disrupting the continuity and structural integrity of periodic patterns.
- The authors' case study clearly demonstrates that while fixed tokenization methods trained on ETTh1 can identify periods of same-scale datasets, their performance degrades significantly when transferring to datasets of different scales.
- This forces the model to require more parameters to compensate for this defect, resulting in increased computational costs.

**Core Idea**: Leverage the scale-invariant intrinsic period in time series as an inductive bias, and design an adaptive periodical tokenization and decoding scheme to compress model parameters while maintaining high performance.

## Method

### Overall Architecture

LightGTS is based on a Transformer Encoder-Decoder architecture, consisting of three core components:

1. **Periodical Tokenization**: Adaptively segmenting time series into period-aligned patches.
2. **Flex Projection Layer**: Handling patches of varying lengths and mapping them into a unified semantic space.
3. **Periodical Parallel Decoding (PPD)**: Utilizing the last token of the encoder to initialize the decoder input.

### 1. Periodical Patching

Given a univariate time series $\mathbf{x} \in \mathbb{R}^L$, the period length is first determined via period detection:

$$P = \text{PeriodsFinding}(\mathbf{x})$$

This is derived directly when prior knowledge (such as a known sampling rate) is available; otherwise, it is automatically detected using FFT. The sequence is then segmented into non-overlapping periodical patches $\mathbf{X}_p \in \mathbb{R}^{P \times N}$, where $N = \lfloor L/P \rfloor$. Each patch precisely aligns with a full period, ensuring consistent semantics carried by tokens across different scales.

### 2. Flex Projection Layer

**Problem**: Different datasets have different period lengths $P$, and fixed-weight patch embeddings cannot handle variable-length patches. Simply resizing weights via linear interpolation introduces bias, causing $\mathbf{x} \cdot \theta \neq \mathbf{x'} \cdot \theta'$.

**Solution**: Formalize the linear interpolation as a linear transformation $\text{Interp}(\mathbf{x})_P^{P'} = \mathbf{x} \cdot \mathbf{A}$, where $\mathbf{A} \in \mathbb{R}^{P \times P'}$. By solving the optimization problem:

$$\theta' = \arg\min_{\theta'} \mathbb{E}_{x \sim \mathcal{X}} \left[ \|\mathbf{x} \cdot \theta - \mathbf{x}\mathbf{A} \cdot \theta'\|_F^2 \right]$$

Theoretical derivation (considering distribution consistency constraints under RevIN normalization) yields a closed-form solution:

$$\theta' = \delta^{-1} (\mathbf{A})^+ \theta, \quad \delta = \sqrt{P/P'}$$

where $(\mathbf{A})^+$ is the Moore-Penrose pseudoinverse. This **Flex-resize** operation requires no additional learning; it guarantees embedding equivalence across different patch sizes solely through mathematical transformation.

The model defines reference weights $\theta_e, \theta_d \in \mathbb{R}^{P^* \times D}$ (defaulting to $P^*=48$), which are dynamically resized during forward propagation to match the period length of the current sequence.

### 3. Encoding

A standard Transformer Encoder is used, incorporating **RoPE (Rotary Position Embedding)** in the attention mechanism to model the relative positional relationships between tokens:

$$\mathbf{S}_{ij} = (\mathbf{W}_Q \mathbf{x}_e^i)^T \mathbf{R}_{i-j} (\mathbf{W}_K \mathbf{x}_e^j)$$

$$\mathbf{Attn}_i = \sum_j \frac{\exp(\mathbf{S}_{ij})}{\sum_k \exp(\mathbf{S}_{ik})} (\mathbf{W}_V \mathbf{x}_e^j)$$

### 4. Periodical Parallel Decoding (PPD)

**Key Designs**: Replicating the last token $\mathbf{e}^N$ of the encoder $K = \lceil F/P \rceil$ times to serve as the decoder input. Key insights include:

- The last token maintains temporal continuity with future predictions.
- Exploiting the consistency of periodic structure between the historical and prediction horizons.
- The non-autoregressive scheme avoids accumulated errors and reduces computational overhead.

Applying an exponentially decaying weight $\omega(\tau) = 1/e^\tau$ to the replicated tokens, and then feeding them in parallel into the decoder:

$$\mathbf{Z} = \text{Decoder}(\{\omega(j) \mathbf{h}^j\}, \mathbf{E})$$

$$\hat{\mathbf{Y}} = \text{Flex-resize}(\theta_d)_{P}^{P^*} \cdot \mathbf{Z}$$

### 5. Loss & Training

Standard MSE loss: $\mathcal{L}_{\text{MSE}} = \|\mathbf{Y} - \hat{\mathbf{Y}}\|_F^2$

### Model Configurations

| Variant | Encoder Layers | Decoder Layers | Hidden Dim | FFN Dim | Parameter Count |
|---|---|---|---|---|---|
| LightGTS-tiny | 1 | 1 | 256 | 512 | 1.3M |
| LightGTS-mini | 3 | 3 | 256 | 512 | 4M |

Pre-training configuration: historical token count $N=10$, prediction token count $K=4$, reference patch size $P^*=48$, batch size = 8192, learning rate $5 \times 10^{-4}$, Adam optimizer with StepLR decay.

## Key Experimental Results

### Pre-training & Evaluation Datasets

- **Pre-training**: Covers 30+ open datasets (Monash, UEA, UCR, etc.) across energy, nature, health, transportation, Web, economy, and other fields, with sampling frequencies ranging from milliseconds to monthly.
- **Evaluation**: 9 benchmark datasets (ETTh1/h2/m1/m2, Weather, Traffic, Electricity, Solar, Exchange), strictly non-overlapping with the pre-training datasets.
- Prediction lengths: $F \in \{96, 192, 336, 720\}$

### Zero-shot Prediction Results (Table 1, Average of Each Prediction Length)

| Dataset | LightGTS-mini | Timer | MOIRAI | Chronos | TimesFM | Time-MoE |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| ETTm1 | **0.327** | 0.768 | 0.390 | 0.551 | 0.435 | 0.376 |
| ETTm2 | **0.247** | 0.315 | 0.276 | 0.293 | 0.347 | 0.315 |
| ETTh1 | **0.388** | 0.562 | 0.510 | 0.533 | 0.479 | 0.394 |
| Weather | **0.208** | 0.292 | 0.260 | 0.288 | - | 0.270 |
| Traffic | **0.561** | 0.613 | - | 0.615 | - | - |
| Solar | **0.191** | 0.771 | 0.714 | 0.393 | 0.500 | 0.411 |
| Electricity | 0.213 | 0.297 | **0.188** | - | - | - |

- LightGTS-mini reduces the average MSE by more than **30%** (vs. the strongest baseline).
- Even LightGTS-tiny (1.3M parameters) achieves a 27% average MSE reduction.

### Full-shot Prediction Results (Table 2, LightGTS-mini vs SOTA Small Models)

| Dataset | LightGTS (full-shot) | PDF | iTransformer | PatchTST |
|--------|:---:|:---:|:---:|:---:|
| ETTm1 | **0.321** | 0.342 | 0.347 | 0.349 |
| Traffic | **0.393** | 0.395 | 0.397 | 0.397 |
| Solar | **0.179** | 0.200 | 0.202 | 0.200 |
| Electricity | **0.156** | 0.160 | 0.163 | 0.171 |

Under the full-shot setting, it achieves a **7%** average MSE reduction compared to 6 SOTA baselines. On 5 datasets, its zero-shot performance already surpasses the full-shot results of the baselines.

### Efficiency Comparison (Table 3, ETTm1 F=720)

| Model | Parameters | MACs | Max Memory (MB) | Inference Time (s) |
|------|--------|------|-------------|-----------|
| Time-MoE | 453M | 5252.9G | 14131 | 2.13 |
| Chronos | 700M | 92327.9G | 10269 | 34.33 |
| MOIRAI | 300M | 97.36G | 2009 | 0.10 |
| Timer | 67.4M | 52.6G | 1435 | 0.08 |
| PatchTST | 6.3M | 225M | 672 | 0.01 |
| **LightGTS** | **4M** | **213M** | 713 | **0.01** |

LightGTS has 17x fewer parameters than Timer and 175x fewer than Chronos, with MACs more than 450x smaller than MOIRAI.

### Ablation Study (Table 4, Zero-shot Average)

| Decoding Method | Tokenization Method | ETT-avg | Weather | Electricity | Traffic |
|---------|---------|:---:|:---:|:---:|:---:|
| PPD | Periodical | **0.328** | **0.208** | **0.213** | **0.561** |
| PPD | Fixed | 0.436 | 0.262 | 0.226 | 0.621 |
| AR | Periodical | 0.341 | 0.226 | 0.229 | 0.634 |
| AR | Fixed | 0.442 | 0.265 | 0.231 | 0.630 |
| MAE | Periodical | 0.388 | 0.260 | 0.322 | 0.746 |

- Periodical Tokenization consistently outperforms Fixed Patching under all decoding methods.
- PPD consistently outperforms AR and MAE, with the gain being more significant when combined with Periodical Patching.
- MAE decoding performs the worst, likely due to the gap between the reconstruction objective and the forecasting task.

### Decoder Input Selection (Table 6)

| Initialization Method | ETT-avg | Weather | Electricity | Traffic |
|-----------|:---:|:---:|:---:|:---:|
| Last token | **0.328** | **0.208** | **0.213** | **0.561** |
| Learnable | 0.342 | 0.278 | 0.231 | 0.627 |
| CLS token | 0.343 | 0.341 | 0.234 | 0.634 |
| Mean token | 0.404 | 0.328 | 0.273 | 0.703 |

Last token achieves the best performance as it is most aligned with the periodicity and most relevant to the prediction task.

## Highlights & Insights

1. **Inductive-Bias-Driven Design Philosophy**: Instead of pursuing model scale, this work deeply explores the periodic inductive bias of time series. Replacing brute-force parameter stacking with the correct inductive bias is a commendable design philosophy.
2. **Theoretical Derivation of Flex Projection**: A closed-form solution is provided via SVD and Moore-Penrose pseudoinverse, adapting to different patch sizes without additional training, which is both elegant and practical.
3. **Cross-Scale Consistency**: Under different sampling granularities (e.g., 10-minute, 30-minute, 1-hour), LightGTS maintains stable performance, whereas Timer and Time-MoE exhibit large fluctuations (Fig. 4).
4. **Plug-and-Play Capability**: Periodical Tokenization can be directly applied to other TSFMs (e.g., Timer), obtaining a 19.23% MSE reduction without retraining (Table 11).

## Limitations & Future Work

1. **Dependence on Periodic Assumption**: For data lacking obvious periodicity (such as exchange rate data in Exchange), the gain of periodical tokenization is limited, and the period detected by FFT may be inaccurate.
2. **Period Length Requires Prior Knowledge or Sufficient Data**: When prior knowledge is missing and the data volume is insufficient, the period length detected by FFT may deviate from the true value, affecting the performance of Periodical Patching.
3. **Channel-Independent Paradigm**: Both pre-training and fine-tuning treat multivariate time series as univariate channels, thereby failing to model the dependencies between variables.
4. **Scope of Evaluation Datasets**: The 9 benchmark datasets are concentrated in domains with obvious periodicity such as energy, transportation, and weather, lacking verification on weakly periodic or non-stationary data (e.g., finance, social media).
5. **Reference Patch Size $P^*$**: Although experiments show insensitivity to $P^*$, selecting 48 as the default is highly empirical and lacks a systematic selection guide.

## Related Work & Insights

- **TimesNet (Wu et al., 2022)**: Also utilizes FFT to discover periodicity, but LightGTS leverages periods at the tokenization level, which is more fundamental.
- **PatchTST (Nie et al., 2023)**: A baseline with fixed patch size; the periodical patching of LightGTS serves as a key improvement over physical patching.
- **Timer (Liu et al., 2024)**: Also a Transformer-based TSFM, against which LightGTS directly compares and demonstrates the limitations of fixed tokenization.
- **MOIRAI (Woo et al., 2024)**: Predefines multiple patch sizes based on sampling frequencies, but still relies on discrete selection rather than continuous adaptation.
- **TTMs**: Uses CV-inspired patch merging for adaptation, but is limited by predefined patch sizes.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of Periodical Tokenization and Flex Projection is ingenious, supported by solid theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 9 datasets × zero-shot/full-shot + detailed ablation studies + efficiency comparisons + cross-resolution robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, intuitive case study, and well-organized theory and experiments.
- Value: ⭐⭐⭐⭐⭐ — Achieving SOTA with only 4M parameters is highly practical for resource-constrained deployments, and Periodical Tokenization is transferable to other TSFMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](../../AAAI2026/model_compression/xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/model_compression/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[ICCV 2025\] SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](../../ICCV2025/model_compression/frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [\[ICML 2025\] Joker: Joint Optimization Framework for Lightweight Kernel Machines](joker_joint_optimization_framework_for_lightweight_kernel_machines.md)
- [\[ICML 2025\] Neutral Residues: Revisiting Adapters for Model Extension](neutral_residues_revisiting_adapters_for_model_extension.md)

</div>

<!-- RELATED:END -->
