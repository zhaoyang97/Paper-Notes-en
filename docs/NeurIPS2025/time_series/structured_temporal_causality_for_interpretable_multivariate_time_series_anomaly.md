---
title: >-
  [Paper Note] Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection
description: >-
  [NeurIPS 2025][Time Series][Multivariate time series] This paper proposes OracleAD, a framework that learns causal embeddings for each variable (via LSTM encoding and attention pooling) and constructs a Stable Latent Str…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Multivariate time series"
  - "anomaly detection"
  - "temporal causal modeling"
  - "stable latent structure"
  - "interpretability"
  - "LSTM"
  - "self-attention"
date: 2026-05-08
content_hash: a157d04ce306ad66
---

# Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection

**Conference**: NeurIPS 2025
**arXiv**: [2510.16511](https://arxiv.org/abs/2510.16511)  
**Authors**: Dongchan Cho, Jiho Han, Keumyeong Kang, Minsang Kim, Honggyu Ryu, Namsoon Jung (SimPlatform Co. Ltd.)
**Code**: Not released  
**Area**: Time Series
**Keywords**: Multivariate time series, anomaly detection, temporal causal modeling, stable latent structure, interpretability, LSTM, self-attention

## TL;DR

This paper proposes OracleAD, a framework that learns causal embeddings for each variable (via LSTM encoding and attention pooling) and constructs a Stable Latent Structure (SLS) to model inter-variable relationships under normal conditions. A dual scoring mechanism combining prediction error and SLS deviation enables interpretable multivariate time series anomaly detection and root cause localization.

## Background & Motivation

### State of the Field
Multivariate time series anomaly detection (MTSAD) is a core task in industrial control, medical monitoring, and cybersecurity. Anomalies are typically rare, unlabeled, and context-dependent, requiring models not only to detect anomalies but also to explain their causal origins.

### Limitations of Prior Work
- **Reconstruction-based methods** (AutoEncoder, OmniAnomaly): process each channel independently, ignoring inter-variable dependencies.
- **Transformer-based methods** (Anomaly Transformer, DCdetector): employ large receptive fields and bidirectional attention, violating the unidirectional irreversibility of time, with high computational cost.
- **Graph neural network methods** (GDN): learn static adjacency matrices, yielding fixed inter-variable relationships at inference time.
- **Frequency-domain/contrastive learning methods** (CATCH, DCdetector): decouple from causal temporal dynamics and artificially separate normal from abnormal data.
- **Evaluation**: common benchmarks (SWaT, SMAP, MSL) contain anomalies that affect only a small number of variables, causing metrics such as Point-adjusted F1 to severely overestimate performance.

### Root Cause
The authors argue that anomalies in multivariate time series fundamentally manifest through two signals: (1) **temporal causal disruption**—the current state of a variable deviates from expectations derived from its history; and (2) **structural deviation**—temporal perturbations propagate and disrupt inter-variable relationships that remain stable under normal conditions. Existing methods do not explicitly model the joint mechanism of these two signals.

## Method

### Overall Architecture
OracleAD operates on sliding windows. For a window of length $L$:
1. A **per-variable LSTM encoder** extracts temporal causal embeddings.
2. **Multi-head self-attention** captures dynamic inter-variable relationships.
3. An **LSTM decoder** performs joint reconstruction and prediction.
4. A **Stable Latent Structure (SLS)** serves as a reference baseline for normal relationships.
5. A **dual scoring mechanism** fuses prediction scores and deviation scores.

### Temporal Causal Modeling
For the historical sequence $\mathbf{x}_i = (x_i^1, \ldots, x_i^{L-1})$ of each variable $i$:
- The LSTM encoder produces a hidden state sequence $\{h_i^1, \ldots, h_i^{L-1}\}$, $h_i^l \in \mathbb{R}^d$.
- Learnable attention pooling aggregates hidden states into a single causal embedding:

$$c_i = \sum_{l=1}^{L-1} \alpha_i^l \, h_i^l, \quad \alpha_i^l = \mathrm{softmax}(w^\top h_i^l + b)$$

- $c_i$ encodes all temporal causal information required to predict $x_i^L$.

**Design Motivation**: Each variable is modeled independently to avoid entangling unrelated temporal patterns via a shared architecture; attention pooling suppresses noise while preserving key temporal information.

### Inter-Variable Relationship Modeling
All causal embeddings are stacked as $C = [c_1, \ldots, c_N]^\top \in \mathbb{R}^{N \times d}$, and multi-head self-attention (MHSA) yields context-aware embeddings $C^* = [c_1^*, \ldots, c_N^*]$. Each $c_i^*$ absorbs contextual information from all other variables, capturing soft dynamic dependencies without requiring a predefined static graph.

### Stable Latent Structure (SLS)
**Training phase**:
- For each time window $k$, the pairwise L2 distance matrix of attention-refined embeddings is computed: $D_{ij}^{(k)} = \|c_i^{*(k)} - c_j^{*(k)}\|_2$.
- At the end of each epoch, these matrices are aggregated into the SLS: $\mathbf{SLS} = \frac{1}{M}\sum_{k=1}^M D^{(k)}$.

**Inference phase**:
- The deviation matrix is computed as $\mathcal{D}_\text{matrix}^t = |D^t - \mathbf{SLS}|$.
- Rows and columns with high values in the deviation matrix indicate root cause variables.

### Loss & Training
The composite loss consists of three terms:

$$\mathcal{L} = \underbrace{\|\mathbf{x}^L - \hat{\mathbf{x}}^L\|^2}_{\text{prediction loss}} + \lambda_\text{recon} \cdot \underbrace{\|\mathbf{x}^{1:L-1} - \hat{\mathbf{x}}^{1:L-1}\|^2}_{\text{reconstruction loss}} + \lambda_\text{dev} \cdot \underbrace{\frac{1}{N^2}\sum_{i,j}(D_{ij} - \mathbf{SLS}_{ij})^2}_{\text{deviation loss}}$$

Default hyperparameters: $\lambda_\text{recon}=0.1$, $\lambda_\text{dev}=3$. No SLS is available during the first epoch; the deviation loss is incorporated starting from the second epoch.

### Anomaly Scoring
Two complementary scores are computed at inference time:
- **Prediction score**: $\mathcal{P}_\text{score}^t = \frac{1}{N}\sum_{i=1}^N |x_i^t - \hat{x}_i^t|$
- **Deviation score**: $\mathcal{D}_\text{score}^t = \|D^t - \mathbf{SLS}\|_F$ (Frobenius norm)
- **Final anomaly score**: $\mathcal{A}_\text{score}^t = \mathcal{P}_\text{score}^t \cdot \mathcal{D}_\text{score}^t$ (multiplicative fusion)

The prediction score is sensitive to abrupt changes but produces short-lived responses; the deviation score captures persistent relational perturbations but exhibits latency. Multiplicative combination balances both: low prediction error suppresses false positives from the deviation score, while sustained deviation compensates for false negatives from the prediction score.

## Key Experimental Results

### Main Results: Multi-Dataset Multi-Metric Comparison

OracleAD is compared against 12 baselines on three benchmark datasets—SMD (38 variables), PSM (25 variables), and SWaT (51 variables)—across 7 evaluation metrics.

| Dataset | Metric | AutoEncoder | OmniAnomaly | A.Transformer | SARAD | CATCH | **OracleAD** |
|--------|------|-------------|-------------|---------------|-------|-------|-------------|
| PSM | F1 | 47.55 | 45.90 | 43.45 | 45.75 | 44.33 | **65.85** |
| PSM | V-PR | 49.66 | 52.49 | 49.76 | 38.64 | 45.95 | **68.17** |
| PSM | A-ROC | 66.79 | 63.95 | 38.35 | 62.86 | 64.75 | **84.78** |
| SMD | F1 | 25.78 | 32.16 | 7.98 | 25.92 | 7.98 | **43.03** |
| SMD | V-PR | 22.50 | 31.18 | 36.86 | 19.33 | 35.25 | **47.52** |
| SMD | A-PR | 19.40 | 27.73 | 4.57 | 25.87 | 17.09 | **44.83** |
| SWaT | F1 | 74.46 | 75.40 | 21.65 | 57.30 | 21.65 | **76.50** |
| SWaT | V-PR | 65.89 | 64.42 | 17.00 | 62.72 | 18.70 | **74.16** |
| SWaT | A-PR | 67.51 | 72.73 | 11.93 | 64.77 | 13.39 | **72.39** |

OracleAD F1 gains: PSM +19.95 pp, SMD +10.87 pp, SWaT +0.9 pp. VUS-PR gains: PSM +15.68 pp, SMD +5.92 pp, SWaT +8.27 pp.

### Ablation Study

| Component | Variant | PSM F1 | PSM V-PR | SMD F1 | SMD V-PR | SWaT F1 | SWaT V-PR |
|------|------|--------|----------|--------|----------|---------|-----------|
| Loss function | w/o reconstruction loss | 58.03 | 54.40 | 56.47 | 54.29 | 76.61 | 71.95 |
| Scoring strategy | Deviation score only | 59.06 | 56.11 | 47.32 | 37.02 | 76.92 | 70.77 |
| Scoring strategy | Prediction score only | 55.33 | 60.99 | 58.98 | 53.71 | 70.49 | 68.50 |
| **Full model** | **OracleAD** | **65.85** | **68.17** | **60.19** | **56.63** | **76.50** | **74.16** |

**Key Findings**:
- Removing the reconstruction loss leads to a drop of 7.82 pp in F1 and 13.77 pp in V-PR on PSM.
- Using only the deviation score performs adequately on SWaT (76.92) but collapses on SMD with V-PR of only 37.02.
- Using only the prediction score reduces F1 by 6.01 pp on SWaT, where anomalies typically affect few variables.
- The dual scoring with multiplicative fusion achieves the best overall performance across all datasets, validating the complementarity of the temporal and structural dimensions.

## Highlights & Insights

- **Explicit anomaly definition**: Multivariate time series anomalies are defined as a two-stage process of "temporal causal disruption → structural deviation," providing stronger causal interpretability than reconstruction error or attention discrepancy.
- **SLS mechanism**: A data-driven reference structure for inter-variable relationships under normal conditions is constructed, serving both as a training regularizer and as an inference-time detection baseline; the deviation matrix directly localizes root cause variables.
- **Minimal yet effective**: The lightweight combination of LSTM, self-attention, and L2 distance with a window length of only $L=10$ substantially outperforms complex Transformer and frequency-domain methods.
- **Comprehensive evaluation**: Seven metrics (including newer metrics such as VUS-PR) are employed, with in-depth analysis of the limitations of metrics such as Affiliation F1.
- **Interpretability**: Visualization of the deviation matrix directly reveals which inter-variable relationships undergo structural change during anomalous periods, providing practical root cause diagnosis capability.

## Limitations & Future Work

- **Global relationship stationarity assumption**: SLS assumes globally stable inter-variable relationships under normal conditions, which may not hold for complex systems with multimodal distributions or regime switching.
- **Continuous input assumption**: The method assumes continuous input and does not address missing values, asynchronous sampling, or related practical issues.
- **Scalability of per-variable modeling**: Each variable has an independent LSTM encoder and decoder, causing parameter count to grow linearly with the number of variables.
- **Single SLS update strategy**: The SLS is aggregated as a mean over all windows at the end of each epoch, without considering temporal decay or online updates.
- **L2 distance only**: Although ablation experiments show L2 outperforms cosine and L1, richer relational metrics are not explored.
- **Fixed window length**: $L=10$ is applied uniformly across all datasets without adaptive adjustment based on anomaly patterns.

## Related Work & Insights

- **Anomaly Transformer**: Detects anomalies via the discrepancy between attention weights and prior temporal associations, but relies on large windows and bidirectional attention, performing poorly on multiple metrics (PSM F1: 43.45 vs. OracleAD: 65.85).
- **OmniAnomaly**: A stochastic RNN-based reconstruction method that processes channels independently; F1 on SMD is 32.16, far below OracleAD's 43.03.
- **SARAD**: Spatial association regularization between adjacent subsequences; achieves higher A-ROC (85.40) and V-ROC (86.30) on SWaT than OracleAD, but F1 is only 57.30.
- **CATCH**: A channel-aware method using frequency-domain patching; leads on PSM Aff-F1 (79.16 vs. 78.07) but trails substantially on F1 and VUS metrics.
- **GDN**: Learns a data-driven graph that is static at inference time; OracleAD's SLS provides dynamic relational comparison.
- **DLinear/NLinear**: Simple linear prediction baselines that confirm more complex architectures do not always yield improvements.

## Rating

- Novelty: ⭐⭐⭐⭐ — The SLS concept and the causal embedding-based dual scoring mechanism are novel contributions, though the core components (LSTM + attention) are standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 12 baselines, 7 metrics, 3 datasets, with comprehensive ablation studies and visualization analysis; validation on larger scales and more domains is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, anomaly definition is rigorous, method derivation is complete, and evaluation discussion is thorough.
- Value: ⭐⭐⭐⭐ — Introduces a new paradigm for MTSAD that combines simplicity and interpretability, with experimental results significantly outperforming mainstream methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](channel_matters_estimating_channel_influence_for_multivariate_time_series.md)
- [\[ICLR 2026\] Contextual and Seasonal LSTMs for Time Series Anomaly Detection](../../ICLR2026/time_series/contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)
- [\[NeurIPS 2025\] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)
- [\[NeurIPS 2025\] Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data](exploring_neural_granger_causality_with_xlstms_unveiling_temporal_dependencies_i.md)

</div>

<!-- RELATED:END -->
