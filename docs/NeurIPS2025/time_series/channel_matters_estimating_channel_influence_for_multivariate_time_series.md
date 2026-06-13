---
title: >-
  [Paper Note] Channel Matters: Estimating Channel Influence for Multivariate Time Series
description: >-
  [NeurIPS 2025][Time Series][Channel influence function] This paper proposes Channel-wise Influence (ChInf)—the first influence function method capable of quantifying the effect of individual channels on model performance…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Channel influence function"
  - "multivariate time series"
  - "anomaly detection"
  - "data pruning"
  - "TracIn"
date: 2026-05-08
content_hash: 4e59020f5717ff27
---

# Channel Matters: Estimating Channel Influence for Multivariate Time Series

**Conference**: NeurIPS 2025
**arXiv**: [2408.14763](https://arxiv.org/abs/2408.14763)  
**Code**: [GitHub](https://github.com/flare200020/Chinf)  
**Area**: Time Series / Influence Functions / Data-Centric Methods
**Keywords**: Channel influence function, multivariate time series, anomaly detection, data pruning, TracIn

## TL;DR
This paper proposes Channel-wise Influence (ChInf)—the first influence function method capable of quantifying the effect of individual channels on model performance in multivariate time series (MTS). By decomposing TracIn from the holistic sample level to the channel level, ChInf enables two downstream applications: channel-level anomaly detection and channel pruning, achieving state-of-the-art performance on 5 anomaly detection benchmarks.

## Background & Motivation

**Background**: The importance of channels (variables) in MTS analysis is widely recognized. iTransformer models channel dependencies via attention, while PatchTST improves generalization through channel independence. However, these approaches are model-centric—they leverage channel information implicitly without explicitly quantifying each channel's contribution to model performance.

**Limitations of Prior Work**:
- Classical influence functions (Koh & Liang 2017) and TracIn are designed for entire data samples and cannot distinguish the contributions of individual channels in MTS.
- TimeInf accounts for temporal dependencies but ignores the channel dimension, yielding suboptimal results on MTS anomaly detection and data pruning tasks.
- No existing tool can answer: "Which channel is most important for model predictions?" or "Which channel is most anomalous?"

**Key Challenge**: Different channels in MTS carry heterogeneous information and exhibit complex correlations, yet existing influence functions cannot disentangle channel-level contributions—they treat all channels as a single sample when computing influence.

**Goal**: (1) Define a channel-level influence function; (2) derive anomaly detection and channel pruning methods based on it.

**Key Insight**: Decompose the gradient inner product in TracIn from the full sample into a sum of per-channel gradient inner products, naturally yielding a channel influence matrix $M_{CInf}$.

**Core Idea**: Decompose influence functions from the sample level to the channel level; use channel self-influence for anomaly detection and channel influence ranking for data pruning.

## Method

### Overall Architecture
ChInf is a post-hoc interpretability tool. Given a trained MTS model, ChInf computes an $N \times N$ channel influence matrix (where $N$ is the number of channels) for any training–test sample pair. Each entry $a_{i,j}$ quantifies the influence of training channel $i$ on the loss of test channel $j$.

### Key Designs

1. **Channel-wise Influence Function (ChInf)**:

    - **Function**: Decomposes TracIn to the channel level.
    - **Mechanism**: Classical TracIn is defined as $\text{TracIn}(z', z) = \eta \nabla_\theta L(z'; \theta)^\top \nabla_\theta L(z; \theta)$. For MTS, $z' = \{c_1', \ldots, c_N'\}$; exploiting the additivity of the loss over channels, one can show that $\text{TracIn}(z', z) = \sum_{i=1}^{N} \sum_{j=1}^{N} \eta \nabla_\theta L(c_i'; \theta)^\top \nabla_\theta L(c_j; \theta)$.
    - The channel influence matrix is defined as $M_{CInf} = [a_{i,j}]_{N \times N}$, where $a_{i,j} = \eta \nabla_\theta L(c_i'; \theta)^\top \nabla_\theta L(c_j; \theta)$.
    - **Design Motivation**: $a_{i,j}$ measures how much training on channel $i$ benefits the loss on channel $j$—channels with similar patterns tend to yield high influence scores.

2. **ChInf Anomaly Detection**:

    - **Function**: Uses channel self-influence (diagonal entries) as anomaly scores.
    - **Mechanism**: Anomaly score $= \max_i \; \eta \nabla_\theta L(c_i'; \theta)^\top \nabla_\theta L(c_i'; \theta)$, i.e., the maximum self-influence across channels. Anomalous samples exhibit higher self-influence due to their distributional divergence from normal training data.
    - **Design Motivation**: Sample-level influence functions cannot localize which specific channel is anomalous; ChInf's channel decomposition enables taking the per-channel maximum rather than an aggregate mean.

3. **ChInf Channel Pruning**:

    - **Function**: Identifies a representative subset of channels, enabling model training with fewer channels.
    - **Mechanism**: Per-channel self-influence (diagonal entries) is computed on a validation set, channels are ranked by influence, and a representative subset $\hat{D}$ is selected via uniform interval sampling.
    - **Design Motivation**: iTransformer demonstrates that a subset of channels can effectively predict all channels, indicating inter-channel redundancy; ChInf provides a data-centric approach to identify such redundancy.

## Key Experimental Results

### Main Results: Anomaly Detection

| Method | SWaT F1 | SMD F1 | SMAP F1 | MSL F1 | WADI F1 |
|--------|---------|--------|---------|--------|---------|
| PCA ERROR (Simple) | 83.3 | 57.2 | 39.2 | 42.6 | 50.1 |
| GCN-LSTM (Simple) | 82.9 | 55.0 | 42.6 | 46.3 | 43.9 |
| iTransformer | 83.7 | 55.9 | 39.6 | 45.5 | 48.8 |
| TimeInf | 79.0 | 54.1 | 35.1 | 39.7 | — |
| **GCN-LSTM + ChInf** | 82.9 | **58.8** | **48.0** | **47.1** | 47.2 |
| **iTransformer + ChInf** | **84.0** | **59.1** | **46.3** | 46.1 | **50.5** |

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| ChInf vs. TracIn | ChInf significantly outperforms TracIn across all datasets, validating the necessity of channel-level analysis. |
| ChInf vs. TimeInf | TimeInf considers only the temporal dimension and ignores channels, leading to inferior performance on MTS tasks. |
| Channel pruning ratio | 30–50% of channels suffice to retain 90%+ forecasting performance. |
| Different backbone models | As a plug-and-play tool, ChInf consistently improves performance across diverse backbone models. |

### Key Findings
- **ChInf ranks first**: Under a fair evaluation protocol (using standard F1 rather than point-adjusted F1), ChInf achieves the best overall ranking across all 5 datasets.
- **Channel-level analysis substantially outperforms sample-level**: TracIn and TimeInf underperform simple baselines on MTS anomaly detection, whereas ChInf's channel decomposition makes influence functions genuinely effective for MTS.
- **Channel pruning is highly effective**: 30–50% of channels are sufficient to maintain near-full-channel forecasting performance.
- **Channel influence matrix provides interpretability**: Visualizations reveal model dependency patterns across channels, with different models exhibiting distinct channel utilization strategies.

## Highlights & Insights
- **A natural and elegant extension from sample level to channel level**: ChInf's derivation is concise—it exploits the additivity of TracIn gradients to decompose influence to the channel level without requiring new approximations.
- **A plug-and-play post-hoc tool**: ChInf does not modify any model; it only requires a trained model and gradient information, making it compatible with any MTS model.
- **Complementarity of data-centric and model-centric perspectives**: Existing methods are model-centric (modifying architectures to leverage channel information), while ChInf is data-centric (quantifying the data value of each channel)—the two paradigms are mutually complementary.

## Limitations & Future Work
- **Computational cost**: Gradients must be computed independently for each channel; with $N$ channels this requires $N$-fold gradient computation, which may be slow for high-dimensional MTS.
- **Assumption of channel-decomposable loss**: The theoretical derivation relies on the additivity of the loss function over channels, which may not hold exactly when the loss involves complex inter-channel interactions.
- **Simple channel pruning sampling strategy**: Uniform interval sampling may not yield the optimal subset selection; more sophisticated subset selection strategies could further improve results.
- **Evaluation limited to anomaly detection and pruning**: The channel influence matrix should be useful for a broader range of tasks (e.g., channel selection, transfer learning) that remain unexplored.

## Related Work & Insights
- **vs. TracIn/IF**: Classical influence functions operate at the sample level; ChInf generalizes them to the channel level and substantially outperforms direct sample-level application in MTS settings.
- **vs. TimeInf**: TimeInf focuses on the temporal dimension while ChInf focuses on the channel dimension—the two are orthogonal and potentially composable.
- **vs. iTransformer**: iTransformer implicitly models channel dependencies, whereas ChInf explicitly quantifies channel influence—ChInf can in turn be used to analyze iTransformer's channel utilization patterns.

## Rating
- Novelty: ⭐⭐⭐⭐ Channel-level influence functions represent a natural yet effective new concept with an elegant derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 5 datasets, fair benchmarking protocol, multi-model validation, and channel pruning experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](../../ICLR2026/time_series/cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [\[ICLR 2026\] T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](../../ICLR2026/time_series/t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)
- [\[AAAI 2026\] C3RL: Rethinking the Combination of Channel-independence and Channel-mixing from Representation Learning](../../AAAI2026/time_series/c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](../../ICLR2026/time_series/routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)

</div>

<!-- RELATED:END -->
