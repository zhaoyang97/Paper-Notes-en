---
title: >-
  [Paper Note] xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories
description: >-
  [NeurIPS2025][Time Series][Time series forecasting] This paper proposes xLSTM-Mixer, the first architecture to combine the Extended Long Short-Term Memory network (sLSTM) with a Mixer framework. Through a three-stage design comprising temporal mixing, joint temporal-variate mixing, and multi-view mixing, the model achieves state-of-the-art performance on multivariate long-term time series forecasting while maintaining an extremely low memory footprint.
tags:
  - NeurIPS2025
  - Time Series
  - Time series forecasting
  - xLSTM
  - mixer architecture
  - recurrent neural networks
  - multivariate forecasting
date: 2026-05-08
content_hash: fb5ad4fc7a54c9f6
---

# xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories

**Conference**: NeurIPS2025
**arXiv**: [2410.16928](https://arxiv.org/abs/2410.16928)
**Authors**: Maurice Kraus, Felix Divo, Devendra Singh Dhami, Kristian Kersting (TU Darmstadt, TU Eindhoven)
**Code**: [mauricekraus/xlstm-mixer](https://github.com/mauricekraus/xlstm-mixer)
**Area**: Time Series
**Keywords**: Time series forecasting, xLSTM, mixer architecture, recurrent neural networks, multivariate forecasting

## TL;DR
This paper proposes xLSTM-Mixer, the first architecture to combine the Extended Long Short-Term Memory network (sLSTM) with a Mixer framework. Through a three-stage design comprising temporal mixing, joint temporal-variate mixing, and multi-view mixing, the model achieves state-of-the-art performance on multivariate long-term time series forecasting while maintaining an extremely low memory footprint.

## Background & Motivation
Time series forecasting is ubiquitous in critical domains such as healthcare, manufacturing, transportation, finance, and weather modeling. However, existing approaches exhibit notable limitations:

- **Transformer-based methods** (PatchTST, iTransformer): The attention mechanism scales quadratically with sequence length, leading to poor efficiency on long sequences and in resource-constrained settings.
- **Pure linear models** (DLinear, NLinear): Although efficient, they lack sufficient expressive power to capture complex nonlinear temporal dynamics.
- **SSM/Mamba-based methods** (S-Mamba, Chimera): These process sequence elements independently, making it difficult to directly learn inter-variate relationships.
- **Existing mixer architectures** (TimeMixer, TSMixer): They lack the memory capacity and long-range dependency modeling advantages of recurrent models.
- **xLSTMTime**: An initial attempt to apply xLSTM to time series, but it failed to surpass strong baselines such as TimeMixer and suffers from reproducibility issues.

Key observations: (1) The channel-independence assumption (e.g., PatchTST) provides regularization at the cost of cross-variate information; (2) joint mixing is more expressive but prone to overfitting; (3) the scalar memory and exponential gating mechanism of sLSTM are naturally suited for sequence mixing. These observations motivate an architecture that integrates the memory capacity of recurrent models with the efficiency of mixer architectures.

## Method

### Overall Architecture
xLSTM-Mixer consists of three stages: temporal mixing → joint mixing → view mixing.

The input is $\bm{X} \in \mathbb{R}^{V \times T}$ ($V$ variates, $T$ time steps), and the model forecasts $\bm{Y} \in \mathbb{R}^{V \times H}$ ($H$ future steps).

### Stage 1: Normalization and Initial Linear Forecast (Temporal Mixing)

1. **RevIN normalization**: Reversible instance normalization is applied to each time series, with learnable scale $\bm{\gamma}$ and shift $\bm{\beta}$:
$$\bm{x}_t^{\text{norm}} = \bm{\gamma} \odot \frac{\bm{x}_t - \mathbb{E}[\bm{x}]}{\sqrt{\text{Var}[\bm{x}]} + \epsilon} + \bm{\beta}$$

2. **NLinear initial forecast**: A shared linear layer is applied independently to each variate, mapping from $T$ steps to $H$ steps:
$$\bm{x}^{\text{initial}} = \text{FC}(\bm{x}_{1:T}^{\text{norm}} - x_T^{\text{norm}}) + x_T^{\text{norm}}$$
   Weights are shared across all variates, yielding fewer parameters and a regularization effect. The linear forecast itself already serves as a strong baseline.

### Stage 2: sLSTM Refinement (Joint Temporal-Variate Mixing)

1. **Up-projection**: The initial forecast $\bm{x}^{\text{initial}} \in \mathbb{R}^{V \times H}$ is projected to a higher hidden dimension $D$: $\bm{x}^{\text{up}} = \text{FC}^{\text{up}}(\bm{x}^{\text{initial}}) \in \mathbb{R}^{V \times D}$, with weights again shared across variates.

2. **Stacked sLSTM blocks**: $M$ layers of sLSTM blocks process the sequence recurrently along the **variate dimension** (rather than the time dimension). Each token represents the temporal embedding of a single variate. Key properties:
    - **Exponential gating**: Input and forget gates employ exponential functions ($\bm{i}_t = \exp(\tilde{\bm{i}}_t - \bm{m}_t)$), enhancing memory control.
    - **Multi-head memory mixing**: The recurrent weight matrix $\bm{R}$ adopts a block-diagonal structure, enabling head specialization.
    - **Numerical stabilization**: $\bm{m}_t = \max(\tilde{\bm{f}}_t + \bm{m}_{t-1}, \tilde{\bm{i}}_t)$ prevents exponential overflow.

3. **Learnable initial embedding $\bm{\eta}$**: Inspired by soft prompts in large language models, a learnable initial token is prepended to the sequence, allowing the sLSTM's initial hidden state to adapt to dataset-specific characteristics.

**Why sLSTM instead of mLSTM**: mLSTM processes sequence elements independently and cannot directly learn inter-variate relationships, whereas sLSTM enables element interactions through its recurrent weight matrix.

**Why recurrence along the variate dimension**: The parameter count does not grow with the number of variates (linear time scaling), and empirical results confirm that this orientation outperforms recurrence along the time dimension.

### Stage 3: Multi-View Mixing

1. The original embedding $\bm{x}^{\text{up}}$ and its **reversed** counterpart $\hat{\bm{x}}^{\text{up}}$ are each passed through a weight-shared sLSTM stack to produce two sets of predictions $\bm{y}', \bm{y}''$.
2. A linear projection fuses the two views: $\bm{y}^{\text{norm}} = \text{FC}^{\text{view}}(\bm{y}', \bm{y}'')$.
3. The final forecast is obtained by inverting RevIN: $\bm{y} = \text{RevIN}^{-1}(\bm{y}^{\text{norm}})$.

Multi-view mixing can be interpreted as an ensemble over different variate orderings (weight-sharing ensembling), providing additional regularization.

## Key Experimental Results

### Table 1: Long-Term Forecasting Main Results (7 datasets, averaged over 4 forecast horizons {96, 192, 336, 720})

| Model | Weather MSE | Electricity MSE | Traffic MSE | ETTh1 MSE | ETTh2 MSE | ETTm1 MSE | ETTm2 MSE | MSE Wins |
|---|---|---|---|---|---|---|---|---|
| **xLSTM-Mixer** | **0.219** | **0.153** | 0.392 | 0.397 | 0.340 | **0.339** | **0.248** | **11** |
| Chimera | 0.219 | 0.154 | 0.403 | 0.405 | **0.318** | 0.345 | 0.250 | 8 |
| TimeMixer | 0.222 | 0.156 | **0.387** | 0.411 | 0.316 | 0.348 | 0.256 | 2 |
| CycleNet | 0.223 | 0.156 | 0.403 | 0.435 | 0.367 | 0.360 | 0.263 | 1 |
| PatchTST | 0.241 | 0.159 | 0.391 | 0.413 | 0.324 | 0.353 | 0.256 | 1 |
| TimeMixer++ | 0.226 | 0.165 | 0.416 | 0.419 | 0.339 | 0.369 | 0.269 | 0 |
| DLinear | 0.246 | 0.166 | 0.434 | 0.423 | 0.431 | 0.357 | 0.267 | 0 |

**Key Findings**:
- xLSTM-Mixer achieves **11 best MSE wins + 16 best MAE wins** across 28 settings.
- It establishes new state-of-the-art results on **6 out of 7** datasets.
- Statistical significance tests (Friedman + Conover post-hoc, $p=0.05$): xLSTM-Mixer significantly outperforms all competing methods except xLSTMTime; however, xLSTM-Mixer's average rank of 1.5 substantially surpasses xLSTMTime's rank of 4.0.

### Table 2: GIFT-Eval Probabilistic Forecasting Benchmark (Top 10)

| Model | MASE ↓ | CRPS ↓ | Rank |
|---|---|---|---|
| TiRex | 0.724 | 0.498 | 1 |
| **xLSTM-Mixer** | **0.780** | **0.510** | **2** |
| TEMPO_ensemble | 0.862 | 0.514 | 3 |
| Toto_Open_Base | 0.750 | 0.517 | 4 |
| TabPFN-TS | 0.771 | 0.544 | 5 |
| timesfm_2_0_500m | 0.758 | 0.550 | 7 |

**Key Findings**:
- Ranked 2nd by CRPS (behind TiRex only), and **1st among purely supervised models**.
- Demonstrates robust performance on a heterogeneous benchmark covering univariate/multivariate and short/long forecasting settings.
- Probabilistic forecasting is enabled by appending a quantile prediction head with no architectural modifications.

### Ablation Study (Table 3 Summary, Weather + ETTm1)

| Ablated Component | MSE Increase | MAE Increase |
|---|---|---|
| sLSTM → LSTM (#3) | +7.0% | +6.2% |
| Variate-wise recurrence → Time-wise recurrence (#5) | +4.7% | +4.3% |
| Remove temporal mixing (#11) | +3.1% | +2.7% |
| Remove initial embedding η (#7) | +0.7% | +0.4% |
| Remove view mixing (#8) | +0.7% | +0.6% |

The sLSTM block and temporal mixing are the most critical components. All components contribute positively, and the full configuration achieves the best performance.

### Efficiency Analysis
- xLSTM-Mixer requires **1–2 orders of magnitude less memory** than TimeMixer.
- Both runtime and memory remain nearly constant as the lookback window $T$ grows, in contrast to the quadratic scaling of Transformer-based methods.
- The model can efficiently exploit longer lookback windows ($T$ from 96 to 1440) with consistently improving performance.

## Highlights & Insights
- **First integration of recurrent and mixer architectures**: Combining the expressive power and memory mixing capability of sLSTM with the efficient structure of Mixer opens a new paradigm for time series forecasting.
- **Extremely low memory footprint**: 1–2 orders of magnitude lower than TimeMixer and substantially lower than Transformer-based approaches, making the model suitable for edge device deployment.
- **Multi-view mixing innovation**: The dual-view ensemble of original and reversed embeddings provides an elegant solution to the variate-ordering sensitivity problem, and weight sharing introduces no additional parameters.
- **Rigorous statistical validation**: Friedman + Conover post-hoc tests confirm statistical significance rather than relying solely on win counts.
- **Strong generality**: The model performs well across long-term point forecasting, GIFT-Eval probabilistic forecasting, and classification tasks.

## Limitations & Future Work
- **Uniform sampling assumption**: All variates are required to be sampled on a regular time grid; irregular or missing timestamps require preprocessing.
- **High-variate bottleneck**: Variate-wise recurrence results in runtime that scales linearly with the number of variates, which may become a bottleneck for extremely high-dimensional settings (e.g., thousands of variates).
- **Variate ordering sensitivity**: Although experiments show that standard ordering is already sufficient, finding the optimal ordering remains an open problem.
- **Limited interpretability**: Multi-view mixing fuses temporal and cross-variate information in a way that makes fine-grained attribution analysis difficult.
- **Point and quantile forecasting only**: Probabilistic forecasting is realized via a quantile head; richer distributional modeling has not been explored.

## Related Work & Insights
- **Recurrent models**: LSTM/GRU → xLSTM (exponential gating + memory mixing) → xLSTMTime (first application to time series, limited effectiveness) → **xLSTM-Mixer achieves substantial improvement via the mixer architecture**.
- **Mixer architectures**: MLP-Mixer → TSMixer/TimeMixer/TimeMixer++ (alternating temporal and variate-dimension mixing) → xLSTM-Mixer replaces MLP with sLSTM for joint mixing.
- **Transformer-based methods**: Autoformer → PatchTST → iTransformer; high accuracy but resource-intensive.
- **SSM-based methods**: S-Mamba, Chimera; support parallel inference but with limited mixing capability.
- **Pre-trained foundation models**: Chronos, Moirai, Timer-XL; require large-scale pretraining data.
- **Positioning of xLSTM-Mixer**: Fills the intersection of recurrent models and mixer architectures, achieving accuracy surpassing Transformer-based methods at a drastically reduced memory cost.

## Rating
- Novelty: ⭐⭐⭐⭐ — First combination of sLSTM with the Mixer architecture; variate-wise recurrence and multi-view mixing represent meaningful methodological contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Seven benchmarks, GIFT-Eval, classification tasks, 13 ablation groups, statistical significance tests, and efficiency analysis; exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated design choices, rigorous ablation analysis; notation is somewhat heavy.
- Value: ⭐⭐⭐⭐ — Provides a practical solution for high-accuracy time series forecasting in resource-constrained settings, and contributes to the revival of recurrent models in the time series domain.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[ICLR 2026\] Free Energy Mixer](../../ICLR2026/time_series/free_energy_mixer.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](channel_matters_estimating_channel_influence_for_multivariate_time_series.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)

<!-- RELATED:END -->
