---
title: >-
  [Paper Note] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting
description: >-
  [ICML 2025][Time Series][Multivariate Time Series Forecasting] This paper proposes the Temporal Query (TQ) technique, which utilizes periodically shifted learnable vectors as queries in the attention mechanism to capture global variable-to-variable correlation patterns, while keys/values are derived from the raw data to preserve sample-level local information. Built upon this, TQNet uses only a single-layer multi-head attention and a shallow MLP to achieve overall state-of-th…
tags:
  - "ICML 2025"
  - "Time Series"
  - "Multivariate Time Series Forecasting"
  - "Temporal Query"
  - "Variable Correlation"
  - "Attention Mechanism"
  - "Periodic Parameters"
  - "Lightweight Models"
date: 2026-05-08
content_hash: 674158a26daea417
---

# TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting

**Conference**: ICML 2025  
**arXiv**: [2505.12917](https://arxiv.org/abs/2505.12917)  
**Code**: [GitHub - TQNet](https://github.com/ACAT-SCUT/TQNet)  
**Area**: Time Series Forecasting  
**Keywords**: Multivariate Time Series Forecasting, Temporal Query, Variable Correlation, Attention Mechanism, Periodic Parameters, Lightweight Models

## TL;DR

This paper proposes the Temporal Query (TQ) technique, which utilizes periodically shifted learnable vectors as queries in the attention mechanism to capture global variable-to-variable correlation patterns, while keys/values are derived from the raw data to preserve sample-level local information. Built upon this, TQNet uses only a single-layer multi-head attention and a shallow MLP to achieve overall state-of-the-art (SOTA) performance across 12 real-world datasets, with computation efficiency approaching the linear model DLinear.

## Background & Motivation

**Background**: The core challenge of multivariate time series forecasting (MTSF) lies in accurately modeling the correlations among variables. Methodological design has evolved through three stages: Channel Mixing (CM) — mixing variables without distinguishing their specific relations (e.g., Informer, Autoformer); Channel Independence (CI) — modeling each variable independently to ignore correlations but enjoy greater robustness (e.g., PatchTST, DLinear); and Channel Dependence (CD) — explicitly modeling variable dependencies but requiring effective mechanisms (e.g., iTransformer, Crossformer).

**Limitations of Prior Work**: Non-stationary noise (such as extreme values, missing data, and noise) causes a significant gap between the inter-variable correlations observed in a single sample and the true correlation patterns on the global training set. Standard self-attention generates Q, K, and V simultaneously from sample data, making it highly susceptible to noise perturbations and leading to unstable learned correlation patterns.

**Key Challenge**: There is a need for a mechanism that can learn globally stable, sample-independent inter-variable correlation patterns while retaining the local specificity of each individual sample. Relying solely on global (learnable parameter) designs loses sample characteristics, whereas purely local (raw data) designs suffer from noise interference.

**Goal**: To design a technique that naturally merges global and local correlations within the attention mechanism, and to construct an extremely simple and efficient forecasting model based on it.

**Key Insight**: Inspired by CycleNet's use of learnable periodic parameters, this work uses learnable vectors as the query in the attention mechanism (encoding global patterns) and uses raw data for keys/values (encoding local features). The global-local fusion is realized through the Q-K interaction of the attention mechanism.

**Core Idea**: By replacing the standard data-generated queries with periodically shifted learnable vectors, the attention score calculation $\frac{QK^\top}{\sqrt{L}}$ naturally integrates global priors (Q from learnable parameters) with sample features (K from raw data). This enables a single layer to capture robust dependencies among variables.

## Method

### Overall Architecture

TQNet consists of three components: (1) a TQ-enhanced single-layer multi-head attention (TQ-MHA) to capture variable correlations; (2) a shallow MLP to model temporal dependencies; and (3) a linear layer with dropout to project to the forecasting target. Input $X_t \in \mathbb{R}^{C \times L}$ (where $C$ is the number of variables, and $L$ is the lookback window length) is processed by TQ-MHA, sent to the MLP via a residual connection, and finally projected to output $\bar{Y}_t \in \mathbb{R}^{C \times H}$ (where $H$ is the forecast horizon). An optional Instance Normalization is adopted to handle distribution shifts.

### Key Designs

1. **Temporal Query (TQ) Technique**
    - **Function**: Encodes global variable correlation patterns using learnable parameters, replacing queries generated from noisy data.
    - **Mechanism**: A learnable parameter matrix $\theta_{TQ} \in \mathbb{R}^{C \times W}$ (where $W$ is the period length of the data) is initialized. For time step $t$, a segment $\theta_{TQ}^{t,L}$ of length $L$ is periodically sliced from $\theta_{TQ}$ based on $t \mod W$ and serves as the query. The periodicity ensures that $\theta_{TQ}^{t,L} = \theta_{TQ}^{(t+i \cdot W),L}, i \in \mathbb{N}$, meaning samples separated by intervals of $W$ steps share the identical TQ vector.
    - **Design Motivation**: Learnable query vectors automatically capture optimal representation of variable relationships during training, periodic shifting enables parameter reuse, and gradient averaging over multiple samples eliminates local noise interference.

2. **TQ-Enhanced Multi-Head Attention (TQ-MHA)**
    - **Function**: An attention mechanism that fuses global and local correlations.
    - **Mechanism**: $Q_h = \theta_{TQ}^{t,L} W_h^Q$ (originated from the TQ vector), $K_h = X_t W_h^K$, $V_h = X_t W_h^V$ (originated from raw data). The attention is computed as $\text{Head}_h = \text{Softmax}(\frac{Q_h K_h^\top}{\sqrt{L}}) V_h$. Since Q encodes global patterns while K and V encode local information, the dot product of Q and K naturally fuses both levels of abstraction.
    - **Design Motivation**: Compared to standard self-attention (where Q, K, V are all data-driven and noise-sensitive) and pure global designs (where Q and K both rely on learnable parameters, losing sample specificity), TQ-MHA reaches the best trade-off. Ablation studies confirm that the (Q=TQ, K=Raw) configuration significantly outperforms the other two settings.

3. **Alignment of Period Length Hyperparameter $W$**
    - **Function**: Controls the periodic shifting interval of the TQ vectors, which needs to be aligned with the inherent periodicity of the data.
    - **Mechanism**: $W$ should be set to the maximum stable period length of the data (e.g., 168 for hourly data to represent one week, 96 for 15-minute data to represent one day). It can be determined via domain knowledge or Auto-Correlation Function (ACF). Alignment with integer multiples of the true period is still effective (though it reduces training samples per parameter), whereas non-alignment introduces semantic inconsistency and performance degradation.
    - **Design Motivation**: The periodicity of time-series data is strong prior knowledge. Aligning TQ with this periodicity allows learnable parameters to represent the global pattern at specific phase steps within a cycle.

### Loss & Training

The model is trained using L2 loss (MSE), evaluated with MSE and MAE metrics. The optimizer is not specified in detail. Instance Normalization adopts the simple scheme from iTransformer: subtracting the mean and dividing by the standard deviation for inputs, followed by an inverse transformation on outputs. Dropout is optionally applied before the output projection layer.

## Key Experimental Results

### Overall Comparison on 12 Datasets (Average of 4 Prediction Horizons)

| Method | ETTh1 MSE | Electricity MSE | Traffic MSE | PEMS03 MSE | Top2 Count / 24 |
|------|----------|----------------|------------|-----------|-----------|
| **TQNet** | **0.441** | **0.164** | 0.445 | **0.097** | **22** |
| TimeXer | 0.437 | 0.171 | 0.466 | 0.112 | 11 |
| CycleNet | 0.457 | 0.168 | 0.472 | 0.118 | 9 |
| iTransformer | 0.454 | 0.178 | 0.428 | 0.113 | 4 |
| PatchTST | 0.469 | 0.205 | 0.481 | - | 0 |
| DLinear | 0.456 | 0.212 | 0.625 | - | 0 |

### Ablation Study: Impact of Query-Key Configuration

| Configuration | Electricity MSE | PEMS03 MSE | PEMS04 MSE | PEMS07 MSE |
|------|----------------|-----------|-----------|-----------|
| Q=Raw, K=Raw (Standard self-attention) | 0.175 | 0.114 | 0.112 | 0.094 |
| **Q=TQ, K=Raw (TQNet default)** | **0.164** | **0.097** | **0.091** | **0.075** |
| Q=TQ, K=TQ (Purely global) | 0.179 | 0.111 | 0.113 | 0.092 |

### Component Ablation (Average on Electricity Dataset)

| Configuration | MSE | MAE | Description |
|------|-----|-----|------|
| TQNet Full | **0.164** | **0.259** | MLP + TQ & MHA |
| W/o MHA | 0.169 | 0.262 | MLP + Channel Identifier only |
| W/o TQ | 0.175 | 0.267 | Standard self-attention + MLP |
| Pure MLP | 0.190 | 0.276 | Without channel correlation modeling |

### Cross-Architecture Transferability of TQ (Electricity Dataset)

| Base Model | Original MSE | +TQ MSE | Gain |
|---------|--------|---------|------|
| iTransformer | 0.175 | **0.163** | -6.9% |
| PatchTST | 0.191 | **0.171** | -10.5% |
| DLinear | 0.210 | **0.182** | -13.3% |

### Key Findings

- TQNet ranks in the Top 2 for 22 out of 24 metrics, achieving overall SOTA by an overwhelming margin.
- The configuration of Q=TQ, K=Raw significantly outperforms standard self-attention and purely global attention on high-dimensional datasets.
- TQ serves as the most critical component for performance advancement — removing TQ causes a larger performance drop (MSE +0.011) than removing MHA (+0.005).
- The TQ technique can be seamlessly transferred to other models (iTransformer, PatchTST, DLinear) with zero modification, bringing considerable improvements across the board.
- t-SNE visualizations demonstrate that the representations learned by TQ align closely with the real channel patterns: similar channels cluster together in the representation space.
- The hyperparameter $W$ yields optimal results when aligned with the data periodicity, and remains highly competitive when set to integer multiples of the period.
- Efficiency is close to DLinear: on the 862-channel Traffic dataset, the training time is comparable to DLinear and substantially faster than iTransformer.

## Highlights & Insights

- **The design of "using learnable queries to encode global patterns" is extremely simple and elegant**: It simplifies the complex global-local fusion problem into a simple choice of where Q comes from. A single attention layer is sufficient to realize optimal fusion.
- **The intuition of periodic shifting is clear**: Since time-series data naturally possesses periodicity, the periodic shifting of TQ vectors aligns seamlessly with the data's cyclic patterns, successfully balancing parameter reuse with semantic consistency.
- **Victory of a minimalist architecture over complex models**: A single-layer attention combined with a shallow MLP defeats complex multi-layer Transformer architectures. This once again demonstrates that in time-series forecasting, architectural design is more vital than raw scale.
- **Cross-architecture transferability**: As an independent technical component, TQ can enhance various existing models, exhibiting immense practical utility.

## Limitations & Future Work

- The setting of the hyperparameter $W$ relies heavily on the periodicity of the data, which may not scale well to data without clear cycles or with multiple overlapping periods.
- When correlation among variables is highly weak, forcing the model to capture multivariate dependencies might introduce unnecessary complexity.
- As the lookback window $L$ increases, the marginal utility of multivariate modeling decreases, as longer historical paths can partially compensate for cross-variable clues.
- The quadratic complexity $O(C^2 L)$ of the attention mechanism might become a performance bottleneck when the number of variables $C > 1000$.
- The model is only trained with L2 loss; richer forecasting paradigms, such as probabilistic forecasting or quantile loss, are left unexplored.

## Related Work & Insights

- **iTransformer (Liu et al., 2024)**: Flips the attention dimension to treat variables as tokens. It serves as the most direct baseline for TQNet. TQNet replaces its queries with TQ, attaining superior performance.
- **CycleNet (Lin et al., 2024)**: The inspiration for using learnable parameters to capture periodic patterns. The periodic design of TQ directly inherits this concept.
- **TimeXer (Wang et al., 2024)**: A cross-attention-based method incorporating exogenous variables. It is outperformed by TQNet but belongs to the same CD (Channel Dependence) method family.
- **DLinear (Zeng et al., 2023)**: The benchmark for efficiency among CI (Channel Independence) methods. TQNet surpasses its accuracy with comparable computational efficiency.
- **Insight**: The source of Q/K/V in attention mechanisms represents an underestimated design space. TQ demonstrates that merely altering the origin of Q can yield substantial architectural improvements.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The global-local fusion strategy of TQ is elegant and neat; the periodic shift design is clever)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Evaluated on 12 datasets, accompanied by comprehensive ablation studies, cross-architecture transfers, representation visualizations, and efficiency analyses)
- **Writing Quality**: ⭐⭐⭐⭐ (Well-structured, highly illustrative figures, and robust ablation logics)
- **Value**: ⭐⭐⭐⭐⭐ (Combining SOTA performance, minimalist architecture, DLinear-like efficiency, and transferability; the overall practical value is high)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[ICML 2025\] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [\[ICML 2025\] Winner-takes-all for Multivariate Probabilistic Time Series Forecasting](winner-takes-all_for_multivariate_probabilistic_time_series_forecasting.md)
- [\[ICML 2025\] Learning Soft Sparse Shapes for Efficient Time-Series Classification](learning_soft_sparse_shapes_for_efficient_time-series_classification.md)

</div>

<!-- RELATED:END -->
