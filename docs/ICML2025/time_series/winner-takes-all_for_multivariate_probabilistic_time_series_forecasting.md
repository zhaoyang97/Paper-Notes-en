---
title: >-
  [Paper Note] Winner-takes-all for Multivariate Probabilistic Time Series Forecasting
description: >-
  [ICML2025][Time Series][Probabilistic Time Series Forecasting] Proposes TimeMCL, which introduces the Winner-Takes-All (WTA) loss of Multiple Choice Learning to multivariate probabilistic time series forecasting. Through a single forward pass of a multi-head network, it generates diverse and representative future trajectories, successfully balancing prediction quality and computational efficiency.
tags:
  - "ICML2025"
  - "Time Series"
  - "Probabilistic Time Series Forecasting"
  - "Winner-Takes-All"
  - "Multiple Choice Learning"
  - "Functional Quantization"
  - "Multimodal Distribution"
date: 2026-05-08
content_hash: a033119f71b3eb98
---

# Winner-takes-all for Multivariate Probabilistic Time Series Forecasting

**Conference**: ICML2025  
**arXiv**: [2506.05515](https://arxiv.org/abs/2506.05515)  
**Code**: [GitHub](https://github.com/Victorletzelter/timeMCL)  
**Area**: Time Series Forecasting  
**Keywords**: Probabilistic Time Series Forecasting, Winner-Takes-All, Multiple Choice Learning, Functional Quantization, Multimodal Distribution

## TL;DR

Proposes TimeMCL, which introduces the Winner-Takes-All (WTA) loss of Multiple Choice Learning to multivariate probabilistic time series forecasting. Through a single forward pass of a multi-head network, it generates diverse and representative future trajectories, successfully balancing prediction quality and computational efficiency.

## Background & Motivation

Time series forecasting is inherently a highly underdetermined problem: input information is insufficient to eliminate uncertainty about the future, and the data itself contains noise. An ideal predictor should provide multiple plausible future trajectories along with their respective probabilities.

Limitations of prior probabilistic time series forecasting approaches:

- **Parametric Distribution Methods** (e.g., DeepAR): These impose an explicit parametric distribution on the output and perform maximum likelihood estimation. While efficient, their flexibility is limited by the chosen distribution family, making it difficult to capture complex multimodal uncertainties.
- **Generative Model Methods** (e.g., TimeGrad diffusion models, TempFlow normalizing flows): These exhibit strong empirical performance on high-dimensional time series, but their inference costs are prohibitively high (TimeGrad's FLOPs are approximately 345 times higher than those of TimeMCL), and they lack an explicit mechanism to guarantee the diversity of hypotheses in a single sampling run.

The motivation of this study: Can $K$ diverse and representative predicted trajectories be generated at the cost of a single forward pass?

## Method

### Overall Architecture

TimeMCL superimposes $K$ prediction heads $f_\theta^k$ and $K$ scoring heads $\gamma_\theta^k$ on top of a shared RNN (LSTM) encoder. The former generate $K$ hypothetical trajectories, while the latter predict the probability of each head "winning".

### Winner-Takes-All Loss

For each training sequence $(x_{1:t_0-1}, x_{t_0:T})$:

1. Compute the negative log-likelihood of each head:

$$\mathcal{L}_\theta^k = -\sum_{t=t_0}^{T} \log p_\theta^k(x_t \mid x_{1:t-1})$$

2. Select the "winner": $k^\star = \arg\min_k \mathcal{L}_\theta^k$
3. Perform backpropagation **only** on the winning head.

Overall WTA objective:

$$\mathcal{L}^{\text{WTA}}(\theta_1,\dots,\theta_K) = \mathbb{E}_{x_{1:T}}\left[\min_{k=1,\dots,K} \mathcal{L}_\theta^k\right]$$

### Scoring Head Loss

The scoring heads are trained using binary cross-entropy to learn the probability of each head winning:

$$\mathcal{L}^s = \mathbb{E}_{x_{1:T}}\left[\sum_{k,t} \text{BCE}\left(\mathbb{1}[k=k^\star],\; \gamma_\theta^k(h_{t-1})\right)\right]$$

### Final Loss

$$\mathcal{L} = \mathcal{L}^{\text{WTA}} + \beta \cdot \mathcal{L}^s$$

Where $\beta > 0$ is the weight of the confidence loss (set to 0.5 in experiments).

### WTA Relaxation Variants

To prevent some heads from being under-trained (similar to the empty cluster problem in K-Means), the authors test two relaxation schemes:

- **Relaxed-WTA (R-WTA)**: The winner is weighted by $1-\varepsilon$, and the remaining heads each receive a weight of $\varepsilon/(K-1)$.
- **Annealed MCL (aMCL)**: Uses softmin to compute weights $q_k(T) \propto \exp(-\mathcal{L}_\theta^k / T)$, where the temperature $T$ linearly anneals during training.

### Inference

A single forward pass yields $K$ trajectories $\hat{x}^1,\dots,\hat{x}^K$ and their corresponding scores. Weighted sampling based on the scores can be performed for subsequent tasks (such as interval estimation).

### Theoretical Guarantee: Conditional Functional Quantizer

**Proposition 5.1**: Under three assumptions (sufficiently large batch size, sufficient network expressiveness, and training convergence), TimeMCL is a conditional stationary functional quantizer, meaning that the output of each head is the conditional expectation of the trajectories in its Voronoi cell:

$$\mathscr{F}_\theta^k(x_{1:t_0-1}) = \mathbb{E}[x_{t_0:T} \mid x_{t_0:T} \in \mathcal{X}_k(x_{1:t_0-1})]$$

This implies that TimeMCL is essentially "conditional K-Means on the trajectory space," representing an optimal $K$-point finite approximation of the target conditional distribution.

**Proposition 5.2**: The scoring heads converge to the true probability mass of each Voronoi cell: $\Gamma_\theta^k = \mathbb{P}(x_{t_0:T} \in \mathcal{X}_k)$.

## Key Experimental Results

### Datasets

6 real-world time series benchmarks: Solar (137-dim), Electricity (370-dim), Exchange (8-dim), Traffic (963-dim), Taxi (1214-dim), Wikipedia (2000-dim).

### Distortion Risk ($K=16$, ↓ lower is better)

| Dataset | TimeGrad | Tactis2 | TempFlow | DeepAR | **TimeMCL(R.)** | TimeMCL(A.) |
|--------|----------|---------|----------|--------|-----------------|-------------|
| Solar | 360.6 | 358.0 | 371.1 | 748.7 | **280.0** | 305.5 |
| Traffic | 0.78 | 0.84 | 1.21 | 2.12 | **0.68** | 0.72 |
| Taxi | 209.6 | 243.6 | 268.7 | 407.4 | **187.8** | 229.3 |
| Elec. | 9872 | 11616 | 14836 | 133107 | **11604** | 11611 |

TimeMCL(R.) achieves the best Distortion on Solar, Traffic, and Taxi, ranking in the top two on most datasets.

### Computational Efficiency (Exchange, $K=16$)

| Metric | TimeGrad | Tactis2 | TempFlow | DeepAR | **TimeMCL** |
|------|----------|---------|----------|--------|-------------|
| FLOPs | 3.05×10⁹ | 1.85×10⁸ | 9.29×10⁷ | **4.65×10⁵** | 8.83×10⁶ |
| Running time (s) | 241.57 | 8.69 | 1.39 | **0.70** | 1.12 |

The FLOPs of TimeMCL are only **1/345** of TimeGrad, and its running time is only 1/215, ranking second in computational efficiency (after DeepAR), while its Distortion is far superior to DeepAR.

### Smoothness (Total Variation, ↓ lower is better)

The predicted trajectories of TimeMCL are significantly smoother on all datasets, which is consistent with the theoretical analysis—the output of each head is the conditional mean of its Voronoi cell, thereby averaging out the noise.

## Highlights & Insights

1. **Elegant Theoretical Insight**: Establishes a rigorous connection between WTA training and optimal functional quantization theory, proving that TimeMCL is a generalization of conditional K-Means on the trajectory space, and the scoring heads converge to the true probability mass of the Voronoi cells.
2. **Excellent Efficiency-Quality Trade-off**: Generates $K$ diverse trajectories in a single forward pass, with FLOPs 2-3 orders of magnitude lower than diffusion models, yet achieving better Distortion.
3. **Theoretical Explanation for Smooth Predictions**: As conditional means, the predicted trajectories are naturally smoother than random sampling; this is a provable mathematical property rather than a coincidence.
4. **Plug-and-Play**: Multi-heads combined with the WTA loss can be superimposed on any RNN/Transformer backbone without changing the underlying architecture.
5. **Dual Validation on Synthetic and Real Data**: Qualitatively validates the quantization properties on Brownian Motion, Brownian Bridge, and AR(5) processes.

## Limitations & Future Work

1. **Number of Hypotheses $K$ Must be Pre-specified**: Similar to K-Means requiring the number of clusters to be pre-determined, the selection of $K$ affects the trade-off between accuracy and efficiency. The paper does not provide an adaptive selection strategy.
2. **Backbone Limited to RNNs**: Experiments only utilize the LSTM backbone; performance on Transformer backbones remains to be validated.
3. **Suboptimal CRPS/RMSE**: TimeMCL is not SOTA under standard CRPS and RMSE metrics, as its training objective is designed for Distortion rather than these traditional metrics.
4. **Under-trained Head Issue**: Certain heads may suffer from under-training due to dominance by the primary mode. Although alleviated by relaxation variants and scoring heads, this issue is not fully resolved.
5. **Inter-variable Dependency**: Each head independently predicts the conditional distribution for each dimension, limiting the capacity to model joint distributions across dimensions (compared to copula-based methods like Tactis2).
6. **Excessive Smoothness can be a Double-edged Sword**: In scenarios where capturing sharp spikes or sudden transitions is critical, over-smoothing can be disadvantageous.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematically introduces the MCL/WTA paradigm to time series forecasting for the first time, backed by functional quantization theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated on 6 benchmarks plus synthetic data under multiple metrics, including computational cost analysis.
- Writing Quality: ⭐⭐⭐⭐ — Tight integration of theory and empirical evidence, with a clear structure.
- Value: ⭐⭐⭐⭐ — Provides a highly efficient and theoretically guaranteed new pathway for probabilistic time series forecasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)
- [\[ICML 2025\] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)
- [\[ICML 2026\] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](../../ICML2026/time_series/parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)
- [\[ICML 2025\] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [\[NeurIPS 2025\] xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories](../../NeurIPS2025/time_series/xlstm-mixer_multivariate_time_series_forecasting_by_mixing_via_scalar_memories.md)

</div>

<!-- RELATED:END -->
