---
title: >-
  [Paper Note] Tuning the Burn-in Phase in RNN Training Improves Performance
description: >-
  [ICLR 2026][Time Series][Recurrent Neural Networks] This paper provides a theoretical analysis of the critical role played by the burn-in length $m$ in Truncated Backpropagation Through Time (TBPTT) training of RNNs. It…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Recurrent Neural Networks"
  - "Truncated Backpropagation Through Time"
  - "Burn-in Phase"
  - "Time Series Forecasting"
  - "System Identification"
date: 2026-05-08
content_hash: 0d29d1f4a4f84d9b
---

# Tuning the Burn-in Phase in RNN Training Improves Performance

**Conference**: ICLR 2026
**arXiv**: [2602.10911](https://arxiv.org/abs/2602.10911)
**Area**: Time Series
**Keywords**: Recurrent Neural Networks, Truncated Backpropagation Through Time, Burn-in Phase, Time Series Forecasting, System Identification

## TL;DR

This paper provides a theoretical analysis of the critical role played by the burn-in length $m$ in Truncated Backpropagation Through Time (TBPTT) training of RNNs. It establishes upper bounds on training regret and validates through system identification and time series forecasting experiments that appropriately tuning the burn-in phase can reduce prediction error by more than 60%.

## Background & Motivation

The standard method for training RNNs is Backpropagation Through Time (BPTT), which faces three major challenges on long sequences:

**High computational and memory cost**: Forward and backward passes must traverse the entire sequence.

**Gradient explosion/vanishing**: Long-sequence BPTT is numerically unstable.

**Complex loss landscape**: Optimization becomes increasingly difficult as sequence length grows.

**Truncated BPTT (TBPTT)** is the standard practical alternative: the long sequence is split into short sub-sequences, each processed independently with BPTT. However, the hidden state at the beginning of each sub-sequence is typically **zero-initialized**, causing initial outputs to be contaminated by transient effects.

**Burn-in phase**: The outputs of the first $m$ steps of each sub-sequence are excluded from the loss function, allowing the network to "warm up." This practice has been adopted in several prior works (Jaeger 2002; Bonassi 2022; Beintema 2021) but has never been theoretically analyzed or systematically tuned — a gap this paper addresses.

## Method

### Overall Architecture

Consider the standard RNN model:

$$h_t = f(h_{t-1}, x_t; \theta_h), \quad y_t = g(h_t, x_t; \theta_y)$$

TBPTT splits the training sequence $D$ into $S$ sub-sequences of length $N$, with the burn-in loss defined as:

$$L(\theta; D_i) = \frac{1}{N-m}\sum_{j=m+1}^{N}\|y_j(0, \theta, X_i^d) - y_{j|i}^d\|^2$$

where $m \in [0, N-1]$ is the burn-in length.

### Key Theoretical Results

**Assumption 1 (Exponential Incremental Output Stability)**: There exist $C > 0$ and $\lambda \in (0,1)$ such that:

$$\|y_t(h_0^{(1)}, \theta, X) - y_t(h_0^{(2)}, \theta, X)\| \leq C\lambda^t \|h_0^{(1)} - h_0^{(2)}\|$$

That is, the dependence of RNN outputs on initialization decays exponentially over time.

**Theorem 1 (Training Regret)**: The regret of the TBPTT solution $\theta^*$ relative to a reference solution $\theta^b$ satisfies:

$$V^* - V^b \leq C_2 \cdot \frac{\lambda^m}{N-m}$$

**Theorem 2 (Performance Regret)**: The performance regret on the full sequence satisfies:

$$P(0, \theta^*; D) - P(h_0^b, \theta^b; D) \leq E_2 \cdot \sqrt{\frac{(S-1)\lambda^{2o_{\min}} + S\lambda^m}{T-m}}$$

### Core Idea

- The regret upper bound **critically depends on the interplay between $m$ and $\lambda$**.
- Smaller $\lambda$ (faster forgetting) allows larger $m$; larger $\lambda$ (slower forgetting) calls for smaller $m$.
- The burn-in length should be treated as a **standard hyperparameter** in RNN training.

## Key Experimental Results

### System Identification (LSTM, $d_h=8$)

| Dataset | $N$ | Baseline $m=0$ Train MSE | Optimal $m^*$ Train MSE | Improvement | Test MSE Improvement |
|--------|-----|-------|---------|------|------------|
| Silver-Box | 100 | 0.242 | 0.042 | **-83%** | -51% |
| RLC | 200 | 0.971 | 0.309 | **-68%** | **-79%** |
| RLC | 500 | 0.442 | 0.186 | **-58%** | -58% |
| W-H | 100 | 0.220 | 0.153 | -31% | -4% |

### Time Series Forecasting Comparison

| Method | Advantages | Disadvantages |
|------|------|------|
| TBPTT ($m=\bar{m}$) | Simple | Suboptimal performance |
| TBPTT ($m=m^*$) | **Universally best** | Requires tuning |
| Stateful TBPTT | Carries hidden state | Numerically unstable |
| Full BPTT | Theoretically optimal | Computationally expensive |

### Key Findings

1. **Large impact of burn-in**: Appropriate tuning reduces training and test MSE by over 60%.
2. **Qualitative consistency**: The effect pattern of burn-in is highly consistent across different window lengths $N$.
3. **TBPTT can outperform BPTT**: The stochasticity introduced by zero-initialized TBPTT yields better numerical stability and faster convergence.
4. **Regularization effect**: Choosing $m < \bar{m}$ provides additional regularization.
5. **Theory-experiment agreement**: Regret decays exponentially as $m$ decreases, consistent with theoretical predictions.

## Highlights & Insights

1. **From heuristic to theory**: The burn-in phase is elevated from an unanalyzed empirical practice to a theoretically grounded training technique.
2. **Optimal control perspective**: TBPTT performance is analyzed via the turnpike property, establishing a bridge between RNN training and optimal control theory.
3. **High practical value**: Tuning burn-in incurs zero overhead (no increase in model complexity) and is directly applicable within existing frameworks.
4. **Broad applicability**: The theory applies to all RNN architectures satisfying exponential output stability (LSTM, GRU, LRU, SSMs, etc.).

## Limitations & Future Work

1. Theoretical analysis is limited to MSE loss; generalization to other losses such as classification objectives requires further investigation.
2. Quantitative verification of Assumption 1 is conservative (practical estimation of $\lambda$ is required), and exact computation of optimal $m$ remains challenging.
3. Only zero-initialized TBPTT is considered; theoretical analysis of stateful training is left for future work.
4. Experiments are conducted solely with LSTM architectures; validation on modern SSMs (e.g., Mamba) is absent.
5. Only univariate forecasting is considered; multivariate settings are not addressed.

## Rating ⭐⭐⭐⭐

The paper offers rigorous theory, sufficient experiments, and strong practical utility. It elevates a neglected training hyperparameter to a theoretically informed tuning target with direct benefits for RNN training practice. Its main limitation is the relatively small experimental scale, with no evaluation on modern large-scale sequential models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition](../../NeurIPS2025/time_series/human-machine_ritual_synergic_performance_through_real-time_motion_recognition.md)
- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming](hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)
- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)

</div>

<!-- RELATED:END -->
