---
title: >-
  [Paper Note] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction
description: >-
  [NeurIPS 2025][Time Series][Jump diffusion] This paper proposes Neural MJD, which parameterizes a non-stationary Merton Jump Diffusion model via neural networks, casting prediction as an SDE simulation problem. The framework combines a time-varying Itô diffusion (capturing continuous drift) with a time-varying compound Poisson process (modeling abrupt jumps), and employs likelihood truncation together with an Euler-Maruyama with Restart solver to enable scalable learning and inference.
tags:
  - NeurIPS 2025
  - Time Series
  - Jump diffusion
  - stochastic differential equations
  - non-stationarity
  - time series forecasting
  - financial modeling
date: 2026-05-08
content_hash: a1987f1682b67190
---

# Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2506.04542](https://arxiv.org/abs/2506.04542)
**Code**: [https://github.com/DSL-Lab/neural-MJD](https://github.com/DSL-Lab/neural-MJD)
**Area**: Time Series
**Keywords**: Jump diffusion, stochastic differential equations, non-stationarity, time series forecasting, financial modeling

## TL;DR

This paper proposes Neural MJD, which parameterizes a non-stationary Merton Jump Diffusion model via neural networks, casting prediction as an SDE simulation problem. The framework combines a time-varying Itô diffusion (capturing continuous drift) with a time-varying compound Poisson process (modeling abrupt jumps), and employs likelihood truncation together with an Euler-Maruyama with Restart solver to enable scalable learning and inference.

## Background & Motivation

**Background**: Deep learning methods for time series (Transformers, SSMs, etc.) achieve strong performance on standard benchmarks, but are fundamentally deterministic models or black-box probabilistic models that lack explicit modeling of the underlying stochastic process.

**Limitations of Prior Work**:
   - Real-world time series frequently exhibit a mixture of **continuous trends and abrupt jumps** (e.g., stock prices surging or crashing due to breaking news, retail revenue spiking due to promotions).
   - Classical statistical models (MJD, Lévy processes) assume independent stationary increments, making them unable to handle non-stationarity or capture multivariate dependencies.
   - Deep learning methods lack explicit jump modeling, while generative models (e.g., diffusion models) are computationally expensive and do not directly model abrupt changes.

**Key Challenge**: Classical jump diffusion theory is mathematically elegant but has fixed parameters → real-world data requires time-varying parameters; deep learning is flexible but lacks mathematical structure → the two must be integrated.

**Goal**: Design a time series forecasting model that possesses an explicit SDE mathematical form (interpretable and amenable to probabilistic inference) while flexibly learning time-varying parameters via neural networks.

**Key Insight**: Replace the fixed parameters $\{\mu, \sigma, \lambda, \nu, \gamma\}$ of classical MJD with time-varying parameters $\{\mu_t, \sigma_t, \lambda_t, \nu_t, \gamma_t\}$ predicted by a neural network, preserving the mathematical structure of the SDE framework.

**Core Idea**: A neural network conditioned on historical data and context predicts all future SDE parameters in a single forward pass, enabling scalable learning of non-stationary jump diffusion.

## Method

### Overall Architecture

The SDE for non-stationary Neural MJD is:

$$dS_t = S_t \big((\mu_t - \lambda_t k_t)dt + \sigma_t dW_t + \int_{\mathbb{R}^d}(y-1)N(dt, dy)\big)$$

where $\mu_t, \sigma_t, \lambda_t, \nu_t, \gamma_t = f_\theta(S_0, S_{-1}, \ldots, S_{-T_p}, C, t)$

### Key Designs

#### 1. Neural Network Prediction of Time-Varying Parameters

**Function**: Output all future SDE parameters in a single forward pass.

**Mechanism**:
- Input: historical sequence $\{S_{-T_p}, \ldots, S_0\}$ + optional context $C$ (spatial graph, features, etc.)
- Output: $\{(\mu_\tau, \sigma_\tau, \lambda_\tau, \nu_\tau, \gamma_\tau)\}_{\tau=1}^{T_f}$
- Jump times $N_t$ follow a Poisson process with time-varying intensity $\lambda_t$
- Jump magnitudes satisfy $\ln Y_t \sim \mathcal{N}(\nu_t, \gamma_t^2)$

**Design Motivation**: Unlike PINNs that incorporate a hand-crafted SDE as a regularization term in the loss, Neural MJD directly parameterizes the non-stationary SDE, yielding greater flexibility.

#### 2. Likelihood Truncation

**Function**: Render the infinite-series likelihood tractable.

**Core Problem**: The exact likelihood $P(\ln S_T | \mathcal{C}) = \sum_{n=0}^{\infty} P(N_t=n) \cdot \phi(\ln S_T; a_n, b_n^2)$ is an infinite series.

**Solution**: Truncate to at most $\kappa$ jumps.

**Theoretical Guarantee** (Theorem 4.1): The truncation error $\Psi_\kappa(t, \delta)$ decays at a super-exponential rate: $O(\kappa^{-\kappa})$. In practice, $\kappa = 5$ is sufficient.

**Piecewise Constant Parameters**: The continuous time axis is discretized such that parameters remain constant over each integer interval $[\tau-1, \tau)$, avoiding numerical integration.

#### 3. Parameter Bootstrapping Training

**Function**: Mitigate the distribution mismatch between training and inference.

**Mechanism**: When computing $\ln P(\ln S_\tau | S_{\tau-1}, \mathcal{C})$, instead of conditioning on the ground-truth $S_{\tau-1}$ (teacher forcing), the conditional mean prediction $\hat{S}_{\tau-1} = \mathbb{E}[S_{\tau-1} | \mathcal{C}]$ is substituted.

The conditional mean admits a closed-form solution:

$$\mathbb{E}[S_t | \mathcal{C}] = S_0 \exp\!\Big(\sum_{j=1}^{\rho_t - 1} \mu_j + (t - \rho_t + 1)\mu_{\rho_t}\Big)$$

#### 4. Euler-Maruyama with Restart Solver

**Function**: Improve the accuracy and stability of trajectory sampling during inference.

**Mechanism**:
- In the standard EM method, errors accumulate exponentially over time: $\epsilon_t^E \leq K\exp(Lt)/M$
- The Restart strategy resets the state to the analytically computed conditional mean at each integer time point
- The error bound improves to: $\epsilon_t^R \leq K\exp(L(t-\lfloor t\rfloor))/M$

**Effect**: The factor $t$ in the exponent is replaced by $t - \lfloor t\rfloor \in [0,1)$, effectively suppressing long-horizon error accumulation.

### Loss & Training

Total loss: $\mathcal{L} = -\sum_{\tau=1}^{T_f} \psi_\tau + \omega \|S_\tau - \hat{S}_\tau\|^2$

- First term: MLE likelihood (truncated to $\kappa=5$ jumps)
- Second term: conditional mean regularization (encouraging mean predictions to remain close to ground truth)
- The loop over time steps is parallelizable, as the conditional mean computation does not depend on sequential steps

## Key Experimental Results

### Synthetic Data (Generated from Standard MJD)

| Model | MAE↓ | R²↑ | minMAE↓ | maxR²↑ |
|-------|------|-----|---------|--------|
| ARIMA | 0.29 | -0.15 | — | — |
| MJD | 0.21 | 0.08 | 0.18 | 0.15 |
| Neural BS | 0.15 | 0.25 | 0.10 | 0.35 |
| **Neural MJD** | **0.09** | **0.32** | **0.07** | **0.39** |

Neural MJD achieves the best performance under all evaluation protocols.

### SafeGraph Business Analytics Data

| Model | MAE↓ | R²↑ | Best-K MAE↓ | Best-K R²↑ |
|-------|------|-----|-------------|-----------|
| GCN | 95.2 | 0.432 | — | — |
| EDM | 57.6 | 0.525 | 49.4 | 0.556 |
| FM | 54.5 | 0.540 | 47.8 | 0.552 |
| Neural BS | 56.4 | 0.539 | 45.6 | 0.561 |
| **Neural MJD** | **54.1** | **0.549** | **42.3** | **0.565** |

### S&P 500 Stock Data

| Model | MAE↓ | R²↑ | minMAE↓ | maxR²↑ |
|-------|------|-----|---------|--------|
| MJD | 64.3 | 0.092 | 40.7 | 0.235 |
| Neural BS | — | — | — | — |
| **Neural MJD** | Best | Best | Best | Best |

### Ablation Study

| Variant | Effect |
|---------|--------|
| Remove jump component (→ Neural BS) | Performance degrades, especially on sequences with abrupt changes |
| Teacher forcing instead of bootstrapping | Training-inference mismatch leads to error accumulation |
| Standard EM instead of Restart EM | Increased variance in long-horizon predictions |
| $\kappa = 1$ vs. $\kappa = 5$ | $\kappa = 5$ is superior; higher values yield marginal improvement |

### Key Findings

1. Explicit jump modeling is more effective than implicit approaches (diffusion models, flow matching).
2. Neural MJD consistently outperforms baselines across all three evaluation protocols: deterministic (mean), probabilistic (Best-K), and likelihood-weighted.
3. Restart EM significantly reduces variance in long-horizon inference.
4. The parameter bootstrapping training strategy effectively mitigates error accumulation.
5. The model is architecture-agnostic — experiments employ a Transformer backbone with optional Graphormer spatial encoding.

## Highlights & Insights

- **Organic integration of theory and practice**: The mathematical structure of MJD (closed-form likelihood, closed-form conditional mean) is preserved while neural networks provide flexibility.
- **Likelihood truncation error bound** (Theorem 4.1) guarantees super-exponential decay at $O(\kappa^{-\kappa})$, making $\kappa=5$ practically exact.
- **Restart EM** (Proposition 4.2) reduces the exponential factor in the weak convergence error from $\exp(Lt)$ to $\exp(L(t-\lfloor t\rfloor))$.
- Jump labels are not required — a key distinction from temporal point process methods that require annotated jump events.
- Continuous-time simulation is supported: although trained at discrete time steps, the model can sample at arbitrary temporal resolutions.

## Limitations & Future Work

1. The piecewise constant parameter assumption (parameters fixed within each integer interval) is a simplification that may lack sufficient granularity for rapidly varying parameters.
2. Jump magnitudes are assumed to follow a log-normal distribution, which may be inadequate for heavy-tailed distributions requiring more flexible parameterization.
3. Each training step requires a full forward pass through the backbone, resulting in high computational cost for long prediction horizons.
4. Comparisons with diffusion-based generative models may not be entirely fair, as the backbone is shared but the task objectives differ.
5. Multi-scale jumps (superposition of jump processes at different frequencies) remain unexplored.

## Related Work & Insights

- **Key distinction from Neural ODE/SDE**: Neural MJD explicitly models the jump process, not merely continuous diffusion.
- **Key distinction from PINNs for finance**: PINNs incorporate BS/MJD as a PDE constraint; Neural MJD directly parameterizes the SDE.
- The Restart strategy bears conceptual similarity to the Parareal algorithm — both adopt a coarse-then-fine hybrid approach.
- The framework is extensible to more general stochastic processes, including Lévy processes and Hawkes processes.

## Rating

⭐⭐⭐⭐

The paper achieves a natural and principled integration of classical financial mathematics (MJD) with deep learning. The design is clean and effective. Theoretical contributions — including the truncation error bound and the Restart EM error improvement — are rigorous. Experiments comprehensively cover both synthetic and real-world data across three evaluation protocols.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](../../AAAI2026/time_series/towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[NeurIPS 2025\] Learning Time-Scale Invariant Population-Level Neural Representations](learning_time-scale_invariant_population-level_neural_representations.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)

<!-- RELATED:END -->
