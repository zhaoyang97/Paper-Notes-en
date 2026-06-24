---
title: >-
  [Paper Note] UnHiPPO: Uncertainty-Aware Initialization for State Space Models
description: >-
  [ICML 2025][state space models] This work extends the HiPPO framework to handle noisy measurements. By reformulating the initialization of State Space Models (SSMs) as a linear stochastic control/estimation problem, the authors derive an uncertainty-aware initialization scheme for SSM dynamics, which significantly enhances noise robustness without increasing runtime overhead.
tags:
  - "ICML 2025"
  - "state space models"
  - "HiPPO"
  - "initialization"
  - "uncertainty"
  - "Kalman filter"
  - "noise robustness"
date: 2026-05-08
content_hash: e5d18dbc3d2c96b6
---

# UnHiPPO: Uncertainty-Aware Initialization for State Space Models

**Conference**: ICML 2025  
**arXiv**: [2506.05065](https://arxiv.org/abs/2506.05065)  
**Code**: [https://cs.cit.tum.de/daml/unhippo](https://cs.cit.tum.de/daml/unhippo)  
**Area**: Sequence Models / State Space Models  
**Keywords**: state space models, HiPPO, initialization, uncertainty, Kalman filter, noise robustness

## TL;DR
This work extends the HiPPO framework to handle noisy measurements. By reformulating the initialization of State Space Models (SSMs) as a linear stochastic control/estimation problem, the authors derive an uncertainty-aware initialization scheme for SSM dynamics, which significantly enhances noise robustness without increasing runtime overhead.

## Background & Motivation
**Background**: State space models (SSMs) such as S4 and Mamba are emerging as dominant architectures for sequence modeling. The HiPPO (High-order Polynomial Projection Operators) framework provides an elegant initialization scheme for SSMs and is key to the success of the S4 family. HiPPO initializes the $A, B$ matrices by on-line approximation of the polynomial projections of the input signal.

**Limitations of Prior Work**: The core assumption of HiPPO is noise-free data—treating the input signal as a deterministic control signal. However, real-world data often contains observational noise (e.g., sensor noise, quantization errors), which significantly degrades SSM training and inference quality.

**Key Challenge**: The initialization yielded by HiPPO is optimal under noise-free conditions, but in noisy scenarios, this initialization indiscriminately propagates noise into the state representation, degrading signal quality.

**Goal**: Design a noise-aware initialization scheme for SSMs.

**Key Insight**: Reinterpret HiPPO as a linear stochastic control/estimation problem.

**Core Idea**: Treat the data not as a deterministic control signal, but as noisy observations of a latent system, and derive a de-noising initialization based on Kalman filtering principles.

## Method

### Overall Architecture
Input: Sequence data (potentially noisy), SSM architecture (e.g., S4, Mamba)  
Output: Uncertainty-aware initialization for $A, B$ matrices  

Pipeline:
1. Formulate standard HiPPO as a control problem: $\dot{x}(t) = Ax(t) + Bu(t)$, where $u(t)$ is the noise-free input.
2. Reformulate as an estimation problem: assume a latent signal $f(t)$ exists, with observations $u(t) = f(t) + \epsilon(t)$.
3. Derive the Kalman-type posterior estimation dynamics.
4. Use the derived new $A', B'$ as the initialization for the SSM.

### Key Designs

1. **Stochastic Control Interpretation of HiPPO**:

    - **Function**: Reframe HiPPO from a function approximation problem to a state estimation problem.
    - **Mechanism**: Standard HiPPO solves $\min_{c(t)} \int_0^t (f(\tau) - \sum_n c_n(t) P_n(\tau))^2 w(\tau) d\tau$, where $f$ is the input function and $P_n$ is the orthogonal polynomial basis. This paper assumes $f$ is a latent variable with observation $u = f + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
    - **Design Motivation**: Under the state estimation framework, the signal and noise can be naturally separated.

2. **Uncertainty-Aware Dynamics Derivation**:

    - **Function**: Derive modified $A, B$ matrices that account for noise.
    - **Mechanism**: Under linear-Gaussian assumptions, the mean and covariance of the posterior distribution are updated via Kalman filtering. The new dynamics are:
    $\dot{x}(t) = (A - K(t)C)x(t) + K(t)u(t)$
      where $K(t)$ is the Kalman gain, which automatically balances prior information and new observations. As noise approaches zero, $K \to B$, reducing to prior HiPPO.
    - **Design Motivation**: Kalman gain automatically suppresses noise—the larger the noise, the more conservative the response to new observations.

3. **Seamless Integration with Standard SSMs**:

    - **Function**: Ensure that the modified initialization does not increase model complexity or runtime.
    - **Mechanism**: The modified $A' = A - KC, B' = K$ remain constant matrices (under steady-state Kalman gain); thus, the architecture and computation of the SSM remain completely unchanged, with only the initial values modified.
    - **Design Motivation**: Ensure the practicality of the method—no architectural changes, no additional inference cost.

### Loss & Training
The initialization scheme does not change the training process. The SSM is trained using standard sequence modeling losses (e.g., cross-entropy, MSE). The key difference lies solely in the initialization of $A, B$.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | UnHiPPO | Standard HiPPO | Random Init | Noise Condition |
|---|---|---|---|---|---|
| Long Range Arena (Noise-free) | ACC | 86.2 | 86.0 | 82.1 | Clean |
| LRA + Gaussian Noise $\sigma=0.1$ | ACC | **83.5** | 79.8 | 75.4 | Light Noise |
| LRA + Gaussian Noise $\sigma=0.3$ | ACC | **78.2** | 68.5 | 63.1 | Moderate Noise |
| Time Series Forecasting (ETTh1) | MSE | **0.372** | 0.385 | 0.412 | Noisy |
| Speech Recognition (Noisy SC09) | ACC | **91.3** | 86.7 | 82.5 | Real Noise |

### Ablation Study

| Configuration | LRA ACC ($\sigma=0.2$) | Description |
|---|---|---|
| UnHiPPO (Full) | **80.8** | Full method |
| Standard HiPPO | 74.2 | No noise considered |
| HiPPO + Input De-noising | 77.5 | De-noise before input |
| Diff. Noise Estimation $\hat{\sigma}=0.1$ (Underestimated) | 79.5 | Small impact from noise estimation |
| Diff. Noise Estimation $\hat{\sigma}=0.5$ (Overestimated) | 79.0 | Over-smoothed but robust |
| Training with noise / Inference noise-free | 85.5 | De-noising during training is beneficial |

### Key Findings
- Under noise-free conditions, UnHiPPO performs on par with standard HiPPO (does not harm clean scenarios).
- The larger the noise, the more pronounced the advantage of UnHiPPO—yielding up to a 10-point accuracy improvement at $\sigma=0.3$.
- The estimation of the noise parameter $\hat{\sigma}$ does not require high precision; the method is highly robust to it.
- Performance gains in real-world noisy scenarios (speech recognition) validate its practical value.

## Highlights & Insights
- **Elegant Theoretical Extension**: Generalizes HiPPO from deterministic approximation to stochastic estimation, offering mathematical completeness.
- **Zero-Cost Improvement**: Only modifies the initialization values, without adding any computational overhead.
- **Clever Application of Kalman Filtering**: Leverages classical control theory tools to address challenges in modern deep learning.
- **No Degradation on Clean Data**: A crucial property for any alternative initialization scheme.

## Limitations & Future Work
- Assumes Gaussian noise; extension to non-Gaussian noise requires further work.
- Handling of time-varying noise levels (e.g., adaptive Kalman gain) could be further improved.
- The efficacy of combining this with selective SSMs like Mamba warrants in-depth investigation.

## Related Work & Insights
- Direct extension of the HiPPO/S4 theory by Gu et al. (2020, 2022).
- Applications of Kalman filtering in deep learning are increasing (e.g., KalmanNet).
- Holds direct value for SSM applications on noisy data in time series analysis and signal processing.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The theoretical contribution of extending HiPPO to stochastic settings is highly significant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across various noise levels and tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear theoretical derivation, with a natural entry point from control theory.
- **Value**: ⭐⭐⭐⭐⭐ A foundational improvement; any HiPPO-based SSM can benefit from this.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](../../NeurIPS2025/others/deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[ICLR 2026\] Exploring State-Space Models for Data-Specific Neural Representations](../../ICLR2026/others/exploring_state-space_models_for_data-specific_neural_representations.md)
- [\[ICML 2025\] Rethinking Aleatoric and Epistemic Uncertainty](rethinking_aleatoric_and_epistemic_uncertainty.md)
- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)
- [\[ICML 2025\] Nonparametric Modern Hopfield Models](nonparametric_modern_hopfield_models.md)

</div>

<!-- RELATED:END -->
