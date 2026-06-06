---
title: >-
  [Paper Note] Doubly Outlier-Robust Online Infinite Hidden Markov Model
description: >-
  [ICML 2026][Time Series][Infinite Hidden Markov Model] This paper proposes BR-iHMM: combining "robust observation update (WoLF)" with "batched state inference (degenerate sticky HDP prior)" to provide bounded Posterior I…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Infinite Hidden Markov Model"
  - "Online Inference"
  - "Robust Bayesian"
  - "Outliers"
  - "Posterior Influence Function"
date: 2026-05-08
content_hash: 4d941c09732242e0
---

# Doubly Outlier-Robust Online Infinite Hidden Markov Model

**Conference**: ICML 2026  
**arXiv**: [2604.14322](https://arxiv.org/abs/2604.14322)  
**Code**: None  
**Area**: Time Series / Bayesian Online Learning / Regime Switching  
**Keywords**: Infinite Hidden Markov Model, Online Inference, Robust Bayesian, Outliers, Posterior Influence Function

## TL;DR
This paper proposes BR-iHMM: combining "robust observation update (WoLF)" with "batched state inference (degenerate sticky HDP prior)" to provide bounded Posterior Influence Function (PIF) in both observation and state spaces for online infinite HMMs. On streaming data with outliers from financial order books, electricity load, and synthetic regression, one-step prediction RMSE is reduced by up to 67%.

## Background & Motivation

**Background**: There are two main approaches for handling non-stationary streaming data. Bayesian changepoint detection (BOCD) and Kalman filtering "reset or forget" upon detecting a changepoint, making it impossible to reuse historical regimes. Online iHMM (HDP-iHMM) maintains a reusable regime library, enabling rapid return to historical regimes when they reappear, making it more suitable for scenarios like finance, electricity, and continual learning where "old regimes reoccur + occasional new regimes".

**Limitations of Prior Work**: The flexibility of iHMM is a double-edged sword—an outlier can (i) contaminate the posterior of the current regime's parameters, degrading subsequent predictions; (ii) mislead the model into creating spurious new regimes, harming interpretability and predictive accuracy. Existing robust methods either focus only on the observation space (robust KF/WoLF) or only on offline iHMM state-space pruning, without addressing both in an online setting.

**Key Challenge**: In the HDP-iHMM framework, "observation robustness" and "state robustness" are **independent** PIF dimensions—the authors prove in Theorem 4.1: even if WoLF ensures bounded PIF$_{\theta_t}$ on the observation side, the state-side PIF$_{s_t}$ can still be driven to infinity by outliers (since large residuals make "new regime" most attractive under the HDP prior).

**Goal**: (1) Formally define double robustness for online iHMM; (2) design an algorithm that bounds both PIF$_{\theta_t}$ and PIF$_{s_t}$; (3) maintain online real-time performance without increasing computational complexity.

**Key Insight**: For the observation side, reuse the WoLF from the generalised Bayes framework (using IMQ weights to downweight outlier likelihoods); for the state side, borrow from batch inference—**a single outlier is insufficient to create a new regime; consistent evidence from multiple consecutive observations is required**.

**Core Idea**: Use a "degenerate sticky HDP prior" to force state transitions to occur only at batch boundaries (intra-batch self-transition probability $\kappa_t=\infty$, inter-batch $\kappa_t=0$), so that a new regime requires sufficient evidence within a window of length $B$; this provides a tunable robustness-adaptivity trade-off parameter $B$.

## Method

### Overall Architecture
BR-iHMM uses Particle Learning (PL) for SMC inference. Every $B$ steps form a batch:
1. Each particle's state $s_t^{(i)}$ predicts $\hat y_{t+1:t+B}$ for the next $B$ steps;
2. IMQ weights $w_{l,t}^{(i)}=W(y_{t+b},\hat y_{l,t+b|t})$ downweight observations;
3. Compute the batched posterior $\nu(s_{1:t+B})$, allowing state transitions only at batch boundaries;
4. Resample when ESS falls below a threshold;
5. WoLF updates the Gaussian posterior $\Psi$ of active states;
6. Antoniak auxiliary variables update HDP structural parameters $\Phi$.

Within each batch, states are forced to self-transition, so state sampling is performed **once per batch**, avoiding the exponential path explosion with batch size $B$.

### Key Designs

1. **WoLF Weighted Observation Update (observation-space robustness)**:

    - **Function**: Caps the influence of a single extreme observation on the posterior of $\theta_{s_t}$.
    - **Mechanism**: Replace the likelihood with a weighted likelihood $P(y_t\mid\theta,x_t)^{W(y_t,\hat y_{s_t})^2}$, where the IMQ weight is $W(y,\hat y)^2=1/(1+c^{-2}\|y-\hat y\|_{R_t}^2)$. For linear Gaussian emissions, conjugacy is preserved; the closed-form update only replaces the covariance $S_{s_t}$ in the Kalman gain with $S_{s_t}=f(x_t)\Sigma_{s_t}f(x_t)^\top+R_t/w_{s_t,t|t-1}^2$. As the residual increases, $w^2\to 0$, $S_{s_t}\to\infty$, and the Kalman gain approaches zero, freezing the posterior.
    - **Design Motivation**: Standard Bayesian updates for LG models have unbounded PIF (arbitrary residuals can arbitrarily shift the posterior); WoLF locks PIF$_{\theta_t}$ via a bounded weight function, while retaining conjugacy.

2. **Batch Inference + Degenerate Sticky HDP (state-space robustness)**:

    - **Function**: Prevents a single outlier from triggering a new regime; state decisions are made only at batch boundaries.
    - **Mechanism**: Define the batched log posterior $\log\nu(s_{1:t+B})=\sum_{b=1}^B w_{s_{t+b},t+b|t}^2\log P(y_{t+b}\mid \dots)+\log\sum_{s_{1:t}}P(s_{1:t}|D)P(s_{t+1}|s_t,\Phi_t)\prod_{b=2}^B\mathbb{1}(s_{t+b-1}=s_{t+b})$. The sticky HDP sets self-transition bias $\kappa_t$ to $0$ (boundary) or $\infty$ (internal), enforcing intra-batch state consistency; only when multiple observations within a batch provide consistent evidence for a new regime does the path posterior switch.
    - **Design Motivation**: Theorem 4.1 proves observation robustness alone is insufficient; batching is mathematically equivalent to defining PIF for "short outlier sequences" (batched PIF), providing a robustness-adaptivity trade-off—larger $B$ increases resistance to persistent noise but also increases detection lag for true regime switches.

3. **Antoniak Auxiliary Variables + State Pruning (scalability)**:

    - **Function**: Ensures the number of states does not diverge on long streams and maintains online updates of HDP structural parameters.
    - **Mechanism**: Each batch samples auxiliary variables $\mathbf{M}_t\sim\text{Antoniak}(\mathbf{N}_t,\alpha,\beta)$ to update global HDP weights $\hat\beta_t$; for particles exceeding MAX_STATES, prune old regimes based on usage frequency and recency heuristics (removing counts and global weights together).
    - **Design Motivation**: iHMM nominally allows infinite states, but without pruning, the bookkeeping matrix $\mathbf{N}_t\in\mathbb{N}^{t\times t}$ explodes in streaming scenarios; pruning keeps the number of states constant, and Propositions D.1/D.2 formally guarantee the batched mechanism's complexity remains O(1) state sampling per batch.

### Loss & Training

- No neural network training; pure Bayesian online inference; implemented in JAX, single RTX 3090 GPU.
- Hyperparameters $B$, IMQ threshold $c$, ESS threshold $\tau_{\text{ESS}}$, and particle number $N$ are tuned via Bayesian optimization on the training split; $B$ ranges for different tasks are given in the appendix.
- Concentration parameters $\hat\alpha_0,\hat\gamma_0\sim\text{Gam}(1,1)$ use non-informative priors, with Escobar–West conjugate updates.

## Key Experimental Results

### Main Results
One-step prediction RMSE (mean ± stdev over 100 runs):

| Model | Synthetic ($d=100$, 1% outliers) | Electricity | OFI |
|------|------|------|------|
| BOCD | 123.12 ± 0.014 | 0.80 ± 0.11 | 0.733 |
| iHMM | 101.7 ± 0.026 | 0.57 ± 0.03 | 0.620 ± 0.080 |
| WoLF-iHMM | 103.8 ± 0.012 | 0.63 ± 0.03 | 0.623 ± 0.089 |
| **BR-iHMM (ours)** | **46.1 ± 0.003** | **0.47 ± 0.04** | **0.616 ± 0.082** |
| offline-iHMM (oracle) | 2.9 | 0.32 | 0.552 |

On the synthetic task, BR-iHMM reduces RMSE by about 55% compared to iHMM and 63% compared to BOCD; on electricity data, BR-iHMM is the only online model to identify the regime switch caused by COVID-19 in March 2020, while iHMM and WoLF-iHMM remain stuck in a single regime.

### Ablation Study

| Configuration | Synthetic RMSE | Failure Mode |
|------|------|------|
| iHMM (baseline) | 101.7 | 30+ spurious regimes, each outlier triggers a new state |
| WoLF-iHMM (observation robustness only) | 103.8 | Parameter posterior stable but states still fragmented, **slightly worse than plain iHMM** |
| BR-iHMM (B=1) | ≈100 | Equivalent to WoLF-iHMM |
| **BR-iHMM (B>1)** | **46.1** | Stabilizes after short-term calibration, recovers true 3 regimes |

### Key Findings
- **Single robustness is insufficient**: WoLF-iHMM performs slightly worse than iHMM, confirming Theorem 4.1—observation robustness alone leads to PIF$_{s_t}$ dominating failure modes.
- **$B$ is a key trade-off parameter**: Appendix Figures E.10 / E.12 show that larger $B$ increases robustness to short outliers but also increases detection lag; $B$ is smaller for financial OFI, larger for electricity.
- **Complexity advantage**: Standard iHMM allows arbitrary switching within a batch, leading to exponentially many paths in $B$; degenerate sticky reduces this to one state sampling per batch, making complexity independent of batch size.
- **Prediction and segmentation win-win**: Table 2 (segmentation) shows BR-iHMM also outperforms DSM-BOCD and iHMM (unknown-var) in changepoint detection metrics.

## Highlights & Insights
- **Theory-driven**: Robustness is rigorously defined as bounded PIF, with Theorems 4.1/4.2 proving double robustness is doubly necessary, grounding the method design.
- **Batch-PIF concept**: Extends PIF from "single-point perturbation" to "short-sequence perturbation", naturally introducing $B$ as an interpretable parameter; this "batched robustness" can transfer to other online Bayesian models (e.g., GP, streaming VI).
- **Dual role of degenerate sticky HDP**: Provides both mathematical state-space contraction (using $\kappa_t\in\{0,\infty\}$ extremes) and computational complexity reduction (eliminating path explosion), achieving both goals.
- **Counterintuitive finding of Theorem 4.1**: Strengthening observation robustness alone can worsen state inference (since suppressed residuals increase the relative likelihood of "new regime"), warning future work.

## Limitations & Future Work
- Full derivation is only provided for LG emissions; the authors claim extensibility to exponential families but do not empirically validate this.
- $B$ is a fixed-a-priori hyperparameter, requiring BayesOpt tuning; adaptive $B$ (e.g., dynamic adjustment based on SNR) is a natural extension.
- The pruning heuristic (usage frequency + recency) is relatively coarse and may mistakenly remove long-tail regimes; theoretically, there is no guarantee pruning preserves PIF bounds.
- Maximum experimental dimension is $d=100$; effectiveness of IMQ weights in ultra-high-dimensional settings (e.g., image features) is untested.
- The offline oracle (offline-iHMM) still significantly outperforms BR-iHMM (synthetic RMSE 2.9 vs. 46.1), indicating a large online–offline gap, essentially due to SMC particle and burn-in limitations.

## Related Work & Insights
- **vs. Standard iHMM (Beal et al. 2001; Teh et al. 2006)**: Adds double robustness with almost no extra computational cost.
- **vs. WoLF (Duran-Martin et al. 2024)**: WoLF only provides robustness for single-state LG models; this work embeds it in the multi-state HDP-iHMM framework and adds state-space robustness.
- **vs. DSM-BOCD (Altamirano et al. 2023)**: BOCD does not support regime reuse; this work retains both robustness and reuse.
- **vs. offline iHMM (Van Gael et al. 2008)**: The latter uses beam sampling for offline MCMC, achieving oracle performance but requiring 1000 iterations; BR-iHMM processes data online in a single pass.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalization of "double robustness" + batched degenerate sticky HDP construction is highly novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Three types of data (synthetic, electricity, order book) + 100 runs, but all relatively low-dimensional
- Writing Quality: ⭐⭐⭐⭐ PIF definition, Theorems 4.1/4.2 derivations, and algorithm pseudocode are clearly organized
- Value: ⭐⭐⭐⭐ Provides a complete toolkit for scenarios (finance, sensors, continual learning) requiring both regime reuse and outlier resistance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](../../ICLR2026/time_series/delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[NeurIPS 2025\] AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](../../NeurIPS2025/time_series/aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)
- [\[NeurIPS 2025\] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](../../NeurIPS2025/time_series/maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)

</div>

<!-- RELATED:END -->
