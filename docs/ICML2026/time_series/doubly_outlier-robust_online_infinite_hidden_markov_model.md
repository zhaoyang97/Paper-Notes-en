---
title: >-
  [Paper Note] Doubly Outlier-Robust Online Infinite Hidden Markov Model
description: >-
  [ICML 2026][Time Series][Infinite Hidden Markov Models] This paper proposes BR-iHMM: it combines "robust observation updates (WoLF)" with "batched state inference (degenerate sticky HDP prior)" to provide bounded Posteri…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Infinite Hidden Markov Models"
  - "Online Inference"
  - "Robust Bayes"
  - "Outliers"
  - "Posterior Influence Function"
date: 2026-05-08
content_hash: 4b4ac9a16a6d6602
---

# Doubly Outlier-Robust Online Infinite Hidden Markov Model

**Conference**: ICML 2026  
**arXiv**: [2604.14322](https://arxiv.org/abs/2604.14322)  
**Code**: None  
**Area**: Time Series / Bayesian Online Learning / Regime Switching  
**Keywords**: Infinite Hidden Markov Models, Online Inference, Robust Bayes, Outliers, Posterior Influence Function

## TL;DR
This paper proposes BR-iHMM: it combines "robust observation updates (WoLF)" with "batched state inference (degenerate sticky HDP prior)" to provide bounded Posterior Influence Functions (PIF) in both observation and state spaces for online infinite hidden Markov models. It reduces one-step prediction RMSE by up to 67% on outlier-contaminated streaming data across financial order books, electricity loads, and synthetic regression.

## Background & Motivation

**Background**: There are two main factions for handling non-stationary streaming data. Bayesian changepoint detection (BOCD) and Kalman filtering "reset or forget" every time a changepoint is detected, failing to reuse historical regimes. Online iHMM (HDP-iHMM) maintains a library of reusable regimes, allowing for rapid return when historical regimes reappear, which is more suitable for scenarios like finance, power grids, and continual learning where "old regimes recur + occasional new regimes."

**Limitations of Prior Work**: The flexibility of iHMM is a double-edged sword—a single outlier can simultaneously (i) contaminate the parameter posterior of the current regime, degrading subsequent predictions; and (ii) mislead the model into creating spurious states by assuming a new regime has occurred, damaging both interpretability and predictive accuracy. Existing robust methods either focus only on the observation space (Robust KF/WoLF) or involve offline iHMM pruning for the state space, failing to resolve both simultaneously in an online setting.

**Key Challenge**: In the HDP-iHMM framework, "observation robustness" and "state robustness" are **independent** PIF dimensions—the authors prove Theorem 4.1: even if the observation side uses WoLF to ensure bounded PIF$_{\theta_t}$, the state-side PIF$_{s_t}$ can still be driven to infinity (as large residuals make a "new regime" the most attractive under the HDP prior).

**Goal**: (1) Formally define double robustness for online iHMM; (2) design an algorithm that simultaneously bounds PIF$_{\theta_t}$ and PIF$_{s_t}$; (3) maintain online real-time performance without sacrificing computational complexity.

**Key Insight**: The observation side reuses the WoLF framework under generalized Bayes (downweighting outlier likelihood using IMQ weights); the state side draws from batch inference ideas—**a single outlier is insufficient to create a new regime; consistent evidence from multiple consecutive observations is required**.

**Core Idea**: Use a "degenerate sticky HDP prior" to force state transitions to contract at batch boundaries (intra-batch self-transition probability $\kappa_t=\infty$, inter-batch $\kappa_t=0$), thereby requiring a new regime to gather sufficient evidence within a window of length $B$. This also provides a tunable robustness-adaptivity trade-off parameter $B$.

## Method

### Overall Architecture
BR-iHMM uses Particle Learning (PL) for SMC inference. Each $B$ steps constitute a batch:
1. Use the state $s_t^{(i)}$ of each particle to make predictions $\hat y_{t+1:t+B}$ for the next $B$ steps;
2. Use IMQ weights $w_{l,t}^{(i)}=W(y_{t+b},\hat y_{l,t+b|t})$ to downweight observations;
3. Calculate the batched posterior $\nu(s_{1:t+B})$, allowing state transitions only at batch boundaries;
4. Resample when ESS falls below a threshold;
5. Update the Gaussian posterior $\Psi$ of active states using WoLF;
6. Update HDP structural parameters $\Phi$ using Antoniak auxiliary variables.

Since internal batch states are forced to self-transition, state sampling is performed **only once per batch**, avoiding the exponential path explosion associated with batch size $B$.

### Key Designs

1. **WoLF Weighted Observation Updates (Observation-space Robustness)**:
    - **Function**: Ensures that the influence of a single extreme observation on the $\theta_{s_t}$ posterior is hard-clipped by an upper bound.
    - **Mechanism**: Replaces the likelihood with a weighted likelihood $P(y_t\mid\theta,x_t)^{W(y_t,\hat y_{s_t})^2}$, where weights take the IMQ form $W(y,\hat y)^2=1/(1+c^{-2}\|y-\hat y\|_{R_t}^2)$. Under linear Gaussian emissions, conjugacy is preserved; the closed-form update simply replaces the covariance $S_{s_t}$ in the Kalman gain with $S_{s_t}=f(x_t)\Sigma_{s_t}f(x_t)^\top+R_t/w_{s_t,t|t-1}^2$. As residuals grow, $w^2\to 0$ and $S_{s_t}\to\infty$, causing the Kalman gain to approach 0, thus freezing the posterior.
    - **Design Motivation**: Standard Bayesian updates have an unbounded PIF for LG models (arbitrarily large residuals lead to arbitrary posterior shifts). WoLF locks PIF$_{\theta_t}$ via a bounded weight function while maintaining conjugacy.

2. **Batch Inference + Degenerate Sticky HDP (State-space Robustness)**:
    - **Function**: Prohibits single outliers from triggering new regimes; state decisions are made only once per batch boundary.
    - **Mechanism**: Defines the batched log posterior $\log\nu(s_{1:t+B})=\sum_{b=1}^B w_{s_{t+b},t+b|t}^2\log P(y_{t+b}\mid \dots)+\log\sum_{s_{1:t}}P(s_{1:t}|D)P(s_{t+1}|s_t,\Phi_t)\prod_{b=2}^B\mathbb{1}(s_{t+b-1}=s_{t+b})$. By setting the sticky HDP self-transition bias $\kappa_t$ to $0$ (boundary) or $\infty$ (internal), intra-batch state consistency is forced; the path posterior switches only if multiple observations within a batch simultaneously provide evidence that a "new regime is more plausible."
    - **Design Motivation**: Theorem 4.1 proves that observation robustness alone is insufficient. Batching mathematically defines PIF for "short outlier sequences" (batched PIF) and provides a robustness-adaptivity trade-off—larger $B$ increases resistance to sustained noise but increases the lag in detecting real regime switches.

3. **Antoniak Auxiliary Variables + State Pruning (Scalability)**:
    - **Function**: Ensures the number of states does not diverge on long streaming data and maintains online updates for HDP structural parameters.
    - **Mechanism**: Each batch uses $\mathbf{M}_t\sim\text{Antoniak}(\mathbf{N}_t,\alpha,\beta)$ to sample auxiliary variables for updating the HDP global weights $\hat\beta_t$. For particles exceeding MAX_STATES, old regimes are heuristically pruned based on frequency and recency (counts and global weights are deleted together).
    - **Design Motivation**: While iHMM nominally allows infinite states, without pruning in a streaming scenario, the bookkeeping matrix $\mathbf{N}_t\in\mathbb{N}^{t\times t}$ would explode. Pruning keeps the state count constant, while Propositions D.1/D.2 formally guarantee that the complexity of the batched mechanism remains O(1) state samplings per batch.

### Loss & Training
- No NN training; pure Bayesian online inference. Implemented in JAX, using a single RTX 3090.
- Hyperparameters $B$, IMQ threshold $c$, ESS threshold $\tau_{\text{ESS}}$, and particle count $N$ are tuned via Bayesian optimization on a training partition. Ranges for $B$ across different tasks are provided in the appendix.
- Concentration parameters $\hat\alpha_0,\hat\gamma_0\sim\text{Gam}(1,1)$ use uninformative priors, combined with Escobar–West conjugate updates.

## Key Experimental Results

### Main Results
One-step prediction RMSE (Mean ± stdev over 100 repetitions):

| Model | Synthetic ($d=100$, 1% Outliers) | Electricity | OFI |
|------|------|------|------|
| BOCD | 123.12 ± 0.014 | 0.80 ± 0.11 | 0.733 |
| iHMM | 101.7 ± 0.026 | 0.57 ± 0.03 | 0.620 ± 0.080 |
| WoLF-iHMM | 103.8 ± 0.012 | 0.63 ± 0.03 | 0.623 ± 0.089 |
| **BR-iHMM (Ours)** | **46.1 ± 0.003** | **0.47 ± 0.04** | **0.616 ± 0.082** |
| offline-iHMM (oracle) | 2.9 | 0.32 | 0.552 |

On the Synthetic task, BR-iHMM reduces RMSE by approximately 55% relative to iHMM and 63% relative to BOCD. On electricity data, BR-iHMM is the only online model to identify the regime switch triggered by COVID-19 in March 2020; iHMM and WoLF-iHMM remained stuck in a single regime throughout.

### Ablation Study

| Configuration | Synthetic RMSE | Failure Mode |
|------|------|------|
| iHMM (Baseline) | 101.7 | 30+ spurious regimes; every outlier triggers a new state |
| WoLF-iHMM (Obs. Robust Only) | 103.8 | Parameter posteriors stable but states still fragmented; **slightly worse than pure iHMM** |
| BR-iHMM (B=1) | ≈100 | Equivalent to WoLF-iHMM |
| **BR-iHMM (B>1)** | **46.1** | Stabilized after short-term calibration; recovers true 3 regimes |

### Key Findings
- **Single Robustness is Insufficient**: WoLF-iHMM performs slightly worse than iHMM, validating Theorem 4.1—ignoring state robustness allows PIF$_{s_t}$ to dominate failure modes.
- **B is the Key Trade-off Parameter**: Appendix Figures E.10 / E.12 show that larger $B$ increases robustness to short outliers but increases detection latency. $B$ is smaller on financial OFI and larger on electricity loads.
- **Complexity Advantage**: Standard iHMM allows arbitrary switches within a batch, leading to exponential path growth in $B$; degenerate sticky HDP reduces this to one state sampling per batch, making complexity independent of batch size.
- **Win-Win for Prediction and Segmentation**: Table 2 (segmentation) shows BR-iHMM outperforms DSM-BOCD and iHMM (unknown-var) on changepoint detection metrics.

## Highlights & Insights
- **Theory-First**: Robustness is strictly defined as bounded PIF, followed by Theorem 4.1/4.2 proving that double robustness is a dual necessity, providing a principled foundation for method design.
- **Batched PIF Concept**: Generalizing PIF from "single-point perturbations" to "short sequence perturbations" naturally yields the interpretable parameter $B$. This "batched robustness" idea is transferable to other online Bayesian models (e.g., GP, streaming VI).
- **Dual-use of Degenerate Sticky HDP**: Serves both as a mathematical state-space contraction (via $\kappa_t\in\{0,\infty\}$ limit) and a computational complexity savior (eliminating path exponential explosion).
- **Counter-intuitive Finding in Theorem 4.1**: Simply strengthening observation robustness can worsen state inference (as suppressed residuals increase the relative likelihood of a "new regime"), serving as a warning for subsequent related work.

## Limitations & Future Work
- Full derivations are only provided for LG emissions; while the authors claim extensibility to exponential families, it remains empirically unverified.
- $B$ is a fixed-a-priori hyperparameter requiring BayesOpt tuning; adaptive $B$ (e.g., dynamically adjusted based on SNR) is a natural extension.
- Pruning heuristics (usage frequency + recency) are relatively crude and might mistakenly delete long-tail regimes; there is no theoretical guarantee that pruning preserves PIF bounds.
- The maximum dimensionality in experiments is $d=100$; the effectiveness of IMQ weights in ultra-high-dimensional cases (e.g., image features) is not validated.
- The offline oracle (offline-iHMM) still significantly outperforms BR-iHMM (synthetic RMSE 2.9 vs. 46.1), indicating a large online–offline gap primarily due to SMC particle counts and burn-in constraints.

## Related Work & Insights
- **vs. Standard iHMM (Beal et al. 2001; Teh et al. 2006)**: Adds double robustness with almost no additional computational overhead.
- **vs. WoLF (Duran-Martin et al. 2024)**: WoLF only performs robustness for single-state LG models; this work embeds it into the multi-state HDP-iHMM framework and complements it with state-space robustness.
- **vs. DSM-BOCD (Altamirano et al. 2023)**: BOCD does not support regime reuse; this work preserves both robustness and reuse capabilities.
- **vs. Offline iHMM (Van Gael et al. 2008)**: The latter uses beam sampling for offline MCMC, achieving oracle performance but requiring 1000 iterations; BR-iHMM requires only a single online pass.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalization of "Double Robustness" + batched construction of degenerate sticky HDP is quite novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + Electricity + Order Book data + 100 repetitions, though mostly low-dimensional.
- Writing Quality: ⭐⭐⭐⭐ PIF definitions, Theorem 4.1/4.2 derivations, and algorithm pseudocode are very clearly organized.
- Value: ⭐⭐⭐⭐ Provides a complete toolchain for scenarios (finance, sensors, continual learning) needing both historical regime reuse and outlier resistance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal](distmatch_adaptive_binning_via_distribution_matching_for_robust_sequential_confo.md)
- [\[ICML 2026\] Divide and Contrast: Learning Robust Temporal Features Without Augmentation](divide_and_contrast_learning_robust_temporal_features_without_augmentation.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](../../ICLR2026/time_series/delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)

</div>

<!-- RELATED:END -->
