---
title: >-
  [Paper Note] Fern: Chaining Spectral Pearls — Ellipsoidal Forecasting Beyond Trajectories for Time Series
description: >-
  [NeurIPS 2025][Time Series][long-term time series forecasting] This paper proposes Fern (Forecasting with Ellipsoidal RepresentatioN), which replaces conventional trajectory prediction with patch-wise ellipsoidal transpo…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "long-term time series forecasting"
  - "optimal transport"
  - "Koopman operator"
  - "spectral decomposition"
  - "chaotic systems"
  - "Wasserstein distance"
date: 2026-05-08
content_hash: 20ff357a56175497
---

# Fern: Chaining Spectral Pearls — Ellipsoidal Forecasting Beyond Trajectories for Time Series

**Conference**: NeurIPS 2025
**arXiv**: [2505.17370](https://arxiv.org/abs/2505.17370)
**Code**: To be confirmed
**Area**: Time Series
**Keywords**: long-term time series forecasting, optimal transport, Koopman operator, spectral decomposition, chaotic systems, Wasserstein distance

## TL;DR

This paper proposes Fern (Forecasting with Ellipsoidal RepresentatioN), which replaces conventional trajectory prediction with patch-wise ellipsoidal transport (rotation–scaling–translation). Fern substantially outperforms baselines on chaotic systems while remaining competitive on standard LTSF benchmarks.

## Background & Motivation

Long-term time series forecasting (LTSF) faces two fundamental challenges:

1. **Evaluation blind spots**: Mainstream evaluation relies on pointwise metrics such as MSE/MAE, and benchmark datasets are dominated by quasi-periodic or noisy data. This masks model fragility under chaotic dynamics — DLinear performs well on Weather but is 24× worse than Fern on Lorenz63.
2. **Lack of actionable interpretability**: Existing models provide no diagnostic tools for failure modes, even when long-term prediction is bound to fail. Users need not only internal model transparency but also tools for stability analysis, regime-shift identification, and direct intervention when necessary.

The authors argue that pointwise metrics over-penalize phase shifts — a precise prediction delayed by one hour may score worse under MSE than a 24-hour mean prediction. Genuine long-term forecasting should focus on **local conditional geometry** rather than exact trajectories.

## Core Problem

How to design an LTSF model that is robust on chaotic systems, competitive on standard benchmarks, and spectrally interpretable?

This decomposes into three sub-problems:
- **How to evaluate?** Introduce distributional metrics (Wasserstein Distance, SWD) and Effective Prediction Time (EPT).
- **How to forecast?** Predict local geometric shapes (ellipsoids) rather than exact trajectories.
- **How to interpret?** Provide transparent diagnostics via explicit spectral factors (eigenvalues/eigenvectors).

## Method

### 1. New Evaluation Protocol

- **Sliced Wsserstein Distance (SWD)**: Treats predictions and ground truth as equally weighted empirical distributions and computes the $W_2$ distance over 1D order statistics. SWD is a permutation-invariant shape metric that complements pointwise indicators.
- **Effective Prediction Time (EPT)**: The first time step at which prediction error exceeds a predefined threshold. Quantifies the boundary between reliable forecasting and failure.
- **Chaotic system stress tests**: Evaluation on low-dimensional chaotic systems (Lorenz63, Rössler, Chua) whose data-generating processes are known, deterministic, easily visualizable, and have short Lyapunov times, fully exposing model behavior under deterministic chaos.

### 2. Fern Model Architecture

Fern frames forecasting as a sequence of controlled ellipsoidal transformations, with the core idea of "predicting geometry rather than dynamics."

**Ellipsoidal Transport (ET) Layer**:

For each patch, starting from an isotropic Gaussian (sphere), three linear operations deform it into an anisotropic Gaussian (ellipsoid):

$$y^* = U(z) K \Lambda(z) K^\top U(z)^\top y_0 + t(z)$$

- $U(z)$: data-dependent orthogonal matrix (rotation), selects local coordinate frame
- $\Lambda(z)$: non-negative diagonal matrix (anisotropic scaling), serves as local eigenvalues
- $K$: fixed learnable $2\times2$ block-diagonal matrix $\begin{bmatrix}a & -b \\ b & a\end{bmatrix}$, simulates complex eigenvalues, captures global dynamics
- $t(z)$: translation vector, captures first-order residuals

**Encoder**:

A variant of Augmented Normalizing Flows (ANF) is adopted. The input $x$ and latent variable $z \sim \mathcal{N}(0,I)$ are mutually refined over $K_{enc}=5$ rounds of alternating scale-shift operations:

- $z \leftarrow s^*(x) \odot z + t(x)$ ($x$ conditions $z$)
- $x \leftarrow s^*(z) \odot x + t(z)$ ($z$ conditions $x$)

**Transport Stage**:

Based on the encoded $z$, the initial Gaussian $y_0$ is partitioned into patches of 24 steps. Ellipsoidal transformation parameters are generated for each patch and concatenated into the final prediction.

### 3. Theoretical Foundations

- **Brenier's theorem**: Under quadratic cost, there exists an almost everywhere unique optimal transport map $T = \nabla\phi$ from an absolutely continuous source distribution to a target distribution, whose Jacobian is symmetric positive semi-definite (SPSD). The ET layer searches precisely within the Brenier class of SPSD Jacobians.
- **Takens' embedding theorem**: Time-delay embedding of a single channel can reconstruct a topological equivalent of a dynamical system's attractor, providing theoretical justification for channel-independent prediction and explaining why patching is effective while naive channel mixing often degrades performance.
- **Koopman perspective**: $U(z)$ selects local Koopman modes, $K$ encodes fixed global complex eigenstructure, and $\Lambda(z)$ modulates instance-level amplitudes. By decomposing local eigenvalues into variable modulation and invariant base frequencies, a clear linear dynamical interpretation is preserved.

### 4. Key Design Choices

- **Allowing zero scaling**: Approximates low-rank mappings; predictions of dimension 336/720 do not require all eigenvalues to be nonzero.
- **Gaussian source $y_0$**: Satisfies Brenier's theorem's requirement for absolute continuity; Gaussian-to-Gaussian maps preserve closure in Koopman coordinates.
- **Updating $y_0$ via translation only**: SPSD matrices are not closed under composition; translation does not affect the Jacobian and is the only admissible dynamic update operation.

## Key Experimental Results

### Chaotic Systems (seq_len=336, simple average)

| Dataset | Fern MSE | TimeMixer MSE | PatchTST MSE | DLinear MSE |
|--------|----------|---------------|--------------|-------------|
| Lorenz63 | **21.82** | 30.94 | 30.11 | 67.76 |
| Rössler | **0.04** | 6.01 | 8.33 | 11.64 |
| Chua | **0.08** | 0.20 | 0.49 | 0.39 |

On Rössler, Fern's MSE is only 0.62% of TimeMixer's, 0.47% of PatchTST's, and 0.36% of DLinear's.

### Standard LTSF Benchmarks

| Dataset | Fern MSE | TimeMixer MSE | PatchTST MSE | DLinear MSE |
|--------|----------|---------------|--------------|-------------|
| ETTm2 | **13.57** | 15.04 | 15.63 | 15.49 |
| ETTh1 | **6.60** | 6.83 | 6.62 | 7.04 |
| ETTm1 | **5.80** | 5.27 | 5.36 | 6.31 |
| Weather* | 0.27 | 0.27 | 0.24 | **0.21** |

Fern achieves the best MSE on ETTm2 and ETTh1 and remains competitive on ETTm1. On Weather, DLinear performs best, consistent with its near-random-walk characteristics.

### Ablation Study (PredLen=192)

- Removing the ET layer (transport only): ETTh2 MSE surges from 11.19 to 408.49.
- Removing rotation and Koopman structure: Lorenz63 MSE increases from 2.06 to 3.02; SWD rises from 0.33 to 0.91.
- Removing patching: ETTh2 MSE increases from 11.19 to 13.78.
- Components are complementary; no single ablation outperforms the full model across all datasets and metrics.

## Highlights & Insights

1. **Geometric forecasting paradigm**: Shifting from "predict exact trajectories" to "predict local ellipsoidal geometry" is particularly effective on chaotic systems — if the model correctly identifies the right region of the attractor, predictions remain qualitatively reasonable.
2. **Explicit spectral interpretability**: Per-patch eigenvalues and eigenvectors are directly usable for stability analysis, pattern recognition, and regime-shift detection.
3. **Theoretical elegance**: Normalizing Flows, Optimal Transport, and the Koopman operator are unified within the ellipsoidal transport framework.
4. **Evaluation methodology contribution**: SWD + EPT + chaotic stress tests constitute a more comprehensive evaluation protocol.

## Limitations & Future Work

1. **Underperforms on simpler datasets**: When data is intrinsically close to a random walk (e.g., Weather), the advantage of geometric prediction diminishes.
2. **Not optimal on ETTh2**: TimeMixer and PatchTST outperform Fern on ETTh2, possibly due to that dataset's particular non-stationarity.
3. **Only three baselines compared**: Comparisons with iTransformer, Crossformer, and other recent SOTA methods are absent.
4. **Low-dimensional chaotic systems only**: Lorenz63, Rössler, and Chua are all 3-dimensional; performance on high-dimensional chaotic systems remains unclear.
5. **Not a density model**: Fern forgoes NF likelihood computation and full OT solutions, which may limit uncertainty quantification.

## Related Work & Insights

| Method | Characteristics | Difference from Fern |
|------|------|----------------|
| DLinear | Simple linear model | Strong on periodic/noisy data; fragile under chaos (24× worse) |
| PatchTST | Transformer + patching | Good generality but no spectral interpretability |
| TimeMixer | Multi-scale mixing | Moderate on chaos; MSE on Rössler is 160× that of Fern |
| Koopman-based methods | Global linearization | Fern uses local linearization + global complex eigenstructure, avoiding closure issues |
| SINDy/HAVOK | Sparse equation discovery | System identification rather than conditional forecasting |
| Iterative one-step methods (DSDL) | Recursive prediction | Chaos-specific; Fern is a direct multi-step general-purpose predictor |

The idea of **"predicting geometry rather than trajectories"** transfers to other domains: predicting motion-field geometry in video forecasting, or predicting local deformation ellipsoids in point cloud forecasting. The interpretability afforded by spectral decomposition has practical value for financial time series (volatility regime shifts) and meteorological forecasting (extreme event detection). Proposing SWD as an evaluation metric deserves broader adoption in the time series community, particularly in long-horizon settings where shape fidelity is more meaningful than pointwise accuracy. Applying Brenier's theorem within a forecasting framework represents a novel angle connecting OT theory to practical prediction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The ellipsoidal forecasting paradigm unifying three major frameworks is highly innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Chaotic experiments are thorough, but standard benchmark comparisons cover too few methods)
- Writing Quality: ⭐⭐⭐⭐ (Combines position paper and model paper styles; structure is clear though information-dense)
- Value: ⭐⭐⭐⭐ (Evaluation methodology and spectral interpretability offer long-term impact for the community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[AAAI 2026\] Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting](../../AAAI2026/time_series/sonnet_spectral_operator_neural_network_for_multivariable_time_series_forecastin.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](../../ICLR2026/time_series/routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)

</div>

<!-- RELATED:END -->
