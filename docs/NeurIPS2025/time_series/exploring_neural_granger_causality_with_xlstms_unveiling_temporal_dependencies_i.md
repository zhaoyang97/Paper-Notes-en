---
title: >-
  [Paper Note] Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data
description: >-
  [NEURIPS2025][Time Series][Granger Causality] This paper proposes GC-xLSTM, which leverages the xLSTM architecture combined with a novel dynamic sparsity optimization strategy to uncover Granger causal relationships in multivariate time series, achieving state-of-the-art performance on multiple datasets.
tags:
  - NEURIPS2025
  - Time Series
  - Granger Causality
  - xLSTM
  - Sparsity
  - Causal Discovery
date: 2026-05-08
content_hash: b711c8ecda87048a
---

# Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data

**Conference**: NEURIPS2025
**arXiv**: [2502.09981](https://arxiv.org/abs/2502.09981)
**Code**: [github.com/harpoonix/GC-xLSTM](https://github.com/harpoonix/GC-xLSTM)
**Area**: Time Series
**Keywords**: Granger Causality, xLSTM, Sparsity, Time Series, Causal Discovery

## TL;DR

This paper proposes GC-xLSTM, which leverages the xLSTM architecture combined with a novel dynamic sparsity optimization strategy to uncover Granger causal relationships in multivariate time series, achieving state-of-the-art performance on multiple datasets.

## Background & Motivation

Granger Causality (GC) is a classical framework for determining whether past values of one time series help predict another. Traditional methods rely on vector autoregression (VAR) models for statistical hypothesis testing, but suffer from the following limitations:

1. **Linearity assumption**: Classical GC methods assume linear relationships between variables, failing to capture nonlinear dependencies.
2. **Short-range dependencies**: Existing Neural GC methods based on MLPs or standard LSTMs exhibit limited capacity to capture long-range dependencies.
3. **Insufficient sparsity enforcement**: Traditional Group Lasso regularization can only shrink parameters close to zero, allowing subsequent layers to re-amplify weak signals. This results in insufficiently strict sparsity and requires manual threshold selection $\tau$.

The recently proposed xLSTM (Extended LSTM), through exponential gating and matrix-valued associative memory, has demonstrated strong capability in sequence modeling, particularly for time series forecasting. This presents an opportunity to leverage superior sequence modeling capacity for GC discovery.

## Core Problem

How to robustly discover Granger causal relationships in complex time series data characterized by nonlinearity, long-range dependencies, and noise, by exploiting the stronger modeling capacity of the xLSTM architecture?

## Method

### Overall Architecture

GC-xLSTM adopts a component-wise architecture, modeling each variable $v$ independently:

1. **Sparse feature projection**: A projection matrix $\mathbf{W}_v \in \mathbb{R}^{D \times V}$ is learned for each variable $v$, projecting $V$ variables into a $D$-dimensional latent space: $\mathbf{x}_v = \mathbf{W}_v \mathbf{S} + \mathbf{b}_v$.
2. **xLSTM prediction**: A single sLSTM block (containing one sLSTM layer) with hidden dimension 32 is used for autoregressive prediction.
3. **Causal relationship extraction**: GC relationships are directly read from the sparse structure of the projection matrix $\mathbf{W}_v$.

The system trains $V$ independent models in total, each responsible for predicting one variable and extracting its incoming edges from the sparse projection.

### Joint Optimization Strategy (Core Innovation)

The paper proposes an alternating optimization scheme that jointly optimizes the predictive model and strict sparsity. Each step consists of two phases:

**Phase 1: Gradient descent update.** The projection weights $\phi$, shrinkage coefficients $\boldsymbol{\alpha}$, and xLSTM parameters $\theta$ are jointly optimized by minimizing:

$$\mathcal{L}_{\text{pred}}(\mathbf{S}; \phi_v, \theta_v) + \lambda \log\left(\sum_{w=1}^{V} \alpha_v^w \| \text{sg}(\mathbf{W}_v^w) \|_2 \right)$$

where $\text{sg}(\cdot)$ denotes stop-gradient, meaning $\mathbf{W}_v$ is not updated through the shrinkage loss and only serves to guide the learning of shrinkage coefficients $\boldsymbol{\alpha}$. The shrinkage coefficients are parameterized via softmax $\boldsymbol{\alpha}_v = \text{softmax}(\boldsymbol{\beta}_v)$ to ensure non-negativity and normalization.

**Phase 2: Proximal gradient descent compression.** A $\boldsymbol{\alpha}_v$-weighted Group Lasso proximal gradient step with soft thresholding is applied to $\mathbf{W}_v$, strictly compressing unimportant columns to zero.

### Key Designs

- **Logarithmic shrinkage loss**: The $\log$ operator provides more balanced weighting across reductions in column norms of different magnitudes, empirically improving robustness to noise and the hyperparameter $\lambda$.
- **Self-reinforcing gradient dynamics**: When $\|\mathbf{W}_v^w\|_2$ is large, the corresponding $\alpha_v^w$ decreases, preserving important features during compression; conversely, unimportant features are more rapidly eliminated, forming a self-reinforcing cycle.
- **Staged training**: $\boldsymbol{\alpha}$ is not updated during the first $K=1500$ steps, allowing the prediction loss to guide the model toward a reasonable initialization.
- **No threshold required**: Unlike traditional methods that require manual threshold selection $\tau$, GC-xLSTM achieves strict sparsity directly through proximal gradient steps.
- **sLSTM over mLSTM**: sLSTM is chosen because its memory mixing is more effective for time series prediction.

### Theoretical Analysis

The paper demonstrates that the sLSTM block possesses at least the same expressive power as RNNs (universal function approximation), ensuring that the GC-xLSTM architecture can approximate the underlying generative process $g_v$ to arbitrary precision, thereby guaranteeing the sufficiency of the model class.

## Key Experimental Results

Extensive evaluation is conducted on 6 datasets.

### Lorenz-96 (Chaotic Nonlinear System)

| Model | F=10 Acc. | F=10 BA | F=40 Acc. | F=40 BA |
|-------|-----------|---------|-----------|---------|
| cMLP | 97.2 | 95.6 | 68.3 | 80.5 |
| GVAR | 98.2 | 98.2 | 94.5 | 88.5 |
| **GC-xLSTM** | **99.1** | **98.5** | **96.3** | **96.6** |

The advantage is particularly pronounced under the high-chaos setting $F=40$, where BA exceeds the second-best GVAR by 8.1 percentage points.

### fMRI Brain Connectivity

| Model | BA |
|-------|----|
| TCDF | 72.8±6.3 |
| cLSTM | 65.5±5.3 |
| **GC-xLSTM** | **73.3±3.0** |

### Ablation Study (F=40 Lorenz / fMRI)

| Configuration | Lorenz BA | fMRI BA |
|---------------|-----------|---------|
| GC-xLSTM (full) | 96.6 | 73.3 |
| Replace with standard LSTM | 93.0 | 62.8 |
| Replace with standard Group Lasso | 73.0 | 65.4 |

Both components (xLSTM architecture + joint optimization) are indispensable, with the joint optimization strategy contributing more substantially.

### Qualitative Analysis on Real-World Data

- **Molène weather**: Spatial dependencies are learned from temperature observations, uncovering local and long-range meteorological patterns without geographic priors.
- **Human motion capture**: Foot→knee→arm driving relationships are discovered in Salsa dancing; lower limbs are identified as the primary motion source in running.
- **Corporate financial indicators**: Extracted causal edges are validated by financial experts as economically meaningful.

## Highlights & Insights

1. **Elegant method design**: Dynamically learned shrinkage coefficients enforce strict sparsity without relying on thresholds, addressing a fundamental limitation of traditional Lasso methods.
2. **Unified and robust**: Aside from $\lambda$, all hyperparameters are essentially shared across six datasets, demonstrating robustness of the approach.
3. **Theory-practice balance**: Gradient dynamics analysis is intuitive and clear; ablation experiments thoroughly validate each component's contribution.
4. **Efficiency**: Training requires no more than 1.5 hours on a single GPU, with time and space complexity scaling approximately linearly with the number of variables.
5. **Interpretability on real data**: Causal relationships extracted from motion capture and weather data carry intuitive physical meaning.

## Limitations & Future Work

1. **Lack of convergence guarantees**: Despite the model class sufficiency analysis, no rigorous mathematical proof of algorithmic convergence is provided.
2. **Limited variable scale**: Experiments involve at most tens of variables; scalability to high-dimensional settings (hundreds or thousands of variables) remains unverified.
3. **$\lambda$ still requires tuning**: Although only one hyperparameter needs adjustment, different datasets require different $\lambda$ values, and AUROC is obtained by sweeping $\lambda \in \{5, \ldots, 15\}$.
4. **Stationarity assumption**: The method assumes strict stationarity of time series; applicability to non-stationary settings remains unexplored.
5. **Insufficient comparison of sLSTM selection**: The superiority of sLSTM over mLSTM is stated without detailed comparative results.

## Related Work & Insights

| Method | Nonlinear | Long-range | Strict Sparsity | No Threshold |
|--------|-----------|------------|-----------------|--------------|
| VAR (classical) | ✗ | ✗ | ✗ | ✗ |
| cMLP | ✓ | ✗ | ✗ | ✗ |
| cLSTM | ✓ | Limited | ✗ | ✗ |
| GVAR | ✓ | ✓ | ✗ | ✗ |
| GC-KAN | ✓ | ✓ | ✗ | ✗ |
| **GC-xLSTM** | ✓ | ✓ | ✓ | ✓ |

GC-xLSTM holds advantages across all dimensions, with the combination of strict sparsity and threshold-free inference being its distinctive selling point.

The following insights emerge from this work:

1. **Transferable sparse optimization paradigm**: The joint optimization framework of dynamic shrinkage coefficients and proximal gradient is not limited to GC discovery and can be generalized to any scenario requiring strict input selection (e.g., feature selection, attention sparsification).
2. **Potential of xLSTM for time series**: Compared to Transformers, sLSTM demonstrates a better computation-performance trade-off for GC discovery, suggesting unique advantages of recurrent architectures in causal tasks.
3. **Extensible to lag-specific modeling**: The paper demonstrates the ability to learn distinct projection matrices $\mathbf{W}^{(\ell)}$ per lag, which is valuable for understanding causal relationships at different temporal scales.
4. **Bridge to causal inference**: Although Granger causality does not equate to Pearl causality, the paper cites theoretical work connecting the two, leaving an interface for future research.

## Rating
- **Novelty**: 8/10 — Applying xLSTM to GC discovery is inherently novel; the dynamic sparsity optimization strategy is elegantly designed.
- **Experimental Thoroughness**: 8/10 — Six datasets covering both simulated and real-world settings with complete ablations; scalability to high dimensions is lacking.
- **Writing Quality**: 8/10 — Well-structured with coherent theory-method-experiment flow; gradient dynamics analysis is intuitive.
- **Value**: 7/10 — Valuable within the time series causal discovery domain, though the application scope is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)
- [\[NeurIPS 2025\] Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] The Human Brain as a Combinatorial Complex](the_human_brain_as_a_combinatorial_complex.md)

</div>

<!-- RELATED:END -->
