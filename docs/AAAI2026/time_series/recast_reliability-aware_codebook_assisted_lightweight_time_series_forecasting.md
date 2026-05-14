---
title: >-
  [Paper Note] ReCast: Reliability-aware Codebook Assisted Lightweight Time Series Forecasting
description: >-
  [AAAI 2026][Time Series][Codebook quantization] This paper proposes ReCast, which encodes time series into discrete embeddings via patch-level vector quantization. It introduces a dual-path architecture consisting of a q…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Codebook quantization"
  - "lightweight forecasting"
  - "dual-path architecture"
  - "reliability-aware update"
  - "distributionally robust optimization"
date: 2026-05-08
content_hash: 7346ae4a855c289b
---

# ReCast: Reliability-aware Codebook Assisted Lightweight Time Series Forecasting

**Conference**: AAAI 2026
**arXiv**: [2511.11991](https://arxiv.org/abs/2511.11991)
**Code**: N/A
**Area**: Time Series
**Keywords**: Codebook quantization, lightweight forecasting, dual-path architecture, reliability-aware update, distributionally robust optimization

## TL;DR

This paper proposes ReCast, which encodes time series into discrete embeddings via patch-level vector quantization. It introduces a dual-path architecture consisting of a quantization path (modeling regular structures) and a residual path (capturing irregular fluctuations), along with a reliability-aware codebook update strategy based on distributionally robust optimization (DRO). ReCast achieves state-of-the-art accuracy with a lightweight architecture across 8 datasets.

## Background & Motivation

Mainstream time series forecasting methods typically adopt global decomposition strategies, decomposing sequences into trend, seasonal, and residual components for independent modeling. However:

**Limitations of global decomposition**: Real-world time series are often dominated by complex, dynamic local patterns rather than clear global regularities. Global decomposition performs poorly on noisy and non-periodic data.

**Model complexity**: Complex Transformer/CNN models incur substantial computational overhead, limiting their applicability in real-time systems and resource-constrained environments.

**Recurrence of local patterns**: Many real-world sequences exhibit "local shape recurrence" (e.g., daily electricity consumption curves sharing similar shapes with varying details), motivating the use of discrete codebooks to capture such recurring patterns.

**Core motivation**: Can vector quantization (VQ) encode local patterns into a finite set of codewords, enabling lightweight forecasting through codeword modeling? The key challenge is that static codebooks cannot adapt to non-stationary data, and handling noise and distribution shift during codebook updates is non-trivial.

## Method

### Overall Architecture

ReCast comprises three main modules:
1. **Patch-wise quantization**: normalization → patching → downsampling → nearest-neighbor codebook matching → discrete embeddings
2. **Dual-path forecasting**: quantization path (MLP predicting future codeword indices) + residual path (MLP predicting quantization residuals)
3. **Codebook construction and update**: clustering to generate a pseudo-codebook → reliability-aware scoring → DRO fusion → incremental update

### Key Designs

1. **Patch-wise vector quantization and downsampling**: The input sequence $\mathbf{X} \in \mathbb{R}^{C \times L}$ is first instance-normalized, then split into $N = \lceil L/L_p \rceil$ patches $\mathbf{p}_i \in \mathbb{R}^{L_p}$. Each patch is downsampled to $L_p/2$ dimensions and matched to the nearest entry in a learnable codebook $\mathbf{S} = \{\mathbf{s}_k\}_{k=1}^K$:

    $q_i = \arg\min_{\mathbf{s}_k \in \mathbf{S}} \|\tilde{\mathbf{p}}_i - \mathbf{s}_k\|_2^2$

   **Design Motivation**:
    - Downsampling: Based on the assumption of "scale invariance of local patterns," low-resolution representations preserve salient structures while suppressing redundant fluctuations, significantly reducing the computational cost of codebook matching and storage.
    - Shared codebook: All variables share a single codebook, implicitly encouraging cross-variable interaction and avoiding the performance bottleneck of channel-independent architectures.
    - Random patch sampling: Only a randomly sampled subset of patches is used during training and codebook updates, reducing the risk of overfitting.

2. **Dual-path forecasting architecture**:

   **Quantization path**: A lightweight MLP $\mathcal{M}_{\text{quant}}$ predicts the discrete indices of future patches $\mathbf{Q}_y = \mathcal{M}_{\text{quant}}(\mathbf{Q}_x)$, and the future sequence $\mathbf{Y}_q$ is reconstructed via codebook lookup and upsampling.

   **Residual path**: Quantization inevitably discards fine-grained details. The residual $\mathbf{X}_r = \mathbf{X} - \mathbf{X}_q$ is computed as the difference between the input $\mathbf{X}$ and its quantized reconstruction $\mathbf{X}_q = \text{Rec}(\mathbf{Q}_x|\mathbf{S})$, and a separate MLP $\mathcal{M}_{\text{res}}$ predicts the future residual $\mathbf{Y}_r$.

   Final prediction: $\hat{\mathbf{Y}} = \sigma_{in}(\mathbf{Y}_q + \mathbf{Y}_r) + \mu_{in}$ (with instance denormalization applied)

   **Design Motivation**: The quantization path focuses on efficiently modeling stable, recurring local patterns (e.g., typical daily electricity curves), while the residual path recovers the irregular fluctuations discarded by quantization (e.g., sudden consumption spikes). The two paths jointly achieve an optimal balance between lightweight design and forecasting accuracy.

3. **Reliability-aware codebook update**: At each epoch, a pseudo-codebook $\hat{\mathbf{S}}^t$ is generated via clustering, and the actual codebook is incrementally updated:

    $\mathbf{S}^t = \mathbf{S}^{t-1} + \frac{1}{t}(\hat{\mathbf{W}}^t \hat{\mathbf{S}}^t - \mathbf{S}^{t-1})$

   The core lies in the computation of update weights $\hat{\mathbf{W}}^t$, which fuse three complementary reliability factors:

    - **Representation quality $w_{rep}$**: Evaluates how accurately each pseudo-codeword reconstructs its assigned patches; higher quality yields higher weight.
    - **Historical consistency $w_\Delta$**: Measures the deviation of pseudo-codewords from those of the previous epoch; larger deviation indicates that the old codebook is insufficient to fit new data and warrants a higher update weight.
    - **OOD sensitivity $w_{je}$**: Detects infrequently assigned codewords based on a joint energy function, preventing the embedding space from collapsing onto a small number of fixed codewords.

   The three factors are fused via **distributionally robust optimization (DRO)**—solving for the worst-case expectation within a KL neighborhood around the uniform distribution, which admits a closed-form solution:

    $\hat{w}_k^t = -\gamma \cdot \log \sum_{i=1}^{3} \exp(-z_{k,i}^t / \gamma)$

   This is a soft-minimum operation that allows the most reliable factor to dominate while softly penalizing the others.

   **Design Motivation**: The three factors are complementary and exhibit varying degrees of reliability under different data conditions. Fixed-weight fusion is prone to failure when a particular factor is noisy. DRO provides a conservative yet robust fusion scheme that yields reasonable reliability estimates even under worst-case conditions.

   Additionally, an embedding regularization loss prevents codeword collapse: $\mathcal{L}_{sep} = \log \sum_{i,j} \exp(-\|\hat{\mathbf{s}}_i^t - \hat{\mathbf{s}}_j^t\|_2^2 / \tau)$

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{pre} + w_{sep} \mathcal{L}_{sep}$$

- $\mathcal{L}_{pre} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_1$ (L1 loss, more robust to outliers)
- The codebook is fixed at inference time; efficient prediction requires only computing Eq. (5).
- Implementation: PyTorch, Nvidia L40 GPU (48GB)

## Key Experimental Results

### Main Results

Comparison with 7 state-of-the-art models on 8 datasets, averaged over 4 forecast horizons $H \in \{96, 192, 336, 720\}$:

| Model | ETTm1 MSE | ETTh1 MSE | ECL MSE | Traffic MSE | Weather MSE | 1st Count |
|------|-----------|-----------|---------|-------------|-------------|-----------|
| **ReCast** | **0.371** | **0.437** | **0.163** | **0.418** | **0.229** | **12/16** |
| PatchMLP | 0.374 | 0.438 | 0.171 | 0.417 | 0.231 | 2 |
| TQNet | 0.377 | 0.441 | 0.164 | 0.445 | 0.242 | 2 |
| CycleNet | 0.379 | 0.457 | 0.168 | 0.472 | 0.243 | 0 |
| iTransformer | 0.407 | 0.454 | 0.178 | 0.428 | 0.258 | 0 |
| PatchTST | 0.387 | 0.469 | 0.216 | 0.555 | 0.259 | 0 |
| DLinear | 0.403 | 0.456 | 0.212 | 0.625 | 0.265 | 0 |

ReCast achieves the best performance on 12 out of 16 MSE/MAE metrics.

### Ablation Study

| Configuration | ETTm1 MSE | Traffic MSE | Weather MSE | Note |
|------|-----------|-------------|-------------|------|
| ReCast (full) | **0.371** | **0.418** | **0.229** | All modules |
| −Residual | 0.377 | 0.435 | 0.248 | Removing residual path; quantization loss uncompensated |
| −Updating | 0.400 | 0.553 | 0.257 | Frozen codebook; **largest degradation** |
| −Random | 0.377 | 0.427 | 0.240 | Removing downsampling and random sampling |
| −Scoring | 0.385 | 0.441 | 0.249 | Removing reliability-aware weights |
| −DRO | 0.375 | 0.424 | 0.237 | Uniform weights replacing DRO fusion |

Transferability experiments:

| Baseline | Original MSE | +ReCast MSE | Dataset |
|------|----------|-------------|--------|
| iTransformer | 0.407 | 0.375 | ETTm1 |
| TimesNet | 0.620 | 0.499 | Traffic |
| iTransformer | 0.258 | 0.231 | Weather |

### Key Findings

1. **Codebook update is the performance core**: The −Updating variant raises MSE from 0.418 to 0.553 (+32%) on Traffic, demonstrating that a static codebook cannot adapt to distributional shifts.
2. **Dual-path complementarity is essential**: The −Residual variant consistently degrades performance, validating that quantization inevitably loses information and residual compensation is critical.
3. **DRO outperforms simple fusion**: The gap between −DRO and −Scoring demonstrates that adaptive weight allocation is superior to uniform or equal weighting.
4. **Architecture is transferable**: Applying ReCast's codebook and dual-path framework to iTransformer and TimesNet yields consistent improvements, validating the generality of the framework.
5. **Efficiency advantage is significant**: ReCast ranks among the top-tier models in both parameter count and training speed.

## Highlights & Insights

- **Paradigm innovation from a local-pattern perspective**: Rather than performing global trend/seasonal decomposition, ReCast uses discrete codebooks to capture recurring local shapes—particularly effective for data with strong local patterns but no clear global regularities.
- **Elegant application of DRO to codebook updates**: The problem of fusing multiple reliability indicators is formalized as a DRO problem with a closed-form solution, which is both theoretically principled and computationally efficient.
- **Division of labor between quantization and residual paths**: Analogous to lossy compression followed by residual coding in image coding, this approach is applied systematically to time series forecasting for the first time.
- **Shared codebook enables implicit cross-variable interaction**: This avoids the limitations of channel-independent architectures without requiring explicit cross-variable dependency modeling.

## Limitations & Future Work

1. **Hyperparameter sensitivity**: The number of codewords $K$ and patch length $L_p$ significantly affect performance and require empirical tuning, lacking adaptive or theoretically grounded selection criteria.
2. **Fixed codebook capacity**: $K$ is determined prior to training and cannot expand dynamically based on data complexity.
3. **Quantization path predicts discrete indices**: This is inherently a classification task rather than regression, and misclassification may lead to severe prediction errors.
4. **Potential directions**: Extending ReCast to a pre-trained foundation model using richer codebooks, diverse patch configurations, and heterogeneous time series for pre-training.

## Related Work & Insights

- **VQ-VAE family**: Vector quantization was first successfully applied to image and speech generation; this paper introduces it into time series forecasting and addresses the dynamic update problem.
- **HDT (AAAI 2025)**: Also employs hierarchical discrete Transformers for time series forecasting, but ReCast's dual-path design and reliability-aware update are more lightweight and robust.
- **PatchTST/PatchMLP**: Patch-based strategies are widely adopted in time series, but this paper augments patching with quantization and discrete embeddings.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The combination of codebook quantization, dual-path architecture, and DRO-based reliability-aware updates is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation and transferability evaluations across 8 datasets, though evaluation in online/streaming settings is absent.
- Writing Quality: ⭐⭐⭐⭐ — Method is clearly described, derivations are complete, and illustrations are intuitive.
- Value: ⭐⭐⭐⭐⭐ — Lightweight, accurate, and transferable; high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[AAAI 2026\] XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](../../NeurIPS2025/time_series/sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[AAAI 2026\] Task-Aware Retrieval Augmentation for Dynamic Recommendation](task-aware_retrieval_augmentation_for_dynamic_recommendation.md)
- [\[AAAI 2026\] Harmonic Dataset Distillation for Time Series Forecasting](harmonic_dataset_distillation_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
