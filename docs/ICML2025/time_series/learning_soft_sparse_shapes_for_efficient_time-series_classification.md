---
title: >-
  [Paper Note] Learning Soft Sparse Shapes for Efficient Time-Series Classification
description: >-
  [ICML2025][Time Series][shapelet] The SoftShape model is proposed to replace the traditional hard filtering of shapelets with soft sparsification based on contribution scores. Combining MoE-driven intra-shape and shared-expert-driven inter-shape dual-mode temporal pattern learning, it achieves SOTA classification accuracy on 128 UCR datasets.
tags:
  - "ICML2025"
  - "Time Series"
  - "shapelet"
  - "time-series classification"
  - "soft sparsification"
  - "Mixture-of-Experts"
  - "interpretability"
date: 2026-05-08
content_hash: 79af9fb8075de233
---

# Learning Soft Sparse Shapes for Efficient Time-Series Classification

**Conference**: ICML2025  
**arXiv**: [2505.06892](https://arxiv.org/abs/2505.06892)  
**Code**: [GitHub](https://github.com/qianlima-lab/SoftShape)  
**Area**: Time Series  
**Keywords**: shapelet, time-series classification, soft sparsification, Mixture-of-Experts, interpretability

## TL;DR
The SoftShape model is proposed to replace the traditional hard filtering of shapelets with soft sparsification based on contribution scores. Combining MoE-driven intra-shape and shared-expert-driven inter-shape dual-mode temporal pattern learning, it achieves SOTA classification accuracy on 128 UCR datasets.

## Background & Motivation

Shapelets (discriminative subsequences) represent a classic paradigm in time-series classification (TSC) that offers both high accuracy and interpretability. However:

**Computational bottleneck**: Brute-force searching through all candidate subsequences is extremely costly, requiring evaluation of subsequences at varied positions and lengths.

**Information loss from hard filtering**: Existing methods (such as shapelet transform) discard a large number of subsequences in a hard manner, potentially missing temporal patterns beneficial for classification.

**Neglected differences in contribution**: Traditional methods fail to distinguish the degree of contribution of different shapelets to the classification, leading to the direct abandonment of "minor but useful" subsequences.

The core motivation of this study is: can all subsequence information be preserved in a **soft manner** while reducing computational overhead through sparsification?

## Method

### Overall Architecture

SoftShape consists of three core modules: Shape Embedding → Soft Shape Sparsification → Soft Shape Learning Block, stacked into $L$ layers and followed by a linear classifier.

### 1. Shape Embedding

A 1D CNN is utilized to transform the input time series $\mathcal{X}_n = \{x_1, \ldots, x_T\}$ into $M = \frac{T-m}{q}+1$ overlapping subsequence embeddings:

$$\hat{\mathcal{S}}_{n,p}^{m} = \sum_{i=0}^{m-1} \mathbf{W}_i \mathcal{S}_{n,p}^{m}, \quad p = 0, q, 2q, \ldots, T-m$$

where $m$ denotes the window size, $q$ denotes the stride, and learnable positional encodings are incorporated to capture temporal dependencies.

### 2. Soft Shape Sparsification

**Key Innovation**: A gated attention mechanism is used to compute a classification contribution score for each shape embedding:

$$\alpha(\hat{\mathcal{S}}_{n,p}^{m}) = \sigma\left(\mathbf{W}_2 \tanh(\mathbf{W}_1 \hat{\mathcal{S}}_{n,p}^{m} + \mathbf{b}_1) + \mathbf{b}_2\right)$$

- **High-scoring shapes** (top-$\eta$ ratio): multiplied by the contribution score to obtain a soft representation $\tilde{\mathcal{S}}_{n,p}^{m} = \alpha \cdot \hat{\mathcal{S}}_{n,p}^{m}$
- **Low-scoring shapes** (the remaining part): weight-fused into a single embedding $\tilde{\mathcal{S}}_{\text{fused}}^{m} = \sum_{p \in \mathcal{E}} \alpha \cdot \hat{\mathcal{S}}_{n,p}^{m}$

This retains information from all subsequences (as opposed to hard discarding) while reducing the input scale for subsequent modules.

### 3. Soft Shape Learning Block

#### 3.1 Intra-Shape Learning — MoE

An MoE router is utilized to activate a small number of class-specific expert networks, learning local temporal patterns for each soft shape embedding:

$$G(\tilde{\mathcal{S}}_{n,p}^{m}) = \text{TOP}_k\left(\text{softmax}(\mathbf{W}_t \tilde{\mathcal{S}}_{n,p}^{m})\right)$$

Each expert is a lightweight MLP, with the output formulated as:

$$h_e(\theta, \tilde{\mathcal{S}}_{n,p}^{m}) = \hat{G}_e(\tilde{\mathcal{S}}_{n,p}^{m}) \cdot \text{GeLU}(\mathbf{W}_e \tilde{\mathcal{S}}_{n,p}^{m} + \mathbf{b}_e)$$

An importance loss $\mathcal{L}_{\text{imp}}$ and a load-balancing loss $\mathcal{L}_{\text{load}}$ are applied to mitigate the expert imbalance issue.

#### 3.2 Inter-Shape Learning — Shared Expert

The sparsified shape embeddings are transposed into a sequence $\mathcal{Q}_n^m = (\tilde{\mathcal{S}}_n^m)^T$ and fed into a shared expert based on an Inception module (comprising three 1D CNNs with different kernel sizes) to learn global temporal patterns across different shapes.

### 4. Loss & Training

The total loss function is a weighted sum of the classification cross-entropy loss and the expert balancing loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \lambda(\mathcal{L}_{\text{imp}} + \mathcal{L}_{\text{load}})$$

Conjunctive pooling is employed to aggregate the final predictions.

## Key Experimental Results

### Main Results: 128 UCR Datasets

| Method | Avg. Acc | Avg. Rank | Win |
|------|----------|-----------|-----|
| InceptionTime | 0.9181 | 4.05 | 29 |
| TSLANet | 0.9205 | 3.68 | 31 |
| MR-H (Non-DL) | 0.8972 | 5.51 | 29 |
| RDST (Non-DL) | 0.8897 | 6.41 | 23 |
| **SoftShape** | **0.9334** | **2.72** | **53** |

- SoftShape achieves an average accuracy of 93.34%, outperforming the runner-up TSLANet by 1.3 percentage points.
- It achieves the best performance on 53/128 datasets, far exceeding other methods.
- All P-values are < 0.05, indicating statistically significant differences.

### Ablation Study

| Variant | Avg. Acc | Avg. Rank |
|------|----------|-----------|
| w/o Soft Sparse | 0.9123 | 3.04 |
| w/o Intra | 0.9245 | 2.75 |
| w/o Inter | 0.9022 | 3.74 |
| w/o Intra & Inter | 0.8696 | 5.02 |
| **SoftShape** | **0.9334** | **2.04** |

- The inter-shape module provides the highest contribution (accuracy drops by 3.1% upon its removal).
- Each module is indispensable.

### Sparsity Rate Analysis (18 UCR Datasets)

| Sparsity Rate | Avg. Acc | P-value |
|--------|----------|---------|
| 0% | 0.9461 | - |
| 10% | 0.9469 | 0.297 |
| 50% | 0.9453 | 0.346 |
| 70% | 0.9323 | 0.009 |
| 90% | 0.9261 | 0.0004 |

- No significant degradation in accuracy is observed when the sparsity rate is $\le 50\%$ ($P > 0.05$); the default is set to 50% to balance performance and efficiency.

## Highlights & Insights

1. **Elegant Soft Sparsification**: Instead of discarding any subsequences, low-contribution shapes are fused into a single representation, which reduces computational cost while preserving information integrity.
2. **Complementary Dual-Path Learning**: MoE captures class-specific local patterns, whereas the Shared Expert captures global cross-shape patterns, combining local and global perspectives.
3. **Intrinsic Interpretability**: The attention score $\alpha$ inherently provides a visual metric of each subsequence's contribution to classification, eliminating the need for auxiliary explanation modules.
4. **Rigorous Evaluation**: The study presents comprehensive experiments on 128 UCR datasets compared against 19 baseline models, accompanied by complete ablation studies and statistical significance tests.
5. **Lossless Performance at 50% Sparsity**: This demonstrates that while a vast number of subsequences are indeed redundant, they should not be discarded in a hard manner.

## Limitations & Future Work

1. **Limited to Univariate Time Series**: The model has not been validated in multivariate scenarios, which are more common in practical applications.
2. **Insufficient Efficiency Analysis**: Despite claiming efficiency gains, there is a lack of detailed comparisons regarding FLOPs and inference speed.
3. **Datasets Concentrated on UCR**: The UCR datasets are of limited length and complexity, lacking validation on long-sequence or large-scale datasets.
4. **Number of MoE Experts Equals Class Count**: When there are a massive number of classes, the number of experts may become prohibitively large.
5. **Manual Configuration of Sparsity Rate $\eta$**: Adaptive sparsity rate strategies have not been explored.

## Related Work & Insights

- **Shapelet Domain**: The transition from hard filtering (Ye & Keogh 2009) to shapelet transform (Hills 2014) and then to the proposed soft sparsification represents a natural evolution of this research direction.
- **MoE in Time Series**: Unlike forecasting tasks that employ MoE to handle multi-domain mixed data, this work applies MoE for class-specific pattern learning at the shape level.
- **Patch Tokenization**: The model borrows the patch concept from PatchTST but redefines patches as shapes and introduces contribution-based weighting.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of soft sparsification and MoE-shape dual-path learning is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Very comprehensive evaluation with 128 datasets, 19 baselines, ablation studies, and statistical tests.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous mathematical derivations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Establishes a new paradigm for shapelet-based methods, though its restriction to univariate scenarios limits its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](../../NeurIPS2025/time_series/maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[ICML 2025\] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)
- [\[ACL 2025\] LETS-C: Leveraging Text Embedding for Time Series Classification](../../ACL2025/time_series/lets-c_leveraging_text_embedding_for_time_series_classification.md)
- [\[ICML 2025\] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)

</div>

<!-- RELATED:END -->
