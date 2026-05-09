---
title: >-
  [Paper Note] T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation
description: >-
  [ICLR 2026][Time Series][time series imputation] This paper proposes T1, a CNN-Transformer hybrid architecture whose core innovation is Channel-Head Binding (CHead Attention): a shared depthwise convolution extracts $C$ types of temporal features (trend, periodicity, abrupt changes, etc.) for each variable, and each CNN channel is then bound one-to-one to a single attention head, enabling cross-variable information transfer to proceed independently at the feature level. When missing data prevents a channel from extracting a valid pattern, the corresponding attention head automatically down-weights, achieving adaptive missing-data handling without explicit design. On 11 benchmark datasets, the average MSE is reduced by 46%, with even larger gains under 70% extreme missingness.
tags:
  - ICLR 2026
  - Time Series
  - time series imputation
  - CNN-Transformer hybrid
  - channel-head binding
  - selective information transfer
  - missing pattern generalization
date: 2026-05-08
content_hash: ac5c882beab5c5b3
---

# T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation

**Conference**: ICLR 2026
**arXiv**: [2602.21043](https://arxiv.org/abs/2602.21043)
**Code**: [GitHub](https://github.com/Oppenheimerdinger/T1)
**Area**: Time Series / Missing Value Imputation
**Keywords**: time series imputation, CNN-Transformer hybrid, channel-head binding, selective information transfer, missing pattern generalization

## TL;DR

This paper proposes T1, a CNN-Transformer hybrid architecture whose core innovation is Channel-Head Binding (CHead Attention): a shared depthwise convolution extracts $C$ types of temporal features (trend, periodicity, abrupt changes, etc.) for each variable, and each CNN channel is then bound one-to-one to a single attention head, enabling cross-variable information transfer to proceed independently at the feature level. When missing data prevents a channel from extracting a valid pattern, the corresponding attention head automatically down-weights, achieving adaptive missing-data handling without explicit design. On 11 benchmark datasets, the average MSE is reduced by 46%, with even larger gains under 70% extreme missingness.

## Background & Motivation

**Background**: Multivariate time-series imputation must simultaneously accomplish two tasks — (1) extracting temporal patterns from sparse observations, and (2) transferring complementary information across variables to aid reconstruction. These tasks are tightly coupled: once temporal features are corrupted by missingness, cross-variable transfer amplifies errors, while naïve cross-variable transfer cannot distinguish which variables are reliable under a given missing pattern.

**Four Architectural Paradigms and Their Limitations**:

| Paradigm | Representative Methods | Temporal Modeling | Cross-Variable Modeling | Core Deficiency |
|---------|---------|---------|----------|---------|
| Time-axis tokenization | SAITS, PatchTST | ✓ Attention models long-range dependency | ✗ All variables mixed in the same token | Missing values directly corrupt token representations → pollution propagates to all computations |
| Variable-axis tokenization | iTransformer | △ Entire sequence compressed into a single token | ✓ Pure inter-variable attention | Loses feature-level selectivity; all temporal patterns are forcibly fused |
| Dual-axis tokenization | ImputeFormer, CSDI | ✓ Time-axis attention | ✓ Variable-axis attention | Missingness creates "broken paths" across both axes, making intermediate representations unreliable |
| Temporal CNN | ModernTCN, TimesNet | ✓ Multi-scale convolution for efficient extraction | △ Static pointwise mixing only | Limited cross-variable transfer capacity with no adaptability to missing patterns |

**Core Observation**: CNNs excel at extracting temporal features from sparse observations (convolutions are inherently robust to local missingness), while Transformers excel at dynamically modeling inter-variable relationships. The key question is how to "correctly interface" the two — naïve concatenation causes multi-channel CNN features to be mixed at the attention layer, allowing corruption-affected channels to contaminate reliable ones.

**Core Idea**: Establish a one-to-one binding between CNN channels and attention heads, so that each attention head handles cross-variable transfer for only one type of feature, realizing a selective information pathway at the feature level.

## Method

### Overall Architecture

T1 consists of three modules: Mask-Aware Embedding → $N$ T1 Blocks → Reconstruction Upsampler.

**Mask-Aware Embedding**: For each variable $x^{(m)}$, instance normalization is applied using only observed values (mean and variance computed only at positions where $\Omega_{m,t}=1$). The normalized sequence and the observation mask are then concatenated into a 2-channel input $[x_{\text{norm}}^{(m)}; \Omega^{(m)}] \in \mathbb{R}^{2 \times T}$, which is downsampled to $z^{(m)} \in \mathbb{R}^{C \times L}$ via a strided 1D convolution with $C$ filters, followed by addition of a learnable variable encoding $E_{\text{var}}^{(m)}$. This ensures the model is aware of missing positions from the outset.

**T1 Block**: Each block contains three components:

1. **Temporal Convolutional QKV Projection**: A depthwise convolution with **weights shared across all variables** independently extracts temporal features for each channel of each variable. Two kernel sizes are applied in parallel for multi-scale analysis, generating Q/K/V:

$$Q_{m,c} = \text{DWConv}_{\text{large},Q}(Z_{m,c}) + \text{DWConv}_{\text{small},Q}(Z_{m,c})$$

Critically, shared weights ensure that the $c$-th channel extracts the **same type** of temporal pattern (e.g., the 3rd channel always captures periodicity) across all variables, providing semantic alignment as the prerequisite for meaningful cross-variable attention per channel.

2. **CHead Attention (Core Innovation)**: Setting the number of attention heads $n_h = C$ (equal to the number of CNN channels), the $c$-th head processes only the $c$-th channel of all variables:

$$O_c = \text{Softmax}\left(\frac{Q_c K_c^T}{\sqrt{L}}\right) V_c, \quad Q_c, K_c, V_c \in \mathbb{R}^{M \times L}$$

Outputs are concatenated, followed by pointwise conv + LayerNorm + residual connection. Each information-transfer pathway thus carries only a single feature type. When the $c$-th channel of a variable fails to extract a valid pattern due to missingness, that channel produces weak features → the corresponding head naturally assigns low weight to that variable → other channels' transfer remains unaffected.

3. **Convolutional FFN**: An inverted bottleneck implemented with pointwise convolutions (rather than linear layers), enabling nonlinear inter-channel interaction while maintaining position-independent processing along the time axis. Across stacked T1 Blocks, the FFN-mixed features form new channel representations for the next layer.

**Reconstruction Upsampler**: A 1D PixelShuffle (parameter-free, rearranging from $\mathbb{R}^{M \times C \times L}$ to $\mathbb{R}^{M \times (C/r) \times (L \cdot r)}$, $r=T/L$) restores the original temporal resolution, avoiding the checkerboard artifacts of transposed convolutions. A final pointwise convolution projects to the target dimension, followed by de-normalization.

### Key Designs

1. **One-to-One Channel-Head Binding**: In conventional multi-head attention, each head processes a subspace of mixed features. T1 enforces $n_h = C$ with the $k$-th head strictly corresponding to the $k$-th CNN channel. This provides feature-level isolation — corruption-affected channels cannot contaminate others, realizing "selective information transfer."

2. **Semantic Alignment via Shared Depthwise Conv**: All variables share the same convolutional kernels, ensuring semantically consistent feature extraction within the same channel across different variables. This is the prerequisite for CHead Attention to function — only semantically aligned features make meaningful cross-variable attention possible.

3. **Mask-Aware Embedding with No Explicit Missing-Data Handling**: The model receives explicit mask information at the embedding layer, but no explicit missing-data handling (e.g., masked attention) is applied in subsequent processing — the model relies entirely on CHead Attention's adaptive down-weighting mechanism. This "implicit handling" is more robust because it makes no prior assumptions about the missing pattern.

## Key Experimental Results

### Point Missingness Scenario: 9 Benchmark Datasets (Average over 4 Missing Rates)

| Dataset | T1 (MSE) | PatchTST | ModernTCN | iTransformer | TimeMixer++ | ImputeFormer | SAITS |
|-------|----------|----------|-----------|-------------|------------|-------------|-------|
| ETTh1 | **0.049** | 0.082 | 0.083 | 0.129 | 0.132 | 0.223 | 0.092 |
| ETTh2 | **0.036** | 0.049 | 0.051 | 0.064 | 0.068 | 0.429 | 0.275 |
| ETTm1 | **0.022** | 0.038 | 0.040 | 0.063 | 0.052 | 0.086 | 0.051 |
| ETTm2 | **0.017** | 0.024 | 0.026 | 0.032 | 0.030 | 0.151 | 0.103 |
| Weather | **0.029** | 0.037 | 0.038 | 0.090 | 0.034 | 0.042 | 0.034 |
| PEMS03 | **0.021** | 0.038 | 0.056 | 0.048 | 0.044 | 0.080 | 0.060 |
| Exchange | **0.002** | 0.003 | 0.009 | 0.004 | 0.002 | 0.031 | 0.180 |
| Illness | **0.038** | 0.130 | 0.260 | 0.205 | 0.238 | 0.636 | 0.614 |
| Electricity | **0.043** | 0.089 | 0.121 | 0.090 | 0.071 | 0.076 | 0.152 |
| **Average** | **0.027** | 0.050 | 0.070 | 0.079 | 0.075 | 0.210 | 0.176 |

T1 achieves an average MSE of 0.027, a **46%** reduction over the second-best PatchTST (0.050) and a **56%** reduction over the dedicated imputer PSW-I (0.062).

### Robustness across Missing Rates (Average over 9 Datasets)

| Test Missing Rate | T1 | PatchTST | ModernTCN | iTransformer | PSW-I |
|-----------|-----|----------|-----------|-------------|-------|
| 10% | **0.017** | 0.040 | 0.063 | 0.057 | 0.048 |
| 30% | **0.021** | 0.038 | 0.048 | 0.061 | 0.058 |
| 50% | **0.027** | 0.048 | 0.059 | 0.076 | 0.068 |
| 70% | **0.049** | 0.092 | 0.135 | 0.128 | 0.093 |

Under 70% extreme missingness, T1's MSE (0.049) is half that of PatchTST (0.092), demonstrating that CHead Attention's selective mechanism provides the greatest benefit at high missing rates. The model is trained with a 40% missing rate and generalizes directly to other rates without retraining.

### Block Missingness Scenario (Simulating Sensor Failure)

Testing combines 5% point missingness with 0.15%-probability contiguous block missing of 24–96 steps. T1 achieves an average MSE of 0.026, a **48%** reduction over PatchTST (0.050). The advantage is largest on the Illness dataset: T1=0.037 vs. PatchTST=0.125.

### Naturally Missing Datasets

- **PhysioNet2012** (ICU data, ~80% inherent missingness + additional artificial missingness): T1 average MSE 0.075, a **23%** reduction over the second-best DLinear (0.097). Reasonable performance (MSE=0.106) is maintained even at a total missing rate of 94%.
- **AQI36** (air quality, 15–30% natural missingness): T1 MSE 0.226, a **13%** reduction over PatchTST (0.262).

### Ablation Study

| Component | Alternative | Avg. MSE | Performance Drop |
|------|---------|--------|---------|
| **T1 (full model)** | — | **0.033** | — |
| Cross-variable mechanism | Pointwise Conv replacing attention | 0.037 | +12.91% |
| Cross-variable mechanism | Fully removed | 0.051 | +56.16% |
| Channel-head binding | 8 channels/head | 0.035 | +7.45% |
| Channel-head binding | 16 channels/head | 0.038 | +16.86% |
| Channel-head binding | 32 channels/head | 0.037 | +14.57% |
| Embedding | Mask channel removed | 0.034 | +3.64% |
| Reconstruction | Linear upsampling replacing PixelShuffle | 0.034 | +3.19% |

Key findings: (1) Removing cross-variable modeling entirely causes a 56% performance drop → cross-variable information is critical for imputation; (2) Replacing attention with static convolution still incurs a 13% drop → dynamic selectivity matters more than fixed patterns; (3) One-to-one binding (128 channels/head) substantially outperforms 8/16/32-channel groupings → feature-level granularity of isolation is the key; (4) Notably, 16 channels/head performs worse than 32 channels/head, revealing a non-monotonic relationship — overly coarse grouping introduces harmful feature mixing.

### Representation Analysis

**Layer-wise Missing Response**: On ETTh1, with all other variables fixed at 40% missingness, the target variable's missing rate is varied from 10% to 70%. First-layer attention weights decrease by 46% (0.195→0.105), while the last layer decreases by only 6% (0.165→0.155). This indicates that shallow layers perform "coarse reconstruction," enabling deeper layers to access more complete information.

**Channel-level Pattern Dependence**: For a target variable, peak vs. non-peak regions and high-variance vs. low-variance regions are separately masked (each at 30%). Different masking patterns produce markedly different attention responses — masking high-variance regions reduces attention by 10.4%, while masking low-variance regions reduces it by 7.5%. This confirms that CHead Attention's modulation depends on **which temporal patterns remain observable**, rather than on the simple missing ratio.

## Highlights & Insights

- **"Channel-head binding" is an elegant interface connecting CNN and Transformer**: Conventional approaches extract CNN features → concatenate/add → feed into Transformer, where features are mixed at the attention layer. By enforcing $n_h=C$, T1 turns each attention head into a "clean information conduit" transmitting only one type of feature. This design adds virtually no overhead yet delivers a qualitative improvement.

- **The philosophy of "no explicit missing-data handling"**: No masked attention, no special treatment of missing positions, no conditioning on missing rate. The architecture itself naturally handles missingness — CNN channels produce weak features for missing regions, and attention down-weights accordingly. This "structural solution" is more elegant and more robust than explicit handling.

- **A 46% MSE reduction is exceptionally rare on a mature problem**. Time-series imputation has attracted many methods; such a large improvement suggests that prior methods harbored fundamental architectural compromises — either strong temporal modeling but weak cross-variable capacity (CNN), or strong cross-variable modeling but temporally corrupted representations (Transformer). T1 identifies the correct division of labor.

- **Practical value of unified hyperparameters**: The same configuration is used across all 11 datasets with no per-dataset tuning, dramatically reducing deployment overhead and suggesting that the architecture's inductive biases are sufficiently well-calibrated.

## Limitations & Future Work

- The sequence length is fixed at 96; scalability to long sequences (e.g., 1000+ steps) is not validated, and PixelShuffle's upsampling ratio $r=T/L$ may require adjustment for very long sequences.
- Training employs simple random-mask self-supervision; more advanced masking strategies (e.g., curriculum learning with progressively increasing missing rates) are unexplored.
- CHead Attention's attention matrix has size $M \times M$ (variables × variables); sparsification may be needed when the number of variables is very large (e.g., thousands of sensors).
- No in-depth comparison with diffusion-based models (CSDI/SSSD) on generation quality and diversity — diffusion models can produce multiple plausible imputations, whereas T1 provides only a point estimate.

## Related Work & Insights

- **vs. ModernTCN**: T1 directly adopts ModernTCN's DWConv design for temporal feature extraction but replaces its static pointwise mixing with dynamic CHead Attention → this substitution accounts for the majority of the 56% performance gain.
- **vs. iTransformer**: Both use variable-axis attention, but iTransformer compresses the entire sequence into a single token, losing feature-level selectivity. T1 maintains $C$ independent information channels through CHead Attention.
- **vs. ImputeFormer**: Dual-axis attention is theoretically comprehensive, but missingness creates "information breaks" across both axes. T1 avoids this by having CNN first perform robust feature extraction along the time axis.
- **Broader Inspiration**: The channel-head binding idea may generalize to other scenarios requiring "reliability-aware cross-dimensional information transfer," such as multi-sensor fusion or multimodal learning with modality dropout.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ CHead Attention is conceptually elegant; the one-to-one binding constraint is simple yet qualitatively impactful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 datasets × 3 missing scenarios × 4 missing rates + natural missingness + ablations + representation analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivational chain is complete, the four-paradigm comparison is intuitive, and the ablation design is insightful.
- Value: ⭐⭐⭐⭐⭐ The 46% MSE reduction is a breakthrough contribution to time-series imputation; unified hyperparameters offer strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [\[CVPR 2026\] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](../../CVPR2026/time_series/a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)
- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](../../NeurIPS2025/time_series/channel_matters_estimating_channel_influence_for_multivariate_time_series.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)

<!-- RELATED:END -->
