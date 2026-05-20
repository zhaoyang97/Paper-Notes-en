---
title: >-
  [Paper Note] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework
description: >-
  [ICLR 2026][Time Series][multivariate time series] This paper proposes ChannelTokenFormer (CTF), a unified Transformer framework that simultaneously addresses three core challenges in real-world multivariate time series…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "multivariate time series"
  - "asynchronous sampling"
  - "block-wise missingness"
  - "channel dependency"
  - "ChannelTokenFormer"
date: 2026-05-08
content_hash: f87faf8a7a93cea8
---

# Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework

**Conference**: ICLR 2026
**arXiv**: [2506.08660](https://arxiv.org/abs/2506.08660)  
**Code**: Available  
**Area**: Time Series / Robust Forecasting
**Keywords**: multivariate time series, asynchronous sampling, block-wise missingness, channel dependency, ChannelTokenFormer

## TL;DR

This paper proposes ChannelTokenFormer (CTF), a unified Transformer framework that simultaneously addresses three core challenges in real-world multivariate time series forecasting: (1) complex inter-channel dependencies — via channel token cross-channel attention; (2) asynchronous sampling across channels — via frequency-domain dynamic patching that preserves original resolution; (3) block-wise missingness at test time — via patch masking during training and direct removal of fully-missing patches at inference. CTF achieves comprehensive state-of-the-art results across six datasets including ETT, SolarWind, Weather, EPA, and CHS.

## Background & Motivation

**Background**: Multivariate time series forecasting is a core task in industrial monitoring, energy systems, and healthcare. Most existing models assume synchronous sampling and complete observations, which is severely misaligned with real-world data characteristics.

**Limitations of Prior Work**:
- **Channel-dependent vs. channel-independent**: CI designs (e.g., PatchTST) are robust but discard cross-channel information; CD designs (e.g., CrossGNN) exploit correlations but are sensitive to distribution shift — a fundamental trade-off.
- **Asynchronous sampling is pervasive**: Different sensors have different physical characteristics and thus different sampling periods (e.g., temperature at 1 hour, pressure at 15 minutes). Most methods assume synchronous alignment, and interpolation introduces signal distortion.
- **Block-wise missingness**: Sensor failures and communication outages cause long consecutive gaps. Naïve interpolation is unreliable on dynamic signals and requires cross-channel inference.
- **No existing method addresses all three simultaneously**: CD methods ignore asynchrony and missingness; CI methods lose dependencies; irregular-time methods do not handle block-wise missingness.

**Key Challenge**: In real-world scenarios, all three challenges co-exist and are mutually coupled; methods designed to address each challenge individually do not compose well.

**Goal**: Design a unified architecture that simultaneously handles channel dependencies, asynchronous sampling, and block-wise missingness without requiring interpolation preprocessing.

**Key Insight**: A channel token serves as a compact channel-level representation that can naturally aggregate local token sequences of varying lengths (handling asynchrony), interact across channels (capturing dependencies), and skip missing patches (handling missingness) — one design resolves three problems.

**Core Idea**: Redefine the channel token from a simple channel summary token into a global attention anchor that simultaneously serves as an asynchrony normalizer, missingness handler, and dependency relay.

## Method

### Overall Architecture

Input multivariate time series → per-channel FFT to detect dominant frequency and determine patch length → non-overlapping patching yields a variable number of local tokens per channel plus a learnable channel token → unified sequence processed via mask-guided self-attention → only channel tokens are fed to the decoder for prediction. Channels sharing the same sampling period share projection layers; the decoder is similarly shared by sampling period.

### Key Designs

1. **Frequency-Domain Dynamic Patching and Tokenization**:
    - **Function**: Adaptively determines patch length for each channel based on its frequency characteristics, handling asynchronous sampling.
    - **Mechanism**: FFT is applied per channel to estimate the dominant period $T_i$, which is used as the patch length for non-overlapping segmentation. A channel with sampling period $s_i$ yields $L_i = \lfloor L/s_i \rfloor$ valid samples over input length $L$. The loss is a channel-aggregated MSE: $\mathcal{L}_\text{total} = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{H_i}\sum_{j=1}^{H_i}(y_j^{(i)} - \hat{y}_j^{(i)})^2$, where $H_i = \lfloor H/s_i \rfloor$ is the number of prediction steps for channel $i$.
    - **Design Motivation**: Preserving the original sampling resolution avoids up/downsampling and eliminates spurious interpolated data. Different channels may produce different numbers of local tokens, which are uniformly aggregated by their channel token.

2. **Mask-Guided Unified Attention**:
    - **Function**: Unifies intra-channel temporal modeling and cross-channel dependency capture within a single attention operation via a carefully designed attention mask.
    - **Mechanism**: All local tokens and channel tokens are concatenated into a unified sequence $\mathbf{X} = [\mathbf{T}^{(1)};\mathbf{C}^{(1)};\dots;\mathbf{T}^{(N)};\mathbf{C}^{(N)}] \in \mathbb{R}^{\mathcal{T} \times d}$, and masked self-attention is applied: $\mathbf{X}_\text{out} = \mathbf{X} + \text{softmax}(\frac{QK^\top}{\sqrt{d}} + \mathbf{M})V$. The mask $\mathbf{M}$ enforces three rules: (1) local tokens can only attend to other local tokens within the same channel (intra-temporal); (2) channel tokens can attend to local tokens of their own channel and to channel tokens of other channels (aggregation + cross-channel interaction); (3) channel tokens do not self-attend (preventing self-reinforcement).
    - **Design Motivation**: Read-write separation — channel tokens act as read-only aggregators, and local tokens cannot attend to channel tokens, preventing information leakage. This structure makes channel tokens serve as inter-channel information relays.

3. **Patch Masking During Training (Simulating Test-Time Missingness)**:
    - **Function**: Randomly masks patch subsets of channels during training as a proxy training strategy for block-wise missingness at test time.
    - **Mechanism**: Inspired by PatchDropout, a subset of patches per channel is randomly removed during training (patches that are entirely zero have their corresponding local tokens dropped), and attention naturally skips those positions. At test time, real missing blocks cause fully-absent patches to be removed, and the model infers missing channels from the available channel tokens.
    - **Design Motivation**: Conventional approaches use zero-filling or interpolation, which propagates invalid signals. This method simply omits missing patches, introducing no erroneous information, while also serving as implicit regularization against overfitting.

### Loss & Training

CMSE (channel-aggregated MSE) and CMAE are used as evaluation metrics, with errors computed only at valid sampling points per channel. The number of channel tokens is tuned per dataset (ETT1/SolarWind=2, EPA=3, Weather/CHS=1), reflecting the inter-channel correlation structure.

## Key Experimental Results

### Main Results: Asynchronous Channel Forecasting (Case 1, CMSE↓ averaged over all prediction lengths)

| Dataset | **CTF** | TimeFilter | DUET | TimeXer | iTransformer | PatchTST | DLinear | Hi-Patch |
|--------|---------|-----------|------|---------|-------------|---------|--------|---------|
| ETT1 | **0.399** | 0.412 | 0.424 | 0.422 | 0.435 | 0.411 | 0.425 | 0.448 |
| ETT2 | **0.377** | 0.383 | 0.388 | 0.380 | 0.396 | 0.390 | 0.455 | 0.397 |
| SolarWind | **0.403** | 0.404 | 0.438 | 0.424 | 0.470 | 0.417 | 0.421 | 0.431 |
| Weather | 0.275 | **0.273** | 0.309 | 0.300 | 0.313 | 0.275 | 0.287 | 0.301 |
| EPA | **0.776** | 0.863 | 0.782 | 0.886 | 0.882 | 0.854 | 1.047 | 0.808 |
| CHS | **0.285** | 0.304 | 0.307 | 0.298 | 0.305 | 0.315 | 0.351 | 0.301 |

### Ablation Study: Block-Wise Missingness (SolarWind, Asynchronous + Block-Wise Missing, CMSE↓)

| Missing Ratio | **CTF** | TimeFilter | TimeXer | iTransformer | PatchTST | BiTGraph |
|---------|---------|-----------|---------|-------------|---------|---------|
| m=0.125 | **0.409** | 0.430 | 0.427 | 0.475 | 0.426 | 0.426 |
| m=0.250 | **0.429** | 0.444 | 0.442 | 0.496 | 0.456 | 0.444 |
| m=0.375 | **0.452** | 0.471 | 0.462 | 0.539 | 0.515 | 0.467 |
| m=0.500 | **0.475** | 0.522 | 0.514 | 0.606 | 0.593 | 0.495 |

### Component Ablation (SolarWind, m=0.375)

| Configuration | CMSE | CMAE | Note |
|------|------|------|------|
| Full CTF | **0.452** | **0.508** | Full model |
| w/o Channel Dependence | 0.474 | 0.521 | Remove cross-channel attention → +4.9% |
| w/o Dynamic Patching | 0.494 | 0.536 | Fixed patch → +9.3% |
| w/o Patch Masking | 0.458 | 0.508 | Remove training-time masking → +1.3% |

### Key Findings

- **Necessity of unified three-in-one design**: Removing any single component degrades performance; dynamic patching has the largest impact (+9.3%), followed by channel dependence (+4.9%).
- **CTF's advantage grows with missingness rate**: At m=0.5, CTF vs. iTransformer: 0.475 vs. 0.606 (21.6% improvement), demonstrating that mask-guided attention significantly outperforms zero-filling and interpolation under high missing rates.
- **The "no-interpolation" philosophy holds**: CTF models data at original resolution directly, outperforming interpolation-dependent methods by 10–35% on the EPA dataset.
- **Optimal channel token count varies by dataset**: Highly correlated channels use 1 token (Weather/CHS); moderately heterogeneous channels use 2 (ETT1/SolarWind); strongly heterogeneous channels use 3 (EPA).
- **Validation on real industrial data**: CTF achieves 0.285 CMSE on the LNG Cargo Handling System (CHS) dataset, outperforming all baselines.

## Highlights & Insights

- **Multiple roles of the channel token**: Inter-channel dependency relay + asynchronous length normalizer + missing information inference anchor — one design solves three problems simultaneously, with impressive architectural unity.
- **The honest methodology of "no interpolation"**: Interpolation appears to solve the problem while introducing spurious data, which is particularly harmful for dynamic signals; direct masking and inference is a more principled approach.
- **Industrial deployability**: Validation on real-world LNG industrial data demonstrates that the framework offers practical deployment potential beyond academic benchmarks.
- **Elegance of mask-guided attention**: Complex token interaction control is achieved purely through the attention mask matrix, without modifying the standard Transformer architecture.

## Limitations & Future Work

- The unified attention is $O(\mathcal{T}^2)$ in total token count, which may be computationally expensive for large-scale settings with thousands of sensors.
- The number of channel tokens requires per-dataset tuning; an adaptive determination mechanism is absent.
- Frequency-domain dynamic patching relies on FFT-based dominant frequency detection, which may be unsuitable for non-stationary signals.
- Robustness under concept drift has not been evaluated.
- Only fixed but heterogeneous sampling periods are considered; truly irregular (event-driven) sampling is not addressed.

## Related Work & Insights

- **vs. iTransformer (Liu et al., 2024c)**: Also employs channel tokens but assumes synchronous sampling and does not handle missingness — CTF extends to asynchronous and missing-data settings.
- **vs. TimeXer (Wang et al., 2024e)**: Also uses auxiliary tokens but is tested for robustness under zero-fill missingness — CTF handles this explicitly via mask-guided attention.
- **vs. PatchTST (Nie et al., 2023)**: The CI design naturally accommodates variable-length inputs but loses cross-channel information — CTF adds CD richness on top of CI flexibility.
- **vs. BiTGraph (Chen et al., 2024b)**: Integrates missingness handling but assumes synchronous sampling — CTF jointly handles asynchrony and missingness.
- **vs. Hi-Patch (Luo et al., 2025)**: Targets highly sparse irregular settings with point-to-point cross-channel relations — not suited for the structured multi-source asynchronous scenario addressed in this paper.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first framework to unify channel dependency, asynchronous sampling, and block-wise missingness; the redefinition of the channel token represents a genuine conceptual innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Six datasets (including real industrial LNG data), two case settings, detailed ablations, channel token count analysis, and comparison against 12 baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and design motivation is well-articulated, though notation is heavy.
- **Value**: ⭐⭐⭐⭐⭐ — Direct industrial value for real-world time series forecasting; the unified framework paradigm can inspire heterogeneous multimodal data fusion in other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[ICLR 2026\] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[NeurIPS 2025\] AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](../../NeurIPS2025/time_series/aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)

</div>

<!-- RELATED:END -->
