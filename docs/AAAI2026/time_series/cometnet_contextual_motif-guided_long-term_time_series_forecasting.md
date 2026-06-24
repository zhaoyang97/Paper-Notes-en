---
title: >-
  [Paper Note] CometNet: Contextual Motif-guided Long-term Time Series Forecasting
description: >-
  [AAAI 2026][Time Series][Long-term time series forecasting] This paper proposes CometNet, which extracts recurrently occurring "contextual motifs" from the full historical sequence to construct a motif library, and employs a motif-guided MoE architecture to dynamically associate the current window with relevant motifs for prediction. This approach breaks the receptive field bottleneck imposed by limited look-back windows and achieves significant improvements over state-of-the…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Long-term time series forecasting"
  - "contextual motif"
  - "mixture of experts"
  - "receptive field bottleneck"
  - "frequency-domain analysis"
date: 2026-05-08
content_hash: b7ab25e67fa02f2b
---

# CometNet: Contextual Motif-guided Long-term Time Series Forecasting

**Conference**: AAAI 2026  
**arXiv**: [2511.08049](https://arxiv.org/abs/2511.08049)  
**Code**: None  
**Area**: Time Series Forecasting  
**Keywords**: Long-term time series forecasting, contextual motif, mixture of experts, receptive field bottleneck, frequency-domain analysis

## TL;DR
This paper proposes CometNet, which extracts recurrently occurring "contextual motifs" from the full historical sequence to construct a motif library, and employs a motif-guided MoE architecture to dynamically associate the current window with relevant motifs for prediction. This approach breaks the receptive field bottleneck imposed by limited look-back windows and achieves significant improvements over state-of-the-art methods such as TimeMixer++ and iTransformer on 8 datasets.

## Background & Motivation

**Background**: Long-term time series forecasting (LTSF) is a core task in data science. Mainstream approaches include Transformer-based models (PatchTST, iTransformer) and MLP-based models (DLinear, TimeMixer++), all of which operate within a fixed look-back window.

**Limitations of Prior Work**: **Receptive field bottleneck** — models can only learn from a window of length $L$ and cannot capture long-range dependencies beyond the window. Gradient backpropagation is confined to a single window, even when sliding windows traverse the entire sequence during training. Simply enlarging the window not only incurs $O(L^2)$ computational complexity but also buries meaningful temporal dependencies in historical noise.

**Key Challenge**: Long-range context is necessary for long-term forecasting, yet directly enlarging the window is both costly and yields diminishing returns.

**Goal**: To provide the model with long-range contextual information beyond the look-back window without increasing window size.

**Key Insight**: Real-world time series are governed by periodic "contextual motifs" — such as factory production cycles or seasonal climate patterns — that recur across thousands of time steps. Extracting these motifs and using them to guide prediction is a natural and principled approach.

**Core Idea**: Mine recurrent contextual motifs from the full history to build a motif library; at inference time, dynamically match the current window to the most relevant motifs via MoE routing to inject long-range context.

## Method

### Overall Architecture
A two-stage paradigm:
1. **Contextual Motif Extraction** (offline): Analyze the entire historical sequence and construct a dominant motif library $\mathcal{M} = \{m_1, ..., m_K\}$.
2. **Motif-guided Forecasting** (online): Given a look-back window, route to the expert network associated with the most relevant motif via MoE routing to produce predictions.

### Key Designs

1. **Cascaded Motif Extraction**:

    - **Function**: Automatically discover multi-scale dominant contextual motifs from historical sequences.
    - **Mechanism**: A three-step cascade — (a) **Multi-scale candidate discovery**: FFT extracts dominant frequencies → top-$N_s$ periods are selected as scales → at each scale, downsampled subsequences are clustered using anchor points (randomly sampled subsequences serve as anchors; a Pearson correlation matrix computes density scores) to yield candidate motifs; (b) **Cross-scale redundancy removal**: candidate motifs form a DTW similarity graph → the most representative motif within each connected component is selected as the prototype; (c) **Benefit-driven selection**: each candidate is evaluated by $B(c|\mathcal{S}) = Q(c) \cdot Cov(c|\mathcal{S}) \cdot Div(c|\mathcal{S})$, and the top-$K$ candidates are iteratively selected to form the final library.
    - **Design Motivation**: Direct multi-scale search produces an exponentially large candidate space with severe cross-scale redundancy. The cascaded strategy — discover, deduplicate, then refine — balances comprehensiveness with efficiency.

2. **Motif-driven Gating Network**:

    - **Function**: Dynamically associate the current window with the most relevant motif in the library.
    - **Mechanism**: The window embedding $e_t = \text{LN}(\text{MLP}(X_{t-L+1:t}))$ is processed by two heads — a **routing head** that produces a $K$-dimensional softmax probability $p_t$ (selecting which motif/expert to use), and a **position head** that produces $s_t \in [0,1]$ (the relative position of the current window within the motif's life cycle).
    - **Design Motivation**: It is necessary not only to identify "which motif matches" but also to determine "at which stage within the motif" — positional information provides fine-grained temporal context.

3. **Context-conditioned Experts**:

    - **Function**: $K$ experts each correspond to one motif and generate predictions based on motif-specific temporal dynamics.
    - **Mechanism**: The positional encoding $e_{pos} = \text{MLP}(s_t)$ is concatenated with the window embedding and fused to obtain a conditioned representation $z_t$. $K$ parallel expert prediction heads each output $\hat{X}_{k}$, and the final prediction is $\hat{X} = \sum_k p_{t,k} \cdot P_k(z_t)$.
    - **Design Motivation**: Different motifs represent distinct temporal dynamics (e.g., weekday/weekend cycles, seasonal cycles). Specialized experts capture the prediction logic of each pattern more precisely than a unified model.

### Loss & Training
- Standard MSE loss.
- Channel-independent strategy for multivariate time series.

## Key Experimental Results

### Main Results
Average MSE across 8 datasets (look-back 96, average over prediction horizons 96/192/336/720):

| Model | ETTh1 Avg | ETTh2 Avg | ETTm1 Avg | ETTm2 Avg |
|------|-----------|-----------|-----------|-----------|
| TimeMixer++ (2025) | 0.419 | 0.356 | 0.351 | - |
| iTransformer (2024) | 0.454 | 0.383 | 0.360 | - |
| PatchTST (2023) | 0.516 | - | - | - |
| **CometNet (Ours)** | **0.373** | **0.284** | **0.324** | - |

On ETTh1 with look-back 96 and prediction horizon 720: MSE 0.391 (vs. TimeMixer++ 0.467), a 16.3% improvement.

### Ablation Study

| Configuration | ETTh1 MSE | Note |
|------|-----------|------|
| w/o Motif (MLP only) | ~0.43 | Baseline without contextual guidance |
| w/o positional encoding | ~0.40 | Loss of within-motif position information |
| w/o cross-scale deduplication | ~0.39 | Redundant motifs degrade library quality |
| **Full CometNet** | **0.373** | Complete model |

### Key Findings
- The advantage of CometNet grows with longer prediction horizons — the gain is most pronounced at 720 steps, confirming that motif context is critical for long-term forecasting.
- Positional encoding contributes substantially: knowing "at which stage within a motif" is more informative than knowing "which motif" alone.
- Even with a short look-back window (96 steps), CometNet leverages context spanning thousands of steps via the motif library.

## Highlights & Insights
- The design of **motif mining as a preprocessing step** is elegant: rather than modifying the training window size, long-range context is "injected" into a limited-window model through offline construction of a motif library, incurring zero additional online computational overhead.
- The **dual-head gating** (routing + position) adds a positional dimension beyond standard MoE, enabling more precise expert predictions — analogous to "telling the model not only the season but also the day within that season."
- The cascaded extraction pipeline (FFT → clustering → graph-based deduplication → benefit-driven selection) is engineering-intensive but demonstrably effective.

## Limitations & Future Work
- Motif extraction is entirely offline and relies on FFT, which may yield low-quality motifs for non-stationary sequences with abrupt trends.
- The motif library size $K$ is a hyperparameter that may require tuning across different datasets.
- The channel-independent strategy ignores inter-variable correlations, which may limit performance in high-dimensional multivariate settings.
- The softmax routing at inference time is a soft selection in which all experts participate in computation; efficiency could be further improved with top-1 routing.

## Related Work & Insights
- **vs. TimeMixer++**: Performs multi-scale mixing within the window but remains constrained by window size. CometNet transcends this limitation via motifs, achieving an average ETTh1 MSE of 0.373 vs. 0.419.
- **vs. BSA (2024)**: BSA enhances long-range modeling via cross-sample spectral attention but still struggles with dependencies spanning thousands of steps. CometNet's motifs directly encode patterns at the thousand-step scale.
- **vs. PatchTST**: PatchTST's patches remain within the window boundary, whereas CometNet's motifs span across window boundaries.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — A new paradigm of motif-guided forecasting that fundamentally addresses the receptive field bottleneck.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 8 datasets, multiple prediction horizons, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear; the motif concept is well illustrated with figures.
- **Value**: ⭐⭐⭐⭐⭐ — Introduces a new direction for long-term time series forecasting; the motif library + MoE framework exhibits strong generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting](../../ICLR2026/time_series/pmdformer_patch-mean_decoupling_information_transformer_for_long-term_forecastin.md)
- [\[ICML 2025\] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](../../ICML2025/time_series/timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [\[AAAI 2026\] iTimER: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning](beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)
- [\[ICLR 2026\] Inferring brain plasticity rule under long-term stimulation with structured recurrent dynamics](../../ICLR2026/time_series/inferring_brain_plasticity_rule_under_long-term_stimulation_with_structured_recu.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)

</div>

<!-- RELATED:END -->
