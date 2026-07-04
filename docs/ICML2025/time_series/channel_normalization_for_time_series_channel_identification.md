---
title: >-
  [Paper Note] Channel Normalization for Time Series Channel Identification
description: >-
  [ICML 2025][Time Series][Channel Normalization] This work proposes Channel Normalization (CN), which enhances the Channel Identifiability (CID) of time series models by assigning independent affine transformation parameters to each channel. It further extends to an adaptive version ACN (dynamically adjusting parameters) and a prototypical version PCN (supporting unknown/variable channel counts), achieving significant performance improvements across various time series models.
tags:
  - "ICML 2025"
  - "Time Series"
  - "Channel Normalization"
  - "Channel Identifiability"
  - "Multivariate Time Series Forecasting"
  - "Affine Transformation"
  - "Foundation Models"
date: 2026-05-08
content_hash: 0354782e5180df40
---

# Channel Normalization for Time Series Channel Identification

**Conference**: ICML 2025  
**arXiv**: [2506.00432](https://arxiv.org/abs/2506.00432)  
**Code**: [https://github.com/seunghan96/CN](https://github.com/seunghan96/CN)  
**Area**: Time Series  
**Keywords**: Channel Normalization, Channel Identifiability, Multivariate Time Series Forecasting, Affine Transformation, Foundation Models

## TL;DR
This work proposes Channel Normalization (CN), which enhances the Channel Identifiability (CID) of time series models by assigning independent affine transformation parameters to each channel. It further extends to an adaptive version ACN (dynamically adjusting parameters) and a prototypical version PCN (supporting unknown/variable channel counts), achieving significant performance improvements across various time series models.

## Background & Motivation

**Channel Identifiability Problem**: In multivariate time series modeling, "Channel Identifiability" (CID) refers to the model's ability to distinguish between different channels. Models lacking CID (such as PatchTST and DLinear) produce the same output for identical inputs, ignoring channel specificity.

**Classification of Existing Methods**:
   - **Non-CID Models**: All channels share parameters (e.g., PatchTST), which is parameter-efficient but loses channel information.
   - **CID Models**: Independent parameters for each channel (e.g., iTransformer), preserving channel information but leading to a large parameter footprint.

**Key Challenge**: How to endow models with channel identification capabilities without significantly increasing the parameter size?

**Information-Theoretic Perspective**: The authors analyze from the perspective of mutual information, showing that when the mutual information between the model input and the channel index is zero, the model cannot distinguish channels—which is precisely the issue with non-CID models.

**Key Insight**: Inject channel-specific affine parameters into the normalization layer to minimally invasively enhance CID.

## Method

### Overall Architecture
The core idea of CN is highly concise: after standard normalization, replace shared parameters with **channel-specific** $\gamma$ and $\beta$:

$$\text{CN}(x_c) = \gamma_c \cdot \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c$$

where $c$ is the channel index, and $\gamma_c \in \mathbb{R}^D$, $\beta_c \in \mathbb{R}^D$ are the affine parameters learned independently for each channel.

### Key Designs

1. **Channel Normalization (CN)**:

    - Each channel has an independent $(\gamma_c, \beta_c)$ parameter pair.
    - The number of parameters only increases by $2 \times C \times D$ (where $C$ is the number of channels, and $D$ is the feature dimension).
    - It can directly replace LayerNorm in any model.

2. **Adaptive Channel Normalization (ACN)**:

    - Motivation: The parameters of CN are static and cannot adapt to dynamic changes in the input.
    - Design: Compute attention weights based on cosine similarity between channels to dynamically aggregate affine parameters.
    - Formula: $\alpha_{ij} = \text{softmax}(\cos(x_i, x_j) / \tau)$
    - Final Parameter: $\tilde{\gamma}_c = \gamma_c^{global} \odot \sum_j \alpha_{cj} \gamma_j$
    - Advantages: Similar channels share information, improving generalization capability.

3. **Prototypical Channel Normalization (PCN)**:

    - Motivation: CN/ACN requires knowing the number of channels $C$ beforehand, making them inapplicable to scenarios with a variable number of channels (e.g., foundation models).
    - Design: Introduce $K$ learnable prototypes $\{p_k\}_{k=1}^K$, where channels obtain affine parameters through similarity with these prototypes.
    - Formula: $\gamma_c = \sum_k \text{softmax}(\text{sim}(x_c, p_k)) \cdot \gamma_k^{proto}$
    - Advantages: $K$ is fixed and independent of the actual number of channels, making it suitable for time series foundation models.

### Loss & Training
- Directly uses the loss function of the original task (e.g., MSE for forecasting tasks).
- CN acts as a plug-and-play module without introducing extra training objectives.
- It only replaces the normalization layers, keeping the training procedure completely unchanged.

## Key Experimental Results

### Main Results: Long-Term Forecasting MSE (ETTh1 Dataset, Prediction Length 96)

| Model | Original MSE | +CN MSE | Gain |
|------|---------|---------|------|
| PatchTST (Non-CID) | 0.386 | 0.370 | -4.1% |
| DLinear (Non-CID) | 0.375 | 0.362 | -3.5% |
| iTransformer (CID) | 0.386 | 0.374 | -3.1% |
| TSMixer (CID) | 0.391 | 0.375 | -4.1% |
| S-Mamba (CID) | 0.382 | 0.368 | -3.7% |

### Ablation Study

| Method | ETTh1 | ETTh2 | ETTm1 | Weather | Description |
|------|-------|-------|-------|---------|------|
| No Normalization | 0.391 | 0.342 | 0.338 | 0.176 | baseline |
| LayerNorm | 0.386 | 0.337 | 0.334 | 0.174 | Standard Scheme |
| **CN** | 0.374 | 0.329 | 0.326 | 0.168 | Channel-specific affine |
| **ACN** | **0.370** | **0.325** | **0.323** | **0.166** | Adaptive optimal |
| PCN | 0.376 | 0.331 | 0.328 | 0.170 | Prototypical version |

### Key Findings
- CN yields improvements on both non-CID and CID models, indicating that even CID models do not fully exploit channel information.
- ACN outperforms CN in most scenarios, validating the value of dynamic parameter adjustment.
- Although PCN is slightly weaker than CN, it supports variable channel counts, making it more suitable for foundation model scenarios.
- Information-theoretic analysis: CN significantly increases the mutual information between inputs and channel indices.

## Highlights & Insights
- **Simplicity and Effectiveness**: Improving various models with just a one-line modification to the normalization code—demonstrating extremely high engineering value.
- **Solid Theoretical Support**: Explaining why CID matters and why CN is effective from an information-theoretic perspective.
- **Three-stage Progressive Design**: CN $\rightarrow$ ACN $\rightarrow$ PCN incrementally addresses static, dynamic, and variable channel count issues.
- **Plug-and-Play**: Can be directly applied to any existing time series model without modifying the architecture.

## Limitations & Future Work
- PCN requires presetting the number of prototypes $K$, and selecting the optimal $K$ still requires tuning.
- Parameter efficiency in scenarios with a massive number of channels (e.g., thousands of channels) remains to be verified.
- The inter-channel attention computation introduced by ACN may become a bottleneck when the number of channels is very large.
- Primarily validated on forecasting tasks; generalizability to tasks like classification and anomaly detection remains to be explored.

## Related Work & Insights
- Difference from RevIN (Reversible Instance Normalization): RevIN performs instance normalization along the temporal dimension, whereas CN specializes along the channel dimension.
- Relationship with C-LoRA: Both focus on channel adaptation, but C-LoRA uses low-rank adaptation, while CN is more direct.
- **Insight**: Normalization layers can serve as a minimal interface for injecting prior knowledge and can be extended to other tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Concise concept but novel perspective (CID + information theory).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple models × multiple datasets, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear chain of motivation-methodological-theoretical analysis.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play with extremely high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](../../NeurIPS2025/time_series/channel_matters_estimating_channel_influence_for_multivariate_time_series.md)
- [\[ICML 2025\] IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction](imts_is_worth_time_times_channel_patches_visual_masked_autoencoders_for_irregula.md)
- [\[AAAI 2026\] C3RL: Rethinking the Combination of Channel-independence and Channel-mixing from Representation Learning](../../AAAI2026/time_series/c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)
- [\[ICLR 2026\] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](../../ICLR2026/time_series/cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](../../ICLR2026/time_series/routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)

</div>

<!-- RELATED:END -->
