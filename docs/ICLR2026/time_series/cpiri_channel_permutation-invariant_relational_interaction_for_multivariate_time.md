---
title: >-
  [Paper Note] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting
description: >-
  [ICLR2026][Time Series][Multivariate time series forecasting] This paper proposes the CPiRi framework, which achieves channel permutation-invariant (CPI) cross-channel relational modeling via a frozen pretrained temporal encoder, a lightweight spatial Transformer, and a channel-shuffling training strategy. CPiRi attains state-of-the-art performance on 5 benchmarks with negligible degradation under channel permutation ($\Delta$WAPE < 0.25%).
tags:
  - ICLR2026
  - Time Series
  - Multivariate time series forecasting
  - channel permutation invariance
  - spatiotemporal decoupling
  - Sundial
  - relational inference
date: 2026-05-08
content_hash: 5169c4515f932e44
---

# CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting

**Conference**: ICLR2026
**arXiv**: [2601.20318](https://arxiv.org/abs/2601.20318)
**Code**: [JasonStraka/CPiRi](https://github.com/JasonStraka/CPiRi)
**Area**: Time Series
**Keywords**: Multivariate time series forecasting, channel permutation invariance, spatiotemporal decoupling, Sundial, relational inference

## TL;DR

This paper proposes the CPiRi framework, which achieves channel permutation-invariant (CPI) cross-channel relational modeling via a frozen pretrained temporal encoder, a lightweight spatial Transformer, and a channel-shuffling training strategy. CPiRi attains state-of-the-art performance on 5 benchmarks with negligible degradation under channel permutation ($\Delta$WAPE < 0.25%).

## Background & Motivation

### State of the Field

**Background**: Multivariate time series forecasting (MTSF) faces a CI–CD dilemma:

### Root Cause

**Key Challenge**: Channel-dependent (CD) models **(e.g., Informer, Crossformer)** can model cross-channel relationships but overfit to channel ordering—under channel-shuffling evaluation, Informer's error increases by >400%, revealing that these models memorize positional rather than semantic relationships.

### Starting Point

**Goal**: **Key Insight**: Channel-independent (CI) models **(e.g., DLinear, PatchTST)** are naturally invariant to channel ordering but disregard cross-channel dependencies.

The authors propose the channel permutation invariance (CPI) diagnostic: a model that truly understands inter-channel relationships should remain stable under channel permutation.

## Method

### Overall Architecture

CPiRi adopts a three-stage spatiotemporal decoupling architecture: a frozen Sundial encoder extracts temporal features → a trainable spatial Transformer models cross-channel relationships → a frozen Sundial decoder independently generates predictions. A channel-shuffling strategy is applied during training to enforce content-based relational reasoning.

### Key Designs

**1. Full Spatiotemporal Decoupling**:
- Stage 1: A frozen Sundial foundation model processes each channel independently, extracting $D$-dimensional temporal features $\{\mathbf{h}_1, \dots, \mathbf{h}_C\}$
- Stage 2: A lightweight spatial Transformer encoder (whose self-attention is naturally permutation-equivariant) models cross-channel relationships over the set of channel features
- Stage 3: A frozen Sundial decoder independently decodes each channel

**2. Permutation-Invariant Regularization (Algorithm 1)**: For each training batch, a random permutation $\pi$ is sampled and applied identically to input $X$ and target $Y$, forcing the spatial module to learn relationships solely from feature content rather than positional cues. The optimization objective is $\min_\theta \mathbb{E}_{(\mathcal{X},\mathcal{Y}),\pi} [\mathcal{L}(f_\theta(\mathcal{X}_\pi), \mathcal{Y}_\pi)]$.

**3. Theoretical Guarantee**: Grounded in the permutation-equivariant function decomposition theorem from Deep Sets (Zaheer et al. 2017), self-attention serves as a canonical realization of $f(\mathbf{h}_i) = \rho(\mathbf{h}_i, \bigoplus_{j=1}^C \phi(\mathbf{h}_j))$. The frozen encoder/decoder are channel-independent (invariant), the spatial module is equivariant, and the full pipeline is thus equivariant end-to-end.

**4. Efficiency Advantage**: The temporal encoder compresses each channel to a single token, reducing spatial attention complexity to $O(C^2)$—far lower than iTransformer's $O((T \times C)^2)$.

## Key Experimental Results

### Main Results

| Dataset | CPiRi WAPE↓ | CPiRi MAE↓ | Runner-up | Runner-up WAPE |
|--------|------------|-----------|---------|----------|
| METR-LA | 9.14% | 4.62 | STID 8.48% | (STID uses external holiday features) |
| PEMS-BAY | **3.90%** | **2.36** | STID 3.91% | Matches/surpasses |
| PEMS-04 | **11.67%** | **23.96** | STID 12.43% | −0.76% |
| PEMS-08 | **9.43%** | **17.46** | iTransformer 10.70% | −1.27% |
| SD | **12.25%** | **26.85** | iTransformer 12.45% | −0.20% |

**Channel-Shuffling Robustness (Table 2)**:

### Ablation Study

| Model | PEMS-04 Original → Shuffled WAPE | Degradation |
|------|------------------------------|---------|
| Informer | 13.57% → 83.53% | **+515%** |
| STID | 12.43% → (significant degradation) | **+235%** |
| CPiRi | 11.67% → ~11.9% | **< 0.25%** |

**Inductive Generalization**: Trained on only half of the channels, CPiRi still demonstrates strong generalization to unseen channels.

## Highlights & Insights

1. **The CPI diagnostic exposes a fundamental flaw in CD models**: Informer's error increases by +515% after channel shuffling, demonstrating that existing CD models essentially memorize positions rather than learn relationships.
2. **Minimal yet effective design**: A frozen pretrained model combined with a single-layer spatial Transformer and data augmentation achieves state-of-the-art performance.
3. **Unified CI+CD paradigm**: Inherits the robustness of CI models while acquiring the relational modeling capability of CD models.
4. **Efficiency and scalability**: $O(C^2)$ complexity, scalable to LargeST with 8,600 channels.

## Limitations & Future Work

- Performance depends on the quality and generalization capacity of the Sundial pretrained model.
- CPiRi does not surpass STID/Crossformer on METR-LA, where the latter methods exploit external holiday features.
- The spatial module consists of only a single Transformer layer, limiting its capacity to model deep cross-channel dependencies.
- Channel-shuffling training increases the number of epochs required for convergence.
- Advantages on non-traffic datasets (e.g., Electricity) are less pronounced.

## Related Work & Insights

- **PatchTST (Nie et al. 2023)**: A representative CI model; CPiRi extends this line by introducing cross-channel modeling.
- **iTransformer (Liu et al. 2024a)**: Achieves CPI by tokenizing channels, but spatiotemporal coupling results in $O((T \times C)^2)$ complexity.
- **Sundial (Liu et al. 2025)**: The temporal backbone of CPiRi; pioneers the use of a foundation model as a frozen feature extractor for multivariate tasks.
- **Deep Sets (Zaheer et al. 2017)**: Theoretical foundation for permutation-invariant functions.
- Insight: The paradigm of frozen pretrained models combined with lightweight trainable modules is broadly applicable to scenarios requiring decoupled modeling across different dimensions.

## Rating

- Novelty: ⭐⭐⭐⭐ (The CPI diagnostic is a novel contribution; full spatiotemporal decoupling with shuffling training is elegant and effective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers standard forecasting, CPI evaluation, inductive generalization, and large-scale scalability experiments)
- Writing Quality: ⭐⭐⭐⭐ (Motivation is clear; theory and experiments are tightly connected)
- Value: ⭐⭐⭐⭐ (The CPI perspective introduces a new evaluation dimension and design principle for the MTSF field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)
- [\[ICLR 2026\] Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative](language_in_the_flow_of_time_time-series-paired_texts_weaved_into_a_unified_temp.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-ser.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](reasoning_on_time-series_for_financial_technical_analysis.md)

</div>

<!-- RELATED:END -->
