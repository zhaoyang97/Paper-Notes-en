---
title: >-
  [Paper Note] Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models
description: >-
  [ICLR 2026][Time Series][Sparse Autoencoder] This work is the first to apply Sparse Autoencoders (SAEs) to a time series foundation model (Chronos-T5-Large), revealing a depth-dependent feature hierarchy through 392 causal ablation experiments: mid-layer encoders concentrate causally critical change-point detection features, whereas the semantically richest final encoder layer exhibits the lowest causal importance.
tags:
  - ICLR 2026
  - Time Series
  - Sparse Autoencoder
  - Time Series Foundation Model
  - mechanistic interpretability
  - Chronos-T5
  - Causal Ablation
  - Feature Hierarchy
date: 2026-05-08
content_hash: b039664ffde1c942
---

# Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models

**Conference**: ICLR 2026
**arXiv**: [2603.10071](https://arxiv.org/abs/2603.10071)
**Code**: Not released
**Area**: Time Series / Interpretability
**Keywords**: Sparse Autoencoder, Time Series Foundation Model, mechanistic interpretability, Chronos-T5, Causal Ablation, Feature Hierarchy

## TL;DR

This work is the first to apply Sparse Autoencoders (SAEs) to a time series foundation model (Chronos-T5-Large), revealing a depth-dependent feature hierarchy through 392 causal ablation experiments: mid-layer encoders concentrate causally critical change-point detection features, whereas the semantically richest final encoder layer exhibits the lowest causal importance.

## Background & Motivation

**Background**: Time series foundation models such as Chronos-T5, TimesFM, MOMENT, and Moirai demonstrate strong zero-shot forecasting performance, yet their internal representations have never been examined at the mechanistic level.

**Limitations of Prior Work**: SAEs have been successfully applied in NLP to decompose dense, superposed activations of language models into interpretable features (Bricken et al., 2023; Templeton et al., 2024), and circuit analysis has identified interpretable computational subgraphs. In contrast, time series interpretability research remains confined to post-hoc methods such as saliency maps, perturbation-based explanations, counterfactual approaches, and concept-based frameworks; only Kalnāre et al. (2025) conducted a preliminary mechanistic analysis on small classifiers, and no prior work has examined foundation models.

**Key Challenge**: The T5 architecture underlying Chronos is mature, SAE training protocols are well-established, and Chronos's discrete tokenization (4,096 bins) provides a natural unit of analysis—yet mechanistic interpretability tools have not been applied in this domain.

**Goal**: This paper investigates whether SAE-learned features are causally relevant, whether a hierarchical structure exists across layers, and whether semantic richness is consistent with causal importance.

## Method

### Overall Architecture

SAEs with TopK sparsity are trained at six extraction points within Chronos-T5-Large (710M parameters, 24 encoder layers + 24 decoder layers, $d_{\text{model}}=1024$). A feature taxonomy is constructed using synthetic data, causal ablation is validated on real ETT data, and a correspondence between semantic labels and causal importance is established for each feature.

### Key Design 1: TopK Sparse Autoencoder

- **Function**: An SAE is trained on residual stream activations at each extraction point to decompose dense activations into sparse, interpretable features.
- **Mechanism**: Given activation $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$, the SAE computes $\mathbf{z} = \text{TopK}(\mathbf{W}_{\text{enc}}(\mathbf{x} - \mathbf{b}_{\text{dec}}) + \mathbf{b}_{\text{enc}}, k)$, retaining only the $k=64$ largest activations, and reconstructs $\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}}\mathbf{z} + \mathbf{b}_{\text{dec}}$.
- **Design Motivation**: TopK provides more direct sparsity control than L1 regularization; the expansion factor $d_{\text{sae}} = 8 \times d_{\text{model}} = 8192$ provides sufficient capacity to decompose superposed features; periodic resampling of dead features ensures utilization.

### Key Design 2: Six-Point Hierarchical Activation Extraction

- **Function**: Forward hooks are registered at six positions—encoder layers 5, 11, and 23, and decoder layers 11 (residual stream + cross-attention output) and 23—to extract activations.
- **Mechanism**: Early, middle, and late representative encoder layers are selected along with corresponding decoder layers, covering the full processing pipeline from input encoding to prediction generation.
- **Design Motivation**: Research on language models has shown that layers at different depths serve distinct functions; whether a similar hierarchical structure exists in time series models is a central research question of this paper.

### Key Design 3: Dual-Source Feature Taxonomy

- **Function**: A synthetic diagnostic dataset (containing known properties such as trends, seasonality, change points, frequency sweeps, and heteroscedastic noise) is used to assign each SAE feature one of 11 temporal concept labels.
- **Mechanism**: The Pearson correlation between each feature's activation pattern on synthetic data and the ground-truth attribute of each diagnostic category is computed; features whose maximum correlation falls below a threshold are labeled as unknown.
- **Design Motivation**: Synthetic data provides ground-truth temporal attributes, avoiding the ambiguity of labeling on real data; the 11 categories cover core temporal concepts including trend, seasonality, change points, frequency, volatility, and noise.

### Key Design 4: Single-Feature and Progressive Causal Ablation

- **Function**: Single-feature ablation zeros out the sparse code of a given feature and measures the resulting change in CRPS; progressive ablation cumulatively removes 1–64 features ranked by their decoder norm contribution.
- **Mechanism**: $\Delta\text{CRPS}_j = \text{CRPS}_{\text{ablated}} - \text{CRPS}_{\text{original}}$; a positive value indicates that the feature carries information necessary for model prediction. Progressive ablation further reveals differences in robustness to feature removal across layers.
- **Design Motivation**: Ablation establishes causal relationships directly, as opposed to correlation analysis; progressive ablation additionally distinguishes features that are "useful but redundant" from those that are "irreplaceable."

### Loss & Training

SAEs are trained for 50,000 steps using MSE reconstruction loss with the Adam optimizer (learning rate $3 \times 10^{-4}$, cosine decay). Ablation experiments are conducted on the ETT benchmark using a context window of 256, prediction length 64, and 4 forecast samples; an extended experiment on the final encoder layer uses 1,024 windows, 8 samples, and 200 features.

## Key Experimental Results

### Table 1: Single-Feature Ablation Summary

| Layer | # Features | Mean $\Delta$CRPS | Median | Max | % Positive | Max/Median |
|---|---|---|---|---|---|---|
| Encoder Block 5 | 64 | 3.05 | 0.95 | 26.32 | 100% | 27.7× |
| Encoder Block 11 | 64 | 5.15 | 1.26 | 38.61 | 100% | 30.5× |
| Encoder Block 23 | 64 | 3.73 | 2.98 | 11.65 | 100% | 3.9× |
| Encoder Block 23† | 200 | 2.37 | 2.37 | 2.44 | 100% | 1.03× |

All 392 ablations produce positive $\Delta$CRPS, confirming that every feature is causally relevant. The mid-layer encoder (Block 11) exhibits the greatest causal influence (max $\Delta$CRPS = 38.61) with a strongly right-skewed distribution.

### Table 2: Feature Taxonomy Distribution by Layer (Selected)

| Concept | Enc 5 | Enc 11 | Enc 23 |
|---|---|---|---|
| Seasonality | 12 | 45 | **1,439** |
| Level shift ↑ | 66 | **1,024** | 1,097 |
| High frequency | 97 | 91 | **668** |
| Noise | 32 | **413** | 315 |
| Label coverage | 4.9% | 25.8% | **59.8%** |

The final encoder layer is semantically richest (59.8% label coverage), while the mid-layer encoder concentrates change-point detection features (1,024 level_shift_up features).

### Key Findings from Progressive Ablation

- **Block 11**: CRPS rises sharply from 2.61 to 25.32 (catastrophic degradation).
- **Block 5**: CRPS rises from 7.05 to 21.54.
- **Block 23**: CRPS **decreases** from 3.62 to 2.73 (an improvement of 0.89); extended experiments confirm this trend is stable.

## Highlights & Insights

1. **Pioneering contribution**: This is the first application of SAEs to a time series foundation model, successfully transferring mechanistic interpretability methodology from NLP.
2. **Counter-intuitive finding**: Causal importance is inversely correlated with semantic richness—mid-layer encoders are causally most critical yet semantically sparse, while the final layer is semantically richest yet yields improved performance upon ablation.
3. **100% causal validation rate**: All 392 ablations produce positive CRPS degradation, providing strong evidence for the causal relevance of SAE features.
4. **Change-point detection as a core mechanism**: Chronos-T5 relies primarily on change-point dynamics rather than periodic pattern recognition, offering guidance for model understanding and improvement.
5. **Plausible explanation for the final-layer ablation paradox**: The final encoder layer likely encodes cross-domain generalization features, and ablation on a specific dataset may function as implicit domain adaptation.

## Limitations & Future Work

1. **Limited dataset coverage**: Causal ablation is conducted exclusively on ETT data; whether findings generalize to other time series domains remains unknown.
2. **Low taxonomy coverage**: 82.8% of features receive no label, and decoder-side coverage is below 6%, indicating that the feature taxonomy remains coarse.
3. **Single model analysis**: Only Chronos-T5-Large is studied; cross-architecture comparisons (e.g., TimesFM, MOMENT) are absent.
4. **Limited statistical precision in ablation configuration**: The fast configuration (256 windows, 4 samples) provides directional conclusions only, with insufficient quantitative precision.
5. **Absence of circuit-level analysis**: Only feature-level ablation is performed; the connectivity and computational graph structure among features are not examined.

## Related Work & Insights

- **Time series foundation models**: Chronos-T5 (Ansari et al., 2024), TimesFM (Das et al., 2024), MOMENT (Goswami et al., 2024), Moirai (Woo et al., 2024).
- **SAEs and mechanistic interpretability**: Bricken et al. (2023) first applied SAEs to decompose language models; Cunningham et al. (2024) proposed TopK SAEs; Templeton et al. (2024) scaled SAEs to Claude 3 Sonnet.
- **Time series interpretability**: Saliency maps (Zhao et al., 2023), perturbation-based explanations (Enguehard, 2023; Liu et al., 2024), counterfactuals (Yan & Wang, 2023), concept-based frameworks (van Sprang et al., 2024).
- **Mechanistic analysis of time series models**: Kalnāre et al. (2025) conducted a preliminary mechanistic analysis on small classifiers; this paper is the first to extend such analysis to foundation models.

## Rating

- Novelty: ⭐⭐⭐⭐ — First transfer of SAE methodology from NLP to time series foundation models; a pioneering contribution.
- Experimental Thoroughness: ⭐⭐⭐ — 392 ablations are compelling but limited to ETT data, a single model, and incomplete taxonomy coverage.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-articulated counter-intuitive findings, and well-designed figures and tables.
- Value: ⭐⭐⭐⭐ — Opens a new direction for mechanistic interpretability of time series models; findings offer actionable guidance for model design and compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)
- [\[ICLR 2026\] Relational Feature Caching for Accelerating Diffusion Transformers](relational_feature_caching_for_accelerating_diffusion_transformers.md)
- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](../../NeurIPS2025/time_series/synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
