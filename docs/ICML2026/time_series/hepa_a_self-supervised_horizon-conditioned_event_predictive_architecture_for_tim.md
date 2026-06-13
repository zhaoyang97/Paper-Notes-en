---
title: >-
  [Paper Note] HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series
description: >-
  [ICML 2026][Time Series][Event Prediction] HEPA learns predictable dynamics in time series through **horizon-conditioned JEPA self-supervised pre-training**. By freezing the encoder and fine-tuning only the predictor…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Event Prediction"
  - "JEPA"
  - "Self-Supervised Pre-training"
  - "Label Efficiency"
  - "Survival Analysis"
date: 2026-05-08
content_hash: bca5e4cddc155d9e
---

# HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.11130](https://arxiv.org/abs/2605.11130)  
**Code**: TBD  
**Area**: Time Series / Self-Supervised Learning / Event Prediction  
**Keywords**: Event Prediction, JEPA, Self-Supervised Pre-training, Label Efficiency, Survival Analysis

## TL;DR
HEPA learns predictable dynamics in time series through **horizon-conditioned JEPA self-supervised pre-training**. By freezing the encoder and fine-tuning only the predictor, it outperforms multiple SOTA methods across 14 benchmarks in 11 domains using a single architecture and fixed hyperparameters, reaching 92% performance with only 2% labeled data.

## Background & Motivation

**Background**: Event prediction tasks such as turbine failure prediction, arrhythmia detection, anomaly detection, and RUL (Remaining Useful Life) prediction are scattered across different communities, each using its own benchmarks, metrics, and architectures. While structurally identical—"given observations at time $t$, estimate $P(\text{event occurs within } \Delta t)$"—methodologies remain heavily fragmented.

**Limitations of Prior Work**:
- Value prediction methods (supervised or pre-trained) force encoders to capture all signal variations, including noise irrelevant to downstream events.
- Existing self-supervised methods using JEPA for classification require labels, while those for anomaly detection are tuned only for specific tasks.
- No single architecture generalizes across domains, requiring domain-specific parameter adjustments for every application.

**Key Challenge**: How to enable the encoder to learn "predictable" temporal dynamics (rather than all variations) while completing downstream event prediction tasks with minimal labels?

**Goal**: To build a universal event prediction system with a unified architecture and fixed hyperparameters capable of handling diverse event types (from mechanical wear to cardiac anomalies) across multiple domains.

**Key Insight**: Instead of forcing the encoder to predict future values (which contain noise), it should predict future representations (retaining only predictable parts)—the core philosophy of JEPA.

**Core Idea**: (1) Pre-train the encoder using horizon-conditioned JEPA to force learning of dynamics across multiple time scales; (2) Freeze the encoder and fine-tune only the predictor and event head, using a survival CDF to output a monotonically increasing event probability surface.

## Method

### Overall Architecture
A two-stage framework—**Stage 1 (Pre-training)**: A causal Transformer encoder learns temporal dynamics from unlabeled data, while a predictor learns to forecast future representations given a horizon $\Delta t$; **Stage 2 (Downstream Fine-tuning)**: Encoder parameters are frozen, and only the predictor and a lightweight event head are fine-tuned to output a discrete-time survival CDF (ensuring predicted probability increases monotonically with $\Delta t$).

### Key Designs

1. **Horizon-Conditioned JEPA Pre-training**:

    - **Function**: Learns multi-scale temporal dynamics by predicting future representations at varying horizons, avoiding noise inherent in value prediction.
    - **Mechanism**: A causal encoder $f_\theta$ maps observations $\mathbf{x}_{\leq t}$ to a summary embedding $\mathbf{h}_t$; a predictor $g_\phi$ receives $\mathbf{h}_t$ and a horizon $\Delta t$ to predict future interval representations $\hat{\mathbf{h}}_{(t, t + \Delta t]}$. During training, $\Delta t$ is sampled from a log-uniform distribution $[1, \Delta t_{\max}]$, forcing the encoder to learn multi-scale patterns. Target representations $\mathbf{h}^*_{(t, t + \Delta t]}$ are derived from a bidirectional encoder and attention pooling. The loss is $\mathcal{L} = (1 - \alpha) \|\hat{\mathbf{h}} - \mathbf{h}^*\|_1 + \alpha \mathcal{L}_{\text{SIG}}$, where SIGReg (Sketched Isotropic Gaussian Regularisation) constrains predicted representations toward an isotropic Gaussian distribution, replacing the standard JEPA EMA momentum mechanism to prevent representation collapse.
    - **Design Motivation**: Horizon conditioning forces the encoder to understand long-term dependencies; SIGReg is more stable with fewer hyperparameters than EMA; L1 loss is more robust than L2.

2. **Frozen Encoder + Predictor Fine-tuning**:

    - **Function**: Fine-tunes only 198K parameters (vs 2.16M end-to-end) for downstream tasks while preserving pre-trained knowledge from JEPA.
    - **Mechanism**: The encoder is frozen while the predictor and a linear event head are jointly fine-tuned. For $K$ discrete horizons $\Delta t = 1, \ldots, K$, the predictor outputs a conditional hazard for each interval $\lambda_{\Delta t}(t) = \sigma(\mathbf{w}^\top \hat{\mathbf{h}}_{(t, t + \Delta t]} + b)$. The discrete-time survival CDF $p(t, \Delta t) = 1 - \prod_{j=1}^{\Delta t} (1 - \lambda_j(t))$ ensures monotonicity. The fine-tuning loss is $\mathcal{L}_{\text{FT}} = \sum_{\Delta t = 1}^K w^+ \text{BCE}(p(t, \Delta t), y(t, \Delta t))$, where $w^+ = N_{\text{neg}} / N_{\text{pos}}$ compensates for class imbalance.
    - **Design Motivation**: Freezing the encoder avoids catastrophic forgetting and overfitting; predictor fine-tuning is more expressive than a linear probe while using far fewer parameters than end-to-end training; survival CDF monotonicity prevents the contradiction of falling event probabilities over longer horizons.

3. **Unified Probability Surface + h-AUROC Evaluation**:

    - **Function**: Unifies the calculation of all metrics using a single probability surface $p(t, \Delta t)$.
    - **Mechanism**: The model outputs probabilities for every observation time $t$ and prediction horizon $\Delta t$. Domain-specific metrics (RMSE for RUL, PA-F1 for anomaly detection) are projected from this same surface. h-AUROC (average AUROC across horizons) is used as the cross-domain metric.
    - **Design Motivation**: A unified framework allows the same model and hyperparameters across 14 datasets and 11 domains; surface representation preserves complete predictive information.

## Key Experimental Results

### Main Results

| Dataset | Domain | h-AUROC (HEPA) | h-AUROC (PatchTST) | h-AUROC (iTransformer) | Leading? |
|--------|------|-----------------|-------------------|----------------------|--------|
| C-MAPSS-1 | Turbine | **0.81 ± 0.03** | 0.80 | 0.70 | ✓ |
| C-MAPSS-3 | Turbine | **0.84 ± 0.01** | 0.79 | 0.76 | ✓ |
| TEP | Chemical | **1.00** | 0.99 | 0.93 | ✓ |
| Weather | Climate | **0.89** | 0.88 | 0.83 | ✓ |
| GECCO | Water | **0.88** | 0.65 | 0.64 | ✓ |
| MBA | Heart | 0.75 | 0.68 | **0.84** | ✗ |

HEPA leads in 10 out of 14 benchmarks while tuning only 198K parameters (11x fewer than PatchTST).

### Ablation Study & Label Efficiency

| Configuration | C-MAPSS-1 h-AUROC | C-MAPSS-3 h-AUROC | Description |
|------|------------------|------------------|------|
| Full Model (100% labels) | 0.786 | 0.853 | Full HEPA |
| 10% labels | 0.772 | 0.830 | Retains 98% / 97% performance |
| 5% labels | 0.730 | 0.709 | Retains 93% / 83% performance |
| **2% labels (2-5 engines)** | **0.724** | **0.635** | **Retains 92% / 74% performance** |
| 1% labels | 0.670 | 0.513 | Performance drops significantly |

### Theoretical Support (Proposition 1: Event Information Retention Bound)
$I(H_t; E_{t + \Delta t}) \geq I(H^*; E_{t + \Delta t}) - C_\eta L^2 \varepsilon$, with $C_\eta = (2 \underline{\eta} (1 - \overline{\eta}))^{-1}$. Lower pre-training loss correlates with higher downstream h-AUROC (validated on C-MAPSS-1/3 and MBA across different domains, Spearman $\rho = -0.67/-0.64/-0.49$, p < 0.05).

### Key Findings
- On lifecycle datasets with extended precursors, HEPA maintains high performance with minimal labels—achieving 92% of full-label performance on C-MAPSS-1 with only 2% labels (2 engines).
- This validates the theoretical prediction of Proposition 1: low pre-training loss $\varepsilon$ positively correlates with high downstream performance.

## Highlights & Insights
- **Innovative Application of Horizon Conditioning**: While standard JEPA for images ignores time scales, HEPA forces the encoder to learn multi-scale dynamics via log-uniform sampling of $\Delta t$—particularly effective for predicting rare events from long-term drifting signals.
- **Predictor Fine-tuning vs. Linear Probe Expressivity Trade-off**: A linear probe uses only 198 parameters but loses horizon-conditional representation capability; end-to-end training requires 2.16M parameters. Predictor fine-tuning elegantly reshapes horizon-conditioned outputs using an MLP, achieving equivalent performance with 1/11th of the parameters.
- **Monotonic Constraint via Survival CDF**: By combining discrete hazards $\lambda_j$ into a survival function $\prod_j (1 - \lambda_j)$, the cumulative event probability is guaranteed to increase strictly monotonically with the horizon—avoiding internal model contradictions.
- **Cross-Domain Generality vs. Domain-Specific Metrics**: The same model achieves competitive or superior performance across turbine, cardiac, and anomaly domains, demonstrating design robustness.

## Limitations & Future Work
- Disadvantage in Sensor-Localized Events: HEPA underperforms iTransformer and PatchTST on MBA (arrhythmia) and BATADAL (cyber-attacks) because event information is concentrated in a few sensor channels, which HEPA's patch tokenization dilutes.
- Performance Instability on Short-Window Anomaly Datasets: Label efficiency advantages disappear on datasets with short anomaly windows like GECCO.
- Cross-Domain Failure of Pre-training Loss Correlation: While the theoretical bound holds within a single dataset, pre-training loss does not correlate with h-AUROC across different datasets (r = -0.05) because Lipschitz constants vary significantly between domains.

## Related Work & Insights
- **vs TS2Vec / TNC / TimesURL**: Contrastive learning methods are sensitive to noise through positive/negative pairs; HEPA's JEPA avoids the complexity of contrastive pair construction by directly predicting representations.
- **vs PatchTST / SimMTM**: Value prediction and masked reconstruction methods learn all signal variations including noise; HEPA is more efficient by focusing only on predictable dynamics.
- **vs Chronos-2 / MOMENT**: Large-scale pre-trained foundation models gain generality via massive external corpora; HEPA pre-trains per-dataset (< 1 minute), and although weights aren't shared across domains, its fixed universal fine-tuning recipe enables practical deployment.
- **vs MTS-JEPA / TS-JEPA**: MTS-JEPA adds codebook regularization for anomaly detection; HEPA replaces EMA with SIGReg to avoid hyperparameter tuning and wins on 8 out of 9 reproduced datasets.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Horizon-conditioned JEPA + predictor fine-tuning for time series event prediction; Proposition 1 validates design principles.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 benchmarks + 11 domains + 5 baselines + ablation tables + theoretical validation + label efficiency curves + representation visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with both formal mathematical expressions and intuitive explanations.
- Value: ⭐⭐⭐⭐⭐ Unified framework, minimal parameter tuning, and high label efficiency provide significant industrial value; the combination of theory and experiments provides practical guidance on when the method succeeds or fails.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](../../NeurIPS2025/time_series/towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[AAAI 2026\] Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching](../../AAAI2026/time_series/detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](../../NeurIPS2025/time_series/universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[AAAI 2026\] Optimal Look-back Horizon for Time Series Forecasting in Federated Learning](../../AAAI2026/time_series/optimal_look-back_horizon_for_time_series_forecasting_in_federated_learning.md)

</div>

<!-- RELATED:END -->
