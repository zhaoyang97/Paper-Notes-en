---
title: >-
  [Paper Note] Online Time Series Prediction Using Feature Adjustment
description: >-
  [ICLR 2026][Time Series][Online Learning] This paper proposes ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space)…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Online Learning"
  - "Distribution Shift"
  - "Feature Space Adaptation"
  - "Delayed Feedback"
  - "Time Series Forecasting"
date: 2026-05-08
content_hash: dd42ceef29ca00f7
---

# Online Time Series Prediction Using Feature Adjustment

**Conference**: ICLR 2026
**arXiv**: [2509.03810](https://arxiv.org/abs/2509.03810)  
**Code**: [Available](https://github.com/xiannanhuang/ADAPT-Z)  
**Area**: Video Understanding
**Keywords**: Online Learning, Distribution Shift, Feature Space Adaptation, Delayed Feedback, Time Series Forecasting

## TL;DR

This paper proposes ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space), which shifts the adaptation objective in online time series forecasting from model parameter updates to feature space correction. A lightweight adapter fuses current features with historical gradients to address delayed feedback in multi-step forecasting. ADAPT-Z consistently outperforms existing online learning methods across 13 datasets.

## Background & Motivation

Time series forecasting faces the core challenge of **distribution shift**: the data distribution continuously changes over time during deployment. Existing online learning methods revolve around two questions: (1) which parameters to update, and (2) how to update them.

Limitations of prior work:

- **Parameter selection bias**: Most methods update the last-layer parameters or introduce small adapter modules, which may not be optimal for adapting to distribution shift.
- **Delayed feedback problem**: In multi-step forecasting (e.g., predicting 24 future steps), the ground truth at time $t$ only arrives at $t+24$, making gradient-based updates from delayed signals potentially unreliable.
- **Train-deploy mismatch**: Training shuffles samples randomly, while deployment receives data in temporal order.

**Core insight**: Apparent distribution shift originates from changes in underlying latent factors (e.g., economic conditions, temperature). The model can be decomposed into an encoder $f$ (extracting latent factor features $z$) and a prediction head $g$. Correcting the feature $z$ more directly addresses the root cause of distribution shift than updating model parameters.

## Method

### Overall Architecture

ADAPT-Z decomposes the forecasting model into an encoder $f$ and a prediction head $g$. The objective is to find a correction term $\delta_t$ such that $g(z_t + \delta_t) \approx y_t$, where $z_t = f(x_t)$ is the current feature representation.

Core components: (1) a dual-path adapter network, (2) historical gradient computation, and (3) a delayed online update mechanism.

### Key Designs

**1. Feature Space Adaptation Paradigm**

The simplest approach—feature-space online gradient descent (fOGD)—is formulated as:

$$\delta_{t+1} = \delta_t - \eta \frac{\partial (g(z_{t-k} + \delta_{t-k}) - y_{t-k})^2}{\partial \delta_{t-k}}$$

where $k$ is the forecasting horizon. However, this approach has limited effectiveness for two reasons: (a) multi-step prediction delay renders gradients stale; (b) the optimal correction $\delta_t$ may depend on the current context $z_t$ rather than being a fixed constant.

Nevertheless, experiments demonstrate that even simple fOGD can match or surpass complex parameter-update methods—challenging the conventional assumption that sophisticated adaptation mechanisms are necessary.

**2. Adapter Network Architecture (Dual-Path Design)**

Directly concatenating $z_t$ and the gradient proves ineffective due to large magnitude discrepancies. ADAPT-Z employs a dual-path structure:

- **Path 1**: A linear layer independently transforms the current feature $z_t$.
- **Path 2**: A linear layer independently transforms the historical gradient.
- **Fusion**: The outputs of both paths are summed and passed through two linear layers to produce the final $\delta_t$.

The input consists of the current feature vector $z_t$ and historical gradient information; the output is the feature correction term $\delta_t$.

**3. Historical Gradient Computation**

To reduce the high variance of single-sample gradients, gradients are computed in a batched manner. Given batch size $b$ and forecasting horizon $k$, the gradient of the average loss over timestamps $t-k-b$ to $t-k$ is used as the historical gradient input at time $t$.

**4. Delayed Online Update**

During deployment, $k$-step delayed online gradient descent is applied:

- Historical gradients, features, and model outputs are cached at each time step.
- Upon receiving the ground truth at time $t$, the loss of the prediction at time $t-k$ is computed.
- Backpropagation updates the adapter parameters.
- The last linear layer parameters are simultaneously updated online.

### Loss & Training

- Training: Standard MSE loss for pretraining the base model.
- Deployment: Online updates to adapter parameters and the last linear layer.
- Enhanced variant: The base model can be fine-tuned on the training set and the adapter trained (3 epochs) prior to online deployment.
- Data split: 60% training / 10% validation / 30% test (more realistic than the 25/5/70 split used in prior work).

## Key Experimental Results

### Main Results

Evaluations span 13 datasets (4 ETT, 4 PEMS, weather, solar, traffic, electricity, exchange), 3 base models (iTransformer, SOFTS, TimesNet), and forecasting horizons of 12/24/48.

| Dataset | Original | fOGD | DSOF | SOLID | ADCSD | Proceed | **ADAPT-Z** | Gain |
|---------|----------|------|------|-------|-------|---------|------------|------|
| ETTm1 | 0.2211 | 0.2178 | 0.2647 | 0.2166 | 0.2169 | 0.2168 | **0.1937** | 12.42% |
| solar | 0.1084 | 0.1074 | 0.1038 | 0.1083 | 0.1075 | 0.1083 | **0.0948** | 12.61% |
| traffic | 0.4075 | 0.4068 | 0.4060 | 0.4070 | 0.4070 | 0.4079 | **0.3689** | 9.49% |
| PEMS04 | 0.1288 | 0.1263 | 0.1465 | 0.1291 | 0.1280 | 0.1290 | **0.1223** | 5.05% |
| weather | 0.1575 | 0.1573 | 0.1975 | 0.1573 | 0.1564 | 0.1575 | **0.1481** | 5.98% |

ADAPT-Z achieves the best performance on all 13 datasets. The DSOF method underperforms the original model on certain datasets.

### Ablation Study

Results for the enhanced variant fine-tuned on the training set:

| Version | ETTh1 | ETTm1 | PEMS03 | solar | traffic |
|---------|-------|-------|--------|-------|---------|
| ADAPT-Z (validation set only) | 0.2626 | 0.1954 | 0.0974 | 0.0940 | 0.3314 |
| Version1 (fine-tuning + online update) | 0.2625 | 0.1948 | 0.0936 | 0.0885 | 0.3197 |
| Version2 (fine-tuning + frozen) | 0.2680 | 0.2104 | 0.0945 | 0.1141 | 0.3224 |

Feature position analysis (iTransformer): Performance remains stable across different layer outputs used as features, but directly modifying the input consistently degrades performance. On average, the output of the first Transformer block yields the best results.

### Key Findings

- **Surprising performance of fOGD**: Applying gradient descent solely in the feature space ranks second on many datasets, validating the feature correction direction.
- **"Learning to adapt" phenomenon in the frozen variant**: Version2, which performs no online updates, still reduces error, indicating that the model learns to leverage previous batch information for self-adaptation during training.
- **Train-test style mismatch**: Current training shuffles samples independently, whereas deployment data arrives in temporal order; future work should account for sample ordering during training.

## Highlights & Insights

1. **Paradigm shift**: Moving from "which parameters to update" to "which features to correct" directly targets the root cause of distribution shift.
2. **A concise yet powerful baseline**: fOGD alone outperforms most complex methods, challenging prevailing assumptions in the field.
3. **"Learning to adapt" phenomenon**: Training with gradient information enables the model to acquire an intrinsic adaptive capability.
4. **Practicality**: The lightweight adapter is plug-and-play and compatible with diverse forecasting models.

## Limitations & Future Work

- The data split (60/10/30) differs from prior work (25/5/70), which may affect online-phase performance comparisons with baselines.
- The selection of feature position (i.e., which block's output to use) lacks theoretical guidance and is currently determined empirically.
- Only point forecasting models are evaluated; adaptation for probabilistic forecasting models remains unexplored.
- The "learning to adapt" phenomenon warrants deeper theoretical analysis; it is currently reported as an empirical observation only.

## Related Work & Insights

- Compared to DSOF and SOLID: these methods update adapter or last-layer parameters, whereas ADAPT-Z operates in the feature space.
- FSNet's dual-stream EMA strategy and ELF's direct fitting strategy represent alternative directions.
- Insight: In the online learning and test-time training literature, feature correction may be an overlooked and superior alternative.

## Rating

- Novelty: ⭐⭐⭐⭐ (Feature space adaptation paradigm is novel; the "learning to adapt" finding is intriguing.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (13 datasets, 3 base models, extensive comparisons and ablations.)
- Writing Quality: ⭐⭐⭐⭐ (Clear exposition, compelling motivation, thorough related work coverage.)
- Value: ⭐⭐⭐⭐ (Provides a new perspective and a concise, effective solution for online time series forecasting.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICLR 2026\] ResCP: Reservoir Conformal Prediction for Time Series Forecasting](rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)
- [\[ICLR 2026\] Relational Feature Caching for Accelerating Diffusion Transformers](relational_feature_caching_for_accelerating_diffusion_transformers.md)
- [\[ICLR 2026\] Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models](dissecting_chronos_sparse_autoencoders_reveal_causal_feature_hierarchies_in_time.md)
- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)

</div>

<!-- RELATED:END -->
