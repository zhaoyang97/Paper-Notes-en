---
title: >-
  [Paper Note] Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting
description: >-
  [ICML 2025][LLM (Other)][Time Series Forecasting] Proposes TimeFuse—a sample-level adaptive model fusion framework. It characterizes input time series features using meta-features and trains a learnable fuser to predict the optimal model combination weights, achieving near-universal improvements (outperforming the best single model on 95.1% of samples) across multiple forecasting benchmarks.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "Time Series Forecasting"
  - "Model Fusion"
  - "Meta-Learning"
  - "Ensemble Methods"
  - "Adaptive Weights"
date: 2026-05-08
content_hash: 4fc9178d4b6a4dd8
---

# Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting

**Conference**: ICML 2025  
**arXiv**: [2505.18442](https://arxiv.org/abs/2505.18442)  
**Code**: [https://github.com/ZhiningLiu1998/TimeFuse](https://github.com/ZhiningLiu1998/TimeFuse)  
**Area**: Time Series  
**Keywords**: Time Series Forecasting, Model Fusion, Meta-Learning, Ensemble Methods, Adaptive Weights

## TL;DR
Proposes TimeFuse—a sample-level adaptive model fusion framework. It characterizes input time series features using meta-features and trains a learnable fuser to predict the optimal model combination weights, achieving near-universal improvements (outperforming the best single model on 95.1% of samples) across multiple forecasting benchmarks.

## Background & Motivation

**Background**: Time series forecasting models continue to advance (Transformers, Mamba, MLPs, etc.), competing closely on benchmark datasets.

**Limitations of Prior Work**: Fine-grained sample-level analysis reveals a neglected fact—no single model is consistently optimal across all samples; even the top-ranked model ranks first on only about 23.2% of test samples; each model has unique areas of strength.

**Key Challenge**: The single-model paradigm wastes the complementary strengths of different models.

**Goal**: How to adaptively leverage the unique advantages of different models on different samples?

**Key Insight**: Characterize the properties of each input time series using meta-features, and train a fuser to predict the optimal model combination weights.

**Core Idea**: Shift from "selecting the best single model" to "finding the optimal model combination for each sample".

## Method

### Overall Architecture
1. Build a model zoo: Independently train $k$ forecasting models.
2. Meta-feature extraction: Compute statistical, temporal, and spectral features for each input.
3. Fuser training: Learn the mapping from meta-features to model combination weights.
4. Inference: Extract meta-features $\rightarrow$ predict weights $\rightarrow$ perform weighted combination of predictions from each model.

### Key Designs

1. **Multi-dimensional Meta-feature Extraction**:

    - **Function**: Compute comprehensive feature descriptions for each input time series.
    - **Mechanism**: Three categories of features—statistical features (skewness, kurtosis), temporal features (stationarity, rate of change), and spectral features (dominant frequency, spectral entropy).
    - **Design Motivation**: These features capture the types of time series that different models excel at—e.g., TimeMixer excels at high spectral complexity, while Non-stationary Transformer excels at low stationarity.

2. **Learnable Fuser**:

    - **Function**: Predict the combination weights $w_1, \dots, w_k$ of the $k$ models from meta-features.
    - **Mechanism**: An MLP network that takes meta-features as input and outputs softmax-normalized weights.
    - **Design Motivation**: End-to-end learning enables the fuser to automatically discover associations between meta-features and model strengths.

3. **Cross-dataset Joint Training**:

    - **Function**: Jointly train the fuser on samples from multiple datasets.
    - **Mechanism**: Meta-features are dataset-agnostic descriptions, enabling the fuser to generalize to unseen datasets.
    - **Design Motivation**: Increasing training diversity improves zero-shot generalization capabilities.

### Loss & Training
- Minimize the MSE loss of the fused predictions.
- Decouple fuser training from base model training.
- Support arbitrary heterogeneous base models.

## Key Experimental Results

### Main Results

| Dataset | Best Single Model MSE | TimeFuse MSE | Improved Sample Ratio |
|--------|-------------|-------------|------------|
| ETTh1 | 0.376 | **0.358** | 89.2% |
| Weather | 0.151 | **0.142** | 92.4% |
| Traffic | 0.360 | **0.344** | 95.1% |

### Ablation Study

| Configuration | MSE | Description |
|------|-----|------|
| Uniform Weight Ensemble | 0.368 | Non-adaptive |
| Statistical Features Only | 0.362 | Lacks spectral information |
| All Meta-features | **0.358** | Optimal |
| Single-dataset Training | 0.364 | Poor generalization |
| Cross-dataset Training | **0.358** | Good generalization |

### Key Findings
- Outperforms the best single model on up to 95.1% of samples—achieving near-universal improvement.
- Interpretable fuser weights: high spectral complexity $\rightarrow$ more weight to TimeMixer; low stationarity $\rightarrow$ more weight to Non-stationary Transformer.
- Remains effective for zero-shot generalization to unseen datasets.

## Highlights & Insights
- The fine-grained discovery of **"no one-size-fits-all model"** is highly convincing, backed by thorough data analysis.
- Utilizing meta-features as a bridge to enable cross-dataset transfer of the fuser is a key design choice.
- The framework is highly versatile—any new model can be directly incorporated into the model zoo.

## Limitations & Future Work
- Maintaining and inferring multiple base models results in linearly growing computational overhead.
- Meta-feature design is currently manual; automated feature learning could be more effective.
- Correlations between models are not considered—which may lead to redundancy.

## Related Work & Insights
- **vs. Traditional Ensembles (bagging/boosting)**: Static combinations, non-adaptive.
- **vs. Model Selection**: Selects only a single model, discarding info from other models.
- Provides insights for AutoML and model selection research.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel perspective with sample-level adaptive fusion
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough analysis with 14 models × 7 datasets
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent visualization and analysis
- Value: ⭐⭐⭐⭐⭐ Practical and general forecasting improvement framework

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser.md)
- [\[ICLR 2026\] FACT: Fine-grained Across-variable Convolution for Multivariate Time Series Forecasting](../../ICLR2026/llm_nlp/fact_fine-grained_across-variable_convolution_for_multivariate_time_series_forec.md)
- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](../../ACL2025/llm_nlp/training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](../../ACL2025/llm_nlp/recurrent_kif_continual_learning.md)

</div>

<!-- RELATED:END -->
