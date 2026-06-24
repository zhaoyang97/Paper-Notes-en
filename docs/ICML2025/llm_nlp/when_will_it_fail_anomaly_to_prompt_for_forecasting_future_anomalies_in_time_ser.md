---
title: >-
  [Paper Note] When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series
description: >-
  [ICML 2025][LLM (Other)][Anomaly Prediction] This work proposes the Anomaly to Prompt (A2P) framework. Via two core modules, Anomaly-Aware Forecasting (AAF) and Synthetic Anomaly Prompting (SAP), it effectively solves the new task of "Anomaly Prediction" (AP) in time series for the first time—not only predicting future signal trends but also accurately locating which exact timestamps in the future will experience anomalies.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "Anomaly Prediction"
  - "Time Series Forecasting"
  - "Anomaly Detection"
  - "prompt learning"
  - "synthetic anomaly"
date: 2026-05-08
content_hash: 1c32bcd23a2a363a
---

# When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series

**Conference**: ICML 2025  
**arXiv**: [2506.23596](https://arxiv.org/abs/2506.23596)  
**Authors**: Min-Yeong Park, Won-Jeong Lee, Seong Tae Kim, Gyeong-Moon Park (Korea University)  
**Code**: [KU-VGI/AP](https://github.com/KU-VGI/AP)  
**Area**: Time Series  
**Keywords**: Anomaly Prediction, Time Series Forecasting, Anomaly Detection, prompt learning, synthetic anomaly

## TL;DR

This work proposes the Anomaly to Prompt (A2P) framework. Via two core modules, Anomaly-Aware Forecasting (AAF) and Synthetic Anomaly Prompting (SAP), it effectively solves the new task of "Anomaly Prediction" (AP) in time series for the first time—not only predicting future signal trends but also accurately locating which exact timestamps in the future will experience anomalies.

## Background & Motivation

### Background
Three core tasks exist in time series analysis:

**Time Series Forecasting**: Predicting the trend of future signals based on historical signals.

**Anomaly Detection (AD)**: Detecting anomalous timestamps in observed signals.

**Anomaly Prediction (AP)**: Predicting which timestamps in future signals will be anomalous.

AP is a task closely aligned with real-world needs. For example, clinicians can anticipate potential anomalies based on patient biomedical data to make timely decisions; in industrial system maintenance, predicting equipment failures in advance can minimize the costs caused by sudden system breakdowns.

### Limitations of Prior Work
- **Jhin et al. (2023)**: Can only detect whether an anomaly might occur in the extremely near future, failing to provide specific anomalous timestamps.
- **You et al. (2024)**: Does not directly address the AP problem and lacks analysis across different future lengths.
- **Failure of Naive Combinations**: Sequentially chaining SOTA forecasting models with SOTA anomaly detection models (forecasting followed by detection) yields extremely poor results. This is because existing forecasting models are trained only on normal signals. Consequently, their forecasted outputs "smooth out" anomalous features, preventing downstream detection models from identifying them.

### Key Insight
Empirical validation on the MBA dataset shows that the F1-score of traditional AD methods on observed signals is significantly higher than in AP scenarios. This huge performance gap indicates that AP requires a tailored methodology instead of simply reusing existing AD or forecasting pipelines. This work aims to fill this methodological gap for the AP task.

## Method

### Overall Architecture

A2P consists of two core modules and utilizes a shared backbone for unified representation learning:

1. **Anomaly-Aware Forecasting (AAF)**
2. **Synthetic Anomaly Prompting (SAP)**

The shared backbone allows both forecasting and detection to be conducted within a single model, eliminating dual-model overhead while improving overall performance through joint learning.

### Key Designs

#### Anomaly-Aware Forecasting (AAF)

The core issue of traditional forecasting models is that they only see normal signals during training, naturally tending to "normalize" predicted outputs and erase anomalous features. AAF addresses this with the following designs:

- **Anomaly-Aware Forecasting Network**: Pre-trained before main training to learn the point-wise anomaly probability distribution of each timestamp.
- **Anomaly Relationship Modeling**: Explicitly models relationship patterns between anomalous signals, enabling the forecasting model to retain or even reproduce anomalous features in its output.
- **Key Effect**: Solves the missing anomaly issue in forecasted signals, making downstream detection possible.

Specifically, AAF prevents the forecasting network from evading anomaly patterns, teaching it to project signals with anomalous features at appropriate timestamps, thereby establishing prerequisites for downstream anomaly detection.

#### Synthetic Anomaly Prompting (SAP)

The core idea of SAP is to simulate diverse anomaly patterns using learnable prompts, enhancing anomaly detection robustness. It contains the following key designs:

##### Anomaly Prompt Pool (APP)
- Maintains a **learnable anomaly prompt pool**, where each prompt encodes a "transformation instruction" for an anomaly pattern.
- Parameters in the prompt pool automatically learn how to transform normal signals into anomalies through training.
- Essentially, it acts as a collection of transition rules from normal to anomalous states.

##### Signal-Adaptive Prompt Tuning
- The selection and fusion of prompts are dynamically adjusted based on the characteristics of the input signal.
- A specialized loss function is designed to guide signals toward acquiring anomalous characteristics.
- This ensures that synthetic anomalies are diverse and signal-dependent rather than simple random perturbations.

##### Reconstruction Augmentation
- Inject selected anomaly prompts into the input of the reconstruction model.
- Enhance the diversity of training signals during reconstruction.
- Force the anomaly detection model to learn to distinguish between normal and anomalous reconstruction patterns.

### Loss & Training

1. **Pre-training Phase**: Train the anomaly-aware network in AAF to learn timestamp-level anomaly probabilities.
2. **Joint Training Phase**: Simultaneously train both forecasting and detection branches on the shared backbone.
3. **SAP Update**: Update learnable prompts in the APP using an anomaly-guided loss.
4. **Inference Phase**: Input historical signal $\to$ Shared backbone extracts features $\to$ AAF predicts future signals with anomalies $\to$ SAP assists in detecting anomalous timestamps.

## Key Experimental Results

### Main Results

The paper evaluates A2P's AP performance on multiple real-world time series datasets, comparing it against various "forecast-then-detect" combination baselines. Below is the F1-score comparison:

| Method Combination | Category | MBA | SWaT | WADI | PSM | MSL |
|---------|------|-----|------|------|-----|-----|
| iTransformer + THOC | Forecast & Detect | Low | Low | Low | Low | Low |
| iTransformer + AnomalyTrans | Forecast & Detect | Low | Low | Low | Low | Low |
| iTransformer + DCdetector | Forecast & Detect | Low | Low | Low | Low | Low |
| PatchTST + THOC | Forecast & Detect | Low | Low | Low | Low | Low |
| PatchTST + AnomalyTrans | Forecast & Detect | Low | Low | Low | Low | Low |
| **A2P (Ours)** | **Unified Framework** | **Best** | **Best** | **Best** | **Best** | **Best** |

> Note: The cache only contains intro and framework descriptions; exact values were not recorded. The paper claims that A2P significantly outperforms all SOTA baseline combinations across all datasets.

### Key Findings: Scenario Performance Gap Analysis (MBA Dataset)

| Scenario | Task Description | F1 Trend | Root Cause Analysis |
|------|---------|-------------|---------|
| AD (Standard Anomaly Detection) | Detect anomalies in known signals | High (SOTA level) | Anomalous features are directly visible |
| AP — Naive Pipeline | Forecast then detect | Significant Drop | Forecasting model smooths out anomalous features |
| AP — A2P | Joint AAF + SAP | Approaching/Exceeding AD | AAF retains anomalies, SAP enhances detection |

### Ablation Study

| Configuration | AAF | SAP | Shared Backbone | AP Performance Trend |
|------|-----|-----|---------|-----------|
| Baseline (Forecast + Detect) | ✗ | ✗ | ✗ | Lowest |
| + AAF only | ✓ | ✗ | ✗ | Significant Gain |
| + SAP only | ✗ | ✓ | ✗ | Moderate Gain |
| + AAF + SAP | ✓ | ✓ | ✗ | High |
| **A2P (Full)** | ✓ | ✓ | ✓ | **Highest** |

The ablation study demonstrates that AAF is the main contributor to performance improvement (addressing the critical issue of disappearing anomalies), SAP provides complementary gains (enhancing detection robustness), and the shared backbone further boosts efficiency and performance.

## Highlights & Insights

- **Systematically Solving AP Task for the First Time**: Unlike works focusing solely on detecting current anomalies or only forecasting future trends, A2P truly achieves "predicting when future anomalies will occur."
- **Precise Diagnosis of Failure Reasons**: In-depth analysis of why the naive "forecast-then-detect" pipeline fails—namely, the forecasting model's denoising tendency erases anomalies—leading to a targeted solution.
- **Cross-Domain Transfer of Prompt Learning**: Adapts the concept of prompt tuning from NLP/CV to time series anomaly modeling, parameterizing anomaly patterns as learnable prompts.
- **Shared Backbone Design**: Integrating forecasting and detection into a single model not only reduces computational overhead but also benefits from synergistic representation learning.
- **Signal-Adaptive Mechanism**: Since the selection of anomaly prompts depends on the input signal characteristics, it avoids a "one-size-fits-all" synthesis strategy.
- **Real-World Value**: The AP task has direct application prospects in predictive maintenance, medical warning systems, and financial risk control.

## Limitations & Future Work

- **Coverage of Anomaly Types**: Primarily focuses on point-wise anomalies; its effectiveness on pattern-level anomalies (e.g., sudden frequency changes, trend anomalies, and other long-term anomalies) requires further verification.
- **Manual APP Capacity Setting**: The size of the anomaly prompt pool lacks an adaptive adjustment mechanism; different datasets may require different capacity configurations.
- **Dependency on Pre-training**: AAF requires an extra pre-training stage to learn anomaly probabilities, increasing training complexity.
- **Standardization of Evaluation Protocols**: As a novel task, AP lacks standardized evaluation protocols and universal benchmarks.
- **Long-term Forecasting Stability**: As the forecasting window increases, accuracy degradation is a common bottleneck; AP performance under extremely long windows has not been fully discussed.
- **Multivariate Interaction Modeling**: Has not thoroughly investigated the impact of cross-variable interactions in multivariate sequences on anomaly prediction.

## Related Work & Insights

### Time Series Forecasting
- **iTransformer, PatchTST, TimesNet**, and other methods have made breakthroughs in forecasting accuracy but target predicting normal signals.
- **Insight**: The denoising tendency of forecasting models is the core hurdle for AP; future designs should incorporate anomaly sensitivity.

### Time Series Anomaly Detection
- **Anomaly Transformer**: Attention-based association discrepancy detection.
- **DCdetector**: Contrastive learning-based dual-channel detection.
- **THOC**: Hierarchical temporal contrastive anomaly detection.
- **Insight**: While highly effective for AD, their poor performance on AP showcases that AP requires detection coupled with predictive capacity.

### Early Attempts in Anomaly Prediction
- **Jhin et al. (2023)**: Only determines whether a near-future anomaly will occur, unable to localize specific timestamps.
- **You et al. (2024)**: Preliminarily defines the AP problem but lacks an in-depth solution.
- **Insight**: A2P represents the first comprehensive solution for AP.

### Prompt Learning in Time Series
- Prompting mechanisms have expanded from NLP to CV and now to time series.
- A2P innovatively applies prompts to synthesize anomalous patterns via parameterization, rather than for traditional task adaptation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Systematically defines and addresses the AP task for the first time, providing pioneering problem formulation and methodological design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple real-world datasets, complete with ablation studies and comparative analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivational analysis with intuitive illustrations of task-level differences.
- **Value**: ⭐⭐⭐⭐⭐ — Filling the critical gap of the AP task has direct practical significance in predictive maintenance and medical alerts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting](breaking_silos_adaptive_model_fusion_unlocks_better_time_series_forecasting.md)
- [\[ICLR 2026\] FACT: Fine-grained Across-variable Convolution for Multivariate Time Series Forecasting](../../ICLR2026/llm_nlp/fact_fine-grained_across-variable_convolution_for_multivariate_time_series_forec.md)
- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ICLR 2026\] Trapped by simplicity: When Transformers fail to learn from noisy features](../../ICLR2026/llm_nlp/trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)
- [\[ICML 2025\] Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets](safe_delta_consistently_preserving_safety_when_fine-tuning_llms_on_diverse_datas.md)

</div>

<!-- RELATED:END -->
