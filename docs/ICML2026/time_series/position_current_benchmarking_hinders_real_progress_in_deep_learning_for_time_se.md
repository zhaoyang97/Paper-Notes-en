---
title: >-
  [Paper Note] Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series
description: >-
  [ICML 2026][Time Series][Time series forecasting] This position paper systematically reveals a core issue in current time series forecasting benchmarking: differences in **design choices** (global/local parameters…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time series forecasting"
  - "benchmarking"
  - "design dimensions"
  - "model comparison"
  - "evaluation methodology"
date: 2026-05-08
content_hash: cf503180da14eb03
---

# Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series

**Conference**: ICML 2026  
**arXiv**: [2512.22702](https://arxiv.org/abs/2512.22702)  
**Code**: TBD  
**Area**: Time Series / Benchmarking Methodology  
**Keywords**: Time series forecasting, benchmarking, design dimensions, model comparison, evaluation methodology

## TL;DR
This position paper systematically reveals a core issue in current time series forecasting benchmarking: differences in **design choices** (global/local parameters, preprocessing, exogenous variables, temporal and spatial processing) are often ignored as "implementation details." This leads to unfair comparisons between papers. Through controlled experiments with 44 datasets × 7 SOTA models × several reference architectures, the authors demonstrate that the impact of these differences ($5-15\%$) often **exceeds the contribution of specific sequence modeling layers** ($1-3\%$).

## Background & Motivation

**Background**: Time series forecasting is popular in deep learning, with many new architectures emerging annually (e.g., PatchTST, iTransformer, TimeMixer, Crossformer). However, the results of these new methods are inconsistent across published benchmarks—the same model on the same dataset often shows significant discrepancies between different papers.

**Limitations of Prior Work**: The rapid overturning of new SOTA results indicates a lack of understanding regarding the fundamental sources of performance improvement. The community is caught in a cycle where new evidence constantly contradicts previous results, yet no one can clearly explain the roots of these contradictions.

**Key Challenge**: Comparisons in papers typically report only final performance metrics while ignoring critical design choices hidden in implementation details: (1) model configuration (global/local/hybrid), (2) preprocessing and exogenous variables, (3) temporal dependence modeling, and (4) spatial dependence processing. When comparing two complex architectures, it is impossible to determine whether performance gains come from a novel attention mechanism or from changes in other design dimensions.

**Goal**: To quantify the impact of four key design dimensions on model performance through rigorous experimental design and to prove that current benchmarking practices are systematically failing.

**Key Insight**: Using "**reference architectures**" (streamlined baselines with well-defined components such as pure MLP, TCN, Transformer, or RNN) for controlled comparisons allows for the direct assessment of the true impact of each dimension by systematically enabling or disabling specific design choices.

**Core Idea**: Benchmarking practices require a complete overhaul. Beyond ensuring hyperparameter consistency, key design choices must be explicitly declared in a **model card** to ensure future comparisons are "apples-to-apples."

## Method

### Overall Architecture
**Diagnosis Phase**: Seven SOTA models (PatchTST, DLinear, TimeMixer, iTransformer, Crossformer, ModernTCN, and Linear baselines) are selected for controlled experiments on 44 real-world datasets. Simultaneously, a set of "reference architectures" (MLP, TCN, RNN, Transformer, Pyramidal Attention, etc.) is implemented. The impact of specific design features is measured in isolation by systematically enabling or disabling them within the reference architectures.

**Analysis Phase**: The roots of performance differences are analyzed across four design dimensions. For each dimension, the magnitude of the impact of different design choices is quantified using comparison tables (changes in MSE/MAE).

**Output**: A "Forecasting Model Card" template that defines essential fields that must be explicitly declared for every newly proposed forecasting architecture.

### Key Designs

1.  **Design Dimensions D1 + D2 (Model Configuration + Preprocessing / Exogenous Variables)**:
    *   **Function**: Determines how model parameters are shared across multiple time series and how data normalization, seasonality, and exogenous covariates are handled.
    *   **Mechanism**: D1 covers three extremes: global models (all parameters shared), local models (each series trained individually), and hybrid models (most parameters shared with some local parameters). D2 involves exogenous variables, calendar features, and normalization methods. Ambiguity is resolved by recording whether the model uses time-series-specific parameters (learnable series embeddings, local parameters in normalization layers).
    *   **Design Motivation**: Many papers currently blur these distinctions. Table 1 shows that adding local parameters to the same model can change MSE by $7\%-30\%$ (e.g., Transformer on Electricity from $0.151 \to 0.136$). Table 2 shows that simply adding or removing calendar features on Traffic changes Transformer MSE from $0.479 \to 0.417$ (a $13\%$ reduction)—differences comparable to any "architectural innovation."

2.  **Design Dimensions D3 + D4 (Temporal Processing + Spatial Processing)**:
    *   **Function**: Determines how intra-series dependencies are modeled along the time axis and how inter-series dependencies are modeled.
    *   **Mechanism**: D3 is the traditional core of "model selection" (MLP, TCN, RNN, Transformer, Pyramidal Attention). D4 involves channel independence, cross-series attention, etc. The individual contribution of each dimension is quantified via ablation experiments on reference architectures.
    *   **Design Motivation**: Surprisingly, Table 3 finds that when other configurations are held constant, performance differences between various temporal processing operators are exaggerated—MLP and Transformer are almost indistinguishable on Electricity and Weather. Table 4 shows that the difference between models with and without spatial attention is $< 5\%$. This suggests that much of the "performance gain from new operators" claimed in papers likely stems from hidden differences in other design dimensions.

3.  **Forecasting Model Card**:
    *   **Function**: Mandates that every newly proposed forecasting architecture explicitly declare key design choices as a metadata standard for future submissions.
    *   **Mechanism**: Includes explicit declarations for the four dimensions (D1-D4), alongside metadata fields for hyperparameter tuning, data preprocessing, and random seeds.
    *   **Design Motivation**: To prevent hidden discrepancies at the source and establish reproducible, comparable benchmarks.

## Key Experimental Results

### Main Results (Quantification of the Impact of Four Design Dimensions)

| Dimension | Model | Config A | Config B | Impact (%) |
| :--- | :--- | :--- | :--- | :--- |
| D1 | Transformer | Global | Local | -10.0 (Electricity) |
| D1 | iTransformer | Hybrid | Global | +9.6 |
| D2 | Transformer | w/ Exogenous | w/o Exogenous | +13.6 (Traffic) |
| D2 | PatchTST | w/o Exogenous | w/ Exogenous | -4.7 (Electricity) |
| D3/D4 | MLP | — | — | $0.129 \pm 0.000$ (Electricity) |
| D3/D4 | Transformer | — | — | $0.129 \pm 0.001$ |
| D3/D4 | TCN | — | — | $0.130 \pm 0.000$ |

**Key Findings**: The impact of D1/D2 ($5-15\%$) is significantly larger than that of D3 (usually $< 5\%$), directly refuting the implicit assumptions of current benchmarking.

### Ablation Study

| Configuration | Electricity | Weather | Traffic | Solar |
| :--- | :--- | :--- | :--- | :--- |
| Global + w/o Exogenous + MLP | 0.129 | 0.148 | 0.376 | 0.194 |
| Global + w/ Exogenous + MLP | 0.127 | 0.146 | 0.342 | 0.191 |
| Hybrid + w/ Exogenous + Transformer | 0.136 | 0.153 | 0.362 | 0.196 |
| Hybrid + w/ Exogenous + TCN | 0.130 | 0.148 | 0.364 | 0.193 |

Simple reference models (e.g., MLP) perform comparably to most SOTA models once design choices are aligned.

### Key Findings
*   **Design Dimensions Outweigh Architectural Innovation**: When holding other conditions constant, the impact of changing preprocessing or parameter-sharing strategies ($5-15\%$) exceeds that of changing the sequence modeling layer ($1-3\%$), overturning the implicit claims of papers from the past decade.
*   **Hidden Heterogeneity**: "PatchTST" in Paper A and Paper B may share the same name but actually represent different entities due to hidden D1-D4 differences, explaining why SOTA results are quickly overturned.
*   **Insights from Reference Architectures**: Simple architectures (even univariate linear models) often compete with complex SOTA models when design choices are aligned; the performance gains of complex models are often disguised "architectural superiority."

## Highlights & Insights
*   **Methodological Breakthrough**: The systematic decomposition using reference architectures and design dimensions makes the contribution of each component transparent. This "isolation of variables" is simple yet effective and can be directly applied to benchmarks in other fields (NLP, CV).
*   **The Biggest "Aha" Moment**: The fact that performance can shift by $10\%$ just by changing the parameter-sharing method reveals an unsettling truth—much of the excitement over "architectural innovation" over the last decade may stem from a neglect of benchmarking practices.
*   **Reusable Ideas**: The "Model Card" concept is a template that mandates the declaration of all critical design choices. This can be extended beyond time series to other domains as a metadata standard for future paper submissions.

## Limitations & Future Work
*   Ours only analyzes long-term forecasting scenarios (96 / 336 steps); short-term forecasting may follow different patterns.
*   While 44 datasets are covered, this is still limited compared to the vast range of deep learning applications.
*   No explicit threshold is provided for "how much difference is too much."
*   It does not demonstrate how to retrospectively evaluate "hidden design dimension scores" for previously published papers.
*   Proposed Improvement: Establish a "Time Series Paper Metadata Database" to automatically extract or require authors to declare D1-D4 values for every paper, collaborating with journal editorial boards to create submission checklists for time series papers.

## Related Work & Insights
*   **vs Brigato et al. (2026)**: Brigato focuses on the evaluation procedure itself (e.g., cross-validation settings); ours goes deeper into architectural design dimensions, identifying design heterogeneity as the root cause.
*   **vs Zeng et al. (2023)**: Zeng found that linear models sometimes outperform Transformers; ours confirms and explains this within a more systematic framework—it is not necessarily that linear models are superior, but that different papers apply different design choices to them.
*   **Insight**: The methodology in this paper draws from classical scientific experimental design principles; the ML community occasionally lacks reflection on foundational experimental design while pursuing new architectures.

## Rating
*   Novelty: ⭐⭐⭐⭐  Not the first to point out benchmarking issues, but the first to systematically identify and quantify the independent impact of four key design dimensions.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐  44 datasets + 7 SOTA + multiple reference architectures + 3 independent runs + clear ablations; top-tier scale and rigor.
*   Writing Quality: ⭐⭐⭐⭐  Clear logic, progressing seamlessly from problem diagnosis to framework proposal, experimental validation, and proposed solutions.
*   Value: ⭐⭐⭐⭐⭐  The potential impact on the time series forecasting community is profound, with the capacity to drive the field toward more mature and standardized evaluation standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](../../NeurIPS2025/time_series/selective_learning_for_deep_time_series_forecasting.md)
- [\[ICML 2026\] Interpretability in Deep Time Series Models Demands Semantic Alignment](interpretability_in_deep_time_series_models_demands_semantic_alignment.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](../../NeurIPS2025/time_series/syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](../../AAAI2026/time_series/counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[ICLR 2026\] Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](../../ICLR2026/time_series/benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)

</div>

<!-- RELATED:END -->
